"""Unconditioned Amazon event marks and explicit one-visit reconciliation.

No vehicle, energy, route, target-city geometry, or solver feasibility enters
donor construction or sampling. Source service values remain *planned* marks.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable
from zoneinfo import ZoneInfo

from .calibration import DEFAULT_CALIBRATION, CALIBRATION_SOURCE_SHA256, GEN_TRAIN_STATION_DAY_IDS, service_curve

import numpy as np
import pandas as pd

SCHEMA = "evrptw_unconditioned_joint_marks_v1"
TIMEZONES = {"DAU": "America/Chicago", "DBO": "America/New_York",
             "DCH": "America/Chicago", "DLA": "America/Los_Angeles",
             "DSE": "America/Los_Angeles"}
ARTIFACT_FILES = {"parcels.parquet", "bundles.parquet", "quarantine.parquet"}


def _checked_manifest(path, *, verify_hashes=True):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported joint-mark schema")
    outputs = manifest.get("outputs_sha256", {})
    sources = manifest.get("source_sha256", {})
    if set(outputs) != ARTIFACT_FILES or set(sources) != {"package_data", "route_data"}:
        raise ValueError("Invalid artifact/source file manifest")
    if not all(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
               for digest in [*outputs.values(), *sources.values()]):
        raise ValueError("SHA-256 fields must contain 64 lowercase hexadecimal characters")
    if verify_hashes:
        for name, digest in outputs.items():
            if sha256_file(root / name) != digest:
                raise ValueError("Changed donor artifact: " + name)
    return manifest


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def finite(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _window(raw, station, source_date):
    window = raw.get("time_window")
    if not isinstance(window, dict):
        missing = window is None or (isinstance(window, (float, np.floating)) and math.isnan(float(window)))
        return None, None, "missing" if missing else "invalid_type"
    values = [window.get("start_time_utc"), window.get("end_time_utc")]
    def absent(value):
        return value is None or (isinstance(value, str) and not value.strip()) or (
            isinstance(value, (float, np.floating)) and math.isnan(float(value)))
    if all(absent(v) for v in values):
        return None, None, "missing"
    if any(not absent(v) and not isinstance(v, str) for v in values):
        return None, None, "invalid_type"
    present = [isinstance(v, str) and bool(v.strip()) for v in values]
    if not any(present):
        return None, None, "missing"
    if not all(present):
        return None, None, "invalid_partial"
    tz_name = TIMEZONES.get(str(station)[:3])
    if tz_name is None:
        return None, None, "timezone_unsupported"
    tz = ZoneInfo(tz_name)
    try:
        midnight = datetime.combine(date.fromisoformat(source_date), time.min, tzinfo=tz)
        offsets = []
        for value in values:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            offsets.append((parsed.astimezone(tz) - midnight).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return None, None, "invalid_parse"
    if offsets[1] < offsets[0]:
        return offsets[0], offsets[1], "invalid_reversed"
    return float(offsets[0]), float(offsets[1]), "specified"


def _record(raw, station, source_date):
    dimensions = raw.get("dimensions") if isinstance(raw.get("dimensions"), dict) else {}
    xyz = [finite(dimensions.get(key)) for key in ("depth_cm", "height_cm", "width_cm")]
    service = finite(raw.get("planned_service_time_seconds"))
    start, end, status = _window(raw, station, source_date)
    valid_dimensions = all(v is not None and v > 0 for v in xyz)
    valid_service = service is not None and service >= 0
    return {"depth_cm": xyz[0], "height_cm": xyz[1], "width_cm": xyz[2],
            "volume_cm3": float(math.prod(xyz)) if valid_dimensions else None,
            "source_planned_service_s": service,
            "source_tw_start_offset_s": start, "source_tw_end_offset_s": end,
            "source_tw_status": status, "source_scan_status": str(raw.get("scan_status") or ""),
            "individual_marks_valid": valid_dimensions and valid_service and status in {"missing", "specified"}}


def build_donor_artifact(*, package_path, route_path, output_dir,
                         partition="training", overwrite=False):
    """Build parse-valid joint donor bundles, with no operating-feasibility gate.

    A source parcel is identified within its source partition and route. One
    DELIVERED event wins over attempts. Multiple DELIVERED events or conflicting
    intrinsic size/window evidence are quarantined; planned-service differences
    across attempts are expected and never treated as intrinsic contradictions.
    Attempt-only observations remain a separately named secondary layer.
    """
    package_path, route_path, out = Path(package_path), Path(route_path), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    inputs = {"package_data": sha256_file(package_path), "route_data": sha256_file(route_path)}
    if manifest_path.exists() and not overwrite:
        manifest = _checked_manifest(out)
        if manifest.get("schema") != SCHEMA or manifest.get("source_sha256") != inputs or manifest.get("partition") != partition:
            raise ValueError("Existing donor artifact has a different schema or source version")
        return manifest
    packages = json.loads(package_path.read_text())
    routes = json.loads(route_path.read_text())
    if not isinstance(packages, dict) or not packages or not isinstance(routes, dict) or not routes:
        raise ValueError("Nonempty source route mappings required")
    rows, bundles, quarantine = [], [], []
    counts = Counter()
    dates = []
    for route_id in sorted(packages):
        route = routes.get(route_id)
        if not isinstance(route, dict) or not isinstance(packages[route_id], dict):
            raise ValueError("Source package route must have route metadata and a stop mapping")
        station = str(route.get("station_code") or "")
        source_date = str(route.get("date_YYYY_MM_DD") or "")
        date.fromisoformat(source_date)
        dates.append(source_date)
        stops = route.get("stops", {})
        events = defaultdict(list)
        for stop_id, payload in sorted(packages[route_id].items()):
            if not isinstance(payload, dict):
                counts["invalid_stop_package_mapping"] += 1
                continue
            if stops.get(stop_id, {}).get("type") != "Dropoff":
                counts["non_dropoff_events"] += len(payload)
                continue
            for parcel_id, raw in sorted(payload.items()):
                counts["raw_dropoff_events"] += 1
                if not isinstance(raw, dict):
                    counts["invalid_package_mapping"] += 1
                    continue
                rec = _record(raw, station, source_date)
                events[parcel_id].append((stop_id, rec))
        grouped = defaultdict(list)
        for parcel_id, observations in sorted(events.items()):
            counts["unique_route_parcels"] += 1
            counts["repeated_extra_events"] += len(observations) - 1
            delivered = [(sid, rec) for sid, rec in observations if rec["source_scan_status"] == "DELIVERED"]
            fingerprints = {(r["depth_cm"], r["height_cm"], r["width_cm"],
                r["source_tw_start_offset_s"], r["source_tw_end_offset_s"], r["source_tw_status"])
                for _, r in observations}
            reason = None
            if len(delivered) > 1:
                reason = "multiple_delivered_events"
            elif len(fingerprints) > 1:
                reason = "intrinsic_mark_conflict"
            if reason:
                counts["quarantine_" + reason] += 1
                quarantine.append({"source_parcel_ref": identity(partition, route_id, parcel_id),
                    "reason": reason, "event_count": len(observations)})
                continue
            if delivered:
                stop_id, rec = delivered[0]
                kind = "delivered"
            else:
                attempted = [(sid, r) for sid, r in observations
                    if r["source_scan_status"] in {"DELIVERY_ATTEMPTED", "REJECTED"}]
                if not attempted:
                    counts["unsupported_status_parcels"] += 1
                    continue
                stop_id, rec = min(attempted, key=lambda x: (x[0], x[1]["source_scan_status"]))
                kind = "attempt_only"
            index = len(rows)
            rows.append({**rec, "source_parcel_ref": identity(partition, route_id, parcel_id),
                "source_event_ref": identity(partition, route_id, stop_id, parcel_id),
                "donor_id": identity(partition, route_id, stop_id, kind),
                "source_kind": kind, "source_station_code": station,
                "source_date": source_date,
                "source_station_day_id": identity(partition, station, source_date),
                "source_route_ref": identity(partition, route_id),
                "source_event_count": len(observations)})
            grouped[(stop_id, kind)].append(index)
            counts["canonical_" + kind + "_parcels"] += 1
            counts["source_tw_" + rec["source_tw_status"]] += 1
        for (stop_id, kind), indices in sorted(grouped.items()):
            items = [rows[i] for i in indices]
            valid = all(r["individual_marks_valid"] for r in items)
            specified = [r for r in items if r["source_tw_status"] == "specified"]
            source_empty = bool(specified) and max(r["source_tw_start_offset_s"] for r in specified) > min(r["source_tw_end_offset_s"] for r in specified)
            bundles.append({"donor_id": identity(partition, route_id, stop_id, kind),
                "source_kind": kind, "source_station_code": station, "source_date": source_date,
                "source_station_day_id": identity(partition, station, source_date),
                "source_route_ref": identity(partition, route_id), "package_count": len(indices),
                "parcel_rows": indices, "individual_marks_valid": valid,
                "source_one_visit_window_intersection_empty": source_empty,
                "source_event_package_count": len(packages[route_id][stop_id]),
                "canonical_event_dedup_applied": True})
    parcel_frame = pd.DataFrame(rows)
    bundle_frame = pd.DataFrame(bundles)
    quarantine_frame = pd.DataFrame(quarantine, columns=["source_parcel_ref", "reason", "event_count"])
    parcel_frame.to_parquet(out / "parcels.parquet", index=False, compression="zstd")
    bundle_frame.to_parquet(out / "bundles.parquet", index=False, compression="zstd")
    quarantine_frame.to_parquet(out / "quarantine.parquet", index=False, compression="zstd")
    primary = bundle_frame.loc[bundle_frame.source_kind.eq("delivered") & bundle_frame.individual_marks_valid]
    manifest = {"schema": SCHEMA, "partition": partition,
        "source_collection": "almrrc2021", "source_release_year": 2021,
        "source_file_names": {"package_data": package_path.name, "route_data": route_path.name},
        "source_paths": {"package_data": str(package_path.resolve()), "route_data": str(route_path.resolve())},
        "source_station_timezone_assumptions": dict(TIMEZONES),
        "source_sha256": inputs, "source_observation_dates": [min(dates), max(dates)],
        "canonical_parcel_rows": len(parcel_frame), "bundle_rows": len(bundle_frame),
        "primary_usable_bundle_count": len(primary),
        "primary_package_count_histogram": {str(k): int(v) for k, v in primary.package_count.value_counts().sort_index().items()},
        "counts": dict(sorted(counts.items())),
        "no_vehicle_or_route_feasibility_filter": True,
        "source_geography_transferred": False,
        "planned_service_is_observed_stop_duration": False,
        "empty_joint_window_bundles_excluded": False,
        "dedup_policy": "unique DELIVERED preferred; multiple delivered/intrinsic conflict quarantined; attempt-only separate",
        "outputs_sha256": {name: sha256_file(out / name) for name in ["parcels.parquet", "bundles.parquet", "quarantine.parquet"]}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class DonorLibrary:
    def __init__(self, artifact_dir, *, verify_hashes=True, allowed_station_day_ids: Iterable[str] | None = None):
        self.path = Path(artifact_dir)
        self.manifest = _checked_manifest(self.path, verify_hashes=verify_hashes)
        self.parcels = pd.read_parquet(self.path / "parcels.parquet")
        pool = pd.read_parquet(self.path / "bundles.parquet")
        pool = pool.loc[pool.source_kind.eq("delivered") & pool.individual_marks_valid].copy()
        if allowed_station_day_ids is not None:
            pool = pool.loc[pool.source_station_day_id.isin(set(allowed_station_day_ids))]
        if pool.empty:
            raise ValueError("No source-supported delivered bundles in the declared donor pool")
        self._declared_pool = pool.copy()
        self.calibration = None
        self._set_pool(pool)

    def _set_pool(self, pool):
        self.bundles = pool.reset_index(drop=True)
        self.by_count = {int(k): g.index.to_numpy(dtype=int) for k, g in self.bundles.groupby("package_count")}
        self.counts = np.array(sorted(self.by_count), dtype=int)
        # Column views avoid constructing two pandas row frames for every
        # arriving address bundle in a large city. They preserve the same pool,
        # random draw, tuple values, and count-extrapolation policy.
        self._parcel_arrays = {name: self.parcels[name].to_numpy() for name in self.parcels}
        self._donor_rows = self.bundles.parcel_rows.to_numpy()
        self._donor_ids = self.bundles.donor_id.to_numpy()
        self._donor_days = self.bundles.source_station_day_id.to_numpy()

    def configure_calibration(self, settings=None):
        """Select the reviewed training partition, without capacity/TW filtering.

        ``declared_pool`` exists for independently declared alternate source
        libraries and fixtures; it is always recorded in output provenance.
        """
        self.calibration = {**DEFAULT_CALIBRATION, **(settings or {})}
        pool = self._declared_pool
        name = self.calibration["donor_pool"]
        if name == "GEN-TRAIN":
            if self.manifest["source_sha256"] != CALIBRATION_SOURCE_SHA256 or self.manifest["partition"] != "training":
                raise ValueError("GEN-TRAIN calibration requires its hash-bound canonical training source")
            pool = pool.loc[pool.source_station_day_id.isin(GEN_TRAIN_STATION_DAY_IDS)]
        elif name != "declared_pool":
            raise ValueError("Declare GEN-TRAIN or an explicit alternate declared_pool")
        if pool.empty:
            raise ValueError("No donors in calibrated source partition")
        self._set_pool(pool)
        self.pool_identity = hashlib.sha256("\n".join(sorted(pool.donor_id.astype(str))).encode()).hexdigest()
        rows = np.concatenate(self._donor_rows).astype(int)
        q = self.parcels.iloc[rows].copy()
        q["_a"] = np.where(q.source_tw_status.eq("specified"), q.source_tw_start_offset_s.to_numpy(dtype=float), -np.inf)
        q["_b"] = np.where(q.source_tw_status.eq("specified"), q.source_tw_end_offset_s.to_numpy(dtype=float), np.inf)
        summary = q.groupby("donor_id", sort=False).agg(service=("source_planned_service_s", "sum"),
            a=("_a", "max"), b=("_b", "min"))
        summary = summary.reindex(self.bundles.donor_id)
        self._donor_service = summary.service.to_numpy(float)
        self._donor_counts = self.bundles.package_count.to_numpy(int)
        self._days_per_count = {k: set(self._donor_days[v]) for k, v in self.by_count.items()}
        self._eligible_cache = {}
        patterns = summary.loc[np.isfinite(summary.a) & np.isfinite(summary.b) & summary.a.le(summary.b)].copy()
        self._tw_patterns = patterns
        self.calibration_summary = {"donor_pool": name, "pool_identity_sha256": self.pool_identity,
            "canonical_outputs_sha256": self.manifest["outputs_sha256"], "bundles": len(pool),
            "station_days": int(pool.source_station_day_id.nunique()),
            "training_max_observed_count": int(self.counts.max()),
            "specified_location_patterns": len(patterns),
            "specified_location_probability": self.calibration["specified_location_tw_probability"],
            "source_partition_hash_bound": name == "GEN-TRAIN",
            "no_source_capacity_or_time_feasibility_filter": True}
        return self.calibration_summary

    def eligible_indices(self, count):
        if self.calibration is None:
            source_k = count if count in self.by_count else min(self.counts, key=lambda v: (abs(math.log(int(v)) - math.log(count)), int(v)))
            return self.by_count[int(source_k)]
        if count in self._eligible_cache:
            return self._eligible_cache[count]
        minimum = self.calibration["minimum_donor_bundles"]
        day_min = self.calibration["minimum_donor_station_days"]
        if count in self.by_count and len(self.by_count[count]) >= minimum and len(self._days_per_count[count]) >= day_min:
            indices = self.by_count[count]
        else:
            chosen, days = [], set()
            for k in sorted(self.counts, key=lambda k: (abs(math.log(int(k) / count)), int(k))):
                chosen.extend(self.by_count[int(k)])
                days.update(self._days_per_count[int(k)])
                if len(chosen) >= minimum and len(days) >= day_min:
                    break
            if len(chosen) < minimum or len(days) < day_min:
                raise ValueError("Declared donor pool cannot satisfy bundle/station-day support thresholds")
            indices = np.asarray(chosen, dtype=int)
        self._eligible_cache[count] = indices
        return indices

    def sample_location_window(self, generator):
        """Draw one joint clock interval for an entire location, independently of K."""
        if self.calibration is None:
            raise ValueError("Configure the calibrated location-window law first")
        if generator.random() >= self.calibration["specified_location_tw_probability"]:
            return {"location_tw_status": "missing", "location_tw_start_offset_s": None,
                    "location_tw_end_offset_s": None, "location_tw_source_donor_id": None}
        if self._tw_patterns.empty:
            raise ValueError("No source-supported specified location-window patterns")
        row = self._tw_patterns.iloc[int(generator.integers(len(self._tw_patterns)))]
        return {"location_tw_status": "specified", "location_tw_start_offset_s": float(row.a),
                "location_tw_end_offset_s": float(row.b), "location_tw_source_donor_id": str(row.name)}

    def sample_bundle(self, package_count, rng: np.random.Generator):
        if isinstance(package_count, (bool, np.bool_)):
            raise ValueError("A boolean is not a parcel count")
        k = int(package_count)
        if k != package_count or k <= 0:
            raise ValueError("A bundle contains a positive integer number of parcels")
        options = self.eligible_indices(k)
        donor_index = int(options[int(rng.integers(len(options)))])
        source_k = int(self.bundles.package_count.iloc[donor_index])
        exact = k == source_k
        rows = np.asarray(self._donor_rows[donor_index], dtype=int)
        if exact:
            selected = rows
        elif k <= len(rows):
            selected = rows[rng.choice(len(rows), size=k, replace=False)]
        else:
            selected = rows[rng.integers(len(rows), size=k)]
        parcels = []
        for occurrence, source_index in enumerate(selected):
            clean = {}
            for key, values in self._parcel_arrays.items():
                value = values[source_index]
                if isinstance(value, np.generic):
                    value = value.item()
                clean[key] = None if isinstance(value, float) and not math.isfinite(value) else value
            clean["target_copy_index"] = occurrence
            parcels.append(clean)
        if self.calibration is not None:
            original_service = float(self._donor_service[donor_index])
            total_service = original_service if exact else float(service_curve(k, self.calibration) *
                original_service / service_curve(source_k, self.calibration))
            total_service *= self.calibration["service_scale"]
            selected_service = sum(p["source_planned_service_s"] for p in parcels)
            for p in parcels:
                p["assigned_planned_service_s"] = (total_service * p["source_planned_service_s"] / selected_service
                                                    if selected_service > 0 else total_service / k)
                p["service_count_model_applied"] = not exact or self.calibration["service_scale"] != 1
        return {"package_count": k, "source_donor_id": str(self._donor_ids[donor_index]),
            "source_station_day_id": str(self._donor_days[donor_index]), "source_package_count": int(source_k),
            "exact_source_count": exact, "count_outside_observed_range": k < int(self.counts.min()) or k > int(self.counts.max()),
            "joint_source_bundle_preserved": exact,
            "joint_package_tuple_preserved": True,
            "count_policy": "exact_source_bundle" if exact else "supported_nearest_log_count_joint_tuple_resample",
            "donor_pool": self.calibration["donor_pool"] if self.calibration else "declared_unconditioned",
            "service_donor_total_s": float(self._donor_service[donor_index]) if self.calibration else None,
            "parcels": parcels}


def instantiate_availability(bundle, *, service_day_start_s, mode="recurring_local_clock"):
    """Evaluate a predeclared daily source pattern; never shift a hard promise.

    For a dated source constraint the caller must instead pass its preassigned
    date as service_day_start_s and mode='preassigned_dated_pattern'. That date
    cannot be changed after observing feasibility.
    """
    if mode not in {"recurring_local_clock", "preassigned_dated_pattern"}:
        raise ValueError("Declare recurring or preassigned dated source-window semantics")
    midnight = float(service_day_start_s)
    if not math.isfinite(midnight):
        raise ValueError("Finite target-day origin required")
    result = []
    for mark in bundle["parcels"]:
        rec = dict(mark)
        effective_status = rec.get("location_tw_status", rec["source_tw_status"])
        rec["availability_status"] = effective_status
        if effective_status == "specified":
            start_field = "location_tw_start_offset_s" if "location_tw_status" in rec else "source_tw_start_offset_s"
            end_field = "location_tw_end_offset_s" if "location_tw_status" in rec else "source_tw_end_offset_s"
            rec["availability_start_s"] = midnight + float(rec[start_field])
            rec["availability_end_s"] = midnight + float(rec[end_field])
        else:
            rec["availability_start_s"] = None
            rec["availability_end_s"] = None
        rec["source_window_mode"] = mode
        result.append(rec)
    return result


def reconcile_one_visit(parcels, *, releases_s, deadlines_s, operating_window_s,
                        source_window_end="service_start", missing_window_policy="operating_horizon"):
    """Aggregate marks and derive service-start TW without changing parcel data.

    Promises and shift end are completion constraints. Source availability ends
    are explicitly service-start or completion bounds. No feasibility rejection
    redraws a donor or alters demand. Releases/deadlines are independent business
    inputs; this function never learns them from source service windows.
    """
    if not parcels or len(releases_s) != len(parcels) or len(deadlines_s) != len(parcels):
        raise ValueError("One independent release/deadline entry per parcel required")
    if source_window_end not in {"service_start", "service_completion"}:
        raise ValueError("Explicit source-window end semantics required")
    if missing_window_policy not in {"operating_horizon", "unknown_unsupported"}:
        raise ValueError("Explicit missing-source-window policy required")
    opening, closing = map(float, operating_window_s)
    if not math.isfinite(opening) or not math.isfinite(closing) or closing < opening:
        raise ValueError("Ordered finite operating window required")
    release = [float(v) for v in releases_s]
    due = [float(v) if v is not None else math.inf for v in deadlines_s]
    if not all(math.isfinite(v) for v in release) or not all(v > -math.inf for v in due):
        raise ValueError("Invalid release/deadline")
    service = [finite(r.get("assigned_planned_service_s", r.get("source_planned_service_s"))) for r in parcels]
    volume = [finite(r.get("volume_cm3")) for r in parcels]
    invalid_service = any(v is None or v < 0 for v in service)
    invalid_volume = any(v is None or v <= 0 for v in volume)
    invalid = invalid_service or invalid_volume
    service_sum = float(sum(v for v in service if v is not None))
    volume_sum = float(sum(v for v in volume if v is not None))
    source_missing = sum(r.get("source_tw_status") == "missing" for r in parcels)
    missing = sum(r.get("availability_status", r.get("location_tw_status", r.get("source_tw_status"))) == "missing" for r in parcels)
    reasons = []
    if invalid:
        reasons.append("individual_mark_invalid")
    if missing and missing_window_policy == "unknown_unsupported":
        reasons.append("source_availability_unknown")
    lower = max([opening, *release])
    upper = min([closing - service_sum, *(v - service_sum for v in due)])
    for r in parcels:
        effective_status = r.get("availability_status", r.get("location_tw_status", r.get("source_tw_status")))
        if effective_status not in {"specified", "missing"}:
            reasons.append("source_window_mark_invalid")
        a, b = r.get("availability_start_s"), r.get("availability_end_s")
        if effective_status == "specified" and a is None and b is None:
            reasons.append("source_window_not_instantiated")
        if (a is None) != (b is None):
            reasons.append("partial_availability_constraint")
            continue
        if a is not None:
            a, b = float(a), float(b)
            if not math.isfinite(a) or not math.isfinite(b) or b < a:
                reasons.append("availability_constraint_invalid")
            else:
                lower = max(lower, a)
                upper = min(upper, b - service_sum if source_window_end == "service_completion" else b)
    if lower > upper:
        reasons.append("empty_one_visit_start_window")
    return {"parcel_count": len(parcels), "demand_cm3": None if invalid_volume else volume_sum,
        "planned_service_time_s": None if invalid_service else service_sum, "service_model": "count_conditioned_joint_donor_planned" if any("assigned_planned_service_s" in r for r in parcels) else "additive_source_package_planned",
        "service_is_measured_stop_duration": False,
        "source_specified_window_count": len(parcels) - source_missing, "source_missing_window_count": source_missing,
        "location_specified_window_count": len(parcels) - missing, "location_missing_window_count": missing,
        "source_window_end_semantics": source_window_end, "missing_window_policy": missing_window_policy,
        "tw_start_s": lower, "tw_end_s": upper, "window_is_service_start_interval": True,
        "promises_and_shift_require_completion": True,
        "supported_one_visit_representation": not reasons, "reasons": sorted(set(reasons)),
        "parcel_data_modified": False, "vehicle_feasibility_checked": False}
