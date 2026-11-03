"""Independent saved-run verification; never regenerate or repair demand.

This module intentionally does not import build, demand, marks, views, world,
physical, or routing. Passing means structural/process/numeric contracts hold.
It never certifies a complete multi-vehicle route solution or all-pair shortest
path optimality. Unsupported bundles and explicitly unreachable numeric states
remain visible outcomes rather than being redrawn or discarded.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import JOINT_CHARGING_POLICY, joint_station_budget


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _same(actual, expected, message, *, tolerance=1e-7):
    _require(np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                        rtol=1e-9, atol=tolerance, equal_nan=True), message)


def _bool(values):
    array = np.asarray(values)
    _require(not pd.isna(array).any() and np.isin(array, [True, False, 0, 1]).all(),
             "Boolean field has missing/nonboolean entries")
    return array.astype(bool)


def _ordered_unique(frame, columns, message):
    _require(not frame[columns].isna().any().any() and not frame.duplicated(columns).any(), message)


class _RunVerifier:
    def __init__(self, path, check_sources):
        self.root = Path(path).resolve()
        self.check_sources = bool(check_sources)
        self.errors, self.checks, self.metrics, self.warnings = [], [], {}, []
        self.config = self.manifest = None
        self.instances = {}

    def local(self, name):
        path = (self.root / str(name)).resolve()
        _require(path.is_relative_to(self.root) and path.is_file(), "Missing/external run file: " + str(name))
        return path

    def read(self, name):
        return _json(self.local(name))

    def table(self, name):
        return pd.read_parquet(self.local(name))

    def stage(self, name, operation):
        try:
            operation()
            self.checks.append({"name": name, "passed": True})
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.errors.append(name + ": " + message)
            self.checks.append({"name": name, "passed": False, "error": message})

    def inventory(self):
        self.manifest = self.read("manifest.json")
        self.config = self.read("config.json")
        _require(self.manifest["schema"] == "evrptw_stage2_run_v1", "Unsupported run schema")
        _require(self.manifest["input"] == self.config["input"], "Run/config sources differ")
        files = self.manifest["files"]
        _require(isinstance(files, dict) and files, "Manifest has no file inventory")
        for name, expected in files.items():
            _require(name not in {"manifest.json", "verification.json"}, "Self-referential output hash inventory")
            _require(_sha(self.local(name)) == expected, "Changed run artifact: " + name)
        actual = {str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()}
        _require(actual-{ "manifest.json", "verification.json"} == set(files), "Run contains missing or unmanifested artifacts")
        _require(self.manifest.get("stage1_modified") is False, "Run claims a Stage 1 mutation")
        _require(self.manifest.get("route_solution_claimed") is False, "Generation cannot claim a route solution")
        self.metrics["hashed_output_file_count"] = len(files)

    def sources(self):
        inputs, expected = self.manifest["input"], self.manifest["source_sha256"]
        self.cle = Path(inputs["cle"])
        self.donor_root = Path(inputs["donor_artifact"])
        self.cle_manifest = _json(self.cle / "cle_manifest.json")
        _require(_sha(self.cle / "cle_manifest.json") == expected["cle_manifest"], "CLE manifest changed")
        from .cle_binding import BUILDING_SCHEMA, cle_authority
        freeze_path, certificate_path = cle_authority(self.cle, self.cle_manifest)
        _require(_sha(freeze_path) == expected["freeze"], "Freeze certificate changed")
        if "cle_verification" in expected or self.cle_manifest["schema"] == BUILDING_SCHEMA:
            _require(_sha(certificate_path) == expected.get("cle_verification"),
                     "Independent Stage 1 validation changed")
        _require(_sha(self.donor_root / "manifest.json") == expected["donor_manifest"], "Donor manifest changed")
        _require(_sha(inputs["communities"]) == expected["community"], "Community source changed")
        self.donor_manifest = _json(self.donor_root / "manifest.json")
        _require(self.donor_manifest["schema"] == "evrptw_unconditioned_joint_marks_v1", "Unsupported donor schema")
        _require(set(self.donor_manifest["outputs_sha256"]) == {"parcels.parquet", "bundles.parquet", "quarantine.parquet"},
                 "Incomplete donor output inventory")
        for name, digest in self.donor_manifest["outputs_sha256"].items():
            _require(_sha(self.donor_root / name) == digest, "Changed donor artifact: " + name)
        if self.check_sources:
            for name, item in self.cle_manifest["files"].items():
                path = (self.cle / name).resolve()
                _require(path.is_relative_to(self.cle.resolve()), "External CLE data file")
                _require(_sha(path) == item["sha256"], "Changed CLE source bytes: " + name)
            paths = self.donor_manifest.get("source_paths", {})
            _require(set(paths) == set(self.donor_manifest["source_sha256"]), "Raw donor source paths not available for full-source verification")
            for name, digest in self.donor_manifest["source_sha256"].items():
                _require(_sha(paths[name]) == digest, "Raw donor source bytes changed: " + name)
        self.canonical = pd.read_parquet(self.cle / "locations.parquet", columns=[
            "location_id", "location_type", "candidate_eligible", "residential_units", "residential_type"]).set_index("location_id")
        _require(self.canonical.index.is_unique, "Duplicate canonical source locations")
        self.metrics["source_verification"] = {"cle_manifest_freeze_verification_bytes": True,
            "community_bytes": True, "donor_manifest_and_outputs_bytes": True,
            "all_CLE_and_raw_donor_source_bytes": self.check_sources}

    def world(self):
        self.world_info = self.read("world/world.json")
        self.catalog = self.table("world/catalog.parquet")
        self.depot = self.table("world/depot.parquet")
        _ordered_unique(self.catalog, ["location_id"], "World customer locations are not unique")
        _require(len(self.depot) == 1 and self.depot.location_id.iloc[0] == self.world_info["depot_id"], "World has no unique canonical depot")
        _require(self.world_info["world_id"] == self.manifest["world_id"], "World ID differs from run")
        _require(self.world_info["config"] == self.config["world"], "World parameters differ from run")
        _require(self.world_info["source"]["cle_manifest_sha256"] == self.manifest["source_sha256"]["cle_manifest"], "World/source CLE differ")
        _require(self.world_info["depot_redraw_count"] == 0 and self.world_info["terminal_through_routing_allowed"] is False,
                 "World admits redraws or terminal through-routing")
        _require(len(self.catalog) == self.world_info["customer_count"], "World customer count mismatch")
        _require(self.catalog.world_id.eq(self.world_info["world_id"]).all(), "Catalog world IDs differ")
        g = self.catalog.G.to_numpy(float)
        _require(np.all(np.isfinite(g) & (g > 0)), "Invalid exposure G")
        radius = self.config["world"]["radius_km"]*1000
        for column in ["depot_outward_distance_m", "depot_inward_distance_m"]:
            values = self.catalog[column].to_numpy(float)
            _require(np.all(np.isfinite(values) & (values >= 0) & (values <= radius+1e-6)), "World violates fixed directed service radius")
        for frame, kind in [(self.catalog, "customer"), (self.depot, "depot")]:
            source = self.canonical.loc[frame.location_id]
            _require(source.location_type.eq(kind).all() and _bool(source.candidate_eligible).all(), "World includes wrong/ineligible canonical locations")
        if self.config["world"].get("demand_model") == "binomial_geometric":
            source = self.canonical.loc[self.catalog.location_id]
            _same(self.catalog.residential_units, source.residential_units,
                  "World overwrote raw residential units")
            modeled = self.catalog.modeled_residential_units.to_numpy(float)
            _require(np.all(np.isfinite(modeled) & (modeled >= 1) & (modeled == np.floor(modeled))),
                     "Modeled units must be positive integers")
            city = self.canonical.loc[self.canonical.location_type.eq("customer") & _bool(self.canonical.candidate_eligible)]
            positive = pd.to_numeric(city.residential_units, errors="coerce").gt(0)
            general = float(city.loc[positive, "residential_units"].median())
            raw = pd.to_numeric(source.residential_units, errors="coerce").to_numpy(float)
            known = np.isfinite(raw) & (raw > 0)
            _require(np.array_equal(_bool(self.catalog.units_imputed), ~known), "Units-imputation flags changed")
            expected = np.where(known, np.maximum(1, np.floor(raw+.5)), np.maximum(1, np.floor(general+.5)))
            for kind in city.residential_type.dropna().unique():
                if kind == "residential_type_ambiguous":
                    continue
                cohort = city.loc[positive & city.residential_type.eq(kind), "residential_units"]
                if len(cohort):
                    mask = (~known) & source.residential_type.eq(kind).to_numpy()
                    expected[mask] = max(1, math.floor(float(cohort.median())+.5))
            _same(modeled, expected, "Modeled units do not follow city/type median policy")
            _same(self.catalog.G, modeled, "Binomial model silently perturbs modeled units")
        self.metrics["world_customer_count"] = len(self.catalog)

    def demand(self):
        self.demand_info = self.read("demand/demand.json")
        self.parcels = self.table("demand/parcels.parquet")
        self.sites = self.table("demand/location_dispatch.parquet")
        self.days = self.table("demand/operational_days.parquet")
        p, s, days, op = self.parcels, self.sites, self.days, self.config["operations"]
        _ordered_unique(p, ["parcel_id"], "Parcel IDs are missing/duplicated")
        _ordered_unique(s, ["day_index", "location_id"], "Duplicate dispatch location/day")
        _require(p.location_id.isin(self.catalog.location_id).all(), "Parcel location outside world")
        _require(self.demand_info["process_id"] == self.manifest["process_id"] and self.demand_info["world_id"] == self.world_info["world_id"], "Demand parent identity mismatch")
        _require(self.demand_info["operations"] == op, "Saved demand operations differ from config")
        _require(self.demand_info["parcel_count"] == len(p), "Parcel total mismatch")
        _require(self.demand_info["donor_source_sha256"] == self.donor_manifest["source_sha256"], "Demand donor source version differs")
        _require(self.demand_info["availability_missing_policy"] == op["missing_window_policy"] and self.demand_info["source_window_end"] == op["source_window_end"], "TW policy not explicitly bound")
        _require(op["availability_pattern"] == "recurring_local_clock", "Unsupported availability semantics")
        _require(op["source_window_end"] in {"service_start", "service_completion"}, "Unknown source-window end semantics")
        _require(op["missing_window_policy"] in {"operating_horizon", "unknown_unsupported"}, "Unknown missing-window policy")
        first = date.fromisoformat(op["start_date"])
        _require(len(days) == op["days"] and np.array_equal(days.day_index.to_numpy(), np.arange(op["days"])), "Incomplete/noncontiguous day horizon")
        _require(not _bool(days.route_solution_claimed).any(), "Daily dispatch falsely claims routing success")
        arrival = p.arrival_day_index.to_numpy(float)
        due = p.promise_day_index.to_numpy(float)
        ready = p.ready_s.to_numpy(float)
        preference = p.preferred_weekday.to_numpy(float)
        eligible = _bool(p.eligible_for_deferral)
        _require(np.all(np.isfinite(arrival) & (arrival == np.floor(arrival)) & (arrival >= 0) & (arrival < op["days"])), "Invalid arrival day")
        _require(np.all(np.isfinite(preference) & (preference == np.floor(preference)) & (preference >= 0) & (preference <= 6)), "Invalid fixed service weekday")
        _same(due, arrival+op["promise_lead_days"], "Promise lead is not the declared business rule")
        dates = np.asarray([(first+timedelta(days=d)).isoformat() for d in range(op["days"])])
        _require(np.array_equal(p.arrival_date.astype(str), dates[arrival.astype(int)]), "Arrival date/index mismatch")
        _require(np.all(np.isfinite(ready) & (ready >= arrival*86400+op["ready_window_s"][0]-1e-7) & (ready <= arrival*86400+op["ready_window_s"][1]+1e-7)), "Parcel ready time outside pre-dispatch arrival window")
        _same(p.promise_s, due*86400+op["shift_end_s"], "Promise is not a completion deadline")
        expected = np.full(len(p), np.nan)
        for day in range(op["days"]):
            cohort = arrival == day
            legal = [d for d in range(day, day+op["promise_lead_days"]+1) if (first+timedelta(days=d)).weekday() in op["operating_weekdays"]]
            if not legal:
                continue
            expected[cohort] = legal[0]
            if op["consolidation_mode"] != "daily":
                for weekday in range(7):
                    preferred = [d for d in legal if (first+timedelta(days=d)).weekday() == weekday]
                    if preferred:
                        expected[cohort & eligible & (preference == weekday)] = preferred[0]
        _same(p.scheduled_day_index, expected, "Dispatch calendar depends on an undeclared rule")
        dispatched = np.isfinite(expected) & (expected < op["days"])
        _same(p.dispatch_day_index, np.where(dispatched, expected, np.nan), "Parcel dispatch changed after scheduling")
        _require(np.array_equal(_bool(p.commitment_failure), ~np.isfinite(expected)), "Commitment failure flag inconsistent")
        _require(np.array_equal(p.status.astype(str), np.where(dispatched, "dispatched_to_workload", "backlog")), "Parcel ledger status inconsistent")
        _require(np.all((arrival[dispatched] <= expected[dispatched]) & (expected[dispatched] <= due[dispatched]) & (ready[dispatched] <= expected[dispatched]*86400+op["shift_start_s"]+1e-7)), "Dispatch violates parcel release/promise or occurs before ready")
        previous = int(self.demand_info["initial_backlog"])
        _require(previous == 0, "Unmodeled initial backlog")
        for row in days.itertuples():
            d = int(row.day_index)
            arrived = int(p.arrival_day_index.eq(d).sum())
            dispatched = int(p.dispatch_day_index.eq(d).sum())
            backlog = previous+arrived-dispatched
            _require(row.date == (first+timedelta(days=d)).isoformat(), "Day date mismatch")
            _require((row.arrivals, row.dispatched, row.previous_backlog, row.backlog) == (arrived, dispatched, previous, backlog), "Daily parcel conservation fails")
            _require(row.active_locations == int(s.day_index.eq(d).sum()), "Day active-location total mismatch")
            previous = backlog
        _require(previous == self.demand_info["end_backlog"] == int(p.dispatch_day_index.isna().sum()), "Final backlog mismatch")
        self._dispatch_aggregates(p, s, op)
        if self.config["world"].get("demand_model") == "binomial_geometric":
            settings = self.config["workload"]
            _require(self.demand_info["capacity_conditioning"] == settings, "Workload calibration is not bound to config")
            successful = s.loc[_bool(s.generation_supported)]
            _require(successful.demand_cm3.gt(0).all(), "Nonpositive successful workload")
            arrival_volumes = p.groupby(["arrival_day_index", "location_id"], sort=False).volume_cm3.sum()
            _require(arrival_volumes.gt(0).all() and arrival_volumes.le(settings["cargo_capacity_cm3"]).all(),
                     "Capacity-invalid arrival proposal exported as generated")
            attempts = p.capacity_conditioning_attempts.to_numpy(float)
            _require(np.all((attempts >= 1) & (attempts <= settings["maximum_full_proposals"])), "Illegal proposal count")
            units = p.location_id.map(self.catalog.set_index("location_id").modeled_residential_units).to_numpy(float)
            _same(p.modeled_residential_units, units, "Arrival generation changed modeled units")
            active = p.active_units.to_numpy(float)
            _require(np.all(np.isfinite(active) & (active >= 1) & (active <= units) & (active == np.floor(active))), "Active-unit counts outside modeled location")
            by_arrival = p.groupby(["arrival_day_index", "location_id"], sort=False)
            _require(by_arrival.active_units.nunique().eq(1).all(), "One arrival location has inconsistent active units")
            _require((by_arrival.size() >= by_arrival.active_units.first()).all(), "Active unit has zero parcel")
            failure = self.table("demand/generation_failures.parquet")
            _require(len(failure) == self.demand_info["capacity_proposal_rejections"], "Rejected proposals omitted")
            bad = s.loc[~_bool(s.generation_supported)]
            _require(len(bad) == self.demand_info["generation_unsupported_location_days"], "Unsupported location-day omitted")
            _require(bad.parcel_count.eq(0).all() and bad.demand_cm3.isna().all() and
                     (~_bool(bad.supported_one_visit_representation)).all(), "Failed proposal masquerades as valid workload")
            if len(failure):
                _require(failure.location_id.isin(self.catalog.location_id).all(), "Proposal failure outside fixed world")
                _require(failure.proposed_volume_cm3.gt(settings["cargo_capacity_cm3"]).all(), "Capacity rejected a supported positive-volume proposal")
                exhausted = failure.loc[_bool(failure.exhausted)]
                _require(exhausted.proposal_index.eq(settings["maximum_full_proposals"]).all(), "Premature proposal exhaustion")
                _require(set(zip(exhausted.day_index, exhausted.location_id)) == set(zip(bad.day_index, bad.location_id)), "Exhausted locations silently removed")
            self.metrics["capacity_rejected_proposals"] = len(failure)
            self.metrics["generation_unsupported_location_days"] = len(bad)
        self.metrics.update(parcel_count=len(p), location_day_count=len(s), end_backlog=previous,
            commitment_failure_parcels=int(p.commitment_failure.sum()), unsupported_location_days=int((~_bool(s.supported_one_visit_representation)).sum()))

    @staticmethod
    def _dispatch_aggregates(parcels, sites, operations):
        q = parcels.dropna(subset=["dispatch_day_index"]).copy()
        _require(q.source_tw_status.isin(["specified", "missing"]).all(), "Unusable raw window mark entered the primary process")
        service_field = "assigned_planned_service_s" if "assigned_planned_service_s" in q else "source_planned_service_s"
        service, volume = q[service_field].to_numpy(float), q.volume_cm3.to_numpy(float)
        _require(np.all(np.isfinite(service) & (service >= 0)) and np.all(np.isfinite(volume) & (volume > 0)), "Unusable service/dimension mark entered primary process")
        missing = q.source_tw_status.eq("missing")
        specified = ~missing
        a, b = q.source_tw_start_offset_s.to_numpy(float), q.source_tw_end_offset_s.to_numpy(float)
        _require(np.all(np.isfinite(a[specified]) & np.isfinite(b[specified]) & (b[specified] >= a[specified])), "Specified availability offsets are invalid")
        _require(np.all(np.isnan(a[missing]) & np.isnan(b[missing])), "Missing availability was silently imputed into source marks")
        if "location_tw_status" in q:
            _require(q.location_tw_status.isin(["specified", "missing"]).all(), "Invalid transferred location TW")
            window_missing = q.location_tw_status.eq("missing").to_numpy()
            a = q.location_tw_start_offset_s.to_numpy(float)
            b = q.location_tw_end_offset_s.to_numpy(float)
            _require(np.all(np.isnan(a[window_missing]) & np.isnan(b[window_missing])), "Missing location TW lost its null provenance")
            _require(np.all(np.isfinite(a[~window_missing]) & np.isfinite(b[~window_missing]) & (a[~window_missing] <= b[~window_missing])), "Invalid joint location window")
            specified = ~window_missing
        else:
            window_missing = missing.to_numpy()
        q["day_index"] = q.dispatch_day_index.astype(int)
        q["_missing"] = missing.astype(int)
        q["_a"] = np.where(specified, a+q.day_index.to_numpy()*86400, -np.inf)
        q["_b"] = np.where(specified, b+q.day_index.to_numpy()*86400, np.inf)
        totals = q.groupby(["day_index", "location_id"], sort=False).agg(
            parcel_count=("parcel_id", "size"), demand_cm3=("volume_cm3", "sum"),
            planned_service_time_s=(service_field, "sum"),
            ready=("ready_s", "max"), promise=("promise_s", "min"),
            source_missing_window_count=("_missing", "sum"), a=("_a", "max"), b=("_b", "min"))
        actual = sites.loc[_bool(sites.generation_supported)] if "generation_supported" in sites else sites
        actual = actual.set_index(["day_index", "location_id"])
        _require(set(totals.index) == set(actual.index), "Dispatch site ledger omits/adds parcel groups")
        totals = totals.reindex(actual.index)
        midnight = actual.index.get_level_values("day_index").to_numpy(float)*86400
        total_service = totals.planned_service_time_s.to_numpy(float)
        totals["tw_start_s"] = np.maximum.reduce([midnight+operations["shift_start_s"], totals.ready.to_numpy(float), totals.a.to_numpy(float)])
        end_availability = totals.b.to_numpy(float)-(total_service if operations["source_window_end"] == "service_completion" else 0)
        totals["tw_end_s"] = np.minimum.reduce([midnight+operations["shift_end_s"]-total_service, totals.promise.to_numpy(float)-total_service, end_availability])
        totals["source_specified_window_count"] = totals.parcel_count-totals.source_missing_window_count
        empty = totals.tw_start_s.to_numpy() > totals.tw_end_s.to_numpy()
        q["_window_missing"] = window_missing.astype(int)
        applied_missing = q.groupby(["day_index", "location_id"], sort=False)._window_missing.sum().reindex(actual.index).to_numpy()
        unknown = (applied_missing > 0) & (operations["missing_window_policy"] == "unknown_unsupported")
        _require(np.array_equal(_bool(actual.supported_one_visit_representation), ~(empty | unknown)), "One-visit support flag incorrect")
        for field in ["parcel_count", "demand_cm3", "planned_service_time_s", "tw_start_s", "tw_end_s", "source_specified_window_count", "source_missing_window_count"]:
            _same(actual[field], totals[field], "Site aggregation differs from immutable parcels: " + field)
        expected_reasons = {(False, False): [], (True, False): ["empty_one_visit_start_window"],
            (False, True): ["source_availability_unknown"], (True, True): ["empty_one_visit_start_window", "source_availability_unknown"]}
        _require(all(json.loads(reason) == expected_reasons[(bool(e), bool(u))]
                     for reason, e, u in zip(actual.reasons, empty, unknown)), "One-visit unsupported reasons differ")

    def donor_tuples(self):
        p = self.parcels
        source = pd.read_parquet(self.donor_root / "parcels.parquet")
        _ordered_unique(source, ["source_parcel_ref"], "Donor canonical parcel IDs are not unique")
        source = source.set_index("source_parcel_ref")
        selected = source.loc[p.source_parcel_ref].reset_index(drop=True)
        _require(selected.source_kind.eq("delivered").all() and _bool(selected.individual_marks_valid).all(), "Target parcel uses unsupported donor layer")
        for field in ["source_event_ref", "source_station_day_id", "source_route_ref", "source_date", "source_tw_status"]:
            _require(np.array_equal(p[field].astype(str).to_numpy(), selected[field].astype(str).to_numpy()), "Joint source tuple/provenance changed: " + field)
        for field in ["depth_cm", "height_cm", "width_cm", "volume_cm3", "source_planned_service_s", "source_tw_start_offset_s", "source_tw_end_offset_s"]:
            _same(p[field], selected[field], "Joint source numeric mark changed: " + field)
        _require(np.array_equal(p.source_donor_id.astype(str), selected.donor_id.astype(str)), "Target parcel does not belong to its declared source bundle")
        _same(p.volume_cm3, p.depth_cm*p.height_cm*p.width_cm, "Parcel volume differs from source dimensions")
        _require(np.all(np.isfinite(p.volume_cm3.to_numpy(float)) & p.volume_cm3.gt(0)), "Invalid target volume")
        _require(np.all(np.isfinite(p.source_planned_service_s.to_numpy(float)) & p.source_planned_service_s.ge(0)), "Invalid target planned service")
        bundles = pd.read_parquet(self.donor_root / "bundles.parquet")
        bundles = bundles.loc[bundles.source_kind.eq("delivered") & bundles.individual_marks_valid].set_index("donor_id")
        _require(bundles.index.is_unique and len(bundles), "No unique valid donor pool")
        calibrated = self.config["world"].get("demand_model") == "binomial_geometric"
        if calibrated:
            from .calibration import CALIBRATION_SOURCE_SHA256, GEN_TRAIN_STATION_DAY_IDS
            _require(self.donor_manifest["source_sha256"] == CALIBRATION_SOURCE_SHA256, "Calibrated source bytes differ")
            bundles = bundles.loc[bundles.source_station_day_id.isin(GEN_TRAIN_STATION_DAY_IDS)]
            _require(p.source_station_day_id.isin(GEN_TRAIN_STATION_DAY_IDS).all(), "Held-out donor leaked into target workload")
            summary = self.demand_info["donor_calibration"]
            _require(summary["donor_pool"] == "GEN-TRAIN" and summary["bundles"] == len(bundles) and
                     summary["station_days"] == bundles.source_station_day_id.nunique(), "Calibrated pool count/provenance mismatch")
        counts = sorted(set(map(int, bundles.package_count)))
        grouped = p.groupby(["arrival_day_index", "location_id"], sort=False)
        summary = grouped.agg(k=("parcel_id", "size"), donor=("source_donor_id", "first"),
            donor_ids=("source_donor_id", "nunique"), source_ks=("source_package_count", "nunique"),
            unique_source_parcels=("source_parcel_ref", "nunique"))
        _require(summary.donor_ids.eq(1).all() and summary.source_ks.eq(1).all(), "Arrival bundle combines unrelated donor bundles")
        source_k = p.source_donor_id.map(bundles.package_count).to_numpy(float)
        actual_source_counts = source.groupby("donor_id", sort=False).size()
        _same(source_k, p.source_donor_id.map(actual_source_counts), "Donor metadata size differs from its canonical parcel population")
        k = grouped.parcel_id.transform("size").to_numpy(int)
        if calibrated:
            settings = self.config["workload"]
            eligible = {}
            by_count = {int(v): group for v, group in bundles.groupby("package_count")}
            for target in np.unique(k):
                target = int(target)
                group = by_count.get(target)
                if group is not None and len(group) >= settings["minimum_donor_bundles"] and group.source_station_day_id.nunique() >= settings["minimum_donor_station_days"]:
                    allowed = {target}
                else:
                    allowed, days, number = set(), set(), 0
                    for candidate in sorted(counts, key=lambda n: (abs(math.log(n / target)), n)):
                        allowed.add(candidate)
                        number += len(by_count[candidate])
                        days.update(by_count[candidate].source_station_day_id)
                        if number >= settings["minimum_donor_bundles"] and len(days) >= settings["minimum_donor_station_days"]:
                            break
                    _require(number >= settings["minimum_donor_bundles"] and len(days) >= settings["minimum_donor_station_days"], "Insufficient declared count support")
                eligible[target] = allowed
            _require(all(int(source) in eligible[int(target)] for source, target in zip(source_k, k)), "Donor outside declared nearest-count support neighborhood")
        else:
            mapping = {int(v): int(v) if v in counts else min(counts, key=lambda n: (abs(math.log(n)-math.log(v)), n)) for v in np.unique(k)}
            expected_k = np.asarray([mapping[v] for v in k])
            _same(source_k, expected_k, "Unsupported count extrapolation rule")
        _same(p.source_package_count, source_k, "Source bundle count metadata incorrect")
        _require(np.array_equal(_bool(p.exact_source_count), source_k == k), "Exact donor-count flag incorrect")
        _require(np.array_equal(_bool(p.source_count_extrapolated), (k < counts[0]) | (k > counts[-1])), "Observed count-range flag incorrect")
        group_source_k = summary.donor.map(bundles.package_count).to_numpy(int)
        without_replacement = summary.k.to_numpy() <= group_source_k
        _require(np.all(summary.unique_source_parcels.to_numpy()[without_replacement] == summary.k.to_numpy()[without_replacement]), "Exact-count bundle changed or downsampling duplicated a parcel")
        if calibrated:
            def central(n):
                n = np.asarray(n, dtype=float)
                cut = settings["service_support_max_count"]
                anchor = settings["service_a_s"]+settings["service_b_s"]*cut**settings["service_alpha"]
                return np.where(n <= cut, settings["service_a_s"]+settings["service_b_s"]*n**settings["service_alpha"],
                                anchor*(n/cut)**settings["service_tail_gamma"])
            raw_sum = source.groupby("donor_id", sort=False).source_planned_service_s.sum()
            source_service = summary.donor.map(raw_sum).to_numpy(float)
            expected_service = central(summary.k.to_numpy())*source_service/central(group_source_k)*settings["service_scale"]
            actual_service = grouped.assigned_planned_service_s.sum().reindex(summary.index).to_numpy(float)
            _same(actual_service, expected_service, "Assigned service broke the joint donor/count residual law")
            _require(np.isfinite(p.assigned_planned_service_s.to_numpy(float)).all() and p.assigned_planned_service_s.ge(0).all(), "Invalid assigned planned service")
            dispatched = p.dropna(subset=["dispatch_day_index"])
            location = dispatched.groupby(["dispatch_day_index", "location_id"], sort=False)
            for name in ["location_tw_status", "location_tw_start_offset_s", "location_tw_end_offset_s", "location_tw_source_donor_id"]:
                _require(location[name].nunique(dropna=False).le(1).all(), "A dispatched location has multiple TWs")
            specified = dispatched.loc[dispatched.location_tw_status.eq("specified")].drop_duplicates(["dispatch_day_index", "location_id"])
            _require(specified.location_tw_source_donor_id.isin(bundles.index).all(), "Transferred TW is not from training pool")
            if len(specified):
                raw_windows = source.loc[source.donor_id.isin(specified.location_tw_source_donor_id)].copy()
                raw_windows["_a"] = np.where(raw_windows.source_tw_status.eq("specified"), raw_windows.source_tw_start_offset_s, -np.inf)
                raw_windows["_b"] = np.where(raw_windows.source_tw_status.eq("specified"), raw_windows.source_tw_end_offset_s, np.inf)
                patterns = raw_windows.groupby("donor_id", sort=False).agg(a=("_a", "max"), b=("_b", "min"))
                _same(specified.location_tw_start_offset_s, specified.location_tw_source_donor_id.map(patterns.a), "Location TW start differs from joint donor pattern")
                _same(specified.location_tw_end_offset_s, specified.location_tw_source_donor_id.map(patterns.b), "Location TW end differs from joint donor pattern")
        self.metrics["joint_tuple_source_crosscheck"] = True

    def views(self):
        info = self.read("views/views.json")
        self.view_table = self.table("views/views.parquet")
        self.order = self.table("views/selection_order.parquet")
        _ordered_unique(self.view_table, ["view_id"], "Duplicate view IDs")
        _ordered_unique(self.order, ["day_index", "rank"], "Duplicate view priority ranks")
        _ordered_unique(self.order, ["day_index", "location_id"], "Repeated location in daily view order")
        _require(info["parent_process_id"] == self.demand_info["process_id"], "View belongs to another demand process")
        _require(info["full_site_bundles_preserved"] is True and info["nested_prefixes"] is True and info["selection_depends_on_vehicle_or_N"] is False,
                 "View lacks immutable full-bundle prefix semantics")
        _require(info["unselected_parcels_are_parent_day_workload_not_backlog"] is True, "View misclassifies unselected parent workload")
        _require(info["sampler"] == self.config["views"]["sampler"] and self.view_table.kind.isin(["operational_day", "benchmark_view"]).all(), "Undeclared view kind or sampling policy")
        for day in self.days.itertuples():
            d = int(day.day_index)
            group = self.order.loc[self.order.day_index.eq(d)].sort_values("rank")
            active = self.sites.loc[self.sites.day_index.eq(d)]
            _require(np.array_equal(group["rank"].to_numpy(), np.arange(len(group))) and set(group.location_id) == set(active.location_id), "Priority list differs from full active day")
            records = self.view_table.loc[self.view_table.day_index.eq(d)]
            natural = records.loc[records.kind.eq("operational_day")]
            fixed = records.loc[records.kind.eq("benchmark_view")]
            _require(len(natural) == 1 and sorted(fixed.requested_n.astype(int)) == self.config["views"]["sizes"], "Missing natural day or fixed-N views")
            _require(natural.requested_n.eq(len(active)).all() and fixed.sampler.eq(self.config["views"]["sampler"]).all(), "Natural workload size or fixed-N sampler was changed")
            for row in records.itertuples():
                supported = row.kind == "operational_day" or len(active) >= row.requested_n
                selected = len(active) if row.kind == "operational_day" else row.requested_n if supported else 0
                materialize = d in self.config["views"]["materialize_day_indices"] and supported and (
                    row.kind != "operational_day" or len(active) <= self.config["views"]["natural_day_dense_matrix_limit"])
                _require(bool(row.supported_size) == supported and row.available_n == len(active) and row.selected_n == selected,
                         "Fixed-N support/coverage numbers are incorrect")
                _require(row.parent_operational_day_id == day.operational_day_id and bool(row.materialize_matrix) == materialize,
                         "View parent or materialization policy differs from config")
        self.metrics["view_count"] = len(self.view_table)
        self.metrics["unsupported_size_views"] = int((~_bool(self.view_table.supported_size)).sum())

    def physical(self):
        self.physical_info = self.read("physical/physical.json")
        self.chargers = self.table("physical/chargers.parquet")
        c, config = self.chargers, self.config["physical"]
        _ordered_unique(c, ["location_id"], "Duplicate physical charging locations")
        _require(self.physical_info["config"] == config and self.physical_info["world_id"] == self.world_info["world_id"], "Physical scenario depends on another world/config")
        source = self.canonical.loc[c.location_id]
        _require(source.location_type.eq("charging").all() and _bool(source.candidate_eligible).all(), "Charging scenario includes excluded canonical sources")
        _require(c.selected_connector.isin(["J1772", "J1772COMBO"]).all(), "Unsupported reference-vehicle connector")
        configurations = pd.read_parquet(self.cle / "charging/charging_configurations.parquet")
        configurations = configurations.loc[_bool(configurations.is_selected)].set_index("configuration_id")
        _require(configurations.index.is_unique, "Ambiguous frozen selected charging configuration")
        original = configurations.loc[c.selected_configuration_id].reset_index(drop=True)
        _require(np.array_equal(original.charging_location_id.astype(str), c.charging_location_id.astype(str)), "Physical charging configuration belongs to another location")
        _require(np.array_equal(original.connector.astype(str), c.selected_connector.astype(str)), "Physical charging connector changed from frozen configuration")
        _same(c.source_assigned_power_kw, original.assigned_power_kw, "Physical source power changed from frozen configuration")
        _same(c.source_reported_power_kw, original.reported_power_kw, "Physical reported source power changed")
        _require(np.array_equal(_bool(c.source_power_is_reported), _bool(original.power_is_reported)), "Physical power imputation status changed")
        caps = np.where(c.reference_charge_mode.eq("dc_fast"), config["vehicle"]["dc_cap_kw"], config["vehicle"]["ac_cap_kw"])
        _require(c.reference_charge_mode.isin(["dc_fast", "ac_level2"]).all(), "Unknown charging mode")
        values = c.source_assigned_power_kw.to_numpy(float)
        _require(np.all(np.isfinite(values) & (values > 0)), "Invalid source assigned charging power")
        _same(c.stage2_vehicle_cap_kw, caps, "Incorrect vehicle power cap")
        _same(c.stage2_capped_pre_derating_kw, np.minimum(values, caps), "Source power cap is incorrect")
        _same(c.stage2_derating_factor, config["charging"]["derating_factor"], "Incorrect charging derating")
        _same(c.effective_power_kw, np.minimum(values, caps)*config["charging"]["derating_factor"], "Charging derating not applied exactly once")
        _require(c.derating_application_count.eq(1).all(), "Charging multiplier applied more than once")
        for field in ["opening_hours_policy", "queue_policy"]:
            _require(c[field].eq(config["charging"][field]).all(), "Charging availability/queue assumption changed")
        for field in ["depot_outward_distance_m", "depot_inward_distance_m"]:
            d = c[field].to_numpy(float)
            _require(np.all(np.isfinite(d) & (d >= 0) & (d <= self.config["world"]["radius_km"]*1000+1e-6)), "Charger outside fixed directed catchment")
        _require(len(c) == self.physical_info["charger_count"], "Charging count mismatch")
        self.metrics["compatible_charging_locations"] = len(c)

    def _instance(self, row):
        prefix = "instances/" + str(row.view_id) + "/"
        info = self.read(prefix+"instance.json")
        terminals = self.table(prefix+"terminals.parquet")
        _require(info["schema"] == "evrptw_scenario_instance_v1", "Unsupported instance schema")
        _require(info["view_id"] == row.view_id and info["world_id"] == self.world_info["world_id"] and info["process_id"] == self.demand_info["process_id"], "Instance parent identity mismatch")
        _require(info["vehicle"] == self.config["physical"]["vehicle"], "Instance vehicle changed parent demand")
        _require(info["operating_window_s"] == [self.config["operations"]["shift_start_s"], self.config["operations"]["shift_end_s"]], "Instance changed its operating window")
        _require(info["assessment"]["route_solution_claimed"] is False, "Instance claims a route solution")
        for field in ["day_index", "kind", "requested_n", "selected_n", "supported_size"]:
            _require(info[field] == getattr(row, field), "Instance/view field mismatch: " + field)
        d, n = int(row.day_index), int(row.selected_n)
        priority = self.order.loc[self.order.day_index.eq(d)].sort_values("rank").location_id.astype(str).tolist()
        ids = priority[:n] if row.supported_size else []
        selection = self.read(prefix+"charging_selection.json")
        _require(selection == info["charging_selection"], "Charging selection differs from instance metadata")
        charger_ids = selection["selected_location_ids"]
        _require(len(set(charger_ids)) == len(charger_ids) and set(charger_ids).issubset(set(self.chargers.location_id.astype(str))), "Instance invented or duplicated charging locations")
        selected_chargers = self.chargers.set_index("location_id").loc[charger_ids]
        charging = self.config["physical"]["charging"]
        policy = charging["supply_policy"]
        if policy in {"road_detour_budget", JOINT_CHARGING_POLICY} and row.materialize_matrix:
            budget = (joint_station_budget(charging, int(row.requested_n))
                      if policy == JOINT_CHARGING_POLICY else charging["external_cs_budget"])
            _require(selection["requested_budget"] == budget and selection["selected_count"] == len(charger_ids), "Charging budget metadata differs")
            _require(bool(selection["supported_budget"]) == (len(charger_ids) == budget), "Charging shortage is hidden")
            _require(selection["candidate_count"] == len(self.chargers) and len(charger_ids) <= budget, "Charging supply count differs")
            _require(selection["road_distance_source"] == "distance_shortest_path_distance", "Charger selector used an inconsistent path family")
            if policy == JOINT_CHARGING_POLICY:
                _require(selection["policy"] == policy and selection["road_time_source"] == "distance_shortest_path_time" and
                         selection["reference_departure_soc"] == charging["reference_departure_soc"],
                         "Joint charging selector policy differs")
                trace = selection["trace"]
                _require([entry["location_id"] for entry in trace] == charger_ids and
                         [entry["rank"] for entry in trace] == list(range(1, len(charger_ids)+1)),
                         "Joint charging trace differs from selected supply")
                counts = selection["phase_counts"]
                _require(set(counts) == {"between_joint", "local_joint", "local_protection", "padding"} and
                         sum(counts.values()) == len(trace) and all(
                             count == sum(item["phase"] == phase for item in trace) for phase, count in counts.items()),
                         "Joint charging selection phase counts differ")
        expected_ids = [str(self.depot.location_id.iloc[0]), *ids, *charger_ids]
        _require(info["node_order"] == expected_ids == terminals.location_id.astype(str).tolist(), "Instance terminal ordering violates complete source bundle/prefix")
        _require(np.array_equal(terminals.node_index.to_numpy(), np.arange(len(terminals))), "Noncontiguous node indices")
        _require(terminals.kind.tolist() == ["depot"]+["customer"]*n+["charging"]*len(charger_ids), "Node kind order incorrect")
        sites = self.sites.loc[self.sites.day_index.eq(d)].set_index("location_id").loc[ids]
        customers = terminals.iloc[1:1+n]
        noncustomers = terminals.loc[~terminals.kind.eq("customer")]
        _same(noncustomers.demand_cm3, np.zeros(len(noncustomers)), "Noncustomer cargo demand is nonzero")
        _same(noncustomers.service_time_s, np.zeros(len(noncustomers)), "Noncustomer service-time policy changed")
        for actual, expected in [("demand_cm3", "demand_cm3"), ("service_time_s", "planned_service_time_s"), ("parcel_count", "parcel_count")]:
            _same(customers[actual], sites[expected], "Instance split or changed the parcel bundle: " + actual)
        for field in ["tw_start_s", "tw_end_s"]:
            _same(customers[field], sites[field]-d*86400, "Instance TW local-day conversion incorrect")
        _require(info["assessment"]["unsupported_one_visit_locations"] == int((~_bool(sites.supported_one_visit_representation)).sum()), "Instance hides unsupported one-visit bundles")
        _require(info["assessment"]["oversize_bundles"] == int(sites.demand_cm3.gt(info["vehicle"]["cargo_capacity_cm3"]).sum()), "Instance hides oversized bundles")
        generation_bad = int((~_bool(sites.generation_supported)).sum()) if "generation_supported" in sites else 0
        _require(info["assessment"].get("unsupported_generation_locations", 0) == generation_bad, "Instance hides proposal-generation failures")
        if len(selected_chargers):
            inherited = terminals.iloc[1+n:]
            for field in ["source_assigned_power_kw", "stage2_vehicle_cap_kw", "stage2_capped_pre_derating_kw", "stage2_derating_factor", "effective_power_kw"]:
                _same(inherited[field], selected_chargers[field], "Instance charging supply/power changed: " + field)
        _require(bool(info["matrix_materialized"]) == bool(row.materialize_matrix), "Instance omitted/invented a requested matrix")
        if not info["matrix_materialized"]:
            _require(info["status"] == ("not_materialized" if row.supported_size else "unsupported_size"), "Nonmaterialized status incorrect")
            self.instances[row.view_id] = (info, None)
            return
        with np.load(self.local(prefix+"costs.npz"), allow_pickle=False) as saved:
            required_arrays = {"distance_m", "time_s", "energy_kwh", "source_option", "destination_option", "path_mode",
                "distance_shortest_path_distance", "distance_shortest_path_time", "time_shortest_path_distance",
                "time_shortest_path_time", "time_shortest_path_energy_kwh", "time_shortest_path_source_option",
                "time_shortest_path_destination_option", "time_shortest_path_mode"}
            _require(set(saved.files) == required_arrays, "Paired cost bundle schema mismatch")
            arrays = {key: saved[key].copy() for key in saved.files}
        count = len(terminals)
        for key, array in arrays.items():
            _require(array.shape == (count, count), "Cost array/terminal dimensions differ: " + key)
        for key in ["source_option", "destination_option", "path_mode", "time_shortest_path_source_option",
                    "time_shortest_path_destination_option", "time_shortest_path_mode"]:
            _require(arrays[key].dtype.kind in {"i", "u"}, "Path incidence array is not integer typed")
        _require(np.isin(arrays["path_mode"], [0, 1, 2, 3]).all(), "Unknown path-incidence mode")
        _require(np.isin(arrays["time_shortest_path_mode"], [0, 1, 2, 3]).all(), "Unknown fastest-path incidence mode")
        for key in ["distance_m", "time_s", "energy_kwh", "distance_shortest_path_distance", "distance_shortest_path_time",
                    "time_shortest_path_distance", "time_shortest_path_time", "time_shortest_path_energy_kwh"]:
            array = arrays[key]
            _require(array.dtype.kind == "f" and not np.isnan(array).any() and not np.isneginf(array).any() and np.all(array >= -1e-7), "Illegal numeric matrix values: " + key)
            _same(np.diag(array), np.zeros(count), "Nonzero cost diagonal: " + key)
        finite = np.isfinite(arrays["time_s"])
        _require(np.array_equal(finite, np.isfinite(arrays["distance_m"])) and np.array_equal(finite, np.isfinite(arrays["energy_kwh"])), "D/T/E reachability masks differ")
        _require(np.array_equal(arrays["path_mode"] == 3, ~finite), "Path mode hides unreachable directed pairs")
        rate = float(info["vehicle"]["specific_energy_kwh_per_km"])
        _same(arrays["energy_kwh"][finite], arrays["distance_m"][finite]*rate/1000, "Energy not derived from the same path distance")
        _same(arrays["distance_m"], arrays["distance_shortest_path_distance"], "Execution distance is not shortest-distance distance")
        _same(arrays["time_s"], arrays["distance_shortest_path_time"], "Execution time belongs to another path")
        _require(np.array_equal(finite, np.isfinite(arrays["time_shortest_path_time"])) and
                 np.array_equal(finite, np.isfinite(arrays["time_shortest_path_distance"])), "Paired path policies have different reachability")
        _require(np.array_equal(arrays["time_shortest_path_mode"] == 3, ~finite), "Fastest-path mode hides unreachable pairs")
        _same(arrays["time_shortest_path_energy_kwh"][finite], arrays["time_shortest_path_distance"][finite]*rate/1000, "Fastest energy belongs to another path")
        _require(np.all(arrays["distance_m"][finite] <= arrays["time_shortest_path_distance"][finite]+1e-6), "Shortest distance exceeds fastest-path distance")
        _require(np.all(arrays["time_shortest_path_time"][finite] <= arrays["time_s"][finite]+1e-6), "Fastest time exceeds shortest-distance-path time")
        unsupported_workload = generation_bad or info["assessment"]["oversize_bundles"] or info["assessment"]["unsupported_one_visit_locations"]
        expected_status = "unsupported_workload" if unsupported_workload else "unsupported_charging_budget" if selection.get("supported_budget") is False else "materialized" if finite.all() else "numeric_unreachable"
        _require(info["status"] == expected_status, "Instance hides unsupported or unreachable state")
        policy = info["cost_policy"]
        expected_day = (date.fromisoformat(self.config["operations"]["start_date"])+timedelta(days=d)).weekday()
        _require(policy["path_policy"] == ("shortest_distance_weekend" if expected_day >= 5 else "shortest_distance_weekday") and policy["distance_is_same_time_path"] is True and policy["terminal_through_routing_allowed"] is False and
                 policy["execution_policy"] == "shortest_distance" and info["primary_objective"] == "total_distance",
                 "Instance cost policy is not one bound directed physical path")
        self._audit_paths(self.read(prefix+"path_audit.json"), info, arrays, expected_policy="shortest_distance")
        fastest_arrays = {"distance_m": arrays["time_shortest_path_distance"], "time_s": arrays["time_shortest_path_time"],
                          "energy_kwh": arrays["time_shortest_path_energy_kwh"]}
        self._audit_paths(self.read(prefix+"fastest_path_audit.json"), info, fastest_arrays, expected_policy="fastest_time")
        opening, closing = info["operating_window_s"]
        earliest = np.maximum(customers.tw_start_s.to_numpy(float), opening+arrays["time_s"][0, 1:1+n])
        vehicle = info["vehicle"]
        witnesses = ((earliest <= customers.tw_end_s.to_numpy(float)) &
                     (earliest+customers.service_time_s.to_numpy(float)+arrays["time_s"][1:1+n, 0] <= closing) &
                     (customers.demand_cm3.to_numpy(float) <= vehicle["cargo_capacity_cm3"]) &
                     (arrays["energy_kwh"][0, 1:1+n]+arrays["energy_kwh"][1:1+n, 0] <=
                      vehicle["battery_capacity_kwh"]*(vehicle["initial_soc"]-vehicle["reserve_soc"])))
        _require(info["assessment"]["direct_single_stop_certificate_count"] == int(witnesses.sum()),
                 "Saved sufficient direct-trip witness count is incorrect")
        self.instances[row.view_id] = (info, arrays)

    def _audit_paths(self, audit, info, arrays, *, expected_policy):
        _require(isinstance(audit, dict) and audit, "Materialized cost bundle has no saved path audit")
        day = "weekend" if info["cost_policy"]["path_policy"].endswith("weekend") else "weekday"
        if not hasattr(self, "speed_lookup"):
            speed = pd.read_parquet(self.cle / "speeds/directed_speeds.parquet", columns=[
                "edge_u", "edge_v", "edge_key", "length_m", "in_public_graph",
                "reference_travel_time_weekday_s", "reference_travel_time_weekend_s"])
            self.speed_lookup = {tuple(map(str, identity)): (float(length), bool(public), float(weekday), float(weekend))
                for identity, (length, public, weekday, weekend) in zip(
                    speed[["edge_u", "edge_v", "edge_key"]].itertuples(index=False, name=None),
                    speed[["length_m", "in_public_graph", "reference_travel_time_weekday_s", "reference_travel_time_weekend_s"]].itertuples(index=False, name=None))}
        for key, record in audit.items():
            i, j = map(int, key.split(","))
            _require(record.get("path_policy") == expected_policy, "Audit path is labeled with another objective")
            _require(0 <= i < len(info["node_order"]) and 0 <= j < len(info["node_order"]), "Invalid path audit indices")
            _require(record["source"] == info["node_order"][i] and record["destination"] == info["node_order"][j], "Path audit references another terminal ordering")
            finite = bool(np.isfinite(arrays["time_s"][i, j]))
            _require(bool(record["reachable"]) == finite, "Path audit reachability differs")
            if not finite:
                _require(all(record[k] is None for k in ["distance_m", "time_s", "energy_kwh"]), "Unreachable audit must use explicit JSON null costs")
                continue
            parts = record["segments"]
            for field in ["distance_m", "time_s", "energy_kwh"]:
                _same(record[field], arrays[field][i, j], "Saved path audit differs from numeric closure")
            _same(sum(x["length_m"] for x in parts), arrays["distance_m"][i, j], "Audited path lengths do not sum to D")
            _same(sum(x["time_s"] for x in parts), arrays["time_s"][i, j], "Audited path times do not sum to T")
            for part in parts:
                length, time = float(part["length_m"]), float(part["time_s"])
                _require(np.isfinite([length, time]).all() and length >= 0 and time >= 0, "Invalid path segment costs")
                if part["kind"] == "geometry_connector":
                    _require(part["source_edge"] is None, "Geometric connector mislabeled as observed native road")
                    _same(time, length*3.6/info["cost_policy"]["connector_speed_kph"], "Geometry connector proxy time differs")
                else:
                    identity = tuple(map(str, part["source_edge"]))
                    source_length, public, weekday, weekend = self.speed_lookup[identity]
                    _require(length <= source_length+1e-6, "Native path segment exceeds source edge")
                    _same(time, (weekend if day == "weekend" else weekday)*length/source_length, "Native segment time not from bound source edge")
                    _require(part["kind"] in {"public_native", "terminal_native"}, "Unknown path segment type")
                    if part["kind"] == "public_native":
                        _require(public, "Private/terminal road opened to public through-routing")

    def instance_group(self):
        _require(set(self.manifest["instances"]) == set(self.view_table.view_id), "Missing instance status documents")
        for row in self.view_table.itertuples():
            self.stage("instance:"+str(row.view_id), lambda row=row: self._instance(row))
        for d in self.view_table.day_index.unique():
            current = [(info, arrays) for info, arrays in self.instances.values()
                       if info["day_index"] == d and arrays is not None]
            if not current:
                continue
            parent, parent_arrays = max(current, key=lambda pair: len(pair[0]["node_order"]))
            lookup = {loc: i for i, loc in enumerate(parent["node_order"])}
            for info, arrays in current:
                common = [loc for loc in info["node_order"] if loc in lookup]
                own = {loc: i for i, loc in enumerate(info["node_order"])}
                ix, jx = [lookup[loc] for loc in common], [own[loc] for loc in common]
                for field in ["distance_m", "time_s", "energy_kwh", "distance_shortest_path_distance", "distance_shortest_path_time",
                              "time_shortest_path_distance", "time_shortest_path_time"]:
                    _require(np.array_equal(arrays[field][np.ix_(jx, jx)], parent_arrays[field][np.ix_(ix, ix)]), "Nested view costs changed for identical physical terminals")
        self.metrics["materialized_instances"] = sum(arrays is not None for _, arrays in self.instances.values())
        self.metrics["unreachable_instances"] = sum(info["status"] == "numeric_unreachable" for info, _ in self.instances.values())


def verify_run(path, check_sources=False):
    """Verify artifacts independently and write ``verification.json``.

    Default checks source manifests/freeze/community bytes and donor outputs.
    ``check_sources=True`` additionally hashes every frozen CLE file and both
    original Amazon source files. A missing/changed required source is a failed
    verification, never silently substituted. No demand or route is regenerated.
    """
    manifest_path = Path(path) / "manifest.json"
    schema = None
    if manifest_path.is_file():
        try:
            header = _json(manifest_path)
            schema = header.get("schema") if isinstance(header, dict) else None
        except (OSError, json.JSONDecodeError):
            pass  # The existing inventory stage records the malformed input.
    if schema == "evrptw_stage2_territory_run_v1":
        from .verify_territory import verify_territory_run
        return verify_territory_run(path, check_sources=check_sources)
    verifier = _RunVerifier(path, check_sources)
    for name, method in [("output_inventory", verifier.inventory), ("source_bindings", verifier.sources),
        ("fixed_world", verifier.world), ("parcel_day_conservation_and_TW", verifier.demand),
        ("unchanged_joint_donor_marks", verifier.donor_tuples), ("immutable_workload_views", verifier.views),
        ("physical_supply_and_power", verifier.physical), ("saved_instances", verifier.instance_group)]:
        verifier.stage(name, method)
    result = {"schema": "evrptw_stage2_verification_v1", "passed": not verifier.errors,
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "verifier_source_sha256": _sha(__file__),
        "manifest_sha256": _sha(verifier.root / "manifest.json") if (verifier.root / "manifest.json").is_file() else None,
        "run_id": (verifier.manifest or {}).get("run_id"), "checks": verifier.checks,
        "errors": verifier.errors, "warnings": verifier.warnings, "metrics": verifier.metrics,
        "scope": {"saved_structure": True, "parcel_process_and_aggregation": True,
            "physical_numeric_and_sampled_path_provenance": True,
            "all_pair_shortest_path_optimality": False, "route_solution_feasibility": False},
        "check_sources": bool(check_sources), "route_solution_claimed": False,
        "unsupported_workloads_repaired": False, "stage1_modified": False}
    (verifier.root / "verification.json").write_text(json.dumps(result, ensure_ascii=False,
        indent=2, sort_keys=True, allow_nan=False)+"\n")
    return result
