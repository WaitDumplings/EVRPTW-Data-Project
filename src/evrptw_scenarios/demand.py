"""Unconditioned route-ready parcel arrivals, commitments, and daily workload.

Calendar seconds below are local planning-clock coordinates (day*86400+clock),
not UTC elapsed seconds. No vehicle or solver can alter this event population.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

from .calibration import DEFAULT_CALIBRATION

from .marks import instantiate_availability, reconcile_one_visit
from .util import identity, rng, write_json


MARK_FIELDS = ["source_parcel_ref", "source_event_ref", "source_donor_id",
               "source_station_day_id", "source_route_ref", "source_date",
               "depth_cm", "height_cm", "width_cm", "volume_cm3",
               "source_planned_service_s", "source_tw_start_offset_s",
               "source_tw_end_offset_s", "source_tw_status",
               "assigned_planned_service_s", "service_count_model_applied",
               "location_tw_status", "location_tw_start_offset_s", "location_tw_end_offset_s",
               "location_tw_source_donor_id"]
PARCEL_FIELDS = ["parcel_id", "location_id", "community_id", "arrival_day_index",
                 "arrival_date", "ready_s", "promise_s", "promise_day_index",
                 "eligible_for_deferral", "preferred_weekday", "scheduled_day_index",
                 "dispatch_day_index", "commitment_failure", "status",
                 "exact_source_count", "source_package_count", "source_count_extrapolated",
                 "active_units", "modeled_residential_units", "capacity_conditioning_attempts",
                 *MARK_FIELDS]
SITE_FIELDS = ["day_index", "date", "location_id", "community_id", "parcel_count",
               "demand_cm3", "planned_service_time_s", "tw_start_s", "tw_end_s",
               "source_specified_window_count", "source_missing_window_count",
               "supported_one_visit_representation", "generation_supported", "capacity_conditioning_attempts",
               "location_tw_status", "location_tw_source_donor_id", "reasons"]


def schedule_day(arrival_day, *, first_date, promise_day, weekdays,
                 mode, eligible, preferred_weekday):
    """Choose once without consulting marks, location costs, or fleet capacity."""
    legal = [d for d in range(arrival_day, promise_day + 1)
             if (first_date + timedelta(days=d)).weekday() in weekdays]
    if not legal:
        return None
    if mode != "daily" and eligible:
        preferred = [d for d in legal if (first_date + timedelta(days=d)).weekday() == preferred_weekday]
        if preferred:
            return preferred[0]
    return legal[0]


def active_location_count(units, probability, mean_parcels, generator, *, conditioned_active=False):
    """Draw active units and positive geometric parcels, with no count cutoff."""
    units = int(units)
    if units <= 0 or not 0 <= probability <= 1 or mean_parcels < 1:
        raise ValueError("Invalid modeled unit/count parameters")
    if conditioned_active:
        if probability == 0:
            raise ValueError("Cannot condition p=0 on an active location")
        zero_mass = float(np.exp(units * np.log1p(-probability))) if probability < 1 else 0.
        quantile = min(np.nextafter(1., 0.), zero_mass + generator.random() * (1 - zero_mass))
        active = int(binom.ppf(quantile, units, probability))
        active = max(1, active)
    else:
        active = int(generator.binomial(units, probability))
    count = active + int(generator.negative_binomial(active, 1 / mean_parcels)) if active else 0
    return active, count


def build_demand(world_info, config, donors, output_dir, *, progress=None):
    """Persist complete parcels and day ledgers, without demand-feasibility repair."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_parquet(world_info["catalog_path"]).sort_values("location_id")
    op, w = config["operations"], config["world"]
    calibrated = w.get("demand_model", "legacy_gamma_poisson") == "binomial_geometric"
    settings = {**DEFAULT_CALIBRATION, **config.get("workload", {})}
    if calibrated:
        donors.configure_calibration(settings)
    start = date.fromisoformat(op["start_date"])
    field_id = world_info["arrival_field_id"]
    workload_id = identity("daily-workload-field-v2", field_id, op, donors.manifest["source_sha256"],
        donors.manifest["partition"], donors.manifest.get("outputs_sha256", {}),
        getattr(donors, "pool_identity", None), settings if calibrated else {})
    process_id = identity("demand-process-v2", workload_id, world_info["world_id"])
    rows, day_rows, failures, failed_sites = [], [], [], []
    communities = catalog.community_id.fillna("").astype(str).to_numpy()
    communities = np.where(communities == "", "unassigned:" + catalog.location_id.astype(str), communities)
    modeled = catalog.get("modeled_residential_units", catalog.G).to_numpy(dtype=float)
    population = list(zip(catalog.location_id.astype(str), modeled, communities))
    for day in range(op["days"]):
        current = start + timedelta(days=day)
        if calibrated:
            cv = w["community_rate_cv"]
            day_factor = {cid: (float(rng("community-day", field_id, cid, current.isoformat()).gamma(1 / cv**2, cv**2))
                               if cv > 0 else 1.0) for cid in sorted(set(communities))}
        else:
            day_factor = {cid: float(rng("community-day", field_id, cid, current.isoformat()).gamma(w["kappa"], 1 / w["kappa"]))
                          for cid in sorted(set(communities))}
        arrivals_today = 0
        for loc, propensity, community in population:
            event_rng = rng("arrivals", field_id, loc, current.isoformat())
            probability = min(1., w["p_unit"] * day_factor[community]) if calibrated else None
            if calibrated:
                active, count = active_location_count(propensity, probability, w["mean_parcels_per_active_unit"], event_rng)
            else:
                active, count = 0, int(event_rng.poisson(w["rho"] * propensity * day_factor[community]))
            if not count:
                continue
            marks_rng = rng("arrival-marks", workload_id, loc, current.isoformat())
            attempts = 0
            while True:
                attempts += 1
                bundle = donors.sample_bundle(count, marks_rng)
                volume = sum(mark["volume_cm3"] for mark in bundle["parcels"])
                if not calibrated or 0 < volume <= settings["cargo_capacity_cm3"]:
                    break
                exhausted = attempts >= settings["maximum_full_proposals"]
                failures.append({"day_index": day, "date": current.isoformat(), "location_id": loc,
                    "community_id": str(community), "modeled_residential_units": int(propensity),
                    "proposal_index": attempts, "proposed_active_units": active, "proposed_package_count": count,
                    "proposed_volume_cm3": volume, "source_donor_id": bundle["source_donor_id"],
                    "source_parcel_refs_json": json.dumps([m["source_parcel_ref"] for m in bundle["parcels"]]),
                    "reason": "capacity_proposal_budget_exhausted" if exhausted else "capacity_proposal_rejected",
                    "exhausted": exhausted})
                if exhausted:
                    bundle = None
                    failed_sites.append({"day_index": day, "date": current.isoformat(), "location_id": loc,
                        "community_id": str(community), "parcel_count": 0, "demand_cm3": None,
                        "planned_service_time_s": None, "tw_start_s": day*86400+op["shift_start_s"],
                        "tw_end_s": day*86400+op["shift_end_s"], "source_specified_window_count": 0,
                        "source_missing_window_count": 0, "supported_one_visit_representation": False,
                        "generation_supported": False, "capacity_conditioning_attempts": attempts,
                        "location_tw_status": "unsupported_generation", "location_tw_source_donor_id": None,
                        "reasons": '["capacity_proposal_budget_exhausted"]'})
                    break
                active, count = active_location_count(propensity, probability, w["mean_parcels_per_active_unit"], event_rng,
                                                       conditioned_active=True)
            if bundle is None:
                continue
            arrivals_today += count
            preferred_key = community if op["consolidation_mode"] == "community_day_experimental" else loc
            preferred = int(rng("preferred-weekday", field_id, preferred_key).integers(7))
            for ordinal, mark in enumerate(bundle["parcels"]):
                pid = identity("parcel-v2", workload_id, loc, current.isoformat(), ordinal)
                business_rng = rng("parcel-business", field_id, pid)
                eligible = bool(business_rng.random() < op["deferral_eligibility_probability"])
                ready = day * 86400 + float(business_rng.uniform(*op["ready_window_s"]))
                due_day = day + op["promise_lead_days"]
                dispatch = schedule_day(day, first_date=start, promise_day=due_day,
                                        weekdays=op["operating_weekdays"],
                                        mode=op["consolidation_mode"], eligible=eligible,
                                        preferred_weekday=preferred)
                is_in_horizon = dispatch is not None and dispatch < op["days"]
                row = {key: mark.get(key) for key in MARK_FIELDS}
                row.update({"parcel_id": pid, "location_id": loc, "community_id": str(community),
                    "arrival_day_index": day, "arrival_date": current.isoformat(),
                    "ready_s": ready, "promise_s": due_day * 86400 + op["shift_end_s"],
                    "promise_day_index": due_day, "eligible_for_deferral": eligible,
                    "preferred_weekday": preferred, "scheduled_day_index": dispatch,
                    "dispatch_day_index": dispatch if is_in_horizon else None,
                    "commitment_failure": dispatch is None,
                    "status": "dispatched_to_workload" if is_in_horizon else "backlog",
                    "source_donor_id": bundle["source_donor_id"],
                    "exact_source_count": bundle["exact_source_count"],
                    "source_package_count": bundle["source_package_count"],
                    "source_count_extrapolated": bundle["count_outside_observed_range"],
                    "active_units": active if calibrated else None,
                    "modeled_residential_units": int(propensity) if calibrated else None,
                    "capacity_conditioning_attempts": attempts})
                rows.append(row)
        day_rows.append({"day_index": day, "date": current.isoformat(), "arrivals": arrivals_today})
        if progress is not None:
            progress(f"arrival day {day}: {arrivals_today} parcels", flush=True)
    parcels = pd.DataFrame(rows, columns=PARCEL_FIELDS)
    if not calibrated:
        parcels = parcels.drop(columns=["assigned_planned_service_s", "service_count_model_applied",
            "location_tw_status", "location_tw_start_offset_s", "location_tw_end_offset_s",
            "location_tw_source_donor_id"])
    sites = list(failed_sites)
    if not parcels.empty:
        for (day, loc), group in parcels.dropna(subset=["dispatch_day_index"]).groupby(["dispatch_day_index", "location_id"], sort=True):
            day = int(day)
            midnight = day * 86400
            if calibrated:
                window = donors.sample_location_window(rng("location-window", workload_id, loc,
                    (start + timedelta(days=day)).isoformat()))
                for name, value in window.items():
                    parcels.loc[group.index, name] = value
                    group[name] = value
            else:
                # Null optional columns must not shadow the legacy source marks.
                group = group.drop(columns=["assigned_planned_service_s", "location_tw_status",
                    "location_tw_start_offset_s", "location_tw_end_offset_s"], errors="ignore")
            availability = instantiate_availability({"parcels": group.to_dict("records")},
                                                    service_day_start_s=midnight,
                                                    mode=op["availability_pattern"])
            aggregate = reconcile_one_visit(availability, releases_s=group.ready_s.to_list(),
                deadlines_s=group.promise_s.to_list(),
                operating_window_s=[midnight + op["shift_start_s"], midnight + op["shift_end_s"]],
                source_window_end=op["source_window_end"], missing_window_policy=op["missing_window_policy"])
            sites.append({"day_index": day, "date": (start + timedelta(days=day)).isoformat(),
                "location_id": loc, "community_id": str(group.community_id.iloc[0]),
                **{k: aggregate[k] for k in SITE_FIELDS if k in aggregate},
                "generation_supported": True,
                "capacity_conditioning_attempts": int(group.capacity_conditioning_attempts.max()),
                "location_tw_status": window["location_tw_status"] if calibrated else "legacy_source_marks",
                "location_tw_source_donor_id": window["location_tw_source_donor_id"] if calibrated else None,
                "reasons": json.dumps(aggregate["reasons"], separators=(",", ":"))})
    site_frame = pd.DataFrame(sites, columns=SITE_FIELDS)
    previous = 0
    for row in day_rows:
        day = row["day_index"]
        n = int(parcels.dispatch_day_index.eq(day).sum()) if not parcels.empty else 0
        row.update(previous_backlog=previous, dispatched=n, backlog=previous + row["arrivals"] - n,
                   active_locations=int(site_frame.day_index.eq(day).sum()),
                   operational_day_id=identity("operational-day-v1", process_id, day),
                   route_solution_claimed=False)
        previous = row["backlog"]
    failure_columns = ["day_index", "date", "location_id", "community_id", "modeled_residential_units", "proposal_index",
        "proposed_active_units", "proposed_package_count", "proposed_volume_cm3", "source_donor_id",
        "source_parcel_refs_json", "reason", "exhausted"]
    pd.DataFrame(failures, columns=failure_columns).to_parquet(out / "generation_failures.parquet", index=False, compression="zstd")
    parcels.to_parquet(out / "parcels.parquet", index=False, compression="zstd")
    site_frame.to_parquet(out / "location_dispatch.parquet", index=False, compression="zstd")
    pd.DataFrame(day_rows).to_parquet(out / "operational_days.parquet", index=False)
    info = {"schema": "evrptw_demand_process_v1", "process_id": process_id,
            "arrival_field_id": field_id, "workload_field_id": workload_id, "world_id": world_info["world_id"],
            "initial_backlog": 0, "warmup_days": 0, "initialization": "empty finite-horizon scenario",
            "end_backlog": previous, "parcel_count": len(parcels), "location_day_count": len(site_frame),
            "arrival_semantics": "parcel ready at local last-mile depot before route dispatch",
            "arrival_model": ("A~Binomial(modeled_units,min(1,p_unit*Z)); K=A+NegativeBinomial(A,1/mean_parcels); Z=1 reference"
                              if calibrated else "K_i,t ~ Poisson(rho * G_i * Z_community,t); Z ~ Gamma(kappa, rate=kappa)"),
            "capacity_proposal_rejections": len(failures), "generation_unsupported_location_days": len(failed_sites),
            "capacity_conditioning": settings if calibrated else None,
            "donor_calibration": donors.calibration_summary if calibrated else None,
            "failed_proposal_path": str(out / "generation_failures.parquet"),
            "failed_proposal_semantics": "Rejected complete proposals retained separately; exhausted location-days have no capacity-valid parcel workload",
            "rho_is_observed_order_rate": False, "time_axis": "local planning days of 86400 clock seconds; not UTC elapsed time",
            "service_model": ("count-conditioned joint donor residual; assigned service separate from source planned marks"
                              if calibrated else "sum of source per-package planned service; not measured dwell"),
            "availability_semantics": op["availability_pattern"],
            "availability_missing_policy": op["missing_window_policy"],
            "source_window_end": op["source_window_end"],
            "dispatch_semantics": "assigned to complete daily workload; does not assert successful delivery or routing feasibility",
            "donor_source_sha256": donors.manifest["source_sha256"],
            "donor_partition": donors.manifest["partition"],
            "operations": op, "world_demand_parameters": w,
            "parcels_path": str(out / "parcels.parquet"),
            "sites_path": str(out / "location_dispatch.parquet"),
            "days_path": str(out / "operational_days.parquet")}
    write_json(out / "demand.json", info)
    return info
