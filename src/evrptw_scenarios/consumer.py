"""Plain canonical EVRPTW export, without adapting to any historical solver API."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .reader import ScenarioRun
from .util import identity, write_json


class UnsupportedSolverRepresentation(ValueError):
    """Saved workload remains intact but cannot use this indivisible-stop adapter."""


def export_solver_input(run_path, view_id, output_path, *, path_policy="shortest_distance",
                        objective="total_distance", specific_energy_kwh_per_km=None):
    """Export saved same-path matrices and unsplit stops to one explicit JSON.

    Passing these structural checks does not establish vehicle-route feasibility.
    Oversize stops and unavailable one-visit windows are rejected, never repaired.
    """
    if path_policy not in {"shortest_distance", "fastest_time"}:
        raise ValueError("path_policy must be shortest_distance or fastest_time")
    if objective not in {"total_distance", "total_energy"}:
        raise ValueError("objective must be total_distance or total_energy")
    run = ScenarioRun(run_path)
    item = run.instance(view_id)
    meta, nodes, costs = item["metadata"], item["terminals"], item["costs"]
    reasons = []
    if meta.get("status") != "materialized" or not meta.get("matrix_materialized"):
        reasons.append("instance_not_materialized:" + str(meta.get("status")))
    assessment = meta.get("assessment", {})
    policy = meta.get("cost_policy")
    if not isinstance(policy, dict) or policy.get("distance_is_same_time_path") is not True:
        reasons.append("same_path_cost_family_not_declared")
    if assessment.get("unsupported_one_visit_locations", 0):
        reasons.append("unsupported_one_visit_locations")
    if assessment.get("oversize_bundles", 0):
        reasons.append("oversize_indivisible_location_bundles")
    required = {"node_index", "location_id", "kind", "demand_cm3", "service_time_s", "tw_start_s", "tw_end_s"}
    if not required.issubset(nodes.columns):
        raise UnsupportedSolverRepresentation("Missing canonical terminal columns: " + str(sorted(required - set(nodes.columns))))
    kinds = nodes.kind.astype(str).tolist()
    if not kinds or kinds[0] != "depot" or kinds.count("depot") != 1:
        reasons.append("exactly_one_depot_at_node_zero_required")
    if any(kind not in {"depot", "customer", "charging"} for kind in kinds):
        reasons.append("unknown_terminal_kind")
    if any(kinds[i] == "customer" and "charging" in kinds[:i] for i in range(len(kinds))):
        reasons.append("customers_must_precede_charging")
    numbers = nodes[["demand_cm3", "service_time_s", "tw_start_s", "tw_end_s"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numbers.to_numpy()).all():
        reasons.append("nonfinite_terminal_constraints")
    if (numbers.demand_cm3 < 0).any() or (numbers.service_time_s < 0).any():
        reasons.append("negative_demand_or_service")
    if (numbers.tw_start_s > numbers.tw_end_s).any():
        reasons.append("empty_one_visit_start_window")
    vehicle = dict(meta.get("vehicle", {}))
    capacity = vehicle.get("cargo_capacity_cm3")
    if not isinstance(capacity, (int, float)) or not np.isfinite(capacity) or capacity <= 0:
        reasons.append("positive_volume_capacity_required")
    elif (numbers.loc[nodes.kind.eq("customer"), "demand_cm3"] > capacity).any():
        reasons.append("oversize_indivisible_location_bundles")
    if "charging" in kinds:
        if "power_usable_kw" not in nodes:
            reasons.append("charging_power_missing")
        else:
            power = pd.to_numeric(nodes.loc[nodes.kind.eq("charging"), "power_usable_kw"], errors="coerce").to_numpy()
            if not np.isfinite(power).all() or (power <= 0).any():
                reasons.append("charging_power_invalid")
    for name in ["distance_m", "time_s", "energy_kwh"]:
        matrix = costs.get(name)
        if matrix is None or matrix.shape != (len(nodes), len(nodes)):
            reasons.append("cost_matrix_missing_or_wrong_shape:" + name)
        elif not np.isfinite(matrix).all() or (matrix < 0).any():
            reasons.append("unreachable_or_invalid_costs:" + name)
    paired_names = ("distance_shortest_path_distance", "distance_shortest_path_time",
                    "time_shortest_path_distance", "time_shortest_path_time")
    paired = {name: costs[name] for name in paired_names if name in costs}
    if paired and len(paired) != len(paired_names):
        reasons.append("incomplete_paired_path_family")
    for name, matrix in paired.items():
        if matrix.shape != (len(nodes), len(nodes)) or not np.isfinite(matrix).all() or (matrix < 0).any():
            reasons.append("invalid_paired_cost_matrix:" + name)
    if len(paired) == 4:
        if path_policy == "shortest_distance":
            distance, time = paired[paired_names[0]], paired[paired_names[1]]
            energy = costs.get("energy_kwh")
        else:
            distance, time = paired[paired_names[2]], paired[paired_names[3]]
            energy = costs.get("time_shortest_path_energy_kwh")
        execution_policy = path_policy
    else:
        # Read old same-path exports honestly; do not relabel a legacy fastest
        # path as shortest distance just because the new default changed.
        declared = (policy or {}).get("execution_policy")
        if declared is None and str((policy or {}).get("path_policy", "")).startswith("fastest_"):
            declared = "fastest_time"
        if declared and path_policy != declared:
            reasons.append("requested_path_family_not_saved")
        elif path_policy == "fastest_time" and not declared:
            reasons.append("fastest_path_family_not_declared")
        execution_policy = declared or "legacy_declared_same_path"
        distance, time, energy = (costs.get(x) for x in ["distance_m", "time_s", "energy_kwh"])
    if specific_energy_kwh_per_km is not None:
        rate = float(specific_energy_kwh_per_km)
        if not np.isfinite(rate) or rate <= 0:
            reasons.append("invalid_scenario_energy_rate")
        elif distance is not None:
            energy = distance * rate / 1000
            vehicle["specific_energy_kwh_per_km"] = rate
    if energy is None:
        reasons.append("same_path_energy_matrix_missing")
    elif energy.shape != (len(nodes), len(nodes)) or not np.isfinite(energy).all() or (energy < 0).any():
        reasons.append("selected_same_path_energy_matrix_invalid")
    if reasons:
        raise UnsupportedSolverRepresentation("Cannot export unsplit EVRPTW representation: " + "; ".join(sorted(set(reasons))))
    # Round-trip JSON makes parquet nulls explicit. Source demand is never changed.
    records = json.loads(nodes.to_json(orient="records"))
    result = {"schema": "evrptw_canonical_solver_input_v1",
        "source": {"run_id": run.manifest.get("run_id"), "view_id": str(view_id),
                   "world_id": meta.get("world_id"), "process_id": meta.get("process_id"),
                   "date": meta.get("date"), "kind": meta.get("kind")},
        "depot_index": 0,
        "customer_indices": [i for i, kind in enumerate(kinds) if kind == "customer"],
        "charging_indices": [i for i, kind in enumerate(kinds) if kind == "charging"],
        "node_order": nodes.location_id.astype(str).tolist(), "nodes": records,
        "distance_matrix_km": (distance / 1000).tolist(),
        "travel_time_matrix_s": time.tolist(),
        "energy_matrix_kwh": energy.tolist(),
        "primary_objective": objective,
        "execution_path_policy": execution_policy,
        "physical_scenario_id": identity("physical-export-v2", run.manifest.get("run_id"),
                                         str(view_id), execution_policy, objective, vehicle),
        "paired_path_matrices": {name: matrix.tolist() for name, matrix in paired.items()},
        "paired_path_matrix_units": {paired_names[0]: "m", paired_names[1]: "s",
                                     paired_names[2]: "m", paired_names[3]: "s"},
        "fastest_time_lower_bound_s": paired[paired_names[3]].tolist() if len(paired) == 4 else None,
        "fastest_time_mask_semantics": "optimistic lower bound; never substitutes for selected execution-path time",
        "vehicle": vehicle, "operating_window_s": meta["operating_window_s"],
        "time_axis": "seconds since this service day's local planning midnight",
        "time_window_semantics": "service start; completion bounds already included in derived customer windows",
        "demand_unit": "cm3", "service_unit": "seconds", "charging_power_unit": "kW",
        "cost_policy": {**policy, "execution_policy": execution_policy,
                        "path_policy": (execution_policy + "_" + policy["day_type"]
                                        if policy.get("day_type") else policy.get("path_policy")),
                        "primary_objective": objective,
                        "specific_energy_kwh_per_km": vehicle.get("specific_energy_kwh_per_km")},
        "physical_revalidation_required": (specific_energy_kwh_per_km is not None or
                                            execution_policy != policy.get("execution_policy", execution_policy)),
        "route_solution_claimed": False, "demand_or_windows_repaired": False,
        "compatibility": "canonical Stage 2 schema; no claim of historical EVRPTW_Core API compatibility"}
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, result)
    return target
