"""Bind compatible supply, then select an instance budget using road detours."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from evrptw_cle.runtime_paths import CONFIG_ROOT
from evrptw_cle.util import sha256_file

from .config import JOINT_CHARGING_DEFAULTS, JOINT_CHARGING_POLICY, resolve_joint_charging
from .world import _canonical_native, _distance_to_frame, _verified_cle, stable_id


PHYSICAL_DEFAULTS = {
    "vehicle": {
        "profile_id": "delivery700_reference_synthetic_energy_v1",
        "cargo_capacity_cm3": 18_500_000.0,
        "battery_capacity_kwh": 100.0,
        "specific_energy_kwh_per_km": 100.0 / 257.0,
        "initial_soc": 1.0, "reserve_soc": 0.1, "fleet_size": None,
        "dc_cap_kw": 100.0, "ac_cap_kw": 11.0,
    },
    "charging": {
        "derating_factor": 0.9, "supply_policy": "road_detour_budget",
        "external_cs_budget": 20, "charge_policy": "full", "charge_curve": "linear",
        "access_policy": "assume_admitted_source_conditions_satisfied",
        "opening_hours_policy": "assume_available_during_shift", "queue_policy": "unlimited_ports",
    },
}


def _configuration(config):
    if set(config) - set(PHYSICAL_DEFAULTS):
        raise ValueError("Only vehicle and charging physical parameter groups are supported")
    result = {}
    for group, defaults in PHYSICAL_DEFAULTS.items():
        incoming = config.get(group, {})
        if group == "charging":
            defaults = {**defaults, **JOINT_CHARGING_DEFAULTS}
        if set(incoming) - set(defaults):
            raise ValueError("Unknown physical parameter in " + group)
        result[group] = {**defaults, **incoming}
    vehicle, charging = result["vehicle"], result["charging"]
    if vehicle["profile_id"] != PHYSICAL_DEFAULTS["vehicle"]["profile_id"]:
        raise ValueError("Only the declared native J1772/CCS1 reference profile is supported")
    for key in ["cargo_capacity_cm3", "battery_capacity_kwh", "specific_energy_kwh_per_km",
                "dc_cap_kw", "ac_cap_kw"]:
        value = float(vehicle[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Vehicle " + key + " must be positive and finite")
        vehicle[key] = value
    if vehicle["dc_cap_kw"] > 100 or vehicle["ac_cap_kw"] > 11:
        raise ValueError("Native reference maxima are DC 100 kW and AC 11 kW; only lower scenario limits are supported")
    for key in ["initial_soc", "reserve_soc"]:
        vehicle[key] = float(vehicle[key])
    if not 0 <= vehicle["reserve_soc"] < vehicle["initial_soc"] <= 1:
        raise ValueError("Vehicle SOC must satisfy 0 <= reserve < initial <= 1")
    if vehicle["fleet_size"] is not None and (
        type(vehicle["fleet_size"]) is not int or vehicle["fleet_size"] <= 0
    ):
        raise ValueError("fleet_size must be a positive integer or null (not specified)")
    charging["derating_factor"] = float(charging["derating_factor"])
    if not 0 < charging["derating_factor"] <= 1:
        raise ValueError("Charging derating_factor must be in (0, 1]")
    if charging["supply_policy"] not in {"road_detour_budget", "all_compatible_catchment", JOINT_CHARGING_POLICY}:
        raise ValueError("Unsupported charging supply_policy")
    if type(charging["external_cs_budget"]) is not int or charging["external_cs_budget"] < 0:
        raise ValueError("external_cs_budget must be a nonnegative integer")
    resolve_joint_charging(charging, vehicle)
    for key in ["charge_policy", "charge_curve", "access_policy", "opening_hours_policy", "queue_policy"]:
        if charging[key] != PHYSICAL_DEFAULTS["charging"][key]:
            raise ValueError("Unsupported charging " + key + "; do not silently reinterpret source conditions")
    return result


def select_chargers(chargers, customer_ids, costs, *, budget=20, depot_id):
    """Select a fixed number of external sites by directed road insertion cost.

    ``costs`` contains all candidate stations, the depot and the requested
    customers. Only the shortest-distance path's distance matrix is consumed;
    coordinates, speeds, vehicle parameters and a fastest-path distance are
    never used as a substitute. The reference legs are both directed depot
    spokes and each customer's nearest outgoing customer leg. Starting with
    the site minimizing average leg insertion cost, greedy additions minimize
    the mean best available insertion detour. This spreads useful alternatives
    across the task without assuming a specific solution route.

    This is a supply-selection heuristic, not a charging-route feasibility
    certificate. All source-compatible supply remains in ``chargers.parquet``.
    A shortage is explicit, and does not trigger a geography or seed change.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("Charging budget must be a nonnegative integer")
    if "location_id" not in chargers or chargers.location_id.astype(str).duplicated().any():
        raise ValueError("Charging candidates must have unique canonical location IDs")
    location_ids = list(map(str, costs.location_ids))
    if len(set(location_ids)) != len(location_ids):
        raise ValueError("Road costs must have unique location IDs")
    try:
        distance = np.asarray(costs.distance_shortest_path_distance, dtype=float)
    except AttributeError as exc:
        raise ValueError("Charger selection requires the shortest-distance path distance matrix") from exc
    if distance.shape != (len(location_ids), len(location_ids)):
        raise ValueError("Shortest-distance road matrix shape disagrees with its location IDs")
    if np.isnan(distance).any() or (distance < 0).any():
        raise ValueError("Road distances must be nonnegative or explicitly unreachable (infinity)")
    index = {value: i for i, value in enumerate(location_ids)}
    customers = list(map(str, customer_ids))
    if len(set(customers)) != len(customers) or str(depot_id) in customers:
        raise ValueError("Customer IDs must be distinct and exclude the depot")
    station_ids = chargers.location_id.astype(str).tolist()
    if set(station_ids) & (set(customers) | {str(depot_id)}):
        raise ValueError("External stations must have distinct depot/customer IDs")
    required = [str(depot_id), *customers, *station_ids]
    missing = sorted(set(required) - set(index))
    if missing:
        raise ValueError(f"Road costs omit selection endpoints: {missing[:5]}")
    depot = index[str(depot_id)]
    customer_indices = [index[value] for value in customers]
    legs = {(depot, customer) for customer in customer_indices}
    legs.update((customer, depot) for customer in customer_indices)
    for customer in customer_indices:
        neighbors = [other for other in customer_indices if other != customer and np.isfinite(distance[customer, other])]
        if neighbors:
            nearest = min(neighbors, key=lambda other: (distance[customer, other], location_ids[other]))
            legs.add((customer, nearest))
    if not legs:
        legs.add((depot, depot))
    legs = sorted(legs)
    finite_legs = [(i, j) for i, j in legs if np.isfinite(distance[i, j])]
    source = np.asarray([i for i, _ in finite_legs], dtype=int)
    target = np.asarray([j for _, j in finite_legs], dtype=int)
    candidates = np.asarray([index[value] for value in station_ids], dtype=int)
    # C x L, not C x N x N: the reference-leg rule bounds memory by O(CN).
    detours = (distance[source[None, :], candidates[:, None]] +
               distance[candidates[:, None], target[None, :]] - distance[source, target][None, :])
    # An actual station visit may connect its permitted inbound and outbound
    # terminal gates, while public-only through-routing cannot use that private
    # shortcut. Thus terminal OD costs need not obey a triangle inequality
    # through a visited station. Record that benefit, but rank additional detour
    # as max(0, raw detour), without treating it as invalid road data.
    negative = np.isfinite(detours) & (detours < -1e-5)
    raw_minimum = float(detours[np.isfinite(detours)].min()) if np.isfinite(detours).any() else None
    detours = np.maximum(detours, 0.)
    usable = np.isfinite(detours).any(axis=1) if len(finite_legs) else np.zeros(len(chargers), dtype=bool)
    remaining = set(np.flatnonzero(usable).tolist())
    selected, scores = [], []
    best = np.full(len(finite_legs), np.inf)
    while remaining and len(selected) < budget:
        def key(candidate):
            updated = np.minimum(best, detours[candidate])
            finite = np.isfinite(updated)
            # Cover reachable reference legs first, then minimize aggregate
            # detour. Stable IDs break ties independently of input row order.
            return (-int(finite.sum()), float(updated[finite].sum()), station_ids[candidate])
        choice = min(remaining, key=key)
        updated = np.minimum(best, detours[choice])
        finite = np.isfinite(updated)
        individual = detours[choice][np.isfinite(detours[choice])]
        scores.append({"location_id": station_ids[choice], "selection_rank": len(selected) + 1,
                       "individual_mean_detour_m": float(individual.mean()),
                       "individual_min_detour_m": float(individual.min()),
                       "covered_reference_legs": int(finite.sum()),
                       "newly_covered_reference_legs": int((finite & ~np.isfinite(best)).sum()),
                       "selected_set_mean_best_detour_m": float(updated[finite].mean())})
        selected.append(choice)
        remaining.remove(choice)
        best = updated
    chosen = chargers.iloc[selected].copy().reset_index(drop=True)
    chosen["charging_selection_rank"] = np.arange(1, len(chosen) + 1)
    report = {
        "schema": "evrptw_road_detour_charger_selection_v1",
        "status": "selected" if len(chosen) == budget else "unsupported_charging_budget",
        "supported_budget": len(chosen) == budget,
        "requested_budget": budget, "selected_count": len(chosen),
        "candidate_count": len(chargers), "usable_candidate_count": int(usable.sum()),
        "selected_location_ids": chosen.location_id.astype(str).tolist(),
        "road_distance_source": "distance_shortest_path_distance",
        "reference_leg_policy": "both_directed_depot_spokes_plus_nearest_outgoing_customer",
        "selection_policy": "greedy_minimum_mean_best_directed_insertion_detour",
        "detour_score_formula": "max(0,D(i,s)+D(s,j)-D(i,j)); permitted station visit may bridge private terminal gates",
        "negative_raw_detour_count": int(negative.sum()), "minimum_raw_detour_m": raw_minimum,
        "requested_reference_leg_count": len(legs), "reachable_reference_leg_count": len(finite_legs),
        "reference_leg_count_with_selected_supply": int(np.isfinite(best).sum()),
        "selection_scores": scores,
        "capacity_or_energy_route_feasibility_claimed": False,
        "source_supply_dropped": False,
    }
    return chosen, report


def bind_physical(cle_path, world_info, config, output_dir):
    """Keep all source-compatible supply within the world's fixed depot radius.

    Source access text, ports, power QA and selected configuration are retained.
    Hours and queue policies are explicit synthetic assumptions, not parsed
    claims about current opening conditions. No customer-count truncation or
    vehicle-driven geography repair is performed.
    """
    parameters = _configuration(config)
    cle = _verified_cle(cle_path)
    if sha256_file(cle.path / "cle_manifest.json") != world_info["source"]["cle_manifest_sha256"]:
        raise ValueError("Physical binding and world must use the identical frozen CLE")
    output = Path(output_dir).resolve()
    if any((output / name).exists() for name in ["chargers.parquet", "physical.json"]):
        raise FileExistsError("Physical outputs already exist; use a new output directory")
    depot_path = Path(world_info["depot_path"])
    if sha256_file(depot_path) != world_info["artifacts"]["depot.parquet"]:
        raise ValueError("World depot artifact changed after construction")
    depot = gpd.read_parquet(depot_path)
    if len(depot) != 1 or depot.location_id.iat[0] != world_info["depot_id"]:
        raise ValueError("World must contain exactly its selected canonical depot")
    chargers = _canonical_native(cle, "charging")
    geographic = world_info.get("schema") == "evrptw_territory_operating_world_v1"
    if geographic:
        from .territory_world import published_coordinates
        chargers = published_coordinates(chargers)
    configurations = pd.read_parquet(cle.path / "charging/charging_configurations.parquet")
    selected = configurations.loc[configurations.is_selected].copy()
    if selected.configuration_id.duplicated().any() or selected.charging_location_id.duplicated().any():
        raise ValueError("Frozen selected charging configurations are not unique")
    if not chargers.selected_configuration_id.isin(selected.configuration_id).all():
        raise ValueError("An eligible charger lacks its frozen selected compatible configuration")
    cfg = selected.set_index("configuration_id").loc[chargers.selected_configuration_id].reset_index(drop=True)
    if not np.array_equal(cfg.charging_location_id.astype(str), chargers.charging_location_id.astype(str)):
        raise ValueError("Selected charging configuration is bound to another location")
    if not cfg.connector.isin(["J1772", "J1772COMBO"]).all():
        raise ValueError("Frozen configuration is incompatible with the no-adapter reference vehicle")
    if not np.array_equal(cfg.reference_charge_mode, chargers.reference_charge_mode):
        raise ValueError("Frozen charging mode and selected configuration disagree")
    if not np.allclose(cfg.assigned_power_kw, chargers.assigned_power_kw):
        raise ValueError("Frozen assigned source power and selected configuration disagree")
    source_power = cfg.assigned_power_kw.to_numpy(dtype=float)
    if not (np.isfinite(source_power) & (source_power > 0)).all():
        raise ValueError("Compatible charger requires positive reported or explicitly imputed site power")
    # Never reuse capped_pre_derating_kw as station power. Stage 1 has already
    # selected one attributable configuration, but has not applied delta.
    caps = np.where(cfg.reference_charge_mode.eq("dc_fast"),
                    parameters["vehicle"]["dc_cap_kw"], parameters["vehicle"]["ac_cap_kw"])
    chargers["source_assigned_power_kw"] = source_power
    chargers["source_reported_power_kw"] = cfg.reported_power_kw.to_numpy()
    chargers["source_power_is_reported"] = cfg.power_is_reported.to_numpy(dtype=bool)
    chargers["source_power_method"] = cfg.power_source.to_numpy()
    chargers["selected_connector"] = cfg.connector.to_numpy()
    chargers["selected_unit_source_row"] = cfg.unit_source_row.to_numpy()
    chargers["selected_afdc_id"] = cfg.afdc_id.to_numpy()
    chargers["stage2_vehicle_cap_kw"] = caps
    chargers["stage2_capped_pre_derating_kw"] = np.minimum(source_power, caps)
    chargers["stage2_derating_factor"] = parameters["charging"]["derating_factor"]
    chargers["effective_power_kw"] = np.minimum(source_power, caps) * parameters["charging"]["derating_factor"]
    chargers["derating_application_count"] = 1
    chargers["source_access_days_time"] = cfg.access_days_time.to_numpy()
    chargers["source_access_detail_code"] = cfg.access_detail_code.to_numpy()
    chargers["source_access_text_machine_evaluated"] = False
    chargers["opening_hours_policy"] = parameters["charging"]["opening_hours_policy"]
    chargers["opening_availability_is_synthetic_assumption"] = True
    chargers["queue_policy"] = parameters["charging"]["queue_policy"]
    chargers["source_port_counts_enforced_as_queue_capacity"] = False
    radius_km = world_info["config"]["external_radius_km" if geographic else "radius_km"]
    outward, inward = _distance_to_frame(cle, depot, chargers, np.inf if geographic else radius_km)
    inside = np.isfinite(outward) & np.isfinite(inward)
    if geographic:
        from .territory_world import great_circle_km
        gc = great_circle_km(float(depot.location_lon.iloc[0]), float(depot.location_lat.iloc[0]),
                             chargers.location_lon.to_numpy(float), chargers.location_lat.to_numpy(float))
        chargers["depot_geodesic_distance_km"] = gc
        if world_info["config"].get("service_policy") != "fastest_time_energy":
            inside &= gc <= radius_km
    else:
        inside &= (outward <= radius_km * 1000) & (inward <= radius_km * 1000)
    chargers["depot_outward_distance_m"] = outward
    chargers["depot_inward_distance_m"] = inward
    eligible_city_count = len(chargers)
    chargers = chargers.loc[inside].copy().reset_index(drop=True)
    contract_path = CONFIG_ROOT / "charging_parameter_contract_v1.json"
    contract_hash = sha256_file(contract_path)
    contract = json.loads(contract_path.read_text())
    component = json.loads((cle.path / "charging/charging_component_manifest.json").read_text())
    if cle.manifest["schema"] == "evrptw_building_first_cle_v1":
        frozen_contract_hash = component["inputs"]["parameter_contract"]["sha256"]
    else:
        frozen_contract_hash = cle.manifest["policy_hashes"]["charging_parameter_contract_v1.json"]
    if component["inputs"]["units"]["sha256"] != contract["unit_source"]["sha256"]:
        raise ValueError("Source-citation contract refers to a different charging-unit snapshot")
    # The cleaned repository's reference document may differ from the original
    # build-time contract. It supplies citations only: actual configurations and
    # powers above come exclusively from manifest-hashed frozen CLE artifacts.
    physical_id = stable_id({"schema": "physical_supply_v1", "world_id": world_info["world_id"],
                             "config": parameters, "charging_contract_sha256": contract_hash})
    chargers["world_id"] = world_info["world_id"]
    chargers["physical_id"] = physical_id
    output.mkdir(parents=True, exist_ok=True)
    chargers.to_parquet(output / "chargers.parquet", index=False, compression="zstd")
    result = {
        "schema": "evrptw_world_physical_supply_v1", "physical_id": physical_id,
        "world_id": world_info["world_id"], "arrival_field_id": world_info["arrival_field_id"],
        "depot_id": world_info["depot_id"], "config": parameters,
        "chargers_path": str(output / "chargers.parquet"), "physical_path": str(output / "physical.json"),
        "artifacts": {"chargers.parquet": sha256_file(output / "chargers.parquet")},
        "compatible_city_chargers": eligible_city_count, "charger_count": len(chargers),
        "reported_power_count": int(chargers.source_power_is_reported.sum()),
        "imputed_power_count": int((~chargers.source_power_is_reported).sum()),
        "source": {"cle_manifest_sha256": world_info["source"]["cle_manifest_sha256"],
                   "reference_document_charging_contract_sha256": contract_hash,
                   "frozen_build_charging_contract_sha256": frozen_contract_hash,
                   "reference_document_matches_frozen_contract": contract_hash == frozen_contract_hash,
                   "configuration_source": "frozen manifest-hashed charging_configurations.parquet",
                   "charging_component_manifest_sha256": sha256_file(cle.path / "charging/charging_component_manifest.json"),
                   "charging_units": contract["unit_source"],
                   "reference_vehicle_source": contract["native_reference_vehicle"]["source"]},
        "vehicle_semantics": {
            "cargo_capacity": "retained reference-profile volumetric capacity; not measured payload weight",
            "battery_capacity_kwh": "explicit synthetic energy scenario; not observed manufacturer battery capacity",
            "specific_energy_kwh_per_km": "synthetic constant-energy scenario; default B/257, not measured route consumption",
            "charging_caps": "reference maximum caps; source pointer in frozen Stage1 charging contract",
            "fleet_size_null": "unlimited homogeneous vehicles in the synthetic EVRPTW model; not observed fleet capacity",
            "weight_demand_modeled": False,
        },
        "power_formula": "effective_power_kw = min(source_assigned_power_kw, stage2_vehicle_cap_kw) * derating_factor",
        "derating_semantics": "one transparent scenario multiplier, not an observed full-charge power curve",
        "opening_hours_semantics": "source text retained, not machine parsed; available during shift is an explicit synthetic override",
        "access_semantics": "synthetic scenario assumes admitted public source conditions can be satisfied; source text retained; payment, membership, parking and eligibility conditions are not machine enforced or verified",
        "queue_semantics": "unlimited simultaneous service assumption; original ports retained as source information",
        "charge_policy": parameters["charging"]["charge_policy"],
        "charge_curve": parameters["charging"]["charge_curve"],
        "charging_time_formula": "(battery_capacity_kwh - arrival_energy_kwh) / effective_power_kw * 3600",
        "supply_policy": {
            "road_detour_budget": "all compatible catchment supply retained here; per-instance road-detour budget applied at materialization",
            JOINT_CHARGING_POLICY: "all compatible catchment supply retained here; per-instance tract joint budget applied at materialization",
            "all_compatible_catchment": "all compatible catchment supply included in each instance",
        }[parameters["charging"]["supply_policy"]],
        "stage1_modified": False, "stage2_frozen": False,
    }
    (output / "physical.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result
