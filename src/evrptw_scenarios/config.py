"""Explicit scenario assumptions. Nothing below is an estimated order-rate panel."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
import math
from pathlib import Path

from .calibration import DEFAULT_CALIBRATION

JOINT_CHARGING_POLICY = "tract_joint_budget"
JOINT_CHARGING_DEFAULTS = {"reference_departure_soc": .8, "external_cs_budgets_by_size": None}
JOINT_CHARGING_BUDGETS = {"100": 20, "500": 30, "1000": 40, "2000": 60}

DEFAULT = {
    "schema": "evrptw_demand_first_config_v1",
    "input": {"cle": "", "communities": "", "donor_artifact": ""},
    "world": {"seed": 20261028, "provider_id": "synthetic_provider_v1", "depot_id": None,
              "radius_km": 25, "alpha": 1, "rho": .23, "kappa": 4,
              "unknown_units_shape": 1, "unknown_units_rate": 1,
              "demand_model": "binomial_geometric", "p_unit": .01,
              "mean_parcels_per_active_unit": 1.4, "community_rate_cv": 0},
    "workload": deepcopy(DEFAULT_CALIBRATION),
    "operations": {"start_date": "2026-11-02", "days": 1,
                   "operating_weekdays": [0, 1, 2, 3, 4, 5, 6],
                   "shift_start_s": 28800, "shift_end_s": 86400,
                   "ready_window_s": [0, 21600], "promise_lead_days": 0,
                   "consolidation_mode": "daily", "deferral_eligibility_probability": 0,
                   "availability_pattern": "recurring_local_clock",
                   "source_window_end": "service_start",
                   "missing_window_policy": "operating_horizon"},
    "views": {"sizes": [100, 500, 1000], "sampler": "community_blocks",
              "materialize_day_indices": [0], "natural_day_dense_matrix_limit": 0},
    "physical": {
        "vehicle": {"profile_id": "delivery700_reference_synthetic_energy_v1",
                    "cargo_capacity_cm3": 18500000, "battery_capacity_kwh": 100,
                    "specific_energy_kwh_per_km": 100 / 257,
                    "initial_soc": 1, "reserve_soc": .1, "fleet_size": None,
                    "dc_cap_kw": 100, "ac_cap_kw": 11},
        "charging": {"derating_factor": .9, "supply_policy": "road_detour_budget",
                     "external_cs_budget": 20, "charge_policy": "full", "charge_curve": "linear",
                     "access_policy": "assume_admitted_source_conditions_satisfied",
                     "opening_hours_policy": "assume_available_during_shift",
                     "queue_policy": "unlimited_ports", **JOINT_CHARGING_DEFAULTS}},
    "output": {"path": "stage2-run"},
}


def resolve_joint_charging(charging, vehicle):
    """Resolve the optional joint selector settings without changing legacy profiles."""
    if charging["supply_policy"] != JOINT_CHARGING_POLICY:
        for key, default in JOINT_CHARGING_DEFAULTS.items():
            if charging.pop(key, default) != default:
                raise ValueError(f"physical.charging.{key} requires {JOINT_CHARGING_POLICY}")
        return
    soc = charging["reference_departure_soc"]
    if (isinstance(soc, bool) or not isinstance(soc, (int, float)) or
            not vehicle["reserve_soc"] < soc <= 1):
        raise ValueError("physical.charging.reference_departure_soc must exceed reserve_soc and be at most 1")
    overrides = charging["external_cs_budgets_by_size"]
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict) or any(
        not isinstance(n, str) or not n.isascii() or not n.isdecimal() or
        n[0] not in "123456789" or type(m) is not int or m < 0
        for n, m in overrides.items()
    ):
        raise ValueError("physical.charging.external_cs_budgets_by_size must map positive size strings to nonnegative integers")
    charging["external_cs_budgets_by_size"] = {**JOINT_CHARGING_BUDGETS, **overrides}


def joint_station_budget(charging, customer_count):
    return charging["external_cs_budgets_by_size"].get(str(customer_count), charging["external_cs_budget"])


def _merge(base, supplied, prefix=""):
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'config'} must be an object")
    unknown = set(supplied) - set(base)
    if unknown:
        raise ValueError(f"Unknown configuration keys at {prefix}: {sorted(unknown)}")
    result = deepcopy(base)
    for key, value in supplied.items():
        result[key] = _merge(base[key], value, prefix + key + ".") if isinstance(base[key], dict) else value
    return result


def resolve_config(config, *, base_dir=None):
    if config.get("schema") == "evrptw_territory_first_config_v1":
        from .territory_config import resolve_config as resolve_territory
        return resolve_territory(config, base_dir=base_dir)
    result = _merge(DEFAULT, config)
    if result["schema"] != DEFAULT["schema"]:
        raise ValueError("Unsupported Stage 2 config schema")
    for key in ["cle", "communities", "donor_artifact"]:
        if not result["input"][key]:
            raise ValueError(f"input.{key} is required")
        path = Path(result["input"][key]).expanduser()
        result["input"][key] = str((Path(base_dir or '.') / path).resolve())
    result["output"]["path"] = str((Path(base_dir or '.') / Path(result["output"]["path"]).expanduser()).resolve())
    w, op, v, phys = (result[k] for k in ["world", "operations", "views", "physical"])
    def positive(value, name, zero=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value < 0 if zero else value <= 0):
            raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    def integer(value, name, minimum=0):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    integer(w["seed"], "world.seed")
    if not isinstance(w["provider_id"], str) or not w["provider_id"]:
        raise ValueError("A provider_id is required")
    for key in ["radius_km", "alpha", "kappa", "unknown_units_shape", "unknown_units_rate"]:
        positive(w[key], "world." + key)
    positive(w["rho"], "world.rho", zero=True)
    if w["demand_model"] not in {"binomial_geometric", "legacy_gamma_poisson"}:
        raise ValueError("Unknown world.demand_model")
    if not isinstance(w["p_unit"], (int, float)) or isinstance(w["p_unit"], bool) or not 0 <= w["p_unit"] <= 1:
        raise ValueError("world.p_unit must be in [0,1]")
    positive(w["mean_parcels_per_active_unit"], "world.mean_parcels_per_active_unit")
    if w["mean_parcels_per_active_unit"] < 1:
        raise ValueError("Positive geometric mean must be >=1")
    positive(w["community_rate_cv"], "world.community_rate_cv", zero=True)
    workload = result["workload"]
    if workload["schema"] != DEFAULT_CALIBRATION["schema"] or workload["donor_pool"] != "GEN-TRAIN":
        raise ValueError("Calibrated workload requires its versioned GEN-TRAIN donor pool")
    for key in ["minimum_donor_bundles", "minimum_donor_station_days", "service_support_max_count", "maximum_full_proposals"]:
        integer(workload[key], "workload." + key, 1)
    for key in ["service_b_s", "service_alpha", "service_tail_gamma", "service_scale", "cargo_capacity_cm3"]:
        positive(workload[key], "workload." + key)
    positive(workload["service_a_s"], "workload.service_a_s", zero=True)
    if not 0 <= workload["specified_location_tw_probability"] <= 1:
        raise ValueError("Location TW probability must be in [0,1]")
    date.fromisoformat(op["start_date"])
    integer(op["days"], "operations.days", 1)
    integer(op["promise_lead_days"], "operations.promise_lead_days")
    weekdays = op["operating_weekdays"]
    if not weekdays or len(set(weekdays)) != len(weekdays) or any(type(x) is not int or x not in range(7) for x in weekdays):
        raise ValueError("operating_weekdays must contain distinct integers in 0..6")
    if not 0 <= op["shift_start_s"] < op["shift_end_s"] <= 86400:
        raise ValueError("One local service day must have 0<=shift_start<shift_end<=86400")
    ready = op["ready_window_s"]
    if len(ready) != 2 or not 0 <= ready[0] <= ready[1] <= op["shift_start_s"]:
        raise ValueError("Ready window must finish before dispatch shift")
    if op["consolidation_mode"] not in {"daily", "address_day", "community_day_experimental"}:
        raise ValueError("Unknown consolidation_mode")
    p = op["deferral_eligibility_probability"]
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
        raise ValueError("Deferral eligibility probability must be in [0,1]")
    if op["availability_pattern"] != "recurring_local_clock":
        raise ValueError("This version implements explicitly declared recurring local-clock availability")
    if op["source_window_end"] not in {"service_start", "service_completion"}:
        raise ValueError("Unknown source_window_end")
    if op["missing_window_policy"] not in {"operating_horizon", "unknown_unsupported"}:
        raise ValueError("Unknown missing_window_policy")
    for n in v["sizes"]:
        integer(n, "views.sizes", 1)
    if len(set(v["sizes"])) != len(v["sizes"]):
        raise ValueError("View sizes must be unique")
    v["sizes"].sort()
    if v["sampler"] not in {"uniform_locations", "community_blocks"}:
        raise ValueError("Unknown location sampler")
    for d in v["materialize_day_indices"]:
        integer(d, "materialize day index")
        if d >= op["days"]:
            raise ValueError("Materialized day index lies outside simulation horizon")
    integer(v["natural_day_dense_matrix_limit"], "natural_day_dense_matrix_limit")
    vehicle = phys["vehicle"]
    if vehicle["profile_id"] != DEFAULT["physical"]["vehicle"]["profile_id"]:
        raise ValueError("This version supports the declared native J1772/CCS1 reference profile")
    for key in ["cargo_capacity_cm3", "battery_capacity_kwh", "specific_energy_kwh_per_km", "dc_cap_kw", "ac_cap_kw"]:
        positive(vehicle[key], "physical.vehicle." + key)
    if vehicle["dc_cap_kw"] > 100 or vehicle["ac_cap_kw"] > 11:
        raise ValueError("Reference vehicle caps cannot exceed native DC 100 kW / AC 11 kW maxima")
    if not 0 <= vehicle["reserve_soc"] < vehicle["initial_soc"] <= 1:
        raise ValueError("Require 0<=reserve_soc<initial_soc<=1")
    if vehicle["fleet_size"] is not None:
        integer(vehicle["fleet_size"], "physical.vehicle.fleet_size", 1)
    charging = phys["charging"]
    if not 0 < charging["derating_factor"] <= 1:
        raise ValueError("Charging derating_factor must be in (0,1]")
    integer(charging["external_cs_budget"], "physical.charging.external_cs_budget")
    if charging["supply_policy"] not in {"road_detour_budget", "all_compatible_catchment", JOINT_CHARGING_POLICY}:
        raise ValueError("Unknown charging supply policy")
    resolve_joint_charging(charging, vehicle)
    for key, expected in [("charge_policy", "full"), ("charge_curve", "linear"), ("access_policy", "assume_admitted_source_conditions_satisfied"), ("opening_hours_policy", "assume_available_during_shift"), ("queue_policy", "unlimited_ports")]:
        if charging[key] != expected:
            raise ValueError(f"This version supports charging.{key}={expected!r}")
    return result


def load_config(path):
    path = Path(path).resolve()
    return resolve_config(json.loads(path.read_text()), base_dir=path.parent)
