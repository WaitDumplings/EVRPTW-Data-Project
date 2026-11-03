"""Configuration for independent territory-day instances."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import math
from pathlib import Path

from .config import DEFAULT as LEGACY_DEFAULT, _merge


SCHEMA = "evrptw_territory_first_config_v1"
DEFAULT = {
    "schema": SCHEMA,
    "input": deepcopy(LEGACY_DEFAULT["input"]),
    "world": {"seed": 20261029, "provider_id": "synthetic_provider_v1", "depot_id": None,
              "external_radius_km": 50.296004439666696, "p_unit": .01,
              "mean_parcels_per_active_unit": 1.4, "depot_sampling": "uniform_id",
              "service_policy": "geodesic", "service_limit_s": 5445.},
    "activation": {"tau": .25, "opening_penalty": .2, "transfer_penalty": .1,
                   "max_centers": 2, "parameter_status": "illustrative_not_amazon_calibrated",
                   "method": "coupled_bg", "candidate_multiplier": 20.,
                   "opening_probability": .1},
    "workload": {**deepcopy(LEGACY_DEFAULT["workload"]),
                 "service_active_unit_share": .5, "service_active_unit_exponent": .5},
    "operations": deepcopy(LEGACY_DEFAULT["operations"]),
    "instances": {"sizes": [100], "materialize_day_indices": [0]},
    "physical": deepcopy(LEGACY_DEFAULT["physical"]),
    "output": deepcopy(LEGACY_DEFAULT["output"]),
}


def resolve_config(config, *, base_dir=None):
    result = _merge(DEFAULT, config)
    if result["schema"] != SCHEMA:
        raise ValueError("Unsupported territory-first schema")
    # Reuse the existing source, vehicle, charging, and calendar contracts.
    legacy = deepcopy(LEGACY_DEFAULT)
    for key in ["input", "operations", "physical", "output"]:
        legacy[key] = result[key]
    legacy["world"].update({k: v for k, v in result["world"].items() if k in legacy["world"]})
    legacy["world"]["radius_km"] = result["world"]["external_radius_km"]
    legacy["views"]["sizes"] = result["instances"]["sizes"]
    legacy["views"]["materialize_day_indices"] = result["instances"]["materialize_day_indices"]
    legacy["workload"] = {k: v for k, v in result["workload"].items()
                          if k in LEGACY_DEFAULT["workload"]}
    from .config import resolve_config as resolve_legacy
    checked = resolve_legacy(legacy, base_dir=base_dir)
    for key in ["input", "operations", "physical", "output"]:
        result[key] = checked[key]
    result["instances"]["sizes"] = checked["views"]["sizes"]
    world, activation, ops = (result[k] for k in ["world", "activation", "operations"])
    if not 0 < world["p_unit"] < 1:
        raise ValueError("Territory-first p_unit must be strictly between zero and one")
    if world["depot_id"] is not None and not isinstance(world["depot_id"], str):
        raise ValueError("depot_id must be a canonical ID or null")
    if world["depot_sampling"] not in {"uniform_id", "bg_uniform"}:
        raise ValueError("Choose world.depot_sampling uniform_id or bg_uniform")
    if world["service_policy"] not in {"geodesic", "fastest_time_energy"}:
        raise ValueError("Choose world.service_policy geodesic or fastest_time_energy")
    if isinstance(world["service_limit_s"], bool) or not isinstance(world["service_limit_s"], (int, float)) or not 0 < world["service_limit_s"] < math.inf:
        raise ValueError("world.service_limit_s must be positive and finite")
    if activation["method"] not in {"coupled_bg", "sequential_tract"}:
        raise ValueError("Choose activation.method coupled_bg or sequential_tract")
    rho, opening = activation["candidate_multiplier"], activation["opening_probability"]
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not 1 <= rho < math.inf:
        raise ValueError("activation.candidate_multiplier must be finite and at least one")
    if isinstance(opening, bool) or not isinstance(opening, (int, float)) or not 0 <= opening <= 1:
        raise ValueError("activation.opening_probability must be in [0,1]")
    for key in ["tau", "opening_penalty", "transfer_penalty"]:
        value = activation[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("activation." + key + " must be finite")
        if value < 0 or (key == "tau" and value == 0):
            raise ValueError("tau must be positive and penalties nonnegative")
    if type(activation["max_centers"]) is not int or not 1 <= activation["max_centers"] <= 2:
        raise ValueError("This reviewed profile supports max_centers of 1 or 2")
    if not isinstance(activation["parameter_status"], str) or not activation["parameter_status"]:
        raise ValueError("Record the activation parameter status")
    share = result["workload"]["service_active_unit_share"]
    exponent = result["workload"]["service_active_unit_exponent"]
    if not 0 <= share <= 1 or not 0 < exponent <= 1:
        raise ValueError("Service unit share must be in [0,1] and exponent in (0,1]")
    if (ops["consolidation_mode"] != "daily" or ops["promise_lead_days"] != 0 or
            ops["deferral_eligibility_probability"] != 0):
        raise ValueError("Territory-day instances use same-day, one-visit orders")
    first = date.fromisoformat(ops["start_date"])
    from datetime import timedelta
    weekdays = [(first + timedelta(days=d)).weekday() for d in range(ops["days"])]
    if any(day not in ops["operating_weekdays"] for day in weekdays):
        raise ValueError("Every requested territory day must be an operating day")
    if world["service_policy"] == "fastest_time_energy" and len({day >= 5 for day in weekdays}) > 1:
        raise ValueError("Fastest-time territory runs must use one day type; split weekday/weekend")
    return result


def load_config(path):
    import json
    path = Path(path)
    return resolve_config(json.loads(path.read_text()), base_dir=path.parent)
