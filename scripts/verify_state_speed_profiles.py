#!/usr/bin/env python3
"""Reproduce every installed state profile from its compact statistical SQL input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evrptw_cle.moves_speed import (
    DEFAULT_MOVES_SPEED_PROFILE_PATH,
    derive_moves_speed_profile_from_sql,
    load_moves_speed_profile,
    resolve_moves_speed_profile,
)


def verify_profiles(registry_path: Path = DEFAULT_MOVES_SPEED_PROFILE_PATH) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records = []
    for state in registry["states"]:
        path = resolve_moves_speed_profile(registry_path, state_fips=state)
        profile = load_moves_speed_profile(path)
        source = path.parent / profile["reproduction"]["canonical_sql"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != profile["reproduction"]["sha256"]:
            raise ValueError(f"Statistical input hash mismatch: {source}")
        reproduced = derive_moves_speed_profile_from_sql(source)
        for field in ("road_types", "source_type_id", "service_window", "low_flow_benchmark"):
            if reproduced[field] != profile[field]:
                raise ValueError(f"State {state} profile differs from its SQL input: {field}")
        records.append(
            {
                "state_fips": state,
                "profile_id": profile["profile_id"],
                "profile_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "statistical_input_sha256": digest,
            }
        )
    cities = {}
    for city, state in registry["city_state_fips"].items():
        path = resolve_moves_speed_profile(registry_path, city_slug=city, state_fips=state)
        cities[city] = load_moves_speed_profile(path)["profile_id"]
    return {"passed": True, "states": records, "cities": cities}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_MOVES_SPEED_PROFILE_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_profiles(args.registry)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
