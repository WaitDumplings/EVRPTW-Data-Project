"""Portable read-only access to a saved Stage 2 run."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
import pandas as pd

from .util import sha256_file


class ScenarioRun:
    """Read saved artifacts without rerunning demand, selection, or routing."""

    def __init__(self, path, *, verify_hashes=True):
        self.root = Path(path).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Scenario run must be a directory")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema") not in {
            "evrptw_stage2_run_v1", "evrptw_stage2_territory_run_v1"
        }:
            raise ValueError("Unsupported scenario run schema")
        files = self.manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("Run manifest must list its hashed files")
        for relative, digest in files.items():
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid file SHA-256: " + str(relative))
            target = self._safe(relative)
            if verify_hashes and sha256_file(target) != digest:
                raise ValueError("Run artifact hash mismatch: " + relative)
        tables = self.manifest.get("tables", {})
        if not isinstance(tables, dict) or set(tables) != {"parcels", "operational_days", "views"}:
            raise ValueError("Run must declare parcels, operational_days, and views tables")
        instances = self.manifest.get("instances", {})
        if not isinstance(instances, dict) or not all(isinstance(x, str) for x in [*tables.values(), *instances.values()]):
            raise ValueError("Table and instance paths must be relative path strings")
        for relative in [*tables.values(), *instances.values()]:
            self._file(relative)

    def _safe(self, relative):
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("Artifact paths must be relative POSIX paths")
        rel = PurePosixPath(relative)
        if not rel.parts or rel.is_absolute() or ".." in rel.parts or ":" in rel.parts[0]:
            raise ValueError("Artifact path escapes run directory: " + relative)
        target = (self.root / relative).resolve(strict=True)
        if not target.is_relative_to(self.root) or not target.is_file():
            raise ValueError("Artifact path escapes run directory or is not a file: " + relative)
        return target

    def _file(self, relative):
        if relative not in self.manifest["files"]:
            raise ValueError("Artifact is absent from run hash manifest: " + str(relative))
        return self._safe(relative)

    def parcels(self):
        return pd.read_parquet(self._file(self.manifest["tables"]["parcels"]))

    def operational_days(self):
        return pd.read_parquet(self._file(self.manifest["tables"]["operational_days"]))

    def views(self):
        return pd.read_parquet(self._file(self.manifest["tables"]["views"]))

    def instance(self, view_id):
        relative = self.manifest.get("instances", {}).get(str(view_id))
        if relative is None:
            raise KeyError("Unknown view ID: " + str(view_id))
        metadata = json.loads(self._file(relative).read_text())
        if metadata.get("schema") != "evrptw_scenario_instance_v1" or str(metadata.get("view_id")) != str(view_id):
            raise ValueError("Instance schema or view identity mismatch")
        folder = str(PurePosixPath(relative).parent)
        terminals = pd.read_parquet(self._file(folder + "/terminals.parquet"))
        if "node_index" not in terminals or "location_id" not in terminals:
            raise ValueError("Instance terminals require canonical node indices and location IDs")
        if not np.array_equal(terminals.node_index.to_numpy(), np.arange(len(terminals))):
            raise ValueError("Instance terminals are not saved in canonical node order")
        if terminals.location_id.astype(str).tolist() != list(map(str, metadata.get("node_order", []))):
            raise ValueError("Instance node order differs from its metadata")
        costs = {}
        cost_relative = folder + "/costs.npz"
        if metadata.get("matrix_materialized"):
            with np.load(self._file(cost_relative), allow_pickle=False) as archive:
                costs = {name: archive[name] for name in archive.files}
            for name in ["distance_m", "time_s", "energy_kwh"]:
                if name not in costs or costs[name].shape != (len(terminals), len(terminals)):
                    raise ValueError("Invalid canonical cost matrix: " + name)
        return {"metadata": metadata, "terminals": terminals, "costs": costs}

    def audited_path(self, view_id, source_index, destination_index, *, policy=None):
        """Read a saved path certificate; arbitrary native-path replay uses routing."""
        relative = self.manifest.get("instances", {}).get(str(view_id))
        if relative is None:
            raise KeyError("Unknown view ID: " + str(view_id))
        folder = str(PurePosixPath(relative).parent)
        if policy not in {None, "shortest_distance", "fastest_time"}:
            raise ValueError("policy must be shortest_distance or fastest_time")
        meta = json.loads(self._file(relative).read_text())
        declared = meta.get("cost_policy", {})
        execution = declared.get("execution_policy", "fastest_time")
        if policy is not None and policy != execution and policy not in declared.get("paired_path_policies", []):
            raise ValueError("Requested audit policy is not retained in this legacy artifact")
        filename = "fastest_path_audit.json" if policy == "fastest_time" and execution != "fastest_time" else "path_audit.json"
        audit = json.loads(self._file(folder + "/" + filename).read_text())
        key = f"{int(source_index)},{int(destination_index)}"
        if key not in audit:
            raise KeyError("Pair has no saved audit certificate; replay it with the frozen routing source")
        return audit[key]

    def path(self, view_id, source_index, destination_index, *, policy=None):
        """Replay any pair from the bound frozen source, preserving D/T/E family.

        External frozen CLE bytes must remain available at the recorded input
        path. Run tables alone support matrix reading, but cannot invent native
        road geometry or a path if that source has been moved or changed.
        """
        from .cle_binding import cle_authority
        from .routing import FrozenCostRouter, canonical_endpoint_frame
        from .world import _verified_cle

        item = self.instance(view_id)
        meta = item["metadata"]
        for index in [source_index, destination_index]:
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(item["terminals"]):
                raise ValueError("Path indices must be canonical integer node indices")
        source = self.manifest.get("input", {}).get("cle")
        if not source:
            raise ValueError("Frozen CLE source path is not recorded")
        source = Path(source).expanduser()
        cle = _verified_cle(source if source.is_absolute() else self.root / source)
        authority, certificate = cle_authority(cle.path, cle.manifest)
        hashes = self.manifest.get("source_sha256", {})
        sources = [("cle_manifest", cle.path / "cle_manifest.json"), ("freeze", authority)]
        if "cle_verification" in hashes:
            sources.append(("cle_verification", certificate))
        for name, path in sources:
            if not hashes.get(name) or sha256_file(path) != hashes[name]:
                raise ValueError("Frozen routing source changed: " + name)
        declared = meta.get("cost_policy", {})
        chosen_policy = policy or declared.get("execution_policy") or (
            "shortest_distance" if str(declared.get("path_policy", "")).startswith("shortest_distance_") else "fastest_time")
        if chosen_policy not in {"shortest_distance", "fastest_time"}:
            raise ValueError("policy must be shortest_distance or fastest_time")
        source_id = str(item["terminals"].location_id.iloc[source_index])
        destination_id = str(item["terminals"].location_id.iloc[destination_index])
        if source_index == destination_index:
            return {"source": source_id, "destination": destination_id, "reachable": True,
                    "path_mode": 2, "path_policy": chosen_policy, "distance_m": 0., "time_s": 0., "energy_kwh": 0., "segments": []}
        policy = meta.get("cost_policy")
        if not isinstance(policy, dict):
            raise ValueError("Instance must declare its lazy or materialized routing policy")
        day_type = policy.get("day_type")
        if day_type is None:
            day_type = str(policy.get("path_policy", "")).removeprefix("fastest_").removeprefix("shortest_distance_")
        router = FrozenCostRouter(pd.read_parquet(cle.path / "speeds/directed_speeds.parquet"),
            day_type=day_type, connector_speed_kph=policy.get("connector_speed_kph"))
        if policy.get("source_cost_sha256") and router.source_cost_sha256 != policy["source_cost_sha256"]:
            raise ValueError("Native routing source cost digest changed")
        if policy.get("connector_speed_kph") is None:
            declared = policy.get("connector_policy")
            computed = router.connector_policy
            algorithm = declared == "public_U_length_weighted_median_else_all_public" and computed in {
                "directed_source_length_weighted_public_U_speed_median", "all_public_speed_median_no_U"}
            if declared != computed and not algorithm:
                raise ValueError("Lazy connector rule differs from the declared frozen-source rule")
        # Only the pair is bound. A large natural day never allocates an N x N closure.
        frame, ledger = canonical_endpoint_frame(cle.path, [source_id, destination_id])
        endpoints = router.bind_endpoints(frame, ledger)
        rate = policy.get("specific_energy_kwh_per_km", meta.get("vehicle", {}).get("specific_energy_kwh_per_km"))
        result = router.path(endpoints, 0, 1, specific_energy_kwh_per_km=rate, policy=chosen_policy)
        if item["costs"]:
            if chosen_policy == "fastest_time" and "time_shortest_path_distance" in item["costs"]:
                matrix_names = [("distance_m", "time_shortest_path_distance"),
                                ("time_s", "time_shortest_path_time"),
                                ("energy_kwh", "time_shortest_path_energy_kwh")]
            elif chosen_policy == "shortest_distance" and "distance_shortest_path_distance" in item["costs"]:
                matrix_names = [("distance_m", "distance_shortest_path_distance"),
                                ("time_s", "distance_shortest_path_time"), ("energy_kwh", "energy_kwh")]
            else:
                execution = policy.get("execution_policy", "fastest_time")
                if chosen_policy != execution:
                    raise ValueError("Requested policy is not retained in this legacy matrix artifact")
                matrix_names = [("distance_m", "distance_m"), ("time_s", "time_s"), ("energy_kwh", "energy_kwh")]
            for key, matrix in matrix_names:
                if not np.isclose(result[key], item["costs"][matrix][source_index, destination_index], rtol=1e-9, atol=1e-7):
                    raise ValueError("Native replay differs from saved canonical matrix: " + key)
        return result
