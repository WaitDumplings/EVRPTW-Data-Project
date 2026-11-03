"""Frozen-CLE adapters for territory-first activation.

The public road graph remains complete and directed for all distance queries.
Community topology is an undirected growth support with retained road witnesses;
its edges are not distance shortcuts and do not authorize terminal through travel.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from evrptw_cle.util import sha256_file
from .routing import FrozenCostRouter
from .territory import contract_empty_units
from .world import (
    WORLD_DEFAULTS, DistanceEngine, _canonical_native, _communities, _propensities,
    _verified_cle, build_endpoints, pair_distances, public_csr, stable_id,
)


def great_circle_km(lon1, lat1, lon2, lat2):
    """Broadcasting haversine distance, matching the published Amazon analysis."""
    x1, y1, x2, y2 = np.broadcast_arrays(lon1, lat1, lon2, lat2)
    x1, y1, x2, y2 = (np.deg2rad(np.asarray(a, dtype=float)) for a in (x1, y1, x2, y2))
    h = np.sin((y2 - y1) / 2) ** 2 + np.cos(y1) * np.cos(y2) * np.sin((x2 - x1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def published_coordinates(frame):
    """Expose frozen point coordinates without substituting a road anchor.

    Customers already carry the published representative coordinate fields;
    native depot and charger tables instead store their published Point geometry.
    """
    if {"location_lon", "location_lat"}.issubset(frame.columns):
        return frame
    points = frame.geometry.to_crs(4326)
    result = frame.copy()
    result["location_lon"], result["location_lat"] = points.x.to_numpy(), points.y.to_numpy()
    return result


class NativeDistanceProvider:
    """Cache one public CSR and use the exact frozen endpoint-access adapters.

    Single-source calls return two n-vectors; multi-source calls return two
    K-by-n arrays. The mutable sparse-root engine is process-local, not threaded.
    """
    def __init__(self, speeds):
        self.graph, self.node_index, self.edge_lookup = public_csr(speeds)
        self.engine = None
        self._target_frame = None
        self._target_endpoints = None

    def distances(self, source_frame, dest_frame):
        source = build_endpoints(source_frame, self.node_index, self.edge_lookup)
        if self._target_frame is not dest_frame:
            self._target_frame = dest_frame
            self._target_endpoints = build_endpoints(
                dest_frame, self.node_index, self.edge_lookup, retain_ref=False,
            )
        targets = self._target_endpoints
        slots = max(1, source.out_nodes.shape[1], source.in_nodes.shape[1])
        if self.engine is None or self.engine.root_slots < slots:
            self.engine = DistanceEngine(self.graph, limit_m=np.inf, root_slots=slots)
        outward = np.empty((len(source), len(targets)))
        inward = np.empty_like(outward)
        target_ids = {identity: j for j, identity in enumerate(targets.ids)}
        for row in range(len(source)):
            outward[row], inward[row] = pair_distances(row, source, targets, self.engine)
            same = target_ids.get(source.ids[row])
            if same is not None:
                # A visit-to-itself is zero, not two trips along its connector.
                outward[row, same] = inward[row, same] = 0.0
        if len(source) == 1:
            return outward[0], inward[0]
        return outward, inward


@dataclass
class NativeTerritoryWorld:
    info: dict
    catalog: gpd.GeoDataFrame
    depot: gpd.GeoDataFrame
    adjacency: dict[str, tuple[str, ...]]
    distance_provider: NativeDistanceProvider

    def anchor_costs(self, indices):
        anchors = self.catalog.iloc[np.asarray(indices, dtype=np.int64)]
        outward, inward = self.distance_provider.distances(anchors, self.catalog)
        return np.atleast_2d(np.maximum(outward, inward)) / 1000.0


def _line_parts(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        if geometry.length > 0:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _line_parts(part)


def _community_polygons(bg, catalog, crs):
    key = next((k for k in ["bg_geoid", "GEOID", "geoid", "census_block_group_geoid"]
                if k in bg), None)
    if key is None or bg.crs is None:
        raise ValueError("Road-supported communities require georeferenced BG identifiers")
    known = catalog.loc[catalog.community_id.notna(), ["bg_geoid", "cell_id"]]
    labels = dict(zip(known.bg_geoid.astype(str), known.cell_id.astype(str)))
    chosen = bg.loc[bg[key].astype(str).isin(labels), [key, "geometry"]].copy().to_crs(crs)
    cells = np.asarray([labels[str(value)] for value in chosen[key]], dtype=str)
    return chosen, cells


def road_supported_community_graph(roads, speeds, bg, catalog):
    """Insert occupied BGs along physical public roads, then contract empty nodes.

    Road IDs stay separate outside all occupied BGs. A shared polygon boundary
    alone creates no adjacency. Witnesses certify the growth topology; native
    directed shortest-path costs remain the only routing costs.
    """
    if roads.crs is None:
        raise ValueError("Physical roads require an explicit CRS")
    active = speeds.loc[speeds.in_public_graph.astype(bool)]
    public_refs = set(active[["edge_u", "edge_v", "edge_key"]].astype(str)
                      .itertuples(index=False, name=None))
    polygons, polygon_cells = _community_polygons(bg, catalog, roads.crs)
    tree = shapely.STRtree(polygons.geometry.to_numpy())
    adjacency = {str(g): set() for g in catalog.cell_id}
    link_evidence = {}
    road_evidence = {}

    def add_path(path, proof_id):
        for a, b in zip(path, path[1:]):
            if a == b:
                continue
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
            link_evidence.setdefault(tuple(sorted((a, b))), set()).add(proof_id)

    retained = 0
    for row in roads.itertuples(index=False):
        refs = json.loads(row.directed_edge_refs)
        matching = [ref for ref in refs
                    if tuple(str(ref[k]) for k in ("u", "v", "key")) in public_refs]
        if not matching:
            continue
        geometry = row.geometry
        if geometry.geom_type == "MultiLineString":
            geometry = shapely.line_merge(geometry)
        if geometry.is_empty or geometry.geom_type != "LineString":
            raise ValueError("Public physical road is not one ordered line: " + row.physical_edge_id)
        road_id = str(row.physical_edge_id)
        reference = matching[0]
        orientation = reference.get("geometry_orientation")
        if orientation == "same_as_physical":
            start, end = str(reference["u"]), str(reference["v"])
        elif orientation == "reverse_of_physical":
            start, end = str(reference["v"]), str(reference["u"])
        else:
            raise ValueError("Public physical road lacks source-bound geometry orientation: " + road_id)
        # edge_u/v denote a selected directed ref, not necessarily the stored
        # geometry's start/end. Wrong orientation would attach junctions to the
        # opposite BG even though the growth support itself is undirected.
        native_u, native_v = "road:" + start, "road:" + end
        adjacency.setdefault(native_u, set())
        adjacency.setdefault(native_v, set())
        hits = []
        for polygon_index in tree.query(geometry, predicate="intersects"):
            intersection = geometry.intersection(polygons.geometry.iloc[int(polygon_index)])
            for part in _line_parts(intersection):
                point = part.interpolate(.5, normalized=True)
                hits.append((float(geometry.project(point)), str(polygon_cells[polygon_index])))
        hits.sort()
        path = [native_u]
        for _, cell in hits:
            if path[-1] != cell:
                path.append(cell)
        path.append(native_v)
        add_path(path, road_id)
        road_evidence[road_id] = {
            "kind": "public_physical_road", "native_u": start,
            "native_v": end, "public_directed_refs": matching,
        }
        retained += 1

    missing = catalog.loc[catalog.community_id.isna()]
    for row in missing.itertuples(index=False):
        options = []
        if bool(row.requires_terminal_paths):
            for side in ("inbound", "outbound"):
                node = str(getattr(row, "terminal_" + side + "_gateway"))
                if "road:" + node in adjacency:
                    options.append((float(getattr(row, "terminal_" + side + "_length_m")), node))
        else:
            for ref in json.loads(row.directed_projection_offsets):
                identity = tuple(str(ref[k]) for k in ("u", "v", "key"))
                if identity not in public_refs:
                    continue
                fraction = float(ref["projection_fraction_from_u"])
                # Select one native gateway. The unit cannot bridge distinct
                # road components by turning a private endpoint into a shortcut.
                options.extend([(fraction, identity[0]), (1 - fraction, identity[1])])
        options = [option for option in options if "road:" + option[1] in adjacency]
        if not options:
            raise ValueError("Unsupported missing-BG location without a frozen public gateway: "
                             + str(row.location_id))
        _, gateway = min(options)
        proof_id = "native-access:" + str(row.location_id)
        road_evidence[proof_id] = {
            "kind": "frozen_native_endpoint_attachment", "location_id": str(row.location_id),
            "public_gateway": gateway, "requires_terminal_paths": bool(row.requires_terminal_paths),
            "source_physical_edge_id": str(getattr(row, "physical_edge_id", "")),
        }
        add_path([str(row.cell_id), "road:" + gateway], proof_id)

    contracted = contract_empty_units(adjacency, catalog.cell_id.astype(str).tolist())
    witness_rows, used = [], set()
    for (source, target), path in sorted(contracted.witness_paths.items()):
        if source >= target:
            continue
        proof_ids = set()
        for a, b in zip(path, path[1:]):
            proof_ids.update(link_evidence[tuple(sorted((a, b)))])
        used.update(proof_ids)
        witness_rows.append({"source": source, "target": target, "unit_path": list(path),
                             "road_evidence_ids": sorted(proof_ids)})
    return {
        "schema": "evrptw_road_supported_community_graph_v1",
        "adjacency": {g: list(neighbors) for g, neighbors in contracted.adjacency.items()},
        "witnesses": witness_rows,
        "road_evidence": {key: road_evidence[key] for key in sorted(used)},
        "public_physical_roads": retained,
        "occupied_cells": len(contracted.adjacency),
        "isolated_occupied_cells": [g for g, neighbors in contracted.adjacency.items() if not neighbors],
        "missing_bg_locations": len(missing),
        "growth_topology": "undirected public-road support; no routing cost inferred from topology",
        "empty_node_policy": "contract real road chains, retaining witnesses; no global outside node",
        "private_through_routing_allowed": False,
    }


def _select_depot(depots, config, city, community_path):
    policy = config.get("depot_sampling", "uniform_id")
    eligible = depots
    bg_count = missing_count = None
    if policy == "bg_uniform":
        eligible = _communities(depots, community_path, city)
        assigned = eligible.loc[eligible.bg_geoid.notna()]
        groups = [group for _, group in assigned.groupby("bg_geoid", sort=True)]
        bg_count, missing_count = len(groups), len(eligible) - len(assigned)
    if config.get("depot_id") is not None:
        depot = eligible.loc[eligible.location_id.eq(config["depot_id"])]
        if len(depot) != 1:
            raise ValueError("Requested depot_id is not an eligible canonical depot")
        label = "explicit canonical ID"
    elif policy == "bg_uniform":
        if not groups:
            raise ValueError("No eligible canonical depot has a Census BG for automatic selection")
        seed = int(stable_id(["depot_bg_uniform_v1", city, config["provider_id"], config["seed"]]), 16)
        rng = np.random.Generator(np.random.PCG64(seed))
        group = groups[int(rng.integers(len(groups)))].sort_values("location_id")
        depot = group.iloc[[int(rng.integers(len(group)))]]
        label = "uniform BG then uniform sorted canonical depot"
    else:
        seed = int(stable_id(["depot_uniform_v1", city, config["provider_id"], config["seed"]]), 16)
        index = int(np.random.Generator(np.random.PCG64(seed)).integers(len(depots)))
        depot = depots.iloc[[index]]
        label = "one uniform sorted-ID draw"
    return depot.copy().reset_index(drop=True), label, bg_count, missing_count


def _fastest_service_catalog(catalog, depot, speeds, config, vehicle, day_type):
    rate = float(vehicle["specific_energy_kwh_per_km"])
    usable = float(vehicle["battery_capacity_kwh"]) * (
        float(vehicle["initial_soc"]) - float(vehicle["reserve_soc"]))
    limit = float(config["service_limit_s"])
    router = FrozenCostRouter(speeds, day_type=day_type)
    frame_columns = ["location_id", "requires_terminal_paths", "connector_length_m",
                     "directed_projection_offsets"]
    ledger_columns = ["location_id", "connector_length_m", "connector_time_s_per_inverse_kph"]
    for side in ("inbound", "outbound"):
        frame_columns.extend([f"terminal_{side}_{suffix}"
                              for suffix in ("gateway", "length_m", "path_json")])
        ledger_columns.extend([f"{side}_terminal_length_m",
                               f"{side}_terminal_reference_{day_type}_time_s"])
    frame = pd.concat([depot[frame_columns], catalog[frame_columns]], ignore_index=True)
    ledger = pd.concat([depot[ledger_columns], catalog[ledger_columns]], ignore_index=True)
    source, *destinations = router.bind_endpoints(frame, ledger)
    distance, elapsed = router.source_costs(source, destinations)
    energy = distance * rate / 1000.
    within_time, within_energy = elapsed <= limit, energy <= usable
    catalog["depot_fastest_outward_distance_m"] = distance
    catalog["depot_fastest_outward_time_s"] = elapsed
    catalog["depot_fastest_outward_energy_kwh"] = energy
    return catalog.loc[within_time & within_energy].copy().reset_index(drop=True), {
        "policy": "fastest_time_energy", "day_type": day_type, "limit_s": limit,
        "usable_energy_kwh": usable, "specific_energy_kwh_per_km": rate,
        "finite_bidirectional_count": len(catalog),
        "time_qualified_count": int(within_time.sum()),
        "energy_qualified_count": int(within_energy.sum()),
        "source_cost_sha256": router.source_cost_sha256,
        "connector_speed_kph": router.connector_speed_kph,
    }


def build_territory_world(cle_path, community_path, world_config, outdir, progress=print, *,
                          vehicle=None, day_type="weekday"):
    """Select one depot and save its road-supported customer catalog."""
    config = dict(world_config)
    cle = _verified_cle(cle_path)
    from .cle_binding import cle_authority

    authority_path, validation_path = cle_authority(cle.path, cle.manifest)
    output, community_path = Path(outdir).resolve(), Path(community_path).resolve()
    names = ["catalog.parquet", "depot.parquet", "world.json", "community_graph.json"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Territory world already exists; use a new output directory")
    city = cle.manifest["city_slug"]
    source = {
        "cle_path": str(cle.path),
        "cle_manifest_sha256": sha256_file(cle.path / "cle_manifest.json"),
        "cle_verification_sha256": sha256_file(validation_path),
        "freeze_sha256": sha256_file(authority_path),
        "community_path": str(community_path), "community_sha256": sha256_file(community_path),
        "frozen_files": {name: item["sha256"] for name, item in cle.manifest["files"].items()},
    }
    depots = published_coordinates(_canonical_native(cle, "depot"))
    if depots.empty:
        raise ValueError("Frozen CLE has no eligible canonical depot; no retry")
    depot, depot_selection, bg_count, missing_bg_count = _select_depot(
        depots, config, city, community_path)
    customers = _communities(_canonical_native(cle, "customer"), community_path, city)
    propensity_config = {**WORLD_DEFAULTS, **config, "demand_model": "binomial_geometric",
                         "community_rate_cv": 0.0}
    customers = _propensities(customers, propensity_config, city)
    geodesic = great_circle_km(
        depot.location_lon.iat[0], depot.location_lat.iat[0],
        customers.location_lon.to_numpy(), customers.location_lat.to_numpy(),
    )
    fastest_policy = config.get("service_policy") == "fastest_time_energy"
    if fastest_policy and vehicle is None:
        raise ValueError("Fastest-time service scope requires the declared vehicle configuration")
    within = (np.ones(len(customers), dtype=bool) if fastest_policy
              else geodesic <= config["external_radius_km"])
    catalog = customers.loc[within].copy().reset_index(drop=True)
    catalog["depot_geodesic_distance_km"] = geodesic[within]
    if progress:
        progress({"stage": "territory_world_unbounded_depot_distances", "city": city,
                  "service_candidates": len(catalog)})
    speed_columns = [
        "edge_u", "edge_v", "edge_key", "length_m", "in_public_graph",
    ]
    if fastest_policy:
        speed_columns.extend(["operating_mode", f"reference_speed_{day_type}_kph",
                              f"reference_travel_time_{day_type}_s"])
    speeds = pd.read_parquet(cle.path / "speeds/directed_speeds.parquet", columns=speed_columns)
    provider = NativeDistanceProvider(speeds)
    outward, inward = provider.distances(depot, catalog)
    finite = np.isfinite(outward) & np.isfinite(inward)
    catalog["depot_outward_distance_m"] = outward
    catalog["depot_inward_distance_m"] = inward
    catalog = catalog.loc[finite].copy().reset_index(drop=True)
    service_scope = None
    if fastest_policy:
        catalog, service_scope = _fastest_service_catalog(
            catalog, depot, speeds, config, vehicle, day_type)
    catalog["cell_id"] = [str(value) if pd.notna(value) else "missing-bg:" + str(identity)
                          for value, identity in zip(catalog.community_id, catalog.location_id)]
    if progress:
        progress({"stage": "territory_world_road_supported_bg_graph", "city": city,
                  "finite_candidates": len(catalog)})
    roads = gpd.read_parquet(cle.path / "roads/physical_roads.parquet", columns=[
        "physical_edge_id", "directed_edge_refs", "edge_u", "edge_v", "geometry",
    ])
    bg = gpd.read_parquet(community_path)
    graph = road_supported_community_graph(roads, speeds, bg, catalog)
    field_id = stable_id({"schema": "territory_arrival_field_v1", "city": city,
                          "cle_manifest_sha256": source["cle_manifest_sha256"],
                          "community_sha256": source["community_sha256"],
                          "seed": config["seed"], "provider_id": config["provider_id"],
                          "p_unit": config["p_unit"],
                          "mean_parcels_per_active_unit": config["mean_parcels_per_active_unit"]})
    world_scope = {"external_radius_km": config["external_radius_km"]}
    if service_scope is not None:
        world_scope = {"service_scope": {k: service_scope[k] for k in (
            "policy", "day_type", "limit_s", "usable_energy_kwh", "specific_energy_kwh_per_km",
        )}}
    world_id = stable_id({"schema": "territory_operating_world_v1", "arrival_field_id": field_id,
                          "depot_id": depot.location_id.iat[0], **world_scope})
    catalog["world_id"], catalog["arrival_field_id"] = world_id, field_id
    depot["world_id"] = world_id
    output.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(output / "catalog.parquet", index=False)
    depot.to_parquet(output / "depot.parquet", index=False)
    (output / "community_graph.json").write_text(json.dumps(graph, ensure_ascii=False,
                                                           indent=2, allow_nan=False) + "\n")
    info = {
        "schema": "evrptw_territory_operating_world_v1", "world_id": world_id,
        "arrival_field_id": field_id, "city_slug": city, "provider_id": config["provider_id"],
        "depot_id": depot.location_id.iat[0], "config": config, "source": source,
        "catalog_path": str(output / "catalog.parquet"), "depot_path": str(output / "depot.parquet"),
        "world_path": str(output / "world.json"),
        "community_graph_path": str(output / "community_graph.json"),
        "artifacts": {name: sha256_file(output / name) for name in
                      ["catalog.parquet", "depot.parquet", "community_graph.json"]},
        "eligible_canonical_depots": len(depots), "eligible_city_customers": len(customers),
        "geodesic_candidate_count": None if fastest_policy else int(within.sum()),
        "customer_count": len(catalog),
        "nonfinite_road_candidates": int((~finite).sum()),
        "community_missing_city_count": int(customers.community_assignment_missing.sum()),
        "community_missing_world_count": int(catalog.community_assignment_missing.sum()),
        "imputed_city_locations": int(customers.units_imputed.sum()),
        "imputed_world_locations": int(catalog.units_imputed.sum()),
        "units_imputation_scope": "entire eligible frozen city before geographic filtering",
        "community_assignment": "frozen lon/lat covered_by; lowest GEOID; missing kept as native-access cell",
        "geography_policy": ("fastest_time_energy_and_finite_return" if fastest_policy
                             else "great_circle_and_unbounded_bidirectional_reachability"),
        "distance_policy": "complete frozen public graph plus native endpoint costs; same-ID distance zero",
        "depot_selection": depot_selection,
        "depot_redraw_count": 0, "terminal_through_routing_allowed": False,
        "real_carrier_territory_claimed": False, "parameters_are_synthetic_scenario": True,
        "stage1_modified": False, "stage2_frozen": False,
    }
    if bg_count is not None:
        info.update(depot_bg_count=bg_count, depot_missing_bg_count=missing_bg_count)
    if service_scope is not None:
        info["service_scope"] = service_scope
    (output / "world.json").write_text(json.dumps(info, ensure_ascii=False, indent=2,
                                                 allow_nan=False) + "\n")
    return NativeTerritoryWorld(info, catalog, depot,
                                {g: tuple(v) for g, v in graph["adjacency"].items()}, provider)
