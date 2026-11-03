"""Construct a fixed last-mile operating world from a verified frozen CLE.

Distance adapters use only public through-arcs and the frozen endpoint ledgers.
No operating day, customer count, vehicle or solver outcome changes this world.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from scipy.special import gammaincinv

from evrptw_cle.cle_reader import CityLogisticsEnvironment
from evrptw_cle.util import sha256_file
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ALIAS_TOL = 1e-10


def public_csr(speeds):
    """Return CSR, str-node index, source-edge lookup (length, unique index).

    Parallel (u,v) costs are minimized before sparse construction; COO duplicate
    summation would be incorrect. Source key identity is retained separately.
    """
    s = speeds.loc[speeds['in_public_graph'].astype(bool),
                   ['edge_u', 'edge_v', 'edge_key', 'length_m']].copy()
    for c in ['edge_u', 'edge_v', 'edge_key']:
        s[c] = s[c].astype(str)
    lengths = s.length_m.to_numpy(dtype=float)
    if not np.all(np.isfinite(lengths) & (lengths >= 0)):
        raise ValueError('Public edge lengths must be finite nonnegative')
    identities = list(s[['edge_u', 'edge_v', 'edge_key']].itertuples(index=False, name=None))
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate public source edge identity')
    edge_lookup = {identity: (float(length), ei)
                   for ei, (identity, length) in enumerate(zip(identities, lengths))}
    nodes = sorted(set(s.edge_u) | set(s.edge_v))
    node_index = {node: i for i, node in enumerate(nodes)}
    minimized = s.groupby(['edge_u', 'edge_v'], sort=False).length_m.min()
    u = np.fromiter((node_index[a] for a, b in minimized.index), dtype=np.int64)
    v = np.fromiter((node_index[b] for a, b in minimized.index), dtype=np.int64)
    graph = csr_matrix((minimized.to_numpy(dtype=float), (u, v)), shape=(len(nodes), len(nodes)))
    graph.sort_indices()
    return graph, node_index, edge_lookup


@dataclass
class Endpoints:
    ids: np.ndarray
    in_nodes: np.ndarray
    in_cost: np.ndarray
    out_nodes: np.ndarray
    out_cost: np.ndarray
    refs_by_row: list
    edge_members: dict
    connector: np.ndarray
    terminal: np.ndarray

    def __len__(self):
        return len(self.ids)


def _pack(rows, dummy):
    width = max([1] + [len(row) for row in rows])
    nodes = np.full((len(rows), width), dummy, dtype=np.int32)
    costs = np.full((len(rows), width), np.inf, dtype=float)
    for i, row in enumerate(rows):
        for j, (node, cost) in enumerate(row.items()):
            nodes[i, j] = node
            costs[i, j] = cost
    return nodes, costs


def build_endpoints(frame, node_index, edge_lookup, *, retain_ref=True):
    """Preserve frame order and use frame['_id'] as stable identity.

    Native columns required: requires_terminal_paths, connector_length_m,
    directed_projection_offsets, terminal_{inbound,outbound}_{gateway,length_m}.
    Public-edge fractions use native source lengths. edge_members maps each
    source edge index to (customer row indices, fractions from u).
    """
    ids = frame['_id'].astype(str).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError('Endpoint identities must be unique')
    terminal = frame['requires_terminal_paths'].astype(bool).to_numpy()
    connector = frame['connector_length_m'].to_numpy(dtype=float)
    if not np.all(np.isfinite(connector) & (connector >= 0)):
        raise ValueError('Invalid connector length')
    offsets = frame['directed_projection_offsets'].to_numpy()
    ing = frame['terminal_inbound_gateway'].astype(str).to_numpy()
    outg = frame['terminal_outbound_gateway'].astype(str).to_numpy()
    inl = frame['terminal_inbound_length_m'].to_numpy(dtype=float)
    outl = frame['terminal_outbound_length_m'].to_numpy(dtype=float)
    in_rows, out_rows, refs_by_row = [], [], []
    member_rows, member_fractions = defaultdict(list), defaultdict(list)
    for i in range(len(frame)):
        c = connector[i]
        if terminal[i]:
            if ing[i] not in node_index or outg[i] not in node_index:
                raise ValueError(f'Terminal gateway absent from public graph: {ids[i]}')
            if not (np.isfinite(inl[i]) and inl[i] >= 0 and np.isfinite(outl[i]) and outl[i] >= 0):
                raise ValueError(f'Invalid terminal path length: {ids[i]}')
            in_rows.append({node_index[ing[i]]: c + inl[i]})
            out_rows.append({node_index[outg[i]]: c + outl[i]})
            refs_by_row.append(())
            continue
        refs = json.loads(offsets[i])
        if not refs:
            raise ValueError(f'Public endpoint has no directed projection refs: {ids[i]}')
        incoming, outgoing, aliases, short_refs = {}, {}, set(), []
        for ref in refs:
            identity = tuple(str(ref[k]) for k in ('u', 'v', 'key'))
            if identity not in edge_lookup:
                raise ValueError(f'Public projection missing source edge {identity}: {ids[i]}')
            length, ei = edge_lookup[identity]
            f = float(ref['projection_fraction_from_u'])
            if not np.isfinite(f) or f < -1e-8 or f > 1 + 1e-8:
                raise ValueError('Invalid projection fraction')
            f = min(1., max(0., f))
            u, v = node_index[identity[0]], node_index[identity[1]]
            if f <= ALIAS_TOL:
                f = 0.
                aliases.add(u)
            elif f >= 1. - ALIAS_TOL:
                f = 1.
                aliases.add(v)
            incoming[u] = min(incoming.get(u, np.inf), c + f * length)
            outgoing[v] = min(outgoing.get(v, np.inf), c + (1. - f) * length)
            short_refs.append((ei, f, length))
            member_rows[ei].append(i)
            member_fractions[ei].append(f)
        if len(aliases) > 1:
            raise ValueError(f'Inconsistent native-node projection aliases: {ids[i]}')
        if aliases:
            # At a real node, all its native public turns are available.
            alias = next(iter(aliases))
            incoming, outgoing = {alias: c}, {alias: c}
        in_rows.append(incoming)
        out_rows.append(outgoing)
        refs_by_row.append(tuple(short_refs) if retain_ref else ())
    in_nodes, in_cost = _pack(in_rows, len(node_index))
    out_nodes, out_cost = _pack(out_rows, len(node_index))
    members = {ei: (np.asarray(rows, dtype=np.int32),
                    np.asarray(member_fractions[ei], dtype=float))
               for ei, rows in member_rows.items()}
    return Endpoints(ids, in_nodes, in_cost, out_nodes, out_cost,
                     refs_by_row, members, connector, terminal)


class DistanceEngine:
    """Two reusable CSR graphs with a mutable sparse virtual-root row.

    Each shortest-path call overwrites only a small number of root entries.
    The graph itself is not copied per depot. Not thread-safe; use one instance
    per process. Set limit_m to the largest R being evaluated for bounded runs.
    """
    def __init__(self, graph, limit_m=np.inf, root_slots=16):
        self.n = graph.shape[0]
        if graph.shape != (self.n, self.n):
            raise ValueError('Square directed graph required')
        self.limit_m = float(limit_m)
        self.root_slots = int(root_slots)
        self.forward = self._root_graph(graph.tocsr())
        self.reverse = self._root_graph(graph.T.tocsr())

    def _root_graph(self, graph):
        n, slots = self.n, self.root_slots
        data = np.concatenate([graph.data.astype(float), np.full(slots, np.inf)])
        indices = np.concatenate([graph.indices.astype(np.int32), np.full(slots, n, dtype=np.int32)])
        indptr = np.concatenate([graph.indptr.astype(np.int32), [graph.nnz + slots]]).astype(np.int32)
        return csr_matrix((data, indices, indptr), shape=(n + 1, n + 1))

    def distances(self, nodes, costs, *, reverse=False):
        graph = self.reverse if reverse else self.forward
        offset = graph.indptr[self.n]
        graph.indices[offset:] = self.n
        graph.data[offset:] = np.inf
        # Coalesce source options so duplicate gateways cannot be ambiguous.
        options = {}
        for node, cost in zip(nodes, costs):
            node, cost = int(node), float(cost)
            if node >= self.n or not np.isfinite(cost):
                continue
            if node < 0 or cost < 0:
                raise ValueError('Invalid source endpoint option')
            options[node] = min(options.get(node, np.inf), cost)
        if len(options) > self.root_slots:
            raise ValueError('Too many endpoint source options for root slots')
        for j, (node, cost) in enumerate(options.items()):
            graph.indices[offset + j] = node
            graph.data[offset + j] = cost
        # Include dummy/root index in output; padded endpoint cost is inf.
        return dijkstra(graph, directed=True, indices=self.n,
                        limit=self.limit_m, return_predecessors=False)


def pair_distances(di, depot_endpoints, customer_endpoints, engine):
    """Return canonical d->customer and customer->d shortest distances in m."""
    d, c = depot_endpoints, customer_endpoints
    forward = engine.distances(d.out_nodes[di], d.out_cost[di])
    reverse = engine.distances(d.in_nodes[di], d.in_cost[di], reverse=True)
    outward = np.min(forward[c.in_nodes] + c.in_cost, axis=1)
    inward = np.min(reverse[c.out_nodes] + c.out_cost, axis=1)
    # Source/destination may lie inside the same retained directed road edge.
    # The native-node formulation alone would force an unnecessary endpoint
    # visit. Direct subsegments keep original one-way directionality.
    for ei, f_depot, length in d.refs_by_row[di]:
        members = c.edge_members.get(ei)
        if members is None:
            continue
        rows, fractions = members
        connectors = d.connector[di] + c.connector[rows]
        delta = (fractions - f_depot) * length
        out_mask = delta >= -1e-9
        rr = rows[out_mask]
        outward[rr] = np.minimum(outward[rr], connectors[out_mask] + np.maximum(delta[out_mask], 0.))
        in_mask = delta <= 1e-9
        rr = rows[in_mask]
        inward[rr] = np.minimum(inward[rr], connectors[in_mask] + np.maximum(-delta[in_mask], 0.))
    return outward, inward


WORLD_DEFAULTS = {
    "seed": 0, "provider_id": "synthetic_provider", "depot_id": None,
    "radius_km": 25.0, "alpha": 1.0, "rho": 0.23, "kappa": 4.0,
    "unknown_units_shape": 1.0, "unknown_units_rate": 1.0,
    "demand_model": "binomial_geometric", "p_unit": 0.01,
    "mean_parcels_per_active_unit": 1.4, "community_rate_cv": 0.0,
}
_NATIVE = {
    "customer": ("customers/latent_locations.parquet", "site_id", "cle_default_instance_eligible"),
    "depot": ("depots/depots.parquet", "candidate_id", "depot_candidate_eligible"),
    "charging": ("charging/chargers.parquet", "charging_location_id", "charging_parameter_eligible"),
}


def stable_id(value):
    """Hash an unambiguous JSON value, independent of paths and Python hash seed."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _configuration(config):
    unknown = set(config) - set(WORLD_DEFAULTS)
    if unknown:
        raise ValueError("Unknown world parameters: " + ", ".join(sorted(unknown)))
    result = {**WORLD_DEFAULTS, **config}
    if type(result["seed"]) is not int or not result["provider_id"]:
        raise ValueError("seed must be an integer and provider_id must be nonempty")
    if not isinstance(result["provider_id"], str):
        raise ValueError("provider_id must be a string")
    for key in ["radius_km", "alpha", "kappa", "unknown_units_shape", "unknown_units_rate"]:
        value = float(result[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(key + " must be finite and positive")
        result[key] = value
    result["rho"] = float(result["rho"])
    if not np.isfinite(result["rho"]) or result["rho"] < 0:
        raise ValueError("rho must be finite and nonnegative")
    if result["depot_id"] is not None and not isinstance(result["depot_id"], str):
        raise ValueError("depot_id must be a canonical location_id or null")
    if result["demand_model"] not in {"binomial_geometric", "legacy_gamma_poisson"}:
        raise ValueError("Unknown demand_model")
    for key in ["p_unit", "community_rate_cv", "mean_parcels_per_active_unit"]:
        result[key] = float(result[key])
        if not np.isfinite(result[key]):
            raise ValueError(key + " must be finite")
    if not 0 <= result["p_unit"] <= 1 or result["community_rate_cv"] < 0 or result["mean_parcels_per_active_unit"] < 1:
        raise ValueError("Invalid binomial/geometric demand parameters")
    return result


def _verified_cle(path):
    from .cle_binding import BUILDING_SCHEMA, cle_authority

    cle = CityLogisticsEnvironment(path, verify_hashes=True)
    if cle.manifest["schema"] == BUILDING_SCHEMA:
        cle_authority(cle.path, cle.manifest)
    elif not cle.frozen:
        raise ValueError("Stage 2 requires an independently verified frozen CLE")
    return cle


def _canonical_native(cle, kind):
    """Join eligible canonical identities to their exact selected native access row."""
    table_name, key, gate = _NATIVE[kind]
    canonical = pd.read_parquet(cle.path / "locations.parquet")
    selected = canonical.loc[
        canonical.location_type.eq(kind) & canonical.candidate_eligible
    ].sort_values("location_id").reset_index(drop=True)
    if selected.location_id.duplicated().any() or selected.source_location_id.duplicated().any():
        raise ValueError("Canonical " + kind + " identities are not unique")
    if not selected.access_table.eq(table_name).all():
        raise ValueError("Canonical native access table differs from the frozen schema")
    native = gpd.read_parquet(cle.path / table_name)
    native[key] = native[key].astype(str)
    if native[key].duplicated().any():
        raise ValueError("Native access row identities are not unique: " + table_name)
    if not selected.access_row_id.astype(str).isin(native[key]).all():
        raise ValueError("Canonical location lacks its native source access row")
    result = native.set_index(key, drop=False).loc[
        selected.access_row_id.astype(str)
    ].reset_index(drop=True)
    if not result[gate].all():
        raise ValueError("Canonical eligible location references an ineligible native row")
    identity_column = "facility_group_id" if kind == "depot" else key
    if not np.array_equal(
        result[identity_column].astype(str).to_numpy(), selected.source_location_id.astype(str)
    ):
        raise ValueError("Canonical location identity does not match its native group/site")
    for column in selected.columns:
        if column == "geometry":
            continue
        if column in result:
            if column == "residential_units" and kind == "customer":
                if not np.allclose(result[column].to_numpy(dtype=float, na_value=np.nan),
                                   selected[column].to_numpy(dtype=float, na_value=np.nan),
                                   equal_nan=True):
                    raise ValueError("Canonical and native residential_units disagree")
            continue
        result[column] = selected[column].to_numpy()
    result["_id"] = result.location_id.astype(str)
    ledger = pd.read_parquet(cle.path / "location_access_costs.parquet")
    if ledger.location_id.duplicated().any():
        raise ValueError("Duplicate canonical access-cost ledger identity")
    if not result.location_id.isin(ledger.location_id).all():
        raise ValueError("Eligible canonical location lacks a source-bound access-cost ledger")
    bound = ledger.set_index("location_id").loc[result.location_id].reset_index(drop=True)
    for native_column, ledger_column in [
        ("connector_length_m", "connector_length_m"),
        ("terminal_inbound_length_m", "inbound_terminal_length_m"),
        ("terminal_outbound_length_m", "outbound_terminal_length_m"),
    ]:
        if not np.allclose(result[native_column].to_numpy(dtype=float),
                           bound[ledger_column].to_numpy(dtype=float), rtol=1e-10, atol=1e-7):
            raise ValueError("Native access and canonical ledger disagree: " + native_column)
    for column in bound:
        if column != "location_id" and column not in result:
            result[column] = bound[column].to_numpy()
    return result


def _distance_to_frame(cle, depot, destinations, radius_km):
    speeds = pd.read_parquet(cle.path / "speeds/directed_speeds.parquet", columns=[
        "edge_u", "edge_v", "edge_key", "length_m", "in_public_graph"
    ])
    graph, nodes, lookup = public_csr(speeds)
    source = build_endpoints(depot, nodes, lookup)
    targets = build_endpoints(destinations, nodes, lookup, retain_ref=False)
    slots = max(1, source.out_nodes.shape[1], source.in_nodes.shape[1])
    engine = DistanceEngine(graph, limit_m=radius_km * 1000, root_slots=slots)
    return pair_distances(0, source, targets, engine)


def _communities(customers, community_path, city):
    bg = gpd.read_parquet(community_path)
    if bg.crs is None:
        raise ValueError("Community polygons require an explicit CRS")
    key = next((k for k in ["bg_geoid", "GEOID", "geoid", "census_block_group_geoid"]
                if k in bg), None)
    if key is None or bg[key].isna().any():
        raise ValueError("Community polygons require nonnull Census BG GEOIDs")
    bg = bg[[key, "geometry"]].rename(columns={key: "bg_geoid"}).to_crs(4326)
    bg["bg_geoid"] = bg.bg_geoid.astype(str)
    if not bg.bg_geoid.str.fullmatch(r"\d{12}").all():
        raise ValueError("Community identifiers must be 12-digit Census block-group GEOIDs")
    if bg.geometry.isna().any() or bg.geometry.is_empty.any() or not bg.geometry.is_valid.all():
        raise ValueError("Community geometry must be nonempty and valid; no automatic repair")
    # Multiple geometry pieces for one GEOID are valid; lowest GEOID resolves
    # coincident boundary memberships without relocating the frozen point.
    bg = bg.sort_values("bg_geoid", kind="stable").reset_index(drop=True)
    xy = customers[["location_lon", "location_lat"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("Frozen customer representative points must be finite")
    pairs = shapely.STRtree(bg.geometry.to_numpy()).query(shapely.points(xy), predicate="covered_by")
    assigned = np.full(len(customers), None, dtype=object)
    hit_counts = np.bincount(pairs[0], minlength=len(customers))
    if pairs.shape[1]:
        ordered = np.lexsort((bg.bg_geoid.to_numpy()[pairs[1]], pairs[0]))
        row, polygon = pairs[:, ordered]
        unique = np.r_[True, row[1:] != row[:-1]]
        assigned[row[unique]] = bg.bg_geoid.to_numpy()[polygon[unique]]
    customers["bg_geoid"] = pd.array(assigned, dtype="string")
    customers["community_id"] = pd.array(
        [None if value is None else city + ":tiger2025:bg:" + value for value in assigned],
        dtype="string",
    )
    customers["community_match_count"] = hit_counts
    customers["community_assignment_missing"] = customers.community_id.isna()
    return customers


def _legacy_propensities(customers, config, city):
    units = customers.residential_units.to_numpy(dtype=float, na_value=np.nan)
    known = np.isfinite(units) & (units > 0)
    if np.any(~known & ~np.isnan(units)):
        raise ValueError("Known residential_units must be finite and positive")
    # Inverse-CDF draws from independent named SHA256 uniform streams. Each
    # site retains its draw when depot, row order, radius or target N changes.
    uniforms = np.fromiter((
        ((int(stable_id(["site_propensity_v1", city, config["provider_id"],
                         config["seed"], site])[:16], 16) >> 11) + 0.5) / (1 << 53)
        for site in customers.site_id.astype(str)
    ), dtype=float, count=len(customers))
    uniforms = np.clip(uniforms, np.nextafter(0., 1.), np.nextafter(1., 0.))
    shape = np.where(known, config["alpha"] * units, config["unknown_units_shape"])
    rate = np.where(known, config["alpha"], config["unknown_units_rate"])
    customers["G"] = gammaincinv(shape, uniforms) / rate
    if not (np.isfinite(customers.G) & customers.G.gt(0)).all():
        raise ValueError("Nonpositive or nonfinite site propensity")
    customers["G_units_known"] = known
    customers["expected_daily_packages"] = config["rho"] * customers.G
    customers["expected_daily_activation_probability"] = -np.expm1(
        -config["kappa"] * np.log1p(config["rho"] * customers.G / config["kappa"])
    )
    return customers


def _propensities(customers, config, city):
    """Model units from the entire eligible city before any depot catchment.

    The frozen raw field is never overwritten. Known values are rounded half-up
    only in the modeled count; missing apt units use their city/type median.
    Ambiguous residential types use the city-wide median to avoid a strongly
    selected, tiny known-unit ambiguous subgroup determining their estimates.
    """
    if config.get("demand_model", "legacy_gamma_poisson") == "legacy_gamma_poisson":
        return _legacy_propensities(customers, config, city)
    raw = pd.to_numeric(customers.residential_units, errors="coerce")
    known = raw.notna() & np.isfinite(raw) & raw.gt(0)
    if not known.any():
        raise ValueError("No positive city residential units for the declared median imputation")
    city_median = float(raw[known].median())
    modeled = raw.copy()
    policies = pd.Series("source_positive_half_up_integer", index=customers.index, dtype=object)
    medians = pd.Series(np.nan, index=customers.index, dtype=float)
    kind = customers.get("residential_type", pd.Series("residential_type_ambiguous", index=customers.index)).fillna("residential_type_ambiguous").astype(str)
    for name, indices in kind.groupby(kind, sort=True).groups.items():
        good = known.loc[indices]
        unknown_type = name == "residential_type_ambiguous"
        values = raw.loc[indices][good]
        estimate = city_median if unknown_type or values.empty else float(values.median())
        reference = max(1, int(np.floor(estimate + .5)))
        missing = indices[~good]
        modeled.loc[missing] = reference
        medians.loc[missing] = estimate
        policies.loc[missing] = ("city_all_residential_median_unknown_type" if unknown_type else
                                 "city_type_median" if len(values) else "city_all_residential_median_no_type_support")
    effective = np.maximum(1, np.floor(modeled.to_numpy(dtype=float) + .5)).astype(np.int64)
    customers["modeled_residential_units"] = effective
    customers["units_imputed"] = ~known
    customers["unit_imputation_policy"] = policies
    customers["unit_imputation_reference_median"] = medians
    customers["unit_imputation_city"] = city
    # G remains an explicit compatibility alias, not a latent Gamma draw.
    customers["G"] = effective.astype(float)
    customers["G_units_known"] = known
    customers["expected_daily_packages"] = effective * config["p_unit"] * config["mean_parcels_per_active_unit"]
    customers["expected_daily_activation_probability"] = (np.ones(len(effective)) if config["p_unit"] == 1 else
        -np.expm1(effective * np.log1p(-config["p_unit"])))
    customers["expected_activation_probability_reference_community_factor"] = 1.0
    return customers


def build_world(cle_path, community_path, world_config, output_dir):
    """Write a fixed depot catchment and stable city/provider/site demand field.

    A depot is selected once, uniformly over sorted eligible canonical IDs.
    The catchment requires both directed depot-to-site distances <= R. Missing
    community membership is retained and explicitly flagged, never snapped.
    No customer-count, vehicle or solver-driven retry is permitted.
    """
    config = _configuration(world_config)
    from .cle_binding import cle_authority
    cle = _verified_cle(cle_path)
    authority, verification = cle_authority(cle.path, cle.manifest)
    community_path = Path(community_path).resolve()
    output = Path(output_dir).resolve()
    names = ["catalog.parquet", "depot.parquet", "world.json"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("World outputs already exist; use a new output directory")
    source = {
        "cle_path": str(cle.path),
        "cle_manifest_sha256": sha256_file(cle.path / "cle_manifest.json"),
        "cle_verification_sha256": sha256_file(verification),
        "freeze_sha256": sha256_file(authority),
        "community_path": str(community_path), "community_sha256": sha256_file(community_path),
        "frozen_files": {name: item["sha256"] for name, item in cle.manifest["files"].items()},
    }
    city = cle.manifest["city_slug"]
    depots = _canonical_native(cle, "depot")
    if depots.empty:
        raise ValueError("Frozen CLE has no eligible canonical depot; no retry")
    if config["depot_id"] is None:
        seed = int(stable_id(["depot_uniform_v1", city, config["provider_id"], config["seed"]]), 16)
        index = int(np.random.Generator(np.random.PCG64(seed)).integers(len(depots)))
        depot = depots.iloc[[index]].copy().reset_index(drop=True)
    else:
        depot = depots.loc[depots.location_id.eq(config["depot_id"])].copy().reset_index(drop=True)
        if len(depot) != 1:
            raise ValueError("Requested depot_id is not an eligible canonical depot")
    customers = _canonical_native(cle, "customer")
    customers = _communities(customers, community_path, city)
    customers = _propensities(customers, config, city)
    outward, inward = _distance_to_frame(cle, depot, customers, config["radius_km"])
    inside = np.isfinite(outward) & np.isfinite(inward)
    inside &= (outward <= config["radius_km"] * 1000) & (inward <= config["radius_km"] * 1000)
    customers["depot_outward_distance_m"] = outward
    customers["depot_inward_distance_m"] = inward
    catalog = customers.loc[inside].copy().reset_index(drop=True)
    field_contract = {key: config[key] for key in [
        "seed", "provider_id", "alpha", "rho", "kappa", "unknown_units_shape", "unknown_units_rate",
        "demand_model", "p_unit", "mean_parcels_per_active_unit", "community_rate_cv"
    ]}
    field_id = stable_id({"schema": "arrival_field_v1", "city": city,
                          "cle_manifest_sha256": source["cle_manifest_sha256"],
                          "community_sha256": source["community_sha256"], "parameters": field_contract})
    world_id = stable_id({"schema": "fixed_depot_catchment_v1", "arrival_field_id": field_id,
                          "depot_id": depot.location_id.iat[0], "radius_km": config["radius_km"]})
    catalog["world_id"] = world_id
    catalog["arrival_field_id"] = field_id
    depot["world_id"] = world_id
    output.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(output / "catalog.parquet", index=False, compression="zstd")
    depot.to_parquet(output / "depot.parquet", index=False, compression="zstd")
    result = {
        "schema": "evrptw_fixed_operating_world_v1", "world_id": world_id,
        "arrival_field_id": field_id, "city_slug": city, "provider_id": config["provider_id"],
        "depot_id": depot.location_id.iat[0], "config": config, "source": source,
        "catalog_path": str(output / "catalog.parquet"),
        "depot_path": str(output / "depot.parquet"), "world_path": str(output / "world.json"),
        "artifacts": {name: sha256_file(output / name) for name in names[:2]},
        "eligible_canonical_depots": len(depots), "eligible_city_customers": len(customers),
        "customer_count": len(catalog),
        "community_missing_city_count": int(customers.community_assignment_missing.sum()),
        "community_missing_world_count": int(catalog.community_assignment_missing.sum()),
        "community_assignment": "frozen location_lon/lat; covered_by; lowest GEOID; no nearest fill",
        "propensity_policy": ("city/type median modeled units; binomial active units and positive geometric parcels"
                              if config["demand_model"] == "binomial_geometric" else
                              "independent named SHA256 uniform inverse-Gamma-CDF per city/provider/seed/site"),
        "units_imputation_scope": "entire eligible frozen city before depot/radius filtering",
        "imputed_city_locations": int(customers.get("units_imputed", pd.Series(False, index=customers.index)).sum()),
        "imputed_world_locations": int(catalog.get("units_imputed", pd.Series(False, index=catalog.index)).sum()),
        "geography_policy": "both directed shortest physical distances <= radius; fixed native endpoints",
        "depot_selection": "explicit canonical ID" if config["depot_id"] else "one uniform sorted-ID draw",
        "depot_redraw_count": 0, "terminal_through_routing_allowed": False,
        "real_carrier_territory_claimed": False, "parameters_are_synthetic_scenario": True,
        "stage1_modified": False, "stage2_frozen": False,
    }
    (output / "world.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result
