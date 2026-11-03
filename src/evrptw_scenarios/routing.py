"""Paired shortest-distance and fastest-time costs on the same frozen CLE.

Only pandas/numpy/scipy are required. No other analysis/output module is used.
The caller verifies the frozen CLE certificate before supplying tables.

API::

    router = FrozenCostRouter(speeds, day_type="weekday")
    endpoints = router.bind_endpoints(canonical_frame, access_costs)
    costs = router.route(endpoints, specific_energy_kwh_per_km=.25,
                         audit_pairs=[(0, 1)], batch_size=8)
    path = router.path(endpoints, 0, 1)

``canonical_frame`` has canonical location_id, requires_terminal_paths,
connector_length_m, directed_projection_offsets, terminal_{inbound,outbound}_
{gateway,length_m,path_json}. ``canonical_endpoint_frame`` joins these from
locations.access_table/access_row_id; it does not choose an alternate entrance.

Each policy accumulates both distance and time from its own predecessor path.
The execution aliases distance_m/time_s/energy_kwh use shortest distance; the
fastest-time family is retained separately for alternative execution and masks. Real terminal segment time comes from
the exact native source edge and fractional length, checked against the frozen
access ledger. Only geometric connector time uses a declared proxy speed.
All private/terminal roads remain outside the public through-routing CSR.

Road travel is a static weekday/weekend reference snapshot, not time-dependent
traffic. No turn penalties, observed signal delays or queueing are invented.
Energy is a declared distance-proportional scenario, not a vehicle telemetry fit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.csgraph import dijkstra

ALIAS_TOL = 1e-10


@dataclass(frozen=True)
class Segment:
    kind: str
    edge: tuple[str, str, str] | None
    length_m: float
    time_s: float
    fraction_start: float | None = None
    fraction_end: float | None = None

    def as_dict(self):
        return {"kind": self.kind, "source_edge": list(self.edge) if self.edge else None,
                "length_m": self.length_m, "time_s": self.time_s,
                "fraction_start": self.fraction_start, "fraction_end": self.fraction_end}


@dataclass(frozen=True)
class EndpointOption:
    node: int
    length_m: float
    time_s: float
    segments: tuple[Segment, ...]


@dataclass(frozen=True)
class Endpoint:
    location_id: str
    inbound: tuple[EndpointOption, ...]
    outbound: tuple[EndpointOption, ...]
    public_refs: tuple[tuple[tuple[str, str, str], float], ...]
    connector: Segment


@dataclass
class CostBundle:
    location_ids: list[str]
    distance_m: np.ndarray
    time_s: np.ndarray
    energy_kwh: np.ndarray
    source_option: np.ndarray
    destination_option: np.ndarray
    # 0=public-path closure, 1=same-native-edge direct, 2=self-zero, 3=unreachable.
    path_mode: np.ndarray
    audit_paths: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    distance_shortest_path_distance: np.ndarray
    distance_shortest_path_time: np.ndarray
    time_shortest_path_distance: np.ndarray
    time_shortest_path_time: np.ndarray
    time_shortest_path_energy_kwh: np.ndarray
    time_shortest_path_source_option: np.ndarray
    time_shortest_path_destination_option: np.ndarray
    time_shortest_path_mode: np.ndarray
    fastest_audit_paths: dict[str, dict[str, Any]]
    matrix_fields: ClassVar[tuple[str, ...]] = (
        "distance_m", "time_s", "energy_kwh", "source_option",
        "destination_option", "path_mode", "distance_shortest_path_distance",
        "distance_shortest_path_time", "time_shortest_path_distance",
        "time_shortest_path_time", "time_shortest_path_energy_kwh",
        "time_shortest_path_source_option", "time_shortest_path_destination_option",
        "time_shortest_path_mode")


def _boolean(values, name):
    arr = np.asarray(values)
    if pd.isna(arr).any() or not np.isin(arr, [True, False, 0, 1]).all():
        raise ValueError(f"{name} must be a nonmissing boolean column")
    return arr.astype(bool)


def _parsed(value):
    return json.loads(value) if isinstance(value, str) else value


def canonical_endpoint_frame(city_dir, location_ids):
    """Read canonical eligible endpoint rows in requested order.

    This helper does not replace CityLogisticsEnvironment verification. The
    caller must verify the frozen CLE before using it. Returns (frame, ledger).
    """
    root = Path(city_dir).resolve()
    ids = list(map(str, location_ids))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate requested canonical locations")
    locations = pd.read_parquet(root / "locations.parquet", columns=[
        "location_id", "candidate_eligible", "location_type", "access_table",
        "access_row_id"]).set_index("location_id")
    if not locations.index.is_unique:
        raise ValueError("Canonical location IDs are not unique")
    selected = locations.loc[ids]
    if not _boolean(selected.candidate_eligible, "candidate_eligible").all():
        raise ValueError("An excluded/pending location was requested")
    native_keys = {"customer": "site_id", "depot": "candidate_id",
                   "charging": "charging_location_id"}
    records = {}
    for access_table, group in selected.groupby("access_table", sort=True):
        path = (root / str(access_table)).resolve()
        if not path.is_relative_to(root):
            raise ValueError("External canonical access table")
        kind = str(group.location_type.iloc[0])
        if set(group.location_type.astype(str)) != {kind} or kind not in native_keys:
            raise ValueError("Unknown or ambiguous canonical table type")
        key = native_keys[kind]
        access_columns = ["requires_terminal_paths", "connector_length_m",
            "directed_projection_offsets", "terminal_inbound_gateway",
            "terminal_outbound_gateway", "terminal_inbound_length_m",
            "terminal_outbound_length_m", "terminal_inbound_path_json",
            "terminal_outbound_path_json"]
        native = pd.read_parquet(path, columns=[key, *access_columns])
        native[key] = native[key].astype(str)
        native = native.set_index(key)
        if not native.index.is_unique:
            raise ValueError("Native access row IDs are not unique")
        for location_id, row in group.iterrows():
            record = native.loc[str(row.access_row_id)].to_dict()
            record["location_id"] = str(location_id)
            records[str(location_id)] = record
    return pd.DataFrame([records[x] for x in ids]), pd.read_parquet(
        root / "location_access_costs.parquet")


class FrozenCostRouter:
    def __init__(self, speeds: pd.DataFrame, *, day_type="weekday",
                 connector_speed_kph: float | None = None):
        if day_type not in {"weekday", "weekend"}:
            raise ValueError("day_type must be weekday or weekend")
        self.day_type = day_type
        self.time_column = f"reference_travel_time_{day_type}_s"
        needed = ["edge_u", "edge_v", "edge_key", "length_m", "in_public_graph",
                  self.time_column]
        source = speeds[needed].copy()
        source[["edge_u", "edge_v", "edge_key"]] = source[
            ["edge_u", "edge_v", "edge_key"]].astype(str)
        if source.duplicated(["edge_u", "edge_v", "edge_key"]).any():
            raise ValueError("Ambiguous native source edge identities")
        numeric = source[["length_m", self.time_column]].to_numpy(dtype=float)
        if not np.all(np.isfinite(numeric) & (numeric > 0)):
            raise ValueError("Native source length and time must be finite positive")
        public_mask = _boolean(source.in_public_graph, "in_public_graph")
        self.edge_lookup = {tuple(identity): (float(length), float(time))
                            for identity, (length, time) in zip(
                                source[["edge_u", "edge_v", "edge_key"]].itertuples(
                                    index=False, name=None), numeric)}
        self.public_edges = set(map(tuple, source.loc[public_mask,
            ["edge_u", "edge_v", "edge_key"]].itertuples(index=False, name=None)))
        public_all = source.loc[public_mask]
        if public_all.empty:
            raise ValueError("No public source edges")
        self.node_ids = sorted(set(public_all.edge_u) | set(public_all.edge_v))
        self.node_index = {node: index for index, node in enumerate(self.node_ids)}
        self.graphs, self.policy_arc_lengths, self.policy_arc_times = {}, {}, {}
        self.policy_arc_identities = {}
        for policy, primary, secondary in [
            ("shortest_distance", "length_m", self.time_column),
            ("fastest_time", self.time_column, "length_m"),
        ]:
            # SciPy sums duplicate CSR arcs. Select the correct native arc for
            # each objective before constructing a graph, retaining its ID.
            public = public_all.sort_values(
                ["edge_u", "edge_v", primary, secondary, "edge_key"], kind="stable"
            ).drop_duplicates(["edge_u", "edge_v"], keep="first")
            u = np.asarray([self.node_index[x] for x in public.edge_u], dtype=np.int32)
            v = np.asarray([self.node_index[x] for x in public.edge_v], dtype=np.int32)
            graph = csr_matrix((public[primary].to_numpy(dtype=float), (u, v)),
                               shape=(len(self.node_ids), len(self.node_ids)))
            graph.sort_indices()
            if not (np.array_equal(graph.indices, v) and np.array_equal(
                    np.repeat(np.arange(len(self.node_ids)), np.diff(graph.indptr)), u)):
                raise ValueError("Unexpected CSR/native-source ordering")
            self.graphs[policy] = graph
            self.policy_arc_lengths[policy] = public.length_m.to_numpy(dtype=float)
            self.policy_arc_times[policy] = public[self.time_column].to_numpy(dtype=float)
            self.policy_arc_identities[policy] = tuple(public[
                ["edge_u", "edge_v", "edge_key"]].itertuples(index=False, name=None))
        # Preserve the old public graph attribute for inspection. Routing always
        # chooses its objective-specific graph explicitly.
        self.graph = self.graphs["fastest_time"]
        self.arc_lengths = self.policy_arc_lengths["fastest_time"]
        self.arc_identities = self.policy_arc_identities["fastest_time"]
        if connector_speed_kph is None:
            speed_column = f"reference_speed_{day_type}_kph"
            pool = speeds.loc[public_mask]
            urban = pool.loc[pool.operating_mode.astype(str).eq("U")]
            chosen = urban if len(urban) else pool
            values = chosen[speed_column].to_numpy(dtype=float)
            weights = chosen.length_m.to_numpy(dtype=float)
            if not np.all(np.isfinite(values) & (values > 0)):
                raise ValueError("Invalid public speed for connector proxy")
            order = np.argsort(values, kind="stable")
            position = min(int(np.searchsorted(np.cumsum(weights[order]),
                           weights.sum() / 2, side="left")), len(order)-1)
            connector_speed_kph = float(values[order[position]])
            self.connector_policy = "directed_source_length_weighted_public_U_speed_median" if len(urban) else "all_public_speed_median_no_U"
        else:
            self.connector_policy = "explicit_scenario_connector_speed"
        self.connector_speed_kph = float(connector_speed_kph)
        if not np.isfinite(self.connector_speed_kph) or self.connector_speed_kph <= 0:
            raise ValueError("Positive finite connector speed is required")
        digest = hashlib.sha256()
        for row in source.sort_values(["edge_u", "edge_v", "edge_key"]).itertuples(index=False, name=None):
            digest.update(json.dumps(row, separators=(",", ":"), default=str).encode())
            digest.update(b"\n")
        self.source_cost_sha256 = digest.hexdigest()

    def _native(self, identity, start=0.0, end=1.0, *, kind="public_native"):
        length, time = self.edge_lookup[identity]
        if not 0 <= start <= end <= 1:
            raise ValueError("Invalid directed native subsegment")
        fraction = end-start
        return Segment(kind, identity, fraction*length, fraction*time, start, end)

    def _terminal_segments(self, raw):
        segments = []
        for ref in _parsed(raw):
            identity = tuple(str(ref[k]) for k in ("u", "v", "key"))
            length, time = self.edge_lookup[identity]
            segment_length = float(ref["length_m"])
            fraction = segment_length/length
            if not np.isfinite(fraction) or fraction < 0 or fraction > 1+1e-8:
                raise ValueError("Terminal segment exceeds bound native source edge")
            segments.append(Segment("terminal_native", identity, segment_length,
                                     time*fraction))
        return tuple(segments)

    @staticmethod
    def _option(node, segments):
        return EndpointOption(node, sum(x.length_m for x in segments),
                              sum(x.time_s for x in segments), tuple(segments))

    @staticmethod
    def _best_options(options):
        # Do not collapse two access paths to the same gateway by time alone:
        # a slower but shorter offset can be optimal for distance routing.
        unique = {str(option): option for option in options}
        return tuple(sorted(unique.values(), key=lambda x: (
            x.node, x.length_m, x.time_s, str(x.segments))))

    @staticmethod
    def _selected_options(options, policy):
        by_node = {}
        for index, option in enumerate(options):
            score = ((option.length_m, option.time_s) if policy == "shortest_distance"
                     else (option.time_s, option.length_m))
            previous = by_node.get(option.node)
            if previous is None or (*score, index) < previous[0]:
                by_node[option.node] = ((*score, index), index, option)
        return {node: (value[1], value[2]) for node, value in sorted(by_node.items())}

    def bind_endpoints(self, frame: pd.DataFrame, access_costs: pd.DataFrame,
                       *, id_column="location_id") -> tuple[Endpoint, ...]:
        ids = frame[id_column].astype(str)
        if not ids.is_unique:
            raise ValueError("Endpoint location IDs must be unique")
        ledger = access_costs.set_index("location_id") if "location_id" in access_costs else access_costs
        if not ledger.index.is_unique:
            raise ValueError("Access ledger identities must be unique")
        selected = ledger.loc[ids]
        terminal_flags = _boolean(frame.requires_terminal_paths, "requires_terminal_paths")
        endpoints = []
        for index, ((_, row), (_, bound)) in enumerate(zip(frame.iterrows(), selected.iterrows())):
            connector_length = float(row.connector_length_m)
            if not np.isfinite(connector_length) or connector_length < 0:
                raise ValueError("Invalid connector distance")
            if not np.isclose(connector_length, float(bound.connector_length_m), rtol=1e-10, atol=1e-7):
                raise ValueError("Endpoint does not use canonical ledger entrance")
            factor = float(bound.connector_time_s_per_inverse_kph)
            if not np.isclose(factor, 3.6*connector_length, rtol=1e-12, atol=1e-9):
                raise ValueError("Connector ledger time factor differs from length")
            connector = Segment("geometry_connector", None, connector_length,
                                factor/self.connector_speed_kph)
            # Check both terminal chains even for public projections (empty).
            chains = {}
            for direction in ("inbound", "outbound"):
                chains[direction] = self._terminal_segments(row[f"terminal_{direction}_path_json"])
                length = sum(x.length_m for x in chains[direction])
                time = sum(x.time_s for x in chains[direction])
                if not (np.isclose(length, float(row[f"terminal_{direction}_length_m"]), rtol=1e-9, atol=1e-6)
                        and np.isclose(length, float(bound[f"{direction}_terminal_length_m"]), rtol=1e-9, atol=1e-6)
                        and np.isclose(time, float(bound[f"{direction}_terminal_reference_{self.day_type}_time_s"]), rtol=1e-9, atol=1e-6)):
                    raise ValueError(f"Native terminal {direction} costs differ from canonical ledger: {ids.iloc[index]}")
            if terminal_flags[index]:
                ing = self.node_index[str(row.terminal_inbound_gateway)]
                outg = self.node_index[str(row.terminal_outbound_gateway)]
                inbound = (self._option(ing, (*chains["inbound"], connector)),)
                outbound = (self._option(outg, (connector, *chains["outbound"])),)
                refs = ()
            else:
                incoming, outgoing, refs, aliases = [], [], [], set()
                for ref in _parsed(row.directed_projection_offsets):
                    identity = tuple(str(ref[k]) for k in ("u", "v", "key"))
                    if identity not in self.public_edges:
                        raise ValueError("Public endpoint references a nonpublic source edge")
                    fraction = float(ref["projection_fraction_from_u"])
                    if not np.isfinite(fraction) or not -1e-8 <= fraction <= 1+1e-8:
                        raise ValueError("Invalid source projection fraction")
                    fraction = min(1., max(0., fraction))
                    if fraction <= ALIAS_TOL:
                        fraction = 0.
                        aliases.add(self.node_index[identity[0]])
                    elif fraction >= 1-ALIAS_TOL:
                        fraction = 1.
                        aliases.add(self.node_index[identity[1]])
                    incoming.append(self._option(self.node_index[identity[0]],
                                    (self._native(identity, end=fraction), connector)))
                    outgoing.append(self._option(self.node_index[identity[1]],
                                    (connector, self._native(identity, start=fraction))))
                    refs.append((identity, fraction))
                if len(aliases) > 1:
                    raise ValueError("Conflicting native node aliases")
                if aliases:
                    alias = next(iter(aliases))
                    incoming = outgoing = [self._option(alias, (connector,))]
                if not incoming:
                    raise ValueError("Public endpoint has no access projection")
                inbound, outbound = self._best_options(incoming), self._best_options(outgoing)
                refs = tuple(refs)
            endpoints.append(Endpoint(str(ids.iloc[index]), inbound, outbound, refs, connector))
        return tuple(endpoints)

    def _augmented(self, endpoints, policy):
        public_graph = self.graphs[policy]
        n, count = public_graph.shape[0], len(endpoints)
        indices, data, indptr = [], [], [0]
        for endpoint in endpoints:
            for _, option in self._selected_options(endpoint.outbound, policy).values():
                indices.append(option.node)
                data.append(option.length_m if policy == "shortest_distance" else option.time_s)
            indptr.append(len(indices))
        roots = csr_matrix((np.asarray(data), np.asarray(indices, dtype=np.int32),
                            np.asarray(indptr, dtype=np.int64)), shape=(count, n+count))
        public = csr_matrix((public_graph.data, public_graph.indices, public_graph.indptr),
                            shape=(n, n+count))
        return vstack([public, roots], format="csr")

    def _arc_position(self, source, destination, policy):
        graph = self.graphs[policy]
        start, stop = graph.indptr[source:source+2]
        offset = int(np.searchsorted(graph.indices[start:stop], destination))
        if offset >= stop-start or graph.indices[start+offset] != destination:
            raise ValueError("Predecessor does not correspond to a public native arc")
        return int(start+offset)

    def _tree_helpers(self, endpoint, predecessor, root, policy):
        cache = {root: (0., 0., -1)}
        by_node = self._selected_options(endpoint.outbound, policy)

        def cost(target):
            node, chain = int(target), []
            while node not in cache:
                parent = int(predecessor[node])
                if parent == -9999:
                    return np.inf, np.inf, -1
                chain.append((parent, node))
                node = parent
            for parent, child in reversed(chain):
                if parent == root:
                    option_index, option = by_node[child]
                    cache[child] = (option.length_m, option.time_s, option_index)
                else:
                    position = self._arc_position(parent, child, policy)
                    length, time, origin = cache[parent]
                    cache[child] = (length + float(self.policy_arc_lengths[policy][position]),
                                   time + float(self.policy_arc_times[policy][position]), origin)
            return cache[int(target)]

        def segments(node):
            arcs = []
            while int(predecessor[node]) != root:
                parent = int(predecessor[node])
                if parent < 0:
                    raise ValueError("Cannot reconstruct unreachable public path")
                arcs.append(self._native(self.policy_arc_identities[policy][
                    self._arc_position(parent, node, policy)]))
                node = parent
            _, option = by_node[int(node)]
            return (*option.segments, *reversed(arcs))
        return cost, segments

    def _source_row(self, source_index, endpoints, distance, predecessor, root,
                    energy_rate, audit_destinations=(), *, policy="shortest_distance"):
        source, count = endpoints[source_index], len(endpoints)
        dd, tt = np.full(count, np.inf), np.full(count, np.inf)
        so, do = np.full(count, -1, dtype=np.int16), np.full(count, -1, dtype=np.int16)
        mode = np.full(count, 3, dtype=np.uint8)
        choice = {}
        tree_cost, tree_segments = self._tree_helpers(source, predecessor, root, policy)
        for j, destination in enumerate(endpoints):
            if j == source_index:
                dd[j] = tt[j] = 0.
                mode[j], choice[j] = 2, ()
                continue
            candidates = []
            for k, option in enumerate(destination.inbound):
                if not np.isfinite(distance[option.node]):
                    continue
                length, time, source_option = tree_cost(option.node)
                length, time = length + option.length_m, time + option.time_s
                score = (length, time) if policy == "shortest_distance" else (time, length)
                candidates.append((*score, k, option, source_option, length, time))
            if candidates:
                _, _, k, option, source_option, length, time = min(candidates, key=lambda x: x[:3])
                dd[j], tt[j] = length, time
                so[j], do[j], mode[j] = source_option, k, 0
                choice[j] = ("closure", option)
            source_refs = dict(source.public_refs)
            for identity, target_fraction in destination.public_refs:
                if identity not in source_refs:
                    continue
                start = source_refs[identity]
                if target_fraction < start-ALIAS_TOL:
                    continue
                direct = self._native(identity, start, max(start, target_fraction))
                segments = (source.connector, direct, destination.connector)
                time, length = sum(x.time_s for x in segments), sum(x.length_m for x in segments)
                primary, secondary = (length, time) if policy == "shortest_distance" else (time, length)
                old_primary, old_secondary = (dd[j], tt[j]) if policy == "shortest_distance" else (tt[j], dd[j])
                if primary < old_primary-1e-10 or (abs(primary-old_primary) <= 1e-10 and secondary < old_secondary):
                    dd[j], tt[j], mode[j] = length, time, 1
                    so[j] = do[j] = -1
                    choice[j] = ("direct", segments)
        audit = {}
        for j in audit_destinations:
            record = {"source": source.location_id, "destination": endpoints[j].location_id,
                      "reachable": bool(np.isfinite(tt[j])), "path_mode": int(mode[j]),
                      "path_policy": policy,
                      "distance_m": float(dd[j]), "time_s": float(tt[j]),
                      "energy_kwh": float(dd[j]*energy_rate/1000), "segments": []}
            if mode[j] == 0:
                option = choice[j][1]
                parts = (*tree_segments(option.node), *option.segments)
            elif mode[j] == 1:
                parts = choice[j][1]
            else:
                parts = ()
            record["segments"] = [segment.as_dict() for segment in parts]
            if np.isfinite(tt[j]) and not (
                np.isclose(sum(x.length_m for x in parts), dd[j], atol=1e-6, rtol=1e-9)
                and np.isclose(sum(x.time_s for x in parts), tt[j], atol=1e-6, rtol=1e-9)):
                raise ValueError("Path audit does not reconstruct matrix distance/time")
            audit[f"{source_index},{j}"] = record
        return dd, tt, so, do, mode, audit

    def _route_policy(self, endpoints, rate, batch_size, pairs, policy):
        count, n = len(endpoints), len(self.node_ids)
        graph = self._augmented(endpoints, policy)
        dd, tt = np.empty((count, count)), np.empty((count, count))
        so, do = np.empty((count, count), dtype=np.int16), np.empty((count, count), dtype=np.int16)
        mode, audit = np.empty((count, count), dtype=np.uint8), {}
        for first in range(0, count, batch_size):
            stop = min(count, first+batch_size)
            distances, predecessors = dijkstra(graph, directed=True,
                indices=np.arange(n+first, n+stop), return_predecessors=True)
            for offset, index in enumerate(range(first, stop)):
                result = self._source_row(index, endpoints, distances[offset],
                    predecessors[offset], n+index, rate,
                    [j for i, j in pairs if i == index], policy=policy)
                dd[index], tt[index], so[index], do[index], mode[index] = result[:5]
                audit.update(result[5])
        return dd, tt, so, do, mode, audit

    def source_costs(self, source: Endpoint, destinations: Iterable[Endpoint],
                     *, policy="fastest_time"):
        """Return one origin's distance and time vectors on the same policy path."""
        if policy not in self.graphs:
            raise ValueError("Unknown source-path policy: " + str(policy))
        root = len(self.node_ids)
        graph = self._augmented((source,), policy)
        distances, predecessors = dijkstra(
            graph, directed=True, indices=root, return_predecessors=True,
        )
        endpoints = (source, *destinations)
        distance, elapsed, *_ = self._source_row(
            0, endpoints, distances, predecessors, root, 1., policy=policy,
        )
        return distance[1:], elapsed[1:]

    def route(self, endpoints: Iterable[Endpoint], *, specific_energy_kwh_per_km,
              batch_size=8, audit_pairs=(), require_all_reachable=True) -> CostBundle:
        endpoints = tuple(endpoints)
        count = len(endpoints)
        if not count or len({x.location_id for x in endpoints}) != count:
            raise ValueError("Nonempty unique canonical endpoints required")
        rate = float(specific_energy_kwh_per_km)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("Positive finite scenario energy rate required")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Positive integer batch_size required")
        pairs = list(audit_pairs)
        if any(not 0 <= i < count or not 0 <= j < count for i, j in pairs):
            raise ValueError("Invalid audit pair indices")
        primary = self._route_policy(endpoints, rate, batch_size, pairs, "shortest_distance")
        fastest = self._route_policy(endpoints, rate, batch_size, pairs, "fastest_time")
        dd, tt, so, do, mode, audit = primary
        fd, ft, fso, fdo, fmode, faudit = fastest
        unreachable = int(np.sum(~np.isfinite(tt)))
        if require_all_reachable and unreachable:
            raise ValueError(f"Selected canonical terminals have {unreachable} unreachable directed pairs")
        metadata = {"schema": "frozen_cle_paired_path_cost_bundle_v2",
            "path_policy": f"shortest_distance_{self.day_type}",
            "day_type": self.day_type,
            "execution_policy": "shortest_distance",
            "primary_objective": "total_distance",
            "paired_path_policies": ["shortest_distance", "fastest_time"],
            "matrix_units": {"distance_shortest_path_distance": "m",
                             "distance_shortest_path_time": "s",
                             "time_shortest_path_distance": "m",
                             "time_shortest_path_time": "s"},
            "fastest_time_mask_role": "optimistic lower bound; not execution time on shortest-distance paths",
            "source_cost_sha256": self.source_cost_sha256,
            "source_time_column": self.time_column,
            "distance_is_same_time_path": True,
            "energy_policy": "declared_scalar_kwh_per_km_on_same_path_distance",
            "specific_energy_kwh_per_km": rate,
            "connector_speed_kph": self.connector_speed_kph,
            "connector_policy": self.connector_policy,
            "measured_connector_speed": False,
            "terminal_through_routing_allowed": False,
            "turn_penalties_s": 0,
            "unreachable_directed_pairs": unreachable,
            "path_reconstruction": "same_router_source_costs_plus_bound_endpoints",
            "tie_policy": "parallel arc minimum policy objective then secondary cost then lexical key; deterministic scipy CSR predecessor; endpoint/direct ties use secondary cost",
            "batch_size": batch_size}
        return CostBundle([x.location_id for x in endpoints], dd, tt, dd*rate/1000,
                          so, do, mode, audit, metadata, dd, tt, fd, ft, fd*rate/1000,
                          fso, fdo, fmode, faudit)

    def path(self, endpoints: Iterable[Endpoint], source_index: int,
             destination_index: int, *, specific_energy_kwh_per_km,
             policy="shortest_distance"):
        """Reconstruct any single matrix pair with exact native edge IDs.

        Supply the explicit scenario energy rate used for the matrices.
        """
        endpoints = tuple(endpoints)
        if not (0 <= source_index < len(endpoints) and 0 <= destination_index < len(endpoints)):
            raise ValueError("Invalid endpoint index")
        rate = float(specific_energy_kwh_per_km)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("Positive finite scenario energy rate required")
        if policy not in self.graphs:
            raise ValueError("policy must be shortest_distance or fastest_time")
        graph = self._augmented(endpoints, policy)
        root = len(self.node_ids)+source_index
        distances, predecessors = dijkstra(graph, directed=True, indices=root,
                                           return_predecessors=True)
        row = self._source_row(source_index, endpoints, distances, predecessors, root,
                              rate, [destination_index], policy=policy)
        return row[-1][f"{source_index},{destination_index}"]
