"""Local-first physical road access, with real OSM terminal paths kept separate.

The public graph never uses terminal ways as through-routing shortcuts. Source
locations match the complete local physical inventory before permission and
connectivity checks. An unavailable local road never triggers a distant snap.
"""

from __future__ import annotations

import gc
import hashlib
import heapq
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import LineString
from shapely.ops import substring, transform
from shapely.strtree import STRtree

from .customer_access import attach_footprint_access
from .road_tags import DELIVERY_ROAD_CLASSES, EXCLUDED_ANCHOR_ROAD_CLASSES
from .terminal_access import _effective_motor_access, classify_terminal_access_way
from .util import sha256_file, write_json

POLICY = "local_first_real_osm_access_v1"
MOTOR_CLASSES = DELIVERY_ROAD_CLASSES | EXCLUDED_ANCHOR_ROAD_CLASSES
BUFFER_LADDER_KM = (0, 1, 2, 5, 10, 20)
PATH_GEOMETRY_POLICY = "metric_geometry_exact_projection_v1"


def repair_terminal_path_geometry(frame, physical):
    """Upgrade an in-progress v1 table; admission/costs are byte-for-byte values.

    Projected metric offsets cannot be reused as normalized longitude/latitude
    curve lengths. Only the first/last partial path geometry is affected.
    """
    if (
        "terminal_path_geometry_policy" in frame
        and frame.terminal_path_geometry_policy.eq(PATH_GEOMETRY_POLICY).all()
    ):
        return frame, False

    result = frame.copy()
    relevant = set(result.loc[result.requires_terminal_paths, "physical_edge_id"])
    selected = physical.loc[physical.physical_edge_id.isin(relevant)]
    geometry = dict(zip(selected.physical_edge_id, selected.geometry))
    subset = result.loc[result.requires_terminal_paths]
    anchors = gpd.GeoSeries(
        gpd.points_from_xy(subset.road_anchor_lon, subset.road_anchor_lat), crs=4326
    ).to_crs(physical.crs)
    to_wgs = Transformer.from_crs(physical.crs, 4326, always_xy=True)
    for row, point in zip(subset.itertuples(), anchors):
        if not np.isfinite(row.road_anchor_lon):
            continue
        original = geometry[row.physical_edge_id]
        for direction in ["inbound", "outbound"]:
            column = "terminal_" + direction + "_path_json"
            route = json.loads(getattr(row, column))
            if not route:
                continue
            ref = route[-1] if direction == "inbound" else route[0]
            line = shapely.reverse(original) if ref["key"].endswith("r") else original
            position = line.project(point)
            part = (
                substring(line, 0.0, position)
                if direction == "inbound"
                else substring(line, position, line.length)
            )
            ref["geometry_wkt"] = transform(to_wgs.transform, part).wkt
            result.at[row.Index, column] = json.dumps(route, separators=(",", ":"))
    result["terminal_path_geometry_policy"] = PATH_GEOMETRY_POLICY
    return result, True


def classify_way(tags):
    highway = tags.get("highway", "")
    _, legal = _effective_motor_access(tags)
    if highway not in MOTOR_CLASSES or tags.get("area") == "yes":
        return None
    terminal = classify_terminal_access_way(tags)
    layer = (
        "osm_terminal_only"
        if highway == "service" or legal == "permit_required"
        else "operational_public"
    )
    allowed = legal not in {"prohibited", "permit_required", "unresolved_tag_value"}
    if highway == "service":
        allowed = allowed and terminal["terminal_core_eligible"]
    if highway in {"construction", "proposed", "raceway"}:
        allowed = False
    return dict(
        highway=highway,
        legal_access_tier=legal,
        access_layer=layer,
        allowed=bool(allowed),
        anchor_class_eligible=highway in DELIVERY_ROAD_CLASSES,
        connector_kind=terminal["connector_kind"]
        if layer == "osm_terminal_only"
        else "through_road",
    )


def _directions(tags):
    value = tags.get("oneway", "").lower()
    if value in {"-1", "reverse"}:
        return (False, True)
    if value in {"yes", "true", "1"} or (
        value not in {"no", "false", "0"} and tags.get("junction") in {"roundabout", "circular"}
    ):
        return (True, False)
    return (True, True)


def _forest(graph, seeds, *, reverse=False):
    """Distance-ranked directed forest to/from public SCC; no geometric joins."""
    distances = {str(n): 0.0 for n in seeds if n in graph}
    roots = {n: n for n in distances}
    parent = {}
    todo = [(0.0, n) for n in distances]
    heapq.heapify(todo)
    adjacency = graph.pred if reverse else graph.succ
    while todo:
        distance, u = heapq.heappop(todo)
        if distance != distances.get(u):
            continue
        for v, edges in adjacency[u].items():
            key, data = min(edges.items(), key=lambda x: (float(x[1]["length"]), str(x[0])))
            value = distance + float(data["length"])
            if value < distances.get(v, float("inf")) - 1e-9:
                distances[v] = value
                roots[v] = roots[u]
                parent[v] = (u, (v, u, key) if reverse else (u, v, key))
                heapq.heappush(todo, (value, v))
    return dict(distance=distances, root=roots, parent=parent, reverse=reverse)


def build_access_context(
    *, source_dir, admin_boundary_path, service_boundary_path, mask_path, output_dir, area_crs
):
    """Build reproducible public/access graphs and the local physical inventory."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extraction = json.loads((source_dir / "extraction.json").read_text())
    if extraction["pbf_replication_timestamp_utc"] != "2026-10-25T21:21:21Z":
        raise ValueError("Unaccepted road snapshot")
    for name, digest in extraction["outputs"].items():
        if sha256_file(source_dir / name) != digest:
            raise ValueError("Changed road extraction: " + name)
    admin = gpd.read_file(admin_boundary_path).to_crs(area_crs).geometry.union_all()
    nodes = gpd.read_parquet(source_dir / "osm_access_nodes.parquet")
    blocked_nodes = set(
        nodes.loc[
            nodes.tags_json.map(lambda v: _effective_motor_access(json.loads(v))[1]).isin(
                ["prohibited", "permit_required", "unresolved_tag_value"]
            ),
            "osmid",
        ]
    )
    masks = gpd.read_parquet(mask_path).to_crs(area_crs)
    masktree = STRtree(masks.geometry.to_numpy()) if len(masks) else None
    ways = gpd.read_parquet(source_dir / "osm_ways.parquet").to_crs(area_crs)
    parsed = []
    usage = Counter()
    for row in ways.itertuples():
        tags = json.loads(row.tags_json)
        kind = classify_way(tags)
        if kind is None:
            continue
        ids = [str(n) for n in json.loads(row.node_ids_json)]
        coords = list(row.geometry.coords)
        if len(ids) != len(coords):
            raise ValueError("Incomplete OSM node geometry")
        usage.update(ids)
        parsed.append((row.osmid, ids, coords, tags, kind))
    del ways
    gc.collect()
    junctions = {n for n, count in usage.items() if count > 1} | blocked_nodes
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    records = []
    metric_geometries = []
    wgs_nodes = {}
    # Coordinate transformation is vectorized after topological segmentation.
    node_metric = {}
    for osmid, ids, coords, tags, kind in parsed:
        cuts = sorted({0, len(ids) - 1, *[i for i, n in enumerate(ids) if n in junctions]})
        for start, end in zip(cuts[:-1], cuts[1:]):
            if start == end:
                continue
            segment = LineString(coords[start : end + 1])
            length = float(segment.length)
            if length <= 0:
                continue
            u, v = ids[start], ids[end]
            node_metric[u] = coords[start]
            node_metric[v] = coords[end]
            identity = hashlib.sha1(
                (osmid + "|" + u + "|" + v + "|" + str(start)).encode()
            ).hexdigest()[:20]
            allowed = kind["allowed"] and not any(n in blocked_nodes for n in ids[start : end + 1])
            mask_hit = False
            if masktree is not None:
                # Do not cut public roads merely because they pass beside a POI
                # mask. Interior facility roads wholly inside a mask are blocked.
                near = masktree.query(segment)
                mask_hit = any(masks.geometry.iloc[int(j)].covers(segment) for j in near)
                allowed = allowed and not mask_hit
            refs = []
            forward, backward = _directions(tags)
            for reversed_, a, b in [(False, u, v), (True, v, u)]:
                if (reversed_ and not backward) or (not reversed_ and not forward):
                    continue
                key = identity + ("r" if reversed_ else "f")
                refs.append(
                    dict(
                        u=a,
                        v=b,
                        key=key,
                        osmid=[osmid],
                        length_m=length,
                        highway=[kind["highway"]],
                        name=[tags.get("name", "")],
                        geometry_orientation="reverse_of_physical"
                        if reversed_
                        else "same_as_physical",
                    )
                )
            record = dict(
                physical_edge_id=identity,
                directed_edge_refs=json.dumps(refs, sort_keys=True),
                directed_edge_ref_count=len(refs),
                edge_u=refs[0]["u"],
                edge_v=refs[0]["v"],
                edge_key=refs[0]["key"],
                highway=kind["highway"],
                road_name=tags.get("name", ""),
                oneway=not (forward and backward),
                **{k: kind[k] for k in ["access_layer", "connector_kind", "legal_access_tier"]},
                motor_access_allowed=allowed,
                source_way_motor_allowed=kind["allowed"],
                node_access_blocked=any(n in blocked_nodes for n in ids[start : end + 1]),
                sensitive_interior_road=mask_hit,
                anchor_class_eligible=kind["anchor_class_eligible"],
                osmid=osmid,
                maxspeed=tags.get("maxspeed", ""),
                maxspeed_forward=tags.get("maxspeed:forward", ""),
                maxspeed_backward=tags.get("maxspeed:backward", ""),
            )
            records.append(record)
            metric_geometries.append(segment)
    del parsed
    gc.collect()
    xy = gpd.GeoSeries(shapely.points(list(node_metric.values())), crs=area_crs).to_crs(4326)
    for n, point in zip(node_metric, xy):
        wgs_nodes[n] = dict(x=float(point.x), y=float(point.y))
    physical = gpd.GeoDataFrame(records, geometry=metric_geometries, crs=area_crs)
    physical.to_parquet(output_dir / "physical_roads.parquet", index=False)
    wgs = physical.to_crs(4326)
    for row in wgs.loc[wgs.motor_access_allowed].itertuples():
        for ref in json.loads(row.directed_edge_refs):
            u, v, key = ref["u"], ref["v"], ref["key"]
            if u not in graph:
                graph.add_node(u, **wgs_nodes[u])
            if v not in graph:
                graph.add_node(v, **wgs_nodes[v])
            reverse = ref["geometry_orientation"] == "reverse_of_physical"
            geom = shapely.reverse(row.geometry) if reverse else row.geometry
            graph.add_edge(
                u,
                v,
                key=key,
                length=ref["length_m"],
                osmid=row.osmid,
                highway=row.highway,
                name=row.road_name,
                oneway=row.oneway,
                reversed=reverse,
                geometry=geom,
                access_layer=row.access_layer,
                legal_access_tier=row.legal_access_tier,
                through_routing_allowed=row.access_layer == "operational_public",
                **{
                    "maxspeed": row.maxspeed,
                    "maxspeed:forward": row.maxspeed_forward,
                    "maxspeed:backward": row.maxspeed_backward,
                },
            )
    public = nx.subgraph_view(
        graph, filter_edge=lambda u, v, k: graph[u][v][k]["through_routing_allowed"]
    )
    public = public.subgraph([n for n in public if public.degree(n) > 0])
    all_pubnodes = list(public.nodes)
    metric_points = shapely.points([node_metric[n] for n in all_pubnodes])
    outside = dict(zip(all_pubnodes, shapely.distance(metric_points, admin)))
    citynodes = {n for n, d in outside.items() if d <= 1e-6}
    trials = []
    selections = []
    for buffer_km in BUFFER_LADDER_KM:
        retain = {n for n in all_pubnodes if outside[n] <= buffer_km * 1000 + 1e-6}
        candidate = public.subgraph(retain)
        components = nx.weakly_connected_components(candidate)
        root = max(components, key=lambda comp: (len(comp & citynodes), len(comp), min(comp)))
        covered = root & citynodes
        trials.append(
            dict(
                buffer_km=buffer_km,
                city_nodes_covered=len(covered),
                city_node_count=len(citynodes),
                city_node_coverage=len(covered) / max(len(citynodes), 1),
            )
        )
        selections.append((set(root), set(covered)))
    maximum = selections[-1][1]
    pick = next(i for i, (_, covered) in enumerate(selections) if maximum <= covered)
    core = public.subgraph(selections[pick][0]).copy()
    for n in core:
        core.nodes[n]["inside_city"] = n in citynodes
        core.nodes[n]["transit_only"] = n not in citynodes
    for u, v, k, data in core.edges(keys=True, data=True):
        data["transit_only"] = u not in citynodes or v not in citynodes
    reference = max(nx.strongly_connected_components(core), key=lambda c: (len(c), min(c)))
    # The terminal graph keeps only non-public-core edges. Its trees terminate
    # at public SCC gateways, so terminal roads never connect two through nodes.
    access = nx.MultiDiGraph(crs="EPSG:4326")
    for u, v, k, data in graph.edges(keys=True, data=True):
        if u in reference and v in reference and data["through_routing_allowed"]:
            continue
        for n in (u, v):
            if n not in access:
                access.add_node(n, **graph.nodes[n])
        access.add_edge(u, v, key=k, **data)
    gateways = set(access) & set(reference)
    inbound = _forest(access, gateways)
    outbound = _forest(access, gateways, reverse=True)
    context = LocalAccessContext(core, access, physical, reference, inbound, outbound, area_crs)
    service_shape = gpd.read_file(service_boundary_path).to_crs(area_crs).geometry.union_all()
    context.water = admin.difference(service_shape)
    context.save(output_dir)
    # Raw city public graph is retained for same-snapshot coverage diagnosis.
    raw = public.subgraph(citynodes).copy()
    ox.save_graphml(raw, output_dir / "graph_all.graphml")
    manifest = dict(
        schema="evrptw_local_first_road_access_v1",
        policy=POLICY,
        city_slug=extraction["city_slug"],
        source=extraction,
        area_crs=area_crs,
        admin_boundary_sha256=sha256_file(Path(admin_boundary_path)),
        service_boundary_sha256=sha256_file(Path(service_boundary_path)),
        mask_sha256=sha256_file(Path(mask_path)),
        selection="smallest buffer recovering every city public node reachable at the maximum real-OSM envelope; never stop at global percentage alone",
        selected_buffer_km=BUFFER_LADDER_KM[pick],
        buffer_trials=trials,
        reference_scc_nodes=len(reference),
        core_nodes=len(core),
        core_directed_edges=core.number_of_edges(),
        terminal_graph_nodes=len(access),
        terminal_graph_directed_edges=access.number_of_edges(),
        local_physical_segments=len(physical),
        terminal_through_routing_allowed=False,
        synthetic_intercomponent_edges=False,
        outputs={
            name: sha256_file(output_dir / name)
            for name in [
                "graph_all.graphml",
                "graph_operational.graphml",
                "graph_terminal_access.graphml",
                "physical_roads.parquet",
                "access_forests.json",
                "connector_water.parquet",
            ]
        },
    )
    write_json(output_dir / "access_context_manifest.json", manifest)
    return context, manifest


@dataclass
class LocalAccessContext:
    core: Any
    access: Any
    physical: Any
    reference: set
    inbound: dict
    outbound: dict
    area_crs: str
    water: Any = None

    def save(self, dest):
        ox.save_graphml(self.core, Path(dest) / "graph_operational.graphml")
        ox.save_graphml(self.access, Path(dest) / "graph_terminal_access.graphml")
        write_json(
            Path(dest) / "access_forests.json",
            dict(reference=sorted(self.reference), inbound=self.inbound, outbound=self.outbound),
        )
        if self.water is not None:
            gpd.GeoDataFrame(
                {"source": ["frozen_admin_minus_service"]}, geometry=[self.water], crs=self.area_crs
            ).to_parquet(Path(dest) / "connector_water.parquet", index=False)

    @classmethod
    def load(cls, dest):
        dest = Path(dest)
        m = json.loads((dest / "access_context_manifest.json").read_text())
        if cls is LocalAccessContext and m.get("policy") == "delivery_terminal_access_v2":
            from .delivery_access import DeliveryAccessContext

            return DeliveryAccessContext.load(dest)
        if m.get("policy") == "delivery_terminal_access_v2":
            from .delivery_access import HOLDS_PATH

            if sha256_file(HOLDS_PATH) != m["review_holds_sha256"]:
                raise ValueError("Changed delivery access review holds; rebuild context explicitly")
        for name, digest in m["outputs"].items():
            if sha256_file(dest / name) != digest:
                raise ValueError("Changed access context " + name)
        f = json.loads((dest / "access_forests.json").read_text())
        water_path = dest / "connector_water.parquet"
        if water_path.exists():
            water = gpd.read_parquet(water_path).to_crs(m["area_crs"]).geometry.union_all()
        else:
            # Compatibility only for this in-progress eleven-city migration.
            base = (
                Path(__file__).resolve().parents[2] / "boundaries/us-11city-2025" / m["city_slug"]
            )
            if (
                sha256_file(base / "admin_boundary.geojson") != m["admin_boundary_sha256"]
                or sha256_file(base / "land_boundary.geojson") != m["service_boundary_sha256"]
            ):
                raise ValueError("Frozen connector-water boundary differs")
            water = (
                gpd.read_file(base / "admin_boundary.geojson")
                .to_crs(m["area_crs"])
                .geometry.union_all()
                .difference(
                    gpd.read_file(base / "land_boundary.geojson")
                    .to_crs(m["area_crs"])
                    .geometry.union_all()
                )
            )
        return cls(
            ox.load_graphml(
                dest / "graph_operational.graphml",
                node_dtypes={"osmid": str},
                edge_dtypes={"osmid": str},
            ),
            ox.load_graphml(
                dest / "graph_terminal_access.graphml",
                node_dtypes={"osmid": str},
                edge_dtypes={"osmid": str},
            ),
            gpd.read_parquet(dest / "physical_roads.parquet"),
            set(f["reference"]),
            f["inbound"],
            f["outbound"],
            m["area_crs"],
            water,
        )

    def _path(self, node, forest):
        if node in self.reference:
            return []
        refs = []
        while node in forest["parent"]:
            parent, (u, v, key) = forest["parent"][node]
            data = self.access[u][v][key]
            refs.append(
                dict(
                    u=u,
                    v=v,
                    key=key,
                    osmid=str(data["osmid"]),
                    length_m=float(data["length"]),
                    fraction=1.0,
                    highway=data["highway"],
                    geometry_wkt=data["geometry"].wkt,
                    delivery_authorization_assumed=str(
                        data.get("delivery_authorization_assumed", False)
                    ).lower()
                    in {"true", "1"},
                )
            )
            node = parent
        if not forest["reverse"]:
            refs.reverse()
        return refs

    def anchor(self, locations, *, masks=None, footprints=None, water=None, selected_edge_ids=None):
        """Match local geometry once; do not fall back when the closest road fails."""
        if water is None:
            water = self.water
        roads = self.physical.loc[self.physical.anchor_class_eligible].reset_index(drop=True)
        chosen = (
            None
            if selected_edge_ids is None
            else pd.Index(roads.physical_edge_id).get_indexer(selected_edge_ids)
        )
        frame = attach_footprint_access(
            locations, roads, self.area_crs, matched_edge_indices=chosen
        )
        info = roads.set_index("physical_edge_id")
        frame["motor_access_allowed"] = frame.physical_edge_id.map(
            info.motor_access_allowed
        ).astype(bool)
        frame["source_way_motor_allowed"] = frame.physical_edge_id.map(
            info.source_way_motor_allowed
        ).astype(bool)
        frame["node_access_blocked"] = frame.physical_edge_id.map(info.node_access_blocked).astype(
            bool
        )
        frame["sensitive_interior_road"] = frame.physical_edge_id.map(
            info.sensitive_interior_road
        ).astype(bool)
        incoming = []
        outgoing = []
        in_roots = []
        out_roots = []
        in_lengths = []
        out_lengths = []
        reasons = []
        metric_anchors = gpd.GeoSeries(
            gpd.points_from_xy(frame.road_anchor_lon, frame.road_anchor_lat), crs=4326
        ).to_crs(self.area_crs)
        to_graph = Transformer.from_crs(self.area_crs, self.core.graph["crs"], always_xy=True)
        for row, metric_anchor in zip(frame.itertuples(), metric_anchors):
            choices_in = []
            choices_out = []
            for ref in json.loads(row.directed_projection_offsets):
                u, v, key = ref["u"], ref["v"], ref["key"]
                if not row.motor_access_allowed:
                    continue
                for target, forest, node, partial in [
                    (choices_in, self.inbound, u, ref["offset_from_u_m"]),
                    (choices_out, self.outbound, v, ref["offset_to_v_m"]),
                ]:
                    public_edge = self.core.has_edge(u, v, key)
                    if public_edge and u in self.reference and v in self.reference:
                        target.append((0.0, "", [], False))
                        continue
                    if node not in self.reference and node not in forest["distance"]:
                        continue
                    path = self._path(node, forest)
                    graph = self.core if public_edge else self.access
                    if not graph.has_edge(u, v, key):
                        continue
                    data = graph[u][v][key]
                    fraction = partial / max(float(ref["length_m"]), 1e-12)
                    geom = info.geometry[row.physical_edge_id]
                    if ref["geometry_orientation"] == "reverse_of_physical":
                        geom = shapely.reverse(geom)
                    position = geom.project(metric_anchor)
                    partgeom = (
                        substring(geom, 0.0, position)
                        if target is choices_in
                        else substring(geom, position, geom.length)
                    )
                    partgeom = transform(to_graph.transform, partgeom)
                    part = dict(
                        u=u,
                        v=v,
                        key=key,
                        osmid=str(data["osmid"]),
                        length_m=float(partial),
                        fraction=float(fraction),
                        highway=data["highway"],
                        geometry_wkt=partgeom.wkt,
                        delivery_authorization_assumed=str(
                            data.get("delivery_authorization_assumed", False)
                        ).lower()
                        in {"true", "1"},
                    )
                    route = path + [part] if target is choices_in else [part] + path
                    root = node if node in self.reference else forest["root"][node]
                    target.append(
                        (
                            float(forest["distance"].get(node, 0.0)) + float(partial),
                            root,
                            route,
                            True,
                        )
                    )
            cin = (
                min(choices_in, key=lambda x: (x[0], x[1]))
                if choices_in
                else (np.nan, "", [], False)
            )
            cout = (
                min(choices_out, key=lambda x: (x[0], x[1]))
                if choices_out
                else (np.nan, "", [], False)
            )
            incoming.append(json.dumps(cin[2], separators=(",", ":")))
            outgoing.append(json.dumps(cout[2], separators=(",", ":")))
            in_roots.append(cin[1])
            out_roots.append(cout[1])
            in_lengths.append(cin[0])
            out_lengths.append(cout[0])
            if row.sensitive_interior_road:
                reason = "local_road_within_excluded_facility"
            elif not row.motor_access_allowed:
                reason = "local_road_" + (
                    "node_access_restricted" if row.node_access_blocked else row.legal_access_tier
                )
            elif not choices_in or not choices_out:
                reason = "local_road_no_verified_directed_roundtrip"
            else:
                reason = ""
            reasons.append(reason)
        frame["terminal_inbound_path_json"] = incoming
        frame["terminal_outbound_path_json"] = outgoing
        frame["terminal_inbound_gateway"] = in_roots
        frame["terminal_outbound_gateway"] = out_roots
        frame["terminal_inbound_length_m"] = in_lengths
        frame["terminal_outbound_length_m"] = out_lengths
        frame["requires_terminal_paths"] = pd.Series(in_roots).ne("") | pd.Series(out_roots).ne("")
        frame["terminal_path_geometry_policy"] = PATH_GEOMETRY_POLICY
        # Source-to-local-road geometry remains an explicit proxy. Reject mapped
        # obstacles; no numeric distance threshold changes eligibility.
        source = locations.to_crs(self.area_crs).geometry.to_numpy()
        accesspoints = (
            gpd.GeoSeries(
                gpd.points_from_xy(frame.road_anchor_lon, frame.road_anchor_lat), crs=4326
            )
            .to_crs(self.area_crs)
            .to_numpy()
        )
        connectors = shapely.shortest_line(source, accesspoints)
        if masks is not None and len(masks):
            tree = STRtree(masks.to_crs(self.area_crs).geometry.to_numpy())
            pairs = tree.query(connectors, predicate="intersects")
            for i in np.unique(pairs[0]):
                reasons[int(i)] = "connector_intersects_excluded_facility"
        if water is not None and not water.is_empty:
            parts = shapely.get_parts(water)
            pairs = STRtree(parts).query(connectors)
            crossed = np.zeros(len(frame), dtype=float)
            if pairs.size:
                lengths = shapely.length(
                    shapely.intersection(connectors[pairs[0]], parts[pairs[1]])
                )
                np.add.at(crossed, pairs[0], lengths)
            hits = crossed > 0.01
            for i in np.flatnonzero(hits):
                if not reasons[i]:
                    reasons[i] = "connector_crosses_mapped_water"
        barrier_mask = self.physical.get(
            "connector_crossing_barrier", ~self.physical.anchor_class_eligible
        )
        barriers = self.physical.loc[barrier_mask]
        if len(barriers):
            pairs = STRtree(barriers.geometry.to_numpy()).query(connectors, predicate="crosses")
            for i in np.unique(pairs[0]):
                if not reasons[int(i)]:
                    reasons[int(i)] = "connector_crosses_nonhosting_motor_road"
        if footprints is not None and len(footprints):
            buildings = footprints.to_crs(self.area_crs).geometry.to_numpy()
            pairs = STRtree(buildings).query(connectors, predicate="crosses")
            for i, j in zip(pairs[0], pairs[1]):
                if not reasons[int(i)] and not shapely.intersects(
                    source[int(i)], buildings[int(j)]
                ):
                    reasons[int(i)] = "connector_crosses_other_mapped_building"
        frame["access_review_reason"] = reasons
        frame["local_access_confirmed"] = pd.Series(reasons).eq("")
        frame["access_status"] = np.where(
            frame.local_access_confirmed,
            np.where(
                frame.requires_terminal_paths,
                "mapped_terminal_roundtrip",
                "public_road_geometric_proxy",
            ),
            "review_required",
        )
        frame["road_access_policy"] = getattr(self, "policy", POLICY)
        frame["protected_inbound_access_eligible"] = (
            np.isfinite(in_lengths) & frame.local_access_confirmed
        )
        frame["protected_outbound_access_eligible"] = (
            np.isfinite(out_lengths) & frame.local_access_confirmed
        )
        frame["protected_roundtrip_eligible"] = frame.local_access_confirmed
        direct = frame.local_access_confirmed & ~frame.requires_terminal_paths
        frame["anchor_scc_id"] = np.where(
            direct,
            "S0001",
            np.where(frame.local_access_confirmed, "S_TERMINAL_ACCESS", "S_UNRESOLVED"),
        )
        frame["reference_scc_id"] = "S0001"
        frame["reference_scc_node_count"] = len(self.reference)
        frame["reference_scc_node_share"] = len(self.reference) / max(len(self.core), 1)
        frame["anchor_in_reference_scc"] = direct
        frame["access_gateway_in_reference_scc"] = frame.local_access_confirmed
        frame["protected_roundtrip_status"] = np.where(
            frame.local_access_confirmed,
            "passed_local_road_and_terminal_directional_access",
            "review_required_local_access",
        )
        # A source-safe location stays in the table, but a sensitive/prohibited
        # facility anchor is not exposed as a new public location.
        redact = pd.Series(reasons).isin(
            ["local_road_within_excluded_facility", "connector_intersects_excluded_facility"]
        )
        frame.loc[redact, ["road_anchor_lon", "road_anchor_lat"]] = np.nan
        frame.loc[redact, "directed_projection_offsets"] = "[]"
        frame.loc[redact, "directed_edge_refs"] = "[]"
        frame.loc[redact, "terminal_inbound_path_json"] = "[]"
        frame.loc[redact, "terminal_outbound_path_json"] = "[]"
        frame.loc[redact, "road_projection_node_id"] = "unresolved_" + frame.loc[
            redact, "latent_service_location_id"
        ].astype(str)
        return frame
