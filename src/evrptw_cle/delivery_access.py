"""National delivery-purpose terminal admission and local frontage alternatives.

The authorization of a selected delivery is a benchmark assumption. It never
turns a private/destination-only edge into a public through-routing edge.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely

from .runtime_paths import CONFIG_ROOT
from .local_road_access import BUFFER_LADDER_KM, LocalAccessContext, _forest
from .public_locations import excluded_location_reason
from .terminal_access import _effective_motor_access
from .util import sha256_file, write_json

POLICY = "delivery_terminal_access_v2"

HOLDS_PATH = CONFIG_ROOT / "us_location_access_review_holds_v1.json"
PHYSICAL_BLOCKS = {
    "block",
    "bollard",
    "cycle_barrier",
    "jersey_barrier",
    "kissing_gate",
    "stile",
    "turnstile",
    "wall",
    "fence",
    "chain",
}


def flag(value):
    return value is True or str(value).lower() in {"true", "1"}


def permission_inventory(physical, source_dir, masks):
    """Apply the accepted authorized-delivery policy to real source segments."""
    roads = physical.copy()
    nodes = pd.read_parquet(
        Path(source_dir) / "osm_access_nodes.parquet", columns=["osmid", "tags_json"]
    )
    private_nodes, hard_nodes = set(), set()
    for row in nodes.itertuples():
        tags = json.loads(row.tags_json)
        _, legal = _effective_motor_access(tags)
        hard = (
            tags.get("access:delivery") in {"no", "prohibited", "none"}
            or tags.get("locked") == "yes"
            or tags.get("barrier") in PHYSICAL_BLOCKS
        )
        if legal == "permit_required" and (
            excluded_location_reason(tags) or tags.get("private") == "military"
        ):
            hard = True
        if legal == "permit_required" and not hard:
            private_nodes.add(str(row.osmid))
        if legal in {"prohibited", "unresolved_tag_value"} or hard:
            hard_nodes.add(str(row.osmid))
    ways = pd.read_parquet(Path(source_dir) / "osm_ways.parquet", columns=["osmid", "tags_json"])
    hard_ways = set()
    for row in ways.itertuples():
        tags = json.loads(row.tags_json)
        _, legal = _effective_motor_access(tags)
        if (
            tags.get("access:delivery") in {"no", "prohibited", "none"}
            or tags.get("service") == "emergency_access"
        ):
            hard_ways.add(str(row.osmid))
        if legal == "permit_required" and (
            excluded_location_reason(tags) or tags.get("private") == "military"
        ):
            hard_ways.add(str(row.osmid))
    private_way = roads.legal_access_tier.eq("permit_required") & roads.anchor_class_eligible
    sensitive = np.zeros(len(roads), dtype=bool)
    ids = np.flatnonzero(private_way)
    if len(ids) and len(masks):
        pairs = shapely.STRtree(masks.to_crs(roads.crs).geometry.to_numpy()).query(
            roads.geometry.to_numpy()[ids], predicate="intersects"
        )
        if pairs.size:
            sensitive[ids[np.unique(pairs[0])]] = True
    unsupported = (
        roads.osmid.isin(hard_ways).to_numpy()
        | roads.sensitive_interior_road.to_numpy()
        | sensitive
    )
    private_endpoint = (
        roads.edge_u.isin(private_nodes) | roads.edge_v.isin(private_nodes)
    ).to_numpy()
    hard_endpoint = (roads.edge_u.isin(hard_nodes) | roads.edge_v.isin(hard_nodes)).to_numpy()
    strict_node = ~roads.node_access_blocked.to_numpy()
    flexible_node = (strict_node | private_endpoint) & ~hard_endpoint
    source_allowed = roads.source_way_motor_allowed.to_numpy()
    roads["baseline_motor_access_allowed"] = roads.motor_access_allowed
    roads["strict_motor_access_allowed"] = (
        source_allowed & strict_node & ~unsupported & ~hard_endpoint
    )
    roads["motor_access_allowed"] = (
        (source_allowed | private_way.to_numpy()) & flexible_node & ~unsupported
    )
    roads["delivery_authorization_assumed"] = private_way.to_numpy() | private_endpoint
    roads["node_access_blocked"] = ~flexible_node
    roads["sensitive_interior_road"] = roads.sensitive_interior_road | sensitive
    # Purpose restrictions permit delivery access but never public transit.
    restricted_purpose = roads.legal_access_tier.eq("destination_or_customers")
    terminal = (
        roads.access_layer.eq("osm_terminal_only")
        | roads.delivery_authorization_assumed
        | restricted_purpose
    )
    roads.loc[terminal, "access_layer"] = "osm_terminal_only"
    roads["connector_crossing_barrier"] = ~roads.anchor_class_eligible | ~roads.motor_access_allowed
    return roads, dict(
        private_node_count=len(private_nodes),
        hard_node_count=len(hard_nodes),
        source_way_hard_count=len(hard_ways),
        private_sensitive_segment_count=int(sensitive.sum()),
    )


def build_delivery_access_context(
    *,
    seed_dir,
    source_dir,
    admin_boundary_path,
    service_boundary_path,
    mask_path,
    output_dir,
    area_crs,
):
    seed_dir, source_dir, output_dir = Path(seed_dir), Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    old = json.loads((seed_dir / "access_context_manifest.json").read_text())
    extraction = json.loads((source_dir / "extraction.json").read_text())
    if (
        extraction != old["source"]
        or extraction["pbf_replication_timestamp_utc"] != "2026-10-25T21:21:21Z"
    ):
        raise ValueError("Source road skeleton and requested snapshot differ")
    seed_path = seed_dir / "physical_roads.parquet"
    if sha256_file(seed_path) != old["outputs"]["physical_roads.parquet"]:
        raise ValueError("Changed source road skeleton")
    masks = gpd.read_parquet(mask_path)
    physical, permissions = permission_inventory(gpd.read_parquet(seed_path), source_dir, masks)
    physical.to_parquet(output_dir / "physical_roads.parquet", index=False)
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    wgs = physical.to_crs(4326)
    node_metric = {}
    for row, metric in zip(wgs.itertuples(), physical.geometry):
        for ref in json.loads(row.directed_edge_refs):
            reversed_ = ref["geometry_orientation"] == "reverse_of_physical"
            geom = shapely.reverse(row.geometry) if reversed_ else row.geometry
            mg = shapely.reverse(metric) if reversed_ else metric
            node_metric[ref["u"]], node_metric[ref["v"]] = mg.coords[0], mg.coords[-1]
            if not row.motor_access_allowed:
                continue
            for n, xy in [(ref["u"], geom.coords[0]), (ref["v"], geom.coords[-1])]:
                if n not in graph:
                    graph.add_node(n, x=float(xy[0]), y=float(xy[1]))
            graph.add_edge(
                ref["u"],
                ref["v"],
                key=ref["key"],
                length=float(ref["length_m"]),
                geometry=geom,
                osmid=str(row.osmid),
                highway=row.highway,
                name=row.road_name,
                oneway=bool(row.oneway),
                reversed=reversed_,
                access_layer=row.access_layer,
                legal_access_tier=row.legal_access_tier,
                through_routing_allowed=row.access_layer == "operational_public",
                delivery_authorization_assumed=bool(row.delivery_authorization_assumed),
                strict_motor_access_allowed=bool(row.strict_motor_access_allowed),
                maxspeed=row.maxspeed,
                **{
                    "maxspeed:forward": row.maxspeed_forward,
                    "maxspeed:backward": row.maxspeed_backward,
                },
            )
    del wgs
    admin = gpd.read_file(admin_boundary_path).to_crs(area_crs).geometry.union_all()
    public = nx.subgraph_view(
        graph, filter_edge=lambda u, v, k: graph[u][v][k]["through_routing_allowed"]
    )
    public = public.subgraph([n for n in public if public.degree(n) > 0])
    ns = list(public)
    outside = dict(zip(ns, shapely.distance(shapely.points([node_metric[n] for n in ns]), admin)))
    citynodes = {n for n, d in outside.items() if d <= 1e-6}
    trials, choices = [], []
    for km in BUFFER_LADDER_KM:
        candidate = public.subgraph([n for n in ns if outside[n] <= 1000 * km + 1e-6])
        component = max(
            nx.weakly_connected_components(candidate),
            key=lambda c: (len(c & citynodes), len(c), min(c)),
        )
        covered = component & citynodes
        choices.append((set(component), set(covered)))
        trials.append(
            dict(
                buffer_km=km,
                city_node_count=len(citynodes),
                city_nodes_covered=len(covered),
                city_node_coverage=len(covered) / max(len(citynodes), 1),
            )
        )
    maximum = choices[-1][1]
    pick = next(i for i, (_, covered) in enumerate(choices) if maximum <= covered)
    core = public.subgraph(choices[pick][0]).copy()
    for n in core:
        core.nodes[n]["inside_city"] = n in citynodes
        core.nodes[n]["transit_only"] = n not in citynodes
    for u, v, k, d in core.edges(keys=True, data=True):
        d["transit_only"] = u not in citynodes or v not in citynodes
    reference = max(nx.strongly_connected_components(core), key=lambda c: (len(c), min(c)))
    access = nx.MultiDiGraph(crs="EPSG:4326")
    for u, v, k, d in graph.edges(keys=True, data=True):
        if u in reference and v in reference and d["through_routing_allowed"]:
            continue
        for n in [u, v]:
            if n not in access:
                access.add_node(n, **graph.nodes[n])
        access.add_edge(u, v, key=k, **d)
    gateways = set(access) & set(reference)
    inbound, outbound = _forest(access, gateways), _forest(access, gateways, reverse=True)
    water = admin.difference(
        gpd.read_file(service_boundary_path).to_crs(area_crs).geometry.union_all()
    )
    context = DeliveryAccessContext(
        core, access, physical, reference, inbound, outbound, area_crs, water
    )
    context.save(output_dir)
    ox.save_graphml(public.subgraph(citynodes).copy(), output_dir / "graph_all.graphml")
    names = [
        "graph_all.graphml",
        "graph_operational.graphml",
        "graph_terminal_access.graphml",
        "physical_roads.parquet",
        "access_forests.json",
        "connector_water.parquet",
    ]
    manifest = dict(
        schema="evrptw_delivery_terminal_access_v2",
        policy=POLICY,
        city_slug=extraction["city_slug"],
        source=extraction,
        area_crs=area_crs,
        admin_boundary_sha256=sha256_file(Path(admin_boundary_path)),
        service_boundary_sha256=sha256_file(Path(service_boundary_path)),
        mask_sha256=sha256_file(Path(mask_path)),
        physical_seed_sha256=sha256_file(seed_path),
        permissions=permissions,
        selected_buffer_km=BUFFER_LADDER_KM[pick],
        selection="smallest buffer recovering all city public nodes reachable at maximum envelope",
        buffer_trials=trials,
        core_nodes=len(core),
        core_directed_edges=core.number_of_edges(),
        reference_scc_nodes=len(reference),
        terminal_graph_nodes=len(access),
        terminal_graph_directed_edges=access.number_of_edges(),
        local_physical_segments=len(physical),
        terminal_through_routing_allowed=False,
        owner_permission_observed=False,
        review_holds_sha256=sha256_file(HOLDS_PATH),
        generation_dependencies={
            "delivery_access.py": sha256_file(Path(__file__)),
            "local_road_access.py": sha256_file(Path(__file__).with_name("local_road_access.py")),
        },
        outputs={n: sha256_file(output_dir / n) for n in names},
    )
    write_json(output_dir / "access_context_manifest.json", manifest)
    del node_metric
    gc.collect()
    return context, manifest


def connector_clearance(source, target, *, masks, water, barriers, buildings):
    lines = shapely.shortest_line(source, target)
    good = np.ones(len(source), dtype=bool)
    if masks is not None:
        pairs = masks.query(lines, predicate="intersects")
        if pairs.size:
            good[np.unique(pairs[0])] = False
    if water is not None and not water.is_empty:
        parts = shapely.get_parts(water)
        pairs = shapely.STRtree(parts).query(lines)
        length = np.zeros(len(lines))
        if pairs.size:
            np.add.at(
                length,
                pairs[0],
                shapely.length(shapely.intersection(lines[pairs[0]], parts[pairs[1]])),
            )
        good[length > 0.01] = False
    pairs = barriers.query(lines, predicate="crosses")
    if pairs.size:
        good[np.unique(pairs[0])] = False
    if buildings is not None:
        pairs = buildings.query(lines, predicate="crosses")
        if pairs.size:
            i, j = pairs
            bad = ~shapely.intersects(source[i], buildings.geometries[j])
            good[np.unique(i[bad])] = False
    return good


class DeliveryAccessContext(LocalAccessContext):
    policy = POLICY

    def anchor(self, locations, *, masks=None, footprints=None, water=None, selected_edge_ids=None):
        result = super().anchor(
            locations,
            masks=masks,
            footprints=footprints,
            water=water,
            selected_edge_ids=selected_edge_ids,
        )
        result["local_alternative_selected"] = False
        result["initial_access_review_reason"] = result.access_review_reason
        # Customer footprints define a local frontage scale. Depot points have
        # no source footprint here, so no arbitrary fixed search radius is used.
        if selected_edge_ids is None:
            select = np.flatnonzero(
                result.access_review_reason.isin(
                    [
                        "connector_crosses_other_mapped_building",
                        "local_road_no_verified_directed_roundtrip",
                    ]
                )
            )
            if len(select):
                replacements = self._alternatives(
                    locations, result, select, masks, footprints, water
                )
                if replacements:
                    ids = np.array(sorted(replacements))
                    edge_ids = [replacements[i] for i in ids]
                    strict = self._strict_context()
                    repl = LocalAccessContext.anchor(
                        strict,
                        locations.iloc[ids].reset_index(drop=True),
                        masks=masks,
                        footprints=footprints,
                        water=water,
                        selected_edge_ids=edge_ids,
                    )
                    accepted = repl.local_access_confirmed.to_numpy()
                    take = ids[accepted]
                    initial = result.loc[take, "initial_access_review_reason"].to_numpy()
                    for col in repl.columns:
                        result.loc[take, col] = repl.loc[accepted, col].to_numpy()
                    result.loc[take, "local_alternative_selected"] = True
                    result.loc[take, "initial_access_review_reason"] = initial
        info = self.physical.set_index("physical_edge_id")
        assumed = []
        for r in result.itertuples():
            value = bool(info.at[r.physical_edge_id, "delivery_authorization_assumed"])
            if r.local_access_confirmed and r.requires_terminal_paths:
                value = value or any(
                    flag(p.get("delivery_authorization_assumed", False))
                    for p in json.loads(r.terminal_inbound_path_json)
                    + json.loads(r.terminal_outbound_path_json)
                )
            assumed.append(value)
        result["delivery_authorization_assumed"] = assumed
        result["owner_permission_observed"] = False
        result["access_permission_basis"] = np.where(
            result.delivery_authorization_assumed,
            "modeled_invited_delivery",
            "source_tags_and_geometric_proxy",
        )
        authorized = result.local_access_confirmed & result.delivery_authorization_assumed
        result.loc[authorized, "access_status"] = "modeled_authorized_terminal_roundtrip"
        holds = json.loads(HOLDS_PATH.read_text())["records"]
        hold_ids = {h["latent_service_location_id"] for h in holds}
        held = result.latent_service_location_id.isin(hold_ids)
        result["access_case_review_hold"] = held
        result.loc[held, "access_review_reason"] = "review_hold_unverified_motor_access"
        result.loc[held, "access_status"] = "review_required"
        for c in [
            "local_access_confirmed",
            "protected_inbound_access_eligible",
            "protected_outbound_access_eligible",
            "protected_roundtrip_eligible",
            "anchor_in_reference_scc",
            "access_gateway_in_reference_scc",
        ]:
            result.loc[held, c] = False
        result.loc[held, "anchor_scc_id"] = "S_UNRESOLVED"
        result.loc[held, "protected_roundtrip_status"] = "review_required_local_access"
        result["road_access_policy"] = POLICY
        return result

    def _strict_context(self):
        if hasattr(self, "_strict_cache"):
            return self._strict_cache
        graph = nx.subgraph_view(
            self.access,
            filter_edge=lambda u, v, k: flag(
                self.access[u][v][k].get("strict_motor_access_allowed", False)
            ),
        )
        gateways = set(graph) & self.reference
        physical = self.physical.copy()
        physical["motor_access_allowed"] = physical.strict_motor_access_allowed
        physical["connector_crossing_barrier"] = (
            ~physical.anchor_class_eligible | ~physical.motor_access_allowed
        )
        ctx = LocalAccessContext(
            self.core,
            graph,
            physical,
            self.reference,
            _forest(graph, gateways),
            _forest(graph, gateways, reverse=True),
            self.area_crs,
            self.water,
        )
        ctx.policy = POLICY
        self._strict_cache = ctx
        return ctx

    def _alternatives(self, locations, frame, selected, masks, footprints, water):
        strict = self._strict_context()
        physical = self.physical
        alt = physical.loc[
            physical.anchor_class_eligible & physical.strict_motor_access_allowed
        ].reset_index(drop=True)
        tree = shapely.STRtree(alt.geometry.to_numpy())
        source = locations.to_crs(self.area_crs).geometry.to_numpy()
        extent = source
        if "access_search_extent_wkt" in locations:
            extent = (
                gpd.GeoSeries(
                    shapely.from_wkt(locations.access_search_extent_wkt.to_numpy()),
                    crs=locations.crs,
                )
                .to_crs(self.area_crs)
                .to_numpy()
            )
        bounds = shapely.bounds(extent)
        diagonal = np.hypot(bounds[:, 2] - bounds[:, 0], bounds[:, 3] - bounds[:, 1])
        gap = frame.road_access_distance_m.to_numpy()
        radius = np.where(diagonal > 0, gap + 2 * diagonal, 2 * gap)
        mt = (
            shapely.STRtree(masks.to_crs(self.area_crs).geometry.to_numpy())
            if masks is not None and len(masks)
            else None
        )
        bt = (
            None
            if footprints is None
            else shapely.STRtree(footprints.to_crs(self.area_crs).geometry.to_numpy())
        )
        barriers = shapely.STRtree(
            physical.loc[
                ~physical.anchor_class_eligible | ~physical.strict_motor_access_allowed
            ].geometry.to_numpy()
        )
        inbound = set(strict.inbound["distance"]) | strict.reference
        outbound = set(strict.outbound["distance"]) | strict.reference
        winners = {}
        for chunk in np.array_split(selected, max(1, len(selected) // 2000 + 1)):
            pairs = tree.query(source[chunk], predicate="dwithin", distance=radius[chunk])
            if not pairs.size:
                continue
            si, ri = chunk[pairs[0]], pairs[1]
            gap = shapely.distance(source[si], alt.geometry.to_numpy()[ri])
            choices = (
                pd.DataFrame(
                    dict(
                        si=si,
                        ri=ri,
                        gap=gap,
                        way=alt.osmid.to_numpy()[ri],
                        pid=alt.physical_edge_id.to_numpy()[ri],
                    )
                )
                .sort_values(["si", "gap", "way", "pid"])
                .drop_duplicates(["si", "way"])
                .groupby("si", sort=False)
                .head(8)
            )
            si, ri = choices.si.to_numpy(), choices.ri.to_numpy()
            r = alt.iloc[ri]
            directed = (r.edge_u.isin(inbound) & r.edge_v.isin(outbound)).to_numpy() | (
                (~r.oneway) & r.edge_v.isin(inbound) & r.edge_u.isin(outbound)
            ).to_numpy()
            good = (
                connector_clearance(
                    source[si],
                    r.geometry.to_numpy(),
                    masks=mt,
                    water=self.water if water is None else water,
                    barriers=barriers,
                    buildings=bt,
                )
                & directed
            )
            for r in choices.loc[good].drop_duplicates("si").itertuples():
                winners[int(r.si)] = str(r.pid)
        return winners
