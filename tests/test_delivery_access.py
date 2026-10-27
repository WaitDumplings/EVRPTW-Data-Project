import json

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon, box
from test_local_road_access import context, locations

from evrptw_cle.delivery_access import DeliveryAccessContext, permission_inventory
from evrptw_cle.local_road_access import _forest
from evrptw_cle.util import sha256_file


def authorized_context():
    c = context(permit=True)
    c.physical["strict_motor_access_allowed"] = False
    c.physical["motor_access_allowed"] = True
    c.physical["delivery_authorization_assumed"] = True
    c.physical["connector_crossing_barrier"] = False
    for u, v, k, d in c.access.edges(keys=True, data=True):
        d["delivery_authorization_assumed"] = True
        d["strict_motor_access_allowed"] = False
    return DeliveryAccessContext(
        c.core, c.access, c.physical, c.reference, c.inbound, c.outbound, c.area_crs
    )


def test_private_is_admitted_only_as_flagged_terminal_with_real_paths():
    c = authorized_context()
    r = c.anchor(locations()).iloc[0]
    assert r.local_access_confirmed and r.delivery_authorization_assumed
    assert not r.owner_permission_observed
    assert r.access_permission_basis == "modeled_invited_delivery"
    assert r.requires_terminal_paths and np.isclose(r.terminal_outbound_length_m, 45)
    assert not c.core.has_edge("a", "c")
    assert json.loads(r.terminal_outbound_path_json)[0]["delivery_authorization_assumed"]


def test_private_gate_is_distinct_from_motor_no_and_locked(tmp_path):
    c = context(permit=True)
    p = c.physical.copy()
    p["osmid"] = "2"
    masks = gpd.GeoDataFrame(geometry=[], crs=3857)
    pd.DataFrame(
        [dict(osmid="a", tags_json=json.dumps({"access": "private", "barrier": "gate"}))]
    ).to_parquet(tmp_path / "osm_access_nodes.parquet", index=False)
    pd.DataFrame(
        [dict(osmid="2", tags_json=json.dumps({"highway": "service", "access": "private"}))]
    ).to_parquet(tmp_path / "osm_ways.parquet", index=False)
    p["node_access_blocked"] = True
    result, _ = permission_inventory(p, tmp_path, masks)
    assert result.motor_access_allowed.iloc[0]
    assert not result.strict_motor_access_allowed.iloc[0]
    for tags in [{"motor_vehicle": "no"}, {"access": "private", "locked": "yes"}]:
        pd.DataFrame([dict(osmid="a", tags_json=json.dumps(tags))]).to_parquet(
            tmp_path / "osm_access_nodes.parquet", index=False
        )
        result, _ = permission_inventory(p, tmp_path, masks)
        assert not result.motor_access_allowed.iloc[0]


def test_case_review_hold_keeps_source_but_never_admits_la_point():
    c = authorized_context()
    x = locations()
    x["latent_service_location_id"] = "cus_site_los-angeles_e11a978cb54f39b269df"
    r = c.anchor(x).iloc[0]
    assert r.access_case_review_hold and not r.local_access_confirmed
    assert r.access_review_reason == "review_hold_unverified_motor_access"
    assert r.geometry.equals(x.geometry.iloc[0])


def test_saved_context_preserves_private_authorization_and_holds(tmp_path):
    from evrptw_cle.delivery_access import HOLDS_PATH, POLICY
    from evrptw_cle.local_road_access import LocalAccessContext

    c = authorized_context()
    c.water = Polygon()
    c.physical.to_parquet(tmp_path / "physical_roads.parquet", index=False)
    c.save(tmp_path)
    manifest = dict(
        policy=POLICY,
        area_crs="EPSG:3857",
        review_holds_sha256=sha256_file(HOLDS_PATH),
        outputs={p.name: sha256_file(p) for p in tmp_path.iterdir() if p.is_file()},
    )
    (tmp_path / "access_context_manifest.json").write_text(json.dumps(manifest))
    loaded = LocalAccessContext.load(tmp_path)
    r = loaded.anchor(locations()).iloc[0]
    assert isinstance(loaded, DeliveryAccessContext)
    assert r.local_access_confirmed and r.delivery_authorization_assumed
    assert all(
        x["delivery_authorization_assumed"] for x in json.loads(r.terminal_outbound_path_json)
    )


def test_pair_retains_authorization_without_opening_reverse_transit():
    from evrptw_cle.terminal_pair import materialize_local_access_pair

    c = authorized_context()
    first, second = locations(), locations()
    second["latent_service_location_id"] = "other"
    second.geometry = second.geometry.translate(yoff=20)
    both = gpd.GeoDataFrame(pd.concat([first, second], ignore_index=True), crs=3857)
    rows = c.anchor(both)
    graph, report = materialize_local_access_pair(
        graph=c.core,
        source=rows.iloc[0],
        destination=rows.iloc[1],
        connector_speed_kph=10,
        terminal_speed_kph=10,
    )
    assert nx.has_path(graph, report["source_node"], report["destination_node"])
    assert not nx.has_path(graph, report["destination_node"], report["source_node"])
    refs = [d for _, _, d in graph.edges(data=True) if d.get("real_osm_terminal_segment")]
    assert refs and all(d["delivery_authorization_assumed"] for d in refs)
    assert all(not d["owner_permission_observed"] for d in refs)


def test_local_frontage_alternative_avoids_neighbor_without_relaxing_permission():
    c = authorized_context()
    p = c.physical.iloc[[0, 0]].copy().reset_index(drop=True)
    graph = nx.MultiDiGraph(crs="EPSG:3857")
    for name, xy in {"a": (0, 0), "b": (100, 0), "c": (0, 100), "d": (100, 100)}.items():
        graph.add_node(name, x=xy[0], y=xy[1])
    for i, (u, v, y) in enumerate([("a", "b", 0), ("c", "d", 100)]):
        geom = LineString([(0, y), (100, y)])
        refs = []
        for a, b, key, line in [
            (u, v, str(i) + "f", geom),
            (v, u, str(i) + "r", LineString(list(geom.coords)[::-1])),
        ]:
            graph.add_edge(
                a, b, key=key, length=100, geometry=line, osmid=str(i), highway="residential"
            )
            refs.append(
                dict(
                    u=a,
                    v=b,
                    key=key,
                    length_m=100,
                    osmid=[str(i)],
                    highway=["residential"],
                    geometry_orientation="same_as_physical"
                    if key.endswith("f")
                    else "reverse_of_physical",
                )
            )
        for col, value in {
            "physical_edge_id": str(i),
            "edge_u": u,
            "edge_v": v,
            "edge_key": str(i) + "f",
            "geometry": geom,
            "directed_edge_refs": json.dumps(refs),
            "strict_motor_access_allowed": True,
            "motor_access_allowed": True,
            "delivery_authorization_assumed": False,
            "access_layer": "operational_public",
            "legal_access_tier": "unspecified",
            "osmid": str(i),
        }.items():
            p.at[i, col] = value
    for u, v in [("a", "c"), ("b", "d")]:
        for a, b in [(u, v), (v, u)]:
            graph.add_edge(
                a,
                b,
                key="join",
                length=100,
                geometry=LineString(
                    [
                        (graph.nodes[a]["x"], graph.nodes[a]["y"]),
                        (graph.nodes[b]["x"], graph.nodes[b]["y"]),
                    ]
                ),
                osmid="join",
                highway="residential",
            )
    access = nx.MultiDiGraph(crs="EPSG:3857")
    c = DeliveryAccessContext(
        graph,
        access,
        p,
        set(graph),
        _forest(access, set()),
        _forest(access, set(), reverse=True),
        "EPSG:3857",
    )
    x = gpd.GeoDataFrame(
        {"latent_service_location_id": ["frontage"]}, geometry=[box(45, 35, 55, 45)], crs=3857
    )
    footprints = gpd.GeoDataFrame(geometry=[x.geometry.iloc[0], box(40, 10, 60, 30)], crs=3857)
    r = c.anchor(x, footprints=footprints).iloc[0]
    assert r.local_alternative_selected and r.local_access_confirmed
    assert r.physical_edge_id == "1" and np.isclose(r.road_access_distance_m, 55)
    assert not r.delivery_authorization_assumed
