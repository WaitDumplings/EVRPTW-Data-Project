"""Consume real terminal ledgers for one OD pair without through shortcuts."""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import LineString, Point

from .service_access import materialize_active_service_graph


def materialize_local_access_pair(
    *, graph, source, destination, connector_speed_kph, terminal_speed_kph
):
    """Materialize only source egress and destination ingress.

    Each path segment is attributed to its real directed OSM edge. Terminal
    paths are never inserted as a public through-routing shortcut. Speeds are
    explicit operating assumptions, separate from location admission.
    """
    if connector_speed_kph <= 0 or terminal_speed_kph <= 0:
        raise ValueError("Positive explicit connector and terminal speeds required")
    for row in (source, destination):
        if not bool(row.get("local_access_confirmed", False)):
            raise ValueError("Unresolved local access cannot be activated")
    if source["latent_service_location_id"] == destination["latent_service_location_id"]:
        raise ValueError("A pair must contain two distinct source locations")
    result = graph.copy()
    public = [row for row in (source, destination) if not bool(row["requires_terminal_paths"])]
    if public:
        services = gpd.GeoDataFrame(
            [
                dict(
                    latent_service_location_id=r["latent_service_location_id"],
                    service_access_node_id=r["service_access_node_id"],
                    geometry=Point(r["building_access_lon"], r["building_access_lat"]),
                )
                for r in public
            ],
            crs=4326,
        )
        projections = gpd.GeoDataFrame(
            [
                dict(
                    road_projection_node_id=r["road_projection_node_id"],
                    directed_projection_offsets=r["directed_projection_offsets"],
                    geometry=Point(r["road_anchor_lon"], r["road_anchor_lat"]),
                )
                for r in public
            ],
            crs=4326,
        ).drop_duplicates("road_projection_node_id")
        cols = [
            "latent_service_location_id",
            "service_access_node_id",
            "road_projection_node_id",
            "service_access_connector_id",
            "connector_length_m",
            "protected_roundtrip_eligible",
        ]
        connectors = pd.DataFrame([{c: r[c] for c in cols} for r in public])
        result, _ = materialize_active_service_graph(
            graph=result,
            service_nodes=services,
            projection_nodes=projections,
            connectors=connectors,
            active_service_location_ids=[r["latent_service_location_id"] for r in public],
            connector_speed_kph=connector_speed_kph,
        )
    for direction, row in [("outbound", source), ("inbound", destination)]:
        sid = row["service_access_node_id"]
        if not bool(row["requires_terminal_paths"]):
            forbidden = (
                list(result.in_edges(sid, keys=True))
                if direction == "outbound"
                else list(result.out_edges(sid, keys=True))
            )
            result.remove_edges_from(forbidden)
            continue
        coords = gpd.GeoSeries(
            [
                Point(row["building_access_lon"], row["building_access_lat"]),
                Point(row["road_anchor_lon"], row["road_anchor_lat"]),
            ],
            crs=4326,
        ).to_crs(result.graph["crs"])
        projection = "pair_projection_" + row["latent_service_location_id"]
        result.add_node(
            sid, x=coords.iloc[0].x, y=coords.iloc[0].y, virtual_node_kind="service_access"
        )
        result.add_node(
            projection, x=coords.iloc[1].x, y=coords.iloc[1].y, virtual_node_kind="road_projection"
        )
        a, b = (sid, projection) if direction == "outbound" else (projection, sid)
        length = float(row["connector_length_m"])
        result.add_edge(
            a,
            b,
            key="pair_geometry_connector",
            length=length,
            travel_time_s=length * 3.6 / connector_speed_kph,
            synthetic_access_connector=True,
            geometry=LineString([coords.iloc[0], coords.iloc[1]])
            if direction == "outbound"
            else LineString([coords.iloc[1], coords.iloc[0]]),
        )
        path = json.loads(row["terminal_" + direction + "_path_json"])
        gateway = str(row["terminal_" + direction + "_gateway"])
        if gateway not in result:
            raise ValueError("Terminal gateway is not in the public graph")
        previous = projection if direction == "outbound" else gateway
        for index, ref in enumerate(path):
            endpoint = (
                (gateway if direction == "outbound" else projection)
                if index == len(path) - 1
                else f"pair_{direction}_{row['latent_service_location_id']}_{index}"
            )
            geometry = shapely.from_wkt(ref["geometry_wkt"])
            xy = list(geometry.coords)[-1]
            if endpoint not in result:
                result.add_node(endpoint, x=xy[0], y=xy[1], virtual_node_kind="terminal_path_clone")
            length = float(ref["length_m"])
            result.add_edge(
                previous,
                endpoint,
                key=f"pair_real_osm_{index}",
                length=length,
                travel_time_s=length * 3.6 / terminal_speed_kph,
                geometry=geometry,
                real_osm_terminal_segment=True,
                source_edge_u=ref["u"],
                source_edge_v=ref["v"],
                source_edge_key=ref["key"],
                source_osmid=ref["osmid"],
                terminal_owner=row["latent_service_location_id"],
                delivery_authorization_assumed=str(
                    ref.get("delivery_authorization_assumed", False)
                ).lower()
                in {"true", "1"},
                owner_permission_observed=False,
            )
            previous = endpoint
    return result, dict(
        source_node=source["service_access_node_id"],
        destination_node=destination["service_access_node_id"],
        terminal_through_routing_allowed=False,
    )
