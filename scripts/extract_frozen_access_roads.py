#!/usr/bin/env python3
"""Freeze complete OSM ways for uniform city + 20km road-access reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import osmium
import shapely
from shapely.geometry import LineString, Point

from evrptw_cle.util import sha256_file, write_json


def extract_region(*, cohort_path, generator_root, output_root, region):
    cohort = json.loads(Path(cohort_path).read_text())
    cfg = cohort["regions"][region]
    pbf = Path(cfg["path"])
    expected = json.loads(Path(cfg["metadata"]).read_text())
    digest = sha256_file(pbf)
    if digest != expected["sha256"]:
        raise ValueError("Frozen OSM PBF hash differs: " + region)
    with osmium.io.Reader(str(pbf)) as reader:
        timestamp = reader.header().get("osmosis_replication_timestamp")
    if timestamp != "2026-10-25T21:21:21Z":
        raise ValueError("Roads must use the accepted October 25 snapshot")
    cities = {}
    boundary_registry = json.loads(
        (Path(generator_root) / "configs/city_boundaries.json").read_text()
    )["cities"]
    for city in cfg["cities"]:
        dest = Path(output_root) / city
        boundary_record = boundary_registry[city]["admin_boundary"]
        boundary_path = Path(generator_root) / boundary_record["path"]
        if sha256_file(boundary_path) != boundary_record["sha256"]:
            raise ValueError("Frozen administrative boundary changed: " + city)
        if (dest / "extraction.json").exists():
            cached = json.loads((dest / "extraction.json").read_text())
            if (
                cached["pbf_sha256"] != digest
                or cached["admin_boundary_sha256"] != boundary_record["sha256"]
                or cached["pbf_replication_timestamp_utc"] != timestamp
            ):
                raise FileExistsError("Cached extraction has different frozen inputs: " + city)
            for name, recorded in cached["outputs"].items():
                if sha256_file(dest / name) != recorded:
                    raise ValueError("Changed cached extraction output: " + name)
            print("VERIFIED REUSE", city, flush=True)
            continue
        boundary = gpd.read_file(boundary_path)
        local_crs = boundary.estimate_utm_crs()
        # Exact complete geometry; no small boundary component is dropped.
        envelope = boundary.to_crs(local_crs).geometry.union_all().buffer(20000)
        envelope = gpd.GeoSeries([envelope], crs=local_crs).to_crs(4326).iloc[0]
        shapely.prepare(envelope)
        cities[city] = dict(
            bounds=envelope.bounds,
            envelope=envelope,
            rows=[],
            nodes=[],
            boundary_path=boundary_path,
        )
    if not cities:
        return
    print("READ", region, list(cities), flush=True)
    processor = (
        osmium.FileProcessor(str(pbf))
        .with_locations()
        .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY | osmium.osm.NODE))
        .with_filter(
            osmium.filter.KeyFilter(
                "highway", "access", "vehicle", "motor_vehicle", "motorcar", "barrier"
            )
        )
    )
    for obj in processor:
        tags = dict(obj.tags)
        if obj.is_node():
            if not obj.location.valid() or not any(
                k in tags for k in ["access", "vehicle", "motor_vehicle", "motorcar", "barrier"]
            ):
                continue
            point = Point(obj.location.lon, obj.location.lat)
            for values in cities.values():
                if values["envelope"].covers(point):
                    values["nodes"].append(
                        dict(
                            osmid=str(obj.id),
                            tags_json=json.dumps(tags, sort_keys=True),
                            geometry=point,
                        )
                    )
            continue
        if "highway" not in tags:
            continue
        coords = [(n.lon, n.lat) for n in obj.nodes if n.location.valid()]
        if len(coords) != len(obj.nodes) or len(coords) < 2:
            continue
        xs, ys = zip(*coords)
        bounds = min(xs), min(ys), max(xs), max(ys)
        near = []
        for city, values in cities.items():
            x0, y0, x1, y1 = values["bounds"]
            if bounds[2] >= x0 and bounds[0] <= x1 and bounds[3] >= y0 and bounds[1] <= y1:
                near.append(city)
        if not near:
            continue
        geometry = LineString(coords)
        record = dict(
            osmid=str(obj.id),
            node_ids_json=json.dumps([n.ref for n in obj.nodes]),
            tags_json=json.dumps(tags, sort_keys=True),
            osm_object_version=obj.version,
            osm_edit_timestamp=obj.timestamp.isoformat(),
            geometry=geometry,
        )
        for city in near:
            if cities[city]["envelope"].intersects(geometry):
                cities[city]["rows"].append(record)
    for city, values in cities.items():
        dest = Path(output_root) / city
        dest.mkdir(parents=True, exist_ok=True)
        ways = gpd.GeoDataFrame(values["rows"], geometry="geometry", crs=4326)
        ways.to_parquet(dest / "osm_ways.parquet", index=False)
        nodes = gpd.GeoDataFrame(
            values["nodes"],
            columns=["osmid", "tags_json", "geometry"],
            geometry="geometry",
            crs=4326,
        )
        nodes.to_parquet(dest / "osm_access_nodes.parquet", index=False)
        write_json(
            dest / "extraction.json",
            dict(
                schema="evrptw_complete_osm_access_ways_v1",
                city_slug=city,
                region=region,
                pbf_sha256=digest,
                pbf_replication_timestamp_utc=timestamp,
                pbf_source_url=cfg["frozen_url"],
                pbf_path=str(pbf),
                admin_boundary_sha256=sha256_file(values["boundary_path"]),
                max_routing_buffer_km=20,
                way_count=len(ways),
                restricted_or_barrier_node_count=len(nodes),
                outputs={
                    name: sha256_file(dest / name)
                    for name in ["osm_ways.parquet", "osm_access_nodes.parquet"]
                },
            ),
        )
        print("COMPLETE", city, len(ways), len(nodes), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--generator-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--regions", nargs="+", required=True)
    args = p.parse_args()
    for region in args.regions:
        extract_region(
            cohort_path=args.cohort,
            generator_root=args.generator_root,
            output_root=args.output_root,
            region=region,
        )
