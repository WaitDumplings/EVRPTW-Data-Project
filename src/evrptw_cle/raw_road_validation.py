"""Independent PBF read-back for complete-way access-node extraction."""

from datetime import UTC, datetime
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .util import sha256_file, write_json


def verify_raw_road_nodes(road_inputs, output_path):
    import osmium

    base=Path(road_inputs)
    groups={}
    for path in sorted(base.glob("*/extraction.json")):
        m=json.loads(path.read_text())
        groups.setdefault(m["pbf_sha256"],[]).append((path.parent,m))
    if not groups:
        raise ValueError("No frozen road extraction inventory")
    results={}
    for digest,group in groups.items():
        pbf=Path(group[0][1]["pbf_path"])
        if sha256_file(pbf)!=digest:
            raise ValueError("Frozen PBF changed before independent node verification")
        cities={}
        for folder,m in group:
            for name,value in m["outputs"].items():
                if sha256_file(folder / name)!=value:
                    raise ValueError("Extracted source bytes changed")
            ways=pd.read_parquet(folder / "osm_ways.parquet",columns=["node_ids_json"])
            used={int(node) for text in ways.node_ids_json for node in json.loads(text)}
            nodes=gpd.read_parquet(folder / "osm_access_nodes.parquet").to_crs(4326)
            if nodes.osmid.duplicated().any():
                raise ValueError("Duplicate extracted restriction node")
            stored={int(row.osmid):(json.loads(row.tags_json),row.geometry.x,row.geometry.y) for row in nodes.itertuples()}
            cities[m["city_slug"]]=dict(used=used,stored=stored,checked=0,source=m)
        all_used=set().union(*(record["used"] for record in cities.values()))
        # Literal accepted-model keys, independently of the extractor constant.
        processor=(osmium.FileProcessor(str(pbf))
                   .with_filter(osmium.filter.EntityFilter(osmium.osm.NODE))
                   .with_filter(osmium.filter.KeyFilter("access","vehicle","motor_vehicle","motorcar","access:delivery","barrier","locked")))
        for node in processor:
            if node.id not in all_used:
                continue
            tags=dict(node.tags)
            for city,record in cities.items():
                if node.id not in record["used"]:
                    continue
                actual=record["stored"].get(node.id)
                if actual is None:
                    raise ValueError(f"Frozen referenced restriction node missing: {city}:{node.id}")
                if tags!=actual[0] or abs(node.location.lon-actual[1])>1e-10 or abs(node.location.lat-actual[2])>1e-10:
                    raise ValueError(f"Frozen restriction-node evidence differs: {city}:{node.id}")
                record["checked"]+=1
        for city,record in cities.items():
            expected=len(record["used"] & record["stored"].keys())
            if record["checked"]!=expected:
                raise ValueError("Extracted restriction node has no corresponding PBF node: "+city)
            results[city]=dict(checked_referenced_access_nodes=record["checked"],
                               pbf_sha256=digest,source_outputs=record["source"]["outputs"])
            print("PBF NODE EVIDENCE VERIFIED",city,record["checked"],flush=True)
    result=dict(schema="evrptw_independent_raw_road_node_verification_v1",passed=True,
                generated_utc=datetime.now(UTC).isoformat(),cities=results,
                verifier_sha256=sha256_file(Path(__file__)),source_data_modified=False)
    write_json(Path(output_path),result)
    return result
