"""Read saved CLE files independently; never repair/promote generator outputs."""

from datetime import UTC, datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .runtime_paths import CONFIG_ROOT
from .util import sha256_file, write_json


def truth(value):
    return value is True or str(value).lower() in {"true", "1", "yes"}


def projected_metres(crs):
    value=CRS.from_user_input(crs)
    return value.is_projected and len(value.axis_info)>=2 and all(abs(axis.unit_conversion_factor-1.0)<1e-12 for axis in value.axis_info[:2])


def same_evidence(original, actual, field):
    """Only the derived optional match key normalizes null to an empty key.

    Occupancy and resunits are native evidence and must retain their values,
    including native missing values. An absent Microsoft match is represented
    by null in the crosswalk and by an empty key in the site membership table.
    """
    if field=="composite_building_id":
        return original.fillna("").eq(actual.fillna("")).all()
    return (original.eq(actual)|(original.isna()&actual.isna())).fillna(False).all()


def check_source_owner_flags(sites, members):
    """Recount saved source-owner competition without the construction ledger."""
    site=sites.set_index("site_id")
    owners=members[["nsi_component_id","site_id"]].drop_duplicates()
    counts=owners.groupby("nsi_component_id").site_id.nunique()
    disputed=set(owners.loc[owners.nsi_component_id.isin(counts.index[counts.gt(1)]),"site_id"])
    if disputed and not site.loc[sorted(disputed),"site_scope_review_required"].all():
        raise ValueError("A source component spans site owners without a pending scope guard")
    roofs=members.loc[members.composite_building_id.fillna("").ne(""),["composite_building_id","site_id"]].drop_duplicates()
    roofs["complex"]=roofs.site_id.map(site.site_source_key).str.startswith("complex:")
    kinds=roofs.groupby("composite_building_id").complex.agg(["any","all"])
    mixed=set(kinds.index[kinds["any"] & ~kinds["all"]])
    affected=set(roofs.loc[roofs.composite_building_id.isin(mixed),"site_id"])
    if affected:
        affected_rows=site.loc[sorted(affected)]
        resolved=affected_rows.site_scope_review_required | affected_rows.site_scope_status.eq("supported_source_building_split")
        if not resolved.all():
            raise ValueError("A shared complex/fallback roof lacks explicit scope disposition")
    return dict(cross_owner_components=int(counts.gt(1).sum()),mixed_scope_roofs=len(mixed),affected_sites=len(disputed|affected))


def check_native_speed_metadata(speeds, native):
    """Costs must use the actual source edge's length and road identity."""
    columns=["edge_u","edge_v","edge_key","length_m","source_length_m","source_osmid","highway"]
    for u,v,key,length,source_length,osmid,highway in speeds[columns].itertuples(index=False,name=None):
        source=native.get((str(u),str(v),str(key)))
        if source is None or not np.isfinite([length,source_length]).all() or abs(float(length)-source["length"])>1e-5 or abs(float(source_length)-source["length"])>1e-5 or str(osmid)!=source["osmid"] or highway!=source["highway"]:
            raise ValueError("Common speed metadata differs from its actual native road edge")
    return True


def graph_edges(path):
    """Stream actual GraphML edges with their native string identities."""
    prefix = "{http://graphml.graphdrawing.org/xmlns}"
    names = {}
    coordinates = {}
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == prefix + "key":
            names[element.attrib["id"]] = element.attrib["attr.name"]
        elif element.tag == prefix + "edge":
            data = {names[x.attrib["key"]]: x.text for x in element.findall(prefix + "data")}
            line = shapely.from_wkt(data["geometry"])
            for node,point in [(element.attrib["source"],line.coords[0]),(element.attrib["target"],line.coords[-1])]:
                if node not in coordinates or np.linalg.norm(np.array(coordinates[node])-np.array(point)) > 1e-7:
                    raise ValueError("Saved graph node coordinates and edge geometry disagree")
            yield (element.attrib["source"], element.attrib["target"], element.attrib["id"]), data
            element.clear()
        elif element.tag == prefix + "node":
            values = {names[x.attrib["key"]]:x.text for x in element.findall(prefix + "data")}
            coordinates[element.attrib["id"]]=(float(values["x"]),float(values["y"]))
            element.clear()


def independent_reference(edges):
    """Compute the saved public graph's SCC using SciPy, not the builder's NX forest."""
    nodes = sorted({n for u, v, _ in edges for n in [u, v]})
    index = {node: i for i, node in enumerate(nodes)}
    row = np.array([index[u] for u, _, _ in edges], dtype=np.int64)
    col = np.array([index[v] for _, v, _ in edges], dtype=np.int64)
    matrix = coo_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(len(nodes), len(nodes))).tocsr()
    _, labels = connected_components(matrix, directed=True, connection="strong")
    counts = np.bincount(labels)
    tied = set(np.flatnonzero(counts == counts.max()))
    choice = next(labels[i] for i in range(len(nodes)) if labels[i] in tied)
    return {node for node, label in zip(nodes, labels) if label == choice}


class AccessRow(dict):
    __getattr__ = dict.__getitem__


def check_path(row, direction, native, graph_keys, reference, speed_index, purpose="delivery", geometry_cache=None, length_transformer=None):
    """Check native graph membership, directions, contiguous chains and partial costs."""
    route = json.loads(row["terminal_" + direction + "_path_json"])
    expected = float(row["terminal_" + direction + "_length_m"])
    if not truth(row.requires_terminal_paths):
        if route or abs(expected) > 1e-7:
            raise ValueError("Public projection unexpectedly carries a terminal route")
        return np.zeros(3), 0
    if not route:
        raise ValueError("Eligible terminal lacks an inbound/outbound chain")
    gateway = str(row["terminal_" + direction + "_gateway"])
    if gateway not in reference or str(route[0]["u"] if direction == "inbound" else route[-1]["v"]) != gateway:
        raise ValueError("Terminal chain does not meet its public SCC gateway")
    if any(str(x["v"]) != str(y["u"]) for x, y in zip(route[:-1], route[1:])):
        raise ValueError("Terminal chain is not topologically contiguous")
    total_length, times = 0.0, np.zeros(3)
    for part in route:
        key = tuple(str(part[field]) for field in ["u", "v", "key"])
        if key not in graph_keys or key not in native or key not in speed_index:
            raise ValueError("Terminal chain references a missing or wrong-purpose edge")
        source = native[key]
        length = float(part["length_m"])
        if length < 0 or length > source["length"] + 1e-5:
            raise ValueError("Terminal partial length exceeds its native edge")
        if abs(float(part["fraction"])-length/source["length"])>1e-8:
            raise ValueError("Saved terminal fraction differs from its native length ratio")
        boundary_part=part is (route[-1] if direction=="inbound" else route[0])
        if not boundary_part and abs(length-source["length"])>1e-7:
            raise ValueError("Only the projected boundary edge may be partial in a terminal chain")
        if str(part["osmid"]) != source["osmid"]:
            raise ValueError("Terminal native OSM identity differs")
        expected_auth = source["auth"] if purpose == "delivery" else False
        if truth(part.get("delivery_authorization_assumed")) != expected_auth:
            raise ValueError("Terminal authorization annotation differs from source")
        if truth(row.get("local_alternative_selected", False)) and not source["strict"]:
            raise ValueError("Local alternative relaxed its strict permission")
        token=(key,length,part["geometry_wkt"],direction if boundary_part else "full")
        if "geometry" in source and (geometry_cache is None or token not in geometry_cache):
            geom=shapely.from_wkt(part["geometry_wkt"])
            source_geom=shapely.reverse(source["geometry"]) if source.get("reverse") else source["geometry"]
            if abs(length-source["length"])<1e-7:
                good=shapely.equals_exact(geom,source_geom,1e-7)
            else:
                good=shapely.covered_by(geom,shapely.buffer(source_geom,1e-7))
            if not good:
                raise ValueError("Saved path geometry differs from the actual native source edge")
            if boundary_part:
                opposite=-1 if direction=="outbound" else 0
                if np.linalg.norm(np.array(geom.coords[opposite])-np.array(source_geom.coords[opposite]))>1e-7:
                    raise ValueError("Projected partial chain does not reach its actual native endpoint")
            if length_transformer is not None:
                xy=np.array(geom.coords)
                x,y=length_transformer.transform(xy[:,0],xy[:,1])
                metric_length=np.hypot(np.diff(x),np.diff(y)).sum()
                if abs(metric_length-length)>.01:
                    raise ValueError(f"Saved terminal path geometry and metric length disagree: edge={key}, stored={length}, geometry={metric_length}")
            if geometry_cache is not None:
                geometry_cache.add(token)
        total_length += length
        times += np.array(speed_index[key][1:]) * length / source["length"]
    if abs(total_length - expected) > 1e-5:
        raise ValueError("Terminal path length does not conserve source segments")
    partial = route[-1] if direction == "inbound" else route[0]
    geom = shapely.from_wkt(partial["geometry_wkt"])
    endpoint = np.array(geom.coords[-1 if direction == "inbound" else 0])
    anchor = np.array([row.road_anchor_lon, row.road_anchor_lat])
    if np.linalg.norm(endpoint - anchor) > 1e-7:
        raise ValueError("Terminal path geometry does not reach the recorded projection")
    return times, 1


def verify_city_cle(city_dir, *, report_path=None):
    """Verify the assembled bytes, source bindings, all eligible locations and paths."""
    base = Path(city_dir).resolve()
    manifest_path = base / "cle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checks = []

    def check(name, condition, details=None):
        checks.append(dict(name=name, passed=bool(condition), **({"details": details} if details is not None else {})))
        if not condition:
            raise ValueError("CLE verification failed: " + name)

    def inputs_hashes(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                path = Path(value["path"])
                if path.is_absolute():
                    check("external_frozen_input:" + path.name, path.is_file() and sha256_file(path) == value["sha256"])
            for item in value.values():
                inputs_hashes(item)
        elif isinstance(value, list):
            for item in value:
                inputs_hashes(item)

    check("stage1_only", manifest["stage"] == 1 and not manifest["instance_activation_included"] and not manifest["public_release"])
    check("known_cle_schema", manifest["schema"] == "evrptw_city_logistics_environment_v1")
    for name, item in manifest["files"].items():
        path = (base / name).resolve()
        check("saved_hash:" + name, path.is_relative_to(base) and path.is_file() and sha256_file(path) == item["sha256"])
    for name, digest in manifest["policy_hashes"].items():
        check("policy_hash:" + name, sha256_file(CONFIG_ROOT / name) == digest)
    inputs_hashes(manifest["sources"])
    components = {kind: json.loads((base / path).read_text()) for kind,path in manifest["components"].items()}
    road = components["roads"]
    check("metric_projected_area_CRS",projected_metres(road["area_crs"]) and CRS.from_user_input(road["area_crs"])==CRS.from_user_input(manifest["area_crs"]))
    check("common_frozen_road_source", road["source"]["pbf_replication_timestamp_utc"] == "2026-10-25T21:21:21Z" == manifest["road_snapshot_utc"])
    check("complete_way_restriction_nodes", road["source"].get("access_nodes_complete_for_retained_ways") is True)
    pbf = Path(road["source"].get("pbf_path", ""))
    # Synthetic end-to-end fixtures declare their non-PBF origin explicitly;
    # production extraction manifests always bind the actual archived PBF.
    if not road["source"].get("synthetic_test_fixture",False):
        check("frozen_raw_OSM_PBF_hash", pbf.is_file() and sha256_file(pbf)==road["source"]["pbf_sha256"])
    for name,digest in road["source"]["outputs"].items():
        check("native_road_source_hash:"+name,sha256_file(base / "private_sources" / name)==digest)
    for field,name in [("admin_boundary_sha256","admin_boundary"),("service_boundary_sha256","service_boundary")]:
        check("road_boundary_binding:"+name,sha256_file(base / manifest["boundaries"][name])==road[field])
    for kind, component in components.items():
        if "city_slug" in component:
            check(kind + "_city_binding", component["city_slug"] == manifest["city_slug"])
        for name,digest in component.get("generation_dependencies",{}).items():
            check(kind + "_code_hash:" + name, sha256_file(Path(__file__).with_name(name)) == digest)
    inputs_hashes(components["customer"]["inputs"])
    inputs_hashes(components["customer"].get("source_provenance",{}))
    inputs_hashes(components["depot"])
    inputs_hashes(components["charging"]["inputs"])
    inputs_hashes(components["speed"]["inputs"])
    road_hash = sha256_file(base / manifest["components"]["roads"])
    check("customer_context_binding", components["customer"]["inputs"]["local_access_context"]["sha256"] == road_hash)
    check("depot_context_binding", components["depot"]["context_sha256"] == road_hash)
    check("charging_context_binding", components["charging"]["inputs"]["road_context"]["sha256"] == road_hash)
    check("speed_context_binding", components["speed"]["inputs"]["road_context"]["sha256"] == road_hash)
    check("all_component_road_snapshots_match", all(components[k]["road_source"]["pbf_sha256"] == road["source"]["pbf_sha256"] for k in ["charging", "speed"]))
    check("depot_native_road_snapshot", components["depot"]["source_provenance"]["pbf_sha256"] == road["source"]["pbf_sha256"])
    check("customer_road_exclusion_mask_binding",components["customer"]["inputs"]["public_masks"]["sha256"]==road["mask_sha256"])
    check("charging_road_exclusion_mask_binding",components["charging"]["inputs"]["masks"]["sha256"]==road["mask_sha256"])
    check("depot_road_exclusion_mask_binding",components["depot"]["mask_sha256"]==road["mask_sha256"])
    check("charging_road_service_domain_binding",components["charging"]["inputs"]["boundary"]["sha256"]==road["service_boundary_sha256"])
    check("depot_road_service_domain_binding",components["depot"]["boundary_sha256"]==road["service_boundary_sha256"])
    customer_provenance=components["customer"].get("source_provenance",{})
    if "archived_osm_complexes" in customer_provenance:
        evidence=json.loads(Path(customer_provenance["archived_osm_complexes"]["path"]).read_text())
        check("customer_OSM_context_same_frozen_PBF",evidence["source"]["pbf_sha256"]==road["source"]["pbf_sha256"] and evidence["source"]["pbf_replication_timestamp_utc"]==manifest["road_snapshot_utc"])
        check("customer_OSM_building_and_complex_evidence_bound",evidence["source"]["osm_buildings_sha256"]==components["customer"]["inputs"]["osm_buildings"]["sha256"] and evidence["output_sha256"]==components["customer"]["inputs"]["osm_complexes"]["sha256"])
        check("customer_OSM_context_boundary_bound",evidence["source"]["boundary_sha256"]==road["service_boundary_sha256"])
    if "archived_public_masks" in customer_provenance:
        evidence=json.loads(Path(customer_provenance["archived_public_masks"]["path"]).read_text())
        check("exclusion_evidence_same_frozen_PBF",evidence["pbf_sha256"]==road["source"]["pbf_sha256"] and evidence["mask_sha256"]==road["mask_sha256"])

    physical = gpd.read_parquet(base / "roads/physical_roads.parquet")
    check("physical_inventory_uses_declared_metric_CRS",physical.crs==CRS.from_user_input(road["area_crs"]))
    check("unique_native_physical_segments", ~physical.physical_edge_id.duplicated().any())
    raw = gpd.read_parquet(base / "private_sources/osm_ways.parquet", columns=["osmid", "tags_json", "node_ids_json", "geometry"])
    source_geometry = raw.to_crs(physical.crs).set_index("osmid").geometry
    for start in range(0,len(physical),10000):
        chunk = physical.iloc[start:start+10000]
        original = source_geometry.reindex(chunk.osmid).to_numpy()
        if not shapely.covered_by(chunk.geometry.to_numpy(),shapely.buffer(original,.01,quad_segs=1)).all():
            raise ValueError("Physical road geometry contains an invented/non-source segment")
    del source_geometry
    tags = {str(key):json.loads(value) for key,value in zip(raw.osmid,raw.tags_json)}
    nodes = pd.read_parquet(base / "private_sources/osm_access_nodes.parquet", columns=["osmid", "tags_json"])
    restricted_nodes = set(nodes.osmid.astype(str))
    endpoints = {}
    direction_errors = 0
    raw_hard = {"block", "bollard", "cycle_barrier", "jersey_barrier", "kissing_gate", "stile", "turnstile", "wall", "fence", "chain"}
    node_tags = {str(key):json.loads(value) for key,value in zip(nodes.osmid,nodes.tags_json)}
    for row in physical.itertuples():
        source_tags = tags[str(row.osmid)]
        oneway = str(source_tags.get("oneway", "")).strip().lower()
        if oneway in {"-1", "reverse"}:
            expected = {"reverse_of_physical"}
        elif oneway in {"yes", "true", "1"} or (not oneway and (source_tags.get("highway") == "motorway" or source_tags.get("junction") in {"roundabout", "circular"})):
            expected = {"same_as_physical"}
        elif not oneway or oneway in {"no", "false", "0"}:
            expected = {"same_as_physical", "reverse_of_physical"}
        else:
            expected = set()
        actual = {ref["geometry_orientation"] for ref in json.loads(row.directed_edge_refs)}
        direction_errors += int(actual != expected)
        endpoints.setdefault(str(row.osmid),set()).update([str(row.edge_u), str(row.edge_v)])
        if row.motor_access_allowed:
            forbidden = source_tags.get("access:delivery") in {"no", "none", "prohibited"} or source_tags.get("locked") == "yes" or source_tags.get("barrier") in raw_hard
            for key in [str(row.edge_u),str(row.edge_v)]:
                nt = node_tags.get(key,{})
                forbidden = forbidden or nt.get("locked") == "yes" or nt.get("barrier") in raw_hard or nt.get("access:delivery") in {"no", "none", "prohibited"}
            if forbidden:
                raise ValueError("Runnable source segment crosses an explicit hard delivery prohibition")
    check("directions_match_frozen_raw_OSM_tags", direction_errors == 0)
    missing_splits = 0
    for row in raw.itertuples():
        if str(row.osmid) in endpoints:
            used = set(str(key) for key in json.loads(row.node_ids_json)) & restricted_nodes
            missing_splits += len(used - endpoints[str(row.osmid)])
    check("all_restriction_nodes_are_segment_endpoints", missing_splits == 0, dict(missing=missing_splits))
    del raw,tags,nodes,node_tags,endpoints
    native = {}
    for row,wgs_geometry in zip(physical.itertuples(),physical.to_crs(4326).geometry):
        for ref in json.loads(row.directed_edge_refs):
            key = tuple(str(ref[field]) for field in ["u", "v", "key"])
            if key in native:
                raise ValueError("Duplicate native directed edge")
            native[key] = dict(
                length=float(ref["length_m"]), osmid=str(row.osmid),
                delivery=bool(row.motor_access_allowed), charging=bool(row.charging_motor_access_allowed),
                public=row.access_layer == "operational_public", tier=row.legal_access_tier,
                auth=bool(row.delivery_authorization_assumed), strict=bool(row.strict_motor_access_allowed),
                highway=row.highway, physical_id=row.physical_edge_id,
                geometry=wgs_geometry, reverse=ref["geometry_orientation"]=="reverse_of_physical",
            )
    public = set()
    delivery = set()
    charge = set()
    for layer, path, target in [
        ("public", "roads/graph_operational.graphml", public),
        ("delivery", "roads/graph_terminal_access.graphml", delivery),
        ("charging", "charging/private_access/graph_terminal_access.graphml", charge),
    ]:
        invalid = 0
        for key, data in graph_edges(base / path):
            source = native.get(key)
            if source is None:
                invalid += 1
                continue
            target.add(key)
            permitted = source["charging"] if layer == "charging" else source["delivery"]
            invalid += int(not permitted or abs(float(data["length"]) - source["length"]) > 1e-5 or str(data["osmid"]) != source["osmid"])
            expected_geometry = shapely.reverse(source["geometry"]) if source["reverse"] else source["geometry"]
            invalid += int(not shapely.equals_exact(shapely.from_wkt(data["geometry"]),expected_geometry,1e-8))
            if layer == "public":
                invalid += int(not source["public"] or source["auth"] or source["tier"] in {"permit_required", "delivery_only", "destination_or_customers"} or source["highway"] == "service" or not truth(data.get("through_routing_allowed")))
            if layer == "charging":
                invalid += int(truth(data.get("delivery_authorization_assumed")))
        check("actual_saved_graph_permission_and_identity:" + layer, invalid == 0, dict(edges=len(target), invalid=invalid))
    reference = independent_reference(public)
    forest_reference = set(json.loads((base / "roads/access_forests.json").read_text())["reference"])
    check("independent_saved_graph_SCC", reference == forest_reference)
    check("public_graph_is_nonempty", len(public) > 0 and len(reference) > 0)
    speeds = pd.read_parquet(base / "speeds/directed_speeds.parquet")
    profile=json.loads((base / "speeds/moves_speed_profile.json").read_text())
    registry=json.loads((CONFIG_ROOT / "us_nei2023_streetlight_state_profiles_v1.json").read_text())
    expected_state=registry["city_state_fips"].get(manifest["city_slug"])
    if expected_state is None:
        source_inventory=json.loads(Path(manifest["sources"]["frozen_inventory"]["path"]).read_text())
        expected_state=source_inventory.get("cities",{}).get(manifest["city_slug"],{}).get("state_fips")
        if expected_state is None and road["source"].get("synthetic_test_fixture"):
            expected_state=profile["state_fips"]
    check("state_profile_assignment_is_explicit",expected_state is not None and str(expected_state).zfill(2)==profile["state_fips"])
    expected_profile=CONFIG_ROOT / registry["states"][str(expected_state).zfill(2)]["profile"]
    check("stored_state_profile_matches_frozen_statistics",sha256_file(expected_profile)==sha256_file(base / "speeds/moves_speed_profile.json"))
    check("commercial_vehicle_statistical_class",profile["source_type_id"]==32)
    check("per_edge_profile_identity",speeds.reference_speed_source.eq(profile["profile_id"]).all())
    for day in ["weekday","weekend"]:
        rho=speeds.moves_road_type.map({key:value["speed_retention_factor"][day] for key,value in profile["road_types"].items()})
        check("per_edge_state_retention:"+day,rho.notna().all() and np.allclose(speeds["reference_speed_"+day+"_kph"],speeds.legal_speed_kph*rho))
    if components["speed"]["inputs"]["hpms"] is None:
        restricted=speeds.highway.isin(["motorway","motorway_link","trunk","trunk_link"])
        expected_type=np.where(restricted,"urban_restricted_access","urban_unrestricted_access")
        check("MOVES_road_type_from_native_highway",np.array_equal(speeds.moves_road_type,expected_type))
        check("no_fabricated_HPMS_observations",components["speed"]["observed_hpms_speed_limit_edge_count"]==0 and speeds.raw_hpms_speed_limit_mph.isna().all())
    speed_index = {
        tuple(key):tuple(value) for key,value in zip(
            speeds[["edge_u", "edge_v", "edge_key"]].astype(str).to_numpy(),
            speeds[["length_m", "legal_travel_time_s", "reference_travel_time_weekday_s", "reference_travel_time_weekend_s"]].to_numpy(),
        )
    }
    check("exact_common_speed_edge_coverage", len(speed_index) == len(speeds) and set(speed_index) == public | delivery | charge)
    check("common_speed_native_lengths_and_source_tags",check_native_speed_metadata(speeds,native))
    for flag, keys in [("in_public_graph", public), ("in_delivery_terminal_graph", delivery), ("in_charging_terminal_graph", charge)]:
        actual = {tuple(row) for row in speeds.loc[speeds[flag], ["edge_u", "edge_v", "edge_key"]].astype(str).to_numpy()}
        check("speed_membership:" + flag, actual == keys)
    for label, speed, time in [
        ("legal", "legal_speed_kph", "legal_travel_time_s"),
        ("weekday", "reference_speed_weekday_kph", "reference_travel_time_weekday_s"),
        ("weekend", "reference_speed_weekend_kph", "reference_travel_time_weekend_s"),
    ]:
        check("speed_cost_formula:" + label, np.isfinite(speeds[speed]).all() and speeds[speed].gt(0).all() and np.allclose(speeds[time], 3.6 * speeds.length_m / speeds[speed]))
        check("reference_within_legal:" + label, (speeds[speed] <= speeds.legal_speed_kph + 1e-9).all())
    del speeds

    customer = gpd.read_parquet(base / "customers/latent_locations.parquet")
    depot = gpd.read_parquet(base / "depots/depots.parquet")
    charging = gpd.read_parquet(base / "charging/chargers.parquet")
    canonical = gpd.read_parquet(base / "locations.parquet")
    members = pd.read_parquet(base / "customers/residential_site_members.parquet")
    check("customer_source_owned_once", members.fd_id.notna().all() and ~members.fd_id.duplicated().any() and members.site_id.isin(customer.site_id).all())
    source_nsi=pd.read_parquet(Path(components["customer"]["inputs"]["nsi_crosswalk"]["path"]),columns=["fd_id","occtype","resunits","composite_building_id"]).set_index("fd_id")
    check("all_customer_members_have_frozen_native_records",source_nsi.index.is_unique and members.fd_id.isin(source_nsi.index).all())
    evidence=source_nsi.reindex(members.fd_id)
    for field in ["occtype","resunits","composite_building_id"]:
        original=evidence[field].reset_index(drop=True)
        actual=members[field].reset_index(drop=True)
        check("unaltered_native_customer_evidence:"+field,same_evidence(original,actual,field))
    del source_nsi,evidence
    check("customer_identity_unique", customer.site_id.notna().all() and ~customer.site_id.duplicated().any())
    counted = members.loc[members.operational_units_counted]
    check("household_component_counted_once", ~counted.nsi_component_id.duplicated().any())
    check("household_values_positive_or_unknown", customer.residential_units.dropna().gt(0).all())
    sums = counted.groupby("site_id").counted_residential_units.sum().reindex(customer.site_id).to_numpy(dtype=float)
    known = customer.residential_units.notna().to_numpy()
    check("site_units_equal_unique_counted_evidence", np.allclose(customer.loc[known, "residential_units"].astype(float), sums[known]))
    check("scope_ambiguity_not_default_available", ~customer.loc[customer.site_scope_review_required, "cle_default_instance_eligible"].any() and customer.loc[customer.site_scope_review_required, "residential_units"].isna().all())
    owner_counts=check_source_owner_flags(customer,members)
    check("independent_saved_source_owner_competition",True,owner_counts)
    check("customer_eligibility_boolean_contract", np.array_equal(customer.cle_default_instance_eligible, customer.road_access_candidate_eligible & ~customer.site_scope_review_required))
    check("no_operating_day_customer_activation", ~customer.active_customer.any())
    check("canonical_identity_unique", ~canonical.location_id.duplicated().any() and canonical.location_id.eq(canonical.location_type + ":" + canonical.source_location_id).all())
    check("canonical_location_counts", len(canonical) == len(customer) + depot.facility_group_id.nunique() + len(charging))
    costs = pd.read_parquet(base / "location_access_costs.parquet").set_index("location_id")
    check("access_costs_only_for_eligible_locations", costs.index.is_unique and set(costs.index) == set(canonical.loc[canonical.candidate_eligible, "location_id"]))
    cost_columns=["connector_length_m","connector_time_s_per_inverse_kph"]
    cost_columns += [direction+"_terminal_"+name for direction in ["inbound","outbound"]
                     for name in ["length_m","legal_time_s","reference_weekday_time_s","reference_weekend_time_s"]]
    cost_values=costs[cost_columns].to_numpy(dtype=float)
    cost_rows={key:i for i,key in enumerate(costs.index)}
    cost_column_index={name:i for i,name in enumerate(cost_columns)}
    all_native = {
        "customer": (customer, "site_id", "cle_default_instance_eligible"),
        "depot": (depot, "candidate_id", "depot_candidate_eligible"),
        "charging": (charging, "charging_location_id", "charging_parameter_eligible"),
    }
    canon_index = canonical.set_index("location_id")
    for kind, (table, key, gate) in all_native.items():
        check(kind + "_native_identity_unique", table[key].notna().all() and ~table[key].duplicated().any())
        selected = canonical.loc[canonical.location_type.eq(kind)]
        check(kind + "_canonical_native_foreign_keys", selected.access_row_id.isin(table[key]).all())
        access = table.set_index(key).loc[selected.access_row_id]
        check(kind + "_canonical_source_geometry_preserved", np.array_equal(selected.geometry.to_wkb(), access.geometry.to_wkb()))
        check(kind + "_canonical_eligibility_preserved", np.array_equal(selected.candidate_eligible, access[gate]))
        if kind=="customer":
            for field, native_field in [("residential_units","residential_units"),("residential_type","service_location_type"),
                                         ("source_identity","site_identity_source"),("scope_unresolved","site_scope_unresolved"),
                                         ("scope_review_required","site_scope_review_required")]:
                left=selected[field].reset_index(drop=True)
                right=access[native_field].reset_index(drop=True)
                check("canonical_customer_auxiliary_evidence:"+field,(left.eq(right)|(left.isna()&right.isna())).fillna(False).all())
    expected_depots = depot.groupby("facility_group_id").depot_candidate_eligible.any()
    actual_depots = canonical.loc[canonical.location_type.eq("depot")].set_index("source_location_id").candidate_eligible
    check("depot_unique_group_eligibility", actual_depots.sort_index().equals(expected_depots.sort_index()))
    masks_path = Path(components["customer"]["inputs"]["public_masks"]["path"])
    masks = gpd.read_parquet(masks_path).to_crs(4326)
    if len(masks):
        pairs = shapely.STRtree(masks.geometry.to_numpy()).query(canonical.geometry.to_numpy(), predicate="intersects")
        check("all_retained_sources_exclude_sensitive_masks", pairs.size == 0)
    shape = physical.set_index("physical_edge_id").geometry
    host_permission=physical.set_index("physical_edge_id")
    metric_masks = masks.to_crs(physical.crs)
    mask_tree = shapely.STRtree(metric_masks.geometry.to_numpy()) if len(masks) else None
    admin_shape = gpd.read_file(base / manifest["boundaries"]["admin_boundary"]).to_crs(physical.crs).geometry.union_all()
    service_shape = gpd.read_file(base / manifest["boundaries"]["service_boundary"]).to_crs(physical.crs).geometry.union_all()
    water = admin_shape.difference(service_shape)
    footprints_path = Path(components["customer"]["inputs"]["microsoft_footprints"]["path"])
    buildings = gpd.read_parquet(footprints_path).to_crs(physical.crs).geometry.to_numpy()
    building_tree = shapely.STRtree(buildings)
    routes = 0
    from pyproj import Transformer
    length_transformer=Transformer.from_crs(4326,physical.crs,always_xy=True)
    geometry_cache=set()
    for kind, (table, key, gate) in all_native.items():
        eligible = table.loc[table[gate]]
        check(kind + "_eligible_has_directed_access", eligible.local_access_confirmed.all() and eligible.protected_roundtrip_eligible.all() and eligible.access_review_reason.eq("").all())
        check(kind + "_no_manual_id_holds", "access_case_review_hold" not in table)
        check(kind + "_frontage_guard_applied", ~eligible.unsupported_frontage_nearer.any())
        permission_field="charging_motor_access_allowed" if kind=="charging" else "motor_access_allowed"
        check(kind + "_host_native_permission",eligible.physical_edge_id.map(host_permission[permission_field]).fillna(False).all())
        check(kind + "_owner_permission_not_claimed",~table.owner_permission_observed.any())
        anchor = gpd.GeoSeries(gpd.points_from_xy(eligible.road_anchor_lon, eligible.road_anchor_lat), crs=4326).to_crs(physical.crs).to_numpy()
        host = shape.reindex(eligible.physical_edge_id).to_numpy()
        source = eligible.to_crs(physical.crs).geometry.to_numpy()
        check(kind + "_metric_projection_and_connector", (shapely.distance(anchor,host) < .01).all() and (np.abs(shapely.distance(source,anchor) - eligible.road_access_distance_m.to_numpy()) < .01).all())
        lines = shapely.shortest_line(source,anchor)
        if mask_tree is not None:
            check(kind + "_connectors_avoid_sensitive_masks", mask_tree.query(lines,predicate="intersects").size==0)
        if not water.is_empty:
            check(kind + "_connectors_do_not_cross_unmapped_water", (shapely.length(shapely.intersection(lines,water))<=.01).all())
        pairs = building_tree.query(lines,predicate="crosses")
        if pairs.size:
            check(kind + "_connectors_do_not_cross_unrelated_buildings", shapely.intersects(source[pairs[0]],buildings[pairs[1]]).all())
        graph_keys = public | (charge if kind == "charging" else delivery)
        route_fields = [key, "requires_terminal_paths", "road_anchor_lon", "road_anchor_lat", "road_access_distance_m", "physical_edge_id", "delivery_authorization_assumed"]
        route_fields += ["terminal_" + direction + "_" + name for direction in ["inbound", "outbound"] for name in ["path_json", "gateway", "length_m"]]
        route_fields += [name for name in ["facility_group_id", "local_alternative_selected"] if name in eligible and name not in route_fields]
        for values in eligible[route_fields].itertuples(index=False, name=None):
            row = AccessRow(zip(route_fields, values))
            authorization=bool(host_permission.at[row.physical_edge_id,"delivery_authorization_assumed"]) if kind!="charging" else False
            for direction in ["inbound", "outbound"]:
                times, count = check_path(row, direction, native, graph_keys, reference, speed_index, purpose="charging" if kind == "charging" else "delivery",geometry_cache=geometry_cache,length_transformer=length_transformer)
                routes += count
                if kind!="charging":
                    authorization=authorization or any(truth(part.get("delivery_authorization_assumed")) for part in json.loads(row["terminal_"+direction+"_path_json"]))
                source_id = str(row["facility_group_id"] if kind == "depot" else row[key])
                location_id = kind + ":" + source_id
                # Multiple depot access rows are retained. Costs bind the
                # deterministic canonical row, not every group member.
                if kind == "depot" and canon_index.at[location_id, "access_row_id"] != str(row[key]):
                    continue
                actual = cost_values[cost_rows[location_id]]
                check_values = actual[[cost_column_index[direction + "_terminal_" + part + "_time_s"] for part in ["legal", "reference_weekday", "reference_weekend"]]]
                if not np.allclose(times, check_values, rtol=1e-9, atol=1e-7):
                    raise ValueError("Saved terminal costs differ from native source-edge sum")
                if abs(actual[cost_column_index[direction+"_terminal_length_m"]]-float(row["terminal_"+direction+"_length_m"]))>1e-7:
                    raise ValueError("Saved terminal cost length differs from native access chain")
                if abs(actual[cost_column_index["connector_time_s_per_inverse_kph"]] - 3.6*float(row.road_access_distance_m)) > 1e-7 or abs(actual[cost_column_index["connector_length_m"]]-float(row.road_access_distance_m))>1e-7:
                    raise ValueError("Connector cost coefficient differs from geometric length")
            if truth(row.delivery_authorization_assumed)!=authorization or (authorization and not truth(row.requires_terminal_paths)):
                raise ValueError("Location invitation flag disagrees with its native host/path permissions")
        check(kind + "_all_saved_paths_and_costs", True, dict(eligible_rows=len(eligible)))
    configurations = pd.read_parquet(base / "charging/charging_configurations.parquet")
    charging_members = pd.read_parquet(base / "charging/charging_members.parquet")
    check("charging_member_ownership", charging_members.charging_location_id.isin(charging.charging_location_id).all() and ~charging_members.ID.duplicated().any())
    selected = configurations.loc[configurations.is_selected]
    check("one_reference_configuration_per_compatible_location", selected.charging_location_id.is_unique and set(selected.charging_location_id) == set(charging.loc[charging.reference_vehicle_compatible, "charging_location_id"]))
    check("charging_configurations_point_to_native_members", set(selected.afdc_id) <= set(charging_members.ID))
    chosen=selected.set_index("charging_location_id")
    compatible=charging.loc[charging.reference_vehicle_compatible]
    check("selected_configuration_and_scalar_AFDC_identity_agree",compatible.ID.eq(compatible.charging_location_id.map(chosen.afdc_id)).all() and compatible.selected_configuration_id.eq(compatible.charging_location_id.map(chosen.configuration_id)).all())
    check("native_reference_connectors_only",(configurations.reference_charge_mode.eq("ac_level2") & configurations.connector.eq("J1772") | configurations.reference_charge_mode.eq("dc_fast") & configurations.connector.eq("J1772COMBO")).all())
    check("charging_positive_power_and_caps",configurations.assigned_power_kw.gt(0).all() and configurations.vehicle_cap_kw.gt(0).all())
    check("charging_power_cap_formula", np.allclose(configurations.capped_pre_derating_kw, np.minimum(configurations.assigned_power_kw, configurations.vehicle_cap_kw)))
    check("charging_does_not_assume_delivery_permission", ~charging.delivery_authorization_assumed.any())
    check("charging_parameter_gate", np.array_equal(charging.charging_parameter_eligible, charging.potential_location_eligible & charging.road_accessible & charging.reference_vehicle_compatible))
    summary = {
        kind: dict(retained=len(group), available=int(group.candidate_eligible.sum()), pending=int((~group.candidate_eligible).sum()))
        for kind,group in canonical.groupby("location_type")
    }
    check("manifest_counts_match_saved_files", summary == manifest["summary"])
    result = dict(
        schema="evrptw_independent_cle_verification_v1", passed=True,
        city_slug=manifest["city_slug"], generated_utc=datetime.now(UTC).isoformat(),
        cle_manifest_sha256=sha256_file(manifest_path), verifier_sha256=sha256_file(Path(__file__)),
        checks=checks, summary=summary, verified_terminal_paths=routes,
        common_speed_edges=len(speed_index), instance_activation_included=False,
        no_data_or_eligibility_modified=True,
        limitations=["Finite frozen public-source model, not surveyed entrances or owner consent", "No turn-restriction/conditional-access/dynamic traffic/queue model", "Pending uncertainty is retained outside default eligibility", "Source unit scopes remain modeled auxiliary evidence; unknown values stay null"],
    )
    authority=(base / manifest.get("freeze_authority","../../stage1_freeze.json")).resolve()
    target=Path(report_path).resolve() if report_path else base / "cle_verification.json"
    if authority.is_file():
        if report_path is None:
            return result  # A frozen certificate must not change on read-back.
        if target.is_relative_to(base):
            raise ValueError("Write frozen CLE re-audits outside the immutable city data directory")
    write_json(target, result)
    return result
