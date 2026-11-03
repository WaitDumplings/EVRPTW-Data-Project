import json

import geopandas as gpd
import pytest
from shapely.geometry import box

from evrptw_cle.customer_building_layer import build_building_customer_layer


def test_building_adapter_uses_full_polygons_and_gates_information_and_roads(tmp_path, monkeypatch):
    context_dir = tmp_path / "roads"
    context_dir.mkdir()
    (context_dir / "access_context_manifest.json").write_text('{"policy":"test"}')
    inputs = tmp_path / "city/customer_inputs"
    inputs.mkdir(parents=True)
    (inputs / "finalization_manifest.json").write_text('{"schema":"test_finalization"}')
    request = tmp_path / "provider_test_record.json"
    request.write_text('{"synthetic_test_fixture":true,"model":"test"}')
    (inputs / "assessments.jsonl").write_text(json.dumps({"entity_id": "a", "ai_runs": [{
        "record_path": str(request), "request_id": "test-request", "model": "test"}]}) + "\n")
    sources = gpd.GeoDataFrame({
        "entity_id": ["a", "b", "c", "d"],
        "residential_use": ["residential", "unknown", "residential", "residential"],
        "residential_source": ["osm", "unresolved", "ai_inferred", "osm"],
        "units_final": [1, 2, 12, 120],
        "units_raw_nsi": [1, 2, 12, 120],
        "units_method": ["nsi_retained"] * 4,
        "units_scope": ["building"] * 4,
        "units_review_status": ["not_flagged"] * 4,
        "units_evidence": [[{"osm_id": "w1"}], [], [], []],
    }, geometry=[box(-117 + i * .001, 33, -116.9995 + i * .001, 33.0005)
                 for i in range(4)], crs=4326)

    class Context:
        policy = "test_full_polygon_access"

        def anchor(self, sites, *, masks, footprints):
            assert masks is None
            assert sites.geometry.geom_type.eq("Polygon").all()
            assert sites.geometry.equals(sources.geometry)
            assert sites.access_search_extent_wkt.equals(sites.geometry.to_wkt())
            assert footprints.geometry.equals(sites.geometry)
            result = sites.copy()
            result["protected_roundtrip_eligible"] = [True, True, False, True]
            result["access_review_reason"] = ["", "", "no_roundtrip", ""]
            result["terminal_inbound_path_json"] = "[]"
            result["terminal_outbound_path_json"] = "[]"
            return result

    monkeypatch.setattr("evrptw_cle.customer_building_layer.LocalAccessContext.load",
                        lambda path: Context())
    manifest = build_building_customer_layer(buildings=sources, context_dir=context_dir,
        output_dir=tmp_path / "city", city_slug="test-city", area_crs="EPSG:32611", profile="u_lt100")
    result = gpd.read_parquet(tmp_path / "city/customers/latent_locations.parquet")
    assert result.cle_default_instance_eligible.tolist() == [True, False, False, False]
    assert result.service_location_type.eq("residential_building").all()
    assert result.site_identity_source.eq("microsoft_osm_canonical_building").all()
    assert result.units_raw_nsi.tolist() == [1, 2, 12, 120]
    assert result.geometry.equals(sources.geometry)
    assert result.quarantine_reason.tolist() == ["", "residential_use_unknown", "no_roundtrip",
                                               "units_outside_selected_profile"]
    assert json.loads(result.units_evidence.iloc[0]) == [{"osm_id": "w1"}]
    assert manifest["summary"]["cle_available_buildings"] == 1
    assert manifest["summary"]["retained_buildings"] == 4
    assert not manifest["apartment_complex_inference"]
    assert (tmp_path / "city/customers/customer_lineage.json").read_bytes() == (
        inputs / "finalization_manifest.json").read_bytes()
    assert (tmp_path / "city/customers/annotation_records.jsonl").read_bytes() == (
        inputs / "assessments.jsonl").read_bytes()
    copied = manifest["ai_request_records"][str(request)]["path"]
    assert (tmp_path / "city/customers" / copied).read_bytes() == request.read_bytes()


@pytest.mark.parametrize("change_geometry", [False, True])
def test_projection_cache_refreshes_metadata_and_routes_only_new_or_changed_shapes(
        tmp_path, monkeypatch, change_geometry):
    context_dir = tmp_path / "roads"
    context_dir.mkdir()
    (context_dir / "access_context_manifest.json").write_text('{"policy":"test"}')
    original = gpd.GeoDataFrame({
        "entity_id": ["a", "b", "c"], "residential_use": ["residential"] * 3,
        "residential_source": ["osm"] * 3, "units_final": [1, 2, 3],
        "units_raw_nsi": [1, 2, 3], "units_method": ["nsi_retained"] * 3,
        "units_scope": ["building"] * 3, "units_review_status": ["not_flagged"] * 3,
        "units_low": [1, 2, 3], "units_high": [1, 2, 3],
    }, geometry=[box(i, 0, i + .4, .4) for i in range(3)], crs=4326)
    calls = []

    class Context:
        policy = "test"

        def anchor(self, sites, *, masks, footprints):
            calls.append(sites.entity_id.tolist())
            assert sites.index.tolist() == list(range(len(sites)))
            assert footprints.geometry.equals(original.geometry)
            result = sites.copy()
            result["protected_roundtrip_eligible"] = True
            result["access_review_reason"] = ""
            result["road_access_distance_m"] = [float(ord(x) - 96) for x in sites.entity_id]
            result["terminal_inbound_path_json"] = "[]"
            result["terminal_outbound_path_json"] = "[]"
            result["directed_projection_offsets"] = [[{"source": x}] for x in sites.entity_id]
            return result

    monkeypatch.setattr("evrptw_cle.customer_building_layer.LocalAccessContext.load",
                        lambda path: Context())
    kwargs = dict(context_dir=context_dir, city_slug="test-city", area_crs="EPSG:3857",
                  profile="u_lt100", obstacle_buildings=original)
    build_building_customer_layer(buildings=original, output_dir=tmp_path / "first", **kwargs)
    previous = gpd.read_parquet(tmp_path / "first/customers/latent_locations.parquet")
    current = original.copy().drop(columns=["units_low", "units_high"])
    current.loc[0, ["units_final", "units_method", "residential_source"]] = [120, "human_corrected", "human_reviewed"]
    current.loc[1, ["residential_use", "residential_source"]] = ["unknown", "unresolved"]
    current.loc[2, "entity_id"] = "d"  # Prior c disappeared; this is a new location.
    if change_geometry:
        current.loc[0, "geometry"] = box(0, 0, .41, .4)
    report = build_building_customer_layer(buildings=current, projection_cache=previous,
                                           output_dir=tmp_path / "second", **kwargs)
    expected = ["a", "d"] if change_geometry else ["d"]
    assert calls == [["a", "b", "c"], expected]
    refreshed = gpd.read_parquet(tmp_path / "second/customers/latent_locations.parquet")
    assert refreshed.entity_id.tolist() == ["a", "b", "d"]
    assert refreshed.units_final.tolist() == [120, 2, 3]
    assert refreshed.residential_source.tolist() == ["human_reviewed", "unresolved", "osm"]
    assert refreshed.cle_default_instance_eligible.tolist() == [False, False, True]
    assert refreshed.quarantine_reason.tolist() == ["units_outside_selected_profile", "residential_use_unknown", ""]
    assert "units_low" not in refreshed and "units_high" not in refreshed
    assert [json.loads(value) for value in refreshed.directed_projection_offsets] == [
        [{"source": "a"}], [{"source": "b"}], [{"source": "d"}]]
    assert report["summary"]["reused_projection_rows"] == 3 - len(expected)
    assert report["summary"]["projected_rows"] == len(expected)
    # All shapes now match; no graph load or anchor call is needed again.
    def no_graph_load(path):
        raise AssertionError("Unchanged projections should not reload the graph")
    monkeypatch.setattr("evrptw_cle.customer_building_layer.LocalAccessContext.load", no_graph_load)
    all_reused = build_building_customer_layer(buildings=current, projection_cache=refreshed,
                                               output_dir=tmp_path / "third", **kwargs)
    assert all_reused["summary"]["reused_projection_rows"] == 3
    assert all_reused["summary"]["projected_rows"] == 0
    assert "directed_projection_offsets" in all_reused["json_encoded_columns"]["latent_locations.parquet"]
