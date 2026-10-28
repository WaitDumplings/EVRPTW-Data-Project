import geopandas as gpd
from shapely.geometry import LineString

from evrptw_cle.frontage import apply_frontage_guard
from test_delivery_access import authorized_context
from test_local_road_access import locations


def source_and_roads(highway):
    context = authorized_context()
    source = context.anchor(locations())
    roads = context.physical.copy()
    other = roads.iloc[[0]].copy()
    other["physical_edge_id"] = "nearby"
    other["osmid"] = "nearby"
    other["highway"] = highway
    # Source centroid is near x=10,y=50; this frontage is closer than x=0.
    other.geometry = [LineString([(9, 0), (9, 100)])]
    roads = gpd.GeoDataFrame(__import__("pandas").concat([roads, other]), crs=roads.crs)
    return source, roads


def test_unsupported_frontage_is_pending_without_source_deletion_or_id_rule():
    source, roads = source_and_roads("track")
    for source_id in ["arbitrary_city_house", "another_source"]:
        source["latent_service_location_id"] = source_id
        result = apply_frontage_guard(source, roads, roads.crs)
        assert result.unsupported_frontage_nearer.all()
        assert not result.local_access_confirmed.any()
        assert result.access_review_reason.eq("nearer_unsupported_motor_frontage").all()
        assert result.geometry.equals(source.geometry)


def test_pedestrian_path_is_not_an_unsupported_motor_frontage():
    source, roads = source_and_roads("footway")
    result = apply_frontage_guard(source, roads, roads.crs)
    assert result.local_access_confirmed.all()
    assert not result.unsupported_frontage_nearer.any()


def test_farther_track_does_not_block_supported_frontage():
    source, roads = source_and_roads("track")
    roads.loc[roads.osmid.eq("nearby"), "geometry"] = LineString([(500, 0), (500, 100)])
    result = apply_frontage_guard(source, roads, roads.crs)
    assert result.local_access_confirmed.all()
