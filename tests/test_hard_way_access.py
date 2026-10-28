import json

import geopandas as gpd
import pandas as pd
import pytest

from evrptw_cle.delivery_access import permission_inventory
from test_local_road_access import context
from test_charging_component import _layers, _road, _station


@pytest.mark.parametrize("hard",[{"locked":"yes"},{"barrier":"bollard"},{"access:delivery":"no"}])
def test_explicit_hard_way_cannot_be_reopened_by_private_invitation(tmp_path,hard):
    physical=context(permit=True).physical
    physical["osmid"]="2"
    pd.DataFrame(columns=["osmid","tags_json"]).to_parquet(tmp_path / "osm_access_nodes.parquet",index=False)
    pd.DataFrame([dict(osmid="2",tags_json=json.dumps({"highway":"service","access":"private",**hard}))]).to_parquet(tmp_path / "osm_ways.parquet",index=False)
    masks=gpd.GeoDataFrame(geometry=[],crs=3857)
    result,_=permission_inventory(physical,tmp_path,masks)
    assert not result.motor_access_allowed.any()


def test_unsupported_track_stays_diagnostic_and_cannot_be_charging_access(tmp_path):
    context,_=_road(tmp_path,terminal_tags={"highway":"track"})
    tracks=context.physical.loc[context.physical.highway.eq("track")]
    assert len(tracks) and not tracks.motor_access_allowed.any()
    assert not tracks.charging_motor_access_allowed.any()
    locations,_,_,_,_=_layers(context,_station())
    assert locations.potential_location_eligible.all()
    assert not locations.road_accessible.any()
    assert locations.unsupported_frontage_nearer.all()
