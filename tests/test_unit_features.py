"""Raw building evidence determines the units cohort without cached trial data."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from evrptw_cle.building_inputs import prepare_building_population
from evrptw_cle.unit_features import prepare_features


def test_raw_native_zero_scenarios_and_independent_building_features(tmp_path):
    polygons = [box(i * .003, 0, i * .003 + .001, .001) for i in range(6)]
    labels = ["yes", "yes", "yes", "retail", "apartments", "yes"]
    microsoft = gpd.GeoDataFrame({
        "building_id": [f"ms-{i}" for i in range(6)],
        "source_kind": ["global"] * 6,
        "height": np.arange(6, dtype=float) + 10.,
    }, geometry=polygons, crs=4326)
    osm = gpd.GeoDataFrame({
        "osm_id": [f"w-{i}" for i in range(6)],
        "building": labels,
        "building_part": [False] * 6,
        "tags": [json.dumps({"building": label, "building:levels": str(i + 2)})
                 for i, label in enumerate(labels)],
    }, geometry=polygons, crs=4326)
    # The third building has no native point.  The last has conflicting zero
    # and positive records from different upstream footprints.
    owners = [0, 1, 3, 4, 5, 5]
    areas = gpd.GeoSeries(polygons, crs=4326).to_crs(3857).area.to_numpy()
    nsi = gpd.GeoDataFrame({
        "fd_id": np.arange(1, 7),
        "bid": [None] * 6,
        "ftprntsrc": ["BING"] * 6,
        "ftprntid": [f"footprint-{i}" for i in range(6)],
        "ftprntsqft": areas[owners] / .09290304,
        "resunits": [0, 1, 4, 400, 0, 4],
        "occtype": ["RES1", "RES1", "RES1", "RES3F", "RES1", "RES3B"],
        "bldheight": [9999.] * 6,
        "num_story": [9999.] * 6,
    }, geometry=[Point(i * .003 + .0005, .0005) for i in owners], crs=4326)
    sources = {"area_crs": "EPSG:3857"}
    for name, frame in [("microsoft", microsoft), ("osm", osm), ("nsi", nsi)]:
        sources[name] = tmp_path / f"{name}.parquet"
        frame.to_parquet(sources[name], index=False)
    base = tmp_path / "city"
    customer_inputs = base / "customer_inputs"
    prepare_building_population(city="fixture", sources=sources,
        output_dir=customer_inputs, axis_r=.3)
    population = pd.read_parquet(customer_inputs / "population.parquet")
    positions = population.entity_id.str.removeprefix("ms-").astype(int)
    pd.DataFrame({
        "entity_id": population.entity_id,
        "canonical_longitude": positions * .003 + .0005,
        "canonical_latitude": .0005,
        "service_domain_eligible": True,
    }).to_parquet(customer_inputs / "scope.parquet", index=False)

    features = prepare_features(city="fixture", sources=sources, base=customer_inputs, axis_r=.3)
    result = features.set_index("entity_id")
    assert set(result.index) == {"ms-1", "ms-2", "ms-4", "ms-5"}
    assert result.loc["ms-1", "scenario_status"] == "residential_after_nsi1"
    assert result.loc["ms-1", "units"] == 1
    assert result.loc["ms-2", "scenario_status"] == "unknown_no_nsi_match"
    assert result.loc["ms-4", "scenario_status"] == "residential_after_nsi1"
    assert result.loc["ms-4", "clean_direct"]
    assert result.loc["ms-4", "units"] == 400
    assert result.loc["ms-4", "use_type"] == "apartments"
    assert result.loc["ms-5", "scenario_status"] == "unknown_zero_positive_conflict"
    assert result.loc["ms-5", "native_conflict_status"]
    expected = population.set_index("entity_id").loc[result.index, "height_m"]
    np.testing.assert_array_equal(result.height_m.to_numpy(), expected.to_numpy())
    assert result.levels.to_dict() == {"ms-1": 3., "ms-2": 4., "ms-4": 6., "ms-5": 7.}
    assert result.loc["ms-1", "use_type"] == "generic"
