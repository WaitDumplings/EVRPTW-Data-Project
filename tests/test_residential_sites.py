import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from evrptw_cle.residential_sites import infer_residential_sites


def inputs():
    records = gpd.GeoDataFrame(
        [
            dict(
                fd_id=i + 1,
                occtype="RES3C",
                resunits=units,
                modeled_units=units or 5,
                ftprntsrc="S",
                ftprntid=identity,
                usastrucid=0,
                cbfips="00100",
                composite_status="matched_unique_containment"
                if building
                else "quarantine_no_containment",
                composite_building_id=building,
                geometry=Point(x, y),
            )
            for i, (x, y, building, identity, units) in enumerate(
                [
                    (1, 1, "a", "a", 10),
                    (1.2, 1, "a", "a", 10),
                    (4, 1, "b", "b", 20),
                    (2.7, 1, "", "c", None),
                    (8, 1, "c", "d", None),
                ]
            )
        ],
        crs=4326,
    )
    footprints = gpd.GeoDataFrame(
        {"building_id": ["a", "b", "c"]},
        geometry=[box(0.5, 0.5, 1.5, 1.5), box(3.5, 0.5, 4.5, 1.5), box(7.5, 0.5, 8.5, 1.5)],
        crs=4326,
    )
    buildings = gpd.GeoDataFrame(
        {"osm_id": ["w1", "w2", "w3"], "usable": [True] * 3},
        geometry=footprints.geometry.copy(),
        crs=4326,
    )
    complexes = gpd.GeoDataFrame(
        {
            "osm_id": ["w4"],
            "site_evidence": ["residential_apartments_area"],
            "member_osm_ids_json": [json.dumps([])],
        },
        geometry=[box(0, 0, 5, 2)],
        crs=4326,
    )
    return records, footprints, buildings, complexes


def infer(*args):
    return infer_residential_sites(*args, city_slug="test", boundary=box(-1, -1, 10, 3))


def test_complex_identity_accepts_courtyard_and_preserves_unknown_units():
    sites, members, summary = infer(*inputs())
    assert len(sites) == 2
    complex_site = sites.loc[
        sites.site_identity_source.eq("osm_explicit_residential_complex")
    ].iloc[0]
    assert members.loc[members.fd_id.le(4), "site_id"].eq(complex_site.site_id).all()
    assert pd.isna(
        complex_site.residential_units
    )  # courtyard estimate has no supported component scope
    assert sites.residential_units.isna().all()
    assert members.fd_id.is_unique
    assert not members.operational_units_counted.any()
    assert summary["complex_courtyard_or_other_non_msft_records"] == 1
    assert sites.location_lon.between(0, 9).all()
    assert sites.residential_unit_band.eq("unknown").all()


def test_nonoverlapping_component_units_counted_once_without_building_scale():
    records, fp, ob, complexes = inputs()
    records = records.loc[records.fd_id.ne(4)]
    sites, members, _ = infer(records, fp, ob, complexes)
    site = sites.loc[sites.site_identity_source.eq("osm_explicit_residential_complex")].iloc[0]
    assert site.residential_units == 30
    assert site.nsi_modeled_unit_source_sum == 40
    assert members.counted_residential_units.sum() == 30
    assert members.loc[members.operational_units_counted, "nsi_component_id"].is_unique
    assert not {"building_count", "building_area", "site_area"} & set(sites.columns)


def test_different_source_components_in_one_roof_have_unknown_units():
    records, fp, ob, complexes = inputs()
    records = records.loc[records.fd_id.le(2)].copy()
    records.loc[records.fd_id.eq(2), "ftprntid"] = "different"
    sites, members, _ = infer(records, fp, ob, complexes.iloc[:0])
    assert len(sites) == 1
    assert pd.isna(sites.iloc[0].residential_units)
    assert not members.operational_units_counted.any()


def test_competing_complexes_fall_back_to_marked_building_scope():
    records, fp, ob, complexes = inputs()
    other = complexes.copy()
    other["osm_id"] = "w5"
    other.geometry = [box(-0.1, -0.1, 5.1, 2.1)]
    sites, _, _ = infer(records, fp, ob, pd.concat([complexes, other], ignore_index=True))
    assert not sites.site_identity_source.eq("osm_explicit_residential_complex").any()
    assert sites.site_scope_ambiguous.any()


def test_duplicate_nsi_record_id_is_rejected():
    records, fp, ob, complexes = inputs()
    with pytest.raises(ValueError, match="fd_id"):
        infer(pd.concat([records, records.iloc[:1]], ignore_index=True), fp, ob, complexes)


def test_site_validator_checks_component_owner_sum():
    from evrptw_cle.residential_sites import validate_site_tables

    records, fp, ob, complexes = inputs()
    sites, members, _ = infer(records.loc[records.fd_id.ne(4)], fp, ob, complexes)
    assert validate_site_tables(sites, members)["passed"]
    members.loc[members.operational_units_counted, "counted_residential_units"] = 99
    with pytest.raises(ValueError, match="Residential units differ"):
        validate_site_tables(sites, members)
