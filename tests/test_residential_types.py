from evrptw_cle.residential_types import (
    AMBIGUOUS_TYPE,
    UNKNOWN_TYPE,
    classify_nsi_occtypes,
)


def test_multiple_single_family_records_do_not_become_an_apartment():
    result = classify_nsi_occtypes(["RES1-1SNB", "RES1-2SWB", "RES1-1SNB"])
    assert result.service_location_type == "house"
    assert result.type_label_resolved and not result.type_conflict
    assert result.coarse_residential_type == "single_family"


def test_multifamily_type_does_not_require_a_unit_count():
    assert classify_nsi_occtypes(["RES3A"]).service_location_type == "small_apt"
    assert classify_nsi_occtypes(["RES3C", "RES3D"]).service_location_type == "medium_apt"
    assert classify_nsi_occtypes(["RES3E", "RES3F"]).service_location_type == "large_apt"


def test_residential_type_conflict_is_retained_without_majority_vote():
    result = classify_nsi_occtypes(["RES1-1SNB"] * 20 + ["RES3A"])
    assert result.service_location_type == AMBIGUOUS_TYPE
    assert result.type_conflict and result.coarse_type_conflict
    assert not result.type_label_resolved
    assert result.nsi_type_candidates == ("house", "small_apt")


def test_coarse_multifamily_can_be_known_when_size_is_ambiguous():
    result = classify_nsi_occtypes(["RES3A", "RES3F"])
    assert result.service_location_type == AMBIGUOUS_TYPE
    assert result.type_conflict and not result.coarse_type_conflict
    assert result.coarse_residential_type == "multi_family"


def test_manufactured_housing_remains_a_separate_source_type():
    assert classify_nsi_occtypes(["RES2"]).service_location_type == "manufactured_home"
    assert classify_nsi_occtypes(["RES2", "RES1-1SNB"]).type_conflict


def test_missing_or_unrecognized_codes_never_default_to_house():
    for codes in [None, [], "", '["RES3A", null]', '["COM1"]', "[broken"]:
        result = classify_nsi_occtypes(codes)
        assert result.service_location_type == UNKNOWN_TYPE
        assert not result.type_label_resolved


def test_code_order_and_existing_json_representation_do_not_change_labels():
    assert classify_nsi_occtypes('["RES3D", "RES3C"]') == classify_nsi_occtypes(
        ("res3c", " RES3D ")
    )
