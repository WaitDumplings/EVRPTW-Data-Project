from evrptw_cle.terminal_access import classify_terminal_access_way


def test_nonprivate_service_way_is_terminal_only_core_candidate() -> None:
    result = classify_terminal_access_way(
        {"highway": "service", "service": "parking_aisle", "access": "customers"}
    )
    assert result["terminal_candidate"]
    assert result["terminal_core_eligible"]
    assert result["legal_access_tier"] == "destination_or_customers"


def test_private_driveway_is_retained_only_for_permit_sensitivity() -> None:
    result = classify_terminal_access_way(
        {"highway": "service", "service": "driveway", "access": "private"}
    )
    assert result["terminal_candidate"]
    assert not result["terminal_core_eligible"]
    assert result["legal_access_tier"] == "permit_required"


def test_specific_motorcar_yes_overrides_generic_private_access() -> None:
    result = classify_terminal_access_way(
        {"highway": "service", "access": "private", "motorcar": "yes"}
    )
    assert result["terminal_candidate"]
    assert result["terminal_core_eligible"]
    assert result["legal_access_source_tag"] == "motorcar"


def test_prohibited_and_emergency_service_ways_are_excluded() -> None:
    prohibited = classify_terminal_access_way({"highway": "service", "motor_vehicle": "no"})
    emergency = classify_terminal_access_way({"highway": "service", "service": "emergency_access"})
    assert not prohibited["terminal_candidate"]
    assert not emergency["terminal_candidate"]


def test_public_residential_road_stays_in_operational_graph() -> None:
    result = classify_terminal_access_way({"highway": "residential"})
    assert not result["terminal_candidate"]
    assert result["terminal_reason"] == "already_operational_or_not_terminal"


def test_unknown_access_and_malformed_service_value_do_not_enter_default_core() -> None:
    result = classify_terminal_access_way(
        {"highway": "service", "service": "Baek Court", "access": "agricultural"}
    )
    assert result["terminal_candidate"]
    assert not result["terminal_core_eligible"]
    assert result["legal_access_tier"] == "unresolved_tag_value"
    assert result["connector_kind"] == "service_other"
