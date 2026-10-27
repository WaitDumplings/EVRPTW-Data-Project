from __future__ import annotations

import pandas as pd


def _module():
    from evrptw_cle import depot_candidates

    return depot_candidates


def test_warehouse_area_is_a_flag_not_a_hard_filter() -> None:
    module = _module()
    row = pd.Series({"building": "warehouse", "name": "Unnamed warehouse"})
    assert module._evidence_tier(row, 1_500.0, 1_000.0)[0] == "B_warehouse_proxy"
    assert module._evidence_tier(row, 1_500.0, 2_000.0)[0] == "B_warehouse_proxy"


def test_unrelated_open_ended_tags_do_not_promote_candidates() -> None:
    module = _module()
    spaceflight = pd.Series(
        {"building": "industrial", "logistics": "spaceflight", "name": "TransAstra"}
    )
    taxi = pd.Series({"depot": "taxi", "name": "Taxi depot"})
    assert module._evidence_tier(spaceflight, 5_000.0, 1_000.0)[0] == "C_industrial_proxy"
    assert not module._is_matched_object(taxi)


def test_named_carrier_facility_is_tier_a() -> None:
    module = _module()
    row = pd.Series(
        {
            "name": "Amazon Delivery Station",
            "operator": "Amazon",
            "building": "warehouse",
        }
    )
    assert module._evidence_tier(row, 10_000.0, 1_000.0)[0] == "A_osm_explicit"


def test_carrier_retail_counter_is_rejected() -> None:
    module = _module()
    row = pd.Series(
        {
            "name": "The UPS Store",
            "operator": "UPS",
            "amenity": "post_depot",
        }
    )
    tier, reason, _ = module._evidence_tier(row, 0.0, 1_000.0)
    assert tier == "C_industrial_proxy"
    assert "retail counter" in reason


def test_carrier_point_requires_dispatch_facility_evidence() -> None:
    module = _module()
    row = pd.Series(
        {
            "operator": "FedEx",
            "amenity": "post_depot",
        }
    )
    tier, reason, _ = module._evidence_tier(row, 0.0, 1_000.0)
    assert tier == "C_industrial_proxy"
    assert "physical dispatch-facility evidence" in reason


def test_noncarrier_named_logistics_facility_remains_optional() -> None:
    module = _module()
    row = pd.Series(
        {
            "name": "SDSU Logistics",
            "amenity": "post_depot",
            "building": "yes",
        }
    )
    assert module._evidence_tier(row, 2_179.0, 1_000.0)[0] == "B_warehouse_proxy"


def test_office_only_remains_b_even_for_known_carrier():
    module = _module()
    row = pd.Series(
        {"name": "FedEx Regional Headquarters", "office": "logistics", "operator": "FedEx"}
    )
    assert module._evidence_tier(row, 0, 1000)[0] == "B_warehouse_proxy"
    row["logistics"] = "spaceflight"
    assert module._evidence_tier(row, 0, 1000)[0] == "C_industrial_proxy"
