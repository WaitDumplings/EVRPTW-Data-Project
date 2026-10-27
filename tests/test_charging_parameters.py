import numpy as np
import pandas as pd

from evrptw_cle.charging_parameters import select_reference_configurations

MEDIANS = {"ac_level2": 6.5, "dc_fast": 240.0}


def fixture():
    members = pd.DataFrame(
        [
            dict(
                ID=i,
                charging_location_id="loc",
                charger_candidate_eligible=True,
                has_j1772=ac,
                has_ccs1=dc,
                l2_ports=int(ac),
                dc_fast_ports=int(dc),
                raw_afdc_longitude=-74.0,
                raw_afdc_latitude=40.0,
                **{
                    "Street Address": "1 Main St",
                    "State": "NY",
                    "ZIP": "10001",
                    "Access Days Time": "Business hours",
                    "Access Detail Code": "CUSTOMERS",
                    "Maximum Vehicle Class": "LD",
                },
            )
            for i, ac, dc in [(1, True, False), (2, True, True)]
        ]
    )
    locations = pd.DataFrame(
        [
            dict(
                charger_id="loc",
                charging_location_id="loc",
                charger_candidate_eligible=True,
                **{"Maximum Vehicle Class": "LD"},
            )
        ]
    )
    units = pd.DataFrame(
        [
            dict(
                ID=i,
                Longitude=-74.0,
                Latitude=40.0,
                **{
                    "Street Address": "1 Main St",
                    "State": "NY",
                    "ZIP": "10001",
                    "Restricted Access": False,
                    "EV J1772 Connector Count": ac,
                    "EV CCS Connector Count": dc,
                    "EV J1772 Power Output (kW)": ap,
                    "EV CCS Power Output (kW)": dp,
                },
            )
            for i, ac, dc, ap, dp in [(1, 1, 0, 7.2, np.nan), (2, 1, 1, 50.0, 80.0)]
        ]
    )
    return locations, members, units


def test_reported_compatible_configuration_keeps_access_and_never_sums():
    locations, members, units = fixture()
    result, _, configs, summary = select_reference_configurations(
        locations, members, units, MEDIANS
    )
    assert len(result) == 1 and result.iloc[0].station_power_kw == 80
    assert result.iloc[0].reference_charge_mode == "dc_fast"
    assert result.iloc[0]["Access Detail Code"] == "CUSTOMERS"
    assert result.iloc[0].reference_vehicle_class_status == "weight_category_not_excluded"
    assert configs.power_qa_flag.sum() == 1 and configs.is_selected.sum() == 1
    assert summary["selected_reported_power"] == 1
    assert result.iloc[0].capped_pre_derating_kw == 80


def test_j1772_ac_is_available_even_if_same_member_has_incompatible_dc():
    locations, members, units = fixture()
    members = members.iloc[1:].copy()
    members["has_ccs1"] = False
    units = units.loc[units.ID.eq(2)].copy()
    units["EV J1772 Power Output (kW)"] = np.nan
    result, _, configs, summary = select_reference_configurations(
        locations, members, units, MEDIANS
    )
    assert result.reference_vehicle_eligible.all()
    assert result.iloc[0].reference_charge_mode == "ac_level2"
    assert pd.isna(result.iloc[0].station_power_kw)
    assert result.iloc[0].assigned_power_kw == 6.5
    assert configs.connector.eq("J1772").all()
    assert summary["selected_imputed_power"] == 1


def test_incompatible_location_is_retained_but_not_vehicle_eligible():
    locations, members, units = fixture()
    members["has_j1772"] = False
    members["has_ccs1"] = False
    result, _, configs, summary = select_reference_configurations(
        locations, members, units, MEDIANS
    )
    assert result.charger_candidate_eligible.all() and not result.reference_vehicle_eligible.any()
    assert configs.empty and summary["incompatible_locations"] == 1


def test_conflicting_unit_address_is_quarantined_without_power_fallback():
    locations, members, units = fixture()
    units["Street Address"] = "other address"
    result, _, _, summary = select_reference_configurations(locations, members, units, MEDIANS)
    assert not result.reference_vehicle_eligible.any()
    assert summary["unit_source_binding_conflicts"] == 2
