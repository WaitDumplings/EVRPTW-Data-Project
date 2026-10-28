"""Reference-vehicle configuration selection from source-attributable AFDC units."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .runtime_paths import CONFIG_ROOT
from .charger_locations import site_address
from .charger_resolution import haversine_m
from .public_locations import excluded_location_reason
from .util import sha256_file

PARAMETER_POLICY = "native_compatible_connector_power_v1"


def load_frozen_charging_units():
    contract = json.loads((CONFIG_ROOT / "charging_parameter_contract_v1.json").read_text())
    path = CONFIG_ROOT.parent / contract["unit_source"]["path"]
    if not path.is_file() or sha256_file(path) != contract["unit_source"]["sha256"]:
        raise ValueError("Missing or changed frozen AFDC charging-unit snapshot")
    frame = pd.read_csv(path, low_memory=False, dtype={"ZIP": str})
    for column, value in {
        "Country": "US",
        "Fuel Type Code": "ELEC",
        "Access Code": "public",
        "Status Code": "E",
    }.items():
        if set(frame[column].dropna()) != {value}:
            raise ValueError("Charging-unit snapshot violates its national source scope")
    return frame, contract


def _positive(value):
    try:
        value = float(value)
        return value if np.isfinite(value) and value > 0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def select_reference_configurations(locations, members, units, medians):
    """Choose one eligible member/connector; never combine power or port counts.

    The location pool is preserved. Vehicle eligibility is a separate flag.
    Positive reported compatible power passing QA is preferred; impute only
    when no reported configuration exists at that location.
    """
    result = locations.copy()
    members = members.copy()
    eligible = members.loc[members.charger_candidate_eligible].copy()
    units = units.copy()
    units["unit_source_row"] = np.arange(len(units), dtype=np.int64)
    units = units.loc[units.ID.isin(eligible.ID)]
    by_id = {int(key): part for key, part in units.groupby("ID", sort=False)}
    configs = []
    binding_conflicts = 0
    for _, member in eligible.iterrows():
        station = by_id.get(int(member.ID))
        compatible = []
        if station is not None:
            conflict = False
            for _, unit_metadata in station.iterrows():
                restricted = str(unit_metadata.get("Restricted Access", "")).lower() in {
                    "true",
                    "t",
                    "1",
                }
                try:
                    lon = float(unit_metadata.Longitude)
                    lat = float(unit_metadata.Latitude)
                    coordinate_conflict = (
                        not (np.isfinite(lon) and np.isfinite(lat))
                        or haversine_m(
                            lon,
                            lat,
                            float(member.raw_afdc_longitude),
                            float(member.raw_afdc_latitude),
                        )
                        > 1.0
                    )
                except (TypeError, ValueError):
                    coordinate_conflict = True
                if (
                    site_address(unit_metadata) != site_address(member)
                    or restricted
                    or excluded_location_reason(unit_metadata)
                    or coordinate_conflict
                ):
                    conflict = True
                    break
            if conflict:
                binding_conflicts += 1
                continue
        for mode, connector, power_col, count_col, cap in [
            ("ac_level2", "J1772", "EV J1772 Power Output (kW)", "EV J1772 Connector Count", 11.0),
            ("dc_fast", "J1772COMBO", "EV CCS Power Output (kW)", "EV CCS Connector Count", 100.0),
        ]:
            supported = (
                bool(member.has_j1772 and member.l2_ports > 0)
                if mode == "ac_level2"
                else bool(member.has_ccs1 and member.dc_fast_ports > 0)
            )
            if not supported:
                continue
            unit_rows = (
                station.loc[pd.to_numeric(station[count_col], errors="coerce").gt(0)]
                if station is not None
                else None
            )
            options = (
                [None]
                if unit_rows is None or unit_rows.empty
                else [u for _, u in unit_rows.iterrows()]
            )
            for unit in options:
                raw = _positive(unit[power_col]) if unit is not None else np.nan
                qa = mode == "ac_level2" and pd.notna(raw) and raw > 19.2
                reported = pd.notna(raw) and not qa
                assigned = float(raw) if reported else float(medians[mode])
                if not np.isfinite(assigned) or assigned <= 0:
                    raise ValueError("No positive national compatible-mode power median")
                compatible.append(
                    {
                        "charging_location_id": member.charging_location_id,
                        "afdc_id": int(member.ID),
                        "city_slug": member.get("city_slug", ""),
                        "unit_source_row": int(unit.unit_source_row) if unit is not None else -1,
                        "connector": connector,
                        "reference_charge_mode": mode,
                        "reported_power_kw": raw,
                        "power_qa_flag": bool(qa),
                        "power_is_reported": bool(reported),
                        "power_source": "reported_compatible_connector"
                        if reported
                        else "national_mode_median_after_qa"
                        if qa
                        else "national_mode_median_missing_power",
                        "assigned_power_kw": assigned,
                        "vehicle_cap_kw": cap,
                        "capped_pre_derating_kw": min(assigned, cap),
                        "access_days_time": str(member.get("Access Days Time", "")),
                        "access_detail_code": str(member.get("Access Detail Code", "")),
                        "restricted_public": False,
                        "port_count_scope": "one_source_configuration_not_location_total",
                    }
                )
        configs.extend(compatible)
    columns = [
        "charging_location_id",
        "afdc_id",
        "city_slug",
        "unit_source_row",
        "connector",
        "reference_charge_mode",
        "reported_power_kw",
        "power_qa_flag",
        "power_is_reported",
        "power_source",
        "assigned_power_kw",
        "vehicle_cap_kw",
        "capped_pre_derating_kw",
        "access_days_time",
        "access_detail_code",
        "restricted_public",
        "port_count_scope",
    ]
    configurations = pd.DataFrame(configs, columns=columns)
    configurations["configuration_id"] = [
        f"afdc_{int(r.afdc_id)}:unit_{int(r.unit_source_row)}:{r.connector}"
        for r in configurations.itertuples()
    ]
    chosen = configurations.sort_values(
        [
            "charging_location_id",
            "power_is_reported",
            "capped_pre_derating_kw",
            "afdc_id",
            "unit_source_row",
        ],
        ascending=[True, False, False, True, True],
        kind="stable",
    ).drop_duplicates("charging_location_id")
    result["reference_vehicle_eligible"] = result.charging_location_id.isin(
        chosen.charging_location_id
    )
    result["charging_parameter_policy"] = PARAMETER_POLICY
    result["selected_configuration_id"] = ""
    result["station_power_kw"] = np.nan
    result["station_power_status"] = "no_native_compatible_configuration"
    source = members.set_index("ID", drop=False)
    selection = chosen.set_index("charging_location_id")
    # Rebinding a representative must carry the complete selected source row:
    # raw equipment, access metadata and source IDs must agree with its normalized
    # fields. These fields belong to the collapsed location rather than a member.
    location_fields = {
        "charger_id",
        "charging_location_id",
        "charging_location_policy",
        "charging_site_address_key",
        "charger_candidate_eligible",
        "source_member_count",
        "source_afdc_ids_json",
        "eligible_afdc_ids_json",
        "equipment_access_members_json",
        "port_count_scope",
        "location_total_ports",
    }
    for idx, row in result.loc[result.reference_vehicle_eligible].iterrows():
        cfg = selection.loc[row.charging_location_id]
        member = source.loc[int(cfg.afdc_id)]
        for field in member.index:
            if field not in location_fields:
                result.at[idx, field] = member[field]
        result.at[idx, "reference_charge_mode"] = cfg.reference_charge_mode
        result.at[idx, "reference_vehicle_connector_compatibility"] = (
            "ccs1_present" if cfg.connector == "J1772COMBO" else "j1772_present"
        )
        result.at[idx, "station_power_kw"] = (
            float(cfg.reported_power_kw) if cfg.power_is_reported else np.nan
        )
        result.at[idx, "station_power_status"] = cfg.power_source
        result.at[idx, "selected_configuration_id"] = (
            f"afdc_{int(cfg.afdc_id)}:unit_{int(cfg.unit_source_row)}:{cfg.connector}"
        )
    result["reference_vehicle_class_status"] = np.where(
        result["Maximum Vehicle Class"].fillna("").isin(["LD", "MD", "HD"]),
        "weight_category_not_excluded",
        "unknown",
    )
    for field in ["assigned_power_kw", "vehicle_cap_kw", "capped_pre_derating_kw", "power_qa_flag"]:
        result[field] = result.charging_location_id.map(selection[field])
    result["charging_parameter_eligible"] = (
        result.charger_candidate_eligible & result.reference_vehicle_eligible
    )
    configurations["is_selected"] = configurations.configuration_id.isin(
        result.selected_configuration_id
    )
    if "release_blocker" in result:
        result.loc[result.charging_parameter_eligible, "release_blocker"] = ""
        result.loc[
            result.charger_candidate_eligible & ~result.reference_vehicle_eligible,
            "release_blocker",
        ] = "no_native_compatible_configuration"
    members["reference_vehicle_configuration_available"] = members.ID.isin(configurations.afdc_id)
    summary = {
        "policy": PARAMETER_POLICY,
        "potential_locations": int(result.charger_candidate_eligible.sum()),
        "compatible_locations": int(result.charging_parameter_eligible.sum()),
        "incompatible_locations": int(
            (result.charger_candidate_eligible & ~result.reference_vehicle_eligible).sum()
        ),
        "compatible_configurations": len(configurations),
        "selected_reported_power": int(chosen.power_is_reported.sum()),
        "selected_imputed_power": int((~chosen.power_is_reported).sum()),
        "unit_source_binding_conflicts": binding_conflicts,
        "derating_factor": 0.90,
        "derating_application": "Stage2 applies once; stored effective power remains pre-derating",
        "unknown_vehicle_clearance": "retained as uncertainty; Class2 reference van is not excluded by LD alone",
    }
    return result, members, configurations, summary
