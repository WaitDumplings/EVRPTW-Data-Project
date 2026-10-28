"""National charging-location identity, quarantine and member preservation."""

from __future__ import annotations

import hashlib
import json
import re

import geopandas as gpd
import pandas as pd

LOCATION_POLICY = "afdc_shared_coordinate_complete_address_v1"
ALIASES = dict(
    zip(
        [
            "street",
            "avenue",
            "boulevard",
            "road",
            "drive",
            "lane",
            "court",
            "parkway",
            "place",
            "highway",
            "north",
            "south",
            "east",
            "west",
        ],
        ["st", "ave", "blvd", "rd", "dr", "ln", "ct", "pkwy", "pl", "hwy", "n", "s", "e", "w"],
        strict=True,
    )
)


def text(value):
    return "" if value is None or pd.isna(value) else str(value).strip()


def site_address(row):
    words = re.sub(r"[^a-z0-9 ]", " ", text(row.get("Street Address")).lower()).split()
    words = [ALIASES.get(word, word) for word in words]
    state = text(row.get("State")).upper()
    postal = re.sub(r"[^0-9]", "", text(row.get("ZIP")).split(".")[0])[:5]
    if (
        len(words) < 2
        or not any(c.isdigit() for c in words[0])
        or len(state) != 2
        or len(postal) != 5
    ):
        return ""
    return "|".join((" ".join(words), state, postal))


def _text_conflict(row):
    name = text(row.get("Station Name")).lower()
    access = text(row.get("Access Days Time")).lower()
    if re.search(
        r"\b(stations? removed|permanently closed|not (?:yet )?available yet|not yet available)\b",
        name + " " + access,
    ):
        return "source_status_text_conflict"
    detail = text(row.get("Access Detail Code")).upper()
    if detail in {"FLEET", "GOVERNMENT", "RESIDENTIAL"} or re.search(
        r"\b(residents?|employees?|tenants?|fleet|government) only\b", name + " " + access
    ):
        return "source_public_access_text_conflict"
    # A structured pipe-separated street address in the title must not name a
    # different house number from the AFDC address. Asset numbers elsewhere do not count.
    street_num = re.match(r"\s*(\d+)\b", text(row.get("Street Address")))
    embedded = [re.match(r"\s*(\d+)\s+[a-z]", part, re.IGNORECASE) for part in name.split("|")[1:]]
    if street_num and any(m and m.group(1) != street_num.group(1) for m in embedded):
        return "source_name_address_conflict"
    return ""


def assign_charging_locations(frame):
    """Exact native coordinate plus full street/state/ZIP; never radius-cluster.

    A coordinate reused by different or incomplete addresses quarantines all its
    records. Unknown addresses remain independent singleton identities.
    """
    result = frame.copy().reset_index(drop=True)
    result["charging_site_address_key"] = result.apply(site_address, axis=1)
    result["location_quarantine_reason"] = result.apply(_text_conflict, axis=1)
    result["charging_location_id"] = ""
    for coordinate, members in result.groupby(
        ["raw_afdc_longitude", "raw_afdc_latitude"], dropna=False, sort=False
    ):
        addresses = members.charging_site_address_key
        collision = len(members) > 1 and (addresses.eq("").any() or addresses.nunique() > 1)
        if collision:
            result.loc[members.index, "location_quarantine_reason"] = (
                "same_coordinate_address_conflict"
            )
        for idx, row in members.iterrows():
            address = row.charging_site_address_key or f"unknown_address_afdc_{int(row.ID)}"
            token = json.dumps([coordinate[0], coordinate[1], address], separators=(",", ":"))
            result.at[idx, "charging_location_id"] = (
                "charging_location_" + hashlib.sha256(token.encode()).hexdigest()[:20]
            )
    result["charging_location_policy"] = LOCATION_POLICY
    return result


def collapse_charging_locations(members):
    """One sampling row per location; keep full original member rows separately.

    Representative legacy equipment fields describe one eligible source record,
    never a sum of possibly overlapping EVSE/port counts. All configurations and
    access conditions remain attributable to their AFDC records.
    """
    members = assign_charging_locations(members)
    members["source_charger_id"] = members["charger_id"]
    members["charger_candidate_eligible"] &= members.location_quarantine_reason.eq("")
    rows = []
    equipment_fields = [
        "ID",
        "EV Network",
        "EV Network Web",
        "EV Connector Types",
        "EV Level1 EVSE Num",
        "EV Level2 EVSE Num",
        "EV DC Fast Count",
        "Access Code",
        "Access Detail Code",
        "Access Days Time",
        "Restricted Access",
        "Maximum Vehicle Class",
        "EV Pricing",
        "reference_charge_mode",
        "station_power_kw",
        "charger_candidate_eligible",
    ]
    for location_id, group in members.groupby("charging_location_id", sort=True):
        # Never expose restricted-member equipment as the public representative.
        ordered = group.assign(_dc=group.reference_charge_mode.eq("dc_fast")).sort_values(
            ["charger_candidate_eligible", "_dc", "ID"],
            ascending=[False, False, True],
            kind="stable",
        )
        representative = ordered.iloc[0].drop(labels="_dc").copy()
        representative["charger_id"] = location_id
        representative["source_member_count"] = len(group)
        representative["source_afdc_ids_json"] = json.dumps(sorted(int(v) for v in group.ID))
        representative["eligible_afdc_ids_json"] = json.dumps(
            sorted(int(v) for v in group.loc[group.charger_candidate_eligible, "ID"])
        )
        configs = group.sort_values("ID").reindex(columns=equipment_fields).astype(object)
        configs = configs.where(pd.notna(configs), None)
        representative["equipment_access_members_json"] = json.dumps(
            configs.to_dict("records"), ensure_ascii=False, default=str
        )
        representative["port_count_scope"] = "representative_source_record_not_location_total"
        representative["location_total_ports"] = None
        representative["charger_candidate_eligible"] = bool(group.charger_candidate_eligible.any())
        rows.append(representative)
    locations = gpd.GeoDataFrame(
        rows,
        columns=list(members.columns)
        + [
            "source_member_count",
            "source_afdc_ids_json",
            "eligible_afdc_ids_json",
            "equipment_access_members_json",
            "port_count_scope",
            "location_total_ports",
        ],
        geometry="geometry",
        crs=members.crs,
    )
    return locations.reset_index(drop=True), members
