"""Independent verification of saved territory-first runs.

Only saved tables, immutable source records and numeric/path certificates are
inspected. No territory, Bernoulli sample, order bundle or route is regenerated.
The legacy verifier supplies shared read-only hash/aggregation/path checks.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import json
import math

import numpy as np
import pandas as pd

from .verify import _RunVerifier, _bool, _ordered_unique, _require, _same, _sha


RUN_SCHEMA = "evrptw_stage2_territory_run_v1"


def _keys(frame):
    return set(frame[["scenario_id", "location_id"]].itertuples(index=False, name=None))


class _TerritoryVerifier(_RunVerifier):
    def inventory(self):
        self.manifest = self.read("manifest.json")
        self.config = self.read("config.json")
        _require(self.manifest["schema"] == RUN_SCHEMA, "Unsupported territory run schema")
        _require(self.config["schema"] == "evrptw_territory_first_config_v1",
                 "Unsupported territory config schema")
        _require(self.manifest["input"] == self.config["input"], "Run/config sources differ")
        files = self.manifest["files"]
        _require(isinstance(files, dict) and files, "Manifest has no file inventory")
        for name, expected in files.items():
            _require(name not in {"manifest.json", "verification.json"},
                     "Self-referential output hash inventory")
            _require(_sha(self.local(name)) == expected, "Changed run artifact: " + name)
        actual = {str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()}
        _require(actual - {"manifest.json", "verification.json"} == set(files),
                 "Run contains missing or unmanifested artifacts")
        _require(set(self.manifest["tables"]) == {"parcels", "operational_days", "views"},
                 "Run table inventory differs")
        for name in [*self.manifest["tables"].values(), *self.manifest["instances"].values()]:
            _require(name in files, "Declared table/instance is not hashed")
        _require(self.manifest.get("stage1_modified") is False, "Run claims Stage 1 mutation")
        _require(self.manifest.get("route_solution_claimed") is False,
                 "Generation cannot claim a route solution")
        self.metrics["hashed_output_file_count"] = len(files)

    def _source_coordinates(self, frame):
        """Read frozen native coordinates, not a new geocoding interpretation."""
        canonical = pd.read_parquet(self.cle / "locations.parquet").set_index("location_id")
        selected = canonical.loc[frame.location_id]
        if {"location_lon", "location_lat"}.issubset(selected.columns):
            return selected[["location_lon", "location_lat"]].to_numpy(float)
        import geopandas as gpd
        result = np.empty((len(selected), 2))
        key = {"customer": "site_id", "depot": "candidate_id", "charging": "charging_location_id"}
        for table in selected.access_table.unique():
            take = selected.access_table.eq(table).to_numpy()
            native = gpd.read_parquet(self.cle / table)
            kind = selected.location_type.iloc[np.flatnonzero(take)[0]]
            native[key[kind]] = native[key[kind]].astype(str)
            native = native.set_index(key[kind]).loc[selected.loc[take, "access_row_id"].astype(str)]
            if {"location_lon", "location_lat"}.issubset(native.columns):
                result[take] = native[["location_lon", "location_lat"]].to_numpy(float)
            else:
                points = native.to_crs(4326).geometry.representative_point()
                result[take] = np.column_stack([points.x, points.y])
        return result

    def _geography(self, frame, *, customer=True):
        if not len(frame):
            return
        for field in ["depot_outward_distance_m", "depot_inward_distance_m"]:
            distance = frame[field].to_numpy(float)
            _require(np.all(np.isfinite(distance) & (distance >= 0)),
                     "Missing/invalid unbounded directed depot reachability")
        xy = self._source_coordinates(frame)
        depot_xy = self._source_coordinates(self.depot)[0]
        if {"location_lon", "location_lat"}.issubset(frame.columns):
            _same(frame[["location_lon", "location_lat"]], xy, "Frozen location coordinates changed")
        lon, lat = np.deg2rad(xy).T
        dlon, dlat = np.deg2rad(depot_xy)
        a = np.sin((lat-dlat)/2)**2 + np.cos(lat)*np.cos(dlat)*np.sin((lon-dlon)/2)**2
        distance = 2*6371.0088*np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        _same(frame.depot_geodesic_distance_km, distance, "Saved great-circle distance differs")
        if self.config["world"].get("service_policy") == "fastest_time_energy":
            if not customer:
                return
            world = self.config["world"]
            vehicle = self.config["physical"]["vehicle"]
            travel = frame.depot_fastest_outward_time_s.to_numpy(float)
            fastest_m = frame.depot_fastest_outward_distance_m.to_numpy(float)
            energy = fastest_m * vehicle["specific_energy_kwh_per_km"] / 1000
            budget = vehicle["battery_capacity_kwh"] * (vehicle["initial_soc"] - vehicle["reserve_soc"])
            _same(frame.depot_fastest_outward_energy_kwh, energy, "Fastest path energy is not bound to its distance")
            _require(np.all(np.isfinite(travel) & (travel >= 0) & (travel <= world["service_limit_s"]+1e-7)) and
                     np.all(np.isfinite(energy) & (energy >= 0) & (energy <= budget+1e-7)),
                     "Location exceeds the one-way time or battery service bound")
        else:
            _require(np.all(distance <= self.config["world"]["external_radius_km"]+1e-7),
                     "Location exceeds the external great-circle bound")

    def world(self):
        self.world_info = self.read("world/world.json")
        self.catalog = self.table("world/catalog.parquet")
        self.depot = self.table("world/depot.parquet")
        _ordered_unique(self.catalog, ["location_id"], "Duplicate world customer identity")
        _require(len(self.depot) == 1 and self.depot.location_id.iloc[0] == self.world_info["depot_id"],
                 "World has no unique canonical depot")
        _require(self.world_info["world_id"] == self.manifest["world_id"], "World ID differs")
        _require(self.world_info["config"] == self.config["world"], "World configuration differs")
        _require(self.world_info["source"]["cle_manifest_sha256"] ==
                 self.manifest["source_sha256"]["cle_manifest"], "World/source CLE differ")
        geography = ("fastest_time_energy_and_finite_return"
                     if self.config["world"].get("service_policy") == "fastest_time_energy"
                     else "great_circle_and_unbounded_bidirectional_reachability")
        _require(self.world_info["geography_policy"] == geography, "Unknown geography policy")
        _require(self.world_info["depot_redraw_count"] == 0 and
                 self.world_info["terminal_through_routing_allowed"] is False,
                 "World allows redraw or terminal through-routing")
        _require(len(self.catalog) == self.world_info["customer_count"], "World count differs")
        _require(self.catalog.world_id.eq(self.manifest["world_id"]).all(), "Catalog world IDs differ")
        for frame, kind in [(self.catalog, "customer"), (self.depot, "depot")]:
            source = self.canonical.loc[frame.location_id]
            _require(source.location_type.eq(kind).all() and _bool(source.candidate_eligible).all(),
                     "World includes an ineligible/wrong canonical location")
        source = self.canonical.loc[self.catalog.location_id]
        _same(self.catalog.residential_units, source.residential_units, "Raw units overwritten")
        units = self.catalog.modeled_residential_units.to_numpy(float)
        _require(np.all(np.isfinite(units) & (units >= 1) & (units == np.floor(units))),
                 "Modeled units are not positive integers")
        city = self.canonical.loc[self.canonical.location_type.eq("customer") &
                                  _bool(self.canonical.candidate_eligible)]
        positive = pd.to_numeric(city.residential_units, errors="coerce").gt(0)
        raw = pd.to_numeric(source.residential_units, errors="coerce").to_numpy(float)
        known = np.isfinite(raw) & (raw > 0)
        if (~known).any():
            general = max(1, math.floor(float(city.loc[positive, "residential_units"].median())+.5))
        else:
            general = 1
        expected = np.where(known, np.maximum(1, np.floor(raw+.5)), general)
        for kind in city.residential_type.dropna().unique():
            if kind == "residential_type_ambiguous":
                continue
            cohort = city.loc[positive & city.residential_type.eq(kind), "residential_units"]
            if len(cohort):
                expected[(~known) & source.residential_type.eq(kind).to_numpy()] = max(
                    1, math.floor(float(cohort.median())+.5))
        _same(units, expected, "Modeled units differ from city/type median policy")
        if "units_imputed" in self.catalog:
            _require(np.array_equal(_bool(self.catalog.units_imputed), ~known), "Imputation flags differ")
        probability = -np.expm1(units*np.log1p(-self.config["world"]["p_unit"]))
        _same(self.catalog.expected_daily_activation_probability, probability,
              "Location activity does not follow its modeled units")
        self._geography(self.catalog)
        self.metrics["world_customer_count"] = len(self.catalog)

    def activation(self):
        self.scenarios = self.table("activation/scenarios.parquet")
        self.territories = self.table("activation/territories.parquet")
        self.selections = self.table("activation/selections.parquet")
        self.view_table = self.table(self.manifest["tables"]["views"])
        _ordered_unique(self.scenarios, ["scenario_id"], "Duplicate scenario identity")
        _ordered_unique(self.scenarios, ["day_index", "requested_n"], "Duplicate day/size scenario")
        _ordered_unique(self.territories, ["scenario_id", "location_id"], "Duplicate territory location")
        _ordered_unique(self.selections, ["scenario_id", "location_id"], "Duplicate selected location")
        _ordered_unique(self.selections, ["scenario_id", "rank"], "Duplicate selection rank")
        _ordered_unique(self.view_table, ["view_id"], "Duplicate view identity")
        ids = set(self.scenarios.scenario_id)
        _require(set(self.view_table.view_id) == ids == set(self.manifest["instances"]),
                 "Scenario/view/instance inventory differs")
        for frame in [self.territories, self.selections]:
            _require(set(frame.scenario_id).issubset(ids) and
                     frame.location_id.isin(self.catalog.location_id).all(),
                     "Activation location outside its scenario/world")
        _require(_keys(self.selections).issubset(_keys(self.territories)),
                 "Selected location outside the saved territory")
        expected_pairs = {(day, size) for day in range(self.config["operations"]["days"])
                          for size in self.config["instances"]["sizes"]}
        _require(set(zip(self.scenarios.day_index, self.scenarios.requested_n)) == expected_pairs,
                 "Requested scenario population was omitted or added")
        catalog = self.catalog.set_index("location_id")
        first = date.fromisoformat(self.config["operations"]["start_date"])
        for row in self.scenarios.itertuples():
            territory = self.territories.loc[self.territories.scenario_id.eq(row.scenario_id)]
            chosen = self.selections.loc[self.selections.scenario_id.eq(row.scenario_id)].sort_values("rank")
            n = int(row.requested_n)
            _require(row.date == (first+timedelta(days=int(row.day_index))).isoformat(), "Scenario date differs")
            _require(bool(row.supported_size) == (int(row.selected_n) == n), "Scenario cardinality status differs")
            _require(len(chosen) == row.selected_n and
                     np.array_equal(chosen["rank"].to_numpy(), np.arange(len(chosen))),
                     "Selected IDs/count/rank differs")
            _require(bool(row.supported_size) or row.selected_n == 0, "Unsupported sampling keeps a partial selection")
            if row.supported_size:
                tract_growth = self.config["activation"].get("method") == "sequential_tract"
                max_centers = len(territory) if tract_growth else self.config["activation"]["max_centers"]
                _require(pd.notna(row.selected_k) and 1 <= row.selected_k <= max_centers
                         and int(row.selected_k) == row.selected_k,
                         "Invalid center count")
                if tract_growth:
                    target = math.ceil(self.config["activation"]["candidate_multiplier"] * n)
                    _require(row.candidate_count == len(territory) and row.candidate_target == target
                             and bool(row.candidate_target_reached) == (len(territory) >= target),
                             "Sequential tract candidate counts differ")
                    _require(catalog.loc[territory.location_id, "bg_geoid"].notna().all(),
                             "Sequential territory uses locations without a tract")
                else:
                    _require(row.reference_length_km > 0 and np.isfinite(row.reference_length_km),
                             "Missing/invalid territory reference length")
                _require(row.activation_attempts >= 0 and
                         int(row.activation_attempts) == row.activation_attempts,
                         "Invalid conditional activation attempt count")
            else:
                _require(not len(territory), "Unsupported sampling exports a territory as selected")
            if len(territory):
                owner = territory.owner.to_numpy(float)
                _require(np.all(np.isfinite(owner) & (owner == np.floor(owner)) &
                                (owner >= 0) & (owner < row.selected_k)), "Invalid territory owner")
                _same(territory.probability,
                      catalog.loc[territory.location_id, "expected_daily_activation_probability"],
                      "Territory probabilities differ from the fixed household model")
                cost = territory.road_cost_km.to_numpy(float)
                _require(np.all(np.isfinite(cost) & (cost >= 0)), "Invalid saved territory road cost")
                if row.supported_size:
                    if len(territory) == n:
                        expected_variance = 0.
                        proposal_mass = float(n)
                    else:
                        _require(pd.notna(row.activation_eta) and np.isfinite(row.activation_eta),
                                 "Nontrivial fixed-N sample has no finite tilt")
                        units = catalog.loc[territory.location_id, "modeled_residential_units"].to_numpy(float)
                        z = -units*np.log1p(-self.config["world"]["p_unit"])
                        theta = z+np.log(-np.expm1(-z))
                        proposal = np.exp(-np.logaddexp(0, -(theta+row.activation_eta)))
                        proposal_mass = float(proposal.sum())
                        expected_variance = float((proposal*(1-proposal)).sum())
                        _require(row.activation_attempts >= 1, "Nontrivial activation has no proposal attempt")
                    _same(proposal_mass, n, "Saved tilt does not center the fixed-N proposal")
                    _same(row.activation_proposal_variance, expected_variance,
                          "Saved conditional proposal variance differs")
            if pd.notna(row.expected_mass):
                _same(row.expected_mass, territory.probability.sum(), "Territory expected-mass diagnostic differs")
            else:
                _require(not len(territory), "Nonempty territory has no mass diagnostic")
            view = self.view_table.set_index("view_id").loc[row.scenario_id]
            for field in ["day_index", "requested_n", "selected_n", "supported_size"]:
                _require(view[field] == getattr(row, field), "View/scenario mismatch: " + field)
            _require(view["kind"] == "territory_day" and
                     view["parent_operational_day_id"] == row.scenario_id,
                     "View has another scenario parent")
        self.metrics.update(scenarios=len(self.scenarios), selected_locations=len(self.selections),
                            sampling_unsupported_scenarios=int((~_bool(self.scenarios.supported_size)).sum()))

    def demand(self):
        self.demand_info = self.read("demand/demand.json")
        self.parcels = self.table(self.manifest["tables"]["parcels"])
        self.sites = self.table("demand/location_dispatch.parquet")
        self.days = self.table(self.manifest["tables"]["operational_days"])
        p, s, op = self.parcels, self.sites, self.config["operations"]
        _ordered_unique(p, ["parcel_id"], "Duplicate parcel ID")
        _ordered_unique(s, ["scenario_id", "location_id"], "Duplicate scenario location workload")
        _ordered_unique(self.days, ["scenario_id"], "Duplicate operational scenario")
        _require(set(self.days.scenario_id) == set(self.scenarios.scenario_id), "Missing operational scenario")
        _require(_keys(s) == _keys(self.selections), "Selected location dropped or workload location substituted")
        _require(_keys(p).issubset(_keys(s)), "Parcel outside selected locations")
        _require(self.demand_info["process_id"] == self.manifest["process_id"] and
                 self.demand_info["world_id"] == self.manifest["world_id"], "Demand parent mismatch")
        _require(self.demand_info["operations"] == op, "Saved operations differ")
        _require(self.demand_info["parcel_count"] == len(p), "Saved parcel count differs")
        _require(self.demand_info["donor_source_sha256"] == self.donor_manifest["source_sha256"],
                 "Demand donor source version differs")
        _require(not _bool(self.days.route_solution_claimed).any(), "Day falsely claims a route solution")
        settings = self.config["workload"]
        catalog = self.catalog.set_index("location_id")
        for row in self.scenarios.itertuples():
            parcels = p.loc[p.scenario_id.eq(row.scenario_id)]
            sites = s.loc[s.scenario_id.eq(row.scenario_id)]
            day = self.days.set_index("scenario_id").loc[row.scenario_id]
            d = int(row.day_index)
            _require((int(day.day_index), day.date, int(day.active_locations)) ==
                     (d, row.date, len(sites)), "Operational day/location count differs")
            _require((day.arrivals, day.dispatched, day.previous_backlog, day.backlog) ==
                     (len(parcels), len(parcels), 0, 0), "Scenario parcel conservation differs")
            for column in ["arrival_day_index", "dispatch_day_index", "scheduled_day_index"]:
                _require(parcels[column].eq(d).all(), "Parcel moved to another scenario day")
            _require(parcels.arrival_date.eq(row.date).all() and
                     not _bool(parcels.commitment_failure).any(), "Parcel date/commitment differs")
            _same(parcels.promise_day_index, np.full(len(parcels), d), "Non-same-day promise")
            _same(parcels.promise_s, np.full(len(parcels), d*86400+op["shift_end_s"]),
                  "Promise completion deadline differs")
            ready = parcels.ready_s.to_numpy(float)-d*86400
            _require(np.all(np.isfinite(ready) & (ready >= op["ready_window_s"][0]-1e-7) &
                            (ready <= op["ready_window_s"][1]+1e-7)), "Ready clock outside declared window")
            self._dispatch_aggregates(parcels, sites, op)
        good = s.loc[_bool(s.generation_supported)]
        bad = s.loc[~_bool(s.generation_supported)]
        _require(good.demand_cm3.gt(0).all() and good.demand_cm3.le(settings["cargo_capacity_cm3"]).all(),
                 "Capacity-invalid successful location bundle")
        _require(bad.parcel_count.eq(0).all() and bad.demand_cm3.isna().all() and
                 bad.planned_service_time_s.isna().all() and
                 (~_bool(bad.supported_one_visit_representation)).all(),
                 "Unsupported bundle was dropped or masquerades as generated")
        _require(not (_keys(p) & _keys(bad)), "Failed location retained an accepted parcel")
        if len(p):
            units = catalog.loc[p.location_id, "modeled_residential_units"].to_numpy(float)
            _same(p.modeled_residential_units, units, "Order changed modeled units")
            active = p.active_units.to_numpy(float)
            _require(np.all(np.isfinite(active) & (active >= 1) & (active <= units) &
                            (active == np.floor(active))), "Active households outside the selected location")
            grouped = p.groupby(["scenario_id", "location_id"], sort=False)
            _require(grouped.active_units.nunique().eq(1).all() and
                     (grouped.size() >= grouped.active_units.first()).all(), "Invalid household/package count")
            for field in ["location_tw_status", "location_tw_start_offset_s",
                          "location_tw_end_offset_s", "location_tw_source_donor_id"]:
                _require(grouped[field].nunique(dropna=False).le(1).all(), "Location has multiple TWs")
        for frame in [p, s]:
            attempts = frame.capacity_conditioning_attempts.to_numpy(float)
            _require(np.all(np.isfinite(attempts) & (attempts >= 1) &
                            (attempts <= settings["maximum_full_proposals"]) &
                            (attempts == np.floor(attempts))), "Invalid capacity proposal count")
        _require(bad.capacity_conditioning_attempts.eq(settings["maximum_full_proposals"]).all(),
                 "Unsupported order stopped before the proposal limit")
        failures = self.table("demand/generation_failures.parquet")
        _ordered_unique(failures, ["scenario_id", "location_id", "proposal_index"],
                        "Repeated rejected order proposal")
        _require(_keys(failures).issubset(_keys(s)), "Rejected proposal outside selected locations")
        _require(failures.proposed_volume_cm3.gt(settings["cargo_capacity_cm3"]).all(),
                 "Feasible proposal incorrectly rejected for volume")
        exhausted = failures.loc[_bool(failures.exhausted)]
        _require(_keys(exhausted) == _keys(bad) and
                 exhausted.proposal_index.eq(settings["maximum_full_proposals"]).all(),
                 "Exhausted location was dropped or invented")
        histories = failures.groupby(["scenario_id", "location_id"]).proposal_index.agg(list).to_dict()
        for key, site in s.set_index(["scenario_id", "location_id"]).iterrows():
            count = int(site.capacity_conditioning_attempts) - int(bool(site.generation_supported))
            _require(np.array_equal(np.sort(histories.get(key, [])), np.arange(1, count+1)),
                     "Capacity rejection history differs from its proposal count")
        self.metrics.update(parcel_count=len(p), location_day_count=len(s),
                            generation_unsupported_location_days=len(bad),
                            capacity_rejected_proposals=len(failures), end_backlog=0)

    def donor_tuples(self):
        p = self.parcels
        source = pd.read_parquet(self.donor_root / "parcels.parquet")
        _ordered_unique(source, ["source_parcel_ref"], "Duplicate donor parcel identity")
        selected = source.set_index("source_parcel_ref").loc[p.source_parcel_ref].reset_index(drop=True)
        _require(selected.source_kind.eq("delivered").all() and
                 _bool(selected.individual_marks_valid).all(), "Unsupported donor parcel layer")
        for field in ["source_event_ref", "source_station_day_id", "source_route_ref",
                      "source_date", "source_tw_status"]:
            _require(np.array_equal(p[field].astype(str), selected[field].astype(str)),
                     "Changed source parcel provenance: " + field)
        for field in ["depth_cm", "height_cm", "width_cm", "volume_cm3", "source_planned_service_s",
                      "source_tw_start_offset_s", "source_tw_end_offset_s"]:
            _same(p[field], selected[field], "Changed joint donor mark: " + field)
        _require(np.array_equal(p.source_donor_id.astype(str), selected.donor_id.astype(str)),
                 "Parcel belongs to a different donor bundle")
        _same(p.volume_cm3, p.depth_cm*p.height_cm*p.width_cm, "Volume differs from source dimensions")
        bundles = pd.read_parquet(self.donor_root / "bundles.parquet").set_index("donor_id")
        _require(bundles.index.is_unique, "Duplicate donor bundle identity")
        groups = p.groupby(["scenario_id", "location_id"], sort=False)
        summary = groups.agg(k=("parcel_id", "size"), donor=("source_donor_id", "first"),
                             donors=("source_donor_id", "nunique"), active=("active_units", "first"))
        _require(summary.donors.eq(1).all(), "One location mixes unrelated donor bundles")
        if len(p):
            donor_counts = p.source_donor_id.map(bundles.package_count).to_numpy(float)
            _same(p.source_package_count, donor_counts, "Donor count metadata changed")
            counts = groups.parcel_id.transform("size").to_numpy(float)
            _require(np.array_equal(_bool(p.exact_source_count), donor_counts == counts),
                     "Exact source count flag differs")
            eligible = bundles.loc[bundles.source_kind.eq("delivered") &
                                   _bool(bundles.individual_marks_valid)]
            _require(p.source_donor_id.isin(eligible.index).all(), "Invalid donor bundle used")
            settings = self.config["workload"]
            def central(count):
                count = np.asarray(count, dtype=float)
                cut = settings["service_support_max_count"]
                base = settings["service_a_s"]+settings["service_b_s"]*count**settings["service_alpha"]
                anchor = settings["service_a_s"]+settings["service_b_s"]*cut**settings["service_alpha"]
                return np.where(count <= cut, base, anchor*(count/cut)**settings["service_tail_gamma"])
            source_service = source.groupby("donor_id").source_planned_service_s.sum()
            source_k = summary.donor.map(bundles.package_count).to_numpy(float)
            residual = summary.donor.map(source_service).to_numpy(float)/central(source_k)*settings["service_scale"]
            reference = np.maximum(1, summary.k.to_numpy(float)/self.config["world"]["mean_parcels_per_active_unit"])
            share = settings["service_active_unit_share"]
            multiplier = (1-share)+share*(summary.active.to_numpy(float)/reference)**settings["service_active_unit_exponent"]
            intercept = settings["service_a_s"]
            expected = residual*(intercept+(central(summary.k)-intercept)*multiplier)
            actual = groups.assigned_planned_service_s.sum().reindex(summary.index)
            _same(actual, expected, "Service differs from source residual/count/household rule")
            windows = p.loc[p.location_tw_status.eq("specified")].drop_duplicates(
                ["scenario_id", "location_id"])
            _require(windows.location_tw_source_donor_id.isin(eligible.index).all(),
                     "Location TW donor is outside the valid pool")
            if len(windows):
                raw = source.copy()
                raw["_a"] = np.where(raw.source_tw_status.eq("specified"), raw.source_tw_start_offset_s, -np.inf)
                raw["_b"] = np.where(raw.source_tw_status.eq("specified"), raw.source_tw_end_offset_s, np.inf)
                patterns = raw.groupby("donor_id").agg(a=("_a", "max"), b=("_b", "min"))
                _same(windows.location_tw_start_offset_s,
                      windows.location_tw_source_donor_id.map(patterns.a), "Location TW start changed its donor")
                _same(windows.location_tw_end_offset_s,
                      windows.location_tw_source_donor_id.map(patterns.b), "Location TW end changed its donor")
        self.metrics["joint_tuple_source_crosscheck"] = True

    def physical(self):
        self.physical_info = self.read("physical/physical.json")
        self.chargers = self.table("physical/chargers.parquet")
        c, config = self.chargers, self.config["physical"]
        _ordered_unique(c, ["location_id"], "Duplicate compatible charging location")
        _require(self.physical_info["config"] == config and
                 self.physical_info["world_id"] == self.manifest["world_id"], "Physical parent differs")
        _require(len(c) == self.physical_info["charger_count"], "Charging count differs")
        if len(c):
            source = self.canonical.loc[c.location_id]
            _require(source.location_type.eq("charging").all() and _bool(source.candidate_eligible).all(),
                     "Ineligible frozen charging location")
            configurations = pd.read_parquet(self.cle / "charging/charging_configurations.parquet")
            configurations = configurations.loc[_bool(configurations.is_selected)].set_index("configuration_id")
            _require(configurations.index.is_unique, "Ambiguous selected charging configuration")
            original = configurations.loc[c.selected_configuration_id].reset_index(drop=True)
            _require(np.array_equal(c.charging_location_id.astype(str), original.charging_location_id.astype(str))
                     and np.array_equal(c.selected_connector.astype(str), original.connector.astype(str)),
                     "Charging location/configuration/connector differs")
            _require(c.selected_connector.isin(["J1772", "J1772COMBO"]).all(), "Incompatible connector")
            _require(c.reference_charge_mode.isin(["dc_fast", "ac_level2"]).all(), "Unknown charging mode")
            _same(c.source_assigned_power_kw, original.assigned_power_kw, "Frozen power changed")
            _same(c.source_reported_power_kw, original.reported_power_kw, "Reported source power changed")
            _require(np.array_equal(_bool(c.source_power_is_reported), _bool(original.power_is_reported)),
                     "Power imputation status changed")
            caps = np.where(c.reference_charge_mode.eq("dc_fast"), config["vehicle"]["dc_cap_kw"],
                            config["vehicle"]["ac_cap_kw"])
            _same(c.stage2_vehicle_cap_kw, caps, "Vehicle charging cap differs")
            _same(c.stage2_capped_pre_derating_kw, np.minimum(c.source_assigned_power_kw, caps), "Power cap differs")
            _same(c.stage2_derating_factor, config["charging"]["derating_factor"], "Derating differs")
            _same(c.effective_power_kw, np.minimum(c.source_assigned_power_kw, caps)*
                  config["charging"]["derating_factor"], "Charging derating is not applied once")
            _require(c.derating_application_count.eq(1).all(), "Repeated charging derating")
            self._geography(c, customer=False)
        self.metrics["compatible_charging_locations"] = len(c)

    def _instance_territory(self, row):
        prefix = "instances/"+str(row.view_id)+"/"
        _require(self.manifest["instances"][row.view_id] == prefix+"instance.json", "Instance path differs")
        info = self.read(prefix+"instance.json")
        terminals = self.table(prefix+"terminals.parquet")
        selected = self.selections.loc[self.selections.scenario_id.eq(row.view_id)].sort_values("rank")
        sites = self.sites.loc[self.sites.scenario_id.eq(row.view_id)].set_index("location_id")
        ids = selected.location_id.astype(str).tolist()
        sites = sites.loc[ids]
        n = len(ids)
        _require(info["schema"] == "evrptw_scenario_instance_v1" and info["view_id"] == row.view_id and
                 info["world_id"] == self.manifest["world_id"] and
                 info["process_id"] == self.manifest["process_id"], "Instance parent identity differs")
        for field in ["day_index", "kind", "requested_n", "selected_n", "supported_size"]:
            _require(info[field] == getattr(row, field), "Instance scenario field differs: "+field)
        _require(info["vehicle"] == self.config["physical"]["vehicle"], "Instance vehicle differs")
        _require(info["assessment"]["route_solution_claimed"] is False, "Instance falsely claims routes")
        _require(info["operating_window_s"] == [self.config["operations"]["shift_start_s"],
                 self.config["operations"]["shift_end_s"]], "Operating window differs")
        selection = self.read(prefix+"charging_selection.json")
        _require(selection == info["charging_selection"], "Charging selection metadata differs")
        charger_ids = selection["selected_location_ids"]
        _require(len(set(charger_ids)) == len(charger_ids) and
                 set(charger_ids).issubset(set(self.chargers.location_id)), "Invented/repeated charging ID")
        expected_ids = [str(self.depot.location_id.iloc[0]), *ids, *charger_ids]
        _require(info["node_order"] == expected_ids == terminals.location_id.astype(str).tolist(),
                 "Instance removed/reordered selected locations")
        _require(np.array_equal(terminals.node_index, np.arange(len(terminals))) and
                 terminals.location_id.is_unique, "Noncanonical instance node identities")
        _require(terminals.kind.tolist() == ["depot"]+["customer"]*n+["charging"]*len(charger_ids),
                 "Instance terminal kinds differ")
        customers = terminals.iloc[1:1+n]
        for target, source in [("demand_cm3", "demand_cm3"), ("service_time_s", "planned_service_time_s"),
                               ("parcel_count", "parcel_count")]:
            _same(customers[target], sites[source], "Instance changed its workload: "+target)
        for field in ["tw_start_s", "tw_end_s"]:
            _same(customers[field], sites[field]-int(row.day_index)*86400, "Instance local TW differs")
        noncustomer = terminals.loc[~terminals.kind.eq("customer")]
        _same(noncustomer.demand_cm3, np.zeros(len(noncustomer)), "Noncustomer demand is nonzero")
        _same(noncustomer.service_time_s, np.zeros(len(noncustomer)), "Noncustomer service is nonzero")
        bad = int((~_bool(sites.generation_supported)).sum())
        bad_tw = int((~_bool(sites.supported_one_visit_representation)).sum())
        oversized = int(sites.demand_cm3.gt(info["vehicle"]["cargo_capacity_cm3"]).sum())
        for field, value in [("unsupported_generation_locations", bad),
                             ("unsupported_one_visit_locations", bad_tw), ("oversize_bundles", oversized)]:
            _require(info["assessment"].get(field, 0) == value, "Instance hides unsupported workload: "+field)
        if info["matrix_materialized"]:
            # Shared legacy checker validates all paired D/T/E arrays and saved
            # path certificates. Scope its old day keys to this one scenario.
            original_sites = self.sites
            self.sites = original_sites.loc[original_sites.scenario_id.eq(row.view_id)]
            self.order = selected.assign(day_index=int(row.day_index))
            try:
                super()._instance(row)
            finally:
                self.sites = original_sites
        else:
            _require(not (self.root / (prefix+"costs.npz")).exists(), "Unmaterialized instance contains dense matrices")
            expected = "unsupported_size" if not row.supported_size else (
                "unsupported_workload" if bad or bad_tw or oversized else "not_materialized")
            _require(info["status"] == expected, "Nonmaterialized instance hides unsupported workload")
            self.instances[row.view_id] = (info, None)

    def instance_group(self):
        for row in self.view_table.itertuples():
            self.stage("instance:"+str(row.view_id), lambda row=row: self._instance_territory(row))
        self.metrics["materialized_instances"] = sum(costs is not None for _, costs in self.instances.values())


def verify_territory_run(path, check_sources=False):
    """Verify a saved territory-day run without recreating generation draws."""
    verifier = _TerritoryVerifier(path, check_sources)
    for name, method in [("output_inventory", verifier.inventory), ("source_bindings", verifier.sources),
                         ("fixed_world", verifier.world), ("territory_and_exact_N", verifier.activation),
                         ("scenario_workload_conservation_and_TW", verifier.demand),
                         ("joint_donor_marks_and_service", verifier.donor_tuples),
                         ("physical_supply_and_power", verifier.physical),
                         ("saved_instances", verifier.instance_group)]:
        verifier.stage(name, method)
    result = {"schema": "evrptw_stage2_verification_v1", "passed": not verifier.errors,
              "verified_at_utc": datetime.now(UTC).isoformat(), "run_id": (verifier.manifest or {}).get("run_id"),
              "manifest_sha256": _sha(verifier.root/"manifest.json"),
              "verifier_source_sha256": _sha(__file__), "checks": verifier.checks,
              "errors": verifier.errors, "warnings": verifier.warnings, "metrics": verifier.metrics,
              "scope": {"saved_structure": True, "scenario_selection_and_workload": True,
                        "paired_path_numeric_and_sampled_provenance": True,
                        "growth_distribution_recreated": False, "all_pair_shortest_path_optimality": False,
                        "complete_city_reachability_recomputed": False, "route_solution_feasibility": False},
              "check_sources": bool(check_sources), "route_solution_claimed": False,
              "unsupported_workloads_repaired": False, "stage1_modified": False}
    (verifier.root/"verification.json").write_text(json.dumps(result, ensure_ascii=False,
        sort_keys=True, indent=2, allow_nan=False)+"\n")
    return result
