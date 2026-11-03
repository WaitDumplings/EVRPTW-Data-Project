"""Sequential tract growth with household-weighted fixed-size sampling."""
from datetime import date, timedelta
from math import ceil

import numpy as np
from scipy.special import expit

from .activation import sample_conditional_bernoulli, unit_log_odds
from .territory import contract_empty_units
from .util import identity, rng


def tract_support(catalog, cell_graph):
    """Collapse existing public-road BG connections by their Census tract."""
    tracts = catalog.bg_geoid.str.slice(0, 11)
    cell_to_tract = dict(zip(catalog.loc[tracts.notna(), "cell_id"], tracts.dropna()))
    graph = {}
    for cell, links in cell_graph.items():
        source = cell_to_tract.get(cell, cell)
        adjacent = graph.setdefault(source, set())
        adjacent.update(target for link in links
                        if (target := cell_to_tract.get(link, link)) != source)
    members = {str(tract): np.asarray(rows, dtype=int)
               for tract, rows in tracts.groupby(tracts, sort=True).groups.items()}
    return members, contract_empty_units(graph, members).adjacency


def activate_tracts(context, config, *, day_index, size):
    from .territory_generation import _component_count, _geometry_diagnostics, _json_number

    catalog, world = context.catalog, context.info
    settings = config["activation"]
    day = date.fromisoformat(config["operations"]["start_date"]) + timedelta(days=day_index)
    spatial_id = identity("sequential-tract-v1", world["world_id"], day.isoformat(), size,
                          settings["candidate_multiplier"], settings["opening_probability"])
    theta = unit_log_odds(catalog.modeled_residential_units.to_numpy(), config["world"]["p_unit"])
    probabilities = expit(theta)
    members, adjacency = tract_support(catalog, context.adjacency)
    available = sum(map(len, members.values()))
    target = ceil(settings["candidate_multiplier"] * size)
    effective_target = min(target, available)
    result = {"spatial_id": spatial_id, "date": day.isoformat(), "day_index": day_index,
              "requested_n": size, "supported_size": False, "selected_n": 0,
              "selected_k": None, "expected_mass": None, "reference_length_km": None,
              "activation_eta": None, "activation_attempts": 0,
              "activation_proposal_variance": None, "candidate_count": 0,
              "candidate_target": target, "effective_candidate_target": effective_target,
              "tract_mapped_candidate_count": available,
              "unmapped_tract_count": len(catalog)-available,
              "activation_method": "sequential_tract",
              "parameter_status": settings["parameter_status"], "candidates": []}
    if available < size:
        return {**result, "status": "unsupported_candidate_count"}, [], [], []

    mass = {tract: probabilities[indices].sum() for tract, indices in members.items()}
    draws = rng("sequential-tract-growth", spatial_id)
    remaining, frontier, owners = set(members), set(), {}
    groups, picked, starts, count = [], [], 0, 0
    while remaining and count < effective_target:
        outside = remaining - frontier
        open_probability = settings["opening_probability"] * (1 - count / effective_target)
        open_new = not frontier or (bool(outside) and draws.random() < open_probability)
        options = sorted(outside if open_new else frontier)
        weights = np.asarray([mass[tract] for tract in options])
        tract = str(draws.choice(options, p=weights / weights.sum()))
        owner = starts if open_new else min(owners[t] for t in adjacency[tract] if t in owners)
        starts += int(open_new)
        owners[tract] = owner
        remaining.remove(tract)
        frontier.discard(tract)
        frontier.update(t for t in adjacency[tract] if t in remaining)
        picked.append(tract)
        groups.append(members[tract])
        count += len(members[tract])

    territory = np.concatenate(groups)
    sample = sample_conditional_bernoulli(theta[territory], size, rng("territory-subset", spatial_id))
    selected = np.sort(territory[sample.indices])
    selected_tracts = catalog.bg_geoid.iloc[selected].str.slice(0, 11).tolist()
    d = sample.diagnostics
    result.update(status="activated", supported_size=True, selected_n=len(selected), selected_k=starts,
                  candidate_count=len(territory), expected_mass=float(probabilities[territory].sum()),
                  selected_tract_ids=picked, candidate_target_reached=len(territory) >= target,
                  activation_eta=_json_number(d.eta), activation_attempts=d.attempts,
                  activation_proposal_variance=d.proposal_variance,
                  activation_proposal_budget=d.proposal_budget,
                  territory_cell_components=_component_count(picked, adjacency),
                  activated_cell_components=_component_count(selected_tracts, adjacency),
                  geometry=_geometry_diagnostics(catalog.iloc[selected], context.depot))
    territory_rows = []
    for tract, indices in zip(picked, groups, strict=True):
        for i in indices:
            row = catalog.iloc[i]
            territory_rows.append({"location_id": str(row.location_id), "owner": owners[tract],
                                   "probability": float(probabilities[i]),
                                   "road_cost_km": float(row.depot_outward_distance_m) / 1000})
    selection_rows = [{"location_id": str(catalog.location_id.iloc[i]), "rank": rank}
                      for rank, i in enumerate(selected)]
    return result, selected.tolist(), territory_rows, selection_rows
