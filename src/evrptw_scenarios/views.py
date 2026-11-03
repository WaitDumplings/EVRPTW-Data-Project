"""Nested fixed-location-count research views of a complete operational day."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .util import identity, rng, write_json


def selection_order(day_sites, *, key, sampler):
    sites = day_sites.sort_values("location_id")
    random = rng("view-priority", key, sampler)
    if sampler == "uniform_locations":
        return sites.location_id.astype(str).to_numpy()[random.permutation(len(sites))].tolist()
    if sampler != "community_blocks":
        raise ValueError("Unsupported view sampler")
    groups = list(sites.groupby("community_id", sort=True, dropna=False))
    # Plackett-Luce size-weighted block order, then uniform within each block.
    priorities = [(float(-math.log(max(random.random(), np.finfo(float).tiny)) / len(g)), str(cid), g)
                  for cid, g in groups]
    result = []
    for _, _, group in sorted(priorities, key=lambda x: (x[0], x[1])):
        ids = group.location_id.astype(str).to_numpy()
        result.extend(ids[random.permutation(len(ids))].tolist())
    return result


def build_views(demand_info, config, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sites = pd.read_parquet(demand_info["sites_path"])
    days = pd.read_parquet(demand_info["days_path"])
    cfg = config["views"]
    records, selections = [], []
    for day in days.to_dict("records"):
        index = int(day["day_index"])
        group = sites.loc[sites.day_index.eq(index)]
        key = day["operational_day_id"]
        order = selection_order(group, key=key, sampler=cfg["sampler"])
        natural = identity("natural-day-v1", key)
        records.append({"view_id": natural, "day_index": index, "kind": "operational_day",
                        "requested_n": len(order), "selected_n": len(order), "available_n": len(order),
                        "supported_size": True, "parent_operational_day_id": key,
                        "materialize_matrix": index in cfg["materialize_day_indices"] and len(order) <= cfg["natural_day_dense_matrix_limit"],
                        "sampler": "all_active_locations"})
        for n in cfg["sizes"]:
            vid = identity("benchmark-view-v1", key, cfg["sampler"], n)
            supported = len(order) >= n
            records.append({"view_id": vid, "day_index": index, "kind": "benchmark_view",
                            "requested_n": n, "selected_n": n if supported else 0,
                            "available_n": len(order), "supported_size": supported,
                            "parent_operational_day_id": key,
                            "materialize_matrix": supported and index in cfg["materialize_day_indices"],
                            "sampler": cfg["sampler"]})
        # Single saved priority per day; every fixed-N view is a prefix.
        selections.extend({"day_index": index, "rank": rank, "location_id": loc}
                          for rank, loc in enumerate(order))
    pd.DataFrame(records).to_parquet(out / "views.parquet", index=False)
    pd.DataFrame(selections, columns=["day_index", "rank", "location_id"]).to_parquet(out / "selection_order.parquet", index=False)
    info = {"schema": "evrptw_workload_views_v1", "parent_process_id": demand_info["process_id"],
            "full_site_bundles_preserved": True, "nested_prefixes": True,
            "selection_depends_on_vehicle_or_N": False,
            "unselected_parcels_are_parent_day_workload_not_backlog": True,
            "sampler": cfg["sampler"], "views_path": str(out / "views.parquet"),
            "selection_order_path": str(out / "selection_order.parquet")}
    write_json(out / "views.json", info)
    return info
