#!/usr/bin/env python3
"""Optional real-graph replay against an external verified upstream bundle.

No source data is distributed here and the external bundle is never modified.
This validates the ported pair consumer, not a full-cohort regeneration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd

from evrptw_cle.delivery_access import flag
from evrptw_cle.terminal_pair import materialize_local_access_pair
from evrptw_cle.util import sha256_file, write_json


def run(bundle, output):
    city = bundle / "cities/dallas"
    graph_path = city / "graph/graph_operational.graphml"
    cp = city / "service_locations/latent_locations.parquet"
    dp = city / "infrastructure/depots.parquet"
    hashes = {
        "graph": sha256_file(graph_path),
        "customer_table": sha256_file(cp),
        "depot_table": sha256_file(dp),
    }
    expected = json.loads((bundle / "dallas_review/verification.json").read_text())
    house = pd.read_parquet(
        cp, filters=[("site_id", "==", "cus_site_dallas_965519749df086b58772")]
    ).iloc[0]
    depot = pd.read_parquet(
        dp, filters=[("latent_service_location_id", "==", "dallas-depot-w1035864837")]
    ).iloc[0]
    graph = ox.load_graphml(graph_path, node_dtypes={"osmid": str}, edge_dtypes={"osmid": str})
    before = len(graph), graph.number_of_edges()
    records = []
    for source, destination in [(house, depot), (depot, house)]:
        pair, report = materialize_local_access_pair(
            graph=graph,
            source=source,
            destination=destination,
            connector_speed_kph=5,
            terminal_speed_kph=10,
        )
        path = nx.shortest_path(
            pair, report["source_node"], report["destination_node"], weight="length"
        )
        edges = [
            min(pair[u][v].values(), key=lambda d: float(d["length"]))
            for u, v in zip(path[:-1], path[1:])
        ]
        distance = sum(float(d["length"]) for d in edges)
        reference = next(
            x
            for x in expected["pair_witnesses"]
            if x["source"] == source.latent_service_location_id
            and x["destination"] == destination.latent_service_location_id
        )
        assert abs(distance - reference["real_road_and_connector_distance_m"]) < 1e-6
        terminals = [d for d in edges if d.get("real_osm_terminal_segment")]
        assert terminals and any(flag(d.get("delivery_authorization_assumed")) for d in terminals)
        assert all(
            not flag(d.get("delivery_authorization_assumed"))
            for d in edges
            if not d.get("real_osm_terminal_segment")
        )
        assert (
            pair.in_degree(report["source_node"]) == 0
            and pair.out_degree(report["destination_node"]) == 0
        )
        records.append(
            dict(
                source=source.latent_service_location_id,
                destination=destination.latent_service_location_id,
                distance_m=distance,
                authorized_terminal_segments=sum(
                    flag(d.get("delivery_authorization_assumed")) for d in terminals
                ),
            )
        )
        del pair
    assert before == (len(graph), graph.number_of_edges())
    assert hashes == {
        "graph": sha256_file(graph_path),
        "customer_table": sha256_file(cp),
        "depot_table": sha256_file(dp),
    }
    result = dict(
        passed=True,
        scope="ported pair consumer on two external production OD witnesses; not complete regeneration",
        external_verification_sha256=sha256_file(bundle / "verification.json"),
        input_sha256=hashes,
        external_inputs_unchanged=True,
        source_egress_destination_ingress_only=True,
        test_only_speeds_kph={"connector": 5, "terminal": 10},
        records=records,
    )
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.bundle, args.output)
