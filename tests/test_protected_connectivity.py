import json

import networkx as nx
import pandas as pd

from evrptw_cle.protected_connectivity import (
    build_directed_component_index,
    label_projection_connectivity,
    projection_reference_access,
    projection_scc_id,
)


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_edge("a", "b", key="ab")
    graph.add_edge("b", "a", key="ba")
    graph.add_edge("b", "c", key="bc")
    return graph


def _offsets(u: str, v: str, fraction: float) -> str:
    return json.dumps(
        [
            {
                "u": u,
                "v": v,
                "key": "0",
                "projection_fraction_from_u": fraction,
            }
        ]
    )


def test_projection_inherits_reference_scc_only_for_roundtrip_edge() -> None:
    index = build_directed_component_index(_graph())
    assert projection_scc_id(_offsets("a", "b", 0.5), index) == "S0001"
    assert projection_scc_id(_offsets("b", "c", 0.5), index) == "S_UNRESOLVED"
    assert projection_scc_id(_offsets("b", "c", 0.0), index) == "S0001"
    assert projection_scc_id(_offsets("b", "c", 1.0), index) != "S0001"


def test_endpoint_projection_requires_both_stage2_access_directions() -> None:
    index = build_directed_component_index(_graph())
    assert projection_reference_access(_offsets("a", "b", 0.5), index) == (
        True,
        True,
    )
    assert projection_reference_access(_offsets("b", "c", 0.0), index) == (
        True,
        False,
    )
    assert projection_reference_access(_offsets("b", "c", 1.0), index) == (
        True,
        False,
    )


def test_label_projection_connectivity_retains_and_quarantines() -> None:
    index = build_directed_component_index(_graph())
    frame = pd.DataFrame(
        {
            "directed_projection_offsets": [
                _offsets("a", "b", 0.5),
                _offsets("b", "c", 0.5),
                _offsets("b", "c", 0.0),
            ]
        }
    )
    result = label_projection_connectivity(frame, index)
    assert result["protected_inbound_access_eligible"].tolist() == [True, True, True]
    assert result["protected_outbound_access_eligible"].tolist() == [True, False, False]
    assert result["protected_roundtrip_eligible"].tolist() == [True, False, False]
    assert result["protected_roundtrip_status"].tolist() == [
        "passed_reference_scc_directional_access",
        "quarantine_no_reference_scc_outbound_access",
        "quarantine_no_reference_scc_outbound_access",
    ]
