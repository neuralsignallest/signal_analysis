"""config/topology.yaml loads into a connected NetworkX graph of four compartments."""

from __future__ import annotations

import networkx as nx
import pytest
from pydantic import ValidationError

from eio_crewbench.api.schema import CompartmentId
from eio_crewbench.config import (
    TOPOLOGY_PATH,
    CompartmentSpec,
    EdgeKind,
    EdgeSpec,
    Topology,
    load_topology,
)


@pytest.fixture(scope="module")
def topology() -> Topology:
    return load_topology()


def test_topology_file_exists() -> None:
    assert TOPOLOGY_PATH.is_file()


def test_four_compartments(topology: Topology) -> None:
    assert set(topology.compartments) == set(CompartmentId)
    assert len(topology.compartments) == 4


def test_volumes_are_positive_m3(topology: Topology) -> None:
    for cid, spec in topology.compartments.items():
        assert spec.volume_m3 > 0, cid


def test_loads_into_connected_networkx_graph(topology: Topology) -> None:
    g = topology.to_graph()
    assert isinstance(g, nx.MultiGraph)
    assert set(g.nodes) == {c.value for c in CompartmentId}
    assert nx.is_connected(g)
    assert topology.is_connected()


def test_graph_carries_volumes_and_edge_attributes(topology: Topology) -> None:
    g = topology.to_graph()
    for node, data in g.nodes(data=True):
        assert data["volume_m3"] == topology.compartments[CompartmentId(node)].volume_m3
    assert g.number_of_edges() == len(topology.edges)
    for _, _, key, data in g.edges(keys=True, data=True):
        assert data["kind"] in {k.value for k in EdgeKind}
        assert data["capacity_m3_per_s"] >= 0
        assert data["transit_time_s"] >= 0
        assert any(e.edge_id == key for e in topology.edges)


def test_has_both_hatches_and_ducts(topology: Topology) -> None:
    kinds = {e.kind for e in topology.edges}
    assert kinds == {EdgeKind.HATCH, EdgeKind.DUCT}


def test_every_compartment_reachable_by_hatch(topology: Topology) -> None:
    """Crew must be able to walk between any two compartments with hatches alone."""
    g = topology.to_graph()
    hatches = nx.MultiGraph()
    hatches.add_nodes_from(g.nodes)
    hatches.add_edges_from((a, b) for a, b, d in g.edges(data=True) if d["kind"] == "hatch")
    assert nx.is_connected(hatches)


def test_topology_round_trips_through_json(topology: Topology) -> None:
    back = Topology.model_validate_json(topology.model_dump_json())
    assert back == topology
    assert nx.utils.graphs_equal(back.to_graph(), topology.to_graph())


def _spec() -> CompartmentSpec:
    return CompartmentSpec(volume_m3=10.0)


def test_disconnected_topology_is_detected() -> None:
    topo = Topology(
        name="split",
        compartments={c: _spec() for c in CompartmentId},
        edges=[
            EdgeSpec(
                edge_id="only",
                kind=EdgeKind.HATCH,
                a=CompartmentId.HABITATION,
                b=CompartmentId.NODE,
                capacity_m3_per_s=1.0,
                transit_time_s=1.0,
            )
        ],
    )
    assert not topo.is_connected()


def test_missing_compartment_is_rejected() -> None:
    with pytest.raises(ValidationError, match="missing compartments"):
        Topology(name="x", compartments={CompartmentId.LAB: _spec()}, edges=[])


def test_self_loop_edge_is_rejected() -> None:
    with pytest.raises(ValidationError, match="to itself"):
        EdgeSpec(
            edge_id="loop",
            kind=EdgeKind.DUCT,
            a=CompartmentId.LAB,
            b=CompartmentId.LAB,
            capacity_m3_per_s=0.0,
            transit_time_s=0.0,
        )


def test_duplicate_edge_ids_are_rejected() -> None:
    edge = EdgeSpec(
        edge_id="dup",
        kind=EdgeKind.DUCT,
        a=CompartmentId.LAB,
        b=CompartmentId.NODE,
        capacity_m3_per_s=0.0,
        transit_time_s=0.0,
    )
    with pytest.raises(ValidationError, match="duplicate edge ids"):
        Topology(name="x", compartments={c: _spec() for c in CompartmentId}, edges=[edge, edge])


def test_unknown_compartment_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EdgeSpec.model_validate(
            {
                "edge_id": "e",
                "kind": "duct",
                "a": "lab",
                "b": "engine_room",
                "capacity_m3_per_s": 0.0,
                "transit_time_s": 0.0,
            }
        )
