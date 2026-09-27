"""Loaders and validators for ``config/params.yaml`` and ``config/topology.yaml``.

``params.yaml`` holds every engineering number the bench uses. Each leaf is a
:class:`Param` with ``value``, ``unit``, ``source`` and ``confidence``; a bare
number anywhere in the file is a validation error. Unknown values are the
string ``"TODO"``.

``topology.yaml`` describes the pressurised volume as a graph: compartments
are nodes with a volume, hatches and ducts are edges with a flow capacity and
a transit time. :meth:`Topology.to_graph` turns it into a NetworkX multigraph.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal, Union

import networkx as nx
import yaml
from pydantic import Field, RootModel, field_validator, model_validator

from eio_crewbench.api.schema import (
    CompartmentId,
    Identifier,
    NonNegFloat,
    PosFloat,
    StrictModel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
PARAMS_PATH = CONFIG_DIR / "params.yaml"
TOPOLOGY_PATH = CONFIG_DIR / "topology.yaml"

TODO = "TODO"
PARAM_FIELDS: frozenset[str] = frozenset({"value", "unit", "source", "confidence"})


# --------------------------------------------------------------------------- #
# params.yaml
# --------------------------------------------------------------------------- #


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class Param(StrictModel):
    """One numeric parameter with its provenance."""

    value: float | Literal["TODO"]
    unit: str = Field(min_length=1)
    source: str = Field(min_length=1)
    confidence: Confidence

    @field_validator("value", mode="before")
    @classmethod
    def _no_bool(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("value must be a number or 'TODO', not a bool")
        return v

    @property
    def is_todo(self) -> bool:
        return self.value == TODO

    @property
    def number(self) -> float:
        """The numeric value; raises if still TODO."""
        if self.is_todo:
            raise ValueError("parameter value is TODO")
        return float(self.value)


class ParamGroup(RootModel[dict[str, Union[Param, "ParamGroup"]]]):
    """A nested mapping of parameter groups ending in :class:`Param` leaves."""

    @model_validator(mode="before")
    @classmethod
    def _reject_bare_leaves(cls, data: object) -> object:
        if not isinstance(data, dict):
            raise ValueError("a parameter group must be a mapping")
        for key, node in data.items():
            if isinstance(node, dict):
                continue
            raise ValueError(
                f"'{key}' is a bare value; every parameter must be a "
                "{value, unit, source, confidence} mapping"
            )
        return data

    def flatten(self, prefix: str = "") -> dict[str, Param]:
        """Return ``{"group.sub.name": Param}`` for every leaf."""
        out: dict[str, Param] = {}
        for key, node in self.root.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(node, Param):
                out[path] = node
            else:
                out.update(node.flatten(path))
        return out

    def __getitem__(self, key: str) -> Param | ParamGroup:
        return self.root[key]


ParamGroup.model_rebuild()


def load_params(path: Path | str = PARAMS_PATH) -> ParamGroup:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ParamGroup.model_validate(raw)


# --------------------------------------------------------------------------- #
# topology.yaml
# --------------------------------------------------------------------------- #


class EdgeKind(StrEnum):
    HATCH = "hatch"
    DUCT = "duct"


class CompartmentSpec(StrictModel):
    volume_m3: PosFloat
    description: str = ""
    source: str = ""


class EdgeSpec(StrictModel):
    edge_id: Identifier
    kind: EdgeKind
    a: CompartmentId
    b: CompartmentId
    capacity_m3_per_s: NonNegFloat = Field(
        description="Volumetric exchange capacity when fully open / at rated speed."
    )
    transit_time_s: NonNegFloat = Field(
        description="Time for gas (duct) or a person (hatch) to cross the edge."
    )
    source: str = ""

    @model_validator(mode="after")
    def _distinct_ends(self) -> EdgeSpec:
        if self.a == self.b:
            raise ValueError(f"edge '{self.edge_id}' connects '{self.a.value}' to itself")
        return self


class Topology(StrictModel):
    name: str = Field(min_length=1)
    compartments: dict[CompartmentId, CompartmentSpec]
    edges: list[EdgeSpec]

    @model_validator(mode="after")
    def _complete_and_unique(self) -> Topology:
        missing = set(CompartmentId) - set(self.compartments)
        if missing:
            names = ", ".join(sorted(m.value for m in missing))
            raise ValueError(f"topology is missing compartments: {names}")
        ids = [e.edge_id for e in self.edges]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate edge ids: {', '.join(dupes)}")
        return self

    def to_graph(self) -> nx.MultiGraph:
        """Undirected multigraph: one node per compartment, one edge per hatch/duct."""
        g = nx.MultiGraph(name=self.name)
        for cid, spec in self.compartments.items():
            g.add_node(cid.value, volume_m3=spec.volume_m3, description=spec.description)
        for e in self.edges:
            g.add_edge(
                e.a.value,
                e.b.value,
                key=e.edge_id,
                kind=e.kind.value,
                capacity_m3_per_s=e.capacity_m3_per_s,
                transit_time_s=e.transit_time_s,
            )
        return g

    def is_connected(self) -> bool:
        return nx.is_connected(self.to_graph())


def load_topology(path: Path | str = TOPOLOGY_PATH) -> Topology:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Topology.model_validate(raw)


__all__ = [
    "CONFIG_DIR",
    "PARAMS_PATH",
    "TOPOLOGY_PATH",
    "TODO",
    "PARAM_FIELDS",
    "Confidence",
    "Param",
    "ParamGroup",
    "load_params",
    "EdgeKind",
    "CompartmentSpec",
    "EdgeSpec",
    "Topology",
    "load_topology",
]
