from __future__ import annotations

import copy
import dataclasses
from typing import Dict, List

from core import Function


@dataclasses.dataclass
class ParentInfo:
    node_id: str
    description: str
    score: float


@dataclasses.dataclass
class PathWiseNode:
    function: Function
    rationale: str
    description: str
    score: float
    node_id: str
    parents: list[ParentInfo] = dataclasses.field(default_factory=list)

    def copy(self) -> "PathWiseNode":
        return copy.deepcopy(self)


@dataclasses.dataclass
class PathWiseEdge:
    parents: list[str]
    rationale: str
    child: str


@dataclasses.dataclass
class PathWiseAction:
    parents: list[str]
    rationale: str


class PathWiseGraph:
    def __init__(self):
        self.nodes: Dict[str, PathWiseNode] = {}
        self.edges: List[PathWiseEdge] = []

    def add_node(self, node: PathWiseNode):
        self.nodes[node.node_id] = node

    def add_edge(self, edge: PathWiseEdge):
        self.edges.append(edge)

    def remove_nodes(self, node_ids: list[str]):
        for node_id in node_ids:
            self.nodes.pop(node_id, None)

    def get_state(self) -> list[PathWiseNode]:
        return list(self.nodes.values())

    def parent_ids(self) -> set[str]:
        parents = set()
        for edge in self.edges:
            parents.update(edge.parents)
        return parents
