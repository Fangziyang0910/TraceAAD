"""Evaluated algorithms and their parent links."""

from dataclasses import dataclass


@dataclass
class Node:
    """One evaluated algorithm in the evolutionary archive."""

    id: int
    code: str
    idea: str
    fitness: float
    parent_id: int | None = None
    operator: str = "Init"
    reference_id: int | None = None


class SearchTree:
    """Keep the full evaluated archive and the parent links used for context."""

    def __init__(self):
        self.nodes = {}
        self.next_id = 0

    def _attach(self, node):
        self.nodes[node.id] = node

    def add(self, *, code, idea, fitness, parent_id, operator, reference_id=None):
        node = Node(
            self.next_id, code, idea, fitness, parent_id, operator, reference_id,
        )
        self.next_id += 1
        self._attach(node)
        return node

    def add_raw(self, node):
        self._attach(node)
        self.next_id = max(self.next_id, node.id + 1)

    def all_nodes(self):
        return list(self.nodes.values())

    def best(self):
        return max(self.nodes.values(), key=lambda node: node.fitness)

    @property
    def roots(self):
        return [node.id for node in self.nodes.values() if node.parent_id is None]
