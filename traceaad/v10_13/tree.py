"""Evaluated algorithms and their formation links."""

from dataclasses import dataclass


@dataclass
class Node:
    """One evaluated algorithm in the evolutionary archive."""

    id: int
    code: str
    idea: str
    fitness: float
    evaluation_id: int | None = None
    parent_id: int | None = None
    operator: str = "Init"
    donor_id: int | None = None
    attempts: int = 0


class SearchTree:
    """Keep the full evaluated archive and the parent links used for context."""

    def __init__(self):
        self.nodes = {}
        self.roots = []
        self.next_id = 0

    def _attach(self, node):
        self.nodes[node.id] = node
        if node.parent_id is None and node.id not in self.roots:
            self.roots.append(node.id)

    def add(self, *, code, idea, fitness, evaluation_id, parent_id, operator,
            donor_id=None, attempts=0):
        node = Node(self.next_id, code, idea, fitness, evaluation_id, parent_id,
                    operator, donor_id, attempts)
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
    def parent_selections(self):
        return sum(node.attempts for node in self.nodes.values())
