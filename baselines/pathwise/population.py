from __future__ import annotations

import copy
import math
from threading import Lock
from typing import Iterable

from .graph import PathWiseNode
from core import Function


class Population:
    def __init__(self, pop_size: int, generation: int = 0, nodes: list[PathWiseNode] | None = None):
        self._pop_size = pop_size
        self._generation = generation
        self._nodes = list(nodes or [])
        self._lock = Lock()

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, item) -> PathWiseNode:
        return self._nodes[item]

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def nodes(self) -> list[PathWiseNode]:
        return self._nodes

    @property
    def population(self) -> list[Function]:
        return [node.function for node in self._nodes]

    @staticmethod
    def is_valid_score(score) -> bool:
        if score is None:
            return False
        try:
            return math.isfinite(float(score))
        except (TypeError, ValueError):
            return False

    def valid_nodes(self) -> list[PathWiseNode]:
        return [node for node in self._nodes if self.is_valid_score(node.score)]

    def _unique_ranked(self, nodes: Iterable[PathWiseNode]) -> list[PathWiseNode]:
        unique = []
        seen_code = set()
        seen_score = set()
        for node in sorted(nodes, key=lambda n: n.score, reverse=True):
            if not self.is_valid_score(node.score):
                continue
            code_key = str(node.function)
            score_key = float(node.score)
            if code_key in seen_code or score_key in seen_score:
                continue
            seen_code.add(code_key)
            seen_score.add(score_key)
            unique.append(copy.deepcopy(node))
        return unique

    def set_nodes(self, nodes: list[PathWiseNode], *, increment_generation: bool = True):
        with self._lock:
            self._nodes = [
                copy.deepcopy(node)
                for node in nodes
                if self.is_valid_score(node.score)
            ][:self._pop_size]
            if increment_generation:
                self._generation += 1

    def extend(self, nodes: list[PathWiseNode]) -> list[PathWiseNode]:
        with self._lock:
            merged = self._unique_ranked([*self._nodes, *nodes])[:self._pop_size]
            before = {node.node_id for node in self._nodes}
            self._nodes = merged
            return [node for node in merged if node.node_id not in before]

