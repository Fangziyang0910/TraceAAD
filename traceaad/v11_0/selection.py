"""Fixed V11.0 scheduling rules: code-level scores and archive references."""

import ast
from bisect import bisect_left, bisect_right
from functools import lru_cache

OPERATORS = ("Refine", "Tune", "Pivot", "Fuse")
OPERATOR_PROBABILITIES = {operator: 0.25 for operator in OPERATORS}
EXPLORATION_C = 0.1
N_REFERENCES = 8


@lru_cache(maxsize=8192)
def code_key(code):
    return ast.dump(ast.parse(code), include_attributes=False)


def quality_percentiles(mean_scores):
    """Midrank percentile in [0, 1] across distinct codes; 0.5 for a single code."""
    total = len(mean_scores)
    if total == 1:
        return [0.5]
    ascending = sorted(mean_scores)
    percentiles = []
    for score in mean_scores:
        low, high = bisect_left(ascending, score), bisect_right(ascending, score)
        lower, equal = low, high - low
        percentiles.append((lower + (equal - 1) / 2) / (total - 1))
    return percentiles


def reference_ranks(fitnesses):
    """Midrank over fitness, better fitness first; ties share the average rank."""
    ascending = sorted(fitnesses)
    total = len(fitnesses)
    ranks = []
    for fitness in fitnesses:
        low, high = bisect_left(ascending, fitness), bisect_right(ascending, fitness)
        ranks.append((total - high) + (high - low + 1) / 2)
    return ranks


def reciprocal_rank_sample(nodes, k, rng):
    """Draw up to k nodes without replacement, weighting each node by 1/rank.

    Ranks and weights stay at their original values during the draw; the
    remaining pool is implicitly renormalized at every pick.
    """
    ordered = sorted(nodes, key=lambda node: (-node.fitness, node.id))
    weights = [1.0 / rank for rank in reference_ranks([node.fitness for node in ordered])]
    pool = list(zip(ordered, weights))
    chosen = []
    while len(chosen) < k and pool:
        pick = rng.random() * sum(weight for _, weight in pool)
        index, cumulative = 0, 0.0
        while index < len(pool) - 1:
            cumulative += pool[index][1]
            if pick < cumulative:
                break
            index += 1
        chosen.append(pool.pop(index)[0])
    return chosen


class CodeBook:
    """Scheduling statistics aggregated per distinct AST code key."""

    def __init__(self):
        self.entries = {}

    def register(self, node):
        """Record one valid evaluation; new codes enter with zero attempts."""
        entry = self.entries.setdefault(code_key(node.code),
                                        {"attempts": 0, "scores": [], "node_ids": []})
        entry["scores"].append(node.fitness)
        entry["node_ids"].append(node.id)

    def note_attempt(self, key):
        self.entries[key]["attempts"] += 1

    def keys(self):
        return list(self.entries)

    def attempts(self, key):
        return self.entries[key]["attempts"]

    def mean_fitness(self, key):
        scores = self.entries[key]["scores"]
        return sum(scores) / len(scores)

    def node_ids(self, key):
        return self.entries[key]["node_ids"]

    def attempts_table(self):
        return {key: entry["attempts"] for key, entry in self.entries.items()}

    def restore_attempts(self, table):
        """Rebuild attempt counts from a checkpoint table."""
        for key in self.entries:
            self.entries[key]["attempts"] = table.get(key, 0)
