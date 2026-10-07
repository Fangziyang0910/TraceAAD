"""Boltzmann parent sampling over programs and reference choice.

Every evaluated program is its own candidate. Equal training scores do not
make programs equivalent; a tie may simply be a change that did not improve.
"""

import math

from traceaad.common.canonical import token_set
from traceaad.common.selection import choose_reference

# Effective number of programs the parent distribution spreads over
# (V10.13's quality ESS).
TARGET_ESS = 8.0


def probabilities(values):
    """Probabilities over programs plus beta, ESS and target ESS."""
    if not values:
        raise ValueError("no eligible parents")
    if not all(math.isfinite(q) for q in values):
        raise ValueError("nonfinite quality")
    n = len(values)
    target = min(float(n), TARGET_ESS)
    if n == 1:
        return [1.0], 0.0, 1.0, target
    minimum = min(values)

    def weighted(beta):
        weights = [math.exp(-beta * (q - minimum)) for q in values]
        total = math.fsum(weights)
        p = [w / total for w in weights]
        return p, 1 / math.fsum(x * x for x in p)

    if weighted(0.0)[1] <= target:
        p, ess = weighted(0.0)
        return p, 0.0, ess, target
    # ESS falls with beta towards the number of programs tied at the minimum;
    # when that many already reach the target, the limit is uniform over them
    # (beta is unbounded and recorded as None).
    top = sum(q == minimum for q in values)
    if top >= target:
        return [1 / top if q == minimum else 0.0 for q in values], None, float(top), target
    lo, hi = 0.0, 1.0
    while weighted(hi)[1] > target:
        hi *= 2
        if not math.isfinite(hi):
            raise ArithmeticError("could not bracket ESS")
    for _ in range(80):
        mid = (lo + hi) / 2
        if weighted(mid)[1] > target:
            lo = mid
        else:
            hi = mid
    p, ess = weighted(hi)
    return p, hi, ess, target


def sample_parent(nodes, rng):
    """Draw one program with probability rising as its training fitness decreases."""
    nodes = sorted(nodes, key=lambda n: n["id"])
    p, beta, ess, target = probabilities([n["fitness"] for n in nodes])
    index = rng.choices(range(len(nodes)), weights=p, k=1)[0]
    return nodes[index], {"beta": beta, "ess": ess, "target_ess": target,
                          "probability": p[index], "eligible": len(nodes)}




def choose_explore_references(parent, archive, rng, count=4):
    """Select distinct idea cards using code diversity as a fallible proxy."""
    def normalize(idea):
        return " ".join((idea or "").split()).casefold()
    parent_idea = normalize(parent.get("idea"))
    pool = [n for n in archive.values() if n["id"] != parent["id"] and
            n["key"] != parent["key"] and normalize(n.get("idea")) and
            normalize(n.get("idea")) != parent_idea]
    # The best program per visible idea: reworded copies would fill several cards.
    ideas, unique = set(), []
    for node in sorted(pool, key=lambda n: (n["fitness"], n["id"])):
        idea = normalize(node["idea"])
        if idea not in ideas:
            ideas.add(idea)
            unique.append(node)
    tokens = {n["id"]: token_set(n["code"]) for n in [parent, *unique]}

    def overlap(left, right):
        a, b = tokens[left["id"]], tokens[right["id"]]
        return len(a & b) / len(a | b) if a or b else 1.0

    chosen, similarities = [], []
    while unique and len(chosen) < count:
        scores = {n["id"]: max(overlap(n, other)
                                for other in [parent, *chosen]) for n in unique}
        minimum = min(scores.values())
        diverse = [n for n in unique if scores[n["id"]] == minimum]
        quality = min(n["fitness"] for n in diverse)
        picked = rng.choice([n for n in diverse if n["fitness"] == quality])
        chosen.append(picked)
        similarities.append(scores[picked["id"]])
        unique.remove(picked)
    return chosen, {"eligible": len(pool), "selected_ids": [n["id"] for n in chosen],
                    "maximum_similarities": similarities}
