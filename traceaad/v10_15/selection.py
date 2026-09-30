"""Quality-only Boltzmann parent sampling and cross-branch reference choice."""

import math
import statistics

from .canonical import similarity, token_set


def probabilities(nodes):
    """Return probabilities aligned with nodes plus beta, ESS and target ESS."""
    if not nodes:
        raise ValueError("no eligible parents")
    values = [float(n["fitness"]) for n in nodes]
    if not all(math.isfinite(q) for q in values):
        raise ValueError("nonfinite quality")
    n = len(nodes)
    target = min(n, max(2.0, 0.1 * n))
    maximum = max(values)
    top = [i for i, q in enumerate(values) if q == maximum]
    if len(top) == n:
        return [1 / n] * n, 0.0, float(n), target
    if len(top) >= target:
        p = [1 / len(top) if i in top else 0.0 for i in range(n)]
        return p, math.inf, float(len(top)), target

    def weighted(beta):
        weights = [math.exp(beta * (q - maximum)) for q in values]
        total = math.fsum(weights)
        p = [w / total for w in weights]
        return p, 1 / math.fsum(x * x for x in p)

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
    p, beta, ess, target = probabilities(nodes)
    index = rng.choices(range(len(nodes)), weights=p, k=1)[0]
    return nodes[index], {"beta": beta if math.isfinite(beta) else "inf",
                          "ess": ess, "target_ess": target, "probability": p[index],
                          "eligible": len(nodes), "top_tie_limit": not math.isfinite(beta)}


def ancestor_ids(node, archive):
    result = set()
    parent_id = node["parent_id"]
    while parent_id is not None:
        if parent_id in result:
            raise ValueError("cyclic lineage")
        result.add(parent_id)
        parent_id = archive[parent_id]["parent_id"]
    return result


def choose_reference(parent, archive, rng):
    population = list(archive.values())
    median = statistics.median(n["fitness"] for n in population)
    eligible = [n for n in population if n["id"] != parent["id"] and
                n["key"] != parent["key"] and n["fitness"] >= median]
    ancestry = ancestor_ids(parent, archive)
    unrelated = [n for n in eligible if n["id"] not in ancestry and
                 parent["id"] not in ancestor_ids(n, archive)]
    pool = unrelated or eligible
    if not pool:
        return None, {"relaxed_lineage": bool(eligible), "eligible": 0}
    scores = {n["id"]: similarity(parent["code"], n["code"]) for n in pool}
    cutoff = statistics.median(scores.values())
    diverse = [n for n in pool if scores[n["id"]] <= cutoff]
    picked = rng.choice(diverse)
    return picked, {"relaxed_lineage": not bool(unrelated), "eligible": len(pool),
                    "diverse": len(diverse), "similarity": scores[picked["id"]],
                    "similarity_median": cutoff}


def choose_explore_references(parent, archive, rng, count=4):
    """Select distinct idea cards using code diversity as a fallible proxy."""
    def normalize(idea):
        return " ".join((idea or "").split())[:300].casefold()
    parent_idea = normalize(parent.get("idea"))
    ancestry = ancestor_ids(parent, archive)
    eligible = [n for n in archive.values() if n["id"] != parent["id"] and
                n["key"] != parent["key"] and normalize(n.get("idea")) and
                normalize(n.get("idea")) != parent_idea]
    unrelated = [n for n in eligible if n["id"] not in ancestry and
                 parent["id"] not in ancestor_ids(n, archive)]
    pool = unrelated or eligible
    # Prefer the best representative of each identical visible idea or program.
    ideas, keys, unique = set(), set(), []
    for node in sorted(pool, key=lambda n: (-n["fitness"], n["id"])):
        idea = normalize(node["idea"])
        if idea not in ideas and node["key"] not in keys:
            ideas.add(idea)
            keys.add(node["key"])
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
        quality = max(n["fitness"] for n in diverse)
        picked = rng.choice([n for n in diverse if n["fitness"] == quality])
        chosen.append(picked)
        similarities.append(scores[picked["id"]])
        unique.remove(picked)
    return chosen, {"relaxed_lineage": bool(eligible) and not bool(unrelated),
                    "eligible": len(pool), "selected_ids": [n["id"] for n in chosen],
                    "maximum_similarities": similarities}
