"""Where the next attempt starts: training quality weighted by the experience of each program.

Each valid program is a candidate. Its weight combines current training
quality with observed attempts that produced a better new program:

    exp(-beta * (q_i - q_min)) * (k_i + s * r) / (n_i + s)

n_i counts attempts from program i, k_i counts improvements, r is the run's
pooled improvement rate, and s = 3 is the retained prior strength. Historical
failure-count groups motivated this setting; they do not establish its net
search benefit or make it a predictor of long-term development value.

beta only converts score differences into weights: it is the temperature at
which the score values observed so far spread over an effective number of 8
(V10.13's quality ESS). Fitting it on the observed values rather than on
programs keeps it finite when many programs share the top score.
"""

import math
import statistics
from collections import Counter

from .canonical import similarity

TARGET_ESS = 8.0
PRIOR_STRENGTH = 3.0
OPERATORS = ("Refine", "Explore", "Crossover", "Develop", "Deepen")


def better(child, parent):
    """Strictly better training fitness, up to the evaluation's floating-point noise."""
    return parent - child > 1e-9 * max(1.0, abs(parent))


def temperature(values):
    """beta at which the distinct observed values have an ESS of min(levels, 8)."""
    levels = sorted(set(values))
    if not levels:
        raise ValueError("no eligible parents")
    if not all(math.isfinite(q) for q in levels):
        raise ValueError("nonfinite quality")
    target = min(float(len(levels)), TARGET_ESS)
    if len(levels) <= TARGET_ESS:
        return 0.0, float(len(levels))
    minimum = levels[0]

    def ess(beta):
        weights = [math.exp(-beta * (q - minimum)) for q in levels]
        total = math.fsum(weights)
        return 1 / math.fsum((w / total) ** 2 for w in weights)

    lo, hi = 0.0, 1.0
    while ess(hi) > target:
        hi *= 2
        if not math.isfinite(hi):
            raise ArithmeticError("could not bracket ESS")
    for _ in range(80):
        mid = (lo + hi) / 2
        if ess(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi, ess(hi)


def experience(attempts, programs):
    """Attempts started from each valid program and how many produced a better new program.

    A repair belongs to the attempt it repairs: that attempt succeeded when
    the repaired program is new and better than where the attempt started.
    """
    repairs = {a["repair_of"]: a for a in attempts.values() if a.get("repair_of") is not None}
    tried, improved = Counter(), Counter()
    for attempt in attempts.values():
        if attempt.get("repair_of") is not None or attempt["action"] not in OPERATORS:
            continue
        source = programs.get(attempt["parent_id"])
        if source is None or not source["valid"]:
            continue
        tried[source["id"]] += 1
        final = repairs.get(attempt["id"], attempt)
        result = programs.get(final.get("program_id"))
        if (final.get("status") == "valid" and result is not None
                and better(result["fitness"], source["fitness"])):
            improved[source["id"]] += 1
    return tried, improved


def weights(nodes, attempts, programs):
    """Selection weights over ``nodes`` and what produced them."""
    beta, levels_ess = temperature([n["fitness"] for n in nodes])
    tried, improved = experience(attempts, programs)
    rate = (sum(improved.values()) + 1) / (sum(tried.values()) + 2)
    minimum = min(n["fitness"] for n in nodes)
    prospect = {n["id"]: (improved[n["id"]] + PRIOR_STRENGTH * rate) / (tried[n["id"]] + PRIOR_STRENGTH)
                for n in nodes}
    raw = [math.exp(-beta * (n["fitness"] - minimum)) * prospect[n["id"]] for n in nodes]
    total = math.fsum(raw)
    return [w / total for w in raw], {"beta": beta, "levels_ess": levels_ess, "rate": rate,
                                      "tried": tried, "improved": improved, "prospect": prospect}


def sample_parent(nodes, attempts, programs, rng):
    nodes = sorted(nodes, key=lambda n: n["id"])
    p, parts = weights(nodes, attempts, programs)
    index = rng.choices(range(len(nodes)), weights=p, k=1)[0]
    chosen = nodes[index]["id"]
    return nodes[index], {"beta": parts["beta"], "levels_ess": parts["levels_ess"],
                          "improvement_rate": parts["rate"], "tried": parts["tried"][chosen],
                          "improved": parts["improved"][chosen], "prospect": parts["prospect"][chosen],
                          "probability": p[index], "ess": 1 / math.fsum(x * x for x in p),
                          "eligible": len(nodes)}


def choose_reference(parent, archive, rng):
    population = list(archive.values())
    median = statistics.median(n["fitness"] for n in population)
    pool = [n for n in population if n["id"] != parent["id"] and
            n["key"] != parent["key"] and n["fitness"] <= median]
    if not pool:
        return None, {"eligible": 0}
    scores = {n["id"]: similarity(parent["code"], n["code"]) for n in pool}
    cutoff = statistics.median(scores.values())
    diverse = [n for n in pool if scores[n["id"]] <= cutoff]
    picked = rng.choice(diverse)
    return picked, {"eligible": len(pool), "diverse": len(diverse),
                    "similarity": scores[picked["id"]], "similarity_median": cutoff}


__all__ = ["better", "temperature", "experience", "weights", "sample_parent", "choose_reference"]
