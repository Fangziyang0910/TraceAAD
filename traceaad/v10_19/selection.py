"""Where the next attempt starts: training quality weighted by the experience of each program.

An expert keeps developing a strong design while attempts on it keep paying
off, and moves on once many attempts have failed. Each valid program is its
own candidate. Its weight is

    exp(beta * (q_i - q_max)) * (k_i + s * r) / (n_i + s)

where n_i attempts have started from program i and k_i of them produced a new
program better than it, r is the run's pooled rate of such improvements and
s = 3 is the prior strength. In V10.15-5/-6 the share of attempts that beat
their starting program fell from about 18-21% on a program's first attempt to
about 9% after one to four failures, 3-6% after five to fourteen and 1% after
fifteen or more, on every task; s = 3 reproduces that decay.

V10.19: an Explore attempt is one change to its starting program. When the
change was made to work in further steps, its outcome is the best version it
reached, so a change that ended up better than the program counts as a success.

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
OPERATORS = ("Refine", "Explore", "Crossover", "Develop")


def better(child, parent):
    """Strictly better training fitness, up to the evaluation's floating-point noise."""
    return child - parent > 1e-9 * max(1.0, abs(parent))


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
    maximum = levels[-1]

    def ess(beta):
        weights = [math.exp(beta * (q - maximum)) for q in levels]
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


def outcome(attempt, attempts, programs, closed=None, repairs=None):
    """The valid program an attempt produced (its repair's, if it had one), or None.

    An Explore proposal whose change was made to work in further steps ends in
    the best version the change reached; ``closed`` maps proposal attempts to
    their closed explorations (``by_proposal``).
    """
    if attempt["action"] == "Explore" and closed:
        record = closed.get(attempt["id"])
        if record is not None and record.get("best_id") is not None:
            return programs.get(record["best_id"])
    if repairs is None:
        repairs = {a["repair_of"]: a for a in attempts.values() if a.get("repair_of") is not None}
    final = repairs.get(attempt["id"], attempt)
    result = programs.get(final.get("program_id"))
    return result if final.get("status") == "valid" and result is not None and result["valid"] else None


def by_proposal(explorations):
    """Closed explorations by the attempt that proposed them."""
    return {e["proposal_attempt"]: e for e in (explorations or {}).values()}


def experience(attempts, programs, explorations=None):
    """Attempts started from each valid program and how many produced a better new program.

    A repair belongs to the attempt it repairs: that attempt succeeded when
    the repaired program is new and better than where the attempt started.
    """
    repairs = {a["repair_of"]: a for a in attempts.values() if a.get("repair_of") is not None}
    closed = by_proposal(explorations)
    tried, improved = Counter(), Counter()
    for attempt in attempts.values():
        if attempt.get("repair_of") is not None or attempt["action"] not in OPERATORS:
            continue
        source = programs.get(attempt["parent_id"])
        if source is None or not source["valid"]:
            continue
        tried[source["id"]] += 1
        result = outcome(attempt, attempts, programs, closed, repairs)
        if result is not None and result["id"] != source["id"] and better(result["fitness"], source["fitness"]):
            improved[source["id"]] += 1
    return tried, improved


def weights(nodes, attempts, programs, explorations=None):
    """Selection weights over ``nodes`` and what produced them."""
    beta, levels_ess = temperature([n["fitness"] for n in nodes])
    tried, improved = experience(attempts, programs, explorations)
    rate = (sum(improved.values()) + 1) / (sum(tried.values()) + 2)
    maximum = max(n["fitness"] for n in nodes)
    prospect = {n["id"]: (improved[n["id"]] + PRIOR_STRENGTH * rate) / (tried[n["id"]] + PRIOR_STRENGTH)
                for n in nodes}
    raw = [math.exp(beta * (n["fitness"] - maximum)) * prospect[n["id"]] for n in nodes]
    total = math.fsum(raw)
    return [w / total for w in raw], {"beta": beta, "levels_ess": levels_ess, "rate": rate,
                                      "tried": tried, "improved": improved, "prospect": prospect}


def sample_parent(nodes, attempts, programs, rng, explorations=None):
    nodes = sorted(nodes, key=lambda n: n["id"])
    p, parts = weights(nodes, attempts, programs, explorations)
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
            n["key"] != parent["key"] and n["fitness"] >= median]
    if not pool:
        return None, {"eligible": 0}
    scores = {n["id"]: similarity(parent["code"], n["code"]) for n in pool}
    cutoff = statistics.median(scores.values())
    diverse = [n for n in pool if scores[n["id"]] <= cutoff]
    picked = rng.choice(diverse)
    return picked, {"eligible": len(pool), "diverse": len(diverse),
                    "similarity": scores[picked["id"]], "similarity_median": cutoff}


__all__ = ["better", "temperature", "outcome", "by_proposal", "experience", "weights", "sample_parent", "choose_reference"]
