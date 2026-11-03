"""Where the next attempt starts: training quality weighted by the experience of each program.

Each valid program is a candidate. Its weight combines current training
quality with observed attempts that produced a better new program:

    exp(-beta * (q_i - q_min)) * (k_i + s * r) / (n_i + s)

n_i counts attempts from program i, k_i counts improvements, r is the run's
pooled improvement rate, and s = 3 is the retained prior strength. Historical
failure-count groups motivated this setting; they do not establish its net
search benefit or make it a predictor of long-term development value.

beta only converts score differences into weights: it is the temperature at
which the score values observed so far spread over an effective number of 8. Fitting it on the observed values rather than on
programs keeps it finite when many programs share the top score.
"""

import math
import statistics
from collections import Counter
from scipy.optimize import brentq
from .config import DEFAULT_OPERATOR_WEIGHTS

TARGET_ESS = 8.0
PRIOR_STRENGTH = 3.0


def better(child, parent):
    """Strictly better training fitness, up to the evaluation's floating-point noise."""
    return parent - child > 1e-09 * max(1.0, abs(parent))


def temperature(values):
    """beta at which the distinct observed values have an ESS of min(levels, 8)."""
    levels = sorted(set(values))
    if not levels:
        raise ValueError("no eligible parents")
    if not all((math.isfinite(q) for q in levels)):
        raise ValueError("nonfinite quality")
    target = min(float(len(levels)), TARGET_ESS)
    if len(levels) <= TARGET_ESS:
        return (0.0, float(len(levels)))
    minimum = levels[0]

    def ess(beta):
        weights = [math.exp(-beta * (q - minimum)) for q in levels]
        total = math.fsum(weights)
        return 1 / math.fsum(((w / total) ** 2 for w in weights))

    lo, hi = (0.0, 1.0)
    while ess(hi) > target:
        hi *= 2
        if not math.isfinite(hi):
            raise ArithmeticError("could not bracket ESS")
    beta = brentq(lambda value: ess(value) - target, lo, hi)
    return beta, ess(beta)


def experience(programs):
    """Direct generated children and improvements, crediting a repair to its original child."""
    repairs = {p.repair_of: p for p in programs.values() if p.repair_of is not None}
    tried, improved = (Counter(), Counter())
    for child in programs.values():
        if child.repair_of is not None or child.action not in DEFAULT_OPERATOR_WEIGHTS:
            continue
        parent = programs.get(child.parent_id)
        if parent is None or not parent.valid:
            continue
        tried[parent.id] += 1
        result = repairs.get(child.id, child)
        if result.valid and better(result.fitness, parent.fitness):
            improved[parent.id] += 1
    return (tried, improved)


def weights(nodes, programs):
    """Selection weights over ``nodes`` and what produced them."""
    beta, levels_ess = temperature([n.fitness for n in nodes])
    tried, improved = experience(programs)
    rate = (sum(improved.values()) + 1) / (sum(tried.values()) + 2)
    minimum = min((n.fitness for n in nodes))
    prospect = {
        n.id: (improved[n.id] + PRIOR_STRENGTH * rate) / (tried[n.id] + PRIOR_STRENGTH)
        for n in nodes
    }
    raw = [math.exp(-beta * (n.fitness - minimum)) * prospect[n.id] for n in nodes]
    total = math.fsum(raw)
    return (
        [w / total for w in raw],
        {
            "beta": beta,
            "levels_ess": levels_ess,
            "rate": rate,
            "tried": tried,
            "improved": improved,
            "prospect": prospect,
        },
    )


def sample_parent(nodes, programs, rng):
    nodes = sorted(nodes, key=lambda n: n.id)
    probabilities, selection_stats = weights(nodes, programs)
    index = rng.choices(range(len(nodes)), weights=probabilities, k=1)[0]
    chosen = nodes[index].id
    return (
        nodes[index],
        {
            "beta": selection_stats["beta"],
            "levels_ess": selection_stats["levels_ess"],
            "improvement_rate": selection_stats["rate"],
            "tried": selection_stats["tried"][chosen],
            "improved": selection_stats["improved"][chosen],
            "prospect": selection_stats["prospect"][chosen],
            "probability": probabilities[index],
            "ess": 1 / math.fsum((x * x for x in probabilities)),
            "eligible": len(nodes),
        },
    )


def choose_reference(parent, archive, rng):
    population = list(archive.values())
    median = statistics.median(n.fitness for n in population)
    pool = [n for n in population if n.id != parent.id and n.fitness <= median]
    return (rng.choice(pool) if pool else None), {"eligible": len(pool)}
