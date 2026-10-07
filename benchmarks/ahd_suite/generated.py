"""Explicit fixed-scale distributions; these do not claim to reproduce OR-Library."""

import numpy as np
from scipy.optimize import linprog

from . import fssp, graph, set_cover
from .dataset import TASKS
from .templates import FUNCTION_NAMES, TEMPLATES

COUNTS = {task: {"train": 16, "val": 32, "test": 100} for task in TASKS}
COUNTS["mdmkp_search"] = {"train": 18, "val": 36, "test": 108}
SEED = 20261007
RECIPES = {
    "fssp_gls": "50 jobs x 20 machines; independent integer processing times uniform in 1..99; reference is NEH makespan",
    "graph_colouring": "G(300,0.5): each undirected edge independently present with probability 0.5; reference is deterministic DSATUR construction",
    "set_cover_construct": "200 elements x 2000 sets; each element chooses 40 distinct sets uniformly, independently of other elements; costs independently uniform integers 1..100; reference is gain/cost construction plus redundant-set deletion",
    "mdmkp_search": "100 items, 10 upper and 5 lower constraints. Fractions alpha=0.25/0.50/0.75 balanced by base group. A uniformly chosen alpha*100-item witness is planted independently of profits. Each coefficient row is sampled from independent uniform integers 1..1000, conditioned on the witness satisfying its bound floor(alpha*row_sum). Positive profits uniform 1..1000; mixed profits uniform -500..1000. Both profit variants share the base constraints and witness. Reference is a continuous LP relaxation upper bound; it is not an integer optimum.",
}


def heuristic(task):
    namespace = {}
    exec(TEMPLATES[task], namespace)
    return namespace[FUNCTION_NAMES[task]]


def constrained_rows(rng, x, alpha, count, upper):
    rows, limits = [], []
    while len(rows) < count:
        row = rng.integers(1, 1001, len(x), dtype=np.int64)
        limit = int(alpha * int(row.sum()))
        used = int(row @ x)
        accepted = used <= limit if upper else used >= limit
        if accepted:
            rows.append(row)
            limits.append(limit)
    return np.array(rows), np.array(limits)


def lp_bound(data):
    result = linprog(-data["cost_vector"].astype(float),
                     A_ub=np.vstack((data["A_leq"], -data["A_geq"])),
                     b_ub=np.r_[data["b_leq"], -data["b_geq"]], bounds=(0, 1), method="highs")
    if not result.success or -result.fun <= 0:
        raise ValueError(f"could not compute positive MDMKP LP reference: {result.message}")
    return float(-result.fun)


def instances(task, split):
    """Yield independent base groups and their input arrays, dimensions and reference."""
    count = COUNTS[task][split]
    phase = ("train", "val", "test").index(split)
    reference_heuristic = heuristic(task)
    for i in range(count // 2 if task == "mdmkp_search" else count):
        entropy = [SEED, TASKS.index(task), phase, i]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        group = f"generated_{split}_base{i:03d}"
        meta = {"group": group, "seed_entropy": entropy, "source_kind": "generated"}
        if task == "mdmkp_search":
            alpha = (0.25, 0.5, 0.75)[i % 3]
            x = np.zeros(100, dtype=np.int8)
            x[rng.choice(100, int(100*alpha), replace=False)] = 1
            a, b = constrained_rows(rng, x, alpha, 10, True)
            g, d = constrained_rows(rng, x, alpha, 5, False)
            base = {"A_leq": a, "b_leq": b, "A_geq": g, "b_geq": d, "initial_solution": x}
            dim = {"items": 100, "upper_constraints": 10, "lower_constraints": 5}
            for cost_type, low in (("positive", 1), ("mixed", -500)):
                arrays = {**base, "cost_vector": rng.integers(low, 1001, 100, dtype=np.int64)}
                yield arrays, {**meta, "id": f"{group}_{cost_type}", "scale": 100, "dimensions": dim,
                               "tightness": alpha, "cost_type": cost_type}, lp_bound(arrays), "LP relaxation upper bound"
            continue
        if task == "fssp_gls":
            times = rng.integers(1, 100, (50, 20)).astype(float)
            arrays = {"processing_times": times}
            dim = {"jobs": 50, "machines": 20}
            reference = float(fssp.makespan(fssp.neh(times), times))
            kind, scale = "NEH feasible upper bound", 50
        elif task == "graph_colouring":
            upper = np.triu(rng.random((300, 300)) < 0.5, k=1)
            arrays = {"adjacency": upper | upper.T}
            dim = {"vertices": 300}
            _, reference = graph.solve(arrays, reference_heuristic, reduction_attempts=0)
            meta["density"] = float(arrays["adjacency"].sum() / (300*299))
            kind, scale = "DSATUR feasible upper bound", 300
        else:
            coverage = np.zeros((200, 2000), dtype=bool)
            for row in coverage:
                row[rng.choice(2000, 40, replace=False)] = True
            arrays = {"coverage": coverage, "costs": rng.integers(1, 101, 2000).astype(float)}
            dim = {"elements": 200, "sets": 2000}
            _, reference = set_cover.solve(arrays, reference_heuristic, improvement_steps=0)
            meta["density"] = float(coverage.mean())
            kind, scale = "greedy feasible upper bound", 2000
        yield arrays, {**meta, "id": group, "scale": scale, "dimensions": dim}, float(reference), kind
