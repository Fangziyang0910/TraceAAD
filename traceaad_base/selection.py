"""Boltzmann parent sampling from current fitness, without visit penalties."""

import ast
from functools import lru_cache
import hashlib
import math
import random

from .config import Config
from .models import Program

SELECTION_RULE = "boltzmann_fitness_v1"


@lru_cache(maxsize=None)
def identity(code: str) -> str:
    """Same parsed source, ignoring layout, comments and docstrings."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    node.body = node.body[1:]
        text = ast.dump(tree, include_attributes=False)
    except (SyntaxError, ValueError):
        text = code.strip()
    return hashlib.sha256(text.encode()).hexdigest()


def sources(archive: dict[int, Program]) -> list[Program]:
    """The earliest valid program of each distinct source, best first."""
    earliest = {}
    valid = (p for p in archive.values() if p.valid and p.evaluation.fitness is not None
             and math.isfinite(p.evaluation.fitness))
    for program in sorted(valid, key=lambda p: p.id):
        earliest.setdefault(identity(program.code), program)
    return sorted(earliest.values(), key=lambda p: (p.evaluation.fitness, p.id))


def boltzmann_weights(fitness: dict[int, float], temperature: float) -> dict[int, float]:
    """Unnormalized exp(-fitness / temperature), shifted so the best weight is one."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("selection_temperature must be finite and positive")
    if not fitness:
        return {}
    if not all(math.isfinite(value) for value in fitness.values()):
        raise ValueError("Boltzmann sampling requires finite fitness values")
    best = min(fitness.values())
    result = {}
    for pid, value in fitness.items():
        gap = value - best
        # Scaling first avoids overflow in a gap spanning both finite float extremes.
        scaled = (value / temperature - best / temperature
                  if math.isinf(gap) else gap / temperature)
        result[pid] = math.exp(-scaled)
    return result


def weights(archive: dict[int, Program], temperature: float) -> dict[int, float]:
    return boltzmann_weights(
        {p.id: p.evaluation.fitness for p in sources(archive)}, temperature
    )


def reference_candidates(fitness: dict[int, float]) -> list[int]:
    """Reference eligibility uses current quality only, with ties at the median retained."""
    ordered = sorted(fitness, key=lambda pid: (fitness[pid], pid))
    if not ordered:
        return []
    n = len(ordered)
    median = fitness[ordered[n // 2]]
    if n % 2 == 0:
        median = median / 2 + fitness[ordered[n // 2 - 1]] / 2
    return [pid for pid in ordered if fitness[pid] <= median]


def reference_pool(archive: dict[int, Program], parent_id: int) -> list[Program]:
    """Distinct sources not worse than the median, without the parent's source."""
    candidates = sources(archive)
    if not candidates:
        return []
    eligible = reference_candidates({p.id: p.evaluation.fitness for p in candidates})
    own = identity(archive[parent_id].code)
    return [archive[pid] for pid in eligible if identity(archive[pid].code) != own]


def random_references(fitness: dict[int, float], rng: random.Random) -> dict[int, int | None]:
    eligible = reference_candidates(fitness)
    result = {}
    for pid in sorted(fitness):
        pool = [reference for reference in eligible if reference != pid]
        result[pid] = rng.choice(pool) if pool else None
    return result


def choose(archive: dict[int, Program], config: Config,
           rng: random.Random) -> tuple[Program, str]:
    allocation = weights(archive, config.selection_temperature)
    if not allocation:
        raise ValueError("No valid parent candidates")
    ids = sorted(allocation)
    parent = archive[rng.choices(ids, [allocation[pid] for pid in ids], k=1)[0]]
    operator = rng.choices(["Refine", "Explore", "Crossover"],
                           [config.refine_weight, config.explore_weight, config.crossover_weight], k=1)[0]
    return parent, operator
