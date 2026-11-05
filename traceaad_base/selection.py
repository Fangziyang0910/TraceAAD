"""Parent choice: better current programs first, fewer chances as attempts from them accumulate."""

import ast
from functools import lru_cache
import hashlib
from itertools import groupby
import random

from .config import Config
from .models import Attempt, Program

RANK_EXPONENT = 1.5  # Above 1, so the share of low ranks does not keep growing with the archive.
VISIT_SMOOTHING = 3


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
    for program in sorted((p for p in archive.values() if p.valid), key=lambda p: p.id):
        earliest.setdefault(identity(program.code), program)
    return sorted(earliest.values(), key=lambda p: (p.evaluation.fitness, p.id))


def qualities(archive: dict[int, Program]) -> dict[int, float]:
    result = {}
    offset = 0
    for _, tied in groupby(sources(archive), key=lambda p: p.evaluation.fitness):
        group = list(tied)
        rank = offset + (len(group) + 1) / 2
        for program in group:
            result[program.id] = rank ** -RANK_EXPONENT
        offset += len(group)
    return result


def weights(archive: dict[int, Program], attempts: list[Attempt]) -> dict[int, float]:
    quality = qualities(archive)
    costs = dict.fromkeys(quality, 0)
    for attempt in attempts:
        if attempt.parent_id in costs:
            costs[attempt.parent_id] += attempt.model_calls
    return {pid: quality[pid] * VISIT_SMOOTHING / (VISIT_SMOOTHING + costs[pid]) for pid in quality}


def reference_pool(archive: dict[int, Program], parent_id: int) -> list[Program]:
    """Distinct sources not worse than the median, without the parent's source."""
    candidates = sources(archive)
    if not candidates:
        return []
    n = len(candidates)
    median = candidates[n // 2].evaluation.fitness
    if n % 2 == 0:
        median = median / 2 + candidates[n // 2 - 1].evaluation.fitness / 2
    own = identity(archive[parent_id].code)
    return [p for p in candidates if identity(p.code) != own and p.evaluation.fitness <= median]


def choose(archive: dict[int, Program], attempts: list[Attempt], config: Config,
           rng: random.Random) -> tuple[Program, str]:
    allocation = weights(archive, attempts)
    if not allocation:
        raise ValueError("No valid parent candidates")
    ids = sorted(allocation)
    parent = archive[rng.choices(ids, [allocation[pid] for pid in ids], k=1)[0]]
    operator = rng.choices(["Refine", "Explore", "Crossover"],
                           [config.refine_weight, config.explore_weight, config.crossover_weight], k=1)[0]
    return parent, operator
