"""Inverse-rank offspring quality per model call; no parent-improvement reward."""

from itertools import groupby
import random

from .config import Config
from .models import Attempt, Program


def qualities(archive: dict[int, Program]) -> dict[int, float]:
    candidates = sorted((p for p in archive.values() if p.valid),
                        key=lambda p: (p.evaluation.fitness, p.id))
    result = {}
    offset = 0
    for _, tied in groupby(candidates, key=lambda p: p.evaluation.fitness):
        group = list(tied)
        rank = offset + (len(group) + 1) / 2
        for program in group:
            result[program.id] = 1 / rank
        offset += len(group)
    return result


def weights(archive: dict[int, Program], attempts: list[Attempt]) -> dict[int, float]:
    quality = qualities(archive)
    costs = dict.fromkeys(quality, 0)
    offspring_quality = dict.fromkeys(quality, 0.0)
    for attempt in attempts:
        if attempt.parent_id in costs:
            costs[attempt.parent_id] += attempt.model_calls
            if attempt.program_id in quality:
                offspring_quality[attempt.parent_id] += quality[attempt.program_id]
    return {pid: (3 * quality[pid] + offspring_quality[pid]) / (3 + costs[pid])
            for pid in quality}


def reference_pool(archive: dict[int, Program], parent_id: int) -> list[Program]:
    candidates = sorted((p for p in archive.values() if p.valid),
                        key=lambda p: (p.evaluation.fitness, p.id))
    if not candidates:
        return []
    n = len(candidates)
    median = candidates[n // 2].evaluation.fitness
    if n % 2 == 0:
        median = median / 2 + candidates[n // 2 - 1].evaluation.fitness / 2
    return [p for p in candidates if p.id != parent_id and p.evaluation.fitness <= median]


def choose(archive: dict[int, Program], attempts: list[Attempt], config: Config,
           rng: random.Random) -> tuple[Program, str, Program | None]:
    allocation = weights(archive, attempts)
    if not allocation:
        raise ValueError("No valid parent candidates")
    ids = sorted(allocation)
    parent = archive[rng.choices(ids, [allocation[pid] for pid in ids], k=1)[0]]
    references = reference_pool(archive, parent.id)
    operator = rng.choices(
        ["Refine", "Explore", "Crossover"],
        [config.refine_weight, config.explore_weight,
         config.crossover_weight if references else 0],
        k=1,
    )[0]
    reference = rng.choice(references) if operator == "Crossover" else None
    return parent, operator, reference

