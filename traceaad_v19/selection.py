"""Trajectory-aware allocation of parent--operator experiments.

An individual is not treated as a single stationary arm.  Each operator
defines a different intervention on that individual's local design trajectory.
The fixed operator mixture chooses the intervention family; a UCB-style score
then allocates that operator's next attempt among eligible parents.
"""

from itertools import groupby
import math
import random

from .config import Config
from .models import Attempt, Program


OPERATORS = ("Refine", "Explore", "Crossover")


def qualities(archive: dict[int, Program]) -> dict[int, float]:
    """Return inverse average-rank quality for valid programs."""
    candidates = sorted(
        (p for p in archive.values() if p.valid),
        key=lambda p: (p.evaluation.fitness, p.id),
    )
    result: dict[int, float] = {}
    offset = 0
    for _, tied in groupby(candidates, key=lambda p: p.evaluation.fitness):
        group = list(tied)
        rank = offset + (len(group) + 1) / 2
        for program in group:
            result[program.id] = 1.0 / rank
        offset += len(group)
    return result


def development_stats(
    archive: dict[int, Program],
    attempts: list[Attempt],
    quality: dict[int, float] | None = None,
    *,
    operator: str | None = None,
    eligible_ids: set[int] | None = None,
) -> dict[int, dict[str, float | int]]:
    """Summarize direct outcomes for one operator's local trajectories.

    A completed generation attempt counts once, including an attempt that used
    the shared repair opportunity.  A valid final child contributes its
    current absolute quality; a format, model, execution, or timeout failure
    contributes zero.  Repair is part of the same parent--operator attempt.
    """
    quality = qualities(archive) if quality is None else quality
    ids = set(quality) if eligible_ids is None else set(eligible_ids)
    stats = {
        program_id: {
            "attempts": 0,
            "successful_children": 0,
            "failed_children": 0,
            "child_quality_sum": 0.0,
            "model_calls": 0,
        }
        for program_id in ids
    }
    for attempt in attempts:
        if not attempt.completed or attempt.operator == "Init":
            continue
        if operator is not None and attempt.operator != operator:
            continue
        parent = stats.get(attempt.parent_id)
        if parent is None:
            continue
        parent["attempts"] += 1
        parent["model_calls"] += attempt.model_calls
        child_quality = quality.get(attempt.program_id, 0.0)
        parent["child_quality_sum"] += child_quality
        if child_quality > 0.0:
            parent["successful_children"] += 1
        else:
            parent["failed_children"] += 1
    return stats


def weights(
    archive: dict[int, Program],
    attempts: list[Attempt],
    config: Config | None = None,
    *,
    operator: str | None = None,
    eligible_ids: set[int] | None = None,
) -> tuple[dict[int, float], dict[int, dict[str, float | int]]]:
    """Calculate UCB-style weights for one parent--operator allocation.

    The current parent quality is a prior.  Direct offspring quality updates
    only the selected operator's trajectory.  The uncertainty term depends on
    how many times this parent has received that operator, not on birth order,
    search phase, or whether a child beat its parent.
    """
    quality = qualities(archive)
    if eligible_ids is not None:
        quality = {pid: value for pid, value in quality.items() if pid in eligible_ids}
    if not quality:
        return {}, {}
    config = config or Config()
    stats = development_stats(
        archive,
        attempts,
        qualities(archive),
        operator=operator,
        eligible_ids=set(quality),
    )
    total_attempts = sum(int(value["attempts"]) for value in stats.values())
    log_total = math.log1p(total_attempts + len(quality))
    diagnostics: dict[int, dict[str, float | int]] = {}
    allocation: dict[int, float] = {}
    for program_id in sorted(quality):
        item = stats[program_id]
        count = int(item["attempts"])
        mean = (
            config.parent_prior_strength * quality[program_id]
            + float(item["child_quality_sum"])
        ) / (config.parent_prior_strength + count)
        uncertainty = config.parent_exploration * math.sqrt(log_total / (1 + count))
        score = mean + uncertainty
        allocation[program_id] = score
        diagnostics[program_id] = {
            **item,
            "quality": quality[program_id],
            "mean": mean,
            "uncertainty": uncertainty,
            "score": score,
            "operator": operator or "all",
        }
    return allocation, diagnostics


def reference_pool(archive: dict[int, Program], parent_id: int) -> list[Program]:
    """Return the quality-screened reference pool for Crossover."""
    candidates = sorted(
        (p for p in archive.values() if p.valid),
        key=lambda p: (p.evaluation.fitness, p.id),
    )
    if not candidates:
        return []
    n = len(candidates)
    median = candidates[n // 2].evaluation.fitness
    if n % 2 == 0:
        median = (median + candidates[n // 2 - 1].evaluation.fitness) / 2
    return [p for p in candidates if p.id != parent_id and p.evaluation.fitness <= median]


def _operator_weights(archive: dict[int, Program], config: Config) -> dict[str, float]:
    """Keep the fixed operator mixture while disabling unavailable Crossover."""
    valid_ids = [p.id for p in archive.values() if p.valid]
    crossover_available = len(valid_ids) > 1 and any(
        reference_pool(archive, parent_id) for parent_id in valid_ids
    )
    return {
        "Refine": config.refine_weight,
        "Explore": config.explore_weight,
        "Crossover": config.crossover_weight if crossover_available else 0.0,
    }


def choose(
    archive: dict[int, Program],
    attempts: list[Attempt],
    config: Config,
    rng: random.Random,
) -> tuple[Program, str, Program | None, dict[str, object]]:
    """Choose an operator first, then allocate that operator to a parent."""
    quality = qualities(archive)
    if not quality:
        raise ValueError("No valid parent candidates")

    operator_weights = _operator_weights(archive, config)
    operator = rng.choices(
        list(OPERATORS), [operator_weights[name] for name in OPERATORS], k=1
    )[0]
    eligible_ids = {
        program_id
        for program_id in quality
        if operator != "Crossover" or reference_pool(archive, program_id)
    }
    allocation, diagnostics = weights(
        archive,
        attempts,
        config,
        operator=operator,
        eligible_ids=eligible_ids,
    )
    if not allocation:
        raise ValueError(f"No valid parent candidates for {operator}")
    ids = sorted(allocation)
    total = math.fsum(allocation.values())
    probabilities = {pid: allocation[pid] / total for pid in ids}
    parent_id = rng.choices(ids, [allocation[pid] for pid in ids], k=1)[0]
    parent = archive[parent_id]
    references = reference_pool(archive, parent.id) if operator == "Crossover" else []
    reference = rng.choice(references) if references else None
    selection = {
        **diagnostics[parent.id],
        "operator": operator,
        "probability": probabilities[parent.id],
        "eligible": len(ids),
        "operator_weights": operator_weights,
        "reference_pool_size": len(references),
    }
    return parent, operator, reference, selection
