"""Trajectory-conditioned joint allocation of parent--operator actions."""

from itertools import groupby
import math
import random

from .config import Config
from .models import Attempt, Program


OPERATORS = ("Refine", "Explore", "Crossover")
Pair = tuple[int, str]


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


def _child_quality(quality: dict[int, float], attempt: Attempt) -> float:
    """Use the final child outcome; a failed attempt contributes zero."""
    return quality.get(attempt.program_id, 0.0)


def development_stats(
    archive: dict[int, Program],
    attempts: list[Attempt],
    quality: dict[int, float] | None = None,
    *,
    operator: str | None = None,
    eligible_ids: set[int] | None = None,
) -> dict[int, dict[str, float | int]]:
    """Summarize direct outcomes for parents, optionally under one operator."""
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
        child_quality = _child_quality(quality, attempt)
        parent["child_quality_sum"] += child_quality
        if child_quality > 0.0:
            parent["successful_children"] += 1
        else:
            parent["failed_children"] += 1
    return stats


def pair_stats(
    archive: dict[int, Program],
    attempts: list[Attempt],
    quality: dict[int, float] | None = None,
    pairs: set[Pair] | None = None,
) -> dict[Pair, dict[str, float | int]]:
    """Summarize the observed response of every eligible parent--operator pair."""
    quality = qualities(archive) if quality is None else quality
    if pairs is None:
        pairs = {(pid, operator) for pid in quality for operator in OPERATORS}
    stats = {
        pair: {
            "attempts": 0,
            "successful_children": 0,
            "failed_children": 0,
            "child_quality_sum": 0.0,
            "model_calls": 0,
        }
        for pair in pairs
    }
    for attempt in attempts:
        if not attempt.completed or attempt.operator == "Init":
            continue
        key = (attempt.parent_id, attempt.operator)
        item = stats.get(key)
        if item is None:
            continue
        item["attempts"] += 1
        item["model_calls"] += attempt.model_calls
        child_quality = _child_quality(quality, attempt)
        item["child_quality_sum"] += child_quality
        if child_quality > 0.0:
            item["successful_children"] += 1
        else:
            item["failed_children"] += 1
    return stats


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


def _operator_priors(archive: dict[int, Program], config: Config) -> dict[str, float]:
    """Return the Base operator mixture as a soft prior over available actions."""
    valid_ids = [p.id for p in archive.values() if p.valid]
    crossover_available = len(valid_ids) > 1 and any(
        reference_pool(archive, parent_id) for parent_id in valid_ids
    )
    values = {
        "Refine": config.refine_weight,
        "Explore": config.explore_weight,
        "Crossover": config.crossover_weight if crossover_available else 0.0,
    }
    total = math.fsum(values.values())
    if total <= 0.0:
        raise ValueError("No available operator")
    return {operator: value / total for operator, value in values.items()}


def _eligible_pairs(
    quality: dict[int, float], archive: dict[int, Program], priors: dict[str, float]
) -> list[Pair]:
    return [
        (program_id, operator)
        for operator, prior in priors.items()
        if prior > 0.0
        for program_id in sorted(quality)
        if operator != "Crossover" or reference_pool(archive, program_id)
    ]


def joint_weights(
    archive: dict[int, Program],
    attempts: list[Attempt],
    config: Config | None = None,
) -> tuple[dict[Pair, float], dict[Pair, dict[str, float | int]]]:
    """Score parent--operator pairs with shared priors and local trajectory data."""
    config = config or Config()
    quality = qualities(archive)
    if not quality:
        return {}, {}
    priors = _operator_priors(archive, config)
    pairs = _eligible_pairs(quality, archive, priors)
    if not pairs:
        return {}, {}
    pair_observations = pair_stats(archive, attempts, quality, set(pairs))
    global_quality = math.fsum(quality.values()) / len(quality)
    operator_observations = {
        operator: {
            "attempts": 0,
            "successful_children": 0,
            "failed_children": 0,
            "child_quality_sum": 0.0,
            "model_calls": 0,
        }
        for operator in OPERATORS
    }
    for (_, operator), item in pair_observations.items():
        aggregate = operator_observations[operator]
        for key in ("attempts", "successful_children", "failed_children", "model_calls"):
            aggregate[key] += item[key]
        aggregate["child_quality_sum"] += item["child_quality_sum"]

    diagnostics: dict[Pair, dict[str, float | int]] = {}
    total_attempts = sum(int(item["attempts"]) for item in pair_observations.values())
    log_total = math.log1p(total_attempts + len(pairs))
    raw: dict[Pair, float] = {}
    for pair in pairs:
        parent_id, operator = pair
        item = pair_observations[pair]
        operator_item = operator_observations[operator]
        operator_count = int(operator_item["attempts"])
        operator_mean = (
            config.operator_prior_strength * global_quality
            + float(operator_item["child_quality_sum"])
        ) / (config.operator_prior_strength + operator_count)
        count = int(item["attempts"])
        mean = (
            config.parent_prior_strength * quality[parent_id]
            + config.operator_prior_strength * operator_mean
            + float(item["child_quality_sum"])
        ) / (config.parent_prior_strength + config.operator_prior_strength + count)
        uncertainty = config.joint_exploration * math.sqrt(log_total / (1 + count))
        score = max(mean + uncertainty, 1e-12)
        raw[pair] = priors[operator] * score
        diagnostics[pair] = {
            **item,
            "quality": quality[parent_id],
            "operator_prior": priors[operator],
            "global_quality": global_quality,
            "global_operator_attempts": operator_count,
            "global_operator_mean": operator_mean,
            "mean": mean,
            "uncertainty": uncertainty,
            "score": score,
            "operator": operator,
            "parent_id": parent_id,
        }
    total = math.fsum(raw.values())
    probabilities = {pair: value / total for pair, value in raw.items()}
    if config.operator_floor:
        eligible_by_operator = {
            operator: [pair for pair in pairs if pair[1] == operator]
            for operator in OPERATORS
        }
        floor = config.operator_floor
        for pair in pairs:
            operator = pair[1]
            floor_probability = priors[operator] / len(eligible_by_operator[operator])
            probabilities[pair] = (
                (1.0 - floor) * probabilities[pair] + floor * floor_probability
            )
    for pair, diagnostic in diagnostics.items():
        diagnostic["raw_probability"] = raw[pair] / total
        diagnostic["joint_probability"] = probabilities[pair]
    return probabilities, diagnostics


def weights(
    archive: dict[int, Program],
    attempts: list[Attempt],
    config: Config | None = None,
    *,
    operator: str | None = None,
    eligible_ids: set[int] | None = None,
) -> tuple[dict[int, float], dict[int, dict[str, float | int]]]:
    """Compatibility view of joint probabilities grouped by parent."""
    probabilities, diagnostics = joint_weights(archive, attempts, config)
    grouped: dict[int, float] = {}
    grouped_diagnostics: dict[int, dict[str, float | int]] = {}
    for (parent_id, selected_operator), diagnostic in diagnostics.items():
        if operator is not None and selected_operator != operator:
            continue
        if eligible_ids is not None and parent_id not in eligible_ids:
            continue
        grouped[parent_id] = grouped.get(parent_id, 0.0) + probabilities[(parent_id, selected_operator)]
        current = grouped_diagnostics.get(parent_id)
        if current is None or diagnostic["score"] > current["score"]:
            grouped_diagnostics[parent_id] = dict(diagnostic)
    return grouped, grouped_diagnostics


def choose(
    archive: dict[int, Program],
    attempts: list[Attempt],
    config: Config,
    rng: random.Random,
) -> tuple[Program, str, Program | None, dict[str, object]]:
    """Sample one parent--operator action from the joint allocation."""
    probabilities, diagnostics = joint_weights(archive, attempts, config)
    if not probabilities:
        raise ValueError("No valid parent--operator actions")
    pairs = sorted(probabilities)
    pair = rng.choices(pairs, weights=[probabilities[item] for item in pairs], k=1)[0]
    parent_id, operator = pair
    parent = archive[parent_id]
    references = reference_pool(archive, parent.id) if operator == "Crossover" else []
    reference = rng.choice(references) if references else None
    parent_probability = math.fsum(
        probability for (candidate_id, _), probability in probabilities.items()
        if candidate_id == parent_id
    )
    operator_probability = math.fsum(
        probability for (_, candidate_operator), probability in probabilities.items()
        if candidate_operator == operator
    )
    selection = {
        **diagnostics[pair],
        "joint_probability": probabilities[pair],
        "parent_probability": parent_probability,
        "operator_probability": operator_probability,
        "operator_conditional_probability": probabilities[pair] / parent_probability,
        "eligible_pairs": len(pairs),
        "operator_floor": config.operator_floor,
        "reference_pool_size": len(references),
    }
    return parent, operator, reference, selection
