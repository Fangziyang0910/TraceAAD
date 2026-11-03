"""Parent choice by the estimated probability that a child becomes a new best."""

import ast
from collections import defaultdict
from functools import lru_cache
import hashlib


@lru_cache(maxsize=4096)
def source_key(code):
    normalized = ast.dump(ast.parse(code), include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def candidates(archive):
    seen, result = set(), []
    for program in sorted(archive.values(), key=lambda p: p.id):
        if program.valid:
            key = source_key(program.code)
            if key not in seen:
                result.append(program)
                seen.add(key)
    return sorted(result, key=lambda p: (p.evaluation.fitness, p.id))


def quality_weights(archive):
    programs = candidates(archive)
    weights = [1 / (i + 1) ** 2 for i in range(len(programs))]
    start = 0
    while start < len(programs):
        stop = start + 1
        while stop < len(programs) and programs[stop].evaluation.fitness == programs[start].evaluation.fitness:
            stop += 1
        weights[start:stop] = [sum(weights[start:stop]) / (stop - start)] * (stop - start)
        start = stop
    return programs, weights


def proposals(archive, rng, count):
    """Distinct programs drawn without replacement with weight 1/(rank+1): good states first, a broad tail."""
    shown, rest = [], list(enumerate(candidates(archive)))
    while rest and len(shown) < count:
        index = rng.choices(range(len(rest)), [1 / (rank + 1) for rank, _ in rest])[0]
        shown.append(rest.pop(index)[1])
    return shown


def rank_distribution(archive, floor=.05):
    programs, weights = quality_weights(archive)
    total = sum(weights)
    return [(p, (1 - floor) * w / total + floor / len(programs)) for p, w in zip(programs, weights)]


def node_distribution(archive, responses, *, mode="neighborhood", window=256, strength=2., floor=.05):
    """Probability that a child of each node becomes a new best: P(improve node) * P(gain covers the gap)."""
    if mode not in {"rank", "direct", "neighborhood"}:
        raise ValueError("Unknown node selection mode")
    if window < 1 or strength <= 0 or not 0 <= floor < 1:
        raise ValueError("Invalid node selection settings")
    if mode == "rank":
        return [(p, probability, {}) for p, probability in rank_distribution(archive, floor)]
    programs = candidates(archive)
    if not programs:
        return []
    eligible = {p.id for p in programs}
    events = {}
    for event in responses:
        if event.evaluation_id in events and events[event.evaluation_id] != event:
            raise ValueError("Conflicting response for one evaluation")
        events[event.evaluation_id] = event
    history = sorted(events.values(), key=lambda e: e.evaluation_id)[-window:]
    emitted, children = defaultdict(list), defaultdict(set)
    for event in history:
        if event.selected_parent_id not in eligible:
            raise ValueError("Response source must be an archived canonical valid node")
        emitted[event.selected_parent_id].append(event)
    for program in programs:
        if program.parent_id in eligible:
            children[program.parent_id].add(program.id)
    gains = [e.gain for e in history if e.response]
    best = programs[0].evaluation.fitness
    base = (1 + len(gains)) / (2 + len(history))
    rows = []
    for program in programs:
        direct = emitted[program.id]
        starts = children[program.id] | ({program.parent_id} if program.parent_id in eligible else set())
        shared = [event for source in sorted(starts) for event in emitted[source]] if mode == "neighborhood" else []
        shared.sort(key=lambda e: e.evaluation_id)
        n, s = len(direct), sum(e.response for e in direct)
        nn, sn = len(shared), sum(e.response for e in shared)
        mn = (strength * base + sn) / (strength + nn) if mode == "neighborhood" else base
        response = (strength * mn + s) / (strength + n)
        gap = program.evaluation.fitness - best
        reach = 1. if gap <= 0 else (1 + sum(g > gap for g in gains)) / (1 + len(gains))
        rows.append((program, {"gap": gap, "reach": reach, "improvement_count": len(gains), "base_rate": base,
                    "neighbor_mean": mn, "direct_count": n, "direct_improvements": s,
                    "neighbor_count": nn, "neighbor_improvements": sn, "response_rate": response,
                    "value": response * reach, "direct_event_ids": [e.evaluation_id for e in direct],
                    "neighbor_event_ids": [e.evaluation_id for e in shared]}))
    total = sum(row["value"] for _, row in rows)
    return [(p, (1 - floor) * row["value"] / total + floor / len(rows), row) for p, row in rows]


def choose(archive, responses, rng, *, mode="neighborhood", window=256, strength=2., floor=.05):
    distribution = node_distribution(archive, responses, mode=mode, window=window, strength=strength, floor=floor)
    if not distribution:
        raise ValueError("No valid source program is available")
    chosen = rng.choices(distribution, [row[1] for row in distribution])[0]
    parent, probability, details = chosen
    return parent, {"selection_mode": mode, "parent_probability": probability,
        "eligible_parents": len(distribution), "response_window": window, "response_strength": strength,
        "eligible_max_id": max(p.id for p, _, _ in distribution),
        "response_event_max": max((e.evaluation_id for e in responses), default=0), **details}
