"""Fitness-rank selection among distinct valid source programs."""

import ast
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


def rank_distribution(archive, floor=.05):
    programs = candidates(archive)
    n = len(programs)
    if not n:
        return []
    weights = [1 / (i + 1) ** 2 for i in range(n)]
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and programs[stop].evaluation.fitness == programs[start].evaluation.fitness:
            stop += 1
        average = sum(weights[start:stop]) / (stop - start)
        weights[start:stop] = [average] * (stop - start)
        start = stop
    total = sum(weights)
    return [(p, (1 - floor) * w / total + floor / n) for p, w in zip(programs, weights)]


def choose(archive, rng, floor=.05):
    distribution = rank_distribution(archive, floor)
    parent = rng.choices([p for p, _ in distribution], [w for _, w in distribution])[0]
    probability = next(w for p, w in distribution if p.id == parent.id)
    return parent, {"parent_probability": probability, "eligible_parents": len(distribution)}
