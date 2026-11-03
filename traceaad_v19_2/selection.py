"""Allocate bounded development episodes, keeping one frontier per segment.

Current ranks select current implementations. They never redefine historical
outcomes. No per-program posterior, ancestor reward, or UCB is estimated.
"""

import ast
from collections import defaultdict
from functools import lru_cache
import hashlib
import math
import random

from .config import Config
from .models import Attempt, Program

OPERATORS = ("Refine", "Explore", "Crossover")
Pair = tuple[int, str]


@lru_cache(maxsize=4096)
def source_key(code: str) -> str:
    """Ignore formatting/comments, retaining names, constants and docstrings.

    This is a conservative implementation identity, not semantic idea recognition.
    """
    try:
        text = ast.dump(ast.parse(code), include_attributes=False)
    except (SyntaxError, ValueError, RecursionError):
        text = code
    return hashlib.sha256(text.encode()).hexdigest()


def segment_state(archive: dict[int, Program]) -> dict:
    """Canonicalize valid source records before deriving operation segments."""
    owners, canonical, membership, members = {}, {}, {}, defaultdict(list)
    for pid in sorted(archive):
        p = archive[pid]
        if not p.valid:
            continue
        first = owners.setdefault(source_key(p.code), pid)
        canonical[pid] = first
        if first != pid:
            membership[pid] = membership[first]
            continue
        if p.operator == "Refine":
            if p.parent_id not in membership:
                raise ValueError(f"Refine program {pid} needs an earlier valid parent")
            segment = membership[p.parent_id]
        elif p.operator in {"Init", "Explore", "Crossover"}:
            segment = pid
        else:
            raise ValueError(f"Invalid origin for program {pid}: {p.operator}")
        membership[pid] = segment
        members[segment].append(pid)
    frontiers = {
        sid: min(ids, key=lambda pid: (archive[pid].evaluation.fitness, pid))
        for sid, ids in members.items()
    }
    return {"canonical": canonical, "membership": membership,
            "members": dict(members), "frontiers": frontiers}


def development_segment(archive: dict[int, Program], program_id: int) -> int:
    return segment_state(archive)["membership"][program_id]


def _references(archive, parent_id, state):
    sid = state["membership"][parent_id]
    ids = [pid for other, pid in state["frontiers"].items() if other != sid]
    # Prefer other segments. A single segment may still have complementary
    # distinct implementations; do not silently disable all recombination.
    if not ids:
        key = source_key(archive[parent_id].code)
        ids = [pid for pid in state["members"][sid]
               if source_key(archive[pid].code) != key]
    ids.sort(key=lambda pid: (archive[pid].evaluation.fitness, pid))
    if not ids:
        return []
    cutoff = archive[ids[(len(ids)-1)//2]].evaluation.fitness
    return [archive[pid] for pid in ids if archive[pid].evaluation.fitness <= cutoff]


def reference_pool(archive: dict[int, Program], parent_id: int) -> list[Program]:
    return _references(archive, parent_id, segment_state(archive))


def rank_weights(archive, ids):
    """Inverse-square rank weights, averaging weights across exact ties.

    This fixed selection pressure is a heuristic. It is invariant under increasing
    score transforms, works with negative objectives, and needs no cohort scale.
    """
    ordered = sorted(ids, key=lambda pid: (archive[pid].evaluation.fitness, pid))
    result, i = {}, 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and archive[ordered[j]].evaluation.fitness == archive[ordered[i]].evaluation.fitness:
            j += 1
        value = math.fsum(1.0 / (k + 1)**2 for k in range(i, j)) / (j-i)
        result.update((pid, value) for pid in ordered[i:j])
        i = j
    total = math.fsum(result.values())
    return {pid: value / total for pid, value in result.items()}


def active_episode(archive, attempts, config, state=None):
    """Reconstruct the bounded continuation entirely from committed attempts."""
    if not attempts or not attempts[-1].completed:
        return None
    last = attempts[-1]
    eid = last.selection.get("episode_id")
    if eid is None:
        return None
    trials = []
    for a in reversed(attempts):
        if not a.completed or a.selection.get("episode_id") != eid:
            break
        trials.append(a)
    trials.reverse()
    spent = sum(a.model_calls for a in trials)
    if spent >= config.development_calls:
        return None
    first = trials[0]
    state = state or segment_state(archive)
    if first.operator in {"Explore", "Crossover"}:
        seed = archive.get(first.program_id)
        # A failed or copied discovery has not created a development process.
        if seed is None or not seed.valid or state["canonical"].get(seed.id) != seed.id:
            return None
        anchor = seed.id
    else:
        anchor = first.parent_id
    sid = state["membership"][anchor]
    tip = anchor
    for trial in trials:
        child = archive.get(trial.program_id)
        if (child is not None and child.valid
                and state["membership"].get(child.id) == sid):
            # A copied anchor is an explicit return to that implementation,
            # not a new competitor or an extension of the episode allowance.
            tip = state["canonical"][child.id]
    return {"episode_id": eid, "episode_calls_before": spent,
            "episode_anchor_id": anchor, "segment_id": sid,
            "parent_id": tip, "origin_operator": first.operator,
            "initial_development": first.selection.get("phase") == "initial_development",
            "phase": "continue", "episode_trial_ids": [a.id for a in trials]}


def _parent_probabilities(archive, ids, config):
    ranked = rank_weights(archive, ids)
    return {pid: (1-config.parent_floor)*p + config.parent_floor/len(ids)
            for pid, p in ranked.items()}


def joint_weights(archive: dict[int, Program], attempts: list[Attempt],
                  config: Config | None = None):
    """Policy at the present state, including deterministic episode steps."""
    config = config or Config()
    state = segment_state(archive)
    frontiers = state["frontiers"]
    if not frontiers:
        return {}, {}
    ongoing = active_episode(archive, attempts, config, state)
    probabilities, rows = {}, {}
    if ongoing:
        # Control: retain the same request block, first-step operator policy and
        # repair limit, but reselect a frontier on subsequent Refine steps.
        # Keep initial protected development identical in the two conditions.
        reselect = config.reselect_parent and not ongoing["initial_development"]
        parent_probs = (_parent_probabilities(archive, list(frontiers.values()), config)
                        if reselect else {ongoing["parent_id"]: 1.0})
        for pid, probability in parent_probs.items():
            pair = (pid, "Refine")
            probabilities[pair] = probability
            rows[pair] = {**ongoing, "parent_id": pid}
            if reselect:
                rows[pair].update(episode_anchor_id=pid, segment_id=state["membership"][pid],
                                  phase="reselect")
    else:
        developed = {state["membership"][a.parent_id] for a in attempts
                     if a.completed and a.operator == "Refine"
                     and a.parent_id in state["membership"]}
        # Discoveries already receive their allowance in the birth episode.
        # Only initialization has no birth episode. Do not grant a second
        # allowance when a discovery's repair exhausted its entire episode.
        pending = [sid for sid in sorted(frontiers)
                   if archive[sid].operator == "Init" and sid not in developed]
        protected = config.protect_new_segments and bool(pending)
        ids = [frontiers[sid] for sid in pending] if protected else list(frontiers.values())
        parent_probs = ({pid: 1/len(ids) for pid in ids} if protected
                        else _parent_probabilities(archive, ids, config))
        eid = max((a.id for a in attempts), default=0) + 1
        for pid, p_parent in parent_probs.items():
            priors = ({"Refine": 1.0} if protected else {
                "Refine": config.refine_weight, "Explore": config.explore_weight,
                "Crossover": config.crossover_weight if (
                    len(frontiers) > 1 or len(state["members"][state["membership"][pid]]) > 1
                ) else 0.0})
            total = math.fsum(priors.values())
            for op, prior in priors.items():
                if prior <= 0:
                    continue
                pair = (pid, op)
                probabilities[pair] = p_parent * prior / total
                rows[pair] = {"episode_id": eid, "episode_calls_before": 0,
                              "episode_anchor_id": pid, "segment_id": state["membership"][pid],
                              "parent_id": pid, "origin_operator": op,
                              "phase": "initial_development" if protected else "start",
                              "episode_trial_ids": []}
    parent_marginals, operator_marginals = defaultdict(float), defaultdict(float)
    for (pid, op), value in probabilities.items():
        parent_marginals[pid] += value
        operator_marginals[op] += value
    for pair, row in rows.items():
        pid, op = pair
        row.update(operator=op, joint_probability=probabilities[pair],
                   parent_probability=parent_marginals[pid],
                   operator_probability=operator_marginals[op],
                   operator_conditional_probability=probabilities[pair]/parent_marginals[pid],
                   development_calls=config.development_calls,
                   segment_count=len(frontiers), segment_size=len(state["members"][row["segment_id"]]),
                   frontier_id=frontiers[row["segment_id"]],
                   current_fitness=archive[pid].evaluation.fitness,
                   eligible_pairs=len(probabilities))
    return probabilities, rows


def choose(archive: dict[int, Program], attempts: list[Attempt], config: Config,
           rng: random.Random):
    probabilities, rows = joint_weights(archive, attempts, config)
    if not probabilities:
        raise ValueError("No valid parent--operator actions")
    pairs = sorted(probabilities)
    pair = rng.choices(pairs, weights=[probabilities[p] for p in pairs], k=1)[0]
    pid, op = pair
    references = reference_pool(archive, pid) if op == "Crossover" else []
    reference = rng.choice(references) if references else None
    return archive[pid], op, reference, {**rows[pair], "reference_pool_size": len(references),
                                        "reference_probability": 1/len(references) if references else None}


def weights(archive, attempts, config=None, *, operator=None, eligible_ids=None):
    """Compatibility view of the actual parent marginal, without renormalizing."""
    probabilities, diagnostics = joint_weights(archive, attempts, config)
    grouped, rows = defaultdict(float), {}
    for pair, probability in probabilities.items():
        pid, op = pair
        if (operator is None or operator == op) and (eligible_ids is None or pid in eligible_ids):
            grouped[pid] += probability
            rows[pid] = diagnostics[pair]
    return dict(grouped), rows
