import random

import pytest

from traceaad_v19_4 import Config, Evaluation, Program
from traceaad_v19_4.knowledge import ExperiencePool, tokens
from traceaad_v19_4.models import Attempt
from traceaad_v19_4.prompts import PromptBuilder
from traceaad_v19_4.selection import rank_distribution


def test_global_bilingual_retrieval_deduplication_and_retention(lesson):
    pool = ExperiencePool()
    content = {**lesson, "applicable_when": "构造式算法需要候选筛选与前瞻估计"}
    notes = [{"polarity": "positive", "content": content}] * 2
    assert pool.replace(notes, {"attempt_id": 1, "reflection_attempt_ids": [1]}) == [1]
    assert pool.retrieve("候选筛选和前瞻估计", 3, random.Random(0))[0]["id"] == 1
    assert not pool.retrieve("unrelated_xyz", 3, random.Random(0))
    assert "前瞻" in tokens("前瞻估计")
    pool.record_usage([1], attempt_id=8, improved=False)
    pool.record_usage([1], attempt_id=8, improved=False)
    assert pool.entries[1].provided_count == 1
    assert pool.retrieve("lookahead", 3, random.Random(0))[0]["id"] == 1
    restored = ExperiencePool(pool.to_list(), pool.versions)
    assert restored.retrieve("lookahead", 3, random.Random(5)) == pool.retrieve("lookahead", 3, random.Random(5))


def test_tied_rank_weights_and_duplicates_do_not_add_mass():
    archive = {i: Program(i, f"def solve(x): return {i}", "rule", "Init", 0, Evaluation("success", fitness))
               for i, fitness in [(1, 1.), (2, 1.), (3, 2.)]}
    archive[4] = Program(4, archive[1].code, "duplicate", "Develop", 3, Evaluation("success", 1.))
    distribution = rank_distribution(archive)
    assert len(distribution) == 3
    assert distribution[0][1] == distribution[1][1]
    assert sum(w for _, w in distribution) == pytest.approx(1)
    assert distribution[-1][1] >= .05 / 3


def test_reflection_modes_only_change_visible_history(task, counter):
    archive = {}
    attempts = []
    for i in range(1, 7):
        parent = 0 if i == 1 else 3 if i >= 4 else i - 1
        program = Program(i, f"def solve(x): return {i}", "bounded lookahead", "Init" if i == 1 else "Develop",
                          parent, Evaluation("success", 10. - i), "change", i, parent)
        archive[i] = program
        attempts.append(Attempt(i, parent, program.operator, program_id=i, version_ids=[i],
            evaluations=[{"status": "success"}], completed=i != 5))
    # P3 -> P4 precedes latest P3 -> P5; future P3 -> P6 must stay hidden.
    modes = {}
    for mode in ("local", "siblings", "trajectory"):
        builder = PromptBuilder(task, Config(reflection_mode=mode), counter)
        modes[mode] = builder.reflection(archive, attempts, attempts[4]).metadata
    assert modes["local"]["main_attempt_ids"] == [5]
    assert modes["local"]["alternative_attempt_ids"] == []
    assert modes["siblings"]["main_attempt_ids"] == [5]
    assert modes["siblings"]["alternative_attempt_ids"] == [4]
    assert modes["trajectory"]["main_attempt_ids"] == [2, 3, 5]
    assert modes["trajectory"]["alternative_attempt_ids"] == [4]


def test_generation_shows_code_only_for_the_current_algorithm(task, counter):
    archive, attempts = {}, []
    for i in range(1, 6):
        parent = 0 if i == 1 else 3 if i >= 4 else i - 1
        program = Program(i, f"def solve(x): return {i} * x  # code_{i}", f"design_{i}", "Init" if i == 1 else "Develop",
                          parent, Evaluation("success", 10. - i), f"analysis_{i}", i, parent)
        archive[i] = program
        attempts.append(Attempt(i, parent, program.operator, program_id=i, version_ids=[i],
            evaluations=[{"status": "success"}], completed=True))
    prompt = PromptBuilder(task, Config(), counter).generation(archive, attempts, archive[3], [])
    assert prompt.metadata["main_attempt_ids"] == [2, 3]
    assert prompt.metadata["alternative_attempt_ids"] == [4, 5]
    assert prompt.metadata["visible_code_version_ids"] == [3]
    assert "code_3" in prompt.text
    assert not any(f"code_{i}" in prompt.text for i in (1, 2, 4, 5))
    assert all(f"design_{i}" in prompt.text and f"analysis_{i}" in prompt.text for i in (2, 4, 5))
