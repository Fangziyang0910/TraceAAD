import random

import pytest

from traceaad_v19_3 import Config, Evaluation, Program
from traceaad_v19_3.knowledge import ExperiencePool, tokens
from traceaad_v19_3.models import Attempt
from traceaad_v19_3.prompts import ContextOverflow, PromptBuilder
from traceaad_v19_3.selection import rank_distribution


def test_global_bilingual_retrieval_exact_merge_and_retirement(lesson):
    pool = ExperiencePool()
    content = {**lesson, "applicable_when": "构造式算法需要候选筛选与前瞻估计"}
    notes = [{"polarity": "positive", "content": content}]
    assert pool.add(notes, {"root_id": 1}) == [1]
    assert pool.add(notes, {"root_id": 999}) == [1]
    assert pool.retrieve("候选筛选和前瞻估计", 3, random.Random(0))[0]["id"] == 1
    assert len(pool.entries[1].origins) == 2
    assert not pool.retrieve("unrelated_xyz", 3, random.Random(0))
    assert "前瞻" in tokens("前瞻估计")
    pool.feedback([1], attempt_id=8, improved=False)
    pool.feedback([1], attempt_id=8, improved=False)
    assert pool.entries[1].score == -1 and pool.entries[1].provided_count == 1
    assert not pool.retrieve("lookahead", 3, random.Random(0))
    assert pool.add(notes, {"root_id": 3}) == [2]
    assert pool.entries[2].score == 0 and pool.entries[2].active
    restored = ExperiencePool(pool.to_list())
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
    for i in range(1, 6):
        parent = 0 if i == 1 else 1 if i == 5 else i - 1
        program = Program(i, f"def solve(x): return {i}", "bounded lookahead", "Init" if i == 1 else "Develop",
                          parent, Evaluation("success", 10. - i), "change", i, parent)
        archive[i] = program
        attempts.append(Attempt(i, parent, program.operator, program_id=i, version_ids=[i],
            evaluations=[{"status": "success"}], completed=i != 4))
    # P3 -> P4 is the newest evaluated attempt; P1 -> P5 is not its same-parent alternative.
    attempts[4].parent_id = 3
    modes = {}
    for mode in ("local", "siblings", "trajectory"):
        builder = PromptBuilder(task, Config(reflection_mode=mode), counter)
        modes[mode] = builder.reflection(archive, attempts, attempts[3], archive[4]).metadata
    assert modes["local"]["main_attempt_ids"] == [4]
    assert modes["local"]["alternative_attempt_ids"] == []
    assert modes["siblings"]["main_attempt_ids"] == [4]
    assert modes["siblings"]["alternative_attempt_ids"] == [5]
    assert modes["trajectory"]["main_attempt_ids"] == [2, 3, 4]


def test_full_change_near_end_of_long_program_is_not_line_truncated(task, counter):
    code = "def solve(x):\n" + "\n".join(f"    value_{i} = x + {i}" for i in range(200)) + "\n    return value_199\n"
    changed = code.replace("value_199 = x + 199", "value_199 = x + 999")
    archive = {1: Program(1, code, "baseline", "Init", 0, Evaluation("success", 5.), attempt_id=1, edited_from=0),
               2: Program(2, changed, "revised", "Develop", 1, Evaluation("success", 4.), attempt_id=2, edited_from=1)}
    attempts = [Attempt(1, 0, "Init", program_id=1, version_ids=[1], completed=True),
                Attempt(2, 1, "Develop", program_id=2, version_ids=[2])]
    prompt = PromptBuilder(task, Config(), counter).reflection(archive, attempts, attempts[1], archive[2])
    assert "value_199 = x + 199" in prompt.text and "value_199 = x + 999" in prompt.text
    assert prompt.metadata["complete_change_attempt_ids"] == [2]
    assert prompt.metadata["material_tokens"] <= 8192


def test_full_code_and_output_reservation_must_fit(task, counter):
    program = Program(1, "def solve(x):\n" + "    x = x + 1\n" * 1000 + "    return x\n", "rule", "Init", 0,
                      Evaluation("success", 5.))
    builder = PromptBuilder(task, Config(context_window=300, max_tokens=32, reflection_max_tokens=32,
                                        protocol_tokens=8), counter)
    with pytest.raises(ContextOverflow):
        builder.generation({1: program}, [], program, [])
