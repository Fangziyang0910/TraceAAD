"""Reflection binds a focused relation to actual implementations and measured results."""

import pytest

from traceaad_base import Evaluation, Program
from traceaad_v21_1.representation import bare
from traceaad_v24_1.evidence import Event, build, questions, render


def program(i, source, parent=None, score=1., failure=None):
    evaluation = (Evaluation("error", failure=failure) if failure is not None
                  else Evaluation("success", score))
    return Program(i, source, f"description {i}", "Refine" if parent else "Init",
                   parent.id if parent else None, None, evaluation, analysis=f"proposal {i}")


def source(k):
    return f"def solve(x):\n    value = x + {k}\n    return value\n"


def packet(events, pending, notes=(), limit=100000):
    return build(events, pending, notes, lambda related, text: str(related) + text, len, limit)


def test_actual_code_changes_and_results_are_adjacent_under_stable_ids():
    filler = "\n".join(f"    v{i} = x + {i}" for i in range(30))
    before = program(101, f"def solve(x):\n{filler}\n    return x + 100\n", score=10.)
    edited = program(102, before.code.replace("x + 100", "x + 101"), before, score=9.)
    rewritten = program(103, "def solve(x):\n    return sum(i * i for i in range(x))\n", before, score=12.)
    failed = program(104, "def solve(x):\n    temporary = []\n    return temporary[1]\n", before,
                     failure='Traceback (most recent call last):\n  File "<candidate>", line 3, in solve\nIndexError: list index out of range')
    events = {7: Event(7, before, edited), 8: Event(8, before, rewritten), 9: Event(9, before, failed)}
    text = render(events, events, events, [])
    assert text.count(bare(before.code)) == 1
    edited_section = text.split("Attempt 7: Program 101 -> Program 102")[1].split("Attempt 8:")[0]
    assert "-    return x + 100" in edited_section and "+    return x + 101" in edited_section
    assert "parent 10.0 -> child 9.0; improved" in edited_section
    assert bare(rewritten.code) in text and failed.code in text
    assert "parent 10.0 -> child 12.0; regressed" in text
    assert "IndexError: list index out of range" in text and "Code 1" not in text
    assert "description 102" not in text and "proposal 102" not in text


def test_connected_small_edits_show_each_diff_and_intermediate_code_only_once():
    filler = "\n".join(f"    v{i} = x + {i}" for i in range(30))
    a = program(10, f"def solve(x):\n{filler}\n    return x + 100\n")
    b = program(11, a.code.replace("return x + 100", "return x + 101"), a)
    c = program(12, b.code.replace("return x + 101", "return x + 102"), b)
    events = {20: Event(20, a, b), 21: Event(21, b, c)}
    text = render(events, events, [21], [20])
    assert text.count(bare(b.code)) == 1
    assert "-    return x + 100" in text and "+    return x + 101" in text
    assert "-    return x + 101" in text and "+    return x + 102" in text
    assert text.index("Attempt 20:") < text.index("Attempt 21:")
    assert "Program 11: complete implementation shown above." in text


def test_same_start_comparison_targets_overlapping_edits_not_all_programs_in_the_cycle():
    a = program(10, "def solve(x):\n    pool = 1\n    rule = 2\n    return pool * x + rule\n")
    b = program(11, a.code.replace("pool = 1", "pool = 3"), a, score=.5)
    c = program(12, a.code.replace("rule = 2", "rule = 4"), a, score=.1)
    d = program(13, a.code.replace("pool = 1", "pool = 5"), a, score=.8)
    events = {1: Event(1, None, a), 2: Event(2, a, b), 3: Event(3, a, c), 4: Event(4, a, d)}
    material = packet(events, [2, 3, 4])
    assert material.focus["kind"] == "same_start" and material.focus["anchor_attempt"] == 4
    assert material.focus["events"] == [2, 4]
    assert set(material.event_ids) == {1, 2, 4} and material.deferred_ids == (3,)
    assert "Program 12" not in material.prompt


def test_supporting_adjustment_preserves_the_development_instead_of_only_the_last_event():
    a = program(10, source(1))
    b = program(11, source(2), a, score=2.)
    c = program(12, source(3), b, score=.5)
    events = {1: Event(1, None, a), 2: Event(2, a, b), 3: Event(3, b, c)}
    material = packet(events, [3])
    assert material.focus["kind"] == "continuation"
    assert material.event_ids == (2, 3) and material.new_event_ids == (3,)
    assert "parent 1.0 -> child 2.0; regressed" in material.prompt
    assert "parent 2.0 -> child 0.5; improved" in material.prompt


def test_equal_observations_merge_but_parameter_changes_and_changed_feedback_remain():
    a = program(1, source(1))
    a_copy = program(2, "def solve(x):\n    # same computation\n    value = x + 1\n    return value\n")
    b = program(3, source(2), a)
    duplicate = program(4, source(2), a_copy)
    parameter = program(5, source(3), a)
    changed_feedback = program(6, source(2), a, score=2.)
    events = {7: Event(7, a, b), 8: Event(8, a_copy, duplicate),
              9: Event(9, a, parameter), 10: Event(10, a, changed_feedback)}
    text = render(events, events, events, [])
    assert "same implementation and feedback as Attempt 7" in text
    assert "Equivalent observations also recorded at Attempts [8]" in text
    assert "Attempt 9: Program 1 -> Program 5" in text and bare(parameter.code) in text
    assert "Measured result for Attempt 10: parent 1.0 -> child 2.0; regressed" in text


def test_unrelated_old_sources_do_not_block_the_new_question_and_related_sources_are_whole():
    a = program(1, source(1))
    b = program(2, source(2), a)
    c = program(3, source(3), b)
    d = program(4, source(4), c)
    large = program(5, "def solve(x):\n" + "\n".join(f"    v{i} = x + {i}" for i in range(1000))
                    + "\n    return x\n")
    events = {1: Event(1, a, b), 2: Event(2, b, c), 3: Event(3, c, d), 4: Event(4, None, large)}
    notes = [{"id": 1, "evidence_ids": [1, 2]}, {"id": 2, "evidence_ids": [4]}]
    material = packet(events, [3], notes, limit=5000)
    assert material.reason is None and material.related_ids == (1,)
    assert material.event_ids == (1, 2, 3)
    assert "Experience source events: [1, 2]" in material.prompt and "v999" not in material.prompt
    base = packet(events, [3])
    restricted = packet(events, [3], notes, limit=base.input_size)
    assert restricted.reason is None and restricted.event_ids == (2, 3)
    assert restricted.related_ids == ()  # A source relation is never half loaded or editable.
    blocked = packet(events, [3], notes, limit=1)
    assert blocked.reason == "focused_relation_exceeds_input_limit" and blocked.deferred_ids == (3,)
    with pytest.raises(ValueError, match="Missing source events"):
        packet(events, [3], [{"id": 3, "evidence_ids": [999]}])


def test_no_code_change_is_an_observation_and_rewrites_do_not_create_false_same_start_pairs():
    a = program(1, source(1))
    b = program(2, a.code, a)
    c = program(3, "def solve(x):\n    for j in x:\n        yield j ** 2\n", a)
    d = program(4, "def solve(x):\n    return sorted(x, reverse=True)\n", a)
    events = {1: Event(1, a, b), 2: Event(2, a, c), 3: Event(3, a, d)}
    assert packet(events, [1]).reason == "no_new_computational_question"
    assert all(q.kind == "single_change" for q in questions(events, [2, 3]))


@pytest.mark.parametrize("recall", ["content", "exposure"])
def test_independent_branch_recalls_complete_experience_sources(recall):
    a = program(10, "def solve(x):\n    distance = x + 1\n    return distance\n")
    a1 = program(11, "def solve(x):\n    distance = x + 1\n    return distance / 10\n", a)
    b = program(20, "def solve(x):\n    scale = x * 2\n    return scale + 3\n")
    b1 = program(21, "def solve(x):\n    scale = x * 2\n    return scale / 5 + 3\n", b)
    note = {"id": 1, "idea": "Scale normalization", "when": "Scores have different scale.",
            "lesson": "Normalize the scale before combination.", "evidence_ids": [1]}
    if recall == "exposure":
        note = {**note, "idea": "Calibration", "when": "Heterogeneous magnitudes.",
                "lesson": "Calibrate heterogeneous magnitudes."}
    events = {1: Event(1, a, a1), 2: Event(2, b, b1, experience_ids=(1,) if recall == "exposure" else ())}
    material = packet(events, [2], [note])
    assert material.related_ids == (1,) and material.event_ids == (1, 2)
    assert bare(a.code) in material.prompt and bare(b.code) in material.prompt
    assert "Attempt 1: Program 10 -> Program 11" in material.prompt
    assert "Attempt 2: Program 20 -> Program 21" in material.prompt
    if recall == "exposure":
        assert "Experience available during generation: [1]" in material.prompt
    # An oversized cross-branch source leaves the focal case available and the note uneditable.
    base = packet(events, [2])
    limited = packet(events, [2], [note], limit=base.input_size)
    assert limited.related_ids == () and limited.event_ids == (2,)
