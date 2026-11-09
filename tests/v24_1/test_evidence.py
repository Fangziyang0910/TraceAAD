"""Reflection must see the evaluated computations and complete related evidence."""

import pytest

from traceaad_base import Evaluation, Program
from traceaad_v21.representation import bare
from traceaad_v24_1.evidence import Event, build, fragment, render


def program(i, source, parent=None, score=1., failure=None):
    evaluation = (Evaluation("error", failure=failure) if failure is not None
                  else Evaluation("success", score))
    return Program(i, source, f"description {i}", "Refine" if parent else "Init",
                   parent.id if parent else None, None, evaluation, analysis=f"proposal {i}")


def source(k):
    return f"def solve(x):\n    return x + {k}\n"


def test_actual_implementations_survive_small_edits_rewrites_and_execution_failures():
    filler = "\n".join(f"    v{i} = x + {i}" for i in range(30))
    before = program(1, f"def solve(x):\n{filler}\n    return x + 100\n", score=10.)
    edited = program(2, before.code.replace("x + 100", "x + 101"), before, score=9.)
    rewritten = program(3, "def solve(x):\n    return sum(i * i for i in range(x))\n", before, score=12.)
    failed = program(4, "def solve(x):\n    temporary = []\n    return temporary[1]\n", before,
                     failure='Traceback (most recent call last):\n  File "<candidate>", line 3, in solve\nIndexError: list index out of range')
    events = {7: Event(7, before, edited), 8: Event(8, before, rewritten), 9: Event(9, before, failed)}
    text = render(events, events, events, [])
    assert text.count(bare(before.code)) == 1
    assert "-    return x + 100" in text and "+    return x + 101" in text
    assert bare(rewritten.code) in text and failed.code in text
    assert "IndexError: list index out of range" in text and "most of the code was replaced" not in text
    assert "Event 7: Algorithm 1 -> Algorithm 2" in text


def test_connected_changes_reuse_the_intermediate_implementation_once():
    a = program(10, source(1), score=9.)
    b = program(11, source(2), a, score=8.)
    c = program(12, source(3), b, score=7.)
    events = {20: Event(20, a, b), 21: Event(21, b, c)}
    text = render(events, events, [21], [20])
    assert text.count(bare(b.code)) == 1
    assert "Algorithm 10 -> Algorithm 11" in text and "Algorithm 11 -> Algorithm 12" in text
    assert text.index("Event 20:") < text.index("Event 21:")


def test_continuous_small_edit_still_shows_its_diff_when_child_code_is_already_present():
    filler = "\n".join(f"    v{i} = x + {i}" for i in range(30))
    a = program(1, f"def solve(x):\n{filler}\n    return x + 100\n")
    b = program(2, a.code.replace("return x + 100", "return x + 101"), a)
    c = program(3, b.code.replace("return x + 101", "return x + 102"), b)
    events = {1: Event(1, a, b), 2: Event(2, b, c)}
    text = render(events, events, events, [])
    assert text.count(bare(b.code)) == 1
    assert "-    return x + 100" in text and "+    return x + 101" in text


def test_related_fragment_keeps_formation_best_and_recent_distinct_same_start_attempts():
    a = program(10, source(1))
    b = program(11, source(2), a)
    best = program(12, source(3), b, score=.1)
    duplicate = program(13, best.code, b, score=.1)
    recent = program(14, source(4), b, score=2.)
    current = program(15, source(5), b)
    events = {1: Event(1, None, a), 10: Event(10, a, b), 20: Event(20, b, best),
              21: Event(21, b, duplicate), 22: Event(22, b, recent), 30: Event(30, b, current)}
    assert fragment(events, 30) == {10, 20, 22, 30}
    # The one-hop rule does not recursively add the grandparent's formation event.
    assert 1 not in fragment(events, 30)


def test_only_equal_observations_merge_constants_and_equal_score_new_code_remain():
    a = program(1, source(1))
    a_copy = program(2, "def solve(x):\n    # same computation\n    return x + 1\n")
    b = program(3, source(2), a)
    duplicate = program(4, source(2), a_copy)
    parameter = program(5, source(3), a)
    changed_feedback = program(6, source(2), a, score=2.)
    events = {7: Event(7, a, b), 8: Event(8, a_copy, duplicate),
              9: Event(9, a, parameter), 10: Event(10, a, changed_feedback)}
    text = render(events, events, events, [])
    assert "Observation events: [7, 8]; occurrences: 2" in text
    assert "Observation events: [9]; occurrences: 1" in text
    assert "Observation events: [10]; occurrences: 1" in text
    assert bare(parameter.code) in text and "Algorithm 2 -> Algorithm 4" in text


def test_same_start_best_is_not_lost_when_new_attempt_repeats_it():
    parent = program(1, source(1))
    best = program(2, source(2), parent, score=.1)
    recent = program(3, source(3), parent, score=.5)
    repeated = program(4, source(2), parent, score=.1)
    events = {1: Event(1, None, parent), 2: Event(2, parent, best),
              3: Event(3, parent, recent), 4: Event(4, parent, repeated)}
    ids = fragment(events, 4)
    assert ids == {1, 2, 3, 4}
    assert "Observation events: [2, 4]; occurrences: 2" in render(events, ids, [4], [])


def test_budget_packs_complete_fragments_and_keeps_all_active_sources():
    a = program(1, source(1))
    b = program(2, source(2), a)
    unrelated = program(3, "def solve(x):\n    return sum(x for _ in range(99))\n")
    c = program(4, source(3), b)
    d = program(5, "def solve(x):\n" + "\n".join(f"    v{i} = x + {i}" for i in range(100))
                + "\n    return x\n", b)
    events = {1: Event(1, a, b), 2: Event(2, None, unrelated),
              3: Event(3, b, c), 4: Event(4, b, d)}
    notes = [{"evidence_ids": [2]}]
    first = {1, 2, 3}
    limit = len(render(events, first, [3, 4], [2]).encode())
    material = build(events, [3, 4], notes, lambda text: text, lambda text: len(text.encode()), limit)
    assert set(material.event_ids) == first and material.new_event_ids == (3,)
    assert material.deferred_ids == (4,) and material.reason is None
    blocked = build(events, [3, 4], notes, lambda text: text, len, 1)
    assert blocked.reason == "experience_sources_exceed_input_limit" and blocked.new_event_ids == ()
    with pytest.raises(ValueError, match="source events are missing"):
        build(events, [3], [{"evidence_ids": [999]}], lambda text: text, len, 10000)


def test_six_realistic_size_sources_leave_room_for_the_current_cadence_in_token_units():
    import math
    filler = "\n".join(f"    value_{i} = x + {i}" for i in range(300))
    events, notes = {}, []
    for i in range(1, 7):
        parent = program(2 * i, f"def solve(x):\n{filler}\n    return x + {i}\n")
        child = program(2 * i + 1, parent.code.replace(f"return x + {i}", f"return x + {i + 20}"), parent)
        events[i] = Event(i, parent, child)
        notes.append({"evidence_ids": [i]})
    parent = events[1].child
    for i in range(20, 30):
        child = program(100 + i, parent.code.replace("return x + 21", f"return x + {i + 40}"), parent)
        events[i] = Event(i, parent, child)
    # This test tokenizer has four bytes per token; the serving adapter is tested separately.
    count_tokens = lambda text: math.ceil(len(text.encode()) / 4)
    material = build(events, list(range(20, 30)), notes, lambda text: text, count_tokens, 44544)
    assert len(material.prompt.encode()) > 32256
    assert material.reason is None and material.new_event_ids == tuple(range(20, 30))
    assert set(range(1, 7)) <= set(material.event_ids)
