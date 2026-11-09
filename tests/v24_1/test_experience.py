"""Local updates preserve minimal evidence relations and protect unrelated experience."""

import json

import pytest

from traceaad_base import Evaluation, Program
from traceaad_v24_1.evidence import Event
from traceaad_v24_1.notes import Experience


def note(i=None, sources=(1,), when="The minimum is taken over a set containing the candidate itself."):
    return {"id": i, "idea": "Minimum-distance feature", "when": when,
            "lesson": "Exclude the candidate itself before computing a distinguishing minimum.",
            "evidence_ids": list(sources)}


def response(updates=(), archives=()):
    return json.dumps({"analysis": "The implementation revealed the relevant condition.",
                       "updates": list(updates), "archive_ids": list(archives)})


def program(i, k, parent=None):
    return Program(i, f"def solve(x):\n    value = x + {k}\n    return value\n", "A computation.",
                   "Refine" if parent else "Init", parent.id if parent else None, None,
                   Evaluation("success", float(k)))


def events(ids):
    parent = program(100, 0)
    return {i: Event(i, parent, program(100 + i, i, parent)) for i in ids}


def apply(experience, value, ids, editable=None):
    visible = events(ids) if not isinstance(ids, dict) else ids
    return experience.apply(value, visible, editable_ids=(
        {n["id"] for n in experience.active} if editable is None else set(editable)))


def test_local_update_keeps_other_entries_and_accumulates_sources_beyond_representatives():
    experience = Experience()
    apply(experience, response([note(sources=(1, 2)), note(sources=(3,))]), {1, 2, 3})
    before = experience.snapshot()
    unchanged = experience.active[1]
    apply(experience, response([note(1, sources=(2, 4), when="Exclude self before the minimum.")]), {2, 4}, [1])
    assert experience.active[1] == unchanged
    assert experience.active[0]["evidence_ids"] == [2, 4]
    assert experience.data["entries"][0]["source_ids"] == [1, 2, 4]
    assert before["entries"][0]["evidence_ids"] == [1, 2]
    revised = experience.snapshot()
    apply(experience, response(), set())
    assert experience.snapshot() == revised
    assert Experience(revised).active == experience.active


def test_capacity_failure_is_atomic_and_archive_does_not_reuse_an_id_or_lose_evidence():
    experience = Experience()
    apply(experience, response([note(sources=(i,)) for i in range(1, 7)]), set(range(1, 7)))
    before = experience.snapshot()
    with pytest.raises(ValueError, match="six active"):
        apply(experience, response([note(sources=(7,))]), {7})
    assert experience.snapshot() == before
    apply(experience, response([note(sources=(7,))], archives=[2]), {2, 7}, [2])
    assert [n["id"] for n in experience.active] == [1, 3, 4, 5, 6, 7]
    archived = next(n for n in experience.data["entries"] if n["id"] == 2)
    assert not archived["active"] and archived["source_ids"] == [2]


@pytest.mark.parametrize("bad", [
    {**note(1), "when": " "}, note(1, sources=(99,)), note(1, sources=()),
    note(1, sources=(1, 1)), note(10), note(True),
    {**note(1), "evidence_ids": [True]}, {**note(1), "extra": "unused"},
])
def test_invalid_update_cannot_apply_an_earlier_valid_update(bad):
    experience = Experience()
    apply(experience, response([note()]), {1})
    before = experience.snapshot()
    with pytest.raises(ValueError):
        apply(experience, response([note(sources=(2,)), bad]), {1, 2})
    assert experience.snapshot() == before


def test_merge_preserves_sources_and_unrelated_entries_cannot_be_changed_or_archived():
    experience = Experience()
    apply(experience, response([note(sources=(1,)), note(sources=(2,)), note(sources=(3,))]), {1, 2, 3})
    apply(experience, response([note(1, sources=(1, 2))], archives=[2]), {1, 2}, [1, 2])
    assert experience.active == [note(1, sources=(1, 2)), note(3, sources=(3,))]
    assert experience.data["entries"][0]["source_ids"] == [1, 2]
    before = experience.snapshot()
    for value in (response([note(3)]), response(archives=[3]), response([note(1)], archives=[1])):
        with pytest.raises(ValueError):
            apply(experience, value, {1, 3}, [1])
        assert experience.snapshot() == before


def test_connected_source_relations_allow_a_comparison_or_supporting_continuation():
    experience = Experience()
    visible = events([1, 2])  # Two different implementations from the same computation.
    b = visible[2].child
    visible[3] = Event(3, b, program(103, 3, b))
    apply(experience, response([note(sources=(1, 2)), note(sources=(2, 3))]), visible)
    assert [n["evidence_ids"] for n in experience.active] == [[1, 2], [2, 3]]
    unrelated = program(900, 20)
    visible[4] = Event(4, unrelated, program(901, 21, unrelated))
    before = experience.snapshot()
    with pytest.raises(ValueError, match="connected shown"):
        apply(experience, response([note(1, sources=(1, 4))]), visible)
    assert experience.snapshot() == before
    repeated = program(104, 1, visible[1].parent)
    visible[5] = Event(5, visible[1].parent, repeated)
    with pytest.raises(ValueError, match="connected shown"):
        apply(experience, response([note(1, sources=(1, 5))]), visible)
    assert experience.snapshot() == before
