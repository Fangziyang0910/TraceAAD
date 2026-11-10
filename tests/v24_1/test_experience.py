"""Local updates preserve minimal evidence relations and protect unrelated experience."""

import json
import os
import subprocess
import sys

import pytest

from traceaad_base import Evaluation, Program
from traceaad_v24_1.evidence import Event
from traceaad_v24_1.notes import Experience, select_notes


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


def test_seventh_experience_is_retained_and_archive_preserves_stable_ids_and_sources():
    experience = Experience()
    apply(experience, response([note(sources=(i,)) for i in range(1, 7)]), set(range(1, 7)))
    apply(experience, response([note(sources=(7,))]), {7}, [])
    assert [n["id"] for n in experience.active] == list(range(1, 8))
    apply(experience, response([note(sources=(8,))], archives=[2]), {2, 8}, [2])
    assert [n["id"] for n in experience.active] == [1, 3, 4, 5, 6, 7, 8]
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


def test_sources_preserve_local_comparisons_and_cross_branch_revisions():
    experience = Experience()
    visible = events([1, 2])  # Two different implementations from the same computation.
    b = visible[2].child
    visible[3] = Event(3, b, program(103, 3, b))
    apply(experience, response([note(sources=(1, 2)), note(sources=(2, 3))]), visible)
    assert [n["evidence_ids"] for n in experience.active] == [[1, 2], [2, 3]]
    unrelated = program(900, 20)
    visible[4] = Event(4, unrelated, program(901, 21, unrelated))
    apply(experience, response([note(1, sources=(1, 4))]), visible)
    assert experience.active[0]["evidence_ids"] == [1, 4]
    assert experience.data["entries"][0]["source_ids"] == [1, 2, 4]
    before = experience.snapshot()
    repeated = program(104, 1, visible[1].parent)
    visible[5] = Event(5, visible[1].parent, repeated)
    with pytest.raises(ValueError, match="distinct observations"):
        apply(experience, response([note(1, sources=(1, 5))]), visible)
    assert experience.snapshot() == before


def test_display_budget_reselects_matching_old_knowledge_and_rotates_retained_entries():
    experience = Experience()
    old = {**note(sources=(1,)), "idea": "Distance normalization", "when": "Distance scores have different scales.",
           "lesson": "Normalize the distance scores before combining them."}
    recent = [{**note(sources=(i,)), "idea": f"Population restart {i}",
               "when": "A population search repeatedly restarts.", "lesson": "Reuse restart candidates."}
              for i in range(2, 9)]
    apply(experience, response([old, *recent[:5]]), set(range(1, 7)))
    apply(experience, response(recent[5:]), {7, 8})
    saved = experience.snapshot()
    # The earlier distance lesson leaves one context, then returns for matching computations.
    first = select_notes(experience.active, "population restart candidates", cursor=7)
    assert len(first) == 6 and 1 not in {n["id"] for n in first}
    returned = select_notes(experience.active, "distance scores normalization scale", cursor=7)
    assert len(returned) == 6 and 1 in {n["id"] for n in returned}
    seen = set()
    for cursor in range(8):
        selected = select_notes(experience.active, "population restart candidates", cursor=cursor)
        assert len({n["id"] for n in selected}) == 6
        seen.update(n["id"] for n in selected)
    assert seen == set(range(1, 9))
    assert experience.snapshot() == saved
    assert select_notes(Experience(saved).active, "distance scores normalization scale", cursor=7) == returned


def test_retrieval_order_is_stable_across_process_hash_seeds():
    script = """
import json
from traceaad_v24_1.notes import ranked_notes, select_notes
topics = ["distance scores", "scores normalization offset", "distance penalty scale",
          "normalization scale offset", "distance scale", "population restart penalty",
          "distance normalization", "scores offset penalty"]
notes = [{"id": i, "idea": text, "when": "scoring", "lesson": "calibrate magnitudes"}
         for i, text in enumerate(topics, 1)]
query = "distance scores normalization scale offset penalty"
print(json.dumps({"ranks": [(score, note["id"]) for score, note in ranked_notes(notes, query)],
                  "shown": [note["id"] for note in select_notes(notes, query, cursor=5)]}))
"""
    outputs = [subprocess.check_output([sys.executable, "-c", script],
                                      env={**os.environ, "PYTHONHASHSEED": seed})
               for seed in ("1", "2", "3")]
    assert outputs[0] == outputs[1] == outputs[2]
