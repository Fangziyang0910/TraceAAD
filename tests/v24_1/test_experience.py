"""Local experience updates preserve conditions, identity and old evidence atomically."""

import json

import pytest

from traceaad_v24_1.notes import Experience


def note(i=None, sources=(1,), when="The minimum is taken over a set containing the candidate itself."):
    return {"id": i, "idea": "Minimum-distance feature", "when": when,
            "lesson": "Exclude the candidate itself before computing a distinguishing minimum.",
            "evidence_ids": list(sources)}


def response(updates=(), archives=()):
    return json.dumps({"analysis": "The implementation revealed the relevant condition.",
                       "updates": list(updates), "archive_ids": list(archives)})


def test_local_update_keeps_other_entries_and_accumulates_sources_beyond_representatives():
    experience = Experience()
    experience.apply(response([note(sources=(1,)), note(sources=(3,))]), {1, 3})
    before = experience.snapshot()
    unchanged = experience.active[1]
    experience.apply(response([note(1, sources=(4,), when="Exclude self before the minimum.")]), {4})
    assert experience.active[1] == unchanged
    assert experience.active[0]["evidence_ids"] == [4]
    assert experience.data["entries"][0]["source_ids"] == [1, 4]
    assert before["entries"][0]["evidence_ids"] == [1]
    revised = experience.snapshot()
    experience.apply(response(), set())
    assert experience.snapshot() == revised
    assert Experience(revised).active == experience.active


def test_capacity_failure_is_atomic_and_archive_does_not_reuse_an_id_or_lose_evidence():
    experience = Experience()
    experience.apply(response([note(sources=(i,)) for i in range(1, 7)]), set(range(1, 7)))
    before = experience.snapshot()
    with pytest.raises(ValueError, match="six active"):
        experience.apply(response([note(sources=(7,))]), {7})
    assert experience.snapshot() == before
    experience.apply(response([note(sources=(7,))], archives=[2]), {7})
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
    experience.apply(response([note()]), {1})
    before = experience.snapshot()
    with pytest.raises(ValueError):
        experience.apply(response([note(sources=(2,)), bad]), {1, 2})
    assert experience.snapshot() == before


def test_merge_can_archive_an_entry_while_preserving_its_sources_and_remaining_identity():
    experience = Experience()
    experience.apply(response([note(sources=(1,)), note(sources=(2,))]), {1, 2})
    experience.apply(response([note(1, sources=(2,))], archives=[2]), {1, 2})
    assert experience.active == [note(1, sources=(2,))]
    assert experience.data["entries"][0]["source_ids"] == [1, 2]
    with pytest.raises(ValueError, match="without archiving"):
        experience.apply(response([note(1)], archives=[1]), {1})
