from dataclasses import FrozenInstanceError
import json
import random

import pytest

from traceaad_v19_4 import Config, Evaluation, Program, ResponseEvent, TraceAAD
from traceaad_v19_4.selection import candidates, node_distribution, proposals, rank_distribution


EMPTY = '{"positive":[],"negative":[]}'


def archive_for(scores, parents=None):
    return {pid: Program(pid, f"def solve(x): return {pid}", "rule", "Develop" if (parents or {}).get(pid) else "Init",
                         (parents or {}).get(pid, 0), Evaluation("success", score)) for pid, score in scores.items()}


def event(index, source, success, gain=1.):
    return ResponseEvent(index, index, source, index + 1000, bool(success), bool(success), bool(success), int(success),
                         gain if success else 0.)


def details(archive, events, **kwargs):
    return {p.id: {**row, "probability": probability} for p, probability, row in node_distribution(archive, events, **kwargs)}


def test_rank_mode_is_the_quality_baseline_and_no_history_is_uniform():
    archive = archive_for({1: -5., 2: -5., 3: 2.})
    expected = {p.id: probability for p, probability in rank_distribution(archive)}
    assert {pid: row["probability"] for pid, row in details(archive, [], mode="rank").items()} == pytest.approx(expected)
    for mode in ("direct", "neighborhood"):
        rows = details(archive, [], mode=mode)
        assert all(row["response_rate"] == .5 and row["reach"] == 1 for row in rows.values())
        assert all(row["probability"] == pytest.approx(1 / 3) for row in rows.values())


def test_direct_response_separates_equal_fitness_nodes():
    archive = archive_for({1: 10., 2: 10.})
    events = [event(i + 1, 1, i < 3) for i in range(4)] + [event(i + 5, 2, False) for i in range(4)]
    rows = details(archive, events, mode="direct")
    base = (1 + 3) / (2 + 8)
    assert rows[1]["base_rate"] == pytest.approx(base)
    assert rows[1]["response_rate"] == pytest.approx((2 * base + 3) / 6)
    assert rows[2]["response_rate"] == pytest.approx(2 * base / 6)
    assert rows[1]["value"] / rows[2]["value"] == pytest.approx((2 * base + 3) / (2 * base))


def test_reach_is_the_share_of_recent_gains_that_exceed_the_gap():
    archive = archive_for({1: 10., 2: 11., 3: 11.5, 4: 13.})
    events = [event(1, 1, True, .5), event(2, 1, True, 1.5), event(3, 1, True, 2.5), event(4, 1, False)]
    rows = details(archive, events)
    assert rows[1]["gap"] == 0 and rows[1]["reach"] == 1
    assert rows[2]["reach"] == pytest.approx((1 + 2) / (1 + 3))
    assert rows[3]["reach"] == pytest.approx((1 + 1) / (1 + 3))
    assert rows[4]["reach"] == pytest.approx(1 / (1 + 3))


def test_local_continuation_informs_ancestor_without_changing_its_label():
    archive = archive_for({1: 10., 2: 11., 3: 9.}, {2: 1, 3: 2})
    events = [event(1, 1, False), event(2, 2, True)]
    rows, direct = details(archive, events), details(archive, events, mode="direct")
    assert rows[1]["response_rate"] == pytest.approx(4 / 9)
    assert direct[1]["response_rate"] == pytest.approx(1 / 3)
    assert rows[1]["direct_improvements"] == 0 and events[0].response == 0
    assert rows[1]["neighbor_event_ids"] == [2]
    assert rows[3]["direct_count"] == 0 and rows[3]["response_rate"] == pytest.approx(2 / 3)
    with pytest.raises(FrozenInstanceError):
        events[0].response = 1


def test_large_neighborhood_stays_a_fixed_strength_reference():
    archive = archive_for({1: 10., 2: 10.}, {2: 1})
    events = [event(i + 1, 1, False) for i in range(40)] + [event(i + 41, 2, True) for i in range(100)]
    row = details(archive, events)[1]
    base = 101 / 142
    assert row["response_rate"] == pytest.approx(2 * ((2 * base + 100) / 102) / 42)
    assert row["response_rate"] < 2 / 42
    assert set(row["direct_event_ids"]).isdisjoint(row["neighbor_event_ids"])


def test_response_window_and_event_identity():
    archive = archive_for({1: 5.})
    events = [event(1, 1, True), event(2, 1, False), event(3, 1, False)]
    row = details(archive, events + events, window=2)[1]
    assert row["direct_event_ids"] == [2, 3]
    assert row["direct_count"] == 2 and row["direct_improvements"] == 0
    with pytest.raises(ValueError, match="Conflicting"):
        node_distribution(archive, [events[0], event(1, 1, False)])


def test_duplicate_versions_add_no_node_or_neighborhood_edge():
    archive = archive_for({1: 5., 2: 4.}, {2: 1})
    archive[3] = Program(3, archive[1].code, "copy", "Develop", 2, Evaluation("success", 5.), duplicate_of=1)
    assert [p.id for p in candidates(archive)] == [2, 1]
    rows = details(archive, [])
    assert len(rows) == 2 and rows[1]["neighbor_count"] == 0


def test_random_forests_obey_probability_and_response_bounds():
    rng = random.Random(194)
    for _ in range(50):
        n = rng.randrange(1, 35)
        archive = archive_for({pid: rng.randrange(-3, 8) for pid in range(1, n + 1)},
                              {pid: rng.randrange(pid) for pid in range(1, n + 1)})
        events = [event(i + 1, rng.randrange(1, n + 1), rng.randrange(2)) for i in range(rng.randrange(500))]
        rows = details(archive, events)
        assert sum(row["probability"] for row in rows.values()) == pytest.approx(1.)
        assert all(0 < row["response_rate"] < 1 and 0 < row["reach"] <= 1
                   and row["probability"] >= .05 / n for row in rows.values())


def test_actual_repair_creates_two_costed_responses(task, model, reply, counter, tmp_path):
    llm = model([reply(5), reply(body="raise ValueError('bad')"), reply(3), EMPTY])
    with TraceAAD(task, llm, config=Config(selection_mode="neighborhood", evaluation_budget=3, initial_size=1), token_counter=counter, run_dir=tmp_path) as search:
        search.run()
        assert [e.response for e in search.responses] == [0, 1]
        assert [e.evaluation_id for e in search.responses] == [2, 3]
        assert [e.selected_parent_id for e in search.responses] == [1, 1]
        assert json.loads((tmp_path / "responses/000003.json").read_text())["response"] == 1
        assert len(search.attempts[1].version_ids) == 2


def test_new_equal_fitness_code_remains_eligible_but_is_not_an_improvement(task, model, reply, counter):
    llm = model([reply(5), reply(body="return 5 + 0"), EMPTY])
    with TraceAAD(task, llm, config=Config(selection_mode="neighborhood", evaluation_budget=2, initial_size=1), token_counter=counter) as search:
        result = search.run()
        assert result.valid_programs == 2 and search.responses[0].response == 0
        assert search.responses[0].first_valid_source and not search.responses[0].improved_parent
        assert result.best.id == 1


def test_empty_initial_archive_continues_until_a_valid_start(task, model, reply, counter):
    llm = model([reply(body="raise ValueError('bad')"), reply(body="raise ValueError('bad')"), reply(5), reply(3), EMPTY])
    with TraceAAD(task, llm, config=Config(selection_mode="neighborhood", evaluation_budget=4, initial_size=1), token_counter=counter) as search:
        search.run()
        assert [a.operator for a in search.attempts] == ["Init", "Init", "Develop"]
        assert len(search.responses) == 1 and search.responses[0].response == 1


@pytest.mark.parametrize("settings", [{"response_window": 0}, {"response_strength": float('nan')},
    {"response_strength": True}, {"generation_history": 0}, {"selection_mode": "ucb"}])
def test_invalid_node_settings_fail_before_search(settings):
    with pytest.raises(ValueError):
        Config(**settings)


def choice(pid, assessment="P2 improved on its initial design."):
    return json.dumps({"assessment": assessment, "choice": pid})


def test_proposals_draw_distinct_programs_with_rank_weights():
    archive = archive_for({1: 9., 2: 3., 3: 7., 4: 5., 5: 8.})
    for seed in range(20):
        assert len({p.id for p in proposals(archive, random.Random(seed), 3)}) == 3
    assert {p.id for p in proposals(archive, random.Random(0), 9)} == {1, 2, 3, 4, 5}
    assert [p.id for p in proposals(archive_for({1: 4.}), random.Random(0), 4)] == [1]
    drawn = [p.id for seed in range(600) for p in proposals(archive, random.Random(seed), 2)]
    assert drawn.count(2) > drawn.count(4) > drawn.count(3) > drawn.count(5) > drawn.count(1) > 0


def test_parent_choice_reads_local_trajectories_and_develops_the_chosen_program(
        task, model, reply, reflection, counter, tmp_path):
    llm = model([reply(5), reply(3), reflection, choice(1), reply(4), EMPTY])
    with TraceAAD(task, llm, config=Config(evaluation_budget=3, initial_size=1), token_counter=counter,
                  run_dir=tmp_path) as search:
        search.run()
        assert [c["role"] for c in search.attempts[2].calls] == ["selector", "generator", "reflector"]
        attempt = search.attempts[2]
        assert sorted(attempt.selection["candidate_ids"]) == [1, 2] and attempt.parent_id == 1
        assert attempt.selection["choice_status"] == "chosen"
        assert attempt.selection["assessment"] == "P2 improved on its initial design."
        assert search.attempts[1].selection["candidate_ids"] == [1] and len(search.attempts[1].calls) == 2
        prompt = " ".join(llm.prompts[3].split())
        assert "Best fitness" not in prompt
        assert "[Program P1] fitness=5.0 Design: A bounded lookahead decision rule. How it was formed: An initial design." in prompt
        assert "Attempts made from it so far: 1. Attempt 2. Selected parent: P1; fitness=5.0." in prompt
        assert "[Program P2] fitness=3.0" in prompt and "Attempts made from it so far: 0." in prompt
        assert "Attempt 2 reflection: Bounded lookahead improved the greedy rule." in prompt
        assert "```python" not in prompt and "return 3" not in prompt
        assert "[Current Algorithm]\nP1:\nfitness=5.0" in llm.prompts[4]
        assert llm.settings[3]["response_format"]["json_schema"]["schema"]["required"] == ["assessment", "choice"]
        raw = json.loads((tmp_path / "calls/000004.json").read_text())
        assert raw["role"] == "selector" and sorted(raw["context"]["candidate_ids"]) == [1, 2]


@pytest.mark.parametrize("answer, status", [(choice(7), "format_failure"), (RuntimeError("down"), "error")])
def test_failed_parent_choice_develops_the_proposed_fallback(task, model, reply, reflection, counter, answer, status):
    llm = model([reply(5), reply(3), reflection, answer, reply(4), EMPTY])
    with TraceAAD(task, llm, config=Config(evaluation_budget=3, initial_size=1), token_counter=counter) as search:
        search.run()
        attempt = search.attempts[2]
        assert attempt.selection["choice_status"] == status
        assert attempt.parent_id == attempt.selection["fallback_id"] in (1, 2)
        assert attempt.program_id is not None
