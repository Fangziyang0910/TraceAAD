from dataclasses import FrozenInstanceError
import json

import pytest

from traceaad_v19_4 import Config, Evaluation, Program, TraceAAD
from traceaad_v19_4.knowledge import ExperiencePool
from traceaad_v19_4.models import Attempt, parse_experience
from traceaad_v19_4.prompts import PromptBuilder
from traceaad_v19_4.selection import candidates


EMPTY = '{"reflection": "No reusable hint."}'


def seed(pool, *lessons):
    pool.replace([{"polarity": "positive", "content": lesson} for lesson in lessons],
                 {"attempt_id": 0, "reflection_attempt_ids": []})


def test_long_term_reflection_merges_short_term_reflections_into_global_pools(
        task, model, reply, reflection, experience, counter, tmp_path):
    llm = model([reply(5), reply(4), reflection, reply(3), reflection, experience])
    config = Config(selection_mode="neighborhood", evaluation_budget=3, initial_size=1, long_term_interval=2)
    with TraceAAD(task, llm, config=config, token_counter=counter, run_dir=tmp_path) as search:
        result = search.run()
        assert (result.evaluations, result.model_calls) == (3, 6)
        assert [c["role"] for a in search.attempts for c in a.calls] == [
            "generator", "generator", "reflector", "generator", "reflector", "long_term_reflector"]
        assert [a.reflection for a in search.attempts[1:]] == ["Bounded lookahead improved the greedy rule."] * 2
        assert search.attempts[2].long_term_status == "completed"
        assert search.pool.versions == [{"version": 1, "attempt_id": 3, "reflection_attempt_ids": [2, 3], "entry_ids": [1]}]
        assert [e.content["summary"] for e in search.pool.current] == ["Bounded lookahead appears promising for greedy decision rules."]
        long_term = llm.prompts[5]
        assert "[Recent Reflections]" in long_term and "not established knowledge" in " ".join(long_term.split())
        assert "Attempt 2: selected parent fitness=5.0; result fitness=4.0; best fitness before this attempt=5.0." in long_term
        assert long_term.count("Reflection: Bounded lookahead improved the greedy rule.") == 2
        assert "Current Positive Design Experience" not in long_term
        assert "Attempt 2 reflection: Bounded lookahead improved the greedy rule." in llm.prompts[3]
        assert [s["max_tokens"] for s in llm.settings] == [16384] * 6
        assert llm.settings[2]["response_format"]["json_schema"]["schema"]["required"] == ["reflection"]
        assert llm.settings[5]["response_format"]["json_schema"]["schema"]["required"] == ["positive", "negative"]
        raw = json.loads((tmp_path / "calls/000003.json").read_text())
        assert raw["response"] == reflection and raw["role"] == "reflector"
        assert raw["context"]["main_attempt_ids"] == [2]


@pytest.mark.parametrize("window, shown", [(2, [2, 3]), (1, [3])])
def test_long_term_rebuilds_pools_from_recent_reflections_only(
        task, model, reply, reflection, experience, counter, window, shown):
    llm = model([reply(5), reply(4), reflection, experience, reply(3), reflection, experience])
    config = Config(selection_mode="neighborhood", evaluation_budget=3, initial_size=1, long_term_interval=1, long_term_window=window)
    with TraceAAD(task, llm, config=config, token_counter=counter) as search:
        search.run()
        second = llm.prompts[6]
        assert [i for i in (2, 3) if f"Attempt {i}: selected parent" in second] == shown
        assert "appears promising" not in second
        assert [v["reflection_attempt_ids"] for v in search.pool.versions] == [[2], shown]
        assert [e.id for e in search.pool.current] == [2]
        assert "not established facts" in " ".join(llm.prompts[4].split())


def test_global_experience_reaches_the_next_generator_and_is_scored_once(task, model, reply, lesson, counter):
    llm = model([reply(5), reply(3)])
    config = Config(selection_mode="neighborhood", evaluation_budget=2, initial_size=1, reflection_mode="none")
    with TraceAAD(task, llm, config=config, token_counter=counter) as search:
        search.step()
        seed(search.pool, {**lesson, "summary": "bounded lookahead"})
        search.run()
        attempt = search.attempts[1]
        assert attempt.selection["retrieved_experience_ids"] == [1]
        assert attempt.context["provided_experience_ids"] == [1]
        assert search.pool.entries[1].provided_count == 1
        assert search.pool.entries[1].outcomes == [{"attempt_id": 2, "improved": True}]
        assert "[Positive Design Experience]\n- bounded lookahead Applies when:" in llm.prompts[1]


def test_rewritten_pools_replace_retrieval_but_keep_history(lesson):
    pool = ExperiencePool()
    seed(pool, lesson)
    pool.replace([{"polarity": "negative", "content": {**lesson, "summary": "Unbounded lookahead times out."}},
                  {"polarity": "negative", "content": {**lesson, "summary": "Unbounded lookahead times out."}}],
                 {"attempt_id": 9, "reflection_attempt_ids": [4, 9]})
    assert [e.id for e in pool.current] == [2] and len(pool.entries) == 2
    assert pool.reflected_attempt_ids == {4, 9}
    assert [e["id"] for e in pool.retrieve("lookahead", 3)] == [2]
    restored = ExperiencePool(pool.to_list(), pool.versions)
    assert restored.retrieve("lookahead", 3) == pool.retrieve("lookahead", 3)


def test_failed_long_term_reflection_keeps_pools_and_retries_later(task, model, reply, reflection, experience, counter):
    llm = model([reply(5), reply(4), reflection, "bad json", reply(3), reflection, experience])
    config = Config(selection_mode="neighborhood", evaluation_budget=3, initial_size=1, long_term_interval=1)
    with TraceAAD(task, llm, config=config, token_counter=counter) as search:
        search.run()
        assert search.attempts[1].long_term_status == "format_failure"
        assert search.attempts[2].long_term_status == "completed"
        assert search.pool.versions[0]["reflection_attempt_ids"] == [2, 3]


def test_malformed_experience_items_are_set_aside(lesson):
    accepted, discarded = parse_experience(json.dumps({"positive": [lesson, {**lesson, "revises": ""}],
                                                       "negative": [{"summary": "", "applicable_when": "x"}]}))
    assert accepted == [{"polarity": "positive", "content": lesson}] and len(discarded) == 2


def test_failed_and_repaired_versions_are_distinct_and_visible(task, model, reply, reflection, counter):
    llm = model([reply(5), reply(body="raise ValueError('broken estimator')"), reply(3), reflection])
    with TraceAAD(task, llm, config=Config(selection_mode="neighborhood", evaluation_budget=3, initial_size=1), token_counter=counter) as search:
        result = search.run()
        attempt = search.attempts[1]
        failed, repaired = (search.archive[i] for i in attempt.version_ids)
        assert failed.id != repaired.id and "raise ValueError" in failed.code
        assert failed.evaluation.status == "error" and repaired.valid
        assert repaired.edited_from == failed.id and repaired.parent_id == 1
        assert result.best.id == repaired.id and result.model_calls == 4
        assert "P2 failure type: ValueError" in llm.prompts[-1] and "broken estimator" not in llm.prompts[-1]
        assert "repair" in llm.prompts[-1] and "| +2 |" in llm.prompts[-1]
        assert "| P2 | generation | P1 | program error | unavailable" in llm.prompts[-1]
        with pytest.raises(FrozenInstanceError):
            failed.code = repaired.code


def test_format_and_program_failures_share_one_repair(task, model, reply, counter):
    llm = model([reply(5), "bad json", reply(body="raise ValueError('bad')"), EMPTY])
    with TraceAAD(task, llm, config=Config(selection_mode="neighborhood", evaluation_budget=2, initial_size=1), token_counter=counter) as search:
        result = search.run()
        assert (result.evaluations, result.model_calls) == (2, 4)
        assert search.attempts[1].repair_used
        assert [c["kind"] for c in search.attempts[1].calls] == ["generate", "format", "reflect"]
        assert len(search.attempts[1].version_ids) == 1
        assert "2-3 sentences" in llm.prompts[2]


def test_last_evaluation_does_not_start_program_repair(task, model, reply, counter):
    llm = model([reply(body="raise RuntimeError('last slot')")])
    with TraceAAD(task, llm, config=Config(selection_mode="neighborhood", evaluation_budget=1), token_counter=counter) as search:
        result = search.run()
        assert result.status == "failed" and result.evaluations == result.model_calls == 1
        assert not search.attempts[0].repair_used


@pytest.mark.parametrize("bad_reflection", ["bad json", RuntimeError("reflector unavailable")])
def test_failed_reflection_preserves_evaluated_algorithm(task, model, reply, counter, bad_reflection):
    llm = model([reply(5), reply(3), bad_reflection])
    with TraceAAD(task, llm, config=Config(selection_mode="neighborhood", evaluation_budget=2, initial_size=1), token_counter=counter) as search:
        result = search.run()
        assert result.status == "finished" and result.best.evaluation.fitness == 3
        assert result.model_calls == 3 and not search.pool.entries and search.attempts[1].reflection is None
        assert search.attempts[1].reflection_status in {"error", "format_failure"}


def test_no_evaluation_neither_scores_experience_nor_calls_reflector(task, model, reply, lesson, counter):
    llm = model([reply(5), RuntimeError("offline"), "bad", "bad"])
    with TraceAAD(task, llm, config=Config(selection_mode="neighborhood", evaluation_budget=3, initial_size=1, max_unevaluated=2), token_counter=counter) as search:
        search.step()
        seed(search.pool, lesson)
        result = search.run()
        assert result.stop_reason == "consecutive_unevaluated_limit"
        assert (result.evaluations, result.model_calls) == (1, 4)
        assert search.pool.entries[1].provided_count == 2
        assert all(outcome["improved"] is None for outcome in search.pool.entries[1].outcomes)
        assert not search.responses
        assert all(c["role"] == "generator" for a in search.attempts for c in a.calls)


def test_duplicate_evaluations_keep_true_edges_but_no_extra_selection_weight(task, model, reply, counter):
    llm = model([reply(5), reply(5), EMPTY])
    with TraceAAD(task, llm, config=Config(selection_mode="neighborhood", evaluation_budget=2, initial_size=1), token_counter=counter) as search:
        result = search.run()
        assert result.evaluations == 2
        assert len(candidates(search.archive)) == 1
        assert search.archive[2].duplicate_of == 1 and search.archive[2].parent_id == 1
        assert search.attempts[1].duplicate_of == 1


def test_new_source_is_reflected_even_at_equal_fitness_but_a_repeat_is_not(task, model, reply, reflection, counter):
    llm = model([reply(5), reply(body="return 5 + 0"), reflection, reply(body="return 5 + 0")])
    with TraceAAD(task, llm, config=Config(selection_mode="neighborhood", evaluation_budget=3, initial_size=1), token_counter=counter) as search:
        search.run()
        assert search.model_calls == 4
        assert [a.reflection_status for a in search.attempts[1:]] == ["completed", "no_new_source"]


def test_short_term_reflection_reads_local_text_only(task, counter):
    archive = {1: Program(1, "def solve(x): return 5  # code_1", "rule", "Init", 0, Evaluation("success", 5.), attempt_id=1, edited_from=0),
               2: Program(2, "def solve(x): return 3  # code_2", "rule two", "Develop", 1, Evaluation("success", 3.), "a", 2, 1),
               3: Program(3, "def solve(x): return 4  # code_3", "rule three", "Develop", 2, Evaluation("success", 4.), "b", 3, 2)}
    attempts = [Attempt(1, 0, "Init", program_id=1, version_ids=[1], completed=True),
                Attempt(2, 1, "Develop", program_id=2, version_ids=[2], completed=True, reflection="Earlier hint."),
                Attempt(3, 2, "Develop", program_id=3, version_ids=[3])]
    prompt = PromptBuilder(task, Config(), counter).reflection(archive, attempts, attempts[2])
    assert "Selected parent P2 design: rule two" in prompt.text and "Attempt 2 reflection: Earlier hint." in prompt.text
    assert "P3 design: rule three" in prompt.text and "P3 analysis: b" in prompt.text
    assert not any(f"code_{i}" in prompt.text for i in (1, 2, 3)) and "Design Experience" not in prompt.text
    assert prompt.metadata["main_attempt_ids"] == [2, 3] and prompt.metadata["alternative_attempt_ids"] == []
