from dataclasses import FrozenInstanceError
import json

import pytest

from traceaad_v19_3 import Config, TraceAAD
from traceaad_v19_3.selection import candidates


EMPTY = '{"positive": [], "negative": []}'


def test_evaluated_experience_reaches_next_generator(task, model, reply, reflection, counter, tmp_path):
    llm = model([reply(5), reply(4), reflection, reply(3), EMPTY])
    with TraceAAD(task, llm, config=Config(evaluation_budget=3, initial_size=1),
                  token_counter=counter, run_dir=tmp_path) as search:
        result = search.run()
        assert (result.evaluations, result.model_calls, result.attempts) == (3, 5, 3)
        assert result.best.evaluation.fitness == 3
        assert [c["role"] for a in search.attempts for c in a.calls] == [
            "generator", "generator", "reflector", "generator", "reflector"]
        assert search.attempts[2].provided_experiences[0]["id"] == 1
        assert search.pool.entries[1].provided_count == 1 and search.pool.entries[1].score == 1
        assert "Positive Design Experience" in llm.prompts[3]
        assert "Adapt bounded lookahead" in llm.prompts[3]
        assert "Latest Evaluated Algorithm" in llm.prompts[2]
        assert "fitness=4" in llm.prompts[2]
        assert [s["max_tokens"] for s in llm.settings] == [16384, 16384, 2048, 16384, 2048]
        assert llm.settings[2]["response_format"]["json_schema"]["schema"]["required"] == ["positive", "negative"]
        raw = json.loads((tmp_path / "calls/000003.json").read_text())
        assert raw["response"] == reflection and raw["role"] == "reflector"
        assert raw["context"]["main_attempt_ids"] == [2]


def test_failed_and_repaired_versions_are_distinct_and_visible(task, model, reply, reflection, counter):
    llm = model([reply(5), reply(body="raise ValueError('broken estimator')"), reply(3), reflection])
    with TraceAAD(task, llm, config=Config(evaluation_budget=3, initial_size=1), token_counter=counter) as search:
        result = search.run()
        attempt = search.attempts[1]
        failed, repaired = (search.archive[i] for i in attempt.version_ids)
        assert failed.id != repaired.id and "raise ValueError" in failed.code
        assert failed.evaluation.status == "error" and repaired.valid
        assert repaired.edited_from == failed.id and repaired.parent_id == 1
        assert result.best.id == repaired.id and result.model_calls == 4
        assert "broken estimator" in llm.prompts[-1]
        assert "repair" in llm.prompts[-1] and "+2 (better)" in llm.prompts[-1]
        assert "| P2 | generation | P1 | program error | unavailable" in llm.prompts[-1]
        with pytest.raises(FrozenInstanceError):
            failed.code = repaired.code


def test_format_and_program_failures_share_one_repair(task, model, reply, counter):
    llm = model([reply(5), "bad json", reply(body="raise ValueError('bad')"), EMPTY])
    with TraceAAD(task, llm, config=Config(evaluation_budget=2, initial_size=1), token_counter=counter) as search:
        result = search.run()
        assert (result.evaluations, result.model_calls) == (2, 4)
        assert search.attempts[1].repair_used
        assert [c["kind"] for c in search.attempts[1].calls] == ["generate", "format", "reflect"]
        assert len(search.attempts[1].version_ids) == 1
        assert "2-3 sentences" in llm.prompts[2]


def test_last_evaluation_does_not_start_program_repair(task, model, reply, counter):
    llm = model([reply(body="raise RuntimeError('last slot')")])
    with TraceAAD(task, llm, config=Config(evaluation_budget=1), token_counter=counter) as search:
        result = search.run()
        assert result.status == "failed" and result.evaluations == result.model_calls == 1
        assert not search.attempts[0].repair_used


@pytest.mark.parametrize("bad_reflection", ["bad json", RuntimeError("reflector unavailable")])
def test_failed_reflection_preserves_evaluated_algorithm(task, model, reply, counter, bad_reflection):
    llm = model([reply(5), reply(3), bad_reflection])
    with TraceAAD(task, llm, config=Config(evaluation_budget=2, initial_size=1), token_counter=counter) as search:
        result = search.run()
        assert result.status == "finished" and result.best.evaluation.fitness == 3
        assert result.model_calls == 3 and not search.pool.entries
        assert search.attempts[1].reflection_status in {"error", "format_failure"}


def test_no_evaluation_neither_scores_experience_nor_calls_reflector(task, model, reply, lesson, counter):
    llm = model([reply(5), RuntimeError("offline"), "bad", "bad"])
    with TraceAAD(task, llm, config=Config(evaluation_budget=3, initial_size=1, max_unevaluated=2), token_counter=counter) as search:
        search.step()
        search.pool.add([{"polarity": "positive", "content": lesson}], {"attempt_id": 0, "root_id": 999})
        result = search.run()
        assert result.stop_reason == "consecutive_unevaluated_limit"
        assert (result.evaluations, result.model_calls) == (1, 4)
        assert search.pool.entries[1].active and search.pool.entries[1].provided_count == 0
        assert all(c["role"] == "generator" for a in search.attempts for c in a.calls)


def test_duplicate_evaluations_keep_true_edges_but_no_extra_selection_weight(task, model, reply, counter):
    llm = model([reply(5), reply(5), EMPTY])
    with TraceAAD(task, llm, config=Config(evaluation_budget=2, initial_size=1), token_counter=counter) as search:
        result = search.run()
        assert result.evaluations == 2
        assert len(candidates(search.archive)) == 1
        assert search.archive[2].duplicate_of == 1 and search.archive[2].parent_id == 1
        assert search.attempts[1].duplicate_of == 1


def test_retrieved_but_unshown_experience_is_not_scored(task, model, reply, lesson, counter):
    llm = model([reply(5), reply(3)])
    config = Config(evaluation_budget=2, initial_size=1, material_tokens=100, reflection_mode="none")
    with TraceAAD(task, llm, config=config, token_counter=counter) as search:
        search.step()
        content = {**lesson, "summary": "bounded lookahead " * 400}
        search.pool.add([{"polarity": "positive", "content": content}], {"attempt_id": 0, "root_id": 99})
        search.run()
        attempt = search.attempts[1]
        assert attempt.selection["retrieved_experience_ids"] == [1]
        assert attempt.context["provided_experience_ids"] == []
        assert search.pool.entries[1].provided_count == 0 and search.pool.entries[1].score == 0


def test_reflection_limit_retains_raw_reply_and_discarded_items(task, model, reply, lesson, counter, tmp_path):
    raw = json.dumps({"positive": [{**lesson, "summary": str(i)} for i in range(4)], "negative": []})
    llm = model([reply(5), reply(3), raw])
    with TraceAAD(task, llm, config=Config(evaluation_budget=2, initial_size=1), token_counter=counter, run_dir=tmp_path) as search:
        search.run()
        assert len(search.pool.entries) == 3
        assert len(search.attempts[1].discarded_lessons) == 1
        assert json.loads((tmp_path / "calls/000003.json").read_text())["response"] == raw
