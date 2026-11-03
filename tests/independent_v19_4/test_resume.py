import json

import pytest

from traceaad_v19_4 import Config, TraceAAD
import traceaad_v19_4.search as module


EMPTY = '{"reflection": "No reusable hint."}'


def test_boundary_resume_reproduces_parent_memory_and_prompt(task, model, reply, reflection, counter, tmp_path):
    responses = [reply(5), reply(4), reflection, reply(3), reflection, reply(2), EMPTY]
    config = Config(selection_mode="neighborhood", evaluation_budget=4, initial_size=1, seed=27)
    full_model = model(responses)
    with TraceAAD(task, full_model, config=config, token_counter=counter) as full:
        expected = full.run()
        archive = [p.to_dict() for p in full.archive.values()]
        memories = full.pool.to_list()
        expected_responses = list(full.responses)
    with TraceAAD(task, model(responses[:3]), config=config, token_counter=counter, run_dir=tmp_path) as first:
        first.step()
        first.step()
    next_model = model(responses[3:])
    with TraceAAD(task, next_model, token_counter=counter, run_dir=tmp_path, resume=True) as resumed:
        assert resumed.run().to_dict() == expected.to_dict()
        assert [p.to_dict() for p in resumed.archive.values()] == archive
        assert resumed.pool.to_list() == memories
        assert resumed.responses == expected_responses
        assert next_model.prompts == full_model.prompts[3:]


@pytest.mark.parametrize("phase", ["generate", "evaluate", "evaluation_record", "response_record", "reflect_before", "reflect_inflight", "reflect_reply", "reflect_committed"])
def test_interruption_resumes_learning_without_replaying_requests(
    task, model, reply, reflection, counter, tmp_path, monkeypatch, phase
):
    config = Config(selection_mode="neighborhood", evaluation_budget=2, initial_size=1)
    responses = [reply(5), KeyboardInterrupt()] if phase == "generate" else [reply(5), reply(3), reflection]
    if phase == "reflect_inflight":
        responses[-1] = KeyboardInterrupt()
    with TraceAAD(task, model(responses), config=config, token_counter=counter, run_dir=tmp_path) as first:
        first.step()
        with monkeypatch.context() as patch:
            if phase == "evaluate":
                patch.setattr(first.evaluator, "evaluate", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
            elif phase in {"evaluation_record", "response_record"}:
                write = first.store.write
                def interrupt_record(path, data):
                    if path == ("evaluations/000002.json" if phase == "evaluation_record" else "responses/000002.json"):
                        raise KeyboardInterrupt()
                    return write(path, data)
                patch.setattr(first.store, "write", interrupt_record)
            elif phase == "reflect_before":
                patch.setattr(first, "_reflect", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
            elif phase == "reflect_reply":
                patch.setattr(module, "parse_reflection", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
            elif phase == "reflect_committed":
                save = first._save
                def interrupt_after_commit():
                    save()
                    a = first.attempts[-1]
                    if a.phase == "finish" and a.reflection_applied:
                        raise KeyboardInterrupt()
                patch.setattr(first, "_save", interrupt_after_commit)
            with pytest.raises(KeyboardInterrupt):
                first.step()
        spent_calls, spent_evals = first.model_calls, first.evaluations_used
    responses = [reply(2), EMPTY] if phase == "generate" else [reflection] if phase in {"reflect_before", "response_record"} else []
    next_model = model(responses)
    with TraceAAD(task, next_model, token_counter=counter, run_dir=tmp_path, resume=True) as resumed:
        assert (resumed.model_calls, resumed.evaluations_used) == (spent_calls, spent_evals)
        result = resumed.run()
        if phase == "generate":
            assert len(next_model.prompts) == 2 and result.model_calls == 4
        elif phase in {"reflect_before", "response_record"}:
            assert len(next_model.prompts) == 1 and result.model_calls == 3
            assert len(resumed.responses) == 1
        else:
            assert not next_model.prompts
        if phase in {"evaluate", "evaluation_record"}:
            assert result.best.evaluation.fitness == 5 and resumed.attempts[1].reflection is None
            assert json.loads((tmp_path / "evaluations/000002.json").read_text())["status"] == "interrupted"
        elif phase not in {"generate", "reflect_inflight"}:
            assert resumed.attempts[1].reflection and resumed.attempts[1].reflection_applied
        if phase == "reflect_inflight":
            assert resumed.attempts[1].reflection_status == "interrupted"
    with TraceAAD(task, model([]), token_counter=counter, run_dir=tmp_path, resume=True) as completed:
        assert completed.run().model_calls == result.model_calls


@pytest.mark.parametrize("phase", ["long_term_inflight", "long_term_reply", "long_term_committed"])
def test_interrupted_long_term_reflection_resumes_without_replaying(
    task, model, reply, reflection, experience, counter, tmp_path, monkeypatch, phase
):
    config = Config(selection_mode="neighborhood", evaluation_budget=2, initial_size=1, long_term_interval=1)
    responses = [reply(5), reply(3), reflection, KeyboardInterrupt() if phase == "long_term_inflight" else experience]
    with TraceAAD(task, model(responses), config=config, token_counter=counter, run_dir=tmp_path) as first:
        first.step()
        with monkeypatch.context() as patch:
            if phase == "long_term_reply":
                patch.setattr(module, "parse_experience", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
            elif phase == "long_term_committed":
                save = first._save
                def interrupt_after_commit():
                    save()
                    if first.attempts[-1].long_term_status == "completed":
                        raise KeyboardInterrupt()
                patch.setattr(first, "_save", interrupt_after_commit)
            with pytest.raises(KeyboardInterrupt):
                first.step()
    next_model = model([])
    with TraceAAD(task, next_model, token_counter=counter, run_dir=tmp_path, resume=True) as resumed:
        result = resumed.run()
        assert not next_model.prompts and result.model_calls == 4
        if phase == "long_term_inflight":
            assert resumed.attempts[1].long_term_status == "interrupted" and not resumed.pool.versions
        else:
            assert resumed.attempts[1].long_term_status == "completed"
            assert [v["reflection_attempt_ids"] for v in resumed.pool.versions] == [[2]]
            assert len(resumed.pool.current) == 1


def test_new_method_and_compatibility_are_checked(task, model, reply, counter, tmp_path):
    with TraceAAD(task, model([reply(5)]), config=Config(selection_mode="neighborhood", evaluation_budget=1), token_counter=counter, run_dir=tmp_path) as search:
        search.run()
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text())
    state["method"] = "traceaad_v19_2"
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="own checkpoint"):
        TraceAAD(task, model([]), token_counter=counter, run_dir=tmp_path, resume=True)
    state["method"] = "traceaad_v19_4"
    state["compatibility"]["model"] = {"model": "different"}
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="matching"):
        TraceAAD(task, model([]), token_counter=counter, run_dir=tmp_path, resume=True)


def test_usage_responses_and_experience_commit_happen_once(task, model, reply, reflection, lesson, counter, tmp_path, monkeypatch):
    with TraceAAD(task, model([reply(5), reply(3)]), config=Config(selection_mode="neighborhood", evaluation_budget=2, initial_size=1),
                  token_counter=counter, run_dir=tmp_path) as first:
        first.step()
        first.pool.replace([{"polarity": "positive", "content": lesson}], {"attempt_id": 0, "reflection_attempt_ids": []})
        with monkeypatch.context() as patch:
            patch.setattr(first, "_reflect", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
            with pytest.raises(KeyboardInterrupt):
                first.step()
        assert first.pool.entries[1].provided_count == 1
        assert len(first.responses) == 1 and first.responses[0].response == 1
    with TraceAAD(task, model([reflection]), token_counter=counter, run_dir=tmp_path, resume=True) as resumed:
        resumed.run()
        assert resumed.pool.entries[1].provided_count == 1
        assert len(resumed.responses) == 1 and resumed.responses[0].response == 1
        assert len(resumed.pool.entries[1].outcomes) == 1
        assert len(resumed.pool.entries) == 1


@pytest.mark.parametrize("phase", ["select_inflight", "select_reply"])
def test_interrupted_parent_choice_resumes_without_replaying(
    task, model, reply, reflection, counter, tmp_path, monkeypatch, phase
):
    config = Config(evaluation_budget=3, initial_size=1)
    choose = json.dumps({"assessment": "P1 leaves more room.", "choice": 1})
    responses = [reply(5), reply(3), reflection, KeyboardInterrupt() if phase == "select_inflight" else choose]
    with TraceAAD(task, model(responses), config=config, token_counter=counter, run_dir=tmp_path) as first:
        first.step()
        first.step()
        with monkeypatch.context() as patch:
            if phase == "select_reply":
                patch.setattr(module, "parse_selection", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
            with pytest.raises(KeyboardInterrupt):
                first.step()
    next_model = model([reply(4), EMPTY])
    with TraceAAD(task, next_model, token_counter=counter, run_dir=tmp_path, resume=True) as resumed:
        resumed.run()
        attempt = resumed.attempts[2]
        assert len(next_model.prompts) == 2 and "[Candidate Programs]" not in next_model.prompts[0]
        if phase == "select_inflight":
            assert attempt.selection["choice_status"] == "interrupted"
            assert attempt.parent_id == attempt.selection["fallback_id"]
        else:
            assert attempt.selection["choice_status"] == "chosen" and attempt.parent_id == 1
