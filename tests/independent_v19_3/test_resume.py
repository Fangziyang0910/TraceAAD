import json

import pytest

from traceaad_v19_3 import Config, TraceAAD
import traceaad_v19_3.search as module


EMPTY = '{"positive": [], "negative": []}'


def test_boundary_resume_reproduces_parent_memory_and_prompt(task, model, reply, reflection, counter, tmp_path):
    responses = [reply(5), reply(4), reflection, reply(3), reflection, reply(2), EMPTY]
    config = Config(evaluation_budget=4, initial_size=1, seed=27)
    full_model = model(responses)
    with TraceAAD(task, full_model, config=config, token_counter=counter) as full:
        expected = full.run()
        archive = [p.to_dict() for p in full.archive.values()]
        memories = full.pool.to_list()
    with TraceAAD(task, model(responses[:3]), config=config, token_counter=counter, run_dir=tmp_path) as first:
        first.step()
        first.step()
    next_model = model(responses[3:])
    with TraceAAD(task, next_model, token_counter=counter, run_dir=tmp_path, resume=True) as resumed:
        assert resumed.run().to_dict() == expected.to_dict()
        assert [p.to_dict() for p in resumed.archive.values()] == archive
        assert resumed.pool.to_list() == memories
        assert next_model.prompts == full_model.prompts[3:]


@pytest.mark.parametrize("phase", ["generate", "evaluate", "evaluation_record", "reflect_before", "reflect_inflight", "reflect_reply", "reflect_committed"])
def test_interruption_resumes_learning_without_replaying_requests(
    task, model, reply, reflection, counter, tmp_path, monkeypatch, phase
):
    config = Config(evaluation_budget=2, initial_size=1)
    responses = [reply(5), KeyboardInterrupt()] if phase == "generate" else [reply(5), reply(3), reflection]
    if phase == "reflect_inflight":
        responses[-1] = KeyboardInterrupt()
    with TraceAAD(task, model(responses), config=config, token_counter=counter, run_dir=tmp_path) as first:
        first.step()
        with monkeypatch.context() as patch:
            if phase == "evaluate":
                patch.setattr(first.evaluator, "evaluate", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
            elif phase == "evaluation_record":
                write = first.store.write
                def interrupt_record(path, data):
                    if path == "evaluations/000002.json":
                        raise KeyboardInterrupt()
                    return write(path, data)
                patch.setattr(first.store, "write", interrupt_record)
            elif phase == "reflect_before":
                patch.setattr(first, "_reflect", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
            elif phase == "reflect_reply":
                patch.setattr(module, "parse_lessons", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
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
    responses = [reply(2), EMPTY] if phase == "generate" else [reflection] if phase == "reflect_before" else []
    next_model = model(responses)
    with TraceAAD(task, next_model, token_counter=counter, run_dir=tmp_path, resume=True) as resumed:
        assert (resumed.model_calls, resumed.evaluations_used) == (spent_calls, spent_evals)
        result = resumed.run()
        if phase == "generate":
            assert len(next_model.prompts) == 2 and result.model_calls == 4
        elif phase == "reflect_before":
            assert len(next_model.prompts) == 1 and result.model_calls == 3
        else:
            assert not next_model.prompts
        if phase in {"evaluate", "evaluation_record"}:
            assert result.best.evaluation.fitness == 5 and len(resumed.pool.entries) == 0
            assert json.loads((tmp_path / "evaluations/000002.json").read_text())["status"] == "interrupted"
        elif phase not in {"generate", "reflect_inflight"}:
            assert len(resumed.pool.entries) == 1 and resumed.attempts[1].reflection_applied
        if phase == "reflect_inflight":
            assert resumed.attempts[1].reflection_status == "interrupted"
    with TraceAAD(task, model([]), token_counter=counter, run_dir=tmp_path, resume=True) as completed:
        assert completed.run().model_calls == result.model_calls


def test_new_method_and_compatibility_are_checked(task, model, reply, counter, tmp_path):
    with TraceAAD(task, model([reply(5)]), config=Config(evaluation_budget=1), token_counter=counter, run_dir=tmp_path) as search:
        search.run()
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text())
    state["method"] = "traceaad_v19_2"
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="own checkpoint"):
        TraceAAD(task, model([]), token_counter=counter, run_dir=tmp_path, resume=True)
    state["method"] = "traceaad_v19_3"
    state["compatibility"]["model"] = {"model": "different"}
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="matching"):
        TraceAAD(task, model([]), token_counter=counter, run_dir=tmp_path, resume=True)


def test_utility_and_experience_commit_happen_once(task, model, reply, reflection, lesson, counter, tmp_path, monkeypatch):
    with TraceAAD(task, model([reply(5), reply(3)]), config=Config(evaluation_budget=2, initial_size=1),
                  token_counter=counter, run_dir=tmp_path) as first:
        first.step()
        first.pool.add([{"polarity": "positive", "content": lesson}], {"attempt_id": 0, "root_id": 999})
        with monkeypatch.context() as patch:
            patch.setattr(first, "_reflect", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
            with pytest.raises(KeyboardInterrupt):
                first.step()
        assert first.pool.entries[1].score == 1
    with TraceAAD(task, model([reflection]), token_counter=counter, run_dir=tmp_path, resume=True) as resumed:
        resumed.run()
        assert resumed.pool.entries[1].provided_count == 1 and resumed.pool.entries[1].score == 1
        assert len(resumed.pool.entries[1].outcomes) == 1
        assert len(resumed.pool.entries) == 1
