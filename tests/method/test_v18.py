"""Executable checks of the V18 design, including real isolated toy evaluation."""

from dataclasses import asdict
import json

import pytest

from core.scheduling import SchedulerError
from tests.support import TinyEvaluation, TokenLLM
from traceaad import Program
from traceaad.storage import committed_rows, read_json, rows
from traceaad.v18 import Config, TraceAADV18
from traceaad.v18.prompts import normalized_source, parse_code


def code(value, suffix=""):
    return f"def score(x):\n    return {value}{suffix}\n"


def root(value):
    return Program(id=1, code=code(value), fitness=-value, status="valid", idea="", analysis=None,
                   action="Provided", generation=0)


class Scripted(TokenLLM):
    def draw_sample_with_details(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response if isinstance(response, dict) else {
            "content": response, "finish_reason": "stop", "usage": {"prompt_tokens": 17, "completion_tokens": 9},
            "model": "scripted",
        }


def search(directory, *answers, initial=(), evaluation=None, **options):
    return TraceAADV18(
        evaluation=evaluation or TinyEvaluation(), llm=Scripted(*answers), run_dir=directory,
        initial_programs=initial,
        config=Config(**{"budget": max(1, len(answers)), "roots": 1, "diverge": False, **options}),
    )


def developed(method):
    return [p for p in method.programs.values() if p.action == "Develop"]


def test_full_tree_windows_expand_once_in_the_specified_order(tmp_path):
    method = search(tmp_path, *(code(i) for i in range(1, 15)), initial=[root(0)])
    summary = method.run()
    assert [p.request["decision_path"] for p in developed(method)] == [
        "", "0", "1", "00", "10", "01", "000", "100", "010", "001", "11", "110", "101", "011",
    ]
    assert [p.request["depth_limit"] for p in developed(method)] == [2]*3 + [3]*3 + [4]*8
    assert len({p.request["state_id"] for p in developed(method)}) == 14
    assert len(method.states) == 29 and method.tasks[1].age == 14
    assert summary["best"]["fitness"] == -14 and summary["budget_used"] == summary["model_calls"] == 14
    assert summary["evaluation_calls"] == 14 and summary["selection_evaluations"] == 0
    assert (tmp_path / "best_program.py").read_text() == code(14)
    assert [r["x_label"] for r in committed_rows(tmp_path) if r["kind"] == "candidate"] == ["模型请求"]*14


def test_six_call_example_and_selected_path_do_not_include_rejected_siblings(tmp_path):
    method = search(tmp_path, *(code(v) for v in (90, 91, 65, 92, 98, 93)), initial=[root(60)])
    method.run()
    parents = [method.programs[p.parent_id].fitness for p in developed(method)]
    assert parents == [-60, -90, -60, -91, -65, -90]
    prompts = [call[0] for call in method.llm.calls]
    assert "return 60" in prompts[2] and "return 90" not in prompts[2] and "return 91" not in prompts[2]
    assert "Attempt 1: score -90" in prompts[2]
    assert "return 65" in prompts[4] and "return 90" not in prompts[4] and "return 91" not in prompts[4]
    assert method.tasks[1].best_curve == [60, 90, 91, 91, 92, 98, 98]
    assert all("response_format" not in options for _, options in method.llm.calls)


@pytest.mark.parametrize("child", [code(50), code(60, " + 0")])
def test_worse_and_tied_distinct_children_get_the_third_request(tmp_path, child):
    method = search(tmp_path, child, code(61), code(70), initial=[root(60)])
    method.run()
    first, _, third = developed(method)
    assert third.parent_id == first.id
    assert method.states[method.states[1].preferred].program_id == 1


def test_identical_code_has_one_edge_but_is_evaluated_and_charged(tmp_path):
    method = search(tmp_path, "\r\ndef score(x):\r\n    return 60\r\n", code(60), code(61), initial=[root(60)])
    method.run()
    assert [p.request["decision_path"] for p in developed(method)] == ["", "0", "00"]
    assert method.states[1].alternative is None
    assert method.tasks[1].age == 3 and method.evaluation_calls == 3
    assert method._summary("finished")["v18"]["duplicate_requests"] == 2


@pytest.mark.parametrize("failure", [
    ConnectionError("offline"), "", "def score(:", code('float("nan")'), code('float("inf")'),
    "def wrong(x):\n    return 1\n", "def score(x):\n    raise ValueError('bad')\n",
    {"content": code(90), "finish_reason": "length", "usage": {}},
])
def test_failure_is_one_request_and_retry_keeps_the_valid_parent(tmp_path, failure):
    method = search(tmp_path, failure, code(70), initial=[root(60)])
    result = method.run()
    failed, retry = developed(method)
    assert not failed.valid and retry.valid and retry.parent_id == 1
    assert method.states[1].alternative is None
    assert result["budget_used"] == result["model_calls"] == method.tasks[1].age == 2
    assert method.tasks[1].best_curve == [60, 60, 70]
    assert len(list(rows(tmp_path / "calls.jsonl.gz"))) == 2


def test_timeout_consumes_one_attempt_and_keeps_parent(tmp_path):
    method = search(tmp_path, "def score(x):\n    while True:\n        pass\n", code(70),
                    initial=[root(60)], eval_timeout_seconds=0.15)
    method.run()
    assert developed(method)[0].status == "timeout"
    assert developed(method)[1].parent_id == 1 and method.tasks[1].age == 2


def test_initialization_caps_and_no_valid_root_are_explicit(tmp_path):
    method = search(tmp_path, "", "", "", budget=9, init_attempt_limit=3)
    summary = method.run()
    assert summary["status"] == "no_valid_root" and summary["budget_used"] == 3
    assert len(method.llm.calls) == 3 and not method.tasks
    limited = search(tmp_path / "budget", "", "", budget=2, init_attempt_limit=16)
    assert limited.run()["budget_used"] == 2


def test_initialization_cost_is_not_task_age_and_partial_roots_can_develop(tmp_path):
    method = search(tmp_path, "", code(60), "", code(90), roots=3, init_attempt_limit=3)
    summary = method.run()
    assert summary["init_attempts"] == 3 and method.tasks[1].age == 1
    assert summary["v18"]["setup_requests"] == 3
    assert summary["v18"]["development_requests"] == 1


def test_same_age_uses_historical_snapshots_waits_for_reference_and_recovers(tmp_path):
    method = search(tmp_path, initial=[root(60), root(55)], roots=2, budget=1)
    a,b = method.tasks.values()
    a.age, a.best_curve = 8, [60]*6+[65,70,90]
    b.age, b.best_curve = 5, [55,55,57,58,60,61]
    method.programs[1].fitness = -90
    method.programs[2].fitness = -61
    method._statuses()
    assert b.status == "ACTIVE" and b.comparison["reference_quality"] == 60
    a.age, a.best_curve = 4, [60,70,75,80,90]
    method._statuses()
    assert b.status == "WAIT_REFERENCE" and method._allocate() is a
    a.age, a.best_curve = 5, [60,70,75,80,90,90]
    method._statuses()
    assert b.status == "PAUSED"
    # A new young champion postpones comparison; its age-5 history then revives B.
    c = root(100)
    c.id = 3
    c.request = {"raw_score": -100, "quality": 100}
    method.programs[3] = c
    method._new_task(c)
    method._statuses()
    assert b.status == "WAIT_REFERENCE"
    champion = method.tasks[3]
    champion.age, champion.best_curve = 6, [50]*6+[100]
    method._statuses()
    assert b.status == "ACTIVE" and b.age == 5
    assert method._allocate() is a  # a and b have age 5; earlier task wins


def test_youngest_selection_includes_champion_and_ties_keep_first_best(tmp_path):
    method = search(tmp_path, code(60, " + 0"), code(60, " + 1"),
                    initial=[root(60), root(50)], roots=2)
    method.step()
    assert developed(method)[0].parent_id == 1
    assert method._best().id == 1
    method.step()
    assert developed(method)[1].parent_id == 2
    assert [t.age for t in method.tasks.values()] == [1,1]


def test_resume_preserves_program_context_cursor_and_age_without_regeneration(tmp_path):
    answers = [code(i) for i in range(1, 15)]
    control = search(tmp_path / "control", *answers, initial=[root(0)])
    expected = control.run()
    first = search(tmp_path / "resume", *answers[:5], initial=[root(0)], budget=14)
    for _ in range(5):
        first.step()
    resumed = search(tmp_path / "resume", *answers[5:], budget=14)
    actual = resumed.run()
    assert [c[0] for c in first.llm.calls+resumed.llm.calls] == [c[0] for c in control.llm.calls]
    assert asdict(resumed.tasks[1]) == asdict(control.tasks[1])
    assert actual["best"]["id"] == expected["best"]["id"]
    assert actual["model_calls"] == actual["evaluation_calls"] == 14
    assert not resumed.llm.calls[0][1].get("response_format")
    with pytest.raises(ValueError, match="configuration"):
        search(tmp_path / "resume", budget=15)


def test_saved_response_survives_evaluator_scheduler_failure(tmp_path, monkeypatch):
    first = search(tmp_path, code(70), initial=[root(60)], budget=2)
    monkeypatch.setattr(first.training, "evaluate", lambda *args: (_ for _ in ()).throw(SchedulerError("busy")))
    assert first.run()["status"] == "service_unavailable"
    assert first.budget_used == 1 and first.tasks[1].age == 0
    resumed = search(tmp_path, code(80), budget=2)
    summary = resumed.run()
    assert len(resumed.llm.calls) == 1 and summary["model_calls"] == 2
    assert resumed.tasks[1].age == 2 and summary["best"]["fitness"] == -80


def test_interrupted_provider_request_is_not_silently_reissued(tmp_path):
    first = search(tmp_path, KeyboardInterrupt(), initial=[root(60)], budget=2)
    with pytest.raises(KeyboardInterrupt):
        first.run()
    resumed = search(tmp_path, code(70), budget=2)
    result = resumed.run()
    assert len(resumed.llm.calls) == 1 and result["budget_used"] == result["model_calls"] == 2
    assert "interrupted request" in developed(resumed)[0].failure["error"]
    assert resumed.tasks[1].best_curve == [60,60,70]


def test_divergence_requires_stagnation_and_budget_for_five_development_calls(tmp_path):
    answers = [code(v) for v in (60,60,60,50,51,52,53,54,55,56)]
    method = search(tmp_path, *answers, initial=[root(60)], diverge=True, stagnation_window=3)
    for _ in range(4):
        method.step()
    divergence = list(method.programs.values())[-1]
    assert divergence.action == "Diverge" and divergence.parent_id is None and divergence.reference_id == 1
    assert method.tasks[2].age == 0
    assert method.last_diverge == 4 and method._divergence_due() is False
    method.run()
    assert sum(t.age for t in method.tasks.values()) + 1 == method.budget_used
    short = search(tmp_path / "short", *(code(60) for _ in range(5)), initial=[root(60)],
                   diverge=True, stagnation_window=1)
    short.run()
    assert len(short.tasks) == 1


@pytest.mark.parametrize("same_age,lds", [(False,False),(True,False),(False,True),(True,True)])
def test_ablation_switches_keep_request_and_evaluation_protocol(tmp_path, same_age, lds):
    method = search(tmp_path, *(code(i) for i in range(61,69)), initial=[root(60)], same_age=same_age, lds=lds)
    summary = method.run()
    addresses = [p.request["decision_path"] for p in developed(method)]
    assert any("1" in a for a in addresses) == lds
    assert summary["budget_used"] == summary["evaluation_calls"] == 8


def test_late_first_changes_only_discrepancy_order(tmp_path):
    method = search(tmp_path, *(code(i) for i in range(1,15)), initial=[root(0)], early_first=False)
    method.run()
    assert [p.request["decision_path"] for p in developed(method)][:6] == ["","0","1","00","01","10"]
    assert len({p.request["decision_path"] for p in developed(method)}) == 14


def test_maximization_is_normalized_before_parent_racing_and_final_selection(tmp_path):
    # TinyEvaluation returns minus the program's number, so -40 is the largest raw score.
    method = search(tmp_path, code(60), code(50), code(40), objective="maximize")
    result = method.run()
    assert result["best"]["request"]["raw_score"] == -40
    assert result["best"]["fitness"] == 40 and method.tasks[1].best_curve == [-60,-50,-40]


def test_prompt_contains_interface_not_template_implementation_and_trims_old_feedback(tmp_path):
    evaluation = TinyEvaluation()
    evaluation.template_program = 'def score(x):\n    """Return a number from x."""\n    return "TEMPLATE_SENTINEL"\n'
    method = search(tmp_path, *(code(i) for i in range(1,6)), initial=[root(0)], evaluation=evaluation)
    for _ in range(3):
        method.step()
    prompts = [p[0] for p in method.llm.calls]
    assert all("TEMPLATE_SENTINEL" not in p and "Return a number from x." in p for p in prompts)
    state = method.states[method.states[1].alternative]
    parent = method.programs[state.program_id]
    full = method.prompts.build("Develop",parent,state)
    method.config.context_tokens = full["input_tokens"] + method.config.output_tokens - 1
    trimmed = method.prompts.build("Develop",parent,state)
    assert not trimmed["local_attempt_ids"] and code(0).strip() in trimmed["prompt"]


def test_context_overflow_does_not_spend_a_model_request(tmp_path):
    method = search(tmp_path, code(70), initial=[root(60)], output_tokens=10, context_tokens=11)
    summary = method.run()
    assert summary["status"] == "prompt_too_large" and summary["budget_used"] == 0
    assert not method.llm.calls


def test_time_budget_freezes_the_best_without_another_request(tmp_path):
    method = search(tmp_path, code(70), initial=[root(60)], max_seconds=1)
    method.progress.elapsed=2
    summary = method.run()
    assert summary["status"] == "finished" and summary["budget_used"] == 0
    assert summary["best"]["fitness"] == -60


@pytest.mark.parametrize("options", [
    {"budget":0}, {"roots":True}, {"attempts_shown":-1}, {"checkpoint_multiplier":1},
    {"initial_depth_limit":1}, {"objective":"unknown"}, {"max_seconds":float("nan")}, {"lds":1},
])
def test_invalid_configuration_fails_before_search(options):
    with pytest.raises(ValueError):
        Config(**options)


def test_plain_or_single_fenced_code_only():
    assert parse_code('```python\n'+code(1)+'```', 'stop') == code(1)
    assert normalized_source('\r\n'+code(1).replace('\n','\r\n')) == code(1).strip()


@pytest.mark.parametrize("reply", [code(1), "```python\n"+code(1), "```py\n"+code(1)+"```",
                                   ("```python\n"+code(1)+"```").replace("\n", "\r\n")])
def test_code_envelopes_preserve_the_submitted_source(reply):
    from traceaad.delivery import DeliveryError
    assert parse_code(reply, "stop") == code(1)
    with pytest.raises(DeliveryError):
        parse_code(reply, "length")


def test_fence_tolerance_does_not_accept_explanations_or_hide_invalid_code(tmp_path):
    replies = ["```python\ndef score(:", "```python\n"+code(1)+"```\nExplanation follows."]
    method = search(tmp_path, *replies, code(2))
    method.run()
    assert [p.status for p in method.programs.values()] == ["invalid_source", "invalid_source", "valid"]
    assert method.budget_used == 3


def test_retry_labels_copies_and_scopes_failed_child_feedback(tmp_path):
    method = search(tmp_path, code(60), "def score(x):\n    raise ValueError('child-only-error')\n", code(70),
                    initial=[root(60)])
    method.run()
    prompt = method.llm.calls[2][0]
    assert "unchanged source; exact copy" in prompt and "child-only-error" in prompt
    assert "current program remains valid" in prompt
    assert "raise ValueError" not in prompt
    assert prompt.index("[Previous Attempts") < prompt.index("[Current Program]") < prompt.index("[Request]")


def test_history_normalization_removes_comment_noise_but_preserves_current_source(tmp_path):
    method = search(tmp_path, code(60)+"# historical-comment-noise\n", initial=[root(60)])
    method.step()
    state = method.states[method.states[1].alternative]
    parent = method.programs[state.program_id]
    request = method.prompts.build("Develop", parent, state)
    history, current = request["prompt"].split("[Current Program]")
    assert "historical-comment-noise" not in history and "historical-comment-noise" in current
    assert "No change after normalizing" in history
    assert parent.code.rstrip() in current and request["history_edge_ids"] == [parent.id]


def test_divergence_reference_is_not_labelled_as_the_development_parent(tmp_path):
    method = search(tmp_path, code(60))
    method.step()
    prompt = method.prompts.build("Diverge", method._best())["prompt"]
    assert "[Reference Program]" in prompt and "[Current Program]" not in prompt
    assert "parameter adjustment" in prompt and "[Output]" in prompt


@pytest.mark.parametrize("revision", [None, "v18-context-1"])
def test_resume_rejects_a_different_prompt_revision(tmp_path, revision):
    from traceaad.storage import write_json
    method = search(tmp_path, code(60), code(70))
    method.step()
    checkpoint = read_json(tmp_path / "resume.json")
    checkpoint["state"]["v18"]["prompt_revision"] = revision
    write_json(tmp_path / "resume.json", checkpoint)
    with pytest.raises(ValueError, match="prompt revision differs"):
        search(tmp_path, code(60), code(70))
