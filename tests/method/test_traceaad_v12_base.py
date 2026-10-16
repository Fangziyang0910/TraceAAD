"""V12.0 base arm: search, prompts, final selection and error handling."""

import pytest
from traceaad.v12_1 import Program


def individual(**fields):
    return Program(
        **{
            "id": 1,
            "generation": 1,
            "code": "",
            "idea": "",
            "analysis": "",
            "action": "Init",
            **fields,
        }
    )


from benchmarks.tasks import (
    CO_TASKS,
    FIXED_TASKS,
    FUNCTION_SECONDS,
    INSTANCE_SECONDS,
    training_task,
)
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v12_1.canonical import canonical
from traceaad.v12_1 import Config, TraceAADV121
from traceaad.v12_1.prompts import PromptBuilder
from traceaad.v12_1 import Config as Config121, TraceAADV121


def builder(task):
    evaluation = training_task(task, condition="traceaad")[0]
    return PromptBuilder(TokenLLM(), task, evaluation, {}, Config(linking=False))


def test_steps_are_refine_explore_and_crossover():
    assert Config(linking=False).operators == {
        "Refine": 0.45,
        "Explore": 0.3,
        "Crossover": 0.25,
    }
    assert not hasattr(TraceAADV121, "EXPLORING")
    assert PromptBuilder.ACTIONS == ("Refine", "Explore", "Crossover")
    assert (
        Config(linking=False).eval_timeout_seconds == INSTANCE_SECONDS == 120
        and FUNCTION_SECONDS == 2
    )


def test_deepen_is_rejected_before_search():
    with pytest.raises(ValueError, match="unknown operators.*Deepen"):
        Config(operators={"Deepen": 1}, linking=False)


@pytest.mark.parametrize("task", CO_TASKS)
def test_prompts_say_nothing_about_time(task):
    prompts = builder(task)
    text = (
        "\n".join(prompts.common)
        + prompts.REPAIR
        + prompts.EXPLORE
        + prompts.REFINE
        + prompts.CROSSOVER
    )
    assert prompts.common[1].startswith("Score: ") and prompts.common[1].endswith(
        ". Lower is better."
    )
    assert "[Evaluation]" not in prompts._render(prompts.common)
    assert prompts.common[2].endswith(
        "include every import, constant and helper it uses."
    )
    assert "computation" not in prompts.common[2]
    for word in ("second", "time limit", "time budget", "efficient"):
        assert word not in text.lower()
    assert text.lower().count("lower is better") == 1
    if task in FIXED_TASKS:
        evaluation = training_task(task, condition="traceaad")[0]
        assert prompts.common[0].startswith("[Task]\n" + evaluation.DESCRIPTION)
        for bookkeeping in (
            "split-specific seeds",
            "copies",
            "clip",
            "hidden from the candidate",
        ):
            assert bookkeeping not in prompts.common[0]
        for name, value in evaluation.outer_settings.items():
            assert f"{name}={value}" in prompts.common[0]
    for detail in ("WHOLE dataset", "tested on", "evaluated on", "training instances"):
        assert detail not in text.lower() if detail.islower() else detail not in text


def test_measurements_show_the_score_only():
    prompts = builder("graph_colouring")
    program = individual(
        fitness=-11.75, eval_seconds=12.3, calls=7113, function_seconds=7.6
    )
    assert prompts.measured(program) == "Score: -11.75"
    rule = "The limit is 10 s of CPU time inside the function per instance, on average over the instances."
    for failure, text in [
        (
            {
                "error_type": "FunctionBudgetExceeded",
                "mean_function_cpu_seconds": 12.345,
            },
            f"Timed out: the function used 12.35 s of CPU time per instance on average. {rule}",
        ),
        (
            {
                "error_type": "TimeoutError",
                "mean_function_cpu_seconds": None,
                "timeout_seconds": 120,
                "function_cpu_seconds": 118.5,
            },
            f"Timed out: one instance did not finish within 120 s; the function had used 118.50 s of CPU time on it. {rule}",
        ),
    ]:
        timeout = individual(
            failure={"kind": "timeout", "function_seconds_limit": 10, **failure},
            idea="x",
            code="def f():\n    pass",
        )
        assert (
            prompts.failure(timeout) == "timed out"
            and prompts.error_text(timeout) == text
        )
        assert "[Your Task: Repair a Timeout]" in prompts.repair(timeout)["prompt"]
    assert prompts.error_text(individual(failure={"kind": "timeout"})) == "Timed out."


def test_legacy_canonical_helper_removes_comments():
    with_notes = (
        'def f(x):\n    # scale the input\n    """doc"""\n    return 2 * x  # double\n'
    )
    assert canonical(with_notes) == canonical("def f(x):\n    return 2 * x\n")


def test_a_tiny_run_completes(tmp_path):
    answers = [response(i) for i in range(1, 9)] + [response(20 + i) for i in range(12)]
    method = TraceAADV121(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*answers),
        run_dir=tmp_path,
        config=Config(budget=14, linking=False, operators={"Refine": 1}),
    )
    summary = method.run()
    assert summary["status"] == "finished" and summary["method"] == "v12_1"
    prompts = [prompt for prompt, _ in method.llm.calls]
    assert len(prompts) == 14 and all(
        ("Write no comments or docstrings" in prompt for prompt in prompts)
    )
    assert not any(
        ("Deepen" in prompt or "time per instance" in prompt for prompt in prompts)
    )


def test_freeze_failure_does_not_export_a_program(tmp_path, monkeypatch):
    method = TraceAADV121(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(response(1), response(2)),
        run_dir=tmp_path,
        config=Config(budget=2, roots=2, linking=False),
    )
    method._roots()
    method._roots()
    monkeypatch.setattr(
        method.training,
        "evaluate",
        lambda *args, **kwargs: {
            "fitness": None,
            "failure": {"kind": "timeout"},
            "evaluations": [],
        },
    )
    method._freeze()
    assert method.phase == "selection_failed" and method.progress.selected_id is None
    assert len(method.progress.selection_results) == 2
    assert not (tmp_path / "best_program.py").exists()
    assert not (tmp_path / "selection.json").exists()


def test_final_training_score_uses_recheck_without_changing_search_history(
    tmp_path, monkeypatch
):
    from experiments.infra.artifacts import pick_best_sample

    method = TraceAADV121(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(response(2)),
        run_dir=tmp_path,
        config=Config(budget=1, roots=1, linking=False),
    )
    method._roots()
    source_before = (tmp_path / "programs.jsonl").read_bytes()
    monkeypatch.setattr(
        method.training,
        "evaluate",
        lambda *args, **kwargs: {"fitness": 123.0, "failure": None, "evaluations": []},
    )
    method._freeze()
    summary = method._summary(method.phase)
    assert summary["best"]["fitness"] == summary["best"]["score"] == 123
    chosen, history = pick_best_sample(tmp_path)
    assert chosen["score"] == 123 and history[0]["score"] == -2
    assert method.archive[1].fitness == -2
    assert (tmp_path / "programs.jsonl").read_bytes() == source_before


@pytest.mark.parametrize("action", ["Refine", "Explore", "Crossover"])
def test_long_prompts_keep_the_selected_action_and_material(
    tmp_path, monkeypatch, action
):
    method = TraceAADV121(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(response(20), response(2), response(3)),
        run_dir=tmp_path,
        config=Config(budget=3, roots=1, linking=False),
    )
    method._roots()
    parent = method.archive[1]
    method._create_program(method.prompts.build("Refine", parent), parent=parent)
    parent = method.archive[2]
    method.prompts.common[0] += "\n" + "context " * 66000 + "end of task"
    monkeypatch.setattr(method, "_choose_parent", lambda eligible: (parent, {}))
    method._ordinary_search(action)
    attempt = method.programs[3]
    prompt = method.llm.calls[-1][0]
    assert attempt.action == attempt.request["sampled_action"] == action
    assert (
        attempt.request["input_tokens"] > 65536 and attempt.request["fallbacks"] == []
    )
    assert method.prompts.common[0] in prompt and parent.code.rstrip() in prompt
    if action == "Crossover":
        assert "[Reference Algorithm]" in prompt and "return 20" in prompt
    if action != "Explore":
        assert attempt.request["history_edge_ids"]


@pytest.mark.parametrize("method_class,config_class", [(TraceAADV121, Config121)])
def test_initialization_keeps_all_roots_without_total_budget_cutoff(
    tmp_path, method_class, config_class
):
    method = method_class(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*(response(i) for i in range(1, 9))),
        run_dir=tmp_path,
        config=config_class(linking=False, budget=4),
    )
    summary = method.run()
    assert summary["status"] == "finished" and summary["budget_used"] == 5
    assert summary["num_roots"] == summary["init_attempts"] == 5
    assert all((attempt.analysis == "" for attempt in method.programs.values()))
    prompts = [prompt for prompt, _ in method.llm.calls]
    assert "[Algorithms Designed So Far]" not in prompts[0]
    for index, prompt in enumerate(prompts[1:], 1):
        assert "[Your Task: Design a Better Algorithm]" in prompt
        assert "listed best first" in prompt and "lower is better" in prompt
        assert all(
            (
                f"Algorithm {i} ·" in prompt and f"Design: return {i}" in prompt
                for i in range(1, index + 1)
            )
        )
        assert prompt.count("```python") == index + 1
        positions = [prompt.index(f"Algorithm {i} ·") for i in range(index, 0, -1)]
        assert positions == sorted(positions)


@pytest.mark.parametrize("method_class,config_class", [(TraceAADV121, Config121)])
def test_initialization_repairs_once_then_generates_a_new_algorithm(
    tmp_path, method_class, config_class
):
    method = method_class(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(response("1 / 0"), response("2 / 0"), response(7)),
        run_dir=tmp_path,
        config=config_class(linking=False, budget=1, roots=1),
    )
    summary = method.run()
    assert summary["status"] == "finished" and summary["num_roots"] == 1
    assert summary["init_attempts"] == summary["budget_used"] == 3
    assert [attempt.action for attempt in method.programs.values()] == [
        "Init",
        "Repair",
        "Init",
    ]
    assert method.archive[3].parent_id is None
    assert method.programs[2].idea == method.programs[1].idea == "return 1 / 0"
    prompt, kwargs = method.llm.calls[1]
    assert (
        kwargs["response_format"] == {"type": "json_object"}
        and "two string fields" in prompt
    )
    assert '"design"' not in prompt and "Raised at: `return 1 / 0`" in prompt


def test_unusable_source_keeps_its_design_through_repair(tmp_path):
    method = TraceAADV121(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(response("("), response("(("), response(7)),
        run_dir=tmp_path,
        config=Config(linking=False, budget=1, roots=1),
    )
    method.run()
    assert method.programs[1].status == "invalid_source"
    assert method.programs[1].idea == method.programs[2].idea == "return ("


@pytest.mark.parametrize("method_class,config_class", [(TraceAADV121, Config121)])
@pytest.mark.parametrize(
    "value,roots,status", [(1, 5, "finished"), ("1 / 0", 0, "no_valid_root")]
)
def test_initialization_stops_at_its_attempt_limit(
    tmp_path, method_class, config_class, value, roots, status
):
    method = method_class(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*[response(value)] * 16),
        run_dir=tmp_path,
        config=config_class(linking=False, budget=1),
    )
    summary = method.run()
    assert summary["status"] == status and summary["num_roots"] == roots
    assert summary["init_attempts"] == summary["budget_used"] == (5 if roots else 16)


def test_scheduler_failure_preserves_the_generated_reply(tmp_path):
    method = TraceAADV121(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(response(1)),
        run_dir=tmp_path,
        config=Config(
            budget=1,
            roots=1,
            scheduler_socket=str(tmp_path / "missing.socket"),
            linking=False,
        ),
    )
    summary = method.run()
    assert summary["status"] == "service_unavailable" and summary["model_calls"] == 1
    assert summary["budget_used"] == 1 and method.phase == "roots"
    assert method.programs[1].status == "pending" and method.progress.pending_id == 1
    import json

    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert (
        len(calls) == 1
        and json.loads(calls[0]["response"])["code"] == "def score(x):\n    return 1"
    )


def test_repair_errors_show_the_task_and_program_frames():
    from traceaad.v12_1.evaluation import HARNESS, clean_traceback, failing_line

    code = "import numpy as np\ndef f(x):\n    y = x + 1\n    return np.ones(2) @ np.ones(3)\n"
    text = f'Traceback (most recent call last):\n  File "{HARNESS[0]}evaluate.py", line 350, in _evaluate_with_details\n    return self._outcome(task.evaluate_program(code, function))\n                         ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^\n  File "/repo/benchmarks/task/evaluation.py", line 52, in solve\n    value = heuristic(data.copy())\n  File "<string>", line 9, in f\n  File "<string>", line 4, in f\nValueError: matmul: Input operand 1 has a mismatch'
    cleaned = clean_traceback(text, code)
    assert (
        cleaned
        == 'Traceback (most recent call last):\n  File "/repo/benchmarks/task/evaluation.py", line 52, in solve\n    value = heuristic(data.copy())\n  File "<string>", line 4, in f\nValueError: matmul: Input operand 1 has a mismatch'
    )
    failed = individual(
        failure={
            "kind": "runtime_error",
            "error": cleaned,
            "line": failing_line(cleaned, code),
        }
    )
    assert (
        builder("tsp_construct").error_text(failed)
        == cleaned + "\nRaised at: `return np.ones(2) @ np.ones(3)`"
    )


def test_initial_reply_is_parsed_before_source_validation():
    import json
    from traceaad.v12_1.delivery import (
        DeliveryError,
        parse_json_response,
        validate_source,
    )

    template = "def score(x):\n    pass"
    reply = json.dumps(
        {
            "analysis": "a",
            "design": " return one ",
            "code": "def score(x):\n    return 1\n",
        }
    )
    assert parse_json_response(reply, "stop") == (
        "def score(x):\n    return 1\n",
        "return one",
        {"strategy": "json", "analysis": "a"},
    )
    with pytest.raises(DeliveryError):
        parse_json_response("invalid JSON", "stop")
    source, _, _ = parse_json_response(
        json.dumps({"design": "x", "code": "def wrong(x): pass"}), "stop"
    )
    with pytest.raises(ValueError, match="one function"):
        validate_source(source, template)


@pytest.mark.parametrize("method_class,config_class", [(TraceAADV121, Config121)])
@pytest.mark.parametrize("repair", [False, True])
def test_interrupted_initial_evaluation_keeps_its_attempt_slot(
    tmp_path, monkeypatch, method_class, config_class, repair
):
    from core.scheduling import SchedulerError

    config = config_class(
        linking=False, budget=1, roots=1, init_attempt_limit=2 if repair else 1
    )
    answers = ([response("(")] if repair else []) + [response(1)]
    method = method_class(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*answers),
        run_dir=tmp_path,
        config=config,
    )

    def unavailable(*args, **kwargs):
        raise SchedulerError("offline")

    monkeypatch.setattr(method.training, "evaluate", unavailable)
    summary = method.run()
    assert summary["status"] == "service_unavailable"
    assert summary["init_attempts"] == summary["budget_used"] == int(repair) + 1
    assert method.progress.pending_id == int(repair) + 1
    resumed = method_class(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(response(2)),
        run_dir=tmp_path,
        config=config,
    )
    summary = resumed.run()
    assert summary["status"] == "finished"
    assert (
        summary["init_attempts"] == summary["budget_used"] == config.init_attempt_limit
    )
    assert summary["model_calls"] == config.init_attempt_limit
    assert not resumed.llm.calls
    assert resumed.programs[config.init_attempt_limit].action == (
        "Repair" if repair else "Init"
    )


@pytest.mark.parametrize(
    "bad_code", ["def score(x):\n    return (", "def wrong(x):\n    return 1"]
)
def test_invalid_source_and_repair_keep_analysis(tmp_path, bad_code):
    import json

    answers = [
        json.dumps(
            {
                "analysis": "initial analysis",
                "design": "original design",
                "code": bad_code,
            }
        ),
        json.dumps(
            {
                "analysis": "repair analysis",
                "design": "unrequested design",
                "code": bad_code + "\n# repair",
            }
        ),
    ]
    method = TraceAADV121(
        evaluation=TinyEvaluation(),
        llm=TokenLLM(*answers),
        run_dir=tmp_path,
        config=Config121(linking=False, budget=1, roots=1, init_attempt_limit=2),
    )
    method.run()
    assert [a.analysis for a in method.programs.values()] == [
        "initial analysis",
        "repair analysis",
    ]
    assert all((a.idea == "original design" for a in method.programs.values()))
    assert method.programs[1].status == "invalid_source"
