"""V10.21: V10.17's three steps, concise prompts and one efficiency rule."""

import pytest

from benchmarks.tasks import CO_TASKS, FIXED_TASKS, FUNCTION_SECONDS, INSTANCE_SECONDS, training_task
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.common.canonical import canonical
from traceaad.common.prompts import ContextTooLong
from traceaad.v10_21 import Config, TraceAADV1021
from traceaad.v10_21.prompts import PromptBuilder


def builder(task):
    evaluation = training_task(task, condition="traceaad")[0]
    return PromptBuilder(TokenLLM(), task, evaluation, {}, {}, Config())


def test_steps_are_refine_explore_and_crossover():
    assert Config().operators == {"Refine": 0.45, "Explore": 0.30, "Crossover": 0.25}
    assert TraceAADV1021.EXPLORING == ("Explore",)
    assert PromptBuilder.ACTIONS == ("Refine", "Explore", "Crossover")
    assert Config().eval_timeout_seconds == INSTANCE_SECONDS == 20 and FUNCTION_SECONDS == 2


def test_deepen_is_rejected_before_search():
    with pytest.raises(ValueError, match="unknown operators.*Deepen"):
        Config(operators={"Deepen": 1})


@pytest.mark.parametrize("task", CO_TASKS)
def test_prompts_say_nothing_about_time(task):
    prompts = builder(task)
    text = "\n".join(prompts.common) + prompts.REPAIR + prompts.EXPLORE + prompts.REFINE + prompts.CROSSOVER
    assert prompts.common[1].startswith("Score: ") and prompts.common[1].endswith(". Lower is better.")
    assert "[Evaluation]" not in prompts._render(prompts.common)
    assert prompts.common[2].endswith("include every import, constant and helper it uses.")
    assert "computation" not in prompts.common[2]
    for word in ("second", "time limit", "time budget", "efficient"):
        assert word not in text.lower()
    assert text.lower().count("lower is better") == 1
    for detail in ("WHOLE dataset", "tested on", "evaluated on", "training instances"):
        assert detail not in text.lower() if detail.islower() else detail not in text


def test_the_score_line_says_what_a_negative_score_means():
    assert "negative score beats the reference" in builder("jssp_construct").common[1]
    assert builder("op_aco").common[1] == ("Score: the negative average total prize of the best tours "
                                           "found by the ant colony. Lower is better.")


@pytest.mark.parametrize("task", FIXED_TASKS)
def test_fixed_tasks_keep_the_problem_and_the_solver_settings(task):
    prompts = builder(task)
    evaluation = training_task(task, condition="traceaad")[0]
    assert prompts.common[0].startswith("[Task]\n" + evaluation.DESCRIPTION)
    for bookkeeping in ("split-specific seeds", "copies", "clip", "hidden from the candidate"):
        assert bookkeeping not in prompts.common[0]
    for name, value in evaluation.outer_settings.items():
        assert f"{name}={value}" in prompts.common[0]


def test_measurements_show_the_score_only():
    prompts = builder("graph_colouring")
    program = {"score": -11.75, "eval_seconds": 12.3, "calls": 7113, "function_seconds": 7.6}
    assert prompts.measured(program) == "Score: -11.75" and prompts.elapsed(12.3) == ""
    for kind in ("FunctionBudgetExceeded", "TimeoutError"):
        timeout = {"failure": {"kind": "timeout", "error_type": kind}, "parent_id": None}
        assert prompts.failure(timeout) == "timed out" and prompts.error_text(timeout) == "Timed out."


def test_comments_never_reach_the_stored_program():
    with_notes = "def f(x):\n    # scale the input\n    \"\"\"doc\"\"\"\n    return 2 * x  # double\n"
    assert canonical(with_notes) == canonical("def f(x):\n    return 2 * x\n")


def test_a_tiny_run_completes(tmp_path):
    answers = [response(i) for i in range(1, 9)] + [response(20 + i) for i in range(12)]
    method = TraceAADV1021(evaluation=TinyEvaluation(), llm=TokenLLM(*answers), run_dir=tmp_path,
                           config=Config(budget=14))
    summary = method.run()
    assert summary["status"] == "finished" and summary["method"] == "v1021"
    prompts = [prompt for prompt, _ in method.llm.calls]
    assert len(prompts) == 14 and all("Write no comments or docstrings" in prompt for prompt in prompts)
    assert not any("Deepen" in prompt or "time per instance" in prompt for prompt in prompts)


def test_freeze_failure_does_not_export_a_program(tmp_path, monkeypatch):
    method = TraceAADV1021(evaluation=TinyEvaluation(), llm=TokenLLM(response(1), response(2)),
                           run_dir=tmp_path, config=Config(budget=2, roots=2))
    method._roots()
    method._roots()
    monkeypatch.setattr(method.training, "evaluate", lambda *args, **kwargs: {
        "fitness": None, "failure": {"kind": "timeout"}, "evaluations": []})
    method._freeze()
    assert method.phase == "selection_failed" and method.progress.selected_id is None
    assert len(method.progress.selection_results) == 2
    assert not (tmp_path / "best_program.py").exists()
    assert not (tmp_path / "selection.json").exists()


def test_final_training_score_uses_recheck_without_changing_search_history(tmp_path, monkeypatch):
    from experiments.infra.artifacts import pick_best_sample
    method = TraceAADV1021(evaluation=TinyEvaluation(), llm=TokenLLM(response(2)),
                           run_dir=tmp_path, config=Config(budget=1, roots=1))
    method._roots()
    source_before = (tmp_path / "programs.jsonl").read_bytes()
    monkeypatch.setattr(method.training, "evaluate", lambda *args, **kwargs: {
        "fitness": 123.0, "failure": None, "evaluations": []})
    method._freeze()
    summary = method._summary(method.phase)
    assert summary["best"]["fitness"] == summary["best"]["score"] == 123
    chosen, history = pick_best_sample(tmp_path)
    assert chosen["score"] == 123 and history[0]["score"] == -2
    assert method.archive[1]["fitness"] == -2
    assert (tmp_path / "programs.jsonl").read_bytes() == source_before


def test_long_refine_falls_back_to_an_available_explore(tmp_path, monkeypatch):
    def large(prefix, value):
        code = "def score(x):\n" + "".join(f"    {prefix}_{i} = {i}\n" for i in range(2250))
        return "Design: test\nCode:\n```python\n" + code + f"    return {value}\n```"

    method = TraceAADV1021(evaluation=TinyEvaluation(),
                           llm=TokenLLM(large("a", 1), large("b", 2), response(3)),
                           run_dir=tmp_path, config=Config(budget=3, roots=1))
    method._roots()
    parent = method.archive[1]
    method._attempt(method.prompts.build("Refine", parent), parent=parent)
    parent = method.archive[2]
    with pytest.raises(ContextTooLong):
        method.prompts.build("Refine", parent)
    assert method.prompts.build("Explore", parent)["input_tokens"] < method.config.max_input_tokens
    monkeypatch.setattr(method, "_choose_parent", lambda eligible: (parent, {}))
    method._ordinary_search("Refine")
    attempt = method.attempts_table[3]
    assert attempt["action"] == "Explore" and attempt["parent_id"] == 2
    assert attempt["sampled_action"] == "Refine" and attempt["exploration"]["step"] == 0
    assert attempt["fallbacks"] == ["explore_context_fallback"] and not method.progress.too_long


def test_development_context_failure_does_not_blacklist_the_program(tmp_path, monkeypatch):
    method = TraceAADV1021(evaluation=TinyEvaluation(), llm=TokenLLM(response(1), response(2)),
                           run_dir=tmp_path, config=Config(budget=4, roots=1, development_probability=1))
    method._roots()
    method._ordinary_search("Explore")
    opened = method._open_exploration()

    def too_long(*args, **kwargs):
        raise ContextTooLong("Refine history does not fit")

    monkeypatch.setattr(method.prompts, "build", too_long)
    method._develop(opened)
    assert method._open_exploration() is None and not method.progress.too_long


def test_initial_context_failure_continues_from_existing_roots(tmp_path):
    method = TraceAADV1021(evaluation=TinyEvaluation(), llm=TokenLLM(*(response(i) for i in range(1, 7))),
                           run_dir=tmp_path, config=Config(budget=6, roots=8, root_tokens=1))
    summary = method.run()
    assert summary['status'] == 'finished' and summary['budget_used'] == 6
    assert summary['num_roots'] == summary['init_attempts'] == 4


def test_initial_context_failure_without_roots_finishes_cleanly(tmp_path):
    method = TraceAADV1021(evaluation=TinyEvaluation(), llm=TokenLLM(), run_dir=tmp_path,
                           config=Config(max_input_tokens=1))
    summary = method.run()
    assert summary['status'] == 'no_valid_root' and summary['model_calls'] == 0


def test_scheduler_failure_preserves_the_generated_reply(tmp_path):
    method = TraceAADV1021(evaluation=TinyEvaluation(), llm=TokenLLM(response(1)), run_dir=tmp_path,
                           config=Config(budget=1, roots=1, scheduler_socket=str(tmp_path / 'missing.socket')))
    summary = method.run()
    assert summary['status'] == 'service_unavailable' and summary['model_calls'] == 1
    assert summary['budget_used'] == 0 and method.phase == 'roots'
    import json
    calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert len(calls) == 1 and calls[0]['response'] == response(1)


def test_repair_errors_show_the_task_and_program_frames():
    from traceaad.common.evaluation import HARNESS, clean_traceback, failing_line
    code = "import numpy as np\ndef f(x):\n    y = x + 1\n    return np.ones(2) @ np.ones(3)\n"
    text = ('Traceback (most recent call last):\n'
            f'  File "{HARNESS[0]}evaluate.py", line 350, in _evaluate_with_details\n'
            '    return self._outcome(task.evaluate_program(code, function))\n'
            '                         ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^\n'
            '  File "/repo/benchmarks/task/evaluation.py", line 52, in solve\n'
            '    value = heuristic(data.copy())\n'
            '  File "<string>", line 9, in f\n'
            '  File "<string>", line 4, in f\n'
            'ValueError: matmul: Input operand 1 has a mismatch')
    cleaned = clean_traceback(text, code)
    assert cleaned == ('Traceback (most recent call last):\n'
                       '  File "/repo/benchmarks/task/evaluation.py", line 52, in solve\n'
                       '    value = heuristic(data.copy())\n'
                       '  File "<string>", line 4, in f\n'
                       'ValueError: matmul: Input operand 1 has a mismatch')
    failed = {"failure": {"kind": "runtime_error", "error": cleaned, "line": failing_line(cleaned, code)}}
    assert builder("tsp_construct").error_text(failed) == cleaned
