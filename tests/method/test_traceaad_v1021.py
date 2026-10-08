"""V10.21: V10.20 with time stated per instance and per call, and no ban on code comments."""

import pytest

from benchmarks.tasks import CO_TASKS, INSTANCE_SECONDS, training_task
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.common.canonical import canonical
from traceaad.v10_20 import Config as V1020Config
from traceaad.v10_21 import Config, TraceAADV1021
from traceaad.v10_21.prompts import DEEPEN, PromptBuilder


def builder(task):
    evaluation = training_task(task, condition="traceaad")[0]
    return PromptBuilder(TokenLLM(), task, evaluation, {}, {}, Config())


def test_the_search_is_v1020s():
    assert Config().operators == V1020Config().operators
    assert TraceAADV1021.EXPLORING == ("Explore", "Deepen")


@pytest.mark.parametrize("task", CO_TASKS)
def test_evaluation_states_the_per_instance_budget(task):
    prompts = builder(task)
    evaluation = prompts.common[1]
    assert f"Each instance has a time budget of {INSTANCE_SECONDS} seconds, which includes the fixed solver." in evaluation
    assert "the whole evaluation must finish within 160 seconds" in evaluation
    assert "under the same per-instance budget (500 seconds in total)" in evaluation
    assert evaluation.lower().count("lower is better") == 1


def test_measurements_are_per_instance_and_per_call():
    prompts = builder("graph_colouring")
    program = {"score": -11.75, "eval_seconds": 12.3, "calls": 7113, "function_seconds": 7.6}
    assert prompts.measured(program) == ("Score: -11.75 · Time per instance: about 0.77 s of the 10 s budget · "
                                         "445 calls to the function per instance, about 1 ms each")
    once = {"score": 8.0, "eval_seconds": 32.0, "calls": 16, "function_seconds": 16.0}
    assert builder("cvrp_aco").measured(once).endswith("1 call to the function per instance, about 1.0 s each")
    assert prompts.elapsed(12.3) == "time per instance about 0.77 s"
    assert prompts.limit_text() == "the 160 s limit for 16 instances (10 s per instance)"


def test_deepen_and_output_format():
    prompts = builder("jssp_construct")
    assert "per-instance time budget" in DEEPEN and prompts.DEEPEN == DEEPEN
    assert "computation per instance" in prompts.ANALYSIS["Deepen"]
    text = prompts.output_format("Deepen")
    assert text.endswith("Write nothing after the code block.") and "comments" not in text


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
    assert len(prompts) == 14 and all("Write no comments" not in prompt for prompt in prompts)
