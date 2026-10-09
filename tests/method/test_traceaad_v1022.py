"""V10.22: V10.21's search; Explore reasons from the exact output of the function."""

import pytest

from benchmarks.tasks import CO_TASKS, training_task
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.common.prompts import REFINE, CROSSOVER, INITIAL
from traceaad.v10_21 import Config as V1021Config
from traceaad.v10_21.prompts import PromptBuilder as V1021Prompts
from traceaad.v10_22 import Config, TraceAADV1022
from traceaad.v10_22.config import EXPERIMENT
from traceaad.v10_22.prompts import EXPLORE_ANALYSIS, REFINE_ANALYSIS, PromptBuilder


def builder(task, programs=None, attempts=None):
    evaluation = training_task(task, condition="traceaad")[0]
    return PromptBuilder(TokenLLM(), task, evaluation, programs or {}, attempts or {}, Config())


def test_search_and_execution_are_v1021s():
    assert EXPERIMENT == "traceaad_v10_22" and TraceAADV1022.METHOD == "v1022"
    assert Config().operators == V1021Config().operators == {"Refine": 0.45, "Explore": 0.30, "Crossover": 0.25}
    assert Config().eval_timeout_seconds == V1021Config().eval_timeout_seconds
    assert PromptBuilder.ACTIONS == ("Refine", "Explore", "Crossover")


def test_only_the_analyses_explore_and_the_further_initial_design_change():
    assert PromptBuilder.REFINE == V1021Prompts.REFINE == REFINE
    assert PromptBuilder.CROSSOVER == V1021Prompts.CROSSOVER == CROSSOVER
    assert PromptBuilder.REPAIR == V1021Prompts.REPAIR and PromptBuilder.INITIAL == INITIAL
    changed = {"Explore", "Refine"}
    assert {k: v for k, v in PromptBuilder.ANALYSIS.items() if k not in changed} == \
        {k: v for k, v in V1021Prompts.ANALYSIS.items() if k not in changed}
    assert PromptBuilder.ANALYSIS["Explore"] == EXPLORE_ANALYSIS
    assert PromptBuilder.ANALYSIS["Refine"] == REFINE_ANALYSIS and "score" in REFINE_ANALYSIS.split(",")[0]
    assert "better than all of them" not in PromptBuilder.ANOTHER_INITIAL
    assert "best found so far" not in PromptBuilder.EXPLORE and "current algorithm" in PromptBuilder.EXPLORE


@pytest.mark.parametrize("task", CO_TASKS)
def test_prompts_still_say_nothing_about_time_or_direction(task):
    prompts = builder(task)
    text = prompts.EXPLORE + prompts.ANOTHER_INITIAL + EXPLORE_ANALYSIS + REFINE_ANALYSIS
    for word in ("second", "time limit", "budget", "efficient", "search", "simulat", "tour", "route"):
        assert word not in text.lower()


def explored_state():
    programs = {
        1: {"id": 1, "key": "a", "parent_id": None, "valid": True, "fitness": 2.0, "score": 2.0,
            "idea": "start", "code": "def score(x):\n    return 2\n", "action": "Init"},
        2: {"id": 2, "key": "b", "parent_id": 1, "valid": True, "fitness": 1.0, "score": 1.0,
            "idea": "better", "code": "def score(x):\n    return 1\n", "action": "Refine"},
    }
    attempts = {
        1: {"id": 1, "action": "Init", "parent_id": None, "status": "valid", "program_id": 1, "idea": "start"},
        2: {"id": 2, "action": "Refine", "parent_id": 1, "status": "valid", "program_id": 2, "idea": "better"},
    }
    return programs, attempts


def test_explore_decides_from_the_current_algorithm_and_its_attempts():
    programs, attempts = explored_state()
    evaluation = TinyEvaluation()
    prompts = PromptBuilder(TokenLLM(), "tiny", evaluation, programs, attempts, Config())
    text = prompts.build("Explore", programs[1])["prompt"]
    assert "[Current Algorithm]" in text and "[Attempts From the Current Algorithm]" in text
    assert "How the Best Score Improved" not in text and "Best score found so far" not in text
    assert "[How the Current Algorithm Was Formed]" not in text
    assert text.index("[Attempts From the Current Algorithm]") < text.index("[Your Task: Explore]")
    assert f"Analysis: <a few sentences: {EXPLORE_ANALYSIS}. It will not be shown again>" in text
    old = V1021Prompts(TokenLLM(), "tiny", evaluation, programs, attempts, V1021Config())
    assert "How the Best Score Improved" in old.build("Explore", programs[1])["prompt"]


def test_further_initial_designs_ask_for_a_different_core_idea():
    programs = {i: {"id": i, "key": str(i), "parent_id": None, "valid": True, "fitness": float(i), "score": float(i),
                    "idea": f"idea {i}", "code": f"def score(x):\n    return {i}\n", "action": "Init"}
                for i in range(1, 5)}
    prompts = PromptBuilder(TokenLLM(), "tiny", TinyEvaluation(), programs, {}, Config())
    assert prompts.initial([])["prompt"].count(INITIAL) == 1
    text = prompts.initial(list(programs.values()))["prompt"]
    assert PromptBuilder.ANOTHER_INITIAL in text and "[Algorithms Designed So Far]" in text


def test_a_tiny_run_completes(tmp_path):
    answers = [response(i) for i in range(1, 9)] + [response(20 + i) for i in range(12)]
    method = TraceAADV1022(evaluation=TinyEvaluation(), llm=TokenLLM(*answers), run_dir=tmp_path,
                           config=Config(budget=14))
    summary = method.run()
    assert summary["status"] == "finished" and summary["method"] == "v1022"
    prompts = [prompt for prompt, _ in method.llm.calls]
    explores = [p for p in prompts if "[Your Task: Explore]" in p]
    assert all("How the Best Score Improved" not in p for p in explores)
