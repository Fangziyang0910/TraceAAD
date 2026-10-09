"""V10.23: steps defined by the kind of change, each deciding from its own material."""

import pytest

from benchmarks.tasks import CO_TASKS, training_task
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v10_22 import Config as V1022Config
from traceaad.v10_22.prompts import PromptBuilder as V1022Prompts
from traceaad.v10_23 import Config, TraceAADV1023
from traceaad.v10_23.config import EXPERIMENT
from traceaad.v10_23.prompts import ANALYSIS, DESIGN, PromptBuilder


def program(i, parent, score, action, code=None):
    return {"id": i, "key": str(i), "parent_id": parent, "valid": True, "fitness": score, "score": score,
            "idea": f"idea {i}", "code": code or f"def score(x):\n    return {i}\n", "action": action}


def state():
    """1 → 2 (Refine) → 3 (Explore) → 4 (Develop); 5 is another root."""
    programs = {1: program(1, None, 4.0, "Init"), 2: program(2, 1, 3.0, "Refine"),
                3: program(3, 2, 3.5, "Explore"), 4: program(4, 3, 2.5, "Develop"),
                5: program(5, None, 3.2, "Init", "def score(x):\n    y = x\n    return y\n")}
    attempts = {i: {"id": i, "action": p["action"], "parent_id": p["parent_id"], "status": "valid",
                    "program_id": i, "idea": p["idea"]} for i, p in programs.items()}
    return programs, attempts


def builder(programs, attempts):
    return PromptBuilder(TokenLLM(), "tiny", TinyEvaluation(), programs, attempts, Config())


def test_search_execution_and_shares_are_v1022s():
    assert EXPERIMENT == "traceaad_v10_23" and TraceAADV1023.METHOD == "v1023"
    assert Config().operators == V1022Config().operators == {"Refine": 0.45, "Explore": 0.30, "Crossover": 0.25}
    assert Config().development_probability == V1022Config().development_probability


def test_refine_reads_the_formation_path_and_changes_one_part():
    programs, attempts = state()
    text = builder(programs, attempts).build("Refine", programs[2])["prompt"]
    assert "[How the Current Algorithm Was Formed]" in text and "one part" in text
    assert f"Analysis: <a few sentences: {ANALYSIS['Refine']}. It will not be shown again>" in text
    assert DESIGN in text and "core idea" not in text


def test_explore_and_crossover_read_code_and_scores_without_formation_paths():
    programs, attempts = state()
    prompts = builder(programs, attempts)
    explore = prompts.build("Explore", programs[2])["prompt"]
    crossover = prompts.build("Crossover", programs[2], reference=programs[5])["prompt"]
    for text in (explore, crossover):
        assert "Was Formed]" not in text and "How the Best Score Improved" not in text
        assert "[Current Algorithm]" in text
    assert "[Reference Algorithm]" in crossover and "return y" in crossover
    assert crossover.index("[Reference Algorithm]") < crossover.index("[Your Task: Crossover]")
    assert "does well" not in crossover
    old = V1022Prompts(TokenLLM(), "tiny", TinyEvaluation(), programs, attempts, V1022Config())
    assert "How the Reference Algorithm Was Formed" in old.build("Crossover", programs[3], reference=programs[5])["prompt"]


def test_develop_reads_the_path_from_where_the_exploration_started():
    programs, attempts = state()
    text = builder(programs, attempts).develop(programs[4], programs[2])["prompt"]
    assert "[Your Task: Develop]" in text
    assert "score 3 → score 3.5" in text and "score 3.5 → score 2.5" in text
    assert "score 4 → score 3" not in text


@pytest.mark.parametrize("task", CO_TASKS)
def test_prompts_name_no_time_and_no_method(task):
    prompts = PromptBuilder(TokenLLM(), task, training_task(task, condition="traceaad")[0], {}, {}, Config())
    text = (prompts.REFINE + prompts.EXPLORE + prompts.CROSSOVER + prompts.DEVELOP
            + " ".join(v for v in ANALYSIS.values() if v)).lower()
    for word in ("second", "time limit", "budget", "efficient", "search", "tour", "route", "2-opt", "rollout", "anneal", "beam"):
        assert word not in text


def test_a_tiny_run_completes_and_develops(tmp_path):
    answers = [response(i) for i in range(1, 9)] + [response(20 + i) for i in range(20)]
    method = TraceAADV1023(evaluation=TinyEvaluation(), llm=TokenLLM(*answers), run_dir=tmp_path,
                           config=Config(budget=20, development_probability=1.0))
    summary = method.run()
    assert summary["status"] == "finished" and summary["method"] == "v1023"
    prompts = [prompt for prompt, _ in method.llm.calls]
    assert any("[Your Task: Explore]" in p for p in prompts)
    developed = [p for p in prompts if "[Your Task: Develop]" in p]
    assert developed and all("[How the Current Algorithm Was Formed]" in p for p in developed)


def test_a_timeout_says_how_far_the_run_got_without_time():
    evaluation = TinyEvaluation()
    evaluation.n_instance = 2
    programs, attempts = state()
    failed = {"id": 9, "parent_id": 2, "valid": False, "action": "Explore", "idea": "x", "code": "def score(x):\n    return 0\n",
              "failure": {"kind": "timeout", "calls": 14, "call_running": True, "function_seconds": 2.0}}

    def text(calls):
        shown = {i: {**p, "calls": calls} for i, p in programs.items()} if calls else {}
        return PromptBuilder(TokenLLM(), "tiny", evaluation, {**shown, 9: failed}, attempts, Config()).error_text(failed)

    assert text(96) == "Timed out: on one instance, the evaluation stopped after the function had returned from 14 of its 48 calls."
    assert text(2) == "Timed out: the evaluation stopped before the function returned from its single call on an instance."
    assert text(None).endswith("had returned from 14 calls.")
    assert "second" not in text(96) and " s " not in text(96)
