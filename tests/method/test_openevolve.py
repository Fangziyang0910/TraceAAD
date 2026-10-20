"""Offline checks for the OpenEvolve loop, its score mapping, and recovery."""

import math
import re

import pytest

from baselines.openevolve import OpenEvolve
from baselines.openevolve.openevolve import MAX_CODE_LENGTH, OBJECTIVE_NOTE
from baselines.shared.profiler import ProfilerBase
from core import Evaluation
from core.evaluate import EvaluationOutcome
from core.code import parse_program
from experiments.infra.resume_baseline import recover
from tests.support import TokenLLM
from traceaad.storage import read_json, rows, write_json


TEMPLATE = "def solve(x):\n    return 1\n"


class ToyTask(Evaluation):
    def __init__(self):
        super().__init__(TEMPLATE, "Minimize the returned value.", safe_evaluate=False)
        self.evaluated = []

    def evaluate_program(self, code, function, **kwargs):
        self.evaluated.append(code)
        value = float(function(0))
        if value <= -1:
            raise RuntimeError("PRIVATE_FAILURE_MESSAGE")
        return value


def current_program(prompt):
    return prompt[1]["content"].split("# Current Program\n```python\n", 1)[1].split("```", 1)[0]


class Mutations(TokenLLM):
    """Decrease the parent's return value; optionally reply with a bad diff or fail transport."""

    def __init__(self, *, stop_after=None, bad_first=None, fail_first=False):
        super().__init__()
        self.stop_after, self.bad_first, self.fail_first = stop_after, bad_first, fail_first

    def draw_sample_with_details(self, prompt, **kwargs):
        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            raise KeyboardInterrupt
        self.calls.append(prompt)
        if self.fail_first and len(self.calls) == 1:
            raise ValueError("Offline transport failure")
        line = re.findall(r"^    return .*$", current_program(prompt), re.M)[0]
        current = int(line.split()[1])
        response = (f"Lower the constant.\n<<<<<<< SEARCH\n{line}\n=======\n"
                    f"    return {current - 1}  # v{len(self.calls)}\n>>>>>>> REPLACE\n")
        if self.bad_first is not None and len(self.calls) == 1:
            response = self.bad_first
        return {"content": response, "model": "offline",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


def method(directory, llm, *, budget=6, **options):
    task = ToyTask()
    write_json(directory / "run_config.json", {"method": "openevolve", "task": "toy", "budget": budget,
                                                "method_params": {}})
    instance = OpenEvolve(llm, task, ProfilerBase(directory), max_sample_nums=budget, random_seed=9,
                          num_islands=2, **options)
    return instance, task


def test_prompt_keeps_upstream_sections_and_states_the_objective_direction(tmp_path):
    llm = Mutations()
    instance, _ = method(tmp_path, llm, budget=3)
    instance.run()
    system, user = llm.calls[-1][0]["content"], llm.calls[-1][1]["content"]
    assert system.startswith("You are an expert software developer tasked with iteratively improving")
    assert "Minimize the returned value." in system and system.endswith(OBJECTIVE_NOTE)
    assert "- Fitness: -" in user and "# Program Evolution History" in user
    assert "## Top Performing Programs" in user and "<<<<<<< SEARCH" in user
    assert current_program(llm.calls[0]).strip() == TEMPLATE.strip()


def test_failed_program_stays_in_database_below_valid_ones_and_shows_its_error(tmp_path):
    instance, task = method(tmp_path, Mutations(), budget=5)
    instance.run()
    database = instance._database
    failed = [p for p in database.programs.values() if not math.isfinite(p.metrics["combined_score"])]
    assert failed and all(p.metrics["error"] == 0.0 for p in failed)
    best = database.programs[database.best_program_id]
    assert best.metrics["combined_score"] > -math.inf and instance.best_function.score == best.metrics["combined_score"] * -1
    artifacts = database.get_artifacts(failed[0].id)
    assert "PRIVATE_FAILURE_MESSAGE" in artifacts["stderr"] and "traceback" not in artifacts
    prompt = instance._prompt(failed[0], [], 9, 0)[1]["content"]
    assert "## Last Execution Output" in prompt and "PRIVATE_FAILURE_MESSAGE" in prompt
    summary = read_json(tmp_path / "summary.json")
    assert summary["best"]["fitness"] == min(r["fitness"] for r in rows(tmp_path / "events.jsonl")
                                            if r["kind"] == "candidate" and r["valid"])


def test_timeout_reports_upstream_fields_without_instance_counts():
    metrics, artifacts = OpenEvolve._metrics(EvaluationOutcome(
        None, failure_kind="timeout", error="mean function CPU time 3.1s exceeded 2s across 16 instances"))
    assert metrics == {"error": 0.0, "combined_score": -math.inf, "timeout": True}
    assert artifacts["timeout"] is True and artifacts["stderr"] == "mean function CPU time 3.1s exceeded 2s"
    for error, shown in (("instance 7 exceeded 120s", "an instance exceeded 120s"),
                         ("instance 3 worker exited with code 9", "an instance worker exited with code 9")):
        assert OpenEvolve._metrics(EvaluationOutcome(None, failure_kind="timeout", error=error))[1]["stderr"] == shown
    assert OpenEvolve._metrics(EvaluationOutcome(4.5)) == ({"combined_score": -4.5}, {})


@pytest.mark.parametrize("reply", [
    "No diff here.",
    "<<<<<<< SEARCH\n    return 7\n=======\n    return 6\n>>>>>>> REPLACE\n",
    "<<<<<<< SEARCH\n    return 1\n=======\n    return 1\n>>>>>>> REPLACE\n",
    "<<<<<<< SEARCH\n    return 1\n=======\n=======\n    return 0\n>>>>>>> REPLACE\n",
    f"<<<<<<< SEARCH\n    return 1\n=======\n    return 0  # {'x' * MAX_CODE_LENGTH}\n>>>>>>> REPLACE\n",
])
def test_rejected_diff_spends_one_attempt_without_evaluation(tmp_path, reply):
    llm = Mutations(bad_first=reply)
    instance, task = method(tmp_path, llm, budget=3)
    instance.run()
    summary = read_json(tmp_path / "summary.json")
    assert summary["status"] == "finished" and summary["budget_used"] == 3
    assert len(llm.calls) == 2 and len(task.evaluated) == 2 == summary["evaluation_calls"]
    events = [r["data"] for r in rows(tmp_path / "events.jsonl") if r["kind"] == "method"]
    assert any(e["event"] == "diff_rejected" for e in events)


def test_transport_failure_does_not_spend_budget(tmp_path):
    llm = Mutations(fail_first=True)
    instance, task = method(tmp_path, llm, budget=3)
    instance.run()
    summary = read_json(tmp_path / "summary.json")
    assert summary["budget_used"] == 3 and len(llm.calls) == 3 and len(task.evaluated) == 3
    candidates = [r for r in rows(tmp_path / "events.jsonl") if r["kind"] == "candidate"]
    assert [r["budget_used"] for r in candidates] == [1, 2, 3]
    assert candidates[-1]["program"]["parent_id"] is not None
    assert summary["token_usage"]["total_tokens"] == 60


def test_islands_follow_round_robin_and_migration_copies_keep_their_source(tmp_path):
    instance, _ = method(tmp_path, Mutations(), budget=9, migration_interval=2)
    instance.run()
    events = [r["data"] for r in rows(tmp_path / "events.jsonl") if r["kind"] == "method"]
    islands = [e["island"] for e in events if e["event"] == "parent_selection"]
    assert islands == [0, 1] * 4
    migrants = [p for p in instance._database.programs.values() if p.metadata.get("migrant")]
    assert migrants and all(instance._candidate(p) == int(p.parent_id) for p in migrants)
    assert any(e["event"] == "island_migration" and e["migrants"] for e in events)


def comparable(database):
    """Program IDs and set order follow str hashes, so compare the population's content."""
    return sorted((p.metadata["candidate_id"], p.metadata["island"], p.code, p.metrics["combined_score"])
                  for p in database.programs.values())


def test_resume_restores_database_content_without_reevaluating(tmp_path):
    first, _ = method(tmp_path, Mutations(stop_after=2))
    first.run()
    resumed, task = method(tmp_path, Mutations())
    report = recover(resumed, tmp_path)
    assert report["recovery"] == "search_checkpoint" and report["budget_used"] == 3 and task.evaluated == []
    assert comparable(resumed._database) == comparable(first._database)
    assert resumed._database.diversity_by_code.keys() == first._database.diversity_by_code.keys()
    resumed.run()
    assert len(task.evaluated) == 3
    candidates = [r for r in rows(tmp_path / "events.jsonl") if r["kind"] == "candidate"]
    assert [r["candidate_id"] for r in candidates] == [1, 2, 3, 4, 5, 6]
    summary = read_json(tmp_path / "summary.json")
    assert summary["budget_used"] == 6 and summary["evaluation_calls"] == 6


@pytest.mark.parametrize("stop_at", [1, 2])
def test_interrupted_evaluation_resumes_without_spending_another_attempt(tmp_path, monkeypatch, stop_at):
    from experiments.infra import resume_baseline
    first, _ = method(tmp_path, Mutations(), budget=2)
    evaluate = first._evaluator.evaluate_program_record_time_with_details
    count = 0

    def interrupted(program):
        nonlocal count
        count += 1
        if count == stop_at:
            raise KeyboardInterrupt
        return evaluate(program)

    monkeypatch.setattr(first._evaluator, "evaluate_program_record_time_with_details", interrupted)
    first.run()
    assert read_json(tmp_path / "summary.json")["status"] == "interrupted"
    llm = Mutations()
    resumed, task = method(tmp_path, llm, budget=2)
    monkeypatch.setattr(resume_baseline, "build", lambda *a: resumed)
    monkeypatch.setattr("benchmarks.tasks.validate_task_interface", lambda *a: None)
    monkeypatch.setattr("sys.argv", ["resume", "--run-dir", str(tmp_path), "--execute"])
    monkeypatch.setenv("TRACEAAD_EVALUATION_LOG", "")
    resume_baseline.main()
    assert read_json(tmp_path / "summary.json")["status"] == "finished"
    assert len(llm.calls) == 2 - stop_at
    assert len(task.evaluated) == 3 - stop_at
    assert resumed.best_function.score == 0 and resumed._tot_sample_nums == 2


@pytest.mark.parametrize("task_name", [
    "tsp_construct", "cvrp_aco", "fssp_gls", "graph_colouring", "jssp_construct", "op_aco"])
def test_real_solver_entry_final_selection_monitor_and_config_rebuild(tmp_path, monkeypatch, task_name):
    from benchmarks.tasks import training_task
    from traceaad.instance_evaluation import _instance_view
    from experiments.openevolve import run
    from experiments.infra import base
    from experiments.infra.resume_baseline import build, close_recovery_method
    from experiments.visualization import ResultsMonitor

    monkeypatch.delenv("TRACEAAD_SCHEDULER_SOCKET", raising=False)
    monkeypatch.setenv("TRACEAAD_EVALUATION_LOG", "")
    original, settings = training_task(task_name)
    task = _instance_view(original, 0, original.timeout_seconds)
    monkeypatch.setattr(base, "training_task", lambda *a, **k: (task, settings))
    last = task.template_program.rstrip("\n").splitlines()[-1]
    reply = f"<<<<<<< SEARCH\n{last}\n=======\n{last}\nOPENEVOLVE_SMOKE = 1\n>>>>>>> REPLACE\n"
    clients = []

    def client(**kwargs):
        llm = TokenLLM(reply)
        clients.append(llm)
        return llm

    monkeypatch.setattr(run, "build_llm_client", client)
    spec = run.RunSpec(task=task_name, budget=2, eval_workers=1,
                       run_name="integration", experiments_root=tmp_path)
    directory = run.run_experiment(spec)
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "finished" and summary["selection_evaluations"] == 0
    candidates = [r for r in rows(directory / "events.jsonl") if r["kind"] == "candidate"]
    assert len(candidates) == 2 and all(r["valid"] for r in candidates)
    assert candidates[0]["fitness"] == candidates[1]["fitness"] == summary["best"]["fitness"]
    assert parse_program((directory / "best_program.py").read_text()) is not None
    detail = ResultsMonitor(tmp_path).run_detail("openevolve", task_name, "integration")
    assert detail["best"]["code"] == (directory / "best_program.py").read_text()
    rebuilt = build(read_json(directory / "run_config.json"), directory)
    try:
        assert clients[-1].calls == []
        report = recover(rebuilt, directory)
        assert report["recovery"] == "search_checkpoint" and rebuilt._tot_sample_nums == 2
    finally:
        close_recovery_method(rebuilt)
