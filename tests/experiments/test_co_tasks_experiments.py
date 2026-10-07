"""All six tasks are reachable from baseline and TraceAAD experiment entries."""

import pytest

from benchmarks.tasks import (CO_TASKS, PRIMARY_SPLITS, SCALES, SPLITS, heldout_task, training_task,
                              FIXED_TASKS)
from experiments import launch
from experiments.infra.search_host_launch import plan_for
from traceaad.v10_20.prompts import PromptBuilder
from traceaad.v10_20 import Config
from tests.support import TokenLLM

TASKS = tuple(task for task in CO_TASKS if task in FIXED_TASKS)


def test_primary_suite_replaces_easy_tasks_with_jssp_and_op():
    assert set(CO_TASKS) == {'tsp_construct', 'cvrp_aco', 'fssp_gls',
                             'graph_colouring', 'jssp_construct', 'op_aco'}


@pytest.mark.parametrize("method", launch.METHODS)
def test_baseline_co6_plan_contains_every_new_task(method):
    args = launch.build_parser().parse_args(["--method", method, "--suite", "co6", "--batch", "test", "--repeats", "2"])
    plan = launch.build_plan(args)
    assert len(plan) == 12 and {item.task for item in plan} == set(CO_TASKS)


def test_traceaad_suite_budget_and_run_names_are_consistent():
    plan = plan_for("test", "traceaad_v10_20_co6", "experiments.traceaad_v10_20.run", tasks=CO_TASKS, repeats=2, budget=100)
    assert len(plan) == 12 and len({item["run_dir"] for item in plan}) == 12
    assert all("--budget=100" in item["command"] for item in plan)


@pytest.mark.parametrize("task", TASKS)
def test_task_conditions_and_prompt_match_primary_data(task):
    training, _ = training_task(task)
    testing = heldout_task(task, PRIMARY_SPLITS[task][0])
    assert testing.outer_settings == training.outer_settings
    assert testing.n_instance == 100
    assert testing.timeout_seconds / testing.n_instance == training.timeout_seconds / training.n_instance
    assert SCALES[task] == (training.problem_size,)
    assert SPLITS[task] == PRIMARY_SPLITS[task]
    text = "\n".join(PromptBuilder(TokenLLM(), task, training, {}, {}, Config()).common)
    assert training.instance_description in text
    assert "100 fixed generated test" in text or "108 fixed generated test" in text
    assert "Lower is better" in text
    assert text.count(training.design_notes) == 1


@pytest.mark.parametrize('task', TASKS)
def test_primary_aliases_have_the_same_evaluation_identity(task):
    from traceaad.common.evaluation import ProgramEvaluator
    primary = heldout_task(task, PRIMARY_SPLITS[task][0])
    alias = heldout_task(task, 'eval')
    assert ProgramEvaluator(primary).protocol == ProgramEvaluator(alias).protocol
