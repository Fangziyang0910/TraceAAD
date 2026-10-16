"""All six tasks are reachable from baseline and TraceAAD experiment entries."""

import pytest

from benchmarks.tasks import (CO_TASKS, PRIMARY_SPLITS, SCALES, SPLITS, TEST_INSTANCES, heldout_task,
                              training_task, FIXED_TASKS)
from experiments import launch
from experiments.infra.search_host_launch import plan_for

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
    plan = plan_for("test", "traceaad_v12_0", "experiments.traceaad_v12_1.run", tasks=CO_TASKS, repeats=2, budget=100)
    assert len(plan) == 12 and len({item["run_dir"] for item in plan}) == 12
    assert all("--budget=100" in item["command"] for item in plan)


@pytest.mark.parametrize("task", TASKS)
def test_task_conditions_match_primary_data(task):
    training, _ = training_task(task)
    testing = heldout_task(task, PRIMARY_SPLITS[task][0])
    assert testing.outer_settings == training.outer_settings
    assert testing.n_instance == TEST_INSTANCES
    assert testing.timeout_seconds / testing.n_instance == training.timeout_seconds / training.n_instance
    assert SCALES[task] == (training.problem_size,)
    assert SPLITS[task] == PRIMARY_SPLITS[task]


@pytest.mark.parametrize('task', TASKS)
def test_primary_aliases_have_the_same_evaluation_identity(task):
    from traceaad.v12_1.evaluation import ProgramEvaluator
    primary = heldout_task(task, PRIMARY_SPLITS[task][0])
    alias = heldout_task(task, 'eval')
    assert ProgramEvaluator(primary).protocol == ProgramEvaluator(alias).protocol


@pytest.mark.parametrize('task', ['graph_colouring', 'jssp_construct'])
def test_complete_solver_interface_is_saved_and_legacy_heldout_is_blocked(task):
    from benchmarks.tasks import validate_task_interface
    from experiments.infra.evaluation_execution import heldout_evaluator
    evaluation, task_config = training_task(task)
    assert evaluation.FUNCTION_NAME == 'solve'
    assert evaluation.outer_settings == {}
    assert task_config['interface'] == evaluation.interface == 'full-solver-v1'
    validate_task_interface(task, {'task_eval': task_config})
    def must_not_build(*args):
        raise AssertionError('legacy result must not be evaluated or overwritten')
    with pytest.raises(ValueError, match='saved source snapshot'):
        heldout_evaluator({'task': task}, PRIMARY_SPLITS[task][0], (730241,),
                          workers=1, timeout_seconds=None, measure_calls=True,
                          task_factory=must_not_build)


def test_old_run_config_is_not_relabelled_on_resume(tmp_path, monkeypatch):
    import json
    from experiments.infra import runner
    from tests.task.test_co_tasks import miniature_evaluation
    from experiments.infra.search_run import build_parser
    args = build_parser('traceaad_test').parse_args([
        '--task', 'jssp_construct', '--run-name', 'old', '--backend', 'local'])
    run = tmp_path / args.task / 'old'
    run.mkdir(parents=True)
    config = run / 'run_config.json'
    config.write_text(json.dumps({'task': args.task, 'task_eval': {'split': 'train'}}))
    (run / 'events.jsonl').write_text('')
    before = config.read_bytes()
    monkeypatch.setattr(runner, 'build_task', lambda *a, **k: (miniature_evaluation(), {'interface': 'full-solver-v1'}))
    with pytest.raises(ValueError, match='new run directory'):
        runner.setup_experiment_run(args, method='test', results_root=tmp_path, resume_file='events.jsonl')
    assert config.read_bytes() == before
