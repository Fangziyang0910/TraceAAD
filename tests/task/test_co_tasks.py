"""Independent feasibility oracles and the search/selection/held-out path."""

import itertools
import json

import numpy as np
import pytest

from benchmarks.fssp_gls import evaluation as fssp
from benchmarks.graph_colouring import evaluation as graph
from benchmarks.jssp_construct import evaluation as jssp
from benchmarks.tasks import CLASSES, FIXED_TASKS as TASKS
from benchmarks.shared.seeded_data import digest_arrays, generate_dataset
from core import SecureEvaluator
from core.evaluate import InvalidEvaluationResult


def seed(task):
    namespace = {}
    exec(CLASSES[task].TEMPLATE, namespace)
    return namespace[CLASSES[task].FUNCTION_NAME]


def test_flowshop_recurrence_and_true_objective_after_perturbation():
    times = np.array([[2, 3], [4, 1], [1, 5]], dtype=float)

    def oracle(order):
        completion = np.zeros((4, 3))
        for i, job in enumerate(order, 1):
            for machine in range(1, 3):
                completion[i, machine] = max(completion[i-1, machine], completion[i, machine-1]) + times[job, machine-1]
        return completion[-1, -1]

    for order in itertools.permutations(range(3)):
        assert fssp.makespan(np.array(order), times) == oracle(order)
    result, cost = fssp.solve({"processing_times": times},
                            lambda seq, t, m, n: (np.zeros_like(t), np.array([seq[-1]])), iterations=2)
    assert sorted(result) == [0, 1, 2]
    assert cost == oracle(result) and cost > 0


@pytest.mark.parametrize("output", [None, (np.ones((3, 2)), np.array([])),
                                   (np.ones((3, 2)), np.array([0, 0])),
                                   (np.ones((2, 3)), np.array([0])),
                                   (np.full((3, 2), np.nan), np.array([0]))])
def test_flowshop_rejects_invalid_perturbations(output):
    with pytest.raises(InvalidEvaluationResult):
        fssp.solve({"processing_times": np.ones((3, 2))}, lambda *args: output, iterations=1)


@pytest.mark.parametrize("adjacency,optimal", [(np.ones((3, 3), dtype=bool) ^ np.eye(3, dtype=bool), 3),
    (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool), 2)])
def test_graph_starter_returns_a_valid_complete_coloring(adjacency, optimal):
    colors, k = graph.solve({"adjacency": adjacency}, seed("graph_colouring"))
    assert graph.validate_coloring(adjacency, colors) == k == optimal


def miniature_evaluation():
    class TinyData:
        TASK = 'jssp_construct'
        SCALE = 2
        COUNTS = {'train': 1, 'test': 1}
        DIMENSIONS = {'jobs': 2, 'machines': 2, 'operations': 4}
        DISTRIBUTION = 'two jobs visiting two machines in opposite orders'

        @staticmethod
        def describe(split, count=None):
            return f"{count} fixed {split} instances with 2 jobs and 2 machines"

        @staticmethod
        def generate_instances(split):
            arrays = {'processing_times': np.array([[3, 2], [2, 4]]),
                      'machine_order': np.array([[0, 1], [1, 0]])}
            yield arrays, {'id': split, 'group': split, 'scale': 2,
                           'dimensions': TinyData.DIMENSIONS}, 8., 'test reference'

    class TinyJSSPEvaluation(jssp.JSSPEvaluation):
        DATASET = TinyData

    return TinyJSSPEvaluation(safe_evaluate=False)


def test_eval_isolates_module_state_and_measures_calls():
    from traceaad.common.evaluation import ProgramEvaluator
    evaluation = miniature_evaluation()
    evaluation._rows *= 2
    evaluation._instances *= 2
    code = ('import numpy as np\n_calls = 0\n'
            'def helper():\n    return np.array([[0,3],[0,3]])\n'
            'def solve(t, m):\n'
            '    global _calls\n    _calls += 1\n    assert _calls == 1\n'
            '    return helper()\n')
    result = ProgramEvaluator(evaluation, (123,)).evaluate(code, "isolation")
    assert result["failure"] is None
    assert result["calls"] == 2
    assert result["fitness"] == pytest.approx(7.)


def test_invalid_scores_are_reported_as_invalid_output():
    from traceaad.common.evaluation import ProgramEvaluator
    evaluation = miniature_evaluation()
    result = ProgramEvaluator(evaluation).evaluate('def solve(*args):\n    return [float("nan")]\n', "bad")
    assert result["failure"]["kind"] == "invalid_output"


def test_candidates_reuse_initial_data_and_references(monkeypatch):
    from traceaad.common.evaluation import ProgramEvaluator
    evaluation = miniature_evaluation()
    before = [digest_arrays(data) for data in evaluation._instances]

    def unexpected_generation(split):
        raise AssertionError('data generation must stay outside candidate evaluation')

    monkeypatch.setattr(evaluation.DATASET, 'generate_instances', unexpected_generation)
    runner = ProgramEvaluator(evaluation)
    first = runner.evaluate(evaluation.template_program, 'first')
    second = runner.evaluate(evaluation.template_program, 'second')
    assert first['failure'] is second['failure'] is None
    assert first['fitness'] == second['fitness'] == pytest.approx(7.)
    assert [digest_arrays(data) for data in evaluation._instances] == before


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('split', ('train', 'test'))
def test_fixed_data_ignore_global_rng_and_keep_the_same_prefix(task, split):
    state = np.random.get_state()
    try:
        np.random.seed(1)
        first = CLASSES[task](split=split, limit=1, safe_evaluate=False)
        np.random.seed(991)
        second = CLASSES[task](split=split, limit=2, safe_evaluate=False)
    finally:
        np.random.set_state(state)
    assert first._datasets == second._datasets[:1]
    assert all(np.array_equal(value, second._instances[0][key]) for key, value in first._instances[0].items())
    assert first._rows[0]['seed_entropy'][2] == (0 if split == 'train' else 2)


@pytest.mark.parametrize("task", TASKS)
def test_packaged_template_runs_through_secure_evaluation(task):
    evaluation = CLASSES[task](limit=1, safe_evaluate=False)
    outcome = SecureEvaluator(evaluation).evaluate_program_with_details(evaluation.template_program)
    assert outcome.failure_kind is None, outcome.error
    assert np.isfinite(outcome.result)


def test_packaged_splits_have_no_base_group_or_content_overlap():
    for task in TASKS:
        data = CLASSES[task].DATASET
        train, _ = generate_dataset(data, 'train')
        test, _ = generate_dataset(data, 'test')
        assert len(train) == data.COUNTS['train'] and len(test) == data.COUNTS['test']
        rows = train + test
        for field in ("group", "content"):
            assert {r['split'] for r in rows} == {'train', 'test'}
            groups = {split: {r[field] for r in rows if r["split"] == split} for split in ("train", "test")}
            assert all(groups[a].isdisjoint(groups[b]) for a, b in itertools.combinations(groups, 2)), task
        assert len({json.dumps(r["dimensions"], sort_keys=True) for r in rows}) == 1


def test_offline_training_winner_and_heldout_use_the_new_task(tmp_path):
    from experiments.infra.evaluate import evaluate_run
    from tests.support import TokenLLM, text_candidate
    from traceaad.v12_0 import BaseConfig as Config, TraceAADV120
    from traceaad.common.storage import write_json
    evaluation = miniature_evaluation()
    run = tmp_path / "run"
    llm = TokenLLM(text_candidate(code=evaluation.template_program))
    config = Config(budget=1, roots=1, final_candidates=1)
    method = TraceAADV120(evaluation=evaluation,
                           llm=llm, run_dir=run, task="jssp_construct", config=config)
    summary = method.run()
    assert summary['status'] == 'finished' and summary['selection_evaluations'] == 0
    assert len(llm.calls) == 1
    write_json(run / "run_config.json", {"task": "jssp_construct", "task_eval": {"interface": "full-solver-v1"},
                                         "method_params": {"evaluation_seeds": [730241]}})
    # Real task data are used for held-out; a frozen valid heuristic must transfer.
    results = evaluate_run(run, ["test_20"], condition="traceaad")
    assert results[0]["fitness"] is not None and not results[0]["failures"]


@pytest.mark.parametrize('task', ['graph_colouring', 'jssp_construct'])
def test_complete_solver_helpers_are_inside_instance_cpu_budget(task):
    from traceaad.common.instance_evaluation import InstanceProgramEvaluator
    evaluation = CLASSES[task](limit=1)
    runner = InstanceProgramEvaluator(evaluation, timeout_seconds=5, function_seconds=.05)
    code = '''import time

def helper():
    start = time.process_time()
    while time.process_time() - start < .2:
        pass

def solve(*args):
    helper()
    return []
'''
    code = evaluation.TEMPLATE.replace('def solve(', 'def solve_original(') + code.replace('return []', 'return solve_original(*args)')
    result = runner.evaluate(code, 'helper-budget')
    assert result['failure']['kind'] == 'timeout'
    assert result['failure']['error_type'] == 'FunctionBudgetExceeded'


def test_complete_solver_search_rejects_a_different_saved_protocol(tmp_path):
    from tests.support import TokenLLM, text_candidate
    from traceaad.v12_0 import BaseConfig as Config, TraceAADV120
    evaluation = miniature_evaluation()
    config = Config(budget=1, roots=1)
    method = TraceAADV120(evaluation=evaluation, llm=TokenLLM(text_candidate(code=evaluation.TEMPLATE)),
                          run_dir=tmp_path, config=config, task='jssp_construct')
    method.run()
    before = (tmp_path / 'resume.json').read_bytes()
    resumed = TraceAADV120(evaluation=evaluation, llm=TokenLLM(), run_dir=tmp_path,
                           config=config, task='jssp_construct')
    assert resumed.attempts == 1
    evaluation.interface = 'different-interface'
    with pytest.raises(ValueError, match='same task interface'):
        TraceAADV120(evaluation=evaluation, llm=TokenLLM(), run_dir=tmp_path,
                      config=config, task='jssp_construct')
    assert (tmp_path / 'resume.json').read_bytes() == before


def test_raw_objective_is_independent_of_reference_values():
    evaluation = miniature_evaluation()
    before = evaluation.evaluate_program(evaluation.template_program, None)
    evaluation._rows[0]['reference'] = 8000.
    assert evaluation.evaluate_program(evaluation.template_program, None) == before == 7.
    assert evaluation.score_metric == 'raw_objective'
