"""V10.15 launch and held-out isolation contracts."""

import json

import pytest

from experiments.traceaad_v10_15 import run
from experiments.infra import search_heldout as heldout
from experiments.traceaad_v10_15.heldout_batch import jobs
from experiments.traceaad_v10_15.launch_batch import build_plan
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v10_15 import Config, TraceAADV1015


class ShiftedEvaluation(TinyEvaluation):
    def __init__(self):
        super().__init__()
        self.shift = 10

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1) + self.shift


def test_dry_run_and_batch_plan(capsys):
    run.main(['--task', 'online_bin_packing', '--dry-run'])
    rendered = json.loads(capsys.readouterr().out)
    assert rendered['search_timeout'] == 30
    assert rendered['config']['budget'] == 1000
    tasks = ['tsp_construct', 'cvrp_aco', 'op_aco', 'online_bin_packing', 'vrptw_construct']
    previous = {'plan': [{'task': task, 'repeat': repeat, 'seed': repeat - 1,
                          'backend': 'server3', 'session': f'old-{task}-{repeat}'}
                         for task in tasks for repeat in range(1, 5)]}
    plan = build_plan(previous, 'newbatch')
    assert len(plan) == len({row['run_name'] for row in plan}) == 20
    assert all(row['command'][3] == '-m' and 'experiments.traceaad_v10_15.run' in row['command']
               and row['seed'] == row['repeat'] - 1 for row in plan)


def test_heldout_only_after_selection_and_uses_selected_key(tmp_path, monkeypatch):
    m = TraceAADV1015(evaluation=TinyEvaluation(), selection_evaluation=ShiftedEvaluation(),
                       llm=TokenLLM(response(2)), run_dir=tmp_path,
                       config=Config(budget=1))
    assert m.run()['status'] == 'finished'
    (tmp_path / 'run_config.json').write_text(json.dumps({
        'task': 'tsp_construct', 'method_params': {'evaluation_seeds': [730241]}}))
    monkeypatch.setattr(heldout, 'heldout_task', lambda task, split, workers, timeout: ShiftedEvaluation())
    result = heldout.evaluate_run(tmp_path)
    assert result['fitness'] == 12
    assert result['key'] == json.loads((tmp_path / 'selection.json').read_text())['selected_key']
    assert heldout.evaluate_run(tmp_path) == result
    (tmp_path / 'best_program.py').write_text('def score(x): return 100\n')
    with pytest.raises(ValueError, match='changed after selection'):
        heldout.evaluate_run(tmp_path)


def test_heldout_batch_lists_every_report_size():
    tasks = ['tsp_construct', 'cvrp_aco', 'op_aco', 'online_bin_packing', 'vrptw_construct']
    plan = [{'task': task, 'repeat': repeat, 'run_dir': f'/tmp/{task}/{repeat}'}
            for task in tasks for repeat in range(1, 5)]
    report = jobs({'method': 'v1015', 'plan': plan})
    assert len(report) == 76
    assert any(task == 'online_bin_packing' and split == 'eval_10000_500'
               for task, _, _, split in report)
    assert any(task == 'tsp_construct' and split == 'eval_200'
               for task, _, _, split in report)


def test_generated_heldout_split_parameters():
    tsp = heldout.heldout_task('tsp_construct', 'eval_100', 2)
    assert tsp.problem_size == 100 and tsp.seed == 2025
    assert tsp.timeout_seconds >= 30
    obp = heldout.heldout_task('online_bin_packing', 'eval_5000_500', 2)
    assert len(obp.dataset_metadata) == 5
    assert all(items == 5000 and capacity == 500 for _, items, capacity in obp.dataset_metadata)
    with pytest.raises(ValueError):
        heldout.heldout_task('online_bin_packing', 'eval_750_100', 2)
