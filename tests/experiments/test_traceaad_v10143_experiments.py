"""Real evaluator, selection, monitoring and launch contracts for V10.14-3."""

import json

import pytest

from experiments.traceaad_v10_14_3 import run as runner
from experiments.traceaad_v10_14_3.launch_batch import build_plan
from experiments.traceaad_v10_14_3.preflight import small_task
from tests.support import TinyEvaluation, TokenLLM, text_candidate as payload
from traceaad.v10_14_3 import Config, TraceAADV10143


@pytest.mark.parametrize('task', ['tsp_construct', 'vrptw_construct', 'online_bin_packing', 'cvrp_aco', 'op_aco'])
def test_text_candidate_real_task_and_independent_selection(tmp_path, task):
    train, selection = small_task(task), small_task(task, seed=11)
    response = payload(idea='Use the supplied task implementation as a protocol check.', code=train.template_program.strip())
    m = TraceAADV10143(evaluation=train, selection_evaluation=selection,
                       llm=TokenLLM(*([response]*3)), run_dir=tmp_path, task=task,
                       config=Config(budget=3, max_evaluations=3, init_proposals=1))
    m._contract = lambda a: ('Pivot', 'None', None)
    result = m.run()
    assert result['status'] == 'finished', result
    assert result['selection_evaluations'] == 1
    assert m.ledger.candidates == 3 and m.ledger.evaluations == 1
    assert all(a['idea'] and a['delivery']['strategy'] == 'explicit_final' for a in m.anchors.values())
    assert m.facts.tables['attempt'][3]['context_mode'] == 'independent'
    from experiments.infra.artifacts import pick_best_sample
    sample, _ = pick_best_sample(tmp_path)
    assert sample['program'] == result['best']['code']


def test_frozen_defaults_dry_run_and_batch_mapping(capsys):
    runner.main(['--task', 'tsp_construct', '--dry-run'])
    data = json.loads(capsys.readouterr().out)
    cfg = data['config']
    assert cfg['parent_policy'] == 'rank_count' and cfg['pivot_context'] == 'independent'
    assert cfg['output_mode'] == 'full' and cfg['budget'] == 1000
    tasks = ['tsp_construct', 'cvrp_aco', 'op_aco', 'online_bin_packing', 'vrptw_construct']
    previous = {'plan': [{'task': task, 'repeat': rep, 'seed': rep-1, 'backend': 'server3',
                          'run_dir': f'/old/{task}/{rep}'} for task in tasks for rep in range(1, 5)]}
    plan = build_plan(previous, 'test_v3')
    assert len(plan) == len({p['run_name'] for p in plan}) == 20
    for old, new in zip(previous['plan'], plan):
        assert all(old[k] == new[k] for k in ('task', 'repeat', 'backend', 'seed'))
        assert 'experiments.traceaad_v10_14_3.run' in new['command']
        assert new['run_dir'] != old['run_dir']


def test_monitor_reads_new_version_candidate_budget(tmp_path):
    from experiments.monitor import ResultsMonitor
    directory = tmp_path / 'traceaad_v10_14_3' / 'tsp_construct' / 'test'
    m = TraceAADV10143(evaluation=TinyEvaluation(), llm=TokenLLM(payload()),
                       run_dir=directory, config=Config(budget=8, max_evaluations=8, init_proposals=1))
    m._initialize()
    (directory / 'run_config.json').write_text(json.dumps({'task': 'tsp_construct', 'method': 'v1014_3'}))
    monitor = ResultsMonitor(tmp_path)
    view = monitor.overview('traceaad_v10_14_3')
    assert view['summary']['running'] == 1 and view['summary']['budget_used'] == 1
    assert monitor._recorded_progress(directory) == (1, 1)
