import json
import pytest
from benchmarks.tasks import CO_TASKS


@pytest.mark.parametrize('task', CO_TASKS)
def test_v13_entry_preserves_shared_task_conditions(task, capsys):
    from experiments.traceaad_v13.run import main
    main(['--task', task, '--budget', '8', '--roots', '2', '--eval-workers', '2', '--dry-run'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['method'] == 'traceaad_v13'
    assert plan['config']['max_model_calls'] == 64
    assert plan['config']['roots'] == 2
    assert plan['config']['output_tokens'] == 8192
    assert plan['final_selection'] == 'training'
    assert plan['evaluation_execution']['function_budget_scope'] == 'mean_instance'


def test_v13_batch_roundtrips_all_search_settings(capsys):
    from experiments.traceaad_v13.launch_host import main
    main(['--batch', 'preview', '--suite', 'co6', '--budget', '1000',
          '--backends', 'server3', '--eval-workers', '8', '--max-model-calls', '6000',
          '--open-floor', '.4', '--library-path', '/tmp/library.json'])
    batch = json.loads(capsys.readouterr().out)
    assert len(batch['plan']) == 18 and batch['method'] == 'traceaad_v13'
    assert batch['search_policy']['max_model_calls'] == 6000
    for item in batch['plan']:
        assert item['command'][1:3] == ['-m', 'experiments.traceaad_v13.run']
        assert '--max-model-calls=6000' in item['command']
        assert '--open-floor=0.4' in item['command']
        assert '--library-path=/tmp/library.json' in item['command']
        assert item['seed'] == item['repeat'] - 1
