import json

import pytest

from experiments.diagnosis_reasoning_effort.run import evaluate_reply
from traceaad.common.instance_evaluation import InstanceProgramEvaluator


@pytest.mark.parametrize('task,cpu_limit,wall_limit', [('tsp_construct', 2, 20), ('graph_colouring', 10, 30)])
@pytest.mark.parametrize('thought', [None, False])
def test_replay_evaluation_uses_current_task_protocol(tmp_path, monkeypatch, task, cpu_limit, wall_limit, thought):
    directory = tmp_path / 'low' / 'case'
    directory.mkdir(parents=True)
    (directory / 'candidate.py').write_text('def solve(x):\n    return x\n')

    def evaluate(self, code, source_key):
        assert self.function_seconds == cpu_limit
        assert self.timeout_seconds == wall_limit
        assert self.n_instance == 16 and self.n_workers == 8
        assert self.seeds == (730241,) and self.scheduler_socket == '/tmp/test.sock'
        return {'fitness': 12.5, 'failure': None, 'seconds': 3.0}

    monkeypatch.setattr(InstanceProgramEvaluator, 'evaluate', evaluate)
    row = {'mode': 'low', 'code_parse_success': True, 'thought_parse_success': thought, 'seconds': 9.0}
    result = evaluate_reply({'id': 'case', 'task': task}, row, tmp_path, workers=8, scheduler_socket='/tmp/test.sock')
    assert result['evaluation_success'] and result['fitness'] == 12.5
    assert result['seconds'] == 9.0 and result['evaluation_seconds'] == 3.0
    assert result['method_delivery_success'] == (thought is not False)
    assert json.loads((directory / 'evaluation.json').read_text())['fitness'] == 12.5


def test_missing_code_is_not_sent_to_evaluation(tmp_path):
    directory = tmp_path / 'high' / 'case'
    directory.mkdir(parents=True)
    row = {'mode': 'high', 'code_parse_success': False, 'thought_parse_success': True}
    result = evaluate_reply({'id': 'case', 'task': 'tsp_construct'}, row, tmp_path, workers=8, scheduler_socket=None)
    assert result['fitness'] is None
    assert result['evaluation_failure']['kind'] == 'delivery_failure'
