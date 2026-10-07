"""Training fixes the winner, even when held-out disagrees or fails."""

import json

import pytest

from experiments.infra import evaluate
from tests.support import TinyEvaluation, TokenLLM, response
from traceaad.v10_20 import Config, TraceAADV1020


@pytest.mark.parametrize('fail_best', (False, True))
def test_heldout_does_not_replace_the_training_winner(tmp_path, monkeypatch, fail_best):
    method = TraceAADV1020(evaluation=TinyEvaluation(), llm=TokenLLM(response(2), response(5), response(1)),
                           run_dir=tmp_path, config=Config(budget=3, roots=3))
    summary = method.run()
    assert summary['status'] == 'finished'
    assert summary['best']['fitness'] == -5
    assert summary['final_selection'] == 'training' and summary['selection_evaluations'] == 0
    choice = json.loads((tmp_path / 'selection.json').read_text())
    assert choice['criterion'] == 'training' and choice['selection_protocol'] is None
    assert choice['results'] == []
    frozen = (tmp_path / 'best_program.py').read_text()

    class OppositeTest(TinyEvaluation):
        def evaluate_program(self, source, function):
            value = function(1)
            if fail_best and value == 5:
                raise ValueError('the frozen training winner fails on held-out')
            return value  # The smaller training competitors would score better here.

    (tmp_path / 'run_config.json').write_text(json.dumps({
        'task': 'tsp_construct', 'method_params': {'evaluation_seeds': [730241]}}))
    monkeypatch.setattr(evaluate, 'heldout_task', lambda *args: OppositeTest())
    result = evaluate.evaluate_run(tmp_path, ['eval_50'], condition='traceaad')[0]
    assert result['key'] == summary['best']['key']
    assert result['fitness'] == (None if fail_best else 5)
    assert (tmp_path / 'best_program.py').read_text() == frozen


def test_cli_defaults_to_training_selection(capsys):
    from experiments.traceaad_v10_20.run import main
    main(['--task', 'tsp_construct', '--dry-run'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['final_selection'] == 'training' and plan['selection'] is None
