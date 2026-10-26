import json
from types import SimpleNamespace

import pytest

from experiments.transfer import loop
from traceaad.storage import read_json


def test_reply_only_unwraps_one_complete_json_fence():
    assert loop.reply_object({'content': '```json\n{"x":1}\n```', 'finish_reason': 'stop'}) == {'x': 1}
    for content, finish in [('prefix {"x":1}', 'stop'), ('{}', 'length'), ('[]', 'stop')]:
        with pytest.raises(ValueError):
            loop.reply_object({'content': content, 'finish_reason': finish})


def test_bank_feedback_excludes_evaluation_internals():
    bank = [dict(id='t1', attempts=0, improvements=0, recent_outcomes=[])]
    parent = SimpleNamespace(id=1, fitness=10)
    child = SimpleNamespace(valid=False, fitness=None, failure=dict(
        kind='timeout', error='mean function CPU time exceeded 2s across 16 instances',
        function_seconds_limit=2, calls=768))
    loop.update_bank(bank, 't1', parent, child)
    assert bank[0]['recent_outcomes'][0]['error'] == 'timeout'
    prompt = loop.bank_prompt('context\n[Your Task: Refine]', bank)
    assert all(text not in prompt for text in ('16 instances', 'function_seconds_limit', '768'))


def test_online_bank_budget_feedback_and_exact_replay(tmp_path, monkeypatch):
    task = SimpleNamespace(task_description='Minimize objective.', template_program='def f(x):\n    return x\n')
    roots = {'parents': [dict(tier=2, code=f'def f(x):\n    return {v}\n', idea='root', fitness=v)
                         for v in (10, 11)]}
    class FakeLLM:
        model = 'fake'
        draws = 0
        def count_prompt_tokens(self, prompt):
            return len(prompt)
        def draw_sample_with_details(self, prompt, **kwargs):
            self.draws += 1
            if prompt.startswith('Extract one reusable'):
                value = dict.fromkeys(('name', 'applicability', 'operation', 'preserve', 'parameters', 'checks'), 'change')
            else:
                value = dict(analysis='change', design='lower', code=f'def f(x):\n    return {9-self.draws}\n')
                if '[Available Transformations]' in prompt:
                    value.update(transformation_id='t3', bindings='f', checks='return')
            return dict(content=json.dumps(value), finish_reason='stop', usage={'total_tokens': 10})
        def close(self):
            pass
    llm = FakeLLM()
    class FakeEvaluator:
        def __init__(self, role):
            self.role = role
        def evaluate(self, code, name):
            if self.role == 'heldout':
                assert (tmp_path / 'selection.json').exists()
            return dict(fitness=float(code.strip().split()[-1]), failure=None)
    monkeypatch.setattr(loop, 'client', lambda _: llm)
    monkeypatch.setattr(loop, 'training_task', lambda _: (task, {}))
    monkeypatch.setattr(loop, 'heldout_task', lambda *_: task)
    monkeypatch.setattr(loop, 'evaluator', lambda task, role: FakeEvaluator(role))
    # request() disables retry in the isolated driver; restore it after this test.
    monkeypatch.setattr(loop.llm_api, 'SERVICE_WAIT_SECONDS', loop.llm_api.SERVICE_WAIT_SECONDS)
    result = loop.run(tmp_path, roots, 'bank', 0, 'qwen', 6)
    state = read_json(tmp_path / 'state.json')
    assert result['calls'] == llm.draws == 6
    assert result['generations'] == 4
    assert result['bank_size'] == 2
    assert state['bank'][0]['attempts'] == state['bank'][0]['improvements'] == 2
    assert [row['best'] for row in state['curve']] == sorted([row['best'] for row in state['curve']], reverse=True)
    assert loop.run(tmp_path, roots, 'bank', 0, 'qwen', 6) == result
    assert llm.draws == 6
    assert read_json(tmp_path / 'state.json') == state


def test_real_tsp_task_reaches_heldout_without_model_calls(tmp_path, monkeypatch):
    code = '''import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
'''
    class FakeLLM:
        model = 'fake'
        def count_prompt_tokens(self, prompt):
            return len(prompt)
        def draw_sample_with_details(self, prompt, **kwargs):
            return dict(content=json.dumps(dict(analysis='nearest', design='nearest', code=code)),
                        finish_reason='stop', usage={'total_tokens': 10})
        def close(self):
            pass
    monkeypatch.setattr(loop, 'client', lambda _: FakeLLM())
    monkeypatch.setattr(loop.llm_api, 'SERVICE_WAIT_SECONDS', loop.llm_api.SERVICE_WAIT_SECONDS)
    monkeypatch.delenv('TRACEAAD_SCHEDULER_SOCKET', raising=False)
    roots = {'parents': [dict(tier=2, code=code, idea='nearest', fitness=10)] * 2}
    result = loop.run(tmp_path, roots, 'open', 0, 'qwen', 1)
    assert result['heldout'] > 0
    assert result['calls'] == 1
    assert read_json(tmp_path / 'evaluations/winner_heldout.json')['failure'] is None
