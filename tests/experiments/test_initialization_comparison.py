import json
import pytest
from tests.method.test_traceaad_v1013 import TinyEvaluation, FakeLLM, response
from experiments.traceaad_initialization.run import InitializationExperiment
from experiments.traceaad_initialization.launch import jobs, validate, check_resume_config

@pytest.mark.parametrize('mode,counts', [('independent',[0]*8),('sequential',list(range(8))),('hybrid',[0,0,0,0,4,5,6,7])])
def test_information_dependency_and_stop(tmp_path, mode, counts):
    llm=FakeLLM(*(response(i) for i in range(8)))
    method=InitializationExperiment(init_mode=mode,init_only=True,evaluation=TinyEvaluation(),llm=llm,run_dir=tmp_path,budget=24)
    method.run()
    records=[json.loads(s) for s in (tmp_path/'init_context.jsonl').read_text().splitlines()]
    assert [len(r['init_context_ids']) for r in records] == counts
    assert method.evaluations_used == 8
    assert len(method.tree.roots) == 8
    assert all(n.parent_id is None for n in method.tree.all_nodes())

def test_invalid_generation_does_not_advance_hybrid_boundary(tmp_path):
    llm=FakeLLM('bad output',*(response(i) for i in range(8)))
    method=InitializationExperiment(init_mode='hybrid',init_only=True,evaluation=TinyEvaluation(),llm=llm,run_dir=tmp_path,budget=24)
    method.run()
    records=[json.loads(s) for s in (tmp_path/'init_context.jsonl').read_text().splitlines()]
    assert [r['root_ordinal'] for r in records][:3] == [1,1,2]
    assert [len(r['init_context_ids']) for r in records] == [0,0,0,0,0,4,5,6,7]
    assert method.evaluations_used == 8


def test_initialization_prompt_conditions(tmp_path):
    from traceaad.v10_13.prompts import PromptBuilder
    from traceaad.v10_13.tree import Node

    class LLM:
        def count_prompt_tokens(self, text):
            return len(text.split())

    nodes = [Node(0, "def score(x):\n    return x", "first", 1.0)]
    builder = PromptBuilder(LLM(), "# Task\nReturn a score.", max_tokens=10000,
                            history_depth=3, lookup=lambda i: nodes[i],
                            all_nodes=lambda: [])
    independent = builder.build_initial("independent").prompt
    informed_builder = PromptBuilder(LLM(), "# Task\nReturn a score.", max_tokens=10000,
                                     history_depth=3, lookup=lambda i: nodes[i],
                                     all_nodes=lambda: nodes)
    informed = informed_builder.build_initial("informed").prompt
    assert "# Initialization Task" in independent
    assert "independent initialization sample" in independent
    assert "Previous Initial Algorithm" not in independent
    assert "# Previous Initial Algorithm" in informed
    assert "meaningful complement" in informed


def test_evaluation_budget_tracks_invalid_output_and_keeps_development(tmp_path):
    llm = FakeLLM('bad output', *(response(i) for i in range(9)))
    method = InitializationExperiment(init_mode='sequential', evaluation=TinyEvaluation(),
                                      llm=llm, run_dir=tmp_path, budget=9)
    method.run()
    summary = json.loads(method.storage.summary_path.read_text())
    assert summary['status'] == 'finished'
    assert summary['budget_basis'] == 'evaluator_calls'
    assert summary['candidate_attempts'] == 10
    assert summary['evaluation_calls'] == 9
    assert summary['num_roots'] == 8
    assert summary['num_nodes'] == 9
    assert len((tmp_path / 'init_context.jsonl').read_text().splitlines()) == 9


def test_evaluation_budget_keeps_incomplete_initialization(tmp_path):
    method = InitializationExperiment(init_mode='hybrid', evaluation=TinyEvaluation(),
                                      llm=FakeLLM(*(response('1 / 0') for _ in range(8))),
                                      run_dir=tmp_path, budget=8)
    method.run()
    summary = json.loads(method.storage.summary_path.read_text())
    assert summary['status'] == 'incomplete_initialization'
    assert summary['candidate_attempts'] == 8
    assert summary['evaluation_calls'] == 8
    assert summary['initialization_complete'] is False


def test_model_call_cap_is_terminal_generation_failure(tmp_path):
    method = InitializationExperiment(init_mode='independent', evaluation=TinyEvaluation(),
                                      llm=FakeLLM(*(['bad output'] * 4)),
                                      run_dir=tmp_path, budget=8, max_calls=4)
    method.run()
    summary = json.loads(method.storage.summary_path.read_text())
    assert summary['status'] == 'call_cap'
    assert summary['candidate_attempts'] == 4
    assert summary['evaluation_calls'] == 0


def test_formal_schedule_and_resume_config(tmp_path, monkeypatch):
    rows = jobs()
    validate(rows)
    assert len(rows) == 60
    assert {r['backend'] for r in rows} == {'server3', 'server3b'}
    assert sum(r['backend'] == 'server3' for r in rows
               if r['task'] == 'tsp_construct' and r['mode'] == 'hybrid') == 2
    row = rows[0]
    run = tmp_path / row['run_name']
    run.mkdir()
    (run / 'tree_state.json').write_text('{}')
    (run / 'run_config.json').write_text(json.dumps({
        'task': row['task'], 'seed': row['seed'], 'repeat': row['repeat'],
        'backend': row['backend'], 'method_params': {
            'budget': 8, 'n_roots': 8, 'init_mode': row['mode'],
            'init_only': False, 'output_tokens': 8192,
        },
    }))
    monkeypatch.setattr('experiments.traceaad_initialization.launch.run_dir', lambda _: run)
    with pytest.raises(RuntimeError, match='budget'):
        check_resume_config(row)


def test_heldout_uses_frozen_training_best(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments.traceaad_initialization import heldout

    job = jobs()[0]
    run = tmp_path / job['run_name']
    (run / 'logs').mkdir(parents=True)
    (run / 'logs' / 'run_summary.json').write_text(json.dumps({
        'status': 'finished',
        'best': {'id': 3, 'fitness': 2.0, 'code': 'def score(x):\n    return 2'},
    }))
    monkeypatch.setattr(heldout, 'run_dir', lambda _: run)
    monkeypatch.setattr(heldout, 'make_evaluator', lambda task: (object(), 'eval'))

    class FakeEvaluator:
        def __init__(self, evaluator):
            pass

        def evaluate_program_with_details(self, code):
            assert code == 'def score(x):\n    return 2'
            return SimpleNamespace(result=1.5, failure_kind=None, error=None)

    monkeypatch.setattr(heldout, 'SecureEvaluator', FakeEvaluator)
    assert heldout.evaluate_one(job, {}) == 'evaluated'
    assert heldout.evaluate_one(job, {}) == 'existing'
    result = json.loads((run / 'heldout.json').read_text())
    assert result['node_id'] == 3
    assert result['heldout_fitness'] == 1.5
