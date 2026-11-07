"""The M1-only control uses Base generation and the shared M1 lifecycle."""

import json
from argparse import Namespace

import pytest

from tests.v22.test_search import Replies, assert_same_selectors, task
from traceaad_base import Config
from traceaad_base.llm import CANDIDATE_RESPONSE_FORMAT
from traceaad_base.models import Attempt
from traceaad_base.search import TraceAAD as Base
from traceaad_v22 import TraceAAD as V22
from traceaad_v23 import TraceAAD


def test_shared_allocation_produces_identical_actions_and_learning(tmp_path):
    config = Config(evaluation_budget=9, initial_size=2, seed=7)
    v22 = V22(task(), Replies(), config=config, run_dir=tmp_path / 'v22')
    v23 = TraceAAD(task(), Replies(), config=config, run_dir=tmp_path / 'v23')
    v22.run()
    v23.run()
    for a, b in zip(v22.attempts, v23.attempts, strict=True):
        assert (a.parent_id, a.operator, a.reference_id) == (b.parent_id, b.operator, b.reference_id)
        assert a.record.get('selection') == b.record.get('selection')
    assert_same_selectors(v22.selector, v23.selector)
    assert v22.selector is not v23.selector
    assert all(v22.selector.models[o] is not v23.selector.models[o] for o in v22.selector.models)
    assert v22.model.prompts != v23.model.prompts


@pytest.mark.parametrize('operator', ['Refine', 'Explore', 'Crossover'])
def test_prompt_is_exactly_base_with_selected_reference(tmp_path, monkeypatch, operator):
    search = TraceAAD(task(), Replies(), config=Config(evaluation_budget=8, initial_size=2), run_dir=tmp_path)
    search.step()
    search.step()
    parent, reference = search.archive[1], search.archive[2] if operator == 'Crossover' else None
    monkeypatch.setattr(search, 'reference', lambda *_: pytest.fail('Reference resampled'))
    attempt = Attempt(3, parent.id, operator, reference.id if reference else None)
    assert search.evolution_prompt(attempt, parent, reference) == Base.evolution_prompt(search, attempt, parent, reference)
    assert search.prompt_type is Base.prompt_type
    prompt = search.evolution_prompt(attempt, parent, reference)
    assert prompt.index('- "analysis"') < prompt.index('- "design"') < prompt.index('- "code"')
    assert 'Score:' in prompt and '```python' in prompt
    assert 'context' not in attempt.record


def test_initialization_and_both_repairs_are_exactly_base():
    search, base = TraceAAD(task(), Replies()), Base(task(), Replies())
    assert search.prompts.initialization(search.archive) == base.prompts.initialization(base.archive)
    assert search.prompts.repair_format('{}', 'missing') == base.prompts.repair_format('{}', 'missing')
    from traceaad_base import Evaluation, Program
    failed = Program(1, 'def solve(x):\n    raise ValueError("bad")', 'design', 'Explore', 0, None,
                     Evaluation('error', failure='ValueError: bad'))
    assert search.prompts.repair_program(failed) == base.prompts.repair_program(failed)


@pytest.mark.parametrize('values', [[100., 110., 'error', 105., 106., 107.],
                                  [100., 100., 100., 105., 103., 102.],
                                  [100., 110., 1e6, -1e6, 110., 110.]])
def test_restore_and_budget_with_repair_ties_and_extremes(tmp_path, values):
    config = Config(evaluation_budget=6, initial_size=2, seed=4)
    search = TraceAAD(task(), Replies(values=values), config=config, run_dir=tmp_path)
    for _ in range(3):
        search.step()
    restored = TraceAAD(task(), Replies(start=search.model_calls, values=values), config=config,
                        run_dir=tmp_path, resume=True)
    assert_same_selectors(search.selector, restored.selector)
    search.run()
    restored.run()
    assert search.evaluations_used == restored.evaluations_used == 6
    assert [a.record for a in search.attempts] == [a.record for a in restored.attempts]
    assert_same_selectors(search.selector, restored.selector)


def test_cross_version_resume_rejected_before_log_repair(tmp_path):
    config = Config(evaluation_budget=3, initial_size=2)
    search = V22(task(), Replies(), config=config, run_dir=tmp_path)
    search.step()
    calls = tmp_path / 'calls.jsonl'
    calls.write_bytes(calls.read_bytes() + b'{"unfinished":')
    before = calls.read_bytes()
    with pytest.raises(ValueError, match='V23 resume settings: method'):
        TraceAAD(task(), Replies(), config=config, run_dir=tmp_path, resume=True)
    assert calls.read_bytes() == before


def test_registry_and_launcher_use_base_schema():
    from experiments.traceaad_v20.run import METHODS, request_options
    from experiments.traceaad_v20.launch_host import build_plan
    assert METHODS['v23'] == ('traceaad_v23', __import__('traceaad_v23'))
    assert request_options('v23') == request_options('base')
    assert request_options('v23')['response_format'] == CANDIDATE_RESPONSE_FORMAT
    args = Namespace(method='v23', batch='test', experiment='traceaad_v23', repeats=3,
                     backend='scheduled', budget=1000, initial_size=5, max_tokens=16384,
                     model_timeout=600, scheduler_socket='/fake', seconds_per_instance=10.,
                     env_file=None, skip_heldout=False)
    plan = build_plan(args)
    assert len(plan) == 18
    assert {row['seed'] for row in plan} == {0, 1, 2}
    assert all(row['command'][row['command'].index('--method') + 1] == 'v23' for row in plan)


def test_real_experiment_entry_publishes_base_protocol_and_m1(tmp_path, monkeypatch, capsys):
    from experiments.traceaad_v20 import run
    from traceaad_base import Evaluator
    class Scheduled(Replies):
        model = 'test-model'
        def __init__(self, path, label):
            super().__init__()
            self.path = path
            self.scheduler = {'endpoints': []}
    def load_task(**options):
        result = task()
        result.metadata = {'n_instance': 16, 'settings': {}}
        return result
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'load_environment', lambda *_: None)
    monkeypatch.setattr(run, 'load_task', load_task)
    monkeypatch.setattr(run, 'ScheduledModel', Scheduled)
    monkeypatch.setattr(run, 'ScheduledEvaluator', lambda task, *_: Evaluator(task))
    run.main(['--method', 'v23', '--task', 'tsp_construct', '--run-name', 'test', '--batch', 'test',
              '--experiment', 'traceaad_v23', '--budget', '6', '--initial-size', '2',
              '--scheduler-socket', '/fake', '--skip-heldout'])
    directory = tmp_path / 'experiments_result/traceaad_v23/tsp_construct/test'
    summary = json.loads((directory / 'summary.json').read_text())
    config = json.loads((directory / 'run_config.json').read_text())
    native = json.loads((directory / 'native/run_config.json').read_text())
    assert summary['method'] == config['method'] == native['method'] == 'traceaad_v23'
    assert summary['evaluation_calls'] == 6 and summary['status'] == 'finished'
    assert config['llm']['sampling']['response_format'] == CANDIDATE_RESPONSE_FORMAT
    assert native['selection_settings']['residual_clip'] == 3.
    records = [json.loads(line)['attempt']['record']
               for line in (directory / 'native/attempts.jsonl').read_text().splitlines()]
    assert all('selection' in r and 'context' not in r for r in records[2:])
    assert all('node' in r for r in records)
    assert (directory / 'native/calls.jsonl.gz').exists()
    capsys.readouterr()
