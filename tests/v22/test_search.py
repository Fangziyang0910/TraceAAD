"""V22's actual attempt, budget, prompt and recovery boundaries."""

import json
from dataclasses import replace

import numpy as np
import pytest

from traceaad_base import Config, Evaluation, Task
from traceaad_v21.context import context
from traceaad_v21.prompts import CANDIDATE_RESPONSE_FORMAT
from traceaad_v22 import TraceAAD
from experiments.traceaad_v20.run import METHODS, request_options


class Replies:
    request_options = {}

    def __init__(self, start=0, values=None):
        self.index = start
        self.values = values
        self.prompts = []

    def complete(self, prompt, **options):
        self.prompts.append(prompt)
        i = self.index
        self.index += 1
        value = self.values[i] if self.values is not None else 100. - i
        if value == 'format':
            return '{}'
        code = 'def solve(x):\n    raise ValueError("bad")\n' if value == 'error' else f'def solve(x):\n    return {value}\n'
        return json.dumps({'analysis': f'change {i}', 'code': code, 'design': f'design {i}'})


def task():
    return Task('Return a finite score.', 'def solve(x):\n    return 1\n',
                lambda source, fn: float(fn(0)), time_limit=5., entrypoint='solve')


def assert_same_selectors(a, b):
    assert a.nodes == b.nodes and a.sources == b.sources and a.processed == b.processed
    assert a.f0 == b.f0 and a.s0 == b.s0 and a.champion == b.champion
    for op, model in a.models.items():
        other = b.models[op]
        assert (model.N, model.U, model.A, model.B) == (other.N, other.U, other.A, other.B)
        np.testing.assert_array_equal(model.m, other.m)
        np.testing.assert_array_equal(model.V, other.V)
        assert model.groups.keys() == other.groups.keys()
        for key, group in model.groups.items():
            saved = other.groups[key]
            assert (group.n, group.u, group.k, group.sy) == (saved.n, saved.u, saved.k, saved.sy)
            np.testing.assert_array_equal(group.sx, saved.sx)


def test_resumed_search_matches_uninterrupted_actions_and_posteriors(tmp_path):
    config = Config(evaluation_budget=9, initial_size=2, seed=7)
    full = TraceAAD(task(), Replies(), config=config, run_dir=tmp_path / 'full')
    result = full.run()
    split = TraceAAD(task(), Replies(), config=config, run_dir=tmp_path / 'split')
    for _ in range(5):
        split.step()
    resumed = TraceAAD(task(), Replies(start=split.model_calls), config=config,
                       run_dir=tmp_path / 'split', resume=True)
    assert_same_selectors(split.selector, resumed.selector)
    resumed.run()
    assert result.evaluations == resumed.evaluations_used == 9
    assert [a.record for a in full.attempts] == [a.record for a in resumed.attempts]
    assert_same_selectors(full.selector, resumed.selector)
    assert full.model.prompts[5:] == resumed.model.prompts
    assert full.best.code == resumed.best.code
    assert set(full.model.request_options['response_format']['json_schema']['schema']['properties']) == {'analysis', 'code', 'design'}


@pytest.mark.parametrize('budget,expected_gate,expected_usable', [(3, 1, 0), (4, 1, 1)])
def test_repair_is_one_learning_event_and_respects_last_evaluation(tmp_path, budget, expected_gate, expected_usable):
    model = Replies(values=[100., 110., 'error', 105.])
    search = TraceAAD(task(), model, config=Config(evaluation_budget=budget, initial_size=2), run_dir=tmp_path)
    search.run()
    last = search.attempts[-1]
    assert search.evaluations_used == budget
    assert sum(m.N for m in search.selector.models.values()) == expected_gate
    assert sum(m.U for m in search.selector.models.values()) == expected_usable
    assert sum(g.k for m in search.selector.models.values() for g in m.groups.values()) == expected_usable
    assert last.repair_used == (budget == 4)
    assert len(last.evaluations) == last.record['selection']['evaluation_calls'] == budget - 2
    assert len(model.prompts) == budget


def test_format_repair_does_not_consume_an_extra_evaluation(tmp_path):
    search = TraceAAD(task(), Replies(values=[100., 110., 'format', 105.]),
                      config=Config(evaluation_budget=3, initial_size=2), run_dir=tmp_path)
    search.run()
    last = search.attempts[-1]
    assert search.evaluations_used == 3 and search.model_calls == 4
    assert len(last.evaluations) == 1 and last.repair_used
    assert sum(m.N for m in search.selector.models.values()) == 1
    assert sum(m.U for m in search.selector.models.values()) == 1


def test_duplicate_final_source_only_updates_gate_and_resume_is_idempotent(tmp_path):
    config = Config(evaluation_budget=5, initial_size=2)
    search = TraceAAD(task(), Replies(values=[100., 110., 100., 100., 100.]), config=config, run_dir=tmp_path)
    search.run()
    assert len(search.selector.nodes) == 2
    assert sum(m.N for m in search.selector.models.values()) == 3
    assert sum(m.U for m in search.selector.models.values()) == 0
    restored = TraceAAD(task(), Replies(), config=config, run_dir=tmp_path, resume=True)
    assert_same_selectors(search.selector, restored.selector)
    restored.restore(restored.attempts[-1])
    assert_same_selectors(search.selector, restored.selector)


def test_tied_scale_freeze_restores_and_continues_identically(tmp_path):
    config = Config(evaluation_budget=6, initial_size=2, seed=4)
    values = [100., 100., 105., 103., 102., 101.]
    search = TraceAAD(task(), Replies(values=values), config=config, run_dir=tmp_path)
    for _ in range(3):
        search.step()
    assert search.selector.s0 == 5.
    restored = TraceAAD(task(), Replies(start=3, values=values), config=config, run_dir=tmp_path, resume=True)
    assert_same_selectors(search.selector, restored.selector)
    search.run()
    restored.run()
    assert_same_selectors(search.selector, restored.selector)
    assert [a.record for a in search.attempts] == [a.record for a in restored.attempts]


def test_v21_context_uses_selected_reference_without_resampling(tmp_path, monkeypatch):
    search = TraceAAD(task(), Replies(), config=Config(evaluation_budget=8, initial_size=2), run_dir=tmp_path)
    search.step()
    search.step()
    monkeypatch.setattr(search, 'reference', lambda *_: pytest.fail('Reference was resampled'))
    parent, operator, reference, record = search.select(3)
    from traceaad_base.models import Attempt
    attempt = Attempt(3, parent.id, operator, reference.id if reference else None, record=record)
    actual = search.evolution_prompt(attempt, parent, reference)
    text, instruction, view = context(search.archive, parent, operator, reference, search.config)
    assert actual == search.prompts._render(text, instruction, has_scores=operator != 'Explore')
    assert attempt.record['context'] == view
    assert attempt.reference_id == record['selection']['reference_id']


def test_evaluator_is_guarded_at_actual_call_site():
    search = TraceAAD(task(), Replies(), config=Config(evaluation_budget=1, initial_size=1))
    search.step()
    from traceaad_base.models import Candidate
    with pytest.raises(RuntimeError, match='budget exhausted'):
        search._evaluate(search.attempts[-1], Candidate('a', 'd', 'def solve(x):\n    return 0'))


def test_v22_registration_and_output_protocol():
    assert METHODS['v22'][0] == 'traceaad_v22'
    assert request_options('v22') == request_options('v21')
    assert request_options('v22')['response_format'] == CANDIDATE_RESPONSE_FORMAT


def test_no_result_attempts_update_gate_and_keep_existing_stop_rule(tmp_path):
    class Failing(Replies):
        def complete(self, prompt, **options):
            if self.index >= 2:
                raise ConnectionError('No reply')
            return super().complete(prompt, **options)
    config = Config(evaluation_budget=8, initial_size=2, max_unevaluated=2)
    search = TraceAAD(task(), Failing(), config=config, run_dir=tmp_path)
    result = search.run()
    assert result.stop_reason == 'consecutive_unevaluated_limit'
    assert result.evaluations == 2 and result.attempts == 4
    assert sum(m.N for m in search.selector.models.values()) == 2
    assert sum(m.U for m in search.selector.models.values()) == 0
    restored = TraceAAD(task(), Failing(), config=config, run_dir=tmp_path, resume=True)
    assert_same_selectors(search.selector, restored.selector)


def test_resume_before_scale_exists_uses_same_uniform_random_stream(tmp_path):
    config = Config(evaluation_budget=6, initial_size=2, seed=9)
    values = [100., 100., 100., 100., 105., 103.]
    search = TraceAAD(task(), Replies(values=values), config=config, run_dir=tmp_path)
    for _ in range(3):
        search.step()
    assert search.selector.s0 is None
    restored = TraceAAD(task(), Replies(start=3, values=values), config=config, run_dir=tmp_path, resume=True)
    search.run()
    restored.run()
    assert_same_selectors(search.selector, restored.selector)
    assert [a.record for a in search.attempts] == [a.record for a in restored.attempts]


def test_experiment_entry_runs_and_publishes_v22_without_external_services(tmp_path, monkeypatch, capsys):
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
    run.main(['--method', 'v22', '--task', 'tsp_construct', '--run-name', 'test', '--batch', 'test',
              '--experiment', 'traceaad_v22', '--budget', '6', '--initial-size', '2',
              '--scheduler-socket', '/fake', '--skip-heldout'])
    directory = tmp_path / 'experiments_result/traceaad_v22/tsp_construct/test'
    summary = json.loads((directory / 'summary.json').read_text())
    config = json.loads((directory / 'run_config.json').read_text())
    assert summary['method'] == config['method'] == 'traceaad_v22'
    assert summary['evaluation_calls'] == 6 and summary['status'] == 'finished'
    assert config['llm']['sampling']['response_format'] == CANDIDATE_RESPONSE_FORMAT
    assert (directory / 'native/calls.jsonl.gz').exists()
    assert (directory / 'best_program.py').exists()
    records = [json.loads(line)['attempt']['record']
               for line in (directory / 'native/attempts.jsonl').read_text().splitlines()]
    assert all('selection' in r and 'context' in r for r in records[2:])
    assert all('node' in r for r in records)
    capsys.readouterr()


def test_launcher_builds_v22_plan_without_launching():
    from argparse import Namespace
    from experiments.traceaad_v20.launch_host import build_plan
    args = Namespace(method='v22', batch='test', experiment='traceaad_v22', repeats=3,
                     backend='scheduled', budget=1000, initial_size=5, max_tokens=16384,
                     model_timeout=600, scheduler_socket='/fake', seconds_per_instance=10.,
                     env_file=None, skip_heldout=False)
    plan = build_plan(args)
    assert len(plan) == 18
    assert all(row['command'][row['command'].index('--method') + 1] == 'v22' for row in plan)
    assert all('traceaad_v22' in row['run_dir'] for row in plan)


@pytest.mark.parametrize('field,value', [('initial_size', 3), ('seed', 8), ('evaluation_budget', 10),
                                         ('ancestor_count', 4)])
def test_resume_rejects_changed_config_before_touching_logs(tmp_path, field, value):
    config = Config(evaluation_budget=6, initial_size=2, seed=7)
    search = TraceAAD(task(), Replies(), config=config, run_dir=tmp_path)
    search.step()
    search.step()
    calls = tmp_path / 'calls.jsonl'
    calls.write_bytes(calls.read_bytes() + b'{"unfinished":')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    model = Replies()
    with pytest.raises(ValueError, match=f'config.{field}'):
        TraceAAD(task(), model, config=replace(config, **{field: value}), run_dir=tmp_path, resume=True)
    assert not model.prompts
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


@pytest.mark.parametrize('field,value', [('update_rule', 'conjugate_v1'), ('residual_clip', 4.),
                                         ('scale_rule', 'other_scale')])
def test_resume_rejects_changed_update_contract(tmp_path, field, value):
    config = Config(evaluation_budget=6, initial_size=2)
    search = TraceAAD(task(), Replies(), config=config, run_dir=tmp_path)
    search.step()
    path = tmp_path / 'run_config.json'
    saved = json.loads(path.read_text())
    saved['selection_settings'][field] = value
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match=f'selection_settings.{field}'):
        TraceAAD(task(), Replies(), config=config, run_dir=tmp_path, resume=True)


def test_old_update_checkpoint_is_not_silently_replayed(tmp_path):
    config = Config(evaluation_budget=6, initial_size=2)
    search = TraceAAD(task(), Replies(), config=config, run_dir=tmp_path)
    search.step()
    path = tmp_path / 'run_config.json'
    saved = json.loads(path.read_text())
    saved.pop('selection_settings')
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='selection_settings'):
        TraceAAD(task(), Replies(), config=config, run_dir=tmp_path, resume=True)


def test_resume_matches_threshold_and_preserves_extreme_true_results(tmp_path):
    config = Config(evaluation_budget=6, initial_size=2)
    values = [100., 110., 1e6, -1e6, 102., 105.]
    search = TraceAAD(task(), Replies(values=values), config=config, residual_clip=2., run_dir=tmp_path)
    for _ in range(4):
        search.step()
    assert search.best.evaluation.fitness == -1e6
    assert search.selector.nodes[3].fitness == 1e6
    assert search.selector.nodes[4].fitness == -1e6
    with pytest.raises(ValueError, match='selection_settings.residual_clip'):
        TraceAAD(task(), Replies(), config=config, run_dir=tmp_path, resume=True)
    resumed = TraceAAD(task(), Replies(start=4, values=values), config=config, residual_clip=2.,
                       run_dir=tmp_path, resume=True)
    assert_same_selectors(search.selector, resumed.selector)
    search.run()
    resumed.run()
    assert_same_selectors(search.selector, resumed.selector)
    assert [a.record for a in search.attempts] == [a.record for a in resumed.attempts]
    resumed.restore(resumed.attempts[0])
    assert_same_selectors(search.selector, resumed.selector)


@pytest.mark.parametrize('field', ['f0', 's0'])
def test_resume_checks_scale_after_each_attempt(tmp_path, field):
    config = Config(evaluation_budget=6, initial_size=2)
    search = TraceAAD(task(), Replies(), config=config, run_dir=tmp_path)
    search.step()
    search.step()
    path = tmp_path / 'attempts.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]['attempt']['record']['selection_scale'][field] += 1.
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='restored scale at attempt 2'):
        TraceAAD(task(), Replies(), config=config, run_dir=tmp_path, resume=True)


@pytest.mark.parametrize('field', ['f0', 's0'])
def test_resume_checks_decision_scale_before_training(tmp_path, field):
    config = Config(evaluation_budget=6, initial_size=2)
    search = TraceAAD(task(), Replies(), config=config, run_dir=tmp_path)
    for _ in range(3):
        search.step()
    path = tmp_path / 'attempts.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]['attempt']['record']['selection'][field] += 1.
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match=f'selection scale: {field}'):
        TraceAAD(task(), Replies(), config=config, run_dir=tmp_path, resume=True)


def test_resume_rejects_changed_task_and_unknown_settings(tmp_path):
    config = Config(evaluation_budget=6, initial_size=2)
    search = TraceAAD(task(), Replies(), config=config, run_dir=tmp_path)
    search.step()
    changed = task()
    changed.time_limit += 1.
    with pytest.raises(ValueError, match='task.time_limit'):
        TraceAAD(changed, Replies(), config=config, run_dir=tmp_path, resume=True)
    path = tmp_path / 'run_config.json'
    saved = json.loads(path.read_text())
    saved['selection_settings']['unsupported_policy'] = True
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='selection_settings.unsupported_policy'):
        TraceAAD(task(), Replies(), config=config, run_dir=tmp_path, resume=True)
