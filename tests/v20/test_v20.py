"""Trajectory scoring: P·h, its online update and its place in the Base search."""
import json
import math
import random

import numpy as np
import pytest

from traceaad_base import Evaluation, Program, Task
import traceaad_base
import traceaad_v20
from traceaad_v20 import Config
from traceaad_v20.selection import (Selector, SIGNS, conversion, constrain, feature_names, features,
                                    laplace, path_best, sigmoid)


def p(i, score, parent=0):
    return Program(i, f'def solve(x):\n    return {score}', 'design', 'Refine', parent, None,
                   Evaluation('success', score))


def test_path_best_and_depth():
    archive = {1: p(1, 5), 2: p(2, 7, 1), 3: p(3, 4, 2), 0: p(0, -100)}
    assert path_best(archive, 3) == 4
    assert path_best(archive, 2) == 5
    assert path_best(archive, 2, 0) == 7
    for i in range(4, 13):
        archive[i] = p(i, 10, i - 1)
    assert path_best(archive, 12, 8) == 10
    assert path_best(archive, 11, 8) == 4


def test_h_is_one_at_zero_gap_and_decreases():
    selector = Selector()
    for value in ('R', 'X'):
        mean = selector.data['h_models'][value]['mean']
        assert conversion(0, mean) == 1
        values = [conversion(g, mean) for g in [.001, .1, 1, 10, 100]]
        assert all(a > b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize('value', ['R', 'X'])
def test_same_value_failures_have_no_fixed_prior_penalty(value):
    beta = Selector().data['models'][value]['mean']
    scores = [float(sigmoid(np.dot(beta, features(1, 1, 1, {value: k}, value)[0]))) for k in (0, 1, 3, 10)]
    assert all(a == scores[0] for a in scores)


@pytest.mark.parametrize('outcome,direction', [(1, 1), (0, -1)])
def test_newton_moves_toward_observation(outcome, direction):
    mean = np.array([-1.]); precision = np.array([[2.]])
    fitted, cov, check = laplace([[1.]] * 20, [outcome] * 20, mean, precision)
    assert (fitted[0] - mean[0]) * direction > 0
    assert cov[0, 0] < .5 and check['gradient_max'] < 1e-7


def test_observe_counts_failure_by_value_and_success_once():
    selector = Selector(); archive = {i: p(i, 10 + i) for i in range(1, 6)}
    parent, op, ref, decision, _ = selector.choose(archive, Config(), random.Random(4))
    selector.observe(decision, parent.id, None)
    assert selector.data['counts'][str(parent.id)][decision['value']] == 1
    assert decision['outcome']['exceeded'] is False
    parent, op, ref, decision, _ = selector.choose(archive, Config(), random.Random(5))
    archive[6] = p(6, 1, parent.id)
    selector.observe(decision, parent.id, archive[6])
    assert decision['outcome']['exceeded'] and decision['outcome']['breakthrough']
    assert selector.data['counts'][str(parent.id)]['s'] == 1


def test_single_parent_omits_crossover():
    selector = Selector(); archive = {1: p(1, -1)}
    for seed in range(20):
        _, op, ref, decision, _ = selector.choose(archive, Config(), random.Random(seed))
        assert op != 'Crossover' and ref is None and decision['selected']['g'] == 0


def test_h_learns_only_exceeded_positive_gap():
    s = Selector(); x, g, _ = features(1.1, 1.1, 1., {}, 'R')
    before = json.dumps(s.data['h_models'])
    s.learn('R', x, 0., True, True)
    s.learn('R', x, g, False, False)
    assert json.dumps(s.data['h_models']) == before
    old = conversion(g, s.data['h_models']['R']['mean'])
    s.learn('R', x, g, True, False)
    assert len(s.data['h_models']['R']['y']) == 1
    assert conversion(g, s.data['h_models']['R']['mean']) < old
    assert json.dumps(s.data['h_models']['X']) == json.dumps(json.loads(before)['X'])


def test_relative_features_and_order():
    x, g, v = features(1.03, 1.01, 1., {'R': 2, 'X': 1, 's': 3}, 'R')
    assert g == pytest.approx(.01) and v == pytest.approx(.02)
    assert x == pytest.approx([1, 1, math.log(2), 1, math.log(3), math.log(3), math.log(2), math.log(4)])
    assert features(1, 1, 1, {}, 'X', True)[0] == [1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert feature_names('X')[1] == 'crossover'


def test_sampled_coefficients_obey_mechanism_signs():
    s = Selector(); archive = {i: p(i, 10 + i) for i in range(1, 6)}
    for v in ('R', 'X'):
        names = feature_names(v)
        for name, sign in SIGNS.items():
            s.data['models'][v]['mean'][names.index(name)] = -3 * sign
        s.data['h_models'][v]['mean'][1] = 3
    for seed in range(40):
        decision = s.choose(archive, Config(), random.Random(seed))[3]
        names = feature_names(decision['value'])
        assert all(sign * decision['beta'][names.index(name)] >= 0 for name, sign in SIGNS.items())
        assert decision['h_parameters'][1] <= -.05
    beta = constrain(np.full(len(feature_names('R')), 5.), 'R')
    worse = features(40., 1., 1., {}, 'R')[0]; same = features(1., 1., 1., {}, 'R')[0]
    assert np.dot(beta, worse) <= np.dot(beta, same)


def test_prior():
    selector = Selector()
    for v in ('R', 'X'):
        m = selector.data['models'][v]; offset = 1 if v == 'X' else 0
        assert len(m['mean']) == 8 + offset
        assert m['mean'][0] == math.log(.1 / .9) and np.diag(m['covariance'])[5 + offset] == 1
        assert selector.data['h_models'][v]['mean'] == [0., -1.]


class Model:
    """Two failures, one format error, otherwise varied scores."""
    def __init__(self):
        self.n = 0
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.n += 1
        self.prompts.append(prompt)
        if self.n == 6:
            return 'bad json'
        code = f'def solve(x):\n    return {10 - self.n % 7}'
        if self.n in (1, 4):
            code = 'def solve(x):\n    raise ValueError("failure")'
        return json.dumps(dict(analysis='a', design='d', code=code))


def search(package, directory=None, resume=False, model=None, budget=10):
    task = Task('minimize', 'def solve(x):\n    return x', lambda source, fn: fn(1), time_limit=2)
    off = {} if package is traceaad_base else dict(trajectory_context=False, reference_selection=False,
                                                    experience_learning=False)
    s = package.TraceAAD(task, model or Model(), config=package.Config(evaluation_budget=budget, initial_size=2, **off),
                         run_dir=directory, resume=resume)

    def evaluate(code):
        if 'raise' in code:
            return Evaluation('error', failure='failure')
        return Evaluation('success', float(code.split('return ')[-1]))
    s.evaluator.evaluate = evaluate
    return s


def test_budget_accounting_matches_base():
    a = search(traceaad_base); b = search(traceaad_v20)
    ra, rb = a.run(), b.run()
    assert (ra.evaluations, ra.model_calls, ra.attempts) == (rb.evaluations, rb.model_calls, rb.attempts)
    assert [(len(t.calls), len(t.evaluations), t.repair_used) for t in a.attempts] == [
        (len(t.calls), len(t.evaluations), t.repair_used) for t in b.attempts]
    assert rb.evaluations == 10
    assert all('selection' in t.record for t in b.attempts if t.operator != 'Init')


def test_resume_rebuilds_posterior_and_continues_as_uninterrupted(tmp_path):
    full = search(traceaad_v20, budget=16)
    full.run()
    first = search(traceaad_v20, tmp_path, budget=16)
    for _ in range(7):
        first.step()
    rows = [json.loads(line) for line in (tmp_path / 'attempts.jsonl').read_text().splitlines()]
    assert len(rows) == 7
    model = Model(); model.n = first.model.n
    resumed = search(traceaad_v20, tmp_path, resume=True, model=model, budget=16)
    assert len(resumed.attempts) == 7
    resumed.run()
    assert json.dumps(resumed.selector.data) == json.dumps(full.selector.data)
    assert [(a.parent_id, a.operator, a.program_id) for a in resumed.attempts] == [
        (a.parent_id, a.operator, a.program_id) for a in full.attempts]
    candidates = [json.loads(line) for line in (tmp_path / 'decisions.jsonl').read_text().splitlines()]
    assert {row['attempt_id'] for row in candidates} == {a.id for a in resumed.attempts if 'selection' in a.record}


def test_base_and_v20_dry_run_share_conditions(monkeypatch, capsys):
    from experiments.traceaad_v20 import run
    task = Task('minimize', 'def solve(x): return x', lambda s, f: f(1), time_limit=10,
                metadata={'n_instance': 16, 'settings': {'timeout_seconds': 10}})
    monkeypatch.setattr(run, 'load_task', lambda **k: task)

    class Scheduled:
        model = 'Qwen3.8-27B'

        def __init__(self, path, label):
            self.path = path; self.scheduler = {'endpoints': []}
    monkeypatch.setattr(run, 'ScheduledModel', Scheduled)
    results = []
    for method in ('base', 'v20'):
        run.main(['--task', 'tsp_construct', '--run-name', 'test', '--batch', 'test', '--method', method,
                  '--scheduler-socket', '/fake', '--dry-run'])
        results.append(json.loads(capsys.readouterr().out))
    a, b = results
    assert a['method_params'] == {k: v for k, v in b['method_params'].items() if k in a['method_params']}
    for key in ('budget', 'evaluation_execution', 'task_eval', 'llm', 'model_routing', 'final_selection'):
        assert a[key] == b[key]
    assert a['llm']['model'] == 'Qwen3.8-27B'
    assert a['llm']['sampling']['chat_template_kwargs'] == {'enable_thinking': False}
