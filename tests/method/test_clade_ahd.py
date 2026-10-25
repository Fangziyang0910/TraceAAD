"""Offline checks for clade evidence, upstream execution quirks, budget and recovery."""

import random

import numpy as np
import pytest

from baselines.clade_ahd import CladeAHD
from baselines.clade_ahd.upstream.bayesian_node import BayesianNode
from baselines.clade_ahd.upstream.mcts import MCTS
from baselines.clade_ahd.upstream.score_normalizer import ScoreNormalizer
from baselines.shared.profiler import ProfilerBase
from experiments.infra.resume_baseline import recover
from tests.support import TinyEvaluation, TokenLLM
from traceaad.storage import read_json, rows, write_json


def program(n):
    return f'{{Return {n}.}}\n```python\ndef score(x):\n    return {n}\n```'


class Scripted(TokenLLM):
    def __init__(self, reply=program, *, n=0, fail_first=False):
        super().__init__()
        self.n, self.reply, self.fail_first = n, reply, fail_first
        self.stages = []

    def draw_sample_with_details(self, prompt, **kwargs):
        self.calls.append(prompt)
        if self.fail_first and len(self.calls) == 1:
            raise ValueError('Offline transport failure')
        alignment = 're-describe the algorithm' in prompt
        self.stages.append('alignment' if alignment else 'generate')
        if alignment:
            content = f'Choose constant {self.n}.'
        else:
            self.n += 1
            content = self.reply(self.n)
        return {'content': content, 'model': 'offline',
                'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}}


def method(directory, llm=None, budget=18, **options):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'run_config.json', {'method': 'clade_ahd', 'task': 'toy',
                                               'budget': budget, 'method_params': {}})
    random.seed(7)
    np.random.seed(7)
    return CladeAHD(llm or Scripted(), TinyEvaluation(), ProfilerBase(directory),
                    max_sample_nums=budget, **options)


def stable_tree(instance):
    state = instance._snapshot_tree()
    for node in state['nodes']:
        node['raw_info'].pop('sample_time', None)
        for child in node['children_info']:
            child.pop('sample_time', None)
    return state


def test_subtree_thompson_uses_descendants_visits_pseudo_evidence_and_cooling(monkeypatch):
    tree = MCTS('Root', num_pseudo=2, cool_down_enabled=True)
    parent = BayesianNode(tree.root, 'a')
    parent.alpha, parent.beta = 1, 3
    child = BayesianNode(parent, 'b')
    child.alpha, child.beta, child.visits = 3, 1, 2
    parent.add_child(child)
    frozen = BayesianNode(tree.root, 'frozen')
    frozen.is_frozen = True
    calls = []
    monkeypatch.setattr(np.random, 'beta', lambda a, b: calls.append((a, b)) or a / (a + b))
    assert parent.get_clade_scores(2) == [.25, .25, .75, .75]
    assert tree.select_clade_thompson([frozen, parent], progress_ratio=.5) == 1
    assert calls == [(6.0, 6.0)]


def test_normalization_and_actual_double_child_update(tmp_path):
    instance = method(tmp_path, budget=1)
    instance.run()
    node = instance._mcts.root.children[0]
    assert type(instance._mcts.normalizer) is ScoreNormalizer
    assert (node.alpha, node.beta, node.visits) == (2, 2, 3)
    assert (instance._mcts.root.alpha, instance._mcts.root.beta) == (1.4, 1.4)
    normalizer = ScoreNormalizer('min')
    normalizer.update(-4)
    normalizer.update(-2)
    assert normalizer.normalize(-4) == pytest.approx(1)
    assert normalizer.normalize(-2) == 0
    assert normalizer.normalize(float('nan')) == 0


def test_freezing_a_subtree_updates_all_descendants():
    tree = MCTS('Root', pruning_enabled=True, pruning_threshold=.5, pruning_min_visits=1)
    parent = BayesianNode(tree.root, 'a')
    child = BayesianNode(parent, 'b')
    parent.visits = child.visits = 2
    child.alpha, child.beta = 1, 99
    tree.root.add_child(parent)
    parent.add_child(child)
    tree.prune_tree(1)
    assert parent.is_frozen and child.is_frozen
    assert child.frozen_reason


def test_bounded_search_and_batch_snapshot(tmp_path):
    llm = Scripted()
    instance = method(tmp_path, llm, budget=9)
    instance.run()
    assert instance._tot_sample_nums == instance._evaluation_count == 9
    assert len(instance._mcts.get_all_nodes()) == 10
    assert llm.stages == ['generate', 'alignment'] * 9
    prompts = [p for p, stage in zip(llm.calls, llm.stages) if stage == 'generate']
    assert [n.operator for n in instance._mcts.get_all_nodes()[1:5]] == ['i1', 'e1', 'e1', 'e1']
    assert all('Return a numeric score.' in prompt for prompt in prompts)
    # Four initial nodes, then e2/m1/m1/m2/m2 at the same leaf; no same-batch offspring is visible.
    assert all('return 5\n' not in prompt for prompt in prompts[5:])
    summary = read_json(tmp_path / 'summary.json')
    assert summary['status'] == 'finished'
    assert summary['budget_used'] == 9 and summary['evaluation_calls'] == 9
    assert summary['selection_evaluations'] == 0
    assert summary['best']['fitness'] == -9
    assert (tmp_path / 'best_program.py').is_file()


def test_bad_and_duplicate_replies_spend_budget_but_transport_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr('baselines.clade_ahd.upstream.evolution.time.sleep', lambda _: None)
    bad = method(tmp_path / 'bad', Scripted(lambda _: 'unparsable'), budget=3)
    bad.run()
    assert bad._tot_sample_nums == 3 and bad._evaluation_count == 0
    assert not bad._mcts.root.children
    duplicated = method(tmp_path / 'duplicates', Scripted(lambda _: program(1)), budget=5)
    duplicated.run()
    assert duplicated._tot_sample_nums == 5
    assert len(duplicated._mcts.root.children) == 1
    transported = method(tmp_path / 'transport', Scripted(fail_first=True), budget=1)
    transported.run()
    assert transported._tot_sample_nums == transported._evaluation_count == 1
    assert not transported._search_aborted
    assert read_json(tmp_path / 'transport' / 'summary.json')['model_calls'] == 3


def test_failed_evaluations_do_not_enter_tree(tmp_path):
    llm = Scripted(lambda n: program(n) if n != 2 else program('1 / 0'))
    instance = method(tmp_path, llm, budget=3)
    instance.run()
    assert instance._tot_sample_nums == instance._evaluation_count == 3
    assert [c.objective for c in instance._mcts.root.children] == [-1, -3]
    summary = read_json(tmp_path / 'summary.json')
    assert summary['candidate_count'] == 3 and summary['valid_candidate_count'] == 2


def test_restored_tree_and_rng_continue_identically_at_round_boundary(tmp_path):
    full_llm = Scripted()
    full = method(tmp_path / 'full', full_llm, budget=22)
    full.run()
    interrupted_llm = Scripted()
    interrupted = method(tmp_path / 'resume', interrupted_llm, budget=22)
    checkpoint = interrupted._checkpoint

    def interrupt(boundary):
        checkpoint(boundary)
        if boundary == 'round':
            raise KeyboardInterrupt

    interrupted._checkpoint = interrupt
    interrupted.run()
    assert read_json(tmp_path / 'resume' / 'summary.json')['status'] == 'interrupted'
    continued_llm = Scripted(n=interrupted_llm.n)
    continued = method(tmp_path / 'resume', continued_llm, budget=22)
    report = recover(continued, tmp_path / 'resume')
    assert report['recovery'] == 'search_checkpoint' and report['pending_budget'] == 0
    assert stable_tree(continued) == stable_tree(interrupted)
    continued.run()
    assert stable_tree(continued) == stable_tree(full)
    assert interrupted_llm.calls + continued_llm.calls == full_llm.calls
    assert continued._tot_sample_nums == continued._evaluation_count == 22


def test_recovery_preserves_frozen_nodes_and_normalization_history(tmp_path):
    instance = method(tmp_path, budget=10)
    for _ in range(3):
        child = instance._offspring(instance._init_population,
                                    'e1' if instance._init_population else 'i1', initial=True)
        instance._attach(instance._mcts.root, child, initial=True)
    instance._mcts.root.children[1].is_frozen = True
    instance._mcts.root.children[1].frozen_reason = 'test'
    instance._checkpoint('initialization')
    restored = method(tmp_path, budget=10)
    recover(restored, tmp_path)
    assert stable_tree(restored) == stable_tree(instance)
    assert restored._mcts.root.children[1].is_frozen
    assert restored._mcts.normalizer.score_history == [-1, -2, -3]


def test_recovery_retains_mid_batch_costs_and_candidates(tmp_path):
    class Interrupting(Scripted):
        def draw_sample_with_details(self, prompt, **kwargs):
            if self.n == 6:
                raise KeyboardInterrupt
            return super().draw_sample_with_details(prompt, **kwargs)

    interrupted = method(tmp_path, Interrupting(), budget=12)
    interrupted.run()
    assert interrupted._tot_sample_nums == 6
    # Generation six returned, but its thought alignment was interrupted: five evaluations survived.
    assert interrupted._evaluation_count == 5
    continued = method(tmp_path, Scripted(n=6), budget=12)
    report = recover(continued, tmp_path)
    assert report['pending_budget'] == 2
    assert continued._evaluation_count == 5 and continued._tot_sample_nums == 6
    continued.run()
    assert continued._tot_sample_nums == 12 and continued._evaluation_count == 11
    summary = read_json(tmp_path / 'summary.json')
    assert summary['evaluation_calls'] == 11 and summary['budget_used'] == 12


def test_parser_keeps_helper_definitions_and_uses_clade_fallback_description(tmp_path):
    reply = 'A direct construction.\n```python\ndef helper(x):\n    return 3\n\ndef score(x):\n    return helper(x)\n```'
    instance = method(tmp_path, Scripted(lambda _: reply), budget=1)
    instance.run()
    assert instance._mcts.root.children[0].objective == -3
    assert 'def helper' in (tmp_path / 'best_program.py').read_text()


def test_cli_and_queue_register_clade_without_launching(tmp_path):
    from experiments.clade_ahd.run import RunSpec, build_parser, run_config
    from experiments.infra.local_queue import build_parser as queue_parser, build_plan

    args = vars(build_parser().parse_args(['--task', 'tsp_construct', '--budget', '9',
                                           '--no-pruning-enabled', '--dry-run']))
    assert args.pop('dry_run')
    spec = RunSpec(**args)
    config = run_config(spec, tmp_path, 'test')
    assert config['method'] == 'clade_ahd' and config['budget'] == 9
    assert config['method_params']['normalization'] == 'minmax'
    assert config['method_params']['pruning_enabled'] is False
    plan = build_plan(queue_parser().parse_args(['--method', 'clade_ahd', '--suite', 'co6',
                                                 '--batch', 'test', '--dry-run']))
    assert len(plan) == 18
    assert all(p.module == 'experiments.clade_ahd.run' for p in plan)


@pytest.mark.parametrize('options', [{'max_sample_nums': 0}, {'num_pseudo_evals': -1},
                                   {'pruning_threshold': float('nan')}, {'backprop_lambda': 2},
                                   {'cool_down_beta': float('inf')}])
def test_invalid_configuration_fails_before_calls(options):
    llm = Scripted()
    with pytest.raises(ValueError):
        CladeAHD(llm, TinyEvaluation(), **options)
    assert not llm.calls
