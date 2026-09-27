"""V10.14-2's information, eligibility and accounting contracts."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from core import SecureEvaluator
from tests.support import FakeLLM, TinyEvaluation, response
from traceaad.v10_14_2 import Config, TraceAADV10142
from traceaad.v10_14_2.edits import SourceError, parse_response
from traceaad.v10_14_2.evaluation import describe_decision, training_probes
from traceaad.v10_14_2.selection import select_parent, unique_archive
from traceaad.v10_14_2.state import Facts


class TokenLLM(FakeLLM):
    """A real local tokenizer for offline tests, never a production substitute."""
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text).ids)

    def count_prompt_tokens(self, text):
        return self.count_tokens(text) + 16


def full(code, idea=""):
    return (idea + '\n' if idea else '') + '```python\n' + code.rstrip() + '\n```'


def method(tmp_path, responses=(), **settings):
    config = Config(**{**dict(budget=40, max_evaluations=40, init_proposals=1), **settings})
    return TraceAADV10142(evaluation=TinyEvaluation(), llm=TokenLLM(*responses), run_dir=tmp_path, config=config)


def test_all_valid_sources_compete_and_duplicate_does_not_reset_opportunity(tmp_path):
    m = method(tmp_path, [response(1), response(.9), response(.9), response(1)], budget=4)
    m.run()
    assert len(m.anchors) == 4 and len(unique_archive(m.anchors)) == 2
    assert m.ledger.evaluations == 2
    pool = unique_archive(m.anchors)
    worse = next(a for a in pool if a['fitness'] == .9)
    better = next(a for a in pool if a['fitness'] == 1)
    selected, trace = select_parent(m.anchors, {better['artifact_id']: 50, worse['artifact_id']: 0})
    assert selected == worse and trace['eligible_unique_sources'] == 2
    selected, _ = select_parent(m.anchors, {better['artifact_id']: 0, worse['artifact_id']: 100})
    assert selected == better  # no guarantee that each worse child gets a turn
    assert sum(m.parent_counts.values()) == 3
    assert m.ledger.channel_used['trial'] == m.ledger.channel_used['recheck'] == 0


def test_equal_quality_candidates_and_v9_tie_breaks():
    a = {1: {'id': 1, 'artifact_id': 'a', 'fitness': -744.5},
         2: {'id': 2, 'artifact_id': 'b', 'fitness': -744.5},
         3: {'id': 3, 'artifact_id': 'b', 'fitness': -744.5}}
    assert select_parent(a, {})[0]['id'] == 1
    assert select_parent(a, {'a': 1})[0]['id'] == 2
    assert select_parent(a, {'a': 1, 'b': 1})[0]['id'] == 1


def test_resume_retains_source_counts_rng_and_charged_failure(tmp_path):
    m = method(tmp_path, [response(1), response(.8), full('def score(x):\n    return missing')])
    m._initialize()
    m._initialize()
    m._initialize()
    m._schedule_search()
    m._step_session()
    m._finish_session()
    counts, state = dict(m.parent_counts), m.rng.getstate()
    restored = method(tmp_path)
    assert restored.parent_counts == counts and restored.rng.getstate() == state
    assert restored.ledger.candidates == 3
    assert restored.facts.tables['attempt'][3]['status'] == 'evaluation_failed'
    assert restored.repairs
    restored.pending = {'kind': 'generation'}
    restored._save()
    with pytest.raises(RuntimeError, match='uncertain'):
        method(tmp_path)


def test_four_actions_equal_and_probes_do_not_exclude_donor(tmp_path):
    m = method(tmp_path, [full('def score(x):\n    a = 1\n    return a'), response(.9)])
    m._initialize()
    m._initialize()
    parent = m.anchors[1]
    # Equal descriptor is deliberately a counterexample to the old donor gate.
    assert parent['profile'] == m.anchors[2]['profile'] == []
    actions = Counter()
    for _ in range(8000):
        scope, ref, donor = m._contract(parent)
        actions['Transfer' if ref == 'Transfer' else scope] += 1
        if ref == 'Transfer':
            assert donor['artifact_id'] != parent['artifact_id']
    assert set(actions) == {'Refine', 'Tune', 'Pivot', 'Transfer'}
    assert all(1750 < n < 2250 for n in actions.values())
    m2 = method(tmp_path / 'single', [response(1)])
    m2._initialize()
    assert {m2._contract(m2.anchors[1])[0] for _ in range(100)} == {'Refine', 'Pivot'}


@pytest.mark.parametrize('label', ['Idea:', '**Idea:**', '**Idea**:'])
def test_idea_end_to_end_raw_parse_journal_reload_next_prompt(tmp_path, label):
    rationale = 'Change the capacity contrast.\nPreserve feasibility.\nCheck late states; this is a hypothesis.'
    m = method(tmp_path, [response(1), full('def score(x):\n    return 2', label + ' ' + rationale)])
    m._initialize()
    m._initialize()
    disk = Facts(tmp_path)
    anchor = disk.tables['anchor'][2]
    assert anchor['idea'] == rationale
    assert disk.tables['revision'][2]['idea'] == rationale
    restored = method(tmp_path)
    packet = restored.prompts.build(restored.anchors[2])
    assert 'Check late states' in packet['prompt']
    assert 'revision:2' in packet['evidence_ids']
    events = restored.prompts.events(restored.anchors[2])
    event = next(e for e in events if e['source'] == 'formation')
    assert event['idea']['text'] == rationale
    assert '-    return 1' in event['diff'] and '+    return 2' in event['diff']
    assert (event['old_score'], event['new_score']) == (1, 2)
    assert 'at most 500 tokens' in packet['prompt']


def test_idea_token_limit_is_a_view_not_fact_loss():
    idea = ' '.join('longdescription' for _ in range(610))
    info = {}
    code, raw = parse_response(full('def score(x):\n    return 1', '**Idea:** ' + idea), 'stop',
        TinyEvaluation().template_program, metadata=info, count_tokens=TokenLLM().count_tokens)
    assert raw == idea and len(raw) > 1000 and info['idea']['raw_tokens'] == 610
    assert info['idea']['display_tokens'] <= 500 and info['idea']['truncated']
    assert parse_response(full(code), 'stop', TinyEvaluation().template_program)[1] == ''


def test_template_constants_helpers_classes_and_imports_are_retained():
    template = '# Target: score\nimport math\nBASE = 2\ndef scale(x):\n    return x * BASE\nclass C:\n    def f(self, x):\n        return scale(x)\ndef score(x):\n    pass'
    submitted = 'LOCAL = 3\ndef helper(x):\n    return C().f(x)\ndef score(x):\n    return helper(x) + LOCAL + math.floor(x)'
    info = {}
    code, _ = parse_response(full(submitted), 'stop', template, metadata=info)
    ns = {}
    exec(code, ns)
    assert ns['score'](2) == 9
    assert submitted in code
    assert set(info['template_additions']) == {'math', 'BASE', 'scale', 'C'}
    override, _ = parse_response(full('BASE = 7\ndef score(x):\n    return BASE'), 'stop', template)
    assert 'BASE = 2' not in override


def test_complete_module_preserved_and_unknown_names_not_invented():
    source = 'import math\nCONST = 2\nclass H:\n    def run(self, x):\n        return math.floor(x)\ndef helper(x):\n    return H().run(x)\ndef score(x):\n    return helper(x) + CONST'
    code, _ = parse_response(full(source), 'stop', TinyEvaluation().template_program)
    assert code == source
    unknown, _ = parse_response(full('def score(x):\n    return forgotten(x)'), 'stop', TinyEvaluation().template_program)
    result = SecureEvaluator(TinyEvaluation()).evaluate_program_with_details(unknown)
    assert result.result is None and 'forgotten' in result.error


def test_multiple_drafts_final_marker_and_explicit_parts():
    template = TinyEvaluation().template_program
    info = {}
    source = full('def score(x):\n    return 1') + '\n' + full('def score(x):\n    return 2')
    code, _ = parse_response(source, 'stop', template, metadata=info)
    assert 'return 2' in code and info['block_indices'] == [1]
    source = 'Final implementation:\n' + source
    code, _ = parse_response(source, 'stop', template, metadata=info)
    assert 'return 1' in code and info['strategy'] == 'explicit_final'
    parts = 'Candidate: solution; Part 1/2\n' + full('C = 2\ndef helper(x):\n    return x+C') + '\nCandidate: solution; Part 2/2\n' + full('def score(x):\n    return helper(x)')
    code, _ = parse_response(parts, 'stop', template, metadata=info)
    ns = {}
    exec(code, ns)
    assert ns['score'](3) == 5 and info['strategy'] == 'explicit_ordered_parts'
    with pytest.raises(ValueError, match='conflicting'):
        parse_response(parts.replace('def score(x):', 'C = 3\ndef score(x):'), 'stop', template)
    with pytest.raises(ValueError, match='one group'):
        parse_response(parts.replace('Candidate: solution; Part 2', 'Candidate: other; Part 2'), 'stop', template)
    # Ambiguous separate helper is never silently spliced into the final block.
    code, _ = parse_response(full('def helper(x):\n    return 3') + '\n' + full('def score(x):\n    return helper(x)'), 'stop', template)
    assert 'def helper' not in code


def test_incomplete_payload_and_conflicting_template_are_rejected():
    with pytest.raises(ValueError, match='incomplete'):
        parse_response(response(1), 'length', TinyEvaluation().template_program)
    with pytest.raises(SourceError, match='overwrite'):
        parse_response(full('A = 3\ndef score(x):\n    return B'), 'stop',
                       'A = B = 2\ndef score(x):\n    pass')


def test_obp_feedback_retains_index_capacity_residual_and_ties():
    p = {'kind': 'ranking', 'args': (4, np.array([8, 8, 10]))}
    assert describe_decision(np.array([1, 1, 0]), p) == [0, 8, 4, 2]
    assert describe_decision(np.array([0, 1, 0]), p) == [1, 8, 4, 1]


def test_aco_descriptor_distinguishes_same_ranking_different_probabilities():
    p = {'kind': 'matrix', 'args': (np.ones((3, 3)), np.ones(3)), 'mask_depot': False,
         'states': [{'current': 0, 'mask': [0, 1, 1], 'pheromone': [1, 1, 1], 'alpha': 1., 'beta': 1.}]}
    a = np.array([[0., 2, 1], [1, 0, 1], [1, 1, 0]])
    b = a.copy()
    b[0, 1] = 200
    assert np.array_equal(np.argsort(a), np.argsort(b))
    assert not np.allclose(describe_decision(a, p), describe_decision(b, p))
    assert sum(describe_decision(b, p)) == pytest.approx(1)


def test_defaults_are_subtractive_and_estimated_tokenizers_rejected(tmp_path):
    config = Config()
    assert not config.online_revalidation and not config.fixed_three_step_commitment and not config.behavior_eligibility_gate
    assert config.exploration_constant == 1 and config.idea_tokens == 500
    with pytest.raises(ValueError, match='fixed_three'):
        replace(config, trial_fraction=.2)
    with pytest.raises(ValueError, match='online_revalidation'):
        replace(config, recheck_fraction=.1)
    with pytest.raises(ValueError, match='serving tokenizer'):
        TraceAADV10142(evaluation=TinyEvaluation(), llm=FakeLLM(), run_dir=tmp_path)


def test_tsp_probes_cover_early_middle_and_tail():
    from tests.experiments.test_traceaad_v1014_experiments import small_task
    evaluation = small_task('tsp_construct')
    remaining = {len(p['args'][2]) for p in training_probes(evaluation, 'tsp_construct')}
    assert {9, 4, 1} <= remaining


def test_assembly_rejects_eager_forward_dependency():
    parts = ('Candidate: one; Part 1/2\n' + full('A = B + 1')
             + '\nCandidate: one; Part 2/2\n' + full('B = 2\ndef score(x):\n    return A'))
    with pytest.raises(SourceError, match='execution order'):
        parse_response(parts, 'stop', TinyEvaluation().template_program)


def test_tokenizer_failure_preserves_complete_source_and_raw_idea():
    def failing_tokenizer(text):
        raise ConnectionError('offline')
    info = {}
    code, idea = parse_response(response(1), 'stop', TinyEvaluation().template_program,
                               metadata=info, count_tokens=failing_tokenizer)
    assert idea == 'return 1' and 'return 1' in code
    assert info['idea']['token_count_status'] == 'unavailable'


def test_repair_is_charged_and_original_action_is_separate(tmp_path):
    m = method(tmp_path, [response(1), response(.9),
                         full('def score(x):\n    return absent'), response(2)])
    m._initialize()
    m._initialize()
    m._initialize()
    parent = m.anchors[1]
    m._begin('main', 1, parent, 0)
    m._step_session()
    m._finish_session()
    m._begin('main', 1, parent, 0)
    m._step_session()
    m._finish_session()
    attempt = m.facts.tables['attempt'][4]
    assert attempt['repair_of'] == 3 and attempt['executed_action'] == 'Repair'
    assert attempt['sampled_action'] in {'Refine', 'Pivot', 'Transfer'}
    assert m.ledger.candidates == m.ledger.calls == m.ledger.evaluations == 4


def test_diagnostic_timeout_never_removes_valid_candidate(tmp_path):
    from traceaad.v10_14_2.evaluation import SeededEvaluation
    m = method(tmp_path, [response(1)])
    probe = {'kind': 'node', 'args': (0,), 'scene': 'timeout diagnostic'}
    adapter = SeededEvaluation(TinyEvaluation(), [probe], diagnostic_only=True)
    adapter.timeout_seconds = .15
    m.probes = [probe]
    m.probe_evaluator = SecureEvaluator(adapter)
    source = 'import time\ndef score(x):\n    if x == 0:\n        time.sleep(3)\n    return 1'
    m.llm.responses = iter([full(source)])
    m._initialize()
    assert m.anchors[1]['fitness'] == 1 and not m.anchors[1]['profile']
    assert m.facts.tables['evaluation'][1]['result']['probe_error']
