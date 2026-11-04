"""Trajectory context and experience learning, alone and in a full V20 run."""
import json
import random

from traceaad_base import Evaluation, ModelReply, Program, Task
from traceaad_base.models import Attempt
from traceaad_v20 import Config, TraceAAD
from traceaad_v20.context import Context
from traceaad_v20.experience import Experience, SEGMENT_LIMIT
from traceaad_v20.representation import ranks, structural
from traceaad_v20.selection import Selector


class Embedding:
    last_usage = {'prompt_tokens': 3, 'total_tokens': 3}

    def encode(self, text):
        return [1., float(sum(map(ord, text)) % 17 + 1), float(len(text) % 13 + 1)]


class Generator:
    def __init__(self):
        self.calls = []

    def complete(self, prompt, **settings):
        self.calls.append((prompt, settings))
        n = len(self.calls)
        return ModelReply(json.dumps({'analysis': 'Use a specific calculation.', 'design': 'Return a fixed lower score.',
                                      'code': f'def solve(x):\n    return {10 - n * .1}'}),
                          usage={'prompt_tokens': 3, 'completion_tokens': 4})


class Reviewer:
    def __init__(self, updates=()):
        self.calls = []
        self.updates = list(updates)

    def complete(self, prompt, **settings):
        self.calls.append((prompt, settings))
        return ModelReply(json.dumps({'analysis': 'The comparison adds a result.', 'review': 'The return computation changed.',
                                      'updates': self.updates}), usage={'prompt_tokens': 5, 'completion_tokens': 6})


def task():
    return Task('Minimize the returned score.', 'def solve(x):\n    return x', lambda code, fn: fn(1), time_limit=2)


def program(i, value, parent=0, code=None):
    return Program(i, code or f'def solve(x):\n    return {value}', str(value), 'Refine', parent, None,
                   Evaluation('success', value), analysis='Change the calculation.')


def score(code):
    return Evaluation('success', float(code.split('return ')[-1]))


def experience(embedder=None):
    return Experience(task(), embedder, Config())


def test_full_flow_reflects_records_costs_and_resumes(tmp_path):
    g, r = Generator(), Reviewer()
    s = TraceAAD(task(), g, config=Config(evaluation_budget=8, initial_size=2), run_dir=tmp_path,
                 embedder=Embedding(), reflection_model=r)
    s.evaluator.evaluate = score
    result = s.run()
    assert result.evaluations == result.model_calls == 8
    assert r.calls and all(x[1]['max_tokens'] == 16384 for x in r.calls)
    assert all('at most 100' not in p for p, _ in g.calls)
    assert s.experience.data['events'] and s.experience.data['reviews']
    roles = {row['role'] for a in s.attempts for row in a.record.get('aux', [])}
    assert 'embedding' in roles and roles & {'local_reflection', 'learning_reflection'}
    # The last attempt ends the budget, so it is observed without a reflection.
    assert not any(row['role'].endswith('reflection') for row in s.attempts[-1].record['aux'])
    saved = json.loads((tmp_path / 'experience.json').read_text())
    assert saved['reviews'] == s.experience.data['reviews']
    before = (len(g.calls), len(r.calls))
    again = TraceAAD(task(), g, config=s.config, run_dir=tmp_path, resume=True, embedder=Embedding(), reflection_model=r)
    assert again.experience.data['events'].keys() == s.experience.data['events'].keys()
    again.run()
    assert before == (len(g.calls), len(r.calls))
    assert all(not any(word in p for word in ['time_limit', 'n_instance', 'evaluation_seed']) for p, _ in r.calls)


def test_reference_shortlist_is_distinct_and_random():
    ctx = Context(task(), Config(), experience())
    archive = {i: program(i, 10 + i) for i in range(1, 7)}
    selected = []
    for seed in range(20):
        ref, info = ctx.reference(archive, archive[1], random.Random(seed))
        selected.append(ref.id)
        assert len(info['shortlist']) <= 4 and ref.id != 1
    assert len(set(selected)) > 1


def test_only_identical_sources_convert_crossover_to_explore(tmp_path):
    archive = {i: program(i, 1., code='def solve(x): return 1') for i in range(1, 6)}
    seen = False
    for seed in range(50):
        decision = Selector().choose(archive, Config(), random.Random(seed), distinct_crossover=False)[3]
        if decision['requested_operator'] == 'Crossover':
            seen = True
            assert decision['operator'] == 'Explore' and decision['value'] == 'X'
            assert decision['selected']['features'][1] == 0
    assert seen

    class Same(Generator):
        def complete(self, prompt, **settings):
            return json.dumps({'analysis': 'a', 'design': 'd', 'code': 'def solve(x): return 1'})
    s = TraceAAD(task(), Same(), config=Config(evaluation_budget=8, initial_size=2, refine_weight=0,
                                               explore_weight=.001, crossover_weight=1),
                 run_dir=tmp_path, embedder=Embedding(), reflection_model=Reviewer())
    s.evaluator.evaluate = lambda code: Evaluation('success', 1.)
    s.run()
    requested = [a for a in s.attempts if a.record.get('selection', {}).get('requested_operator') == 'Crossover']
    assert requested and all(a.operator == 'Explore' and a.reference_id is None for a in requested)


def test_note_create_revise_retire():
    e = experience()
    e.data['events'] = {'e1': {}, 'e2': {}}
    create = {'action': 'create', 'experience_id': None, 'lesson': 'Use a computation',
              'applicable_when': 'Inputs support it', 'new_source_event_ids': ['e1', 'unknown']}
    assert e.apply(json.dumps({'analysis': '', 'review': 'r', 'updates': [create]}), 'e1', ['e1'], [])
    assert e.data['notes']['1']['source_event_ids'] == ['e1'] and e.data['reviews']['e1'] == 'r'
    revise = {**create, 'action': 'revise', 'experience_id': 1, 'lesson': 'Revised', 'new_source_event_ids': ['e2']}
    # A note the reflection did not see cannot be changed.
    e.apply(json.dumps({'analysis': '', 'review': '', 'updates': [revise]}), 'e2', ['e2'], [])
    assert e.data['notes']['1']['lesson'] == 'Use a computation'
    e.apply(json.dumps({'analysis': '', 'review': '', 'updates': [revise]}), 'e2', ['e2'], [{'id': 1}])
    assert e.data['notes']['1']['source_event_ids'] == ['e1', 'e2'] and e.data['notes']['1']['lesson'] == 'Revised'
    retire = {**create, 'action': 'retire', 'experience_id': 1}
    e.apply(json.dumps({'analysis': '', 'review': '', 'updates': [retire]}), 'e2', ['e2'], [{'id': 1}])
    assert not e.data['notes']['1']['active'] and len(e.data['notes']['1']['history']) == 2
    assert e.data['processed'] == ['e1', 'e2']


def test_parse_failure_and_local_mode_do_not_consume_events():
    e = experience()
    e.data['events']['e'] = {}
    assert not e.apply('not json', 'e', ['e'], [])
    e.apply(json.dumps({'analysis': '', 'review': '', 'updates': [{'action': 'create'}]}), 'e', [], [])
    assert not e.data['processed'] and not e.data['notes']


def test_single_note_has_no_similarity_threshold_and_keeps_both_outcomes():
    exp = experience(Embedding())
    for i, (value, result) in enumerate([(1., 'improved'), (3., 'regressed')], 1):
        exp.data['events'][f'e{i}'] = {'event_id': f'e{i}', 'attempt_id': i, 'parent_id': 1, 'child_id': i + 1,
            'operator': 'Refine', 'parent_fitness': 2., 'child_fitness': value, 'status': 'success', 'result': result,
            'parent_code': 'def solve(x): return 2', 'parent_design': 'Return two', 'code': f'def solve(x): return {value}',
            'structural': structural('def solve(x): return 2'), 'analysis': 'Change return', 'design': 'Return score'}
    exp.data['notes']['1'] = {'id': 1, 'active': True, 'lesson': 'Change a computation',
                              'applicable_when': 'A return can change', 'source_event_ids': ['e1', 'e2']}
    notes, _ = exp.retrieve(program(1, 2.), 'Refine')
    assert len(notes) == 1 and len(notes[0]['observations']) == 2
    assert any('worse than the parent' in x for x in notes[0]['observations'])


def test_ancestor_change_points_keep_nearest():
    ctx = Context(task(), Config(), experience())
    archive = {i: program(i, float(30 - i), i - 1) for i in range(1, 21)}
    selected, path = ctx.ancestors(archive, archive[20])
    assert len(path) == 19 and len(selected) == 8 and [p.id for p in selected][-3:] == [17, 18, 19]


def event(i, parent, fitness, operator='Refine'):
    return {'event_id': str(i), 'attempt_id': i, 'operator': operator, 'parent_id': parent, 'child_id': i + 1,
            'child_fitness': fitness, 'code_identity': str(i), 'status': 'success', 'best_before': 0.}


def test_followup_segment_uses_unprocessed_children_of_an_exploration():
    exp = experience()
    exp.data['events'] = {'1': event(1, 1, 4., 'Explore'), '2': event(2, 2, 3.), '3': event(3, 2, 2.)}
    assert exp.segment('3', 3) == ['1', '2', '3']
    exp.data['processed'] = ['1', '2', '3']
    assert exp.segment('3', 3) == []


def test_large_segment_keeps_required_and_most_recent():
    exp = experience()
    for i in range(1, 21):
        exp.data['events'][str(i)] = event(i, 1, float(i))
    selected = exp.segment('20', 21)
    assert len(selected) == SEGMENT_LIMIT
    assert {'1', '20'} <= set(selected) and selected[-11:] == [str(i) for i in range(10, 21)]


def test_results_are_judged_against_the_trajectory_best():
    exp = experience(Embedding())
    # 1 (score 7) -> 2 (score 30) -> 3 recovers to 8, 4 beats 7, 5 regresses.
    archive = {1: program(1, 7.), 2: program(2, 30., parent=1), 3: program(3, 8., parent=2),
               4: program(4, 6., parent=2), 5: program(5, 31., parent=2)}
    labels = {}
    for child in (3, 4, 5):
        attempt = Attempt(child, 2, 'Refine', program_id=child, evaluations=[{'id': child}],
                          record={'context': {'best_before': 6.}})
        labels[child] = exp.data['events'][exp.observe(attempt, archive)]
    assert labels[3]['result'] == 'recovered' and labels[3]['reference_fitness'] == 7.
    assert labels[4]['result'] == 'improved' and labels[5]['result'] == 'regressed'
    card = exp.card(labels[3]['event_id'])
    assert 'not better than the trajectory best' in card and 'trajectory best 7.0' in card


def test_equal_channels_give_no_ranking():
    assert ranks({1: None, 2: None}) is None and ranks({1: 0., 2: 0.}) is None


def test_each_section_has_one_header_and_guidance_follows_shown_material():
    ctx = Context(task(), Config(), experience())
    archive = {i: program(i, float(30 - i), i - 1) for i in range(1, 6)}
    for i, value in [(6, 20.), (7, 24.), (8, 21.)]:
        archive[i] = program(i, value, 5)
    archive[9] = Program(9, 'def solve(x):\n    return y', 'e', 'Refine', 5, None,
                         Evaluation('error', None, 'NameError'), analysis='Use y.')
    sections, required, view = ctx.render(archive, archive[5], 'Explore', None, [])
    history = ctx.assemble(sections, view)
    for header in ['[Ancestor History]', '[Evaluated Offspring]', '[Failed Offspring]']:
        assert history.count(header) == 1
    assert 'Direct attempts from Algorithm 5: 3 valid; showing 3.' in history
    assert 'Failed direct attempts from Algorithm 5: 1; showing 1.' in history
    assert '[Relevant Experience]' not in history
    instruction = ctx.instruction('Explore', sections)
    assert 'formation path' in instruction and 'already tried from this parent' in instruction
    assert 'execution feedback' in instruction and 'Use a note' not in instruction
    assert instruction.endswith('not measured improvements.')
    assert 'need not retain the current representation' in instruction
    bare = ctx.instruction('Refine', [])
    assert 'formation path' not in bare and 'already tried' not in bare and 'execution feedback' not in bare
