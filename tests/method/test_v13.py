import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from core.llm import generate, ModelCallError
from core.scheduling import SchedulerError
from traceaad.program import Program
from traceaad.storage import read_json, rows, append_jsonl
from traceaad.v13 import Config, TraceAADV13
from traceaad.v13.knowledge import (ancestors, eligible, bindings, apply_edits, specification,
                                    distribution, evidence_rate, trial_view, unchanged_regions)
from tests.support import TinyEvaluation


def spec(**kw):
    return dict(name='decrease offset', applicability='numeric return', operation='reduce return by one',
                dependencies='preserve inputs', parameters='one unit', preserve='interface', checks='new value',
                requires=[], **kw)


def code(value):
    return f'def f(x):\n    return {value}\n'


class ScriptedLLM:
    model = 'fixture'
    def __init__(self):
        self.calls = []
        self.reject_audit = False
        self.no_matches = False
        self.revision = 'keep'
    def count_prompt_tokens(self, text):
        return len(text) // 4
    count_tokens = count_prompt_tokens
    def draw_sample_with_details(self, text, **kwargs):
        stage = re.search(r'\[V13:(\w+)\]', text)[1]
        data = json.loads(text.rsplit('\n', 1)[1])
        self.calls.append((stage, text))
        if stage == 'generate':
            parent = data.get('CURRENT_PARENT')
            value = int(parent['code'].split('return ')[-1]) - 1 if parent else 20 + len(data['context'][0]['existing_roots']) * 10
            value = dict(code=code(value), analysis='decrease', design='offset')
        elif stage == 'describe':
            pid = data['candidate']['id']
            route = 'r1' if pid % 2 else 'r2'
            value = dict(change='reduce offset', key='offset', lineage='offset path', features=['numeric return'],
                         route=route if route in data['routes'] else None, route_name=route)
        elif stage == 'learn':
            before = data['evaluated_edits'][-1]['parent']['code']
            value = dict(kind='attach' if data['library'] else 'new',
                         id=data['library'][0]['id'] if data['library'] else None,
                         spec=spec(), bindings=[dict(role='return', old=before)])
        elif stage == 'match':
            value = dict(matches=[] if self.no_matches else [dict(id=t['id'], bindings=[dict(role='return', old=data['CURRENT_PARENT']['code'])])
                                                            for t in data['candidates']])
        elif stage == 'edit':
            before = data['CURRENT_PARENT']['code']
            n = int(before.split('return ')[-1]) - 1
            value = dict(design='decrease', analysis='apply', edits=[dict(old=before, new=code(n))])
        elif stage == 'audit':
            value = dict(summary='decrease', operation_ok=not self.reject_audit, dependencies_ok=True,
                         preservation_ok=True, parameters_ok=True, reason='fixture audit')
        elif stage == 'revise':
            value = dict(kind=self.revision, specs=[] if self.revision=='keep' else [spec()], components=[], reason='evidence')
        else:
            raise AssertionError(stage)
        return dict(content=json.dumps(value), finish_reason='stop', usage={'prompt_tokens': 20, 'completion_tokens': 10})
    def close(self):
        pass


class Evaluator:
    def __init__(self):
        self.calls = []
    def evaluate(self, source, pid):
        self.calls.append(pid)
        value = int(source.split('return ')[-1])
        return dict(fitness=value, failure=None, seconds=.01, calls=1, function_seconds=.01,
                    evaluations=[dict(role='search', program_id=pid, score=value)])


def make(path, llm=None, **config):
    task = TinyEvaluation()
    task.template_program = code(0)
    search = TraceAADV13(evaluation=task, llm=llm or ScriptedLLM(), run_dir=path, task='fixture',
                        config=Config(**dict(budget=18, roots=2, revision_interval=2, **config)))
    search.training = Evaluator()
    return search


def test_end_to_end_all_stages_and_costs(tmp_path):
    search = make(tmp_path)
    result = search.run()
    assert result['status'] == 'finished'
    assert result['budget_used'] == 18
    assert result['model_calls'] == len(search.llm.calls) == result['reserved_model_calls']
    assert {'generate','describe','learn','match','edit','audit','revise'} <= set(result['stages'])
    assert len(search.state['library']) == 1  # equivalent source edits attach, not duplicate
    assert any(p.action == 'Crossover' for p in search.programs.values())
    assert result['best']['fitness'] == min(p.fitness for p in search.archive.values())
    assert read_json(tmp_path/'selection.json')['selected_node'] == result['best']['id']
    assert result['measured_input_tokens'] == 20 * result['model_calls']
    assert result['missing_usage'] == 0
    assert read_json(tmp_path/'transformations.json')['schema'] == 13
    assert all('function_seconds_limit' not in text for _, text in search.llm.calls)
    assert all(len(t['records']) >= t['reviewed'] for t in search.state['library'].values())
    generated = [a for a in search.state['attempts'] if a['mode'] != 'Init']
    assert sum(a['mode']=='Open' for a in generated) >= int(search.config.open_floor*len(generated))


@pytest.mark.parametrize('stage', ['generate', 'describe', 'learn', 'match', 'edit', 'audit', 'revise'])
def test_committed_reply_replays_without_regeneration(tmp_path, monkeypatch, stage):
    control = make(tmp_path/'control')
    expected = control.run()
    interrupted = make(tmp_path/'interrupted')
    original = interrupted._json
    def stop_after_response(reply):
        if reply['stage'] == stage:
            raise KeyboardInterrupt()
        return original(reply)
    monkeypatch.setattr(interrupted, '_json', stop_after_response)
    with pytest.raises(KeyboardInterrupt):
        interrupted.run()
    resumed = make(tmp_path/'interrupted')
    actual = resumed.run()
    assert interrupted.llm.calls + resumed.llm.calls == control.llm.calls
    assert actual['model_calls'] == expected['model_calls']
    assert actual['best']['fitness'] == expected['best']['fitness']
    assert resumed.state['library'] == control.state['library']


def test_unknown_inflight_request_keeps_its_budget_slot(tmp_path, monkeypatch):
    first = make(tmp_path, max_model_calls=2)
    monkeypatch.setattr(first.llm, 'draw_sample_with_details', lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        first.run()
    resumed = make(tmp_path, max_model_calls=2)
    result = resumed.run()
    assert result['reserved_model_calls'] == 2
    assert result['model_calls'] == 1 and result['uncertain_calls'] == 1
    assert result['stop_reason'] == 'model_call_budget'
    assert result['best']['fitness'] == 20


def test_evaluation_resume_does_not_repeat_model_calls(tmp_path, monkeypatch):
    first = make(tmp_path)
    monkeypatch.setattr(first.training, 'evaluate', lambda *a: (_ for _ in ()).throw(SchedulerError('offline')))
    assert first.run()['status'] == 'service_unavailable'
    assert len(first.llm.calls) == 1
    resumed = make(tmp_path)
    assert resumed.run()['status'] == 'finished'
    assert resumed.llm.calls[0][0] == 'describe'


def test_audit_rejection_is_not_performance_failure(tmp_path):
    model = ScriptedLLM()
    model.reject_audit = True
    search = make(tmp_path, llm=model)
    search.run()
    rejected = [a for a in search.state['attempts'] if a['mode'] == 'Transform']
    assert rejected and all(a['outcome'] == 'implementation_failed' for a in rejected)
    assert all(a['child'] not in search.training.calls for a in rejected)
    assert all(search.programs[a['child']].status == 'implementation_failed' for a in rejected)


def node(pid, parent=None, reference=None, text=None):
    return Program(id=pid, code=text or code(pid), idea='', analysis=None, action='Crossover' if reference else 'Open',
                   generation=pid, parent_id=parent, reference_id=reference, fitness=pid, status='valid')


def test_dag_late_sibling_failure_and_changed_regions():
    programs = {1:node(1), 2:node(2,1), 3:node(3,1), 4:node(4,2,3), 5:node(5,3)}
    assert ancestors(4, programs) == {1,2,3,4}
    library = {'t1':dict(spec=spec(),family='t1',components=[])}
    records = [dict(parent=3,child=5,transform='t1',outcome='no_improvement',bindings=[dict(role='return',old='return 4')])]
    assert trial_view(4,programs,records)[0]['count'] == 1
    assert eligible(programs[4],programs,library,records) == []
    programs[4].code = code(9)
    assert eligible(programs[4],programs,library,records) == ['t1']
    records[0].update(child=2, outcome='improved')
    assert eligible(programs[4],programs,library,records) == []
    records[0]['outcome'] = 'implementation_failed'
    assert eligible(programs[4],programs,library,records) == ['t1']


def test_patch_boundaries_and_interface_validation():
    before = code(10)
    regions = bindings([dict(role='return',old='return 10')],before)
    assert apply_edits(before,[dict(old='10',new='9')],regions)==code(9)
    with pytest.raises(ValueError,match='outside'):
        apply_edits(before,[dict(old='f(x)',new='g(x)')],regions)
    with pytest.raises(ValueError,match='overlapping'):
        apply_edits(before,[dict(old='return 10',new='return 9'),dict(old='10',new='9')],regions)
    assert unchanged_regions('def f(x):\n    return    10 # comment\n', regions)
    assert not unchanged_regions(code(11),regions)


def test_revision_split_compose_prerequisites_and_import(tmp_path):
    search = make(tmp_path/'a')
    search.run()
    tid = 't1'
    old = deepcopy(search.state['library'][tid]['spec'])
    changed = spec();changed['applicability']='narrow numeric return'
    search._apply_revision(tid,dict(kind='revise',specs=[changed],reason='observed scope'))
    assert search.state['library'][tid]['version']==2
    assert search.state['library'][tid]['versions'][0]['spec']==old
    search._apply_revision(tid,dict(kind='split',specs=[changed,old],reason='two contexts'))
    assert len(search.state['library'])==3
    assert search.state['library']['t2']['family']=='t1'
    search._apply_revision(tid,dict(kind='compose',specs=[old],components=['t1','t2']))
    assert search.state['library']['t4']['components']==['t1','t2']
    invalid=spec();invalid['requires']=['t4']
    with pytest.raises(ValueError,match='contrasting'):
        search._apply_revision('t1',dict(kind='revise',specs=[invalid]))
    invalid['requires']=['t1']
    with pytest.raises(ValueError):
        specification(invalid,search.state['library'],own_id='t1')
    search._export_library()
    imported=make(tmp_path/'b',library_path=str(tmp_path/'a/transformations.json'))
    assert not imported.programs and not imported.state['attempts']
    assert len(imported.state['library'])==4
    assert all(r['external'] for t in imported.state['library'].values() for r in t['records'])


def test_single_transport_attempt_and_counted_failure(tmp_path,monkeypatch):
    search=make(tmp_path)
    def fail(*a,**k):
        raise ConnectionError('unavailable')
    monkeypatch.setattr(search.llm,'draw_sample_with_details',fail)
    result=search.run()
    assert result['status']=='service_unavailable'
    assert result['model_calls']==result['reserved_model_calls']==1
    assert search.state['work']['stage']=='generate'


def test_resume_refuses_different_config(tmp_path):
    search=make(tmp_path)
    search._save()
    with pytest.raises(ValueError,match='configuration'):
        make(tmp_path,open_floor=.5)


def test_pending_sources_produce_one_completed_candidate_and_work_with_readers(tmp_path):
    from experiments.infra.artifacts import load_scored_samples, pick_best_sample
    search = make(tmp_path)
    search.run()
    candidates = [r for r in rows(tmp_path/'events.jsonl') if r['kind'] == 'candidate']
    assert len(candidates) == len(search.programs)
    assert all(r['status'] != 'pending' for r in candidates)
    assert len(list(rows(tmp_path/'programs.jsonl'))) == len(search.programs)
    samples = load_scored_samples(tmp_path)
    assert {s['node_id'] for s in samples} == set(search.archive)
    for limit in (2, 7, 18):
        chosen, _ = pick_best_sample(tmp_path, max_sample_order=limit)
        assert chosen['score'] == min(p.fitness for p in search.archive.values() if p.generation <= limit)


def test_runtime_failure_is_separate_from_performance_and_allocation_counts_it(tmp_path):
    search = make(tmp_path)
    evaluate = search.training.evaluate
    def sometimes_broken(source, pid):
        if search.programs[pid].action != 'Transform':
            return evaluate(source, pid)
        return dict(fitness=None, failure=dict(kind='runtime_error', error='bad indices'),
                    seconds=.01, calls=1, function_seconds=.01, evaluations=[])
    search.training.evaluate = sometimes_broken
    search.run()
    attempts = [a for a in search.state['attempts'] if a['mode'] == 'Transform']
    assert attempts and all(a['outcome'] == 'implementation_failed' for a in attempts)
    assert evidence_rate(attempts) == .5
    weights = distribution(['Transform', 'Open'], attempts, 'mode', 20)
    assert weights[0] < weights[1]


def test_nonmatches_are_reviewed_without_claiming_application(tmp_path):
    model = ScriptedLLM()
    model.no_matches = True
    search = make(tmp_path, llm=model)
    search.run()
    attempts = [a for a in search.state['attempts'] if a['mode']=='Transform']
    assert attempts and all(a['outcome']=='not_applicable' and a['child'] is None for a in attempts)
    reviews = [text for stage, text in model.calls if stage == 'revise']
    assert any('not_applicable' in text for text in reviews)


def test_prerequisite_contrast_and_dependency_cycle(tmp_path):
    search = make(tmp_path)
    a = search._add_transform(spec(), [])
    b = search._add_transform(spec(), [])
    search.state['library'][a]['records'] = [dict(outcome='improved', already_used=[b]),
                                            dict(outcome='no_improvement', already_used=[])]
    changed=spec(); changed['requires']=[b]
    search._apply_revision(a, dict(kind='revise', specs=[changed]))
    assert search.state['library'][a]['spec']['requires'] == [b]
    changed=spec(); changed['requires']=[a]
    with pytest.raises(ValueError, match='cyclic'):
        specification(changed, search.state['library'], own_id=b)


def test_checkpoint_reply_mismatch_is_fatal_not_bad_model_output(tmp_path):
    search = make(tmp_path)
    search._start()
    search.state['reply'] = dict(key='wrong', stage='generate', content='{}', finish_reason='stop')
    with pytest.raises(RuntimeError, match='checkpointed'):
        search.run()
    assert not search.llm.calls and search.state['work']['stage']=='generate'


def test_prompt_never_exposes_evaluator_configuration():
    from traceaad.v13.prompts import node_view, trajectory
    p = node(1)
    p.failure = dict(kind='timeout', n_instances=16, seed=730241, function_seconds_limit=2, seconds=8,
                     error='private solver reference, instance size, seed or count')
    text = json.dumps(dict(node=node_view(p), trajectory=trajectory(p, {1:p}, [], 8)))
    assert 'timeout' in text
    assert all(word not in text for word in ('n_instances', '730241', 'function_seconds', 'private solver'))


@pytest.mark.parametrize('stage', ['generate', 'revise'])
def test_resume_keeps_auxiliary_catalog_stable_with_large_library(tmp_path, monkeypatch, stage):
    def seeded(path):
        search = make(path, retrieval_size=2)
        for i in range(10):
            item=spec(); item['operation'] += f' variant {i}'
            search._add_transform(item, [])
        return search
    control = seeded(tmp_path/'control')
    expected = control.run()
    interrupted = seeded(tmp_path/'interrupted')
    original = interrupted._json
    def stop(reply):
        if reply['stage'] == stage and interrupted.state['work']['parent'] is not None:
            raise KeyboardInterrupt()
        return original(reply)
    monkeypatch.setattr(interrupted, '_json', stop)
    with pytest.raises(KeyboardInterrupt):
        interrupted.run()
    resumed = make(tmp_path/'interrupted', retrieval_size=2)
    actual = resumed.run()
    assert interrupted.llm.calls + resumed.llm.calls == control.llm.calls
    assert actual['model_calls'] == expected['model_calls']
    assert resumed.state['library'] == control.state['library']


def test_resume_repairs_interrupted_final_export_without_model_calls(tmp_path):
    from traceaad.storage import write_json
    search = make(tmp_path)
    expected = search.run()
    summary = dict(expected); summary.pop('reserved_model_calls')
    write_json(tmp_path/'summary.json', summary)
    (tmp_path/'transformations.json').unlink()
    resumed = make(tmp_path)
    actual = resumed.run()
    assert actual['reserved_model_calls'] == expected['reserved_model_calls']
    assert not resumed.llm.calls
    assert read_json(tmp_path/'transformations.json')['library']


def test_partial_usage_is_reported_as_missing(tmp_path):
    search = make(tmp_path)
    call = dict(usage=dict(prompt_tokens=12, completion_tokens=None), error=None)
    search._record_calls([call], dict(action='generate', input_tokens=12))
    assert search.state['missing_usage'] == 1
    assert search.state['measured_input_tokens'] == 12
    assert search.state['measured_output_tokens'] == 0


def test_observed_source_regions_come_from_parent_diff_not_model_copy():
    from traceaad.v13.knowledge import changed_regions
    before = 'def f(x):\n    return 1\n\ndef g(x):\n    return 1\n'
    after = before.replace('def g(x):\n    return 1', 'def g(x):\n    return 2')
    regions = changed_regions(before, after)
    assert all(before.count(r['old']) == 1 for r in regions)
    assert any('def g(x)' in r['old'] and 'return 1' in r['old'] for r in regions)
    insertion = changed_regions(before, before.replace('def g(x):', 'def g(x):\n    x += 1'))
    assert all(before.count(r['old']) == 1 for r in insertion)
    with pytest.raises(ValueError, match='nonempty'):
        changed_regions(before, before)


def test_specification_accepts_text_lists_without_losing_semantic_checks():
    value=spec(); value['dependencies']=['numpy', 'finite weights']; value['checks']=['valid index', 'same interface']
    normalized=specification(value,{})
    assert normalized['dependencies']=='numpy\nfinite weights'
    assert normalized['checks']=='valid index\nsame interface'
    value['checks']=[dict(ambiguous='do not silently stringify unknown structures')]
    with pytest.raises(ValueError, match='computational fields'):
        specification(value,{})
