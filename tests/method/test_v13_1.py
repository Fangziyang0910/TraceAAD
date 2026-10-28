import json
import math
import re
from copy import deepcopy
from pathlib import Path
import pytest

from tests.support import TinyEvaluation
from tests.method.test_v13 import Evaluator, code, node
from traceaad.storage import read_json
from traceaad.v13_1 import Config, TraceAADV131
from traceaad.v13_1 import models
from traceaad.v13_1.knowledge import comparable, specification, identity


def spec():
    return dict(name='decrease one',applicability='numeric return',operation='subtract exactly one from return',
        dependencies='none',parameters='one',preserve='interface',checks='return decreases one',requires=[],
        binding_policy='bind numeric return as role result',conditions=[dict(name='positive',kind='static',
            predicate='returned constant is positive',rationale='positive and negative backgrounds may differ')])


class Model:
    model='fixture'
    def __init__(self): self.calls=[]; self.reject=False
    def count_prompt_tokens(self,text): return len(text)//4
    count_tokens=count_prompt_tokens
    def draw_sample_with_details(self,text,**kwargs):
        stage=re.search(r'\[V13.1:(\w+)\]',text)[1]; data=json.loads(text.rsplit('\n',1)[1]); self.calls.append((stage,text))
        if stage=='generate':
            parent=data.get('CURRENT_PARENT')
            value=int(parent['code'].split('return ')[-1])-1 if parent else 100+len(data['context'][0]['existing_roots'])*10
            out=dict(code=code(value),design='offset',analysis='one-step')
        elif stage=='learn':
            out=dict(kind='attach' if data['library'] else 'new',spec=spec(),id='t1',reason='exact same operation')
        elif stage=='match':
            before=data['CURRENT_PARENT']['code']
            out=dict(matches=[dict(id=t['id'],bindings=[dict(role='result',old=before)],symbols={'function':'f','input':'x'},
                conditions=[dict(value=True,evidence=[before])]) for t in data['candidates']],rejections={})
        elif stage=='edit':
            before=data['CURRENT_PARENT']['code']; value=int(before.split('return ')[-1])-1
            out=dict(edits=[dict(old=before,new=code(value))],design='offset',analysis='frozen one')
        elif stage=='audit':
            out=dict(operation_ok=not self.reject or 'purpose' in data,dependencies_ok=True,preservation_ok=True,parameters_ok=True,
                     summary='subtract one',reason='fixture')
        else: raise AssertionError(stage)
        return dict(content=json.dumps(out),finish_reason='stop',usage=dict(prompt_tokens=20,completion_tokens=10))
    def close(self): pass


def make(path,**kw):
    task=TinyEvaluation(); task.template_program=code(0)
    search=TraceAADV131(evaluation=task,llm=Model(),run_dir=path,task='fixture',
        config=Config(**dict(budget=35,roots=2,warmup=2,**kw)))
    search.training=Evaluator()
    return search


def test_end_to_end_and_completed_resume(tmp_path):
    s=make(tmp_path); result=s.run()
    assert result['status']=='finished' and result['budget_used']==35
    assert result['model_calls']==result['reserved_model_calls']==len(s.llm.calls)
    assert {'generate','learn','match','edit','audit'}<=set(result['stages'])
    assert 'describe' not in result['stages'] and 'revise' not in result['stages']
    assert len(s.state['library'])==1
    assert s.state['prior'] and s.state['scale']>0
    t=s.state['library']['t1']; assert t['records']
    assert all(r['source'] in ('ordinary_replay','diagnostic_replay') for r in t['records'])
    assert all(r['attempt']!=t['discovery']['attempt'] for r in t['records'])
    assert any(a.get('cache_hit') for a in s.state['attempts'])
    assert any(a['mode']=='Transform' for a in s.state['attempts'])
    assert read_json(tmp_path/'transformations.json')['schema']=='13.1'
    again=make(tmp_path); assert again.run()==result and not again.llm.calls


@pytest.mark.parametrize('stage',['generate','learn','match','edit','audit'])
def test_every_model_stage_resumes_exactly(tmp_path,monkeypatch,stage):
    baseline=make(tmp_path/'base'); baseline.run()
    s=make(tmp_path/'cut'); original=s._json
    def interrupt(reply):
        if reply['stage']==stage: raise KeyboardInterrupt()
        return original(reply)
    monkeypatch.setattr(s,'_json',interrupt)
    with pytest.raises(KeyboardInterrupt): s.run()
    resumed=make(tmp_path/'cut'); resumed.run()
    assert s.llm.calls+resumed.llm.calls==baseline.llm.calls
    assert resumed.state['library']==baseline.state['library']
    assert resumed.state['attempts']==baseline.state['attempts']


def test_predictive_math_and_unknown_condition():
    prior=[1.,1.,1.,1.]
    records=[dict(credit=True,A=True,Y=True,Z=2.,conditions=[True],parent=1),
             dict(credit=True,A=True,Y=False,Z=0.,conditions=[False],parent=2),
             dict(credit=True,A=False,Y=False,Z=0.,conditions=[True],parent=1),
             dict(credit=False,A=True,Y=True,Z=100.,conditions=[True],parent=1)]
    c=models.cell(prior,records)
    assert c==[3.,2.,2.,2.,4.,4.]
    assert models.mean(c)==pytest.approx(.4)
    t=dict(records=records); programs={1:node(1),2:node(2,1)}
    pred=models.predictions(t,prior,[None],2,programs,.5)
    assert pred[0]==pred[1]
    assert models.disagreement([c,c],[.5,.5])==0
    assert models.log_predictive(c,dict(A=False))==pytest.approx(math.log(2/5))
    assert models.weights([0,-1000])==[1.,0.]


def test_pair_mapping_and_nonadjacent_backgrounds():
    q=node(1,text='def f(x):\n    return x\n')
    p=node(2,1,text='def f(x):\n    y = 0\n    return x\n')
    b=[dict(role='result',old='return x')]
    assert comparable(1,2,b,b,{1:q,2:p})
    p.code='def g(x):\n    return x\n'
    assert not comparable(1,2,b,b,{1:q,2:p})


def test_unchanged_known_conditions_can_enter_pair_and_forward_score_only(tmp_path):
    s=make(tmp_path);s.run()
    t=s.state['library']['t1']
    old=next(r for r in t['records'] if r['A'] and r['credit'])
    parent=s.programs[old['parent']]
    new=node(max(s.programs)+1,parent.id,text=parent.code.replace('return ','return 1 + '))
    s.programs[new.id]=new
    match=dict(bindings=[dict(role='result',old=new.code)],conditions=old['conditions'])
    pairs=s._pairs(new.id,'t1',match)
    assert pairs and all(not p['changed'] for p in pairs)
    c=models.cell([1.,1.,1.,1.],[])
    assert models.disagreement([c,c],[.5,.5])==0  # no retrospective q prediction enters this function


def test_failed_audit_one_repair_no_performance_credit(tmp_path):
    s=make(tmp_path);s.llm.reject=True;s.run()
    attempted=[a for a in s.state['attempts'] if a['mode']=='Transform']
    assert attempted and all(a['repair_count']==1 for a in attempted)
    assert all(a['outcome']=='implementation_failed' for a in attempted)
    records=s.state['library']['t1']['records']
    assert records and all(not r['A'] for r in records)
    assert all(not r['Y'] and r['Z']==0 for r in records)


def test_cache_not_independent_and_discovery_not_validation(tmp_path):
    s=make(tmp_path);s.run()
    for t in s.state['library'].values():
        assert all(r['attempt']!=t['discovery']['attempt'] for r in t['records'])
        assert any(not r['credit'] for r in t['records'])
    assert len(s.training.calls)==len(s.state['evaluation_cache'])


def test_frozen_library_import_and_configuration_guard(tmp_path):
    s=make(tmp_path/'a');s.run()
    imported=make(tmp_path/'b',library_path=str(tmp_path/'a/transformations.json'))
    assert imported.state['library'] and not imported.programs
    assert all(r['external'] for t in imported.state['library'].values() for r in t['records'])
    with pytest.raises(ValueError,match='configuration'):
        make(tmp_path/'a',lineage_decay=.4)
    value=spec();value['parameters']='two'
    assert identity(specification(value))!=identity(specification(spec()))


def test_entry_preview(capsys):
    from experiments.traceaad_v13_1.run import main
    main(['--task','tsp_construct','--dry-run'])
    value=json.loads(capsys.readouterr().out)
    assert value['method']=='traceaad_v13_1' and value['config']['warmup']==16
    assert value['config']['free_per_ten']==2


def test_template_requires_three_examples_then_resets_executor_credit(tmp_path):
    from traceaad.v13_1.knowledge import token_template,render_template,canonical
    s=make(tmp_path);s.run();t=s.state['library']['t1']
    before='def f(x):\n    return x + 1\n';after=before.replace('+ 1','- 1')
    from traceaad.v13.knowledge import bindings
    regions=bindings([dict(role='result',old=before)],before)
    edits=[dict(old='x + 1',new='x - 1')]
    symbols={'input':'x'}
    template=token_template(edits,symbols)
    rendered,_=render_template(template,symbols,before,regions)
    assert canonical(rendered)==canonical(after)
    examples=[]
    for i in range(3):
        a=max(s.programs)+1;b=a+1
        s.programs[a]=node(a,text=before);s.programs[b]=node(b,a,text=after)
        examples.append(dict(parent=a,child=b,edits=edits,symbols=symbols,bindings=regions))
    t['examples']=examples;t['template']=None;old_version=t['version']
    s._compile_template(t)
    assert t['template'] and t['version']==old_version+1
    assert t['records']==[] and t['versions'][-1]['records']


def test_budget_abort_and_evaluation_receipt_resume(tmp_path,monkeypatch):
    s=make(tmp_path/'budget',max_model_calls=2);result=s.run()
    assert result['reserved_model_calls']==2 and result['stop_reason']=='model_call_budget'
    s=make(tmp_path/'cut');orig=s._advance
    def cut(stage,**record):
        if stage=='describe' and s.state['work']['stage']=='evaluate': raise KeyboardInterrupt()
        return orig(stage,**record)
    monkeypatch.setattr(s,'_advance',cut)
    with pytest.raises(KeyboardInterrupt): s.run()
    assert len(s.training.calls)==1
    resumed=make(tmp_path/'cut');resumed.run()
    assert 1 not in resumed.training.calls


def test_named_parameter_object_preserves_fixed_semantic_values():
    value=spec();value['parameters']={'slope':.01,'candidate_source':'unvisited_nodes[:K]'}
    result=specification(value)
    assert json.loads(result['parameters'])==value['parameters']
    value['parameters']['slope']=float('nan')
    with pytest.raises(ValueError): specification(value)


def test_incomplete_discovery_is_rewritten_before_admission(tmp_path):
    s=make(tmp_path);call=s.llm.draw_sample_with_details;seen=0
    def reject_first_source(text,**kwargs):
        nonlocal seen
        result=call(text,**kwargs)
        if '[V13.1:audit]' in text and 'Discovery completeness:' in text:
            seen+=1
            if seen==1:
                r=json.loads(result['content']);r['parameters_ok']=False;r['reason']='missing cooperating parameter change'
                result['content']=json.dumps(r)
        return result
    s.llm.draw_sample_with_details=reject_first_source
    s.run()
    retried=[a for a in s.state['attempts'] if a.get('induction_retries')]
    assert retried and retried[0]['induction_retries']==1
    assert s.programs[retried[0]['child']].valid
    assert s.state['library']


@pytest.mark.parametrize('bad_match', ['t1', None, 42])
def test_malformed_match_item_records_failure_and_continues(tmp_path,bad_match):
    search=make(tmp_path)
    original=search.llm.draw_sample_with_details
    def malformed(text,**kwargs):
        reply=original(text,**kwargs)
        if '[V13.1:match]' in text:
            value=json.loads(reply['content']);value['matches']=[bad_match]
            reply['content']=json.dumps(value)
        return reply
    search.llm.draw_sample_with_details=malformed
    result=search.run()
    assert result['status']=='finished' and result['budget_used']==35
    errors=[e for a in search.state['attempts'] for e in a.get('stage_errors',[]) if e['stage']=='match']
    assert errors and all(e['error']=='each match must be an object' for e in errors)
    assert all(not t['records'] for t in search.state['library'].values())
    assert result['reserved_model_calls']==len(search.llm.calls)


def test_cached_malformed_match_is_consumed_without_another_request(tmp_path,monkeypatch):
    search=make(tmp_path)
    original=search.llm.draw_sample_with_details
    def malformed(text,**kwargs):
        reply=original(text,**kwargs)
        if '[V13.1:match]' in text: reply['content']=json.dumps(dict(matches=['t1'],rejections={}))
        return reply
    search.llm.draw_sample_with_details=malformed
    parse=search._json
    def interrupt(reply):
        if reply['stage']=='match': raise KeyboardInterrupt()
        return parse(reply)
    monkeypatch.setattr(search,'_json',interrupt)
    with pytest.raises(KeyboardInterrupt): search.run()
    resumed=make(tmp_path);requests=resumed.state['requests'];work=resumed.state['work']
    index=work['match_index'];batch=work['current_match_batch']
    with pytest.raises(ValueError,match='each match must be an object') as exc:
        resumed._match(work)
    resumed._stage_error(exc.value)
    assert resumed.state['requests']==requests and not resumed.llm.calls
    assert work['match_index']==index+batch and resumed.state['reply'] is None
    assert all(not t['records'] for t in resumed.state['library'].values())
    assert resumed.run()['status']=='finished'


@pytest.mark.parametrize('status',[500,429])
def test_service_retry_is_checkpointed_and_budgeted(tmp_path,monkeypatch,status):
    import time
    search=make(tmp_path,max_model_calls=3)
    original=search.llm.draw_sample_with_details
    requests=[]
    class ServiceError(RuntimeError):
        status_code=status
    def fail_once(text,**kwargs):
        requests.append(kwargs)
        if len(requests)==1: raise ServiceError('Internal server error')
        return original(text,**kwargs)
    search.llm.draw_sample_with_details=fail_once
    monkeypatch.setattr(time,'sleep',lambda _:None)
    reply=search._ask('generate',dict(mode='Init'),[dict(existing_roots=[])])
    assert reply['code']
    assert search.state['requests']==search.progress.model_calls==2
    assert search.progress.service_failures==1
    assert requests[0]['response_format']=={'type':'json_object'}
    assert requests[1]==requests[0]
    calls=[json.loads(line) for line in (tmp_path/'calls.jsonl').read_text().splitlines()]
    assert len(calls)==2 and calls[0]['error'] and not calls[1]['error']
    resumed=make(tmp_path,max_model_calls=3)
    assert resumed._ask('generate',dict(mode='Init'),[dict(existing_roots=[])])==reply
    assert not resumed.llm.calls


def test_repeated_service_failures_cannot_exceed_call_budget(tmp_path,monkeypatch):
    import time
    search=make(tmp_path,max_model_calls=3)
    class ServiceError(RuntimeError):
        status_code=500
    def fail(*args,**kwargs): raise ServiceError('Internal server error')
    search.llm.draw_sample_with_details=fail
    monkeypatch.setattr(time,'sleep',lambda _:None)
    result=search.run()
    assert result['reserved_model_calls']==result['model_calls']==3
    assert result['stop_reason']=='model_call_budget'
    from traceaad.storage import rows
    assert len(list(rows(tmp_path/'calls.jsonl.gz')))==3


def test_permanent_service_error_is_not_retried(tmp_path):
    from core.llm import ModelCallError
    search=make(tmp_path)
    class ServiceError(RuntimeError):
        status_code=401
    def fail(*args,**kwargs): raise ServiceError('unauthorized')
    search.llm.draw_sample_with_details=fail
    with pytest.raises(ModelCallError):
        search._ask('generate',dict(mode='Init'),[dict(existing_roots=[])])
    assert search.state['requests']==search.progress.model_calls==1


def test_service_retry_stops_after_backoff_window_with_checkpoint(tmp_path,monkeypatch):
    import time
    from core.llm import ModelCallError, SERVICE_WAIT_SECONDS
    search=make(tmp_path,max_model_calls=100)
    class ServiceError(RuntimeError):
        status_code=500
    def fail(*args,**kwargs): raise ServiceError('Internal server error')
    search.llm.draw_sample_with_details=fail
    pauses=[]
    monkeypatch.setattr(time,'sleep',pauses.append)
    with pytest.raises(ModelCallError):
        search._ask('generate',dict(mode='Init'),[dict(existing_roots=[])])
    assert sum(pauses)>=SERVICE_WAIT_SECONDS and max(pauses)<=60
    assert search.state['requests']==search.progress.model_calls==len(pauses)+1
    resumed=make(tmp_path,max_model_calls=100)
    assert resumed.state['requests']==search.state['requests']
    assert resumed.state['inflight'] is None and resumed.state['reply'] is None
