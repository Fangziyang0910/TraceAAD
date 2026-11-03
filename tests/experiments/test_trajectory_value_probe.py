"""Guard against false semantic claims and future leakage in the probe."""
from experiments.trajectory_value_probe.features import edge_features,dynamics,node_features


def program(code,pid=1,parent=0,f=10):
    return {'id':pid,'code':code,'parent_id':parent,'reference_id':None,'operator':'Refine',
            'design':'','analysis':'','evaluation':{'status':'success','fitness':f,'failure':None}}


BASE='def solve(adjacency):\n    n = len(adjacency)\n    return list(range(n))\n'
SMALL='def solve(adjacency):\n    n = len(adjacency)\n    def exact():\n        return [0]*n\n    if n <= 20:\n        return exact()\n    return list(range(n))\n'


def test_known_domain_and_unknown_domain_are_distinct():
    assert edge_features(program(BASE),program(SMALL),300)['known_inactive_only']
    assert not edge_features(program(BASE),program(SMALL))['known_inactive_only']
    assert not edge_features(program(BASE),program(SMALL),10)['known_inactive_only']


def test_reassigned_size_is_not_pruned():
    changed=SMALL.replace('    def exact():','    n = 10\n    def exact():')
    assert not edge_features(program(BASE),program(changed),300)['known_inactive_only']


def test_unused_helper_default_side_effect_is_preserved():
    changed=SMALL.replace('def exact():','def exact(x=side_effect()):')
    assert not edge_features(program(BASE),program(changed),300)['known_inactive_only']


def test_dynamic_scope_disables_size_inference():
    changed=SMALL.replace('    n = len(adjacency)','    globals()\n    n = len(adjacency)')
    assert not edge_features(program(BASE),program(changed),300)['known_inactive_only']


def test_numeric_change_not_treated_as_copy():
    a=program('def f():\n return 100\n');b=program('def f():\n return 150\n')
    e=edge_features(a,b)
    assert e['numeric_only'] and not e['same_ast'] and not e['known_inactive_only']


def test_trajectory_keeps_trend_and_plateau_separate():
    d=dynamics([12,11,10,10])
    assert d['slope_2']==0 and d['slope_4']>0 and d['trailing_flat']==1


def test_no_children_is_unknown_not_failed():
    p=program(BASE)
    f=node_features(1,{1:p},[],{}, {})
    assert f['coverage']['all']['attempts']==0
    assert f['coverage']['all']['effective_subtree_diversity'] is None
    assert f['dynamics']['slope_2'] is None


def test_request_cap_repair_cost_and_explore_continuation(tmp_path,monkeypatch):
    import ast,json
    from contextlib import nullcontext
    from types import SimpleNamespace
    from experiments.trajectory_value_probe import continue_probe as runner
    from experiments.traceaad_base import scheduling
    from traceaad_base import Task
    from traceaad_base.models import Evaluation
    replies=iter(['not json']+[json.dumps({'analysis':'a','design':'d','code':f'def solve():\n return {n}\n'}) for n in (12,11,9,8,7)])
    class Model:
        model='stub'
        def __init__(self,*a):pass
        def complete(self,prompt,**kw):return next(replies)
    class Evaluator:
        last_dispatch={}
        def __init__(self,*a):pass
        def evaluation_context(self):return nullcontext()
        def evaluate(self,code):return Evaluation('success',float(ast.parse(code).body[0].body[0].value.value))
    monkeypatch.setattr(scheduling,'ScheduledModel',Model)
    monkeypatch.setattr(scheduling,'ScheduledEvaluator',Evaluator)
    task=Task('toy','def solve():\n return 0\n',lambda *a:0,1)
    monkeypatch.setattr(runner,'setup',lambda *a:task)
    job={'job_id':'toy','candidate_key':'c','first_operator':'Explore','repeat':0,'request_budget':6}
    candidate={'candidate_key':'c','prefix_file':'prefix.json','task':'toy','node_id':1,'role':'target'}
    (tmp_path/'candidates.locked.json').write_text(json.dumps({'jobs':[job],'candidates':[candidate]}))
    (tmp_path/'prefix.json').write_text(json.dumps({'archive':[program('def solve():\n return 10\n')]}))
    (tmp_path/'pilot').mkdir();(tmp_path/'pilot/preflight.json').write_text(json.dumps({'results':[{'candidate_key':'c','matches':True}]}))
    args=SimpleNamespace(data=tmp_path,job='toy',base_root=tmp_path,socket='unused')
    runner.run_job(args)
    result=json.loads((tmp_path/'pilot/jobs/toy/result.json').read_text())
    state=json.loads((tmp_path/'pilot/jobs/toy/native/state.json').read_text())
    assert result['requests']==6 and result['evaluations']==5
    assert result['checkpoints'][1]['best_fitness']==10
    assert state['attempts'][0]['repair_used']
    assert state['attempts'][1]['parent_id']==state['attempts'][0]['program_id']
    assert result['best_fitness']==7
def test_changed_statement_probe_counts_body_not_function_declaration():
    from experiments.trajectory_value_probe.output_probe import changed_statements
    parent='def solve(x):\n    return x\n'
    child='def solve(x):\n    def unused():\n        return x + 1\n    return x\n'
    assert changed_statements(parent,child)==[3]


def test_output_equality_compares_all_matrix_cells():
    import numpy as np
    from experiments.trajectory_value_probe.output_probe import equal
    assert equal((np.eye(2),[0,1]),(np.eye(2),np.array([0,1])))
    assert not equal(np.eye(2),np.array([[1.,0.],[.000001,1.]]))

