from evotrace_v3.models import Edge, Node, TreePopulation
from evotrace_v3.reflection import build_variation_prompt, reflect_and_apply
from evotrace_v3.topology import ConsolidationPlan, PathContraction


def add(tree, parent, score, code, op='Refine'):
    nid = tree.latest_node_id + 1
    tree.add(Node(nid, code, f'Design {nid}', score),
             Edge(parent, nid, op if parent else None), evaluations=1)
    return nid


def test_negative_scores_are_compared_by_program_and_persisted_on_the_edge():
    import json
    tree = TreePopulation()
    start = add(tree, 0, -14.848125, 'def solve(x): return x')
    middle = add(tree, start, -14.696875, 'def solve(x): return x + 1', 'Explore')
    end = add(tree, middle, -11.508125, 'def solve(x): return x * 2', 'Explore')
    region = PathContraction(start, end, (middle,))
    prompt = build_variation_prompt(tree, region, task_description='Minimize negative reward')
    assert 'Overall measured outcome: regressed' in prompt
    assert f'Step 1 [Explore] (Intermediate), comparison {start}->{middle}: regressed' in prompt
    assert f'Step 2 [Explore] (Endpoint), comparison {middle}->{end}: regressed' in prompt
    assert f'{start}->{end} [overall path; net development]' in prompt
    class Model:
        def complete(self, *args, **kwargs):
            return json.dumps({'experience': 'The computation changed from identity to multiplication.'})
    new, records = reflect_and_apply(
        tree, ConsolidationPlan((start, end), (region,)), model=Model())
    assert not records[0].used_fallback
    assert new.edges[end].variation_experience == 'The computation changed from identity to multiplication.'
    fact = next(f for f in new.edges[end].variation_observations if f["comparison_id"] == f"{start}->{end}")
    assert (fact['parent_fitness'], fact['fitness']) == (-14.848125, -11.508125)


def test_lineage_subtree_effort_cools_surpassed_ancestor_and_shares_non_improving_effort():
    import pytest
    from evotrace_v3.allocation import lineage_peaks, parent_probabilities
    t = TreePopulation()
    a = add(t, 0, 2, 'def f(): return 2')
    b = add(t, a, 1, 'def f(): return 1')
    c = add(t, 0, 3, 'def f(): return 3')
    assert lineage_peaks(t) == {a: a, b: b, c: c}
    before = parent_probabilities(t)
    add(t, b, None, 'def f(): raise ValueError()')
    after = parent_probabilities(t)
    assert after[a] / after[c] == pytest.approx(before[a] / before[c] * 2 / 3)
    assert after[b] / after[c] == pytest.approx(before[b] / before[c] / 2)
    d = add(t, b, 1.5, 'def f(): return 1.5')
    assert lineage_peaks(t)[d] == b
    final = parent_probabilities(t)
    assert final[b] > final[d] > 0


def test_observations_keep_original_comparison_when_owner_is_removed():
    import json
    from dataclasses import asdict
    from evotrace_v3.reflection import comparison_catalog
    from evotrace_v3.search import deserialize_tree
    from evotrace_v3.topology import BranchConsolidation
    t = TreePopulation()
    a = add(t, 0, -10, 'def f(): return 0')
    b = add(t, a, -8, 'def f(): return 1')
    x = add(t, b, -9, 'def f(): return 2')
    prompts = []
    class Model:
        def complete(self, prompt, **kwargs):
            prompts.append(prompt)
            import re
            return json.dumps({'experience': 'Changed the return value.', 'parent_fitness': 999, 'fitness': 999})
    first = BranchConsolidation(b, (x,))
    t, records = reflect_and_apply(t, ConsolidationPlan((a,b), branch_consolidations=(first,)), model=Model())
    assert not records[0].used_fallback
    fact = t.nodes[b].selection_observations[0]
    assert (fact['parent_id'], fact['child_id'], fact['parent_fitness'], fact['fitness']) == (b,x,-8,-9)
    second = BranchConsolidation(a, (b,))
    t, records = reflect_and_apply(t, ConsolidationPlan((a,), branch_consolidations=(second,)), model=Model())
    assert not records[0].used_fallback
    facts = {f['comparison_id']: f for f in t.nodes[a].selection_observations}
    assert (facts[f'{b}->{x}']['parent_fitness'], facts[f'{b}->{x}']['fitness']) == (-8, -9)
    assert (facts[f'{a}->{b}']['parent_fitness'], facts[f'{a}->{b}']['fitness']) == (-10, -8)
    assert 'Comparison' not in t.nodes[a].selection_experience
    assert f'[Supporting comparisons: {b}->{x}] Changed the return value.' in prompts[-1]
    assert deserialize_tree(json.loads(json.dumps(asdict(t)))) == t


def test_region_sources_are_automatic_and_empty_experience_falls_back():
    import json
    from evotrace_v3.reflection import _bind_experience, VARIATION_SCHEMA
    t=TreePopulation()
    a=add(t,0,10,'def f(): return 0')
    b=add(t,a,15,'def f(): return 1')
    c=add(t,b,12,'def f(): return 2')
    region=PathContraction(a,c,(b,))
    note,facts=_bind_experience({'experience':'A useful algorithmic insight.',
                                 'comparison_ids':['99->100'], 'fitness':-999},t,region)
    assert note=='A useful algorithmic insight.'
    assert {f['comparison_id'] for f in facts}=={f'{a}->{b}',f'{b}->{c}',f'{a}->{c}'}
    assert next(f for f in facts if f['comparison_id']==f'{a}->{c}')['fitness']==12
    assert VARIATION_SCHEMA['json_schema']['schema']['required']==['experience']
    class Empty:
        def complete(self,*args,**kwargs):
            return json.dumps({'experience':'  '})
    new,records=reflect_and_apply(t,ConsolidationPlan((a,c),(region,)),model=Empty())
    assert records[0].used_fallback
    assert 'nonempty' in records[0].error
    assert f'{a}->{c}: 10 -> 12 (regressed)' in new.edges[c].variation_experience
    assert f'{b}->{c}: 15 -> 12 (improved)' in new.edges[c].variation_experience


def test_virtual_root_is_not_a_net_comparison():
    from evotrace_v3.reflection import comparison_catalog, render_observations
    t=TreePopulation()
    a=add(t,0,15,'def f(): return 1')
    b=add(t,a,12,'def f(): return 2')
    region=PathContraction(0,b,(a,))
    catalog,current,net=comparison_catalog(t,region)
    assert net is None and f'0->{b}' not in catalog
    assert f'{a}->{b}' in current
    prompt=build_variation_prompt(t,region,task_description='Minimize')
    assert 'no overall relative outcome exists' in prompt
    observation=dict(catalog[f'0->{a}'],changes='Return a constant.',interpretation='')
    text=render_observations([observation])
    assert f'Evaluation {a}; no parent comparison' in text
    assert 'Comparison 0->' not in text


def test_one_note_can_combine_comparisons_and_outage_keeps_its_sources():
    import json
    from evotrace_v3.topology import BranchConsolidation
    from evotrace_v3.reflection import build_selection_prompt
    t=TreePopulation()
    a=add(t,0,10,'def f(): return 0')
    b=add(t,a,8,'def f(): return 1')
    c=add(t,a,12,'def f(): return 2')
    note='The two changes offered different tradeoffs under the same starting computation.'
    class Model:
        def complete(self,*args,**kwargs):
            return json.dumps(dict(experience=note))
    t,records=reflect_and_apply(t,ConsolidationPlan((a,),branch_consolidations=(BranchConsolidation(a,(b,c)),)),model=Model())
    assert not records[0].used_fallback
    assert t.nodes[a].selection_experience==note
    assert len(t.nodes[a].selection_observations)==2
    d=add(t,a,None,'def f(): raise ValueError()')
    region=BranchConsolidation(a,(d,))
    prompt=build_selection_prompt(t,region,task_description='Minimize')
    assert f'Supporting comparisons: {a}->{b}, {a}->{c}' in prompt
    t,records=reflect_and_apply(t,ConsolidationPlan((a,),branch_consolidations=(region,)),model=None)
    assert records[0].used_fallback and note in t.nodes[a].selection_experience
    assert {o['comparison_id'] for o in t.nodes[a].selection_observations}=={f'{a}->{b}',f'{a}->{c}',f'{a}->{d}'}


def test_reflection_has_same_start_reference_without_champion_code_or_other_experience():
    from evotrace_v3.reflection import build_selection_prompt
    from evotrace_v3.topology import BranchConsolidation
    from dataclasses import asdict
    t=TreePopulation()
    a=add(t,0,10,'def f(): return 0')
    b=add(t,a,12,'def f(): return 1')
    better=add(t,a,8,'def f(): return 12345')
    other=add(t,a,9,'def f(): return 2')
    unrelated=add(t,0,1,'def f(): return 54321')
    t.nodes[better].design='Joint routing and capacity model'
    t.nodes[better].selection_experience='Unrelated notes should stay on their owner.'
    before=asdict(t)
    prompt=build_selection_prompt(t,BranchConsolidation(a,(b,)),task_description='Minimize')
    related=prompt.split('[Related Results]')[1].split('[Earlier Algorithm-Design Experience]')[0].split('[Comparison Catalog]')[0]
    assert 'fitness=8' in related and t.nodes[better].design in related
    assert 'fitness=9' not in related and 'fitness=1;' not in related
    assert '12345' not in related and '54321' not in related
    assert t.nodes[better].selection_experience not in prompt
    assert asdict(t)==before


def test_contracted_side_attempt_keeps_measured_scores_beside_its_diff():
    t=TreePopulation()
    a=add(t,0,10,'def f(): return 0')
    b=add(t,a,11,'def f(): return 1')
    c=add(t,b,8,'def f(): return 2')
    x=add(t,b,12,'def f(): return 3')
    region=PathContraction(a,c,(b,),(x,))
    prompt=build_variation_prompt(t,region,task_description='Minimize')
    assert f'comparison {b}->{x}: 11 -> 12 (regressed)' in prompt
