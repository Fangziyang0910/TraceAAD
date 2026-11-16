from evotrace_v3.models import Edge, Node, TreePopulation
from evotrace_v3.reflection import build_edge_reflection_prompt, build_node_reflection_prompt, reflect_and_apply
from evotrace_v3.topology import PathContraction, SimplificationPlan, SubtreePruning


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
    prompt = build_edge_reflection_prompt(tree, region, task_description='Minimize negative reward')
    assert 'Overall measured outcome: regressed' in prompt
    assert f'Step 1 [Explore] (Intermediate), comparison {start}->{middle}: regressed' in prompt
    assert f'Step 2 [Explore] (Endpoint), comparison {middle}->{end}: regressed' in prompt
    assert f'{start}->{end} [overall path; net development]' in prompt
    class Model:
        def complete(self, *args, **kwargs):
            return json.dumps({'experience': 'The computation changed from identity to multiplication.'})
    new, records = reflect_and_apply(
        tree, SimplificationPlan((start, end), (region,)), model=Model())
    assert not records[0].used_fallback
    assert new.edges[end].experience == 'The computation changed from identity to multiplication.'
    fact = next(f for f in new.edges[end].observations if f["comparison_id"] == f"{start}->{end}")
    assert (fact['parent_fitness'], fact['fitness']) == (-14.848125, -11.508125)


def test_representative_cover_excludes_worse_nearby_offspring_and_resets_direct_effort_on_novel_or_improving_child():
    import pytest
    from evotrace_v3.allocation import parent_probabilities
    from evotrace_v3.topology import representatives
    t = TreePopulation()
    a = add(t, 0, 2.0, 'def f(x): return x + 2')
    b = add(t, a, 1.0, 'def f(x): return x + 1')  # Better nearby tweak covers `a` (d == 0 < 0.15)
    c = add(t, 0, 3.0, 'def f(x): return [y * y for y in range(x) if y % 2 == 0]')
    assert representatives(t) == (b, c)
    before = parent_probabilities(t)
    assert set(before) == {b, c}
    add(t, b, None, 'def f(x): raise ValueError()')
    after = parent_probabilities(t)
    assert after[b] / after[c] == pytest.approx(before[b] / before[c] / 2)
    # Worse nearby tweak under `b` is covered by `b` and does not enter R or invert `b`.
    worse_tweak = add(t, b, 1.5, 'def f(x): return x + 5')
    assert worse_tweak not in parent_probabilities(t)
    # Structurally distinct Explore child under `b` enters R with fresh direct effort = 0.
    novel = add(t, b, 2.5, 'def f(x):\n    acc = 0\n    for k in range(x):\n        acc ^= (k << 1)\n    return acc', 'Explore')
    probs = parent_probabilities(t)
    assert set(probs) == {b, novel, c}
    assert t.nodes[novel].direct_evaluations == 0
    assert probs[novel] > probs[c] > 0


def test_observations_keep_original_comparison_while_reflection_prompt_stays_bounded():
    import json
    from dataclasses import asdict
    from evotrace_v3.search import deserialize_tree
    t = TreePopulation()
    a = add(t, 0, -10, 'def f(): return 0')
    b = add(t, a, -8, 'def f(): return 1')
    x = add(t, b, -9, 'def f(): return 2')
    prompts = []
    class Model:
        def complete(self, prompt, **kwargs):
            prompts.append(prompt)
            return json.dumps({'experience': 'Changed the return value.', 'parent_fitness': 999, 'fitness': 999})
    first = SubtreePruning(b, (x,))
    t, records = reflect_and_apply(t, SimplificationPlan((a, b), subtree_prunings=(first,)), model=Model())
    assert not records[0].used_fallback
    fact = t.nodes[b].observations[0]
    assert (fact['parent_id'], fact['child_id'], fact['parent_fitness'], fact['fitness']) == (b, x, -8, -9)
    second = SubtreePruning(a, (b,))
    t, records = reflect_and_apply(t, SimplificationPlan((a,), subtree_prunings=(second,)), model=Model())
    assert not records[0].used_fallback
    facts = {f['comparison_id']: f for f in t.nodes[a].observations}
    assert (facts[f'{b}->{x}']['parent_fitness'], facts[f'{b}->{x}']['fitness']) == (-8, -9)
    assert (facts[f'{a}->{b}']['parent_fitness'], facts[f'{a}->{b}']['fitness']) == (-10, -8)
    assert 'Comparison' not in t.nodes[a].experience
    assert 'Changed the return value.' in prompts[-1]
    # Historical deleted comparison IDs stay in node.observations on disk, NOT re-dumped into prompt catalog!
    assert f'{b}->{x}' not in prompts[-1]
    assert deserialize_tree(json.loads(json.dumps(asdict(t)))) == t

    # Verify across 12 successive prunings into node `a` that prompt length stays O(current region), not growing!
    lengths = []
    for step in range(12):
        k1 = add(t, a, -7 + step, f'def f(): return {10 + step}')
        k2 = add(t, a, -6 + step, f'def f(): return {30 + step}')
        pruning = SubtreePruning(a, (k1, k2))
        t, recs = reflect_and_apply(t, SimplificationPlan((a,), subtree_prunings=(pruning,)), model=Model())
        lengths.append(recs[0].prompt_chars)
    assert len(t.nodes[a].observations) == 2 + 24
    assert max(lengths) - min(lengths) < 60


def test_region_sources_are_automatic_and_empty_experience_falls_back():
    import json
    from evotrace_v3.reflection import _bind_experience, EDGE_EXPERIENCE_SCHEMA
    t = TreePopulation()
    a = add(t, 0, 10, 'def f(): return 0')
    b = add(t, a, 15, 'def f(): return 1')
    c = add(t, b, 12, 'def f(): return 2')
    region = PathContraction(a, c, (b,))
    note, facts = _bind_experience({'experience': 'A useful algorithmic insight.',
                                    'comparison_ids': ['99->100'], 'fitness': -999}, t, region)
    assert note == 'A useful algorithmic insight.'
    assert {f['comparison_id'] for f in facts} == {f'{a}->{b}', f'{b}->{c}', f'{a}->{c}'}
    assert next(f for f in facts if f['comparison_id'] == f'{a}->{c}')['fitness'] == 12
    assert EDGE_EXPERIENCE_SCHEMA['json_schema']['schema']['required'] == ['experience']
    class Empty:
        def complete(self, *args, **kwargs):
            return json.dumps({'experience': '  '})
    new, records = reflect_and_apply(t, SimplificationPlan((a, c), (region,)), model=Empty())
    assert records[0].used_fallback
    assert 'nonempty' in records[0].error
    assert f'{a}->{c}: 10 -> 12 (regressed)' in new.edges[c].experience
    assert f'{b}->{c}: 15 -> 12 (improved)' in new.edges[c].experience


def test_virtual_root_is_not_a_net_comparison():
    from evotrace_v3.reflection import comparison_catalog, render_observations
    t = TreePopulation()
    a = add(t, 0, 15, 'def f(): return 1')
    b = add(t, a, 12, 'def f(): return 2')
    region = PathContraction(0, b, (a,))
    catalog, current, net = comparison_catalog(t, region)
    assert net is None and f'0->{b}' not in catalog
    assert f'{a}->{b}' in current
    prompt = build_edge_reflection_prompt(t, region, task_description='Minimize')
    assert 'no overall relative outcome exists' in prompt
    observation = dict(catalog[f'0->{a}'])
    text = render_observations([observation])
    assert f'Evaluation {a}; no parent comparison' in text
    assert 'Comparison 0->' not in text


def test_one_note_can_combine_comparisons_and_outage_keeps_its_sources():
    import json
    t = TreePopulation()
    a = add(t, 0, 10, 'def f(): return 0')
    b = add(t, a, 8, 'def f(): return 1')
    c = add(t, a, 12, 'def f(): return 2')
    note = 'The two changes offered different tradeoffs under the same starting computation.'
    class Model:
        def complete(self, *args, **kwargs):
            return json.dumps(dict(experience=note))
    t, records = reflect_and_apply(t, SimplificationPlan((a,), subtree_prunings=(SubtreePruning(a, (b, c)),)), model=Model())
    assert not records[0].used_fallback
    assert t.nodes[a].experience == note
    assert len(t.nodes[a].observations) == 2
    d = add(t, a, None, 'def f(): raise ValueError()')
    region = SubtreePruning(a, (d,))
    prompt = build_node_reflection_prompt(t, region, task_description='Minimize')
    assert note in prompt
    t, records = reflect_and_apply(t, SimplificationPlan((a,), subtree_prunings=(region,)), model=None)
    assert records[0].used_fallback and note in t.nodes[a].experience
    assert {o['comparison_id'] for o in t.nodes[a].observations} == {f'{a}->{b}', f'{a}->{c}', f'{a}->{d}'}


def test_reflection_has_same_start_reference_without_champion_code_or_other_experience():
    from dataclasses import asdict
    t = TreePopulation()
    a = add(t, 0, 10, 'def f(): return 0')
    b = add(t, a, 12, 'def f(): return 1')
    better = add(t, a, 8, 'def f(): return 12345')
    other = add(t, a, 9, 'def f(): return 2')
    unrelated = add(t, 0, 1, 'def f(): return 54321')
    t.nodes[better].design = 'Joint routing and capacity model'
    t.nodes[better].experience = 'Unrelated notes should stay on their owner.'
    before = asdict(t)
    prompt = build_node_reflection_prompt(t, SubtreePruning(a, (b,)), task_description='Minimize')
    related = prompt.split('[Related Results]')[1].split('[Earlier Algorithm-Design Experience]')[0].split('[Comparison Catalog]')[0]
    assert 'fitness=8' in related and t.nodes[better].design in related
    assert 'fitness=9' not in related and 'fitness=1;' not in related
    assert '12345' not in related and '54321' not in related
    assert t.nodes[better].experience not in prompt
    assert asdict(t) == before


def test_contracted_branch_attempt_keeps_measured_scores_beside_its_diff():
    t = TreePopulation()
    a = add(t, 0, 10, 'def f(): return 0')
    b = add(t, a, 11, 'def f(): return 1')
    c = add(t, b, 8, 'def f(): return 2')
    x = add(t, b, 12, 'def f(): return 3')
    region = PathContraction(a, c, (b,), (x,))
    prompt = build_edge_reflection_prompt(t, region, task_description='Minimize')
    assert f'comparison {b}->{x}: 11 -> 12 (regressed)' in prompt
