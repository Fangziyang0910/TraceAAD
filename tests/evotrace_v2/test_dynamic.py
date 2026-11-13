from copy import deepcopy
from dataclasses import asdict
import json
import random

import pytest

from evotrace_v2 import Config, EvoTrace, Task
from evotrace_v2.allocation import parent_probabilities
from evotrace_v2.context import build_evolution_context
from evotrace_v2.models import Edge, History, Node, TreePopulation
from evotrace_v2.reflection import build_selection_prompt, build_variation_prompt, reflect_and_apply
from evotrace_v2.search import deserialize_tree
from evotrace_v2.topology import TopologyConfig, apply_consolidation, plan_consolidation


def add(tree, parent, fit, *, op='Refine', reference=None, evaluations=1, failures=None):
    i = tree.latest_node_id + 1
    tree.add(Node(i, f'def solve(x):\n    return x + {i}\n', f'Idea {i}', fit,
                  failure='BAD_BRANCH' if fit is None else None),
             Edge(parent, i, None if parent == 0 else op, reference),
             evaluations=evaluations, failures=failures)
    return i


def removed(plan):
    return [i for c in (*plan.path_contractions, *plan.branch_consolidations) for i in c.removed_ids]


def total_history(tree):
    result = tree.root_selection_history
    for i, node in tree.nodes.items():
        result = result + node.own_history + node.selection_history + tree.edges[i].variation_history
    return result


def test_quality_is_selected_once_and_not_recursively_halved():
    t = TreePopulation()
    for i in range(8):
        add(t, 0, i+1)
    t.evaluations_used += 40
    cfg = TopologyConfig(clone_distance=0, same_fitness_clone_distance=0)
    plan = plan_consolidation(t, cfg)
    assert plan.kept_ids == (1, 2, 3, 4)
    assert len(removed(plan)) == len(set(removed(plan))) == 4


def test_recent_birth_and_reference_use_delay_retirement_at_exact_boundary():
    t = TreePopulation()
    a = add(t, 0, 1)
    ref = add(t, 0, 9)
    add(t, a, 2, op='Crossover', reference=ref, evaluations=2, failures=1)
    assert t.evaluations_used == 4
    assert t.nodes[a].last_used == t.nodes[ref].last_used == 4
    assert t.nodes[a].children_count == t.nodes[ref].reference_count == 1
    cfg = TopologyConfig(cold_evaluations=40, quality_floor=1)
    t.evaluations_used = 43
    assert ref in plan_consolidation(t, cfg).kept_ids
    t.evaluations_used = 44
    assert ref in removed(plan_consolidation(t, cfg))
    assert total_history(t) == History(3, 4, 1, 1, 1)


def test_champion_quality_and_required_forks_survive_without_capacity_claim():
    t = TreePopulation()
    for i in range(1, 256):
        add(t, 0 if i == 1 else i//2, i if i >= 128 else 1000+i)
    t.evaluations_used += 40
    cfg = TopologyConfig(clone_distance=0, same_fitness_clone_distance=0)
    plan = plan_consolidation(t, cfg)
    assert len(plan.kept_ids) == 255  # all leaves are representatives; all forks are needed
    assert not plan.triggers


def test_weak_structural_anchor_cannot_exclude_better_nearby_versions():
    t = TreePopulation()
    a = add(t, 0, 100)
    left = add(t, a, 1)
    right = add(t, a, 2)
    t.nodes[right].last_used = t.evaluations_used
    t.nodes[a].last_used = 0
    t.evaluations_used = 40
    plan = plan_consolidation(t)
    assert {a, left, right} <= set(plan.kept_ids)
    if plan.triggers:
        assert left in plan.triggers[0]['protected_ids']


def test_path_and_side_regions_partition_original_tree_and_carry_history():
    t = TreePopulation()
    a = add(t, 0, 10)
    b = add(t, a, 9)
    c = add(t, b, 8)
    d = add(t, c, 1)
    side = add(t, b, None, evaluations=2, failures=2)
    t.nodes[b].selection_experience = 'OLDER_BRANCH_EVIDENCE'
    t.edges[c].variation_experience = 'OLDER_PATH_EVIDENCE'
    t.evaluations_used += 40
    t.nodes[a].last_used = t.nodes[d].last_used = t.evaluations_used
    before = asdict(t)
    plan = plan_consolidation(t, TopologyConfig(quality_floor=1))
    assert plan.kept_ids == (a, d)
    assert plan.path_contractions[0].removed_ids == (b, c)
    assert plan.branch_consolidations[0].removed_ids == (side,)
    new, records = reflect_and_apply(t, plan, model=None)
    assert asdict(t) == before
    assert total_history(new) == total_history(t)
    assert new.edges[d].variation_history == History(2, 2, 0, 3, 0)
    assert new.nodes[a].selection_history == History(1, 2, 2, 0, 0)
    assert new.nodes[a].children_count == 1
    assert 'OLDER_BRANCH_EVIDENCE' in new.edges[d].variation_experience
    assert 'OLDER_PATH_EVIDENCE' in new.edges[d].variation_experience
    assert 'BAD_BRANCH' in new.nodes[a].selection_experience
    assert len(records) == 2
    assert new.nodes[a].last_used == t.nodes[a].last_used
    assert not removed(plan_consolidation(new, TopologyConfig(quality_floor=1)))
    for op, reference in [('Refine', None), ('Explore', None), ('Crossover', a)]:
        prompt, _, _, _ = build_evolution_context(new, d, op, reference)
        assert 'OLDER_BRANCH_EVIDENCE' in prompt or 'OLDER_PATH_EVIDENCE' in prompt


def test_root_absorbs_dead_directions_without_losing_direction_accounting():
    t = TreePopulation()
    a = add(t, 0, 10)
    b = add(t, a, 9, op='Explore')
    c = add(t, 0, 1)
    t.evaluations_used += 40
    plan = plan_consolidation(t, TopologyConfig(quality_floor=1))
    new, _ = reflect_and_apply(t, plan, model=None)
    assert new.nodes.keys() == {c}
    assert new.root_selection_history == History(2, 2, 0, 1, 0)
    assert new.direction_evaluations == {a: 1, b: 1, c: 1}
    assert total_history(new) == total_history(t)
    restored = deserialize_tree(json.loads(json.dumps(asdict(new))))
    assert restored == new


@pytest.mark.parametrize('seed', range(8))
def test_many_growth_compression_rounds_preserve_unique_event_ownership(seed):
    rng = random.Random(seed)
    tree = TreePopulation()
    owners = {('selection', 0): set()}
    events = {}
    cfg = TopologyConfig(cold_evaluations=5)
    saw_path = saw_branch = False
    for k in range(160):
        ids = list(tree.nodes)
        parent = rng.choice(ids) if ids else 0
        op = rng.choice(['Refine', 'Explore', 'Crossover'])
        reference = rng.choice(ids) if ids and op == 'Crossover' else None
        failed = bool(ids) and rng.random() < .12
        n_evals = rng.choice([1, 1, 2])
        n_fail = n_evals if failed else n_evals-1
        nid = add(tree, parent, None if failed else rng.random()*10, op=op,
                  reference=reference, evaluations=n_evals, failures=n_fail)
        owners['node', nid] = set()
        owners['selection', nid] = set()
        owners['edge', nid] = set()
        for role, owner, value in [('version', ('node', nid), History(1, n_evals, n_fail)),
                                  ('parent', ('node', parent), History(parent_uses=1)),
                                  ('reference', ('node', reference), History(reference_uses=1))]:
            if (role == 'parent' and not parent) or (role == 'reference' and reference is None):
                continue
            key = (k, role)
            events[key] = value
            owners[owner].add(key)
        before = deepcopy(tree)
        plan = plan_consolidation(tree, cfg)
        assert set(removed(plan)) == set(tree.nodes) - set(plan.kept_ids)
        assert len(removed(plan)) == len(set(removed(plan)))
        saw_path |= bool(plan.path_contractions)
        saw_branch |= bool(plan.branch_consolidations)
        for region in plan.path_contractions:
            dest = owners['edge', region.child_id]
            for i in region.removed_ids:
                for kind in ['node', 'selection', 'edge']:
                    dest.update(owners.pop((kind, i)))
        for region in plan.branch_consolidations:
            dest = owners['selection', region.parent_id]
            for i in region.removed_ids:
                for kind in ['node', 'selection', 'edge']:
                    dest.update(owners.pop((kind, i)))
        tree = apply_consolidation(tree, plan)
        if parent in tree.nodes:
            assert before.nodes[parent].children_count == tree.nodes[parent].children_count
        observed = {('selection', 0): tree.root_selection_history}
        for i, node in tree.nodes.items():
            observed['node', i] = node.own_history
            observed['selection', i] = node.selection_history
            observed['edge', i] = tree.edges[i].variation_history
            assert tree.edges[i].parent_id == 0 or tree.edges[i].parent_id < i and tree.edges[i].parent_id in tree.nodes
        assert observed.keys() == owners.keys()
        assert sum(map(len, owners.values())) == len(set().union(*owners.values())) == len(events)
        for owner, keys in owners.items():
            expected = History()
            for key in keys:
                expected += events[key]
            assert observed[owner] == expected
        assert sum(tree.direction_evaluations.values()) == tree.evaluations_used
        assert tree == deserialize_tree(json.loads(json.dumps(asdict(tree))))
    assert saw_path and saw_branch


class Model:
    request_options = {}
    def __init__(self):
        self.state = {'count': 0, 'reflections': 0}
    def complete(self, prompt, **kwargs):
        if self.request_options['response_format']['json_schema']['name'].startswith('evotrace_'):
            self.state['reflections'] += 1
            return json.dumps({'experience': 'Earlier offsets had larger scores.'})
        self.state['count'] += 1
        return json.dumps({'analysis': 'Reduce the offset',
                           'code': f"def solve(x):\n    return {100-self.state['count']}\n", 'design': 'Constant offset'})


def make_search(model, path):
    task = Task('Minimize', 'def solve(x):\n    pass\n', lambda src, fn: fn(1), 2)
    return EvoTrace(task, model, config=Config(initial_size=1, evaluation_budget=24, seed=3),
                    topology_config=TopologyConfig(cold_evaluations=2), run_dir=path)


def test_resume_matches_uninterrupted_search_and_rejects_old_policy(tmp_path):
    full = make_search(Model(), tmp_path/'full')
    full.run()
    model = Model()
    search = make_search(model, tmp_path/'resumed')
    for _ in range(14):
        search.step()
    assert search.consolidations_count > 0
    before = dict(model.state)
    restored = EvoTrace(search.task, model, config=search.config, topology_config=search.topology_config,
                        run_dir=tmp_path/'resumed', resume=True)
    assert restored.tree == search.tree and model.state == before
    restored.run()
    assert restored.tree == full.tree
    assert [(a.parent_id, a.reference_id, a.operator) for a in restored.attempts] == [(a.parent_id, a.reference_id, a.operator) for a in full.attempts]
    settings = tmp_path/'resumed'/'run_config.json'
    data = json.loads(settings.read_text())
    data['topology'] = {'branch_width': 8, 'path_length': 4}
    settings.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='topology'):
        EvoTrace(search.task, model, config=search.config, topology_config=search.topology_config,
                 run_dir=tmp_path/'resumed', resume=True)


@pytest.mark.parametrize('phase', ['before_commit', 'after_commit'])
def test_interruption_recovers_only_committed_tree_without_repeating_reflection(tmp_path, phase):
    model = Model()
    search = make_search(model, tmp_path)
    for _ in range(12):
        search.step()
    before_tree = deepcopy(search.tree)
    before_attempts = len(search.attempts)
    if phase == 'before_commit':
        original = search.store.append
        def append(name, row):
            if name == 'attempts.jsonl':
                raise OSError('interrupted before commit')
            return original(name, row)
        search.store.append = append
    else:
        def write(name, row):
            raise OSError('interrupted after commit')
        search.store.write = write
    # A different random allocation may not consolidate at the very next step.
    for _ in range(8):
        before_tree = deepcopy(search.tree)
        before_attempts = len(search.attempts)
        try:
            search.step()
        except OSError:
            break
    else:
        pytest.fail('Expected interruption was not exercised')
    calls = dict(model.state)
    restored = EvoTrace(search.task, model, config=search.config, topology_config=search.topology_config,
                        run_dir=tmp_path, resume=True)
    assert model.state == calls
    assert len(restored.attempts) == before_attempts + (phase == 'after_commit')
    assert restored.tree == (before_tree if phase == 'before_commit' else search.tree)


def test_model_outage_does_not_age_tree_and_reflection_failure_keeps_history(tmp_path):
    model = Model()
    search = make_search(model, tmp_path)
    search.step()
    before = deepcopy(search.tree)
    def outage(*args, **kwargs):
        raise RuntimeError('provider unavailable')
    search.model.complete = outage
    attempt = search.step()
    assert not attempt.evaluations and search.tree == before
    t = TreePopulation()
    for i in range(4):
        add(t, 0, i+1)
    t.evaluations_used += 40
    class Broken:
        complete = staticmethod(outage)
    plan = plan_consolidation(t)
    new, records = reflect_and_apply(t, plan, model=Broken())
    assert records and all(r.used_fallback and r.error for r in records)
    assert total_history(new) == total_history(t)
    assert new.root_selection_experience


def test_history_is_input_evidence_and_does_not_penalize_current_parent():
    t = TreePopulation()
    a = add(t, 0, 1)
    b = add(t, a, 2)
    add(t, b, 3, evaluations=2, failures=1)
    t.evaluations_used += 40
    plan = plan_consolidation(t, TopologyConfig(quality_floor=1))
    prompt = build_selection_prompt(t, plan.branch_consolidations[0], task_description='Minimize')
    assert '3 evaluations including 1 failures' in prompt
    new, _ = reflect_and_apply(t, plan, model=None)
    assert new.nodes[a].children_count == 1
    probs = parent_probabilities(new)
    new.nodes[a].selection_history = History(parent_uses=999)
    assert parent_probabilities(new) == probs


def test_three_round_reflection_reuses_old_notes_and_reaches_all_operators():
    t = TreePopulation()
    a = add(t, 0, 10)
    tip = a
    prompts = []
    class Notes:
        def complete(self, prompt, **kwargs):
            prompts.append(prompt)
            marker = 'PATH' if '[Path Evidence]' in prompt else 'SIDE'
            return json.dumps({'experience': f'{marker}_NOTE_{round_number}'})
    for round_number in range(3):
        mid = add(t, tip, 9)
        tip = add(t, mid, 1/(round_number+1))
        add(t, mid, None)
        t.evaluations_used += 40
        t.nodes[a].last_used = t.nodes[tip].last_used = t.evaluations_used
        plan = plan_consolidation(t, TopologyConfig(quality_floor=1))
        assert plan.path_contractions and plan.branch_consolidations
        prompt_start = len(prompts)
        old_total = total_history(t)
        t, _ = reflect_and_apply(t, plan, model=Notes())
        assert total_history(t) == old_total
        if round_number:
            used = '\n'.join(prompts[prompt_start:])
            assert f'PATH_NOTE_{round_number-1}' in used
            assert f'SIDE_NOTE_{round_number-1}' in used
        for op, reference in [('Refine', None), ('Explore', None), ('Crossover', a)]:
            context, _, _, _ = build_evolution_context(t, tip, op, reference)
            assert f'PATH_NOTE_{round_number}' in context
            # SIDE is at the retained ancestor, available in cross/global views.
            if op != 'Refine':
                assert f'SIDE_NOTE_{round_number}' in context
        t = deserialize_tree(json.loads(json.dumps(asdict(t))))
