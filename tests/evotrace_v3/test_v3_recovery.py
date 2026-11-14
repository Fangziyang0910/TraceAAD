from copy import deepcopy
import json
import pytest
from evotrace_v3 import Config, EvoTrace, Task
from evotrace_v3.models import Edge, Node, TreePopulation
from evotrace_v3.reflection import reflect_and_apply
from evotrace_v3.topology import TopologyConfig, plan_consolidation


def add(tree, parent, fit):
    i = tree.latest_node_id + 1
    tree.add(Node(i, f'def solve(x): return x + {i}', f'Idea {i}', fit),
             Edge(parent, i, 'Refine' if parent else None), evaluations=1)
    return i


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


@pytest.mark.parametrize('field, old_value', [
    ('topology', {'branch_width': 8, 'path_length': 4}),
    ('selection_rule', 'route_max'),
    ('selection_rule', 'individual_tree_allocation'),
    ('allocation', {'exploration_weight': 1.0}),
])
def test_resume_matches_uninterrupted_search_and_rejects_old_policy(tmp_path, field, old_value):
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
    data[field] = old_value
    settings.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=field):
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


def test_model_outage_does_not_age_tree_and_reflection_failure_keeps_experience(tmp_path):
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
    assert new.root_selection_experience


