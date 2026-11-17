from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from evotrace_v3 import Config, EvoTrace, Task
from evotrace_v3.models import Edge, Node, TreePopulation
from evotrace_v3.reflection import reflect_and_apply
from evotrace_v3.topology import TopologyConfig, plan_simplification


def add(tree, parent, fit):
    i = max(tree.nodes, default=0) + 1
    tree.add(Node(i, f"def solve(x): return x + {i}", f"Idea {i}", fit),
             Edge(parent, i, "Refine" if parent else None), evaluations=1)
    return i


class Model:
    request_options = {}

    def __init__(self):
        self.state = {"count": 0, "reflections": 0}

    def complete(self, prompt, **kwargs):
        if self.request_options["response_format"]["json_schema"]["name"].startswith("evotrace_"):
            self.state["reflections"] += 1
            return json.dumps({"experience": "Changed the constant offset."})
        self.state["count"] += 1
        return json.dumps({
            "analysis": "Reduce the offset",
            "code": f"def solve(x):\n    return {100 - self.state['count']}\n",
            "design": "Constant offset",
        })


def simplifications(search):
    return sum("aux" in a.record for a in search.attempts)


def make_search(model, path):
    task = Task("Minimize", "def solve(x):\n    pass\n", lambda src, fn: fn(1), 2)
    return EvoTrace(
        task, model, config=Config(initial_size=1, evaluation_budget=24, seed=3),
        topology_config=TopologyConfig(max_representatives=8, cover_distance=0.15, region_size=2),
        run_dir=path,
    )


@pytest.mark.parametrize("field, old_value", [
    ("topology", {"branch_width": 8, "path_length": 4}),
    ("topology", {"quality_slots": 32, "recent_evaluations": 40}),
    ("topology", {"cold_evaluations": 40, "quality_floor": 0.5, "clone_distance": 0.08}),
    ("topology", {"branch_width": 4, "path_depth": 4, "cold_evaluations": 40}),
    ("topology", {"active_evaluations": 40, "region_size": 4}),
    ("mechanisms", ["tree_population", "dynamic_reflective_simplification", "individual_tree_allocation", "multiscale_context"]),
    ("mechanisms", ["tree_population", "reflective_tree_simplification", "individual_tree_allocation", "multiscale_context"]),
    ("mechanisms", ["tree_population", "reflective_tree_simplification", "lineage_effort_allocation", "tree_context"]),
    ("selection_rule", "route_max"),
    ("selection_rule", "rank_decay_subtree_sampling"),
    ("selection_rule", "rank_decay_direct_sampling"),
    ("selection_rule", "rank_decay_lineage_sampling"),
    ("selection_rule", "individual_tree_allocation"),
    ("selection_rule", "rank_subtree_sampling"),
    ("allocation", {"quality_ratio": 10.0}),
    ("allocation", {"exploration_weight": 1.0}),
])
def test_resume_matches_uninterrupted_search_and_rejects_old_policy(tmp_path, field, old_value):
    full = make_search(Model(), tmp_path / "full")
    full.run()
    model = Model()
    search = make_search(model, tmp_path / "resumed")
    for _ in range(14):
        search.step()
    assert simplifications(search) > 0
    before = dict(model.state)
    restored = EvoTrace(
        search.task, model, config=search.config, topology_config=search.topology_config,
        run_dir=tmp_path / "resumed", resume=True,
    )
    assert restored.tree == search.tree and model.state == before
    restored.run()
    assert restored.tree == full.tree
    assert [(a.parent_id, a.reference_id, a.operator, a.record.get("explore_ids")) for a in restored.attempts] == [
        (a.parent_id, a.reference_id, a.operator, a.record.get("explore_ids")) for a in full.attempts
    ]
    settings = tmp_path / "resumed" / "run_config.json"
    data = json.loads(settings.read_text())
    data[field] = old_value
    settings.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=field):
        EvoTrace(
            search.task, model, config=search.config, topology_config=search.topology_config,
            run_dir=tmp_path / "resumed", resume=True,
        )


def test_interruption_before_commit_recovers_only_committed_tree_without_repeating_reflection(tmp_path):
    model = Model()
    search = make_search(model, tmp_path)
    for _ in range(12):
        search.step()
    original = search.store.append

    def append(name, row):
        if name == "attempts.jsonl":
            raise OSError("interrupted before commit")
        return original(name, row)

    search.store.append = append
    before_tree = deepcopy(search.tree)
    before_attempts = len(search.attempts)
    with pytest.raises(OSError):
        search.step()
    calls = dict(model.state)
    restored = EvoTrace(
        search.task, model, config=search.config, topology_config=search.topology_config,
        run_dir=tmp_path, resume=True,
    )
    assert model.state == calls
    assert len(restored.attempts) == before_attempts
    assert restored.tree == before_tree


def test_model_outage_does_not_age_tree_and_reflection_failure_keeps_experience(tmp_path):
    model = Model()
    search = make_search(model, tmp_path)
    search.step()
    before = deepcopy(search.tree)

    def outage(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    search.model.complete = outage
    attempt = search.step()
    assert not attempt.evaluations and search.tree == before
    t = TreePopulation()
    for i in range(5):
        add(t, 0, i + 1)

    class Broken:
        complete = staticmethod(outage)

    t.root_experience = "Earlier root note."
    regions = plan_simplification(t)
    new, records = reflect_and_apply(t, regions, model=Broken(), problem="[Problem]")
    assert records and all(r["error"] for r in records)
    assert new.root_experience == "Earlier root note." and len(new.nodes) < len(t.nodes)


def test_two_hundred_evaluations_resume_with_identical_trees(tmp_path):
    task = Task("Minimize", "def solve(x):\n    pass\n", lambda src, fn: fn(1), 2)
    config = Config(initial_size=1, evaluation_budget=200, seed=3)
    topology = TopologyConfig(max_representatives=8, cover_distance=0.15, region_size=4)

    def make(model, path, resume=False):
        return EvoTrace(task, model, config=config, topology_config=topology, run_dir=path, resume=resume)

    def advance(search, count):
        states = []
        for _ in range(count):
            assert search.step() is not None
            states.append(asdict(search.tree))
        return states

    full = make(Model(), tmp_path / "full200")
    expected = advance(full, 200)
    model = Model()
    interrupted = make(model, tmp_path / "split200")
    assert advance(interrupted, 100) == expected[:100]
    calls = dict(model.state)
    restored = make(model, tmp_path / "split200", resume=True)
    assert model.state == calls
    assert advance(restored, 100) == expected[100:]
    assert restored.evaluations_used == full.evaluations_used == 200
    assert simplifications(restored) == simplifications(full) > 0


def test_explore_programs_start_new_lineages_at_the_root(tmp_path):
    search = make_search(Model(), tmp_path / "run")
    added = []
    attach = search._attach_evaluated_attempt

    def record(attempt):
        attach(attempt)
        if attempt.program_id in search.tree.edges:
            prog = search.archive[attempt.program_id]
            added.append((attempt.operator, attempt.parent_id, prog.parent_id, search.tree.edges[attempt.program_id].parent_id))

    search._attach_evaluated_attempt = record
    search.run()
    operators = {op for op, _, _, _ in added}
    assert {"Explore", "Refine"} <= operators
    for op, parent, prog_parent, attached in added:
        assert parent == prog_parent == attached
        if op in ("Init", "Explore"):
            assert parent == 0
        else:
            assert parent > 0
