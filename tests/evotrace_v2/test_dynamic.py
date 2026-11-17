from copy import deepcopy
from dataclasses import asdict
import json
import random

import pytest

from evotrace_v2 import Config, EvoTrace, Task
from evotrace_v2.allocation import parent_probabilities
from evotrace_v2.context import build_evolution_context
from evotrace_v2.models import Edge, Node, TreePopulation
from evotrace_v2.reflection import build_selection_prompt, build_variation_prompt, reflect_and_apply
from evotrace_v2.topology import PathContraction, TopologyConfig, apply_consolidation, plan_consolidation, quality_representatives


def add(tree, parent, fit, *, op="Refine", reference=None, evaluations=1, nid=None):
    i = nid if nid is not None else max(tree.nodes, default=0) + 1
    tree.add(
        Node(i, f"def solve(x):\n    return x + {i}\n", f"Idea {i}", fit, failure="BAD_BRANCH" if fit is None else None),
        Edge(parent, i, None if parent == 0 else op, reference),
        evaluations=evaluations,
    )
    return i


def removed(regions):
    return [
        i for r in regions
        for i in ((*r.removed_ids, *r.side_ids) if isinstance(r, PathContraction) else r.removed_ids)
    ]


def test_quality_is_selected_once_and_not_recursively_halved():
    t = TreePopulation()
    for i in range(8):
        add(t, 0, i + 1)
    t.evaluations_used += 40
    cfg = TopologyConfig(clone_distance=0, same_fitness_clone_distance=0)
    regions = plan_consolidation(t, cfg)
    after = apply_consolidation(t, regions)
    assert tuple(sorted(after.nodes)) == (1, 2, 3, 4)
    assert len(removed(regions)) == len(set(removed(regions))) == 4


def test_explore_and_reference_do_not_refresh_parent_or_reference_clock():
    t = TreePopulation()
    a = add(t, 0, 1)
    ref = add(t, 0, 9)
    add(t, a, 2, op="Crossover", reference=ref, evaluations=2)
    assert t.evaluations_used == 4
    assert t.nodes[a].last_used == 4 and t.nodes[ref].last_used == 2
    assert t.nodes[a].children_count == 1 and t.nodes[ref].children_count == 0
    exp_child = add(t, a, 8, op="Explore", evaluations=1)
    assert t.evaluations_used == 5
    assert t.nodes[a].last_used == 4 and t.nodes[a].children_count == 1
    assert t.nodes[exp_child].last_used == 5 and t.direction_of(exp_child) == exp_child
    cfg = TopologyConfig(cold_evaluations=40, quality_floor=1)
    t.evaluations_used = 41
    assert ref not in removed(plan_consolidation(t, cfg))
    t.evaluations_used = 42
    assert ref in removed(plan_consolidation(t, cfg))
    add(t, a, 3, op="Refine", evaluations=1)
    t.evaluations_used = 45
    regions = plan_consolidation(t, cfg)
    assert exp_child in removed(regions) and a not in removed(regions)


def test_champion_quality_and_required_forks_survive_without_capacity_claim():
    t = TreePopulation()
    for i in range(1, 256):
        add(t, 0 if i == 1 else i // 2, i if i >= 128 else 1000 + i)
    t.evaluations_used += 40
    cfg = TopologyConfig(clone_distance=0, same_fitness_clone_distance=0)
    assert plan_consolidation(t, cfg) == ()


def test_weak_structural_anchor_cannot_exclude_better_nearby_versions():
    t = TreePopulation()
    a = add(t, 0, 100)
    left = add(t, a, 1)
    right = add(t, a, 2)
    t.nodes[right].last_used = t.evaluations_used
    t.nodes[a].last_used = 0
    t.evaluations_used = 40
    assert left in quality_representatives(t)
    after = apply_consolidation(t, plan_consolidation(t))
    assert {a, left, right} <= set(after.nodes)


def test_path_and_side_regions_partition_original_tree_and_carry_experience():
    t = TreePopulation()
    a = add(t, 0, 10)
    b = add(t, a, 9)
    c = add(t, b, 8)
    d = add(t, c, 1)
    side = add(t, b, None, evaluations=2)
    other = add(t, a, 5)
    t.nodes[b].selection_experience = "OLDER_BRANCH_EVIDENCE"
    t.edges[c].variation_experience = "OLDER_PATH_EVIDENCE"
    t.evaluations_used += 40
    t.nodes[other].last_used = t.nodes[d].last_used = t.evaluations_used
    before = asdict(t)
    regions = plan_consolidation(t, TopologyConfig(quality_floor=1))
    (contr,) = regions
    assert contr.removed_ids == (b, c) and contr.side_ids == (side,)
    new, _ = reflect_and_apply(t, regions, model=None)
    assert asdict(t) == before
    assert tuple(sorted(new.nodes)) == (a, d, other)
    assert new.nodes[a].children_count == 2
    assert "OLDER_BRANCH_EVIDENCE" in new.edges[d].variation_experience
    assert "OLDER_PATH_EVIDENCE" in new.edges[d].variation_experience
    assert "BAD_BRANCH" in new.edges[d].variation_experience
    assert new.nodes[a].selection_experience == ""
    assert new.nodes[a].last_used == t.nodes[a].last_used
    assert plan_consolidation(new, TopologyConfig(quality_floor=1)) == ()
    for op, reference in [("Refine", None), ("Explore", None), ("Crossover", a)]:
        prompt, _, _ = build_evolution_context(new, d, op, reference)
        assert "OLDER_BRANCH_EVIDENCE" in prompt or "OLDER_PATH_EVIDENCE" in prompt
        assert "BAD_BRANCH" in prompt


def test_root_absorbs_dead_directions_and_roundtrips_snapshot():
    t = TreePopulation()
    a = add(t, 0, 10)
    add(t, a, 9, op="Explore")
    c = add(t, 0, 1)
    t.evaluations_used += 40
    regions = plan_consolidation(t, TopologyConfig(quality_floor=1))
    new, _ = reflect_and_apply(t, regions, model=None)
    assert new.nodes.keys() == {c} and new.root_selection_experience
    restored = TreePopulation.from_dict(json.loads(json.dumps(asdict(new))))
    assert restored == new


@pytest.mark.parametrize("seed", range(8))
def test_many_growth_compression_rounds_keep_a_valid_tree(seed):
    rng = random.Random(seed)
    tree = TreePopulation()
    cfg = TopologyConfig(cold_evaluations=5)
    saw_path = saw_branch = False
    for k in range(160):
        ids = list(tree.nodes)
        parent = rng.choice(ids) if ids else 0
        op = rng.choice(["Refine", "Explore", "Crossover"])
        reference = rng.choice(ids) if ids and op == "Crossover" else None
        failed = bool(ids) and rng.random() < 0.12
        n_evals = rng.choice([1, 1, 2])
        add(tree, parent, None if failed else rng.random() * 10, op=op, reference=reference, evaluations=n_evals, nid=k + 1)
        before = deepcopy(tree)
        regions = plan_consolidation(tree, cfg)
        rem = removed(regions)
        assert len(rem) == len(set(rem))
        saw_path |= any(isinstance(r, PathContraction) for r in regions)
        saw_branch |= any(not isinstance(r, PathContraction) for r in regions)
        tree = apply_consolidation(tree, regions)
        assert set(rem) == set(before.nodes) - set(tree.nodes)
        if parent in tree.nodes:
            assert before.nodes[parent].children_count == tree.nodes[parent].children_count
        for i in tree.nodes:
            assert tree.edges[i].parent_id == 0 or (tree.edges[i].parent_id < i and tree.edges[i].parent_id in tree.nodes)
        assert tree == TreePopulation.from_dict(json.loads(json.dumps(asdict(tree))))
    assert saw_path and saw_branch


class Model:
    request_options = {}

    def __init__(self):
        self.state = {"count": 0, "reflections": 0}

    def complete(self, prompt, **kwargs):
        if self.request_options["response_format"]["json_schema"]["name"].startswith("evotrace_"):
            self.state["reflections"] += 1
            return json.dumps({"experience": "Earlier offsets had larger scores."})
        self.state["count"] += 1
        return json.dumps({
            "analysis": "Reduce the offset",
            "code": f"def solve(x):\n    return {100 - self.state['count']}\n",
            "design": "Constant offset",
        })


def make_search(model, path):
    task = Task("Minimize", "def solve(x):\n    pass\n", lambda src, fn: fn(1), 2)
    return EvoTrace(
        task, model, config=Config(initial_size=1, evaluation_budget=24, seed=3),
        topology_config=TopologyConfig(cold_evaluations=2), run_dir=path,
    )


@pytest.mark.parametrize("field, old_value", [
    ("topology", {"branch_width": 8, "path_length": 4}),
    ("selection_rule", "route_max"),
])
def test_resume_matches_uninterrupted_search_and_rejects_old_policy(tmp_path, field, old_value):
    full = make_search(Model(), tmp_path / "full")
    full.run()
    model = Model()
    search = make_search(model, tmp_path / "resumed")
    for _ in range(14):
        search.step()
    assert any("aux" in a.record for a in search.attempts)
    before = dict(model.state)
    restored = EvoTrace(
        search.task, model, config=search.config, topology_config=search.topology_config,
        run_dir=tmp_path / "resumed", resume=True,
    )
    assert restored.tree == search.tree and model.state == before
    restored.run()
    assert restored.tree == full.tree
    assert [(a.parent_id, a.reference_id, a.operator) for a in restored.attempts] == [
        (a.parent_id, a.reference_id, a.operator) for a in full.attempts
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
    assert model.state == calls and len(restored.attempts) == before_attempts and restored.tree == before_tree


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
    for i in range(4):
        add(t, 0, i + 1)
    t.evaluations_used += 40

    class Broken:
        complete = staticmethod(outage)

    regions = plan_consolidation(t)
    new, records = reflect_and_apply(t, regions, model=Broken())
    assert records and all(r["error"] for r in records) and new.root_selection_experience


def test_reflection_reads_algorithm_evidence_only():
    t = TreePopulation()
    a = add(t, 0, 1)
    b = add(t, a, 2)
    add(t, b, 3, evaluations=2)
    t.evaluations_used += 40
    regions = plan_consolidation(t, TopologyConfig(quality_floor=1))
    prompt = build_selection_prompt(t, regions[0], task_description="Minimize")
    assert "Code change from its parent" in prompt and "regressed" in prompt
    assert "evaluations including" not in prompt and "child attempts" not in prompt
    new, _ = reflect_and_apply(t, regions, model=None)
    assert new.nodes[a].children_count == 1


def test_three_round_reflection_reuses_old_notes_and_reaches_all_operators():
    t = TreePopulation()
    a = add(t, 0, 10)
    other = add(t, a, 5)
    tip = a
    prompts = []

    class Notes:
        def complete(self, prompt, **kwargs):
            prompts.append(prompt)
            marker = "PATH" if "[Path Evidence]" in prompt else "SIDE"
            return json.dumps({"experience": f"{marker}_NOTE_{round_number}"})

    for round_number in range(3):
        mid = add(t, tip, 9)
        tip = add(t, mid, 1 / (round_number + 1))
        add(t, mid, None)
        add(t, a, None)
        t.evaluations_used += 40
        t.nodes[other].last_used = t.nodes[tip].last_used = t.evaluations_used
        regions = plan_consolidation(t, TopologyConfig(quality_floor=1))
        assert any(isinstance(r, PathContraction) for r in regions)
        assert any(not isinstance(r, PathContraction) for r in regions)
        prompt_start = len(prompts)
        t, _ = reflect_and_apply(t, regions, model=Notes())
        if round_number:
            used = "\n".join(prompts[prompt_start:])
            assert f"PATH_NOTE_{round_number - 1}" in used and f"SIDE_NOTE_{round_number - 1}" in used
        for op, reference in [("Refine", None), ("Explore", None), ("Crossover", a)]:
            context, _, _ = build_evolution_context(t, tip, op, reference)
            assert f"PATH_NOTE_{round_number}" in context and f"SIDE_NOTE_{round_number}" in context
        t = TreePopulation.from_dict(json.loads(json.dumps(asdict(t))))


def test_attempts_from_one_base_wait_until_that_stage_ends():
    t = TreePopulation()
    a = add(t, 0, 1)
    first = add(t, a, 5)
    t.evaluations_used += 39
    second = add(t, a, 6)
    cfg = TopologyConfig(quality_floor=1)
    assert plan_consolidation(t, cfg) == ()
    t.evaluations_used += 40
    regions = plan_consolidation(t, cfg)
    assert len(regions) == 1 and regions[0].parent_id == a and regions[0].removed_ids == (first, second)


def test_version_overtaken_by_its_own_better_descendant_is_not_a_representative():
    t = TreePopulation()
    a = add(t, 0, 3)
    b = add(t, a, 2)
    c = add(t, b, 1)
    apart = add(t, 0, 2.5)
    t.evaluations_used += 40
    cfg = TopologyConfig(quality_floor=0, clone_distance=0, same_fitness_clone_distance=0)
    assert quality_representatives(t, cfg) == {c, apart}
    regions = plan_consolidation(t, cfg)
    assert tuple(sorted(apply_consolidation(t, regions).nodes)) == (c, apart)
    assert regions[0].removed_ids == (a, b)
    tie = TreePopulation()
    a = add(tie, 0, 1)
    b = add(tie, a, 1)
    tie.evaluations_used += 40
    assert tuple(sorted(apply_consolidation(tie, plan_consolidation(tie, cfg)).nodes)) == (a, b)


def test_cross_route_explore_does_not_overtake_parent_and_draft_can_contract():
    t = TreePopulation()
    a = add(t, 0, 3.0)
    e = add(t, a, 4.0, op="Explore")
    v = add(t, e, 1.0, op="Refine")
    filler = [add(t, 0, 10.0 + k) for k in range(4)]
    t.evaluations_used += 40
    cfg = TopologyConfig(quality_floor=0.5, clone_distance=0, same_fitness_clone_distance=0)
    reps = quality_representatives(t, cfg)
    assert a in reps and v in reps
    regions = plan_consolidation(t, cfg)
    assert any(isinstance(r, PathContraction) and r.parent_id == a and r.child_id == v
               and r.removed_ids == (e,) and r.merged_operator == "Explore" for r in regions)
    new_tree = apply_consolidation(t, regions)
    assert e not in new_tree.nodes and filler[-1] in removed(regions)
    assert new_tree.edges[v].parent_id == a and new_tree.edges[v].operator == "Explore"
    assert new_tree.direction_of(v) == e


def test_route_max_weighted_allocation_eliminates_additive_version_bias_and_keeps_old_bases_eligible():
    t = TreePopulation()
    a0 = add(t, 0, 2.5)
    a_best = add(t, a0, 2.0, op="Refine")
    a_extra = [add(t, a0, 2.6 + 0.1 * k, op="Refine") for k in range(3)]
    b_draft = add(t, a0, 2.0, op="Explore")
    for k in range(6):
        add(t, 0, 6.0 + k)
    probs = parent_probabilities(t)
    assert abs(sum(probs.values()) - 1.0) < 1e-12
    p_route_a = sum(probs[i] for i in [a0, a_best, *a_extra])
    assert abs(p_route_a - probs[b_draft]) < 1e-12
    assert probs[a0] > 0.0 and probs[a_best] > probs[a0]


def test_contracted_intermediate_sides_are_rendered_in_variation_prompt():
    t = TreePopulation()
    a = add(t, 0, 10.0)
    b = add(t, a, 8.0, op="Refine")
    c = add(t, b, 5.0, op="Refine")
    s_ref = add(t, b, 9.5, op="Refine")
    s_exp = add(t, b, 11.0, op="Explore")
    other = add(t, a, 6.0, op="Refine")
    t.evaluations_used += 40
    t.nodes[c].last_used = t.nodes[other].last_used = t.evaluations_used
    regions = plan_consolidation(t, TopologyConfig(quality_floor=1.0))
    (pc,) = regions
    assert pc.removed_ids == (b,) and pc.side_ids == (s_ref, s_exp)
    prompt = build_variation_prompt(t, pc, task_description="Minimize")
    assert "Side Attempts from Contracted Intermediates:" in prompt
    assert "Side Trial 1 [Refine] from Step 1: regressed" in prompt
    assert "Side Trial 2 [Explore] from Step 1: regressed" in prompt
