import json

from evotrace_v1.models import Edge, Node, TreePopulation
from evotrace_v1.topology import TopologyConfig, plan_consolidation
from evotrace_v1.reflection import build_selection_prompt, build_variation_prompt, reflective_simplify
from evotrace_v1.models import ModelReply


class FakeReflectionModel:
    def __init__(self, fail=False):
        self.fail = fail
        self.request_options = {}
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append((prompt, dict(self.request_options)))
        if self.fail:
            raise RuntimeError("endpoint timeout")
        schema_name = self.request_options["response_format"]["json_schema"]["name"]
        if schema_name == "evotrace_variation_experience":
            payload = {
                "experience": "Switched from linear to quadratic scaling. Linear tweak scored 9.5; side attempt with cubic scaling scored 14.0.",
            }
        else:
            payload = {
                "experience": "Linear scaling (3.0x) regressed to 15.0; invalid guard raised ValueError.",
            }
        return ModelReply(content=json.dumps(payload), usage={"total_tokens": 120})


def _sample_tree() -> TreePopulation:
    tree = TreePopulation(root_id=0)

    def add(nid, parent, op, fit, body, failure=None, sel_exp=""):
        node = Node(
            id=nid,
            code=f"def solve(x):\n    {body}\n",
            design=f"Design of #{nid}",
            fitness=fit,
            failure=failure,
            selection_experience=sel_exp,
        )
        tree.add(node, Edge(parent_id=parent, child_id=nid, operator=op), evaluations=1)

    add(1, 0, None, 10.0, "return x * 1.0 + 1.0")
    add(2, 1, "Refine", 9.5, "return x * 1.1 + 1.0", sel_exp="[Eliminated] Cubic scaling scored 14.0.")
    add(3, 2, "Refine", 9.2, "return x * 1.2 + 1.0")
    add(4, 3, "Refine", 7.0, "return (x ** 2) / (1.0 + abs(x))")
    add(5, 1, "Refine", 9.6, "return min(x, 0.0) - 1.0")
    add(6, 5, "Refine", 7.2, "return x * 0.5 if x > 1.0 else min(x, 0.0)")
    add(7, 1, "Refine", 15.0, "return x * 3.0 + 1.0")
    add(8, 1, "Refine", None, "raise ValueError('bad x')", failure="ValueError: bad x")
    add(9, 0, None, 8.0, "acc = 0.0\n    for i in range(4):\n        acc += x + i\n    return acc")
    add(10, 4, "Explore", 11.0, "return max(x, 2.0) - min(x, 0.2)")
    add(11, 7, "Refine", 14.2, "return x * 2.8 + 1.0")  # Multi-node dead-end subtree 7 -> 11
    add(12, 0, None, 25.0, "return x + 100.0")  # Pruned initial node under Virtual Root #0
    return tree


def test_prompts_include_step_diffs_blackhole_selection_ledger_subtree_grouping_and_root_ledger():
    tree = _sample_tree()
    cfg = TopologyConfig(base_capacity=6, trigger_capacity=10, min_protected_evaluations=3)
    plan = plan_consolidation(tree, cfg)

    # 1. Variation prompt includes step diffs AND intermediate node #2's prior selection_experience!
    contr = next(c for c in plan.path_contractions if c.child_id == 4)
    v_prompt = build_variation_prompt(
        tree, contr,
        task_description="Minimize score.",
        function_template="def solve(x: float) -> float:\n    pass\n",
    )
    assert "Surviving Ancestor (fitness = 10)" in v_prompt
    assert "Surviving Descendant (fitness = 7)" in v_prompt
    assert "Step code change:" in v_prompt
    assert "return x * 1.1 + 1.0" in v_prompt
    assert "Earlier Side-Branch Experience: [Eliminated] Cubic scaling scored 14.0." in v_prompt
    assert "Net Code Diff (Ancestor -> Descendant):" in v_prompt
    assert "def solve(x: float)" in v_prompt

    # 2. Selection prompt includes parent baseline code + per-trial parent diff + failure without leaking #IDs
    reconnected, _, _ = reflective_simplify(tree, config=cfg, model=None)
    b_cons = next(b for b in plan.branch_consolidations if b.parent_id == 1)
    s_prompt = build_selection_prompt(tree, b_cons, task_description="Minimize score.")
    assert "Retained Parent Baseline (fitness = 10;" in s_prompt
    assert "return x * 1.0 + 1.0" in s_prompt
    assert "Code change from its parent:" in s_prompt
    assert "Removed Trial 1 [Refine]" in s_prompt
    assert "from Trial 1: 15 -> 14.2 (improved)" in s_prompt
    assert "ValueError: bad x" in s_prompt

    # 3. An eliminated initial algorithm contributes to the virtual root's experience.
    cfg_cap = TopologyConfig(base_capacity=5, trigger_capacity=10, min_protected_evaluations=3)
    reconn_cap, plan_cap, _ = reflective_simplify(tree, config=cfg_cap, model=None)
    assert any(b.parent_id == 0 for b in plan_cap.branch_consolidations)
    assert reconn_cap.root_selection_experience.startswith("[Branch experience]")


def test_reflective_simplify_writes_structured_experiences_and_root_ledger_with_model():
    tree = _sample_tree()
    cfg = TopologyConfig(base_capacity=5, trigger_capacity=10, min_protected_evaluations=3)
    model = FakeReflectionModel()

    new_tree, plan, records = reflective_simplify(
        tree,
        config=cfg,
        model=model,
        task_description="Minimize score.",
    )

    assert plan.path_contractions
    assert all(not r.used_fallback for r in records)
    assert all(r.prompt_chars == len(r.prompt) and json.loads(r.response) for r in records)
    assert all(r.error is None for r in records)
    exp_4 = new_tree.edges[4].variation_experience
    assert exp_4.startswith("[10 -> 7 (improved)] Switched")
    assert "quadratic scaling" in exp_4
    assert "Linear tweak scored 9.5" in exp_4

    sel_1 = new_tree.nodes[1].selection_experience
    assert sel_1.startswith("[Branch experience]")
    assert "ValueError" in sel_1
    assert "Surviving:" not in sel_1  # No redundant surviving branch prose in ledger!
    assert new_tree.root_selection_experience.startswith("[Branch experience]")


def test_reflective_simplify_falls_back_cleanly_on_model_error_and_preserves_prior_ledger():
    tree = _sample_tree()
    tree.nodes[1].selection_experience = "[Eliminated] Old failed attempt scored 18.0."
    cfg = TopologyConfig(base_capacity=6, trigger_capacity=10, min_protected_evaluations=3)

    new_tree, _, records = reflective_simplify(
        tree,
        config=cfg,
        model=FakeReflectionModel(fail=True),
        task_description="Minimize score.",
    )
    assert all(r.used_fallback for r in records)
    assert all(r.prompt and r.error == "RuntimeError: endpoint timeout" for r in records)
    assert "[10 -> 7 (improved)] Development:" in new_tree.edges[4].variation_experience
    assert "Cubic scaling scored 14.0." in new_tree.edges[4].variation_experience
    assert "Old failed attempt scored 18.0." in new_tree.nodes[1].selection_experience
    assert "ValueError: bad x" in new_tree.nodes[1].selection_experience


def test_invalid_reflection_keeps_raw_response_and_usage_for_audit():
    class InvalidReply:
        def complete(self, prompt, **kwargs):
            return ModelReply(content="{invalid", usage={"total_tokens": 73})

    _, _, records = reflective_simplify(
        _sample_tree(), model=InvalidReply(),
        config=TopologyConfig(base_capacity=6, trigger_capacity=10),
    )
    assert records
    assert all(r.used_fallback and r.error.startswith("JSONDecodeError:") for r in records)
    assert all(r.response == "{invalid" and r.usage["total_tokens"] == 73 for r in records)


def test_empty_or_nontext_experience_falls_back_without_erasing_evidence():
    for value in (None, [], "", "   ", "#123"):
        class EmptyReply:
            def complete(self, prompt, **kwargs):
                return ModelReply(content=json.dumps({"experience": value}), usage={"total_tokens": 12})
        tree, _, records = reflective_simplify(
            _sample_tree(), model=EmptyReply(),
            config=TopologyConfig(base_capacity=6, trigger_capacity=10),
        )
        assert all(r.used_fallback and r.error for r in records)
        assert "ValueError" in tree.nodes[1].selection_experience


def test_selection_prompt_uses_actual_parent_for_side_branch_hanging_off_contracted_node_and_lists_all_branches():
    tree = TreePopulation(root_id=0)

    def add(nid, parent, op, fit, body):
        tree.add(
            Node(nid, f"def solve(x):\n    {body}\n", f"Design {nid}", fit),
            Edge(parent, nid, op),
            evaluations=1,
        )

    # Path 1 (fit=10) -> 2 (fit=8.0, contracted) -> 3 (fit=5.0, kept).
    # Side branch 4 (fit=9.0) hangs off #2: relative to its actual parent #2 (8.0), #4 REGRESSED (8 -> 9),
    # whereas relative to anchor #1 (10.0) it would falsely look "improved"!
    add(1, 0, None, 10.0, "return x * 1.0")
    add(2, 1, "Refine", 8.0, "return x * 1.1")
    add(3, 2, "Refine", 5.0, "return (x ** 2) / (1.0 + abs(x))")
    add(4, 2, "Refine", 9.0, "return x * 1.2 + 5.0")
    for extra in range(5, 13):
        add(extra, 1, "Refine", 11.0 + extra, f"return x + {extra}.0")
    add(13, 0, None, 6.0, "acc = 0.0\n    for i in range(4):\n        acc += x + i\n    return acc")

    cfg = TopologyConfig(base_capacity=3, trigger_capacity=10)
    plan = plan_consolidation(tree, cfg)
    assert plan.kept_ids == (1, 3, 13)
    b_cons = next(b for b in plan.branch_consolidations if b.parent_id == 1)
    assert 4 in b_cons.removed_ids and len(b_cons.removed_ids) == 9

    reconnected, _, _ = reflective_simplify(tree, config=cfg, model=None)
    s_prompt = build_selection_prompt(tree, b_cons, task_description="Minimize score.")

    # Must list ALL 9 eliminated branches (no 6-branch drop) and score #4 against its actual parent #2 (8 -> 9 regressed)!
    assert "Removed Trial 1 [Refine] from Additional parent 1: 8 -> 9 (regressed)" in s_prompt
    assert "Additional parent 1 (fitness = 8):" in s_prompt
    assert "return x * 1.1" in s_prompt
    for idx in range(1, 10):
        assert f"Removed Trial {idx} [Refine]" in s_prompt
