from dataclasses import asdict
import json

import pytest

from evotrace_v1.allocation import AllocationConfig
from evotrace_v1.context import ContextConfig, build_evolution_context
from evotrace_v1.models import Edge, Node, TreePopulation
from evotrace_v1 import EvoTrace
from evotrace_v1.topology import TopologyConfig
from evotrace_v1 import Config, Task
from evotrace_v1.models import ModelReply


def _body_for(idx: int) -> str:
    patterns = [
        f"return x * {0.1 * idx + 0.5:.2f}",
        f"return (x ** 2) / (1.0 + abs(x)) + {0.05 * idx:.2f}",
        f"acc = 0.0\n    for i in range(3):\n        acc += (x + i) * {0.02 * idx + 0.1:.2f}\n    return acc",
        f"if x > {idx}:\n        return x * 0.2\n    return min(x, 0.0) + {0.1 * idx:.2f}",
    ]
    return patterns[idx % len(patterns)]


class DualSchemaScriptedModel:
    """Handles candidate generation schemas and reflection schemas deterministically."""

    def __init__(self):
        self.request_options = {}
        self.calls = []
        self._counter = 0

    def complete(self, prompt: str, **kwargs) -> ModelReply:
        schema = (self.request_options.get("response_format") or {}).get("json_schema", {}).get("name", "")
        self.calls.append((schema, prompt))
        self._counter += 1

        if schema == "evotrace_variation_experience":
            return ModelReply(
                content=json.dumps({
                    "experience": "Adopted smoother nonlinear term. Linear intermediate tweak was superseded.",
                }),
                usage={"total_tokens": 80},
            )
        if schema == "evotrace_selection_experience":
            return ModelReply(
                content=json.dumps({
                    "experience": "High-offset variants regressed in score.",
                }),
                usage={"total_tokens": 90},
            )

        idx = self._counter
        code = f"def solve(x):\n    {_body_for(idx)}\n"
        return ModelReply(
            content=json.dumps({
                "analysis": f"Step {idx} modification.",
                "code": code,
                "design": f"Design {idx}: compute pattern {idx % 4}.",
            }),
            usage={"total_tokens": 100},
        )


def _sample_tree_with_experiences() -> TreePopulation:
    tree = TreePopulation(root_id=0, root_selection_experience="[Eliminated] Constant heuristic scored 50.0.")
    tree.add(
        Node(1, "def solve(x):\n    return x * 2.0\n", "Linear baseline.", 10.0,
             selection_experience="[Eliminated] Cubic term regressed to 25.0."),
        Edge(0, 1, None),
        evaluations=1,
    )
    tree.add(
        Node(3, "def solve(x):\n    return (x ** 2) / (1.0 + abs(x))\n", "Normalized quadratic.", 7.0,
             selection_experience="[Eliminated] Log scaling raised ValueError."),
        Edge(1, 3, "Refine", variation_experience="[10 -> 7 (improved)] Retained: Replaced linear with normalized quadratic."),
        evaluations=1,
    )
    tree.add(
        Node(4, "def solve(x):\n    return (x ** 2) / (2.0 + abs(x))\n", "Damped quadratic.", 7.5),
        Edge(3, 4, "Refine"),
        evaluations=1,
    )
    tree.add(
        Node(5, "def solve(x):\n    s = 0.0\n    for i in range(4):\n        s += x + i\n    return s\n", "Loop accumulator.", 6.8),
        Edge(1, 5, "Explore"),
        evaluations=1,
    )
    return tree


def test_refine_crossover_and_explore_contexts_extract_multiscale_tree_views():
    tree = _sample_tree_with_experiences()

    # 1. Refine(#3): path from root (1 -> 3 with variation_experience) + #3's selection_experience & active child #4 + #3's code
    r_ctx, r_inst, r_scores, r_view = build_evolution_context(tree, 3, "Refine")
    assert r_scores is True
    assert "[Evolution Path]" in r_ctx
    assert r_ctx.index("[Current Algorithm]") < r_ctx.index("[Evolution Path]")
    assert "Replaced linear with normalized quadratic" in r_ctx
    assert "[Attempts from Current Algorithm]" in r_ctx
    assert "Log scaling raised ValueError" in r_ctx
    assert "Active/Recent Child 1 (Refine): 7 -> 7.5 (regressed)" in r_ctx
    assert r_ctx.rstrip().endswith("```")
    assert r_view["path"] == [1, 3]

    # 2. Crossover(#3, ref=#5): LCA is #1! Shows Common Ancestor #1 + branch 1->3 + branch 1->5 + Reference #5 + Current #3
    c_ctx, _, c_scores, c_view = build_evolution_context(tree, 3, "Crossover", reference_id=5)
    assert c_scores is True
    assert c_view["lca_id"] == 1
    assert "[Common Ancestor]\nAlgorithm 1" in c_ctx
    assert c_ctx.count("Design: Linear baseline.") == 1
    assert "Cubic term regressed to 25.0" in c_ctx
    assert "[Current Branch since Common Ancestor]" in c_ctx
    assert "[Reference Branch since Common Ancestor]" in c_ctx
    assert "[Reference Algorithm]\nAlgorithm 5" in c_ctx
    assert c_ctx.index("[Current Algorithm]") < c_ctx.index("[Reference Algorithm]") < c_ctx.index("[Common Ancestor]")
    assert "for i in range(4):" in c_ctx
    assert "Log scaling raised ValueError" in c_ctx
    assert "[Attempts from Current Algorithm]" in c_ctx

    # 3. Explore reads ideas and experience; management counts stay in metadata.
    e_ctx, _, e_scores, e_view = build_evolution_context(tree, 3, "Explore")
    assert e_scores is True
    assert "[Algorithmic Ideas and Experience]" in e_ctx
    assert "Constant heuristic scored 50.0" in e_ctx
    assert "Approach 1: Linear baseline." in e_ctx
    assert "Approach 2: Loop accumulator." in e_ctx
    assert "evaluation(s) explored" not in e_ctx
    assert "structurally distinct" not in e_ctx
    assert "best retained fitness" not in e_ctx
    assert "Best retained design" not in e_ctx
    assert "Score: 7" not in e_ctx.split("[Current Algorithm]")[-1]
    assert len(e_view["directions"]) == 2

    # 4. Ablation flag `include_experiences=False` cleanly strips variation & selection experiences
    #    while keeping the exact same endpoint/design layout on contracted edges!
    no_exp_ctx, _, _, _ = build_evolution_context(
        tree, 3, "Refine", config=ContextConfig(include_experiences=False),
    )
    assert "Replaced linear with normalized quadratic" not in no_exp_ctx
    assert "Log scaling raised ValueError" not in no_exp_ctx
    assert "Resulting design: Normalized quadratic." in r_ctx
    assert "Resulting design: Normalized quadratic." in no_exp_ctx
    assert "-    return x * 2.0" not in no_exp_ctx


def test_refine_keeps_experience_on_a_single_compressed_initial_edge():
    tree = TreePopulation()
    tree.add(
        Node(3, "def solve(x):\n    return abs(x)\n", "Magnitude scoring.", 2.0),
        Edge(0, 3, None, variation_experience="Earlier signed scores lost magnitude information.",
             contracted=True),
        evaluations=3,
    )
    context, *_ = build_evolution_context(tree, 3, "Refine")
    without, *_ = build_evolution_context(
        tree, 3, "Refine", config=ContextConfig(include_experiences=False),
    )
    assert "Earlier signed scores lost magnitude information." in context
    assert "Earlier signed scores lost magnitude information." not in without
    assert "Initial Algorithm 3" in context and "Initial Algorithm 3" in without


def test_explore_reads_eliminated_direction_ledgers_on_non_origin_nodes_and_zero_ast_dist_allows_distinct_code():
    from evotrace_v1.allocation import select_crossover_reference
    from evotrace_v1.reflection import reflect_and_apply
    from evotrace_v1.topology import BranchConsolidation, ConsolidationPlan

    # 1. Explore pruned under non-origin Refine node #2 is still visible to Explore!
    tree = TreePopulation()
    tree.add(Node(1, "def solve(x):\n    return x\n", "Identity", 3.0), Edge(0, 1, None), evaluations=1)
    tree.add(Node(2, "def solve(x):\n    return x * x\n", "Quadratic", 1.0), Edge(1, 2, "Refine"), evaluations=1)
    tree.add(Node(3, "def solve(x):\n    return abs(x)\n", "PRUNED_EXPLORE_ABS", 5.0), Edge(2, 3, "Explore"), evaluations=1)
    plan = ConsolidationPlan(
        kept_ids=(1, 2), selected_ids=(2,), lca_anchor_ids=(), direction_anchor_ids=(1,),
        branch_consolidations=(BranchConsolidation(2, (3,)),),
    )
    pruned, _ = reflect_and_apply(tree, plan, model=None)
    exp_ctx, *_ = build_evolution_context(pruned, 2, "Explore")
    assert "PRUNED_EXPLORE_ABS" in exp_ctx

    # 2. `x / y` vs `y / x` have AST shingle distance 0.0, yet distinct normalized source -> eligible for Crossover!
    pair_tree = TreePopulation()
    pair_tree.add(Node(1, "def solve(x, y):\n    return x / y\n", "x/y", 1.0), Edge(0, 1, None), evaluations=1)
    pair_tree.add(Node(2, "def solve(x, y):\n    return y / x\n", "y/x", 2.0), Edge(0, 2, None), evaluations=1)
    ref_id, dist = select_crossover_reference(pair_tree, 1)
    assert ref_id == 2 and dist == 0.0


def test_explore_reads_every_retained_nodes_pruned_direction_experience():
    from evotrace_v1.reflection import reflect_and_apply
    from evotrace_v1.topology import BranchConsolidation, ConsolidationPlan

    tree = TreePopulation()
    for nid in (1, 2, 3):
        tree.add(
            Node(nid, f"def solve(x):\n    return x ** {nid}\n", f"Core {nid}", 4.0 - nid),
            Edge(nid - 1, nid, None if nid == 1 else "Refine"), evaluations=1,
        )
    for parent_id in (1, 2, 3):
        child_id = parent_id + 3
        tree.add(
            Node(child_id, f"def solve(x):\n    return abs(x) + {child_id}\n",
                 f"REMOVED_DIRECTION_{parent_id}", 20.0 + child_id),
            Edge(parent_id, child_id, "Explore"), evaluations=1,
        )
    plan = ConsolidationPlan(
        kept_ids=(1, 2, 3), selected_ids=(3,), lca_anchor_ids=(), direction_anchor_ids=(1,),
        branch_consolidations=tuple(BranchConsolidation(p, (p + 3,)) for p in (1, 2, 3)),
    )
    pruned, _ = reflect_and_apply(tree, plan, model=None)

    context, *_ = build_evolution_context(pruned, 3, "Explore")
    without_experiences, *_ = build_evolution_context(
        pruned, 3, "Explore", config=ContextConfig(include_experiences=False),
    )
    for nid in (1, 2, 3):
        marker = f"REMOVED_DIRECTION_{nid}"
        assert marker in pruned.nodes[nid].selection_experience
        assert context.count(marker) == 1
        assert marker not in without_experiences


def test_explore_reads_direction_entry_transition_without_dumping_internal_refinement_stages():
    tree = TreePopulation()
    entry_exp = "Entry exploration stage. " * 15 + "ENTRY_FACT"
    internal_exp = "Internal micro-refinement stage. " * 10 + "INTERNAL_TWEAK"
    tree.add(Node(1, "def solve(x):\n    return x\n", "Core 1", 4.0), Edge(0, 1, None), evaluations=1)
    tree.add(
        Node(2, "def solve(x):\n    acc = 0.0\n    for i in range(4):\n        acc += x + i\n    return acc\n", "Core 2", 3.0),
        Edge(1, 2, "Explore", variation_experience=entry_exp, contracted=True),
        evaluations=1,
    )
    tree.add(
        Node(3, "def solve(x):\n    acc = 0.0\n    for i in range(4):\n        acc += (x + i) * 1.1\n    return acc\n", "Core 3", 2.0),
        Edge(2, 3, "Refine", variation_experience=internal_exp, contracted=True),
        evaluations=1,
    )

    context, *_ = build_evolution_context(tree, 3, "Explore")
    without_experiences, *_ = build_evolution_context(
        tree, 3, "Explore", config=ContextConfig(include_experiences=False),
    )
    assert entry_exp in context
    assert internal_exp not in context
    assert entry_exp not in without_experiences
    # Internal refinement stages remain visible in Refine path context!
    refine_ctx, *_ = build_evolution_context(tree, 3, "Refine")
    assert internal_exp in refine_ctx


def test_explore_reads_compressed_direction_entry_and_keeps_original_count_after_restore():
    from evotrace_v1.search import deserialize_tree
    from evotrace_v1.reflection import reflect_and_apply
    from evotrace_v1.topology import ConsolidationPlan, PathContraction

    tree = TreePopulation()
    tree.add(Node(1, "def solve(x):\n    return x\n", "Initial", 3.0), Edge(0, 1, None), evaluations=1)
    tree.add(Node(2, "def solve(x):\n    return abs(x)\n", "Exploration", 5.0,
                  selection_experience="EARLIER_DIRECTION_ATTEMPT"), Edge(1, 2, "Explore"), evaluations=2)
    tree.add(Node(3, "def solve(x):\n    return x * x\n", "Improved", 1.0),
             Edge(2, 3, "Refine"), evaluations=1)
    plan = ConsolidationPlan(
        kept_ids=(1, 3), selected_ids=(3,), lca_anchor_ids=(), direction_anchor_ids=(1,),
        path_contractions=(PathContraction(1, 3, (2,), "Explore", None),),
    )
    rebuilt, _ = reflect_and_apply(tree, plan, model=None)
    restored = deserialize_tree(asdict(rebuilt))
    context, _, _, view = build_evolution_context(restored, 3, "Explore")
    assert "EARLIER_DIRECTION_ATTEMPT" in context
    assert "Approach 2: Improved" in context
    assert "evaluation(s) explored" not in context
    assert "best retained fitness" not in context
    direction = next(d for d in view["directions"] if d["origin"] == 2)
    assert direction["entry"] == 3 and direction["evaluations"] == 3
    restored.add(Node(4, "def solve(x):\n    return x * x / 2\n", "Next", 0.5),
                 Edge(3, 4, "Refine"), evaluations=2)
    assert restored.direction_of(4) == 2
    assert restored.direction_evaluations[2] == 5


def test_evotrace_full_search_loop_consolidates_reflects_and_resumes_identically(tmp_path):
    task = Task(
        "Minimize positive score.",
        "def solve(x):\n    pass\n",
        lambda source, fn: abs(float(fn(2.0))),
        score_definition="Absolute value at x=2; lower is better.",
        time_limit=5,
    )
    cfg = Config(initial_size=3, evaluation_budget=12, seed=42)
    topo_cfg = TopologyConfig(base_capacity=4, trigger_capacity=7, min_protected_evaluations=2)
    alloc_cfg = AllocationConfig(refine_weight=0.45, explore_weight=0.30, crossover_weight=0.25)

    run_dir = tmp_path / "full_run"
    model = DualSchemaScriptedModel()
    engine = EvoTrace(
        task,
        model,
        config=cfg,
        topology_config=topo_cfg,
        allocation_config=alloc_cfg,
        run_dir=run_dir,
    )

    # Run first 8 steps (triggers at least 1 consolidation + LLM reflection at 7 nodes!)
    for _ in range(8):
        assert engine.step() is not None

    assert engine.consolidations_count >= 1
    assert engine.reflection_calls_count >= 1
    assert len(engine.tree.nodes) <= topo_cfg.trigger_capacity
    rows = [json.loads(line) for line in (run_dir / "attempts.jsonl").read_text().splitlines()]
    assert any("tree_after" in row for row in rows)
    assert not (run_dir / "evotrace_settings.json").exists()
    assert not (run_dir / "consolidations.jsonl").exists()
    assert (run_dir / "tree_latest.json").is_file()

    # Resume from `run_dir` for the remaining 4 evaluations and verify state restoration works cleanly
    resumed_model = DualSchemaScriptedModel()
    resumed = EvoTrace(
        task,
        resumed_model,
        config=cfg,
        topology_config=topo_cfg,
        allocation_config=alloc_cfg,
        run_dir=run_dir,
        resume=True,
    )
    assert resumed.evaluations_used == 8
    assert set(resumed.tree.nodes) == set(engine.tree.nodes)

    result = resumed.run()
    assert result.status == "finished"
    assert result.stop_reason == "evaluation_budget_exhausted"
    assert result.evaluations == 12
    assert result.best is not None

    # Verify resume rejects mismatched EvoTrace settings and repairs a truncated attempts journal
    with pytest.raises(ValueError, match="Incompatible EvoTrace resume settings"):
        EvoTrace(
            task,
            DualSchemaScriptedModel(),
            config=cfg,
            topology_config=topo_cfg,
            allocation_config=AllocationConfig(parent_selection_lambda=0.0),
            run_dir=run_dir,
            resume=True,
        )

    # Context switches are checked in saved configuration, independently of protocol labels.
    for context_config in (
        ContextConfig(include_experiences=False),
        ContextConfig(include_reference_code=False),
    ):
        with pytest.raises(ValueError, match="context"):
            EvoTrace(
                task, DualSchemaScriptedModel(), config=cfg,
                topology_config=topo_cfg, allocation_config=alloc_cfg,
                context_config=context_config, run_dir=run_dir, resume=True,
            )

    with (run_dir / "attempts.jsonl").open("ab") as handle:
        handle.write(b'{"attempt_id": ')
    repaired_resume = EvoTrace(
        task,
        DualSchemaScriptedModel(),
        config=Config(initial_size=3, evaluation_budget=14, seed=42),
        topology_config=topo_cfg,
        allocation_config=alloc_cfg,
        run_dir=tmp_path / "nonexistent",
    )
    assert repaired_resume.evaluations_used == 0
    clean_resume = EvoTrace(
        task,
        DualSchemaScriptedModel(),
        config=cfg,
        topology_config=topo_cfg,
        allocation_config=alloc_cfg,
        run_dir=run_dir,
        resume=True,
    )
    assert clean_resume.evaluations_used == 12


def test_context_handles_stale_references_text_limits_and_same_fitness_clones():
    from evotrace_v1.allocation import parent_probabilities
    from evotrace_v1.reflection import (
        clean_prompt_text,
        fallback_selection_experience,
        format_selection_experience,
        format_variation_experience,
    )
    from evotrace_v1.topology import BranchConsolidation, plan_consolidation

    # 1. clean_prompt_text restores unescaped LaTeX \beta (\x08eta) and \frac (\x0crac)
    dirty = "Ratio $d_{cv}/(1+\x08eta d_{vs}) + \x0crac{1}{2}$"
    assert clean_prompt_text(dirty) == r"Ratio $d_{cv}/(1+\beta d_{vs}) + \frac{1}{2}$"

    # 2. Stale #ID stripping + clause-boundary truncation + no double [Eliminated] prefix
    raw_sel = (
        "[Eliminated] Attempt #45 [Explore]: Angle detour: score 8.17; "
        + "; ".join(f"Refine #{100+i}: variant_{i} formula: score {7.0+0.01*i:.2f}" for i in range(30))
    )
    formatted_sel = format_selection_experience(raw_sel)
    assert formatted_sel.startswith("[Branch experience] ")
    assert format_selection_experience(formatted_sel) == formatted_sel
    assert "[Eliminated]" not in formatted_sel
    assert "#" not in formatted_sel
    assert "variant_29 formula: score 7.29" in formatted_sel
    assert len(formatted_sel) > 960

    raw_var = format_variation_experience(
        "Replaced linear with quadratic from #12.",
        "#15 (cubic): score 9.1; Refine #18: score 8.4.",
        parent_fit=10.0,
        child_fit=7.0,
    )
    assert "#" not in raw_var

    # 3. Fallback selection experience never duplicates '[Eliminated]' and preserves new removals
    t_fb = TreePopulation()
    t_fb.add(
        Node(1, "def solve(x):\n    return x\n", "Base", 5.0,
             selection_experience="[Eliminated] " + "Old item: score 9.0; " * 45),
        Edge(0, 1, None),
        evaluations=1,
    )
    t_fb.add(
        Node(2, "def solve(x):\n    return x[99]\n", "NEW_BAD_INDEXING", None, failure="IndexError: out of bounds"),
        Edge(1, 2, "Refine"),
        evaluations=1,
    )
    fb_exp = fallback_selection_experience(t_fb, BranchConsolidation(1, (2,)))
    assert fb_exp.startswith("[Branch experience] ")
    assert "[Eliminated]" not in fb_exp
    assert "NEW_BAD_INDEXING" in fb_exp and "IndexError" in fb_exp

    # 4. Dangling reference_id and 'Explore with reference Algorithm X' are never rendered in prompts!
    t_ref = TreePopulation()
    t_ref.add(Node(1, "def solve(x):\n    return x\n", "Init 1", 10.0), Edge(0, 1, None), evaluations=1)
    t_ref.add(Node(2, "def solve(x):\n    return x * 2\n", "Alive ref", 8.0), Edge(0, 2, None), evaluations=1)
    # Edge 1->5 contracted across Explore + pruned Crossover ref #99 -> operator="Explore", reference_id=99
    t_ref.add(
        Node(5, "def solve(x):\n    return x ** 2\n", "Explored", 6.0),
        Edge(1, 5, "Explore", reference_id=99, contracted=True),
        evaluations=1,
    )
    # Edge 5->6 is Crossover with pruned ref #88, Edge 6->7 is Crossover with alive ref #2
    t_ref.add(
        Node(6, "def solve(x):\n    return x ** 2 + 1\n", "Cross dead ref", 5.5),
        Edge(5, 6, "Crossover", reference_id=88, contracted=True),
        evaluations=1,
    )
    t_ref.add(
        Node(7, "def solve(x):\n    return x ** 2 + 0.5\n", "Cross alive ref", 5.0),
        Edge(6, 7, "Crossover", reference_id=2),
        evaluations=1,
    )
    r_ctx, *_ = build_evolution_context(t_ref, 7, "Refine")
    assert "Explore with reference" not in r_ctx
    assert "Algorithm 99" not in r_ctx
    assert "Algorithm 88" not in r_ctx
    assert "Crossover with reference Algorithm 2" in r_ctx

    # 5. Deep path (> max_path_steps) anchors Initial Algorithm at top while keeping recent steps
    deep = TreePopulation()
    for nid in range(1, 14):
        deep.add(
            Node(nid, f"def solve(x):\n    return x + {nid}\n", f"StepDesign {nid}", 20.0 - nid),
            Edge(nid - 1, nid, None if nid == 1 else "Refine"),
            evaluations=1,
        )
    deep_ctx, *_ = build_evolution_context(deep, 13, "Refine", config=ContextConfig(max_path_steps=6))
    assert "Initial Algorithm 1: fitness = 19" in deep_ctx
    assert "earlier step(s) omitted" in deep_ctx
    assert "Step 12 (Refine)" in deep_ctx

    # 6. Same-fitness minor AST rewrites (0.08 <= dist < 0.20) do not crowd out diverse candidates,
    #    and MAD == 0 uses IQR/range fallback instead of collapsing to 1e-6 step function!
    crowd = TreePopulation()
    base_code = (
        "def solve(x):\n"
        "    a = x + 1.0\n"
        "    b = a * 2.0\n"
        "    c = b / (1.0 + abs(x))\n"
        "    d = c - 0.5 * x\n"
        "    return d + a\n"
    )
    rewrite_code = (
        "def solve(x):\n"
        "    a = x + 1.0\n"
        "    b = a * 2.0\n"
        "    c = b / (1.0 + abs(x))\n"
        "    d = c - 0.5 * x\n"
        "    res = float(d + a)\n"
        "    return res\n"
    )
    alt_code = "def solve(x):\n    s = 0.0\n    for k in range(4):\n        s += (x - k) ** 2\n    return s\n"
    crowd.add(Node(1, base_code, "Champ", 5.0), Edge(0, 1, None), evaluations=1)
    crowd.add(Node(2, rewrite_code, "SameFitRewrite", 5.0), Edge(0, 2, None), evaluations=1)
    crowd.add(Node(3, alt_code, "DistinctRunnerUp", 5.2), Edge(0, 3, None), evaluations=1)
    crowd.add(Node(4, "def solve(x):\n    return x * 10.0\n", "Tail", 9.0), Edge(0, 4, None), evaluations=1)
    plan = plan_consolidation(crowd, TopologyConfig(base_capacity=2, trigger_capacity=4))
    assert set(plan.kept_ids) == {1, 3}

    # Check MAD == 0 fallback uses median nonzero deviation instead of 1e-6 step collapse!
    mad_tree = TreePopulation()
    for nid, fit in [(1, 4.95), (2, 5.0), (3, 5.0), (4, 5.0), (5, 5.2)]:
        mad_tree.add(Node(nid, f"def solve(x):\n    return x + {nid}\n", f"D{nid}", fit), Edge(0, nid, None), evaluations=1)
    probs = parent_probabilities(mad_tree)
    assert probs[1] > probs[2] > probs[5] > 0.0



@pytest.mark.parametrize("crash_stage", ["before_attempt_commit", "before_preview_write"])
def test_resume_uses_committed_attempt_even_when_consolidation_is_interrupted(tmp_path, monkeypatch, crash_stage):
    task = Task("Minimize the score", "def solve(x):\n    pass\n", lambda src, fn: abs(fn(2.0)), 2)
    config = Config(initial_size=3, evaluation_budget=12, seed=42)
    topology = TopologyConfig(base_capacity=4, trigger_capacity=7)
    search = EvoTrace(task, DualSchemaScriptedModel(), config=config,
                      topology_config=topology, run_dir=tmp_path)
    for _ in range(6):
        search.step()
    committed_tree = asdict(search.tree)
    append, write = search.store.append, search.store.write

    def append_or_crash(name, row):
        if name == "attempts.jsonl" and crash_stage == "before_attempt_commit":
            raise OSError("simulated interruption before commit")
        append(name, row)

    def write_or_crash(name, value):
        if name == "tree_latest.json" and crash_stage == "before_preview_write":
            raise OSError("simulated interruption after commit")
        write(name, value)

    monkeypatch.setattr(search.store, "append", append_or_crash)
    monkeypatch.setattr(search.store, "write", write_or_crash)
    with pytest.raises(OSError, match="simulated interruption"):
        search.step()
    restored = EvoTrace(task, DualSchemaScriptedModel(), config=config,
                        topology_config=topology, run_dir=tmp_path, resume=True)
    if crash_stage == "before_attempt_commit":
        assert restored.evaluations_used == 6 and restored.consolidations_count == 0
        assert asdict(restored.tree) == committed_tree
    else:
        assert restored.evaluations_used == 7 and restored.consolidations_count == 1
        assert asdict(restored.tree) == asdict(search.tree)
