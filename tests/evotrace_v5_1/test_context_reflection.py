"""算子隔离上下文与剪枝反思测试：覆盖真实生成父代归因、截断回退、两层笔记上限、共享指令与回退标签。"""

import json

from evotrace_v5_1 import context as ctx_mod
from evotrace_v5_1.context import build_evolution_context, evolution_prompt, repair_program_prompt
from evotrace_v5_1.models import ROOT_ID, Evaluation, Program, TreePopulation
from evotrace_v5_1.reflection import (
    build_cleanup_merge_prompt,
    build_leaf_eviction_prompt,
    reflect_reduction,
    strip_stale_node_ids,
)
from evotrace_v5_1.task import Task
from evotrace_v5_1.topology import Reduction


def sample_task() -> Task:
    return Task(
        "Minimize the scalar output.",
        "def solve(x):\n    pass\n",
        lambda _src, fn: fn(1),
        5.0,
        score_definition="Lower is better.",
    )


def make_prog(
    pid: int,
    fitness: float,
    *,
    code: str,
    design: str,
    analysis: str = "Secret pre-coding monologue",
    parent_id: int = ROOT_ID,
) -> Program:
    return Program(pid, code, design, "Refine", parent_id, None, Evaluation("success", fitness), analysis)


def test_explore_hides_all_code_all_reflections_and_has_no_domain_hints():
    task = sample_task()
    archive = {
        1: make_prog(1, 5.0, code="def solve(x):\n    return x * 111\n", design="Linear scaling"),
        2: make_prog(2, 7.0, code="def solve(x):\n    return x * 222\n", design="Quadratic term"),
    }
    tree = TreePopulation(
        parents={1: ROOT_ID, 2: ROOT_ID},
        idea_notes={1: "UNREALIZED_SECRET_NOTE_A", 2: "UNREALIZED_SECRET_NOTE_B"},
    )
    prompt, info = evolution_prompt(task, tree, archive, ROOT_ID, "Explore", explore_ids=(1, 2))
    assert info["memory_shown"] is False
    assert "Linear scaling" in prompt and "Quadratic term" in prompt
    assert "x * 111" not in prompt and "x * 222" not in prompt
    assert "UNREALIZED_SECRET_NOTE" not in prompt
    lower = prompt.lower()
    assert "tentative solution" not in lower
    assert "downstream consequences" not in lower
    assert "subproblems" not in lower


def test_crossover_blocks_reflections_and_distinguishes_same_vs_cross_lineage():
    task = sample_task()
    archive = {
        1: make_prog(1, 5.0, code="def solve(x):\n    return x + 10\n", design="Offset ten"),
        2: make_prog(2, 8.0, code="def solve(x):\n    return x * 20\n", design="Scale twenty"),
    }
    tree = TreePopulation(
        parents={1: ROOT_ID, 2: ROOT_ID},
        idea_notes={1: "PRIMARY_NOTE_HIDDEN", 2: "DONOR_NOTE_HIDDEN"},
    )
    prompt_cross, info_cross = evolution_prompt(
        task, tree, archive, 1, "Crossover", reference_id=2, cross_lineage=True
    )
    assert info_cross["memory_shown"] is False and info_cross["cross_lineage"] is True
    assert "[Complementary Algorithm (Different Lineage)]" in prompt_cross
    assert "worse than primary by 60%" in prompt_cross
    assert "PRIMARY_NOTE_HIDDEN" not in prompt_cross and "DONOR_NOTE_HIDDEN" not in prompt_cross

    prompt_same, info_same = evolution_prompt(
        task, tree, archive, 1, "Crossover", reference_id=2, cross_lineage=False
    )
    assert info_same["cross_lineage"] is False
    assert "Different Lineage" not in prompt_same
    assert "different lineages" not in prompt_same
    assert "[Complementary Algorithm]" in prompt_same


def test_refine_uses_shared_instruction_caps_notes_to_self_and_parent_and_tracks_truncation(monkeypatch):
    # Grandparent 1 -> Parent 2 -> Child 3 (all in tree); grandparent note must NOT broadcast to 3
    archive = {
        1: make_prog(1, 5.0, code="def solve(x):\n    return x + 1\n", design="Root"),
        2: make_prog(2, 6.0, code="def solve(x):\n    return x + 2\n", design="Parent", parent_id=1),
        3: make_prog(3, 8.0, code="def solve(x):\n    return x + 8\n", design="Child", parent_id=2),
    }
    tree = TreePopulation(
        parents={1: ROOT_ID, 2: 1, 3: 2},
        idea_notes={1: "GRANDPARENT_STALE_NOTE", 2: "DIRECT_PARENT_NOTE", 3: "SELF_NOTE"},
    )
    ctx_on, instr_on, info_on = build_evolution_context(tree, archive, 3, "Refine", show_memory=True)
    ctx_off, instr_off, info_off = build_evolution_context(tree, archive, 3, "Refine", show_memory=False)
    # Shared instruction guarantees zero instruction confounder in within-run masking
    assert instr_on == instr_off
    assert info_on["memory_available"] is True and info_on["memory_source"] == "both"
    assert info_on["memory_shown"] is True and info_on["memory_truncated"] is False
    assert info_on["regressed_step_shown"] is True
    assert "SELF_NOTE" in ctx_on and "DIRECT_PARENT_NOTE" in ctx_on
    assert "GRANDPARENT_STALE_NOTE" not in ctx_on
    assert info_off["memory_masked"] is True and info_off["memory_shown"] is False
    assert "SELF_NOTE" not in ctx_off and "DIRECT_PARENT_NOTE" not in ctx_off
    assert "[Unrealized Ideas Worth a Second Chance]" not in ctx_off

    # Contracted edge (3's true parent_id=2, rewired in tree under 1) MUST NOT trigger regressed step
    tree_rewired = TreePopulation(parents={1: ROOT_ID, 3: 1})
    ctx_rw, _, info_rw = build_evolution_context(tree_rewired, archive, 3, "Refine", show_memory=True)
    assert info_rw["regressed_step_shown"] is False
    assert "[Recent Step That Regressed]" not in ctx_rw

    # Post-fit telemetry: if CONTEXT_CHARS clips memory_block, memory_shown must report False & memory_truncated True
    monkeypatch.setattr(ctx_mod, "CONTEXT_CHARS", len(ctx_off) + 10)
    ctx_clip, _, info_clip = build_evolution_context(tree, archive, 3, "Refine", show_memory=True)
    assert info_clip["memory_assigned"] is True
    assert info_clip["memory_shown"] is False
    assert info_clip["memory_truncated"] is True
    assert "[Unrealized Ideas Worth a Second Chance]" not in ctx_clip


def test_leaf_eviction_uses_true_generating_parent_and_falls_back_on_truncated_diff():
    task = sample_task()
    # Reproduces Astra Max's exact CVRP counterexample:
    # #2 (9.335) -> #17 (14.726, earlier removed) -> #23 (13.507, improved over #17, rewired under #2)
    archive = {
        2: make_prog(2, 9.335, code="def solve(x):\n    return x + 1\n", design="Grandparent"),
        17: make_prog(17, 14.726, code="def solve(x):\n    return x + 20\n", design="Bio parent", parent_id=2),
        23: make_prog(23, 13.507, code="def solve(x):\n    return x + 15\n", design="Improved child", parent_id=17),
    }
    tree = TreePopulation(parents={2: ROOT_ID, 23: 2}, idea_notes={2: "Old note on 2"})
    prompt = build_leaf_eviction_prompt(task, tree, archive, receiver_id=2, leaf_id=23)
    assert "[Surviving Ancestor Receiving This Note]" in prompt
    assert "[Evicted Trial's Direct Generating Parent (Earlier Removed)]" in prompt
    # Must show improvement over true generating parent #17, NOT mislabel it as regression!
    assert "Outcome vs direct generating parent: 14.726 -> 13.507 (improved)" in prompt
    assert "Outcome vs surviving ancestor: 9.335 -> 13.507 (regressed)" in prompt

    # Long diff (>28 lines) must NOT be sent truncated — must fall back to complete evicted trial code
    long_base = "def solve(x):\n" + "".join(f"    v{i} = x + {i}\n" for i in range(40)) + "    return v0\n"
    long_leaf = "def solve(x):\n" + "".join(f"    v{i} = x + {i + 100}\n" for i in range(40)) + "    return v0\n"
    archive[30] = make_prog(30, 5.0, code=long_base, design="Long base")
    archive[31] = make_prog(31, 9.0, code=long_leaf, design="Long leaf", parent_id=30)
    tree_long = TreePopulation(parents={30: ROOT_ID, 31: 30})
    long_prompt = build_leaf_eviction_prompt(task, tree_long, archive, 30, 31)
    assert "more diff lines" not in long_prompt
    assert "Complete evicted trial code" in long_prompt
    assert "v39 = x + 139" in long_prompt


def test_reflection_cleans_inherited_notes_labels_sources_accurately_and_preserves_notes_on_error():
    task = sample_task()
    archive = {
        1: make_prog(1, 5.0, code="def solve(x):\n    return x + 1\n", design="D1", analysis="SECRET_A1"),
        2: make_prog(2, 9.0, code="def solve(x):\n    return x + 2\n", design="D2", analysis="SECRET_A2", parent_id=1),
        3: make_prog(3, 3.0, code="def solve(x):\n    return x + 3\n", design="D3", analysis="SECRET_A3", parent_id=1),
    }

    # 1. Own note on target without any inherited ancestor note -> ZERO model calls
    calls: list[str] = []

    class RecordingModel:
        def complete(self, prompt, **kwargs):
            calls.append(prompt)
            return json.dumps({"experience": "Cleaned note from Algorithm #2 and (#3)."})

    tree_own = TreePopulation(parents={1: ROOT_ID, 3: 1}, idea_notes={3: "Own note on 3"})
    note_own, report_own = reflect_reduction(
        task, tree_own, archive, Reduction("upgrade", ROOT_ID, 3, (1,)), RecordingModel(), 30.0
    )
    assert note_own == "Own note on 3" and report_own is None and calls == []

    # 2. Single note inherited from replaced ancestor -> triggers cleanup against target's code & labels accurately!
    tree_inh = TreePopulation(parents={1: ROOT_ID, 3: 1}, idea_notes={1: "Ancestor idea A"})
    note_inh, report_inh = reflect_reduction(
        task, tree_inh, archive, Reduction("upgrade", ROOT_ID, 3, (1,)), RecordingModel(), 30.0
    )
    assert report_inh is not None and report_inh["kind"] == "upgrade_cleanup" and len(calls) == 1
    assert "inherited from replaced ancestor" in calls[0]
    assert "existing on surviving algorithm" not in calls[0]
    assert "#2" not in note_inh and "#3" not in note_inh

    # 3. Reflection error on evict_leaf preserves BOTH parent note AND carried leaf note
    class BrokenModel:
        def complete(self, prompt, **kwargs):
            raise RuntimeError("transient outage")

    tree_err = TreePopulation(
        parents={1: ROOT_ID, 2: 1}, idea_notes={1: "Parent note P", 2: "Carried note L"}
    )
    fallback_note, err_report = reflect_reduction(
        task, tree_err, archive, Reduction("evict_leaf", 1, 1, (2,)), BrokenModel(), 30.0
    )
    assert err_report is not None and err_report["status"] == "error"
    assert "Parent note P" in fallback_note and "Carried note L" in fallback_note

    # 4. build_cleanup_merge_prompt respects explicit source labels
    merge_p = build_cleanup_merge_prompt(
        task, archive, 3, [("inherited from replaced ancestor", "Note 1")]
    )
    assert "Note A (inherited from replaced ancestor):" in merge_p


def test_strip_stale_node_ids_and_repair_prompt_preserves_raw_code_and_line_numbers():
    assert strip_stale_node_ids("Attempt #12 tried X; compared to Algorithm #3 (#4) it failed.") == (
        "tried X; compared to it failed."
    )
    task = sample_task()
    raw_code = "def solve(x):\n    y = min(i for i in range(x))\n    return 1 / 0\n"
    prog = Program(
        1,
        raw_code,
        "Divide zero",
        "Init",
        ROOT_ID,
        None,
        Evaluation("error", failure='File "<string>", line 3\nZeroDivisionError: division by zero'),
    )
    prompt = repair_program_prompt(task, prog)
    assert "min(i for i in range(x))" in prompt
    assert "return 1 / 0" in prompt
    assert "ZeroDivisionError: division by zero" in prompt
