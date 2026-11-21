"""演化脉络上下文、三算子提示与原位段落反思单元测试。"""

import json

from evotrace_v5.context import (
    CONTEXT_CHARS,
    build_evolution_context,
    evolution_prompt,
    init_prompt,
    repair_format_prompt,
    repair_program_prompt,
)
from evotrace_v5.models import ROOT_ID, Candidate, Evaluation, Program, TreePopulation
from evotrace_v5.reflection import (
    bare_code,
    build_reflection_prompt,
    compact_code_diff,
    reflect_region,
)
from evotrace_v5.task import Task
from evotrace_v5.topology import Reduction


def _task():
    return Task(
        "Minimize the objective value.",
        "def score(x):\n    '''Compute priority.'''\n    return x\n",
        lambda _s, fn: float(fn(1)),
        time_limit=5.0,
    )


def _prog(
    pid,
    fitness,
    code,
    design,
    *,
    parent_id=None,
    operator="Refine",
    analysis=None,
    status="success",
    failure=None,
    dup=None,
):
    return Program(
        id=pid,
        code=code,
        design=design,
        operator=operator,
        parent_id=parent_id,
        reference_id=None,
        evaluation=Evaluation(
            status, fitness=fitness if status == "success" else None, failure=failure
        ),
        analysis=analysis,
        duplicate_of=dup,
    )


def test_bare_code_and_code_diff_strip_comments_and_omit_rewrites():
    code_a = 'def score(x):\n    """Doc."""\n    # noisy comment\n    y = x + 1\n\n    return y\n'
    code_b = "def score(x):\n    # different comment\n    y = x + 2\n    return y\n"
    bare = bare_code(code_a)
    assert "Doc." not in bare and "noisy comment" not in bare
    assert "y = x + 1" in bare

    diff = compact_code_diff(code_a, code_b)
    assert "-    y = x + 1" in diff and "+    y = x + 2" in diff
    assert "comment" not in diff

    rewrite = compact_code_diff(
        "def score(x):\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n",
        "def score(x):\n    p = x * 3\n    q = p - 7\n    r = q * q\n    return r\n",
    )
    assert rewrite == ""
    assert compact_code_diff(code_a, code_b, max_lines=1, omit_if_truncated=True) == ""


def test_designs_are_never_truncated_and_code_is_stripped_in_generation():
    long_design = "Compute normalized distance plus adaptive load balance factor. " * 12
    p1 = _prog(
        1,
        8.0,
        "def score(x):\n    # parent comment\n    return x + 1\n",
        long_design,
        parent_id=ROOT_ID,
        operator="Init",
        analysis="VERBOSE ANCESTOR ANALYSIS SHOULD NOT APPEAR",
    )
    p2 = _prog(
        2,
        7.5,
        "def score(x):\n    # child comment\n    return x * 0.5\n",
        long_design + "TAIL_MARKER",
        parent_id=1,
        operator="Refine",
    )
    p3 = _prog(
        3,
        7.1,
        "def score(x):\n    return x * 0.4\n",
        "Active sibling design",
        parent_id=2,
        operator="Refine",
    )
    tree = TreePopulation(
        parents={1: ROOT_ID, 2: 1, 3: 2},
        edge_notes={2: "Halving linear term improved score along path."},
        node_notes={2: "Quadratic penalties regressed from this parent."},
    )
    archive = {1: p1, 2: p2, 3: p3}

    ctx, instr = build_evolution_context(tree, archive, 2, "Refine", include_experiences=True)
    assert "TAIL_MARKER" in ctx
    assert "VERBOSE ANCESTOR ANALYSIS" not in ctx
    assert "child comment" not in ctx
    assert "Halving linear term improved score along path." in ctx
    assert "Quadratic penalties regressed from this parent." in ctx
    assert "Do not re-implement them; pursue a distinct change." in ctx
    assert "earlier experiences" in instr

    ctx_no_exp, instr_no_exp = build_evolution_context(
        tree, archive, 2, "Refine", include_experiences=False
    )
    assert "Halving linear term" not in ctx_no_exp
    assert "Quadratic penalties" not in ctx_no_exp
    assert "earlier experiences" not in instr_no_exp
    prompt_no_exp = evolution_prompt(_task(), tree, archive, 2, "Refine", include_experiences=False)
    assert "relevant experience" not in prompt_no_exp


def test_crossover_supports_weak_donor_and_explore_hides_reference_code():
    p1 = _prog(1, 7.2, "def score(x):\n    return x * 0.5\n", "Main parent design", parent_id=ROOT_ID)
    p2 = _prog(
        2,
        7.8,
        "def score(x):\n    # donor comment\n    return x * 0.25 + 1\n",
        "Weaker donor design",
        parent_id=ROOT_ID,
    )
    p3 = _prog(
        3,
        6.9,
        "def score(x):\n    return SECRET_CODE_MARKER\n",
        "Stronger explore ref design",
        parent_id=ROOT_ID,
    )
    tree = TreePopulation(
        parents={1: ROOT_ID, 2: ROOT_ID, 3: ROOT_ID},
        node_notes={2: "Donor branch note.", 3: "Ref branch note."},
    )
    archive = {1: p1, 2: p2, 3: p3}

    cross_ctx, _ = build_evolution_context(tree, archive, 1, "Crossover", reference_id=2)
    assert "does not beat the current algorithm overall" in cross_ctx
    assert "worse than current 7.2" in cross_ctx
    assert "regressed" not in cross_ctx
    assert "donor comment" not in cross_ctx
    assert "return x * 0.25 + 1" in cross_ctx
    assert "Donor branch note." in cross_ctx
    assert "[Development of the Current Algorithm]" not in cross_ctx

    exp_ctx, _ = build_evolution_context(tree, archive, 1, "Explore", explore_ids=[2, 3])
    assert "return x * 0.5" in exp_ctx
    assert "SECRET_CODE_MARKER" not in exp_ctx
    assert "return x * 0.25 + 1" not in exp_ctx
    assert exp_ctx.index("Stronger explore ref design") < exp_ctx.index("Weaker donor design")
    assert "[Relevant Experiences]" in exp_ctx
    assert "Ref branch note." in exp_ctx


def test_refine_shows_duplicates_and_recent_failures_without_clutter():
    p1 = _prog(1, 7.5, "def score(x):\n    return x\n", "Parent design", parent_id=ROOT_ID)
    dup = _prog(2, 7.5, "def score(x):\n    return x\n", "Dup", parent_id=1, dup=1)
    fail = _prog(
        3,
        None,
        "def score(x):\n    return x / 0\n",
        "Div zero design",
        parent_id=1,
        status="error",
        failure='Traceback:\n  File "<candidate>", line 2, in score\nZeroDivisionError: division by zero',
    )
    tree = TreePopulation(parents={1: ROOT_ID})
    ctx, _ = build_evolution_context(tree, {1: p1, 2: dup, 3: fail}, 1, "Refine")
    assert "Exact code duplicates: 1" in ctx
    assert "ZeroDivisionError" in ctx


def test_reflection_is_code_grounded_and_omits_reported_design():
    task = _task()
    p1 = _prog(1, 8.0, "def score(x):\n    return x + 5\n", "BIASED_DESIGN_1", parent_id=ROOT_ID)
    p2 = _prog(2, 8.4, "def score(x):\n    return x + 9\n", "BIASED_DESIGN_2", parent_id=1)
    p3 = _prog(3, 7.4, "def score(x):\n    return x + 1\n", "BIASED_DESIGN_3", parent_id=2)
    tree = TreePopulation(
        parents={1: ROOT_ID, 2: 1, 3: 2},
        node_notes={2: "Earlier side-branch tried negative multiplier and regressed."},
    )
    archive = {1: p1, 2: p2, 3: p3}
    event = Reduction("path", 1, 3, (2,))

    prompt, prior = build_reflection_prompt(task, tree, archive, event)
    assert "return x + 9" in prompt
    assert "BIASED_DESIGN" not in prompt
    assert "Earlier side-branch tried negative multiplier" in prompt
    assert "#1" not in prompt and "#2" not in prompt and "#3" not in prompt
    assert prior == ["Earlier side-branch tried negative multiplier and regressed."]

    class FakeModel:
        def complete(self, prompt, **_kwargs):
            return json.dumps(
                {
                    "analysis": "Step 1 raised offset from 5 to 9; Step 2 lowered it to 1.",
                    "experience": "Reducing the additive offset improved the score from 8 to 7.4, whereas raising it to 9 regressed.",
                }
            )

    note, aux = reflect_region(task, tree, archive, event, FakeModel())
    assert (
        note
        == "Reducing the additive offset improved the score from 8 to 7.4, whereas raising it to 9 regressed."
    )
    assert aux is not None and aux["status"] == "success"

    note_none, aux_none = reflect_region(task, tree, archive, event, None)
    assert note_none == prior[0] and aux_none is None

    class BrokenModel:
        def complete(self, prompt, **_kwargs):
            raise RuntimeError("backend down")

    note_err, aux_err = reflect_region(task, tree, archive, event, BrokenModel())
    assert note_err == prior[0] and aux_err["status"] == "error"

    class FencedModel:
        def complete(self, prompt, **_kwargs):
            return '```json\n{"analysis": "ok", "experience": "Fenced reply works."}\n```'

    note_fenced, aux_fenced = reflect_region(task, tree, archive, event, FencedModel())
    assert note_fenced == "Fenced reply works." and aux_fenced["status"] == "success"
    cand = Candidate.parse(
        '```json\n{"analysis": "a", "code": "def score(x):\\n    return x\\n", "design": "d"}\n```'
    )
    assert cand.design == "d"


def test_init_and_repair_prompts_keep_raw_source_code():
    task = _task()
    raw_code = "def score(x):\n    # keep line numbers\n    return 1 / 0\n"
    failed = _prog(
        1,
        None,
        raw_code,
        "Broken design",
        parent_id=ROOT_ID,
        operator="Init",
        status="error",
        failure='Traceback:\n  File "<candidate>", line 3, in score\nZeroDivisionError: division by zero',
    )
    init_p = init_prompt(task, {1: failed})
    assert "# keep line numbers" in init_p and "ZeroDivisionError" in init_p
    assert "chosen modification" not in init_p and "relevant experience" not in init_p

    rep_p = repair_program_prompt(task, failed)
    assert "# keep line numbers" in rep_p and "return 1 / 0" in rep_p

    fmt_p = repair_format_prompt(task, "not json", "Expecting value")
    assert "Expecting value" in fmt_p

    p1 = _prog(1, 7.0, "def score(x):\n    return x\n", "d", parent_id=ROOT_ID)
    evo = evolution_prompt(task, TreePopulation(parents={1: ROOT_ID}), {1: p1}, 1, "Refine")
    assert "[Output]" in evo and len(evo) < CONTEXT_CHARS
