"""三算子隔离上下文与提示词构造：Refine 共享中性指令且仅读 p 与直接父代笔记、Crossover 无反思、Explore 无源码无反思。"""

from __future__ import annotations

from collections.abc import Sequence

from .evaluation import TIMEOUT_PREFIX, compact_failure, failure_feedback
from .models import ROOT_ID, Operator, Program, TreePopulation, clean_text, fitness_text
from .reflection import compact_code_diff, outcome_summary
from .task import Task
from .topology import bare_code, beats

CONTEXT_CHARS = 90_000
MAX_RECENT_FAILURES = 3

COMMON_TEMPLATE = """[Algorithm-design task]
{task_description}

Design the computation performed by the following function to improve the complete solver's objective. Use the available inputs and the role of its returned result in the solver.

[Function interface]
```python
{function_template}
```

[Context]
{context}

[Instruction]
{instruction}

[Output]
Return exactly one valid JSON object with three string fields, in this order:
- "analysis": Reason carefully through the problem structure, the computational mechanism you choose, and why it can improve the complete solver's decisions, grounding your reasoning in the supplied evidence when relevant.
- "code": Provide the complete implementation using only the inputs in [Function interface], including necessary imports and helper definitions. Keep the code clean and self-contained; put explanations in "analysis" rather than code comments.
- "design": In about 80-120 words, give another algorithm designer a concise, factual description of what the code actually computes and its key parameters. Describe the implementation itself, not expected performance.
Preserve the required function signature and satisfy the task constraints and input/output requirements.
Return only the JSON object, without Markdown fences or additional text."""

# Shared neutral instruction across masked/unmasked arms so within-run masking isolates ONLY the memory block.
REFINE_INSTRUCTION = (
    "Improve the current algorithm by diagnosing how it makes decisions and where its computation is limited.\n"
    "Choose one focused, non-duplicate modification—revising the problem representation, decision logic, "
    "search/construction procedure, or key parameters.\n"
    "If unrealized ideas from earlier evicted trials are provided above, you may revisit one ONLY when you "
    "can identify a specific flaw to remove or a sub-component to test cleanly in isolation (skip any idea "
    "already implemented in the current algorithm); treat all notes as optional hypotheses, never as "
    "prohibitions or required directions.\n"
    "Avoid duplicating the exact implementation of an active branch, and implement a coherent improved algorithm."
)

CROSSOVER_INSTRUCTION_CROSS = (
    "Compare how the two implementations from different lineages compute their decisions and identify "
    "complementary strengths in each (an algorithm with a weaker overall score can still contain a valuable "
    "local mechanism or representation).\n"
    "Develop a unified algorithm that genuinely integrates their useful strengths—you may reorganize the "
    "algorithm's architecture rather than merely adding another weighted scoring term.\n"
    "If the two mechanisms cannot be cleanly combined, use the insight revealed by comparing them to construct "
    "a substantially stronger improvement to the primary algorithm."
)

CROSSOVER_INSTRUCTION_SAME = (
    "Compare how the two implementations compute their decisions and identify complementary strengths in each "
    "(an algorithm with a weaker overall score can still contain a valuable local mechanism or representation).\n"
    "Develop a unified algorithm that genuinely integrates their useful strengths—you may reorganize the "
    "algorithm's architecture rather than merely adding another weighted scoring term.\n"
    "If the two mechanisms cannot be cleanly combined, use the insight revealed by comparing them to construct "
    "a substantially stronger improvement to the primary algorithm."
)

EXPLORE_INSTRUCTION = (
    "Rethink how this task can be solved from first principles and implement a new algorithm whose core "
    "computational mechanism differs fundamentally from all existing lineages above.\n"
    "Do not produce another re-weighting or minor tweak of the same local features. Instead, change the "
    "information used, the internal representation, or the computational procedure that produces the decision.\n"
    "Choose a principled approach suited to the task and implement its complete main computation cleanly."
)


def _local_unrealized_notes(tree: TreePopulation, parent_id: int) -> tuple[list[str], str]:
    """仅读取主父代 p 及其存活直接父节点 π(p) 的笔记，避免长寿祖先向深层子树广播陈旧想法。"""
    notes: list[str] = []
    seen: set[str] = set()
    has_self = False
    has_parent = False

    own = clean_text(tree.idea_notes.get(parent_id, ""))
    if own:
        seen.add(own)
        notes.append(own)
        has_self = True

    direct_parent = tree.parents.get(parent_id, ROOT_ID)
    if direct_parent != ROOT_ID:
        p_note = clean_text(tree.idea_notes.get(direct_parent, ""))
        if p_note and p_note not in seen:
            seen.add(p_note)
            notes.append(f"Earlier note from direct parent in this lineage: {p_note}")
            has_parent = True

    source = (
        "both"
        if has_self and has_parent
        else "self"
        if has_self
        else "parent"
        if has_parent
        else "none"
    )
    return notes, source


def _recent_failures(
    archive: dict[int, Program], parent_id: int, *, exclude_ids: frozenset[int] = frozenset()
) -> list[Program]:
    failed = [
        p
        for p in archive.values()
        if p.id != 0
        and p.id not in exclude_ids
        and p.parent_id == parent_id
        and p.evaluation is not None
        and not p.valid
    ]
    failed.sort(key=lambda p: p.id)
    return failed[-MAX_RECENT_FAILURES:]


def _active_branches_and_failures_block(
    tree: TreePopulation,
    archive: dict[int, Program],
    parent_id: int,
    *,
    exclude_failures: frozenset[int] = frozenset(),
) -> str:
    parent = archive[parent_id]
    lines: list[str] = []
    for idx, cid in enumerate(tree.children_of(parent_id), 1):
        child = archive[cid]
        is_direct = child.parent_id == parent_id
        rel_note = outcome_summary(parent, child)
        scope = "" if is_direct else ", multi-step descendant"
        entry = [
            f"- Active branch {idx} ({rel_note}{scope}; avoid duplicating this exact implementation)",
            f"  Design: {clean_text(child.design)}",
        ]
        diff = compact_code_diff(
            parent.code, child.code, max_lines=14, max_chars=700, omit_if_truncated=True
        )
        if diff:
            diff_label = "Code diff" if is_direct else "Cumulative code diff"
            entry.append(f"  {diff_label}:\n```diff\n{diff}\n```")
        lines.append("\n".join(entry))

    for idx, prog in enumerate(_recent_failures(archive, parent_id, exclude_ids=exclude_failures), 1):
        reason = compact_failure(prog.evaluation.failure, prog.code, status=prog.evaluation.status)
        lines.append(
            f"- Recent failure {idx}: {clean_text(prog.design) or 'attempted modification'} -> {reason}"
        )

    if not lines:
        return ""
    return "[Already Active Branches & Recent Failures from Current Algorithm]\n" + "\n".join(lines)


def _regressed_step_block(tree: TreePopulation, archive: dict[int, Program], parent_id: int) -> str:
    """仅当直接生成父节点仍在树上（非收缩重连边）且严格优于当前节点时，提供一步完整客观对比。"""
    cur_prog = archive[parent_id]
    direct_parent = tree.parents.get(parent_id, ROOT_ID)
    if direct_parent == ROOT_ID or direct_parent != cur_prog.parent_id or direct_parent not in archive:
        return ""
    prev_prog = archive[direct_parent]
    if not beats(prev_prog, cur_prog):
        return ""
    diff = compact_code_diff(prev_prog.code, cur_prog.code, omit_if_truncated=True)
    if diff:
        body = f"Complete code change from previous version to current:\n```diff\n{diff}\n```"
    else:
        body = f"Previous better implementation:\n```python\n{bare_code(prev_prog.code)}\n```"
    return (
        "[Recent Step That Regressed]\n"
        f"Previous version fitness = {fitness_text(prev_prog.fitness)} -> "
        f"current fitness = {fitness_text(cur_prog.fitness)} (regressed)\n"
        f"{body}"
    )


def _current_block(program: Program) -> str:
    return (
        "[Current Algorithm]\n"
        f"fitness = {fitness_text(program.fitness)}\n"
        f"```python\n{bare_code(program.code)}\n```"
    )


def _fit_named_blocks(
    optional_blocks: list[tuple[str, str]], required_block: str
) -> tuple[str, set[str]]:
    """保留必选块，按顺序纳入可选块，返回最终文本与实际保留的区块名集合。"""
    kept: list[str] = []
    included: set[str] = set()
    for name, block in optional_blocks:
        if not block:
            continue
        candidate = "\n\n".join([*kept, block, required_block])
        if len(candidate) <= CONTEXT_CHARS:
            kept.append(block)
            included.add(name)
    return "\n\n".join([*kept, required_block]), included


def _relative_donor_label(parent: Program, donor: Program) -> str:
    p_fit = float(parent.fitness)
    d_fit = float(donor.fitness)
    if beats(donor, parent):
        pct = 100.0 * abs(p_fit - d_fit) / max(abs(p_fit), 1e-12)
        return f"better than primary by {pct:.2g}%"
    if beats(parent, donor):
        pct = 100.0 * abs(d_fit - p_fit) / max(abs(p_fit), 1e-12)
        return f"worse than primary by {pct:.2g}%"
    return "tied with primary"


def build_evolution_context(
    tree: TreePopulation,
    archive: dict[int, Program],
    parent_id: int,
    operator: Operator,
    reference_id: int | None = None,
    *,
    explore_ids: Sequence[int] = (),
    show_memory: bool = True,
    cross_lineage: bool = True,
    exclude_failures: frozenset[int] = frozenset(),
) -> tuple[str, str, dict]:
    """构造三算子隔离上下文、指令与裁剪后的准确记忆暴露诊断信息。"""
    if operator == "Explore":
        cards = []
        for idx, nid in enumerate(explore_ids, 1):
            if nid in archive:
                prog = archive[nid]
                cards.append(
                    f"Lineage {idx}: fitness = {fitness_text(prog.fitness)}\n"
                    f"Design: {clean_text(prog.design)}"
                )
        context = (
            "[Existing Active Lineages]\n"
            "Ordered from best to worst fitness (one leading representative per distinct active lineage).\n\n"
            + "\n\n".join(cards)
            if cards
            else ""
        )
        info = {
            "memory_available": False,
            "memory_source": "none",
            "memory_assigned": False,
            "memory_shown": False,
            "memory_masked": False,
            "memory_truncated": False,
        }
        return context, EXPLORE_INSTRUCTION, info

    parent = archive[parent_id]
    if operator == "Crossover":
        if reference_id is None or reference_id not in archive:
            raise ValueError("Crossover requires a valid reference_id")
        donor = archive[reference_id]
        primary_block = (
            "[Primary Algorithm]\n"
            f"fitness = {fitness_text(parent.fitness)}\n"
            f"Design: {clean_text(parent.design)}\n"
            f"```python\n{bare_code(parent.code)}\n```"
        )
        heading = (
            "[Complementary Algorithm (Different Lineage)]"
            if cross_lineage
            else "[Complementary Algorithm]"
        )
        donor_block = (
            f"{heading}\n"
            f"fitness = {fitness_text(donor.fitness)} ({_relative_donor_label(parent, donor)})\n"
            f"Design: {clean_text(donor.design)}\n"
            f"```python\n{bare_code(donor.code)}\n```"
        )
        context = f"{primary_block}\n\n{donor_block}"
        instruction = CROSSOVER_INSTRUCTION_CROSS if cross_lineage else CROSSOVER_INSTRUCTION_SAME
        info = {
            "memory_available": False,
            "memory_source": "none",
            "memory_assigned": False,
            "memory_shown": False,
            "memory_masked": False,
            "memory_truncated": False,
            "cross_lineage": cross_lineage,
        }
        return context, instruction, info

    if operator != "Refine":
        raise ValueError(f"Unsupported evolution operator: {operator!r}")

    notes, memory_source = _local_unrealized_notes(tree, parent_id)
    memory_available = bool(notes)
    memory_assigned = memory_available and show_memory
    memory_masked = memory_available and not show_memory

    memory_block = ""
    if memory_assigned:
        memory_block = (
            "[Unrealized Ideas Worth a Second Chance]\n"
            "These ideas come from earlier evicted trials in this lineage; a low score alone does not "
            "prove whether the idea or its implementation failed. Skip any idea already implemented "
            "in the current algorithm.\n" + "\n".join(notes)
        )

    regressed_block = _regressed_step_block(tree, archive, parent_id)
    active_block = _active_branches_and_failures_block(
        tree, archive, parent_id, exclude_failures=exclude_failures
    )
    required = _current_block(parent)
    # 固定非记忆区块优先排布，确保 memory_block 绝不挤出另一臂存在的常规上下文
    context, included = _fit_named_blocks(
        [("regressed", regressed_block), ("active", active_block), ("memory", memory_block)],
        required,
    )
    memory_shown = "memory" in included
    info = {
        "memory_available": memory_available,
        "memory_source": memory_source,
        "memory_assigned": memory_assigned,
        "memory_shown": memory_shown,
        "memory_masked": memory_masked,
        "memory_truncated": memory_assigned and not memory_shown,
        "regressed_step_shown": "regressed" in included,
    }
    return context, REFINE_INSTRUCTION, info


def _render(task: Task, context: str, instruction: str, *, has_scores: bool = False) -> str:
    template = COMMON_TEMPLATE
    if not context:
        template = template.replace("[Context]\n{context}\n\n", "")
    elif has_scores and task.score_definition not in task.task_description:
        context = f"Score definition: {task.score_definition}\n\n{context}"
    return template.format_map(
        {
            "task_description": task.task_description.strip(),
            "function_template": task.function_template.strip(),
            "context": context,
            "instruction": instruction,
        }
    )


def evolution_prompt(
    task: Task,
    tree: TreePopulation,
    archive: dict[int, Program],
    parent_id: int,
    operator: Operator,
    reference_id: int | None = None,
    *,
    explore_ids: Sequence[int] = (),
    show_memory: bool = True,
    cross_lineage: bool = True,
    exclude_failures: frozenset[int] = frozenset(),
) -> tuple[str, dict]:
    context, instruction, info = build_evolution_context(
        tree,
        archive,
        parent_id,
        operator,
        reference_id,
        explore_ids=explore_ids,
        show_memory=show_memory,
        cross_lineage=cross_lineage,
        exclude_failures=exclude_failures,
    )
    return _render(task, context, instruction, has_scores=True), info


def init_prompt(task: Task, archive: dict[int, Program], failure_count: int = 2) -> str:
    blocks: list[str] = []
    if 0 in archive:
        example = archive[0]
        header = "[Example Algorithm]\n"
        if design := clean_text(example.design):
            header += f"Design: {design}\n"
        blocks.append(header + f"```python\n{bare_code(example.code)}\n```")

    valid = sorted(
        (p for p in archive.values() if p.valid and p.operator == "Init" and p.duplicate_of is None),
        key=lambda p: (float(p.fitness), p.id),
    )
    if valid:
        items = [
            f"Approach {idx}: fitness = {fitness_text(p.fitness)}\nDesign: {clean_text(p.design)}"
            for idx, p in enumerate(valid, 1)
        ]
        blocks.append("[Existing Initial Approaches]\nFrom best to worst fitness.\n\n" + "\n\n".join(items))

    failed = (
        sorted(
            (
                p
                for p in archive.values()
                if p.id != 0
                and p.parent_id == ROOT_ID
                and p.evaluation is not None
                and not p.valid
            ),
            key=lambda p: p.id,
        )[-failure_count:]
        if failure_count
        else []
    )
    if failed:
        items = [
            f"Failed attempt {idx}\nDesign: {clean_text(p.design)}\n"
            f"```python\n{p.code.rstrip()}\n```\n"
            f"Failure: {failure_feedback(p.evaluation.failure, p.code)}"
            for idx, p in enumerate(failed, 1)
        ]
        blocks.append("[Recent Failed Attempts]\n" + "\n\n".join(items))

    if valid:
        instruction = (
            "Develop an effective algorithm with a fundamentally different main computation from the "
            "initial approaches already represented above. Do not produce another re-weighting of the "
            "same local features; change the information used, representation, or computational procedure."
        )
    else:
        instruction = (
            "Design an effective algorithm for this task from first principles. Consider how the returned "
            "decision shapes the complete solver's solution and choose a computation suited to that role."
        )
        if 0 in archive:
            instruction += " The example illustrates the interface only, not a structure to preserve."
    if failed:
        instruction += "\nUse the failed attempts and error feedback to avoid repeating those runtime or timeout issues."
    return _render(task, "\n\n".join(blocks), instruction, has_scores=bool(valid))


def repair_format_prompt(task: Task, response: str, feedback: str) -> str:
    context = f"[Original Response]\n{response}\n\n[Parsing Feedback]\n{feedback}"
    instruction = (
        "Convert the original response into the exact JSON object specified in [Output], using the "
        "parsing feedback to fix structure, required fields, and string escaping.\n"
        "Preserve the existing algorithmic idea and implementation logic cleanly."
    )
    return _render(task, context, instruction)


def repair_program_prompt(task: Task, program: Program) -> str:
    failure = program.evaluation.failure or ""
    is_timeout = program.evaluation.status == "timeout" or failure.startswith(TIMEOUT_PREFIX)
    feedback = (
        f"The evaluation did not finish within the total time limit of {task.time_limit:g} seconds."
        if is_timeout
        else failure_feedback(failure, program.code)
    )
    context = (
        f"[Current Algorithm]\nDesign: {clean_text(program.design)}\n"
        f"```python\n{program.code.rstrip()}\n```\n\n[Failure Feedback]\n{feedback}"
    )
    instruction = (
        "Make the algorithm finish well within the reported time limit. "
        "Reduce the asymptotic or constant-factor cost of its core computation, using a clean "
        "approximation if needed, while preserving the intended algorithmic idea."
        if is_timeout
        else (
            "Trace the reported exception or invalid result to its exact cause in the code, using "
            "the design to understand the intended computation.\n"
            "Fix the faulty logic or data handling so the function executes cleanly and satisfies "
            "all interface and task constraints while keeping the intended core mechanism."
        )
    )
    return _render(task, context, instruction)
