"""演化脉络上下文与三算子提示组织：Init、Repair 及 Refine / Crossover / Explore。"""

from collections.abc import Sequence
from typing import Any

from .evaluation import TIMEOUT_PREFIX, compact_failure, failure_feedback
from .models import ROOT_ID, Operator, Program, TreePopulation, clean_text, fitness_text
from .reflection import compact_code_diff, outcome_summary
from .topology import bare_code, better

INIT_ANALYSIS = (
    "Briefly state your core algorithmic idea and how you expect it to improve the solver's "
    "decisions (up to 150 words)."
)
EVO_ANALYSIS = (
    "Briefly state the chosen change and how you expect it to affect the solver's choices "
    "as a testable hypothesis (up to 200 words)."
)
EVO_ANALYSIS_WITH_EXP = (
    "Briefly state the chosen change, how you expect it to affect the solver's choices "
    "(as a testable hypothesis), and why any relevant experience applies or warrants re-testing "
    "(up to 200 words)."
)
EXPERIENCE_NOTE = (
    " Treat earlier experiences as condition-dependent observations that may be re-tested "
    "under a different computation."
)
EXPERIENCE_MARKERS = (
    "Path experience:",
    "Branch experience:",
    "Retracted attempts experience:",
)

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
- "analysis": {analysis_instruction}
- "code": Provide the complete implementation using only the inputs in [Function interface], including necessary imports and helper definitions. Do not write explanatory comments; put the reasoning in "analysis".
- "design": In about 120 words, describe for another algorithm designer the main computation the code performs and its key parameters. Describe the implementation, not expected performance.
Keep "design" faithful to "code". Preserve the required function signature and satisfy the task constraints and input/output requirements.
Return only the JSON object, without Markdown fences or additional text."""


def _node_experiences(
    tree: TreePopulation, node_id: int, *, include_node: bool = True
) -> list[str]:
    lines = []
    if note := clean_text(tree.edge_notes.get(node_id)):
        lines.append(f"Path experience: {note}")
    if include_node and (note := clean_text(tree.node_notes.get(node_id))):
        lines.append(f"Branch experience: {note}")
    return lines


def _current_code_block(prog: Program, *, include_design: bool = False) -> str:
    lines = ["[Current Algorithm]", outcome_summary(None, prog)]
    if include_design:
        lines.append(f"Reported design: {clean_text(prog.design)}")
    lines.append(f"```python\n{bare_code(prog.code)}\n```")
    return "\n".join(lines)


def _donor_comparison(parent: Program, donor: Program) -> str:
    base = outcome_summary(None, donor)
    if not (parent.valid and donor.valid):
        return base
    p_fit = fitness_text(parent.fitness)
    if better(donor.fitness, parent.fitness):
        relation = f"better than current {p_fit}"
    elif better(parent.fitness, donor.fitness):
        relation = f"worse than current {p_fit}"
    else:
        relation = f"tied with current {p_fit}"
    return f"{base} ({relation})"


def _lineage_sections(
    tree: TreePopulation,
    archive: dict[int, Program],
    parent_id: int,
    *,
    include_experiences: bool,
    exclude: int | None = None,
) -> tuple[list[str], str | None, set[int]]:
    """构造主父代的纵向演化步骤列表与本地试错区块（单节点路径免去重复的开发区块）。"""
    path = tree.path_to(parent_id)
    seen_nodes = set(path)

    path_steps: list[str] = []
    if len(path) > 1:
        prev_prog: Program | None = None
        for k, nid in enumerate(path):
            prog = archive[nid]
            label = ("First version" if k == 0 else f"Step {k}") + (
                ", the current algorithm" if nid == parent_id else ""
            )
            lines = [
                f"{label}: {outcome_summary(prev_prog, prog)}",
                f"Reported design: {clean_text(prog.design)}",
            ]
            if prev_prog is not None and (diff := compact_code_diff(prev_prog.code, prog.code)):
                lines.append(f"Code change:\n{diff}")
            if include_experiences:
                lines.extend(_node_experiences(tree, nid, include_node=(nid != parent_id)))
            path_steps.append("\n".join(lines))
            prev_prog = prog

    parent_prog = archive[parent_id]
    attempts: list[str] = []
    if include_experiences and (note := clean_text(tree.node_notes.get(parent_id))):
        attempts.append(f"Retracted attempts experience: {note}")

    active_children = sorted(
        cid
        for cid, pid in tree.parents.items()
        if pid == parent_id and cid != exclude and cid in archive
    )
    if active_children:
        attempts.append(
            "Existing active attempts below are already being explored as separate branches from "
            "the current algorithm. Do not re-implement them; pursue a distinct change."
        )
        for k, cid in enumerate(active_children, 1):
            child = archive[cid]
            lines = [
                f"Active attempt {k}: {outcome_summary(parent_prog, child)}",
                f"Reported design: {clean_text(child.design)}",
            ]
            if include_experiences:
                lines.extend(_node_experiences(tree, cid))
            attempts.append("\n".join(lines))
            seen_nodes.add(cid)

    off_tree = [
        archive[pid]
        for pid in sorted(archive)
        if pid != ROOT_ID and pid not in tree.parents and archive[pid].parent_id == parent_id
    ]
    if dups := sum(p.duplicate_of is not None for p in off_tree):
        attempts.append(
            f"Exact code duplicates: {dups} earlier attempt(s) from this parent reproduced an "
            "existing algorithm verbatim without changing the computation."
        )
    failures = [p for p in off_tree if p.duplicate_of is None and p.evaluation and not p.valid]
    for prog in failures[-MAX_RECENT_FAILURES:]:
        reason = compact_failure(prog.evaluation.failure, prog.code, status=prog.evaluation.status)
        attempts.append(
            f"Failed attempt ({prog.operator or 'Refine'}): {reason}\n"
            f"Reported design: {clean_text(prog.design)}"
        )

    attempts_block = (
        "\n\n".join(["[Attempts from the Current Algorithm]", *attempts]) if attempts else None
    )
    return path_steps, attempts_block, seen_nodes


def _build_explore_context(
    tree: TreePopulation,
    archive: dict[int, Program],
    parent_id: int,
    explore_ids: Sequence[int],
    *,
    include_experiences: bool,
) -> tuple[str, str]:
    blocks = [_current_code_block(archive[parent_id], include_design=True)]
    refs = sorted(
        {
            rid
            for rid in explore_ids
            if rid != parent_id and rid in archive and archive[rid].fitness is not None
        },
        key=lambda i: (archive[i].fitness, i),
    )

    shown = [parent_id]
    labels = {parent_id: "Current Algorithm"}
    ref_cards: list[str] = []
    for idx, rid in enumerate(refs, 1):
        rep = archive[rid]
        label = f"Approach {idx}"
        card = f"{label}: {outcome_summary(None, rep)}\nReported design: {clean_text(rep.design)}"
        candidate_block = "\n\n".join(
            ["[Other Existing Approaches]\nFrom best to worst fitness.", *ref_cards, card]
        )
        if len("\n\n".join([*blocks, candidate_block])) <= CONTEXT_CHARS:
            ref_cards.append(card)
            shown.append(rid)
            labels[rid] = label
    if ref_cards:
        blocks.append(
            "\n\n".join(["[Other Existing Approaches]\nFrom best to worst fitness.", *ref_cards])
        )

    has_exp = False
    if include_experiences:
        seen_notes: set[str] = set()
        exp_lines: list[str] = []
        for nid in shown:
            for note in _node_experiences(tree, nid):
                if note in seen_notes:
                    continue
                seen_notes.add(note)
                entry = f"{labels[nid]} — {note}"
                candidate_block = "\n\n".join(["[Relevant Experiences]", *exp_lines, entry])
                if len("\n\n".join([*blocks, candidate_block])) <= CONTEXT_CHARS:
                    exp_lines.append(entry)
        if exp_lines:
            blocks.append("\n\n".join(["[Relevant Experiences]", *exp_lines]))
            has_exp = True

    instruction = (
        "Develop a genuinely different way to use the available information, represent the problem, "
        "or compute the result compared with the current algorithm and existing approaches. "
        "State how you expect the new computation to change the solver's choices."
        + (EXPERIENCE_NOTE if has_exp else "")
    )
    return "\n\n".join(blocks), instruction


def _build_donor_block(
    tree: TreePopulation,
    archive: dict[int, Program],
    parent_id: int,
    donor_id: int,
    *,
    include_experiences: bool,
    seen_nodes: set[int],
    used_chars: int,
) -> str:
    parent, donor = archive[parent_id], archive[donor_id]
    intro = "A donor algorithm whose distinct computation may complement the current algorithm."
    if parent.fitness is not None and donor.fitness is not None and donor.fitness >= parent.fitness:
        intro += (
            " Although this donor does not beat the current algorithm overall, "
            "a specific component in it may still help."
        )

    lines = [
        "[Donor Algorithm]",
        intro,
        _donor_comparison(parent, donor),
        f"Reported design: {clean_text(donor.design)}",
    ]
    donor_code = bare_code(donor.code)
    code_section = (
        "Same displayed code as the current algorithm."
        if donor_code == bare_code(parent.code)
        else f"```python\n{donor_code}\n```"
    )
    if include_experiences:
        if donor_id not in seen_nodes:
            lines.extend(_node_experiences(tree, donor_id))
            seen_nodes.add(donor_id)
        for anc_id in tree.path_to(donor_id)[:-1]:
            if anc_id in seen_nodes:
                continue
            seen_nodes.add(anc_id)
            if anc_notes := _node_experiences(tree, anc_id):
                line = "Donor lineage: " + " | ".join(anc_notes)
                if used_chars + len("\n".join([*lines, line, code_section])) <= CONTEXT_CHARS:
                    lines.append(line)
    lines.append(code_section)
    return "\n".join(lines)


def build_evolution_context(
    tree: TreePopulation,
    archive: dict[int, Program],
    parent_id: int,
    operator: Operator,
    reference_id: int | None = None,
    explore_ids: Sequence[int] = (),
    include_experiences: bool = True,
) -> tuple[str, str]:
    """按算子返回 (演化上下文文本, 指令文本)。"""
    if operator == "Explore":
        return _build_explore_context(
            tree, archive, parent_id, explore_ids, include_experiences=include_experiences
        )
    if operator not in ("Refine", "Crossover"):
        raise ValueError(f"Unsupported evolution operator: {operator!r}")
    donor = reference_id if operator == "Crossover" else None
    if operator == "Crossover" and donor not in archive:
        raise ValueError(f"Crossover requires a valid reference_id in archive, got {donor!r}")

    path_steps, attempts_block, seen_nodes = _lineage_sections(
        tree, archive, parent_id, include_experiences=include_experiences, exclude=donor
    )
    current = _current_code_block(archive[parent_id], include_design=(not path_steps))

    tail: list[str] = []
    if attempts_block:
        tail.append(attempts_block)
    if donor is not None:
        used_chars = len("\n\n".join([*path_steps, *tail, current]))
        tail.append(
            _build_donor_block(
                tree,
                archive,
                parent_id,
                donor,
                include_experiences=include_experiences,
                seen_nodes=seen_nodes,
                used_chars=used_chars,
            )
        )
    tail.append(current)

    # 超长路径时优先裁剪最早祖先步骤，始终保留当前算法、本地尝试与供体。
    while len(path_steps) > 1 and (
        len("\n\n".join(["[Development of the Current Algorithm]", *path_steps, *tail]))
        > CONTEXT_CHARS
    ):
        path_steps.pop(0)

    blocks = (
        ["\n\n".join(["[Development of the Current Algorithm]", *path_steps]), *tail]
        if path_steps
        else tail
    )
    context = "\n\n".join(blocks)
    has_exp = include_experiences and any(m in context for m in EXPERIENCE_MARKERS)

    if donor is not None:
        instruction = (
            "Identify a specific computation from the donor algorithm that addresses a weakness of the current algorithm, "
            "and integrate it into the current implementation. Consider how the solver's choices will change after integration "
            "and whether the conditions under which that computation worked still hold. A weaker donor may still contribute "
            "a useful component."
        )
    else:
        instruction = (
            "Identify situations where the current implementation may make poor choices, and modify its computation "
            "to improve the objective. You may refine specific details or replace a substantial part of the implementation."
            + (EXPERIENCE_NOTE if has_exp else "")
        )
    return context, instruction


def _render(
    task: Any,
    context: str,
    instruction: str,
    *,
    has_scores: bool = False,
    analysis_instruction: str = EVO_ANALYSIS,
) -> str:
    task_desc = getattr(task, "task_description", "")
    score_def = getattr(task, "score_definition", "Task objective; lower is better.")
    template = COMMON_TEMPLATE if context else COMMON_TEMPLATE.replace("[Context]\n{context}\n\n", "")
    if context and has_scores and score_def and score_def not in task_desc:
        context = f"Score definition: {score_def}\n\n{context}"
    return template.format_map(
        {
            "task_description": task_desc,
            "function_template": getattr(task, "function_template", None)
            or getattr(task, "template_program", ""),
            "context": context,
            "instruction": instruction,
            "analysis_instruction": analysis_instruction,
        }
    )


def evolution_prompt(
    task: Any,
    tree: TreePopulation,
    archive: dict[int, Program],
    parent_id: int,
    operator: Operator,
    reference_id: int | None = None,
    explore_ids: Sequence[int] = (),
    include_experiences: bool = True,
) -> str:
    context, instruction = build_evolution_context(
        tree, archive, parent_id, operator, reference_id, explore_ids, include_experiences
    )
    has_exp = include_experiences and any(m in context for m in EXPERIENCE_MARKERS)
    return _render(
        task,
        context,
        instruction,
        has_scores=True,
        analysis_instruction=EVO_ANALYSIS_WITH_EXP if has_exp else EVO_ANALYSIS,
    )


def init_prompt(task: Any, archive: dict[int, Program], failure_count: int = 2) -> str:
    blocks: list[str] = []
    if ROOT_ID in archive:
        ex = archive[ROOT_ID]
        design = f"Design: {d}\n" if (d := clean_text(ex.design)) else ""
        blocks.append(f"[Example Algorithm]\n{design}```python\n{bare_code(ex.code)}\n```")

    valid = sorted(
        (p for p in archive.values() if p.operator == "Init" and p.valid and p.duplicate_of is None),
        key=lambda p: (p.fitness, p.id),
    )
    if valid:
        rows = [
            f"Approach {k}: {outcome_summary(None, p)}\nReported design: {clean_text(p.design)}"
            for k, p in enumerate(valid, 1)
        ]
        blocks.append("[Existing Approaches]\nFrom best to worst fitness.\n\n" + "\n\n".join(rows))

    failed = (
        [
            archive[i]
            for i in sorted(archive)
            if i != ROOT_ID
            and archive[i].parent_id == ROOT_ID
            and archive[i].evaluation is not None
            and not archive[i].valid
        ][-failure_count:]
        if failure_count > 0
        else []
    )
    if failed:
        rows = []
        for k, prog in enumerate(failed, 1):
            ev = prog.evaluation
            fb = (
                "Execution timed out."
                if ev.status == "timeout" or (ev.failure or "").startswith(TIMEOUT_PREFIX)
                else failure_feedback(ev.failure, prog.code)
            )
            rows.append(
                f"Failed attempt {k}\nReported design: {clean_text(prog.design)}\n"
                f"```python\n{prog.code.rstrip()}\n```\nFailure: {fb}"
            )
        blocks.append("[Failed Attempts]\n" + "\n\n".join(rows))

    if valid:
        instruction = (
            "Develop an effective algorithm with a different main computation from the ideas already represented. "
            "Seek a different way to compute the result, rather than another weighting of the same local features. "
            "You can change the representation, the information used, or the computational procedure. "
            "Implement the chosen approach as a complete algorithm."
        )
    else:
        example_note = (
            " The example illustrates an implementation, not a structure to preserve."
            if ROOT_ID in archive
            else ""
        )
        instruction = (
            "Design an effective algorithm for this task. Consider how the returned result affects the complete solution "
            f"and choose a computational approach suited to that role.{example_note} Implement the specified function."
        )
    if failed:
        instruction += "\nUse the failed attempts and failure feedback to avoid repeating the same execution errors."
    return _render(
        task,
        "\n\n".join(blocks),
        instruction,
        has_scores=bool(valid),
        analysis_instruction=INIT_ANALYSIS,
    )


def repair_format_prompt(task: Any, response: str, feedback: str) -> str:
    context = f"[Original Response]\n{response[:CONTEXT_CHARS]}\n\n[Parsing Feedback]\n{feedback}"
    instruction = (
        "Convert the original response into the JSON object specified in [Output], using the "
        "parsing feedback to correct its structure, required fields, and string escaping.\n"
        "Preserve the existing algorithmic idea and computational logic.\n"
        "Complete any missing or truncated content consistently with that approach and the "
        "task requirements."
    )
    return _render(
        task,
        context,
        instruction,
        analysis_instruction="Keep the original rationale if present. Otherwise briefly state the intended computation.",
    )


def repair_program_prompt(task: Any, program: Program) -> str:
    ev = program.evaluation
    assert ev is not None
    failure = (ev.failure or "").strip()
    is_timeout = ev.status == "timeout" or failure.startswith(TIMEOUT_PREFIX)
    limit = getattr(task, "time_limit", None)
    if is_timeout:
        feedback = failure or (
            f"The evaluation did not finish within the total time limit of {limit:g} seconds."
            if limit is not None
            else "The evaluation timed out."
        )
        if limit is not None and f"{limit:g}" not in feedback:
            feedback = f"{feedback} (time limit: {limit:g} seconds)"
        instruction = (
            "Make the algorithm finish within the reported time limit. "
            "Reduce the cost of its core computation, using an approximation when needed, "
            "while retaining the intended algorithmic idea."
        )
    else:
        feedback = failure_feedback(failure, program.code)
        instruction = (
            "Trace the reported exception or invalid result to its cause in the code, using the "
            "design to understand the intended algorithm.\n"
            "Correct the faulty computations or data handling so the function executes correctly "
            "and satisfies the task requirements, while retaining the intended core computation."
        )
    context = (
        f"[Current Algorithm]\nReported design: {clean_text(program.design)}\n"
        f"```python\n{program.code.rstrip()}\n```\n\n[Failure Feedback]\n{feedback}"
    )
    return _render(
        task,
        context,
        instruction,
        analysis_instruction="Explain the correction for the reported failure.",
    )
