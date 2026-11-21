"""原位局部反思：随路径收缩与分支剪枝凝练一段基于源码变化与实测结果的算法设计经验。"""

import difflib
from contextlib import nullcontext
from typing import Any

from .evaluation import compact_failure
from .models import (
    Program,
    TreePopulation,
    clean_text,
    fitness_text,
    json_schema,
    parse_json_object,
)
from .topology import bare_code, better

EXPERIENCE_SCHEMA = json_schema("evotrace_experience", "analysis", "experience")

MAX_DIFF_LINES = 24
MAX_DIFF_CHARS = 1600
REWRITE_SIMILARITY = 0.3

PROMPT_TEMPLATE = """{problem}

[Situation]
{situation}

[Comparisons]
{records}

[Task]
{question}
Ground every claim in the actual code and measured outcomes shown above:
- Identify each computational component by what it computes and how it enters the decision, not by its variable name.
- Attribute a gain or loss to a single change only when a step isolates it; when several changes occur together, report their combined outcome and mark single-change explanations as hypotheses.
- Record only applicable conditions, observed computational changes, and measured outcomes; do not prescribe what to try next.
- Capture useful observations from both improving and losing steps, including any promising component inside a losing branch.{merge}

[Output]
Return a JSON object with two string fields:
- "analysis": Briefly identify the actual computational changes between the programs and how the score changed (up to 150 words).
- "experience": Write one concise, factual paragraph (about 60-120 words) of design experience grounded in the code changes and measured outcomes. Return an empty string only if there is nothing new to record or revise."""

MERGE_GUIDANCE = (
    "\n- Earlier experience summarizes prior comparisons here. Merge still-valid lessons with the "
    "new results into one updated paragraph; narrow or retract any earlier claim contradicted by "
    "new evidence while keeping the conditions under which the earlier result occurred."
)


def outcome_summary(base: Program | None, candidate: Program) -> str:
    """纯黑盒实测目标值与相对父代的胜负摘要。"""
    ev = candidate.evaluation
    if ev is None:
        return "not evaluated"
    if ev.status != "success" or ev.fitness is None:
        return f"failed - {compact_failure(ev.failure, candidate.code, status=ev.status)}"
    cand_fit = fitness_text(ev.fitness)
    if base is None or not base.valid or base.evaluation is None or base.evaluation.fitness is None:
        return f"fitness = {cand_fit}"
    base_fit = fitness_text(base.evaluation.fitness)
    verdict = (
        "improved"
        if better(ev.fitness, base.evaluation.fitness)
        else "regressed"
        if better(base.evaluation.fitness, ev.fitness)
        else "tied"
    )
    return f"{base_fit} -> {cand_fit} ({verdict})"


def compact_code_diff(
    code_a: str,
    code_b: str,
    *,
    max_lines: int | None = MAX_DIFF_LINES,
    max_chars: int | None = MAX_DIFF_CHARS,
    omit_if_truncated: bool = False,
) -> str:
    """紧凑源码差异；无变化或整体重写返回空串，omit_if_truncated=True 时超长亦略去。"""
    lines_a = [line.rstrip() for line in bare_code(code_a).splitlines() if line.strip()]
    lines_b = [line.rstrip() for line in bare_code(code_b).splitlines() if line.strip()]
    if lines_a == lines_b:
        return ""
    kept_lines = sum(
        b.size
        for b in difflib.SequenceMatcher(None, lines_a, lines_b, autojunk=False).get_matching_blocks()
    )
    if kept_lines < REWRITE_SIMILARITY * max(len(lines_a), len(lines_b)):
        return ""
    diff = [
        line
        for line in difflib.unified_diff(lines_a, lines_b, n=1, lineterm="")
        if not line.startswith(("---", "+++"))
    ]
    if not diff:
        return ""
    kept: list[str] = []
    total = 0
    for line in diff:
        if (max_lines is not None and len(kept) >= max_lines) or (
            max_chars is not None and total + len(line) + 1 > max_chars
        ):
            if omit_if_truncated:
                return ""
            kept.append(f"... (+{len(diff) - len(kept)} more diff lines)")
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)


def _program_block(
    tree: TreePopulation,
    archive: dict[int, Program],
    node_id: int,
    *,
    label: str,
    parent_prog: Program | None = None,
    parent_label: str = "",
    anchor_prog: Program | None = None,
    include_edge_note: bool = True,
    node_note_label: str | None = None,
) -> str:
    prog = archive[node_id]
    head = (
        f"{label} (from {parent_label}): {outcome_summary(parent_prog, prog)}"
        if parent_prog is not None
        else f"{label}: {outcome_summary(None, prog)}"
    )
    if anchor_prog is not None and anchor_prog is not parent_prog:
        head += f" | vs Start of path: {outcome_summary(anchor_prog, prog)}"
    lines = [head]
    if parent_prog is not None and (
        diff := compact_code_diff(parent_prog.code, prog.code, omit_if_truncated=True)
    ):
        lines.append(f"Code change from {parent_label}:\n{diff}")
    if include_edge_note and (edge_note := clean_text(tree.edge_notes.get(node_id))):
        lines.append(f"Earlier path experience leading into this step: {edge_note}")
    if node_note_label and (node_note := clean_text(tree.node_notes.get(node_id))):
        lines.append(f"{node_note_label}: {node_note}")
    lines.append(f"```python\n{bare_code(prog.code)}\n```")
    return "\n".join(lines)


def _collect_prior_texts(tree: TreePopulation, event: Any) -> list[str]:
    removed_notes = [
        note
        for nid in event.removed
        for note in (tree.edge_notes.get(nid), tree.node_notes.get(nid))
    ]
    notes = (
        [*removed_notes, tree.edge_notes.get(event.target)]
        if event.kind == "path"
        else [tree.node_notes.get(event.target), *removed_notes]
    )
    return list(dict.fromkeys(cleaned for n in notes if (cleaned := clean_text(n))))


def build_reflection_prompt(
    task: Any, tree: TreePopulation, archive: dict[int, Program], event: Any
) -> tuple[str, list[str]]:
    """构造局部淘汰区域的反思提示，返回 (prompt, 先前去重经验列表)。"""
    prior_texts = _collect_prior_texts(tree, event)
    anchor_prog = archive[event.anchor]
    records: list[str] = []

    if event.kind == "path":
        target_prog = archive[event.target]
        records.append(
            _program_block(
                tree, archive, event.anchor, label="Start of path", include_edge_note=False
            )
        )
        prev_prog, prev_label = anchor_prog, "Start of path"
        for idx, nid in enumerate(event.removed, 1):
            step_label = f"Intermediate step {idx}"
            records.append(
                _program_block(
                    tree,
                    archive,
                    nid,
                    label=step_label,
                    parent_prog=prev_prog,
                    parent_label=prev_label,
                    node_note_label="Side-branch experience previously pruned onto this step",
                )
            )
            prev_prog, prev_label = archive[nid], step_label
        records.append(
            _program_block(
                tree,
                archive,
                event.target,
                label="End of path",
                parent_prog=prev_prog,
                parent_label=prev_label,
                anchor_prog=anchor_prog,
            )
        )
        net_diff = compact_code_diff(anchor_prog.code, target_prog.code, omit_if_truncated=True)
        situation = (
            "The final algorithm was developed along a linear path from Start of path to End of path.\n"
            f"Net outcome (Start of path -> End of path): {outcome_summary(anchor_prog, target_prog)}."
            + (f"\nNet code change (Start of path -> End of path):\n{net_diff}" if net_diff else "")
        )
        question = (
            "From Start of path to End of path, which computational changes remained in the final implementation, "
            "which step introduced regression or useful side-branch observations, and where did recovery or improvement occur?"
        )
    else:
        records.append(
            _program_block(
                tree,
                archive,
                event.anchor,
                label="Retained parent",
                include_edge_note=False,
                node_note_label="Earlier branch experience at this node",
            )
        )
        for idx, nid in enumerate(event.removed, 1):
            records.append(
                _program_block(
                    tree,
                    archive,
                    nid,
                    label=f"Retracted attempt {idx}",
                    parent_prog=anchor_prog,
                    parent_label="Retained parent",
                    node_note_label="Earlier sub-branch experience folded into this attempt",
                )
            )
        situation = (
            f"{len(event.removed)} weaker branch attempt(s) derived from Retained parent are being folded back into it."
        )
        question = (
            "Which distinct computational changes from Retained parent were tested, how did each affect the score, "
            "and is any component still worth borrowing despite the branch losing overall?"
        )

    template = (
        getattr(task, "function_template", None) or getattr(task, "template_program", "")
    ).strip()
    problem_lines = [
        "[Problem]",
        getattr(task, "task_description", "").strip(),
        f"Score: {getattr(task, 'score_definition', 'Task objective; lower is better.').strip()}",
    ]
    if template:
        problem_lines.append(f"Function interface:\n```python\n{template}\n```")

    prompt = PROMPT_TEMPLATE.format(
        problem="\n".join(problem_lines),
        situation=situation,
        records="\n\n".join(records),
        question=question,
        merge=MERGE_GUIDANCE if prior_texts else "",
    )
    return prompt, prior_texts


def reflect_region(
    task: Any,
    tree: TreePopulation,
    archive: dict[int, Program],
    event: Any,
    model: Any = None,
    timeout: float = 180.0,
    *,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> tuple[str, dict | None]:
    """对操作前的树与单个局部淘汰事件执行原位反思，返回 (experience_text, aux_report | None)。"""
    if model is None:
        return " ".join(_collect_prior_texts(tree, event)), None

    prompt, prior_texts = build_reflection_prompt(task, tree, archive, event)
    usage: dict[str, Any] = {}
    error = analysis = raw_response = None
    experience_text = ""

    try:
        with getattr(model, "request_context", nullcontext)():
            reply = model.complete(
                prompt, temperature=temperature, max_tokens=max_tokens, timeout=timeout
            )
        raw_response = reply if isinstance(reply, str) else getattr(reply, "content", None)
        usage = dict(getattr(reply, "usage", None) or {})
        payload = parse_json_object(raw_response)
        if not isinstance(payload.get("experience"), str):
            raise TypeError(
                "Reflection reply must be a JSON object with a string 'experience' field"
            )
        experience_text = clean_text(payload["experience"])
        if isinstance(payload.get("analysis"), str):
            analysis = clean_text(payload["analysis"])
    except Exception as exc:  # noqa: BLE001 - fall back to prior experience on backend/parse failure
        error = f"{type(exc).__name__}: {exc}"

    final_text = experience_text or " ".join(prior_texts)
    aux = {
        "role": "reflection",
        "kind": event.kind,
        "anchor": event.anchor,
        "target": event.target,
        "removed": list(event.removed),
        "status": "error" if error else "success",
        "usage": usage,
        "error": error,
        "prompt": prompt,
        "response": raw_response,
        "analysis": analysis,
        "experience": final_text,
    }
    return final_text, aux
