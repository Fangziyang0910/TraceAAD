"""剪枝反思：仅在容量淘汰叶节点时基于完整代码证据提取未兑现想法，并在跨节点继承或合并时对照存活源码清理。"""

from __future__ import annotations

import difflib
import re
from contextlib import nullcontext
from typing import Any

from .evaluation import compact_failure
from .models import (
    ROOT_ID,
    Program,
    TreePopulation,
    clean_text,
    fitness_text,
    json_schema,
    parse_json_object,
)
from .topology import Reduction, bare_code, better

EXPERIENCE_SCHEMA = json_schema("evotrace_experience", "experience")

MAX_DIFF_LINES = 28
MAX_DIFF_CHARS = 1800
REWRITE_SIMILARITY = 0.30
MAX_FALLBACK_NOTE_CHARS = 1200

_STALE_ID_PREFIX_RE = re.compile(
    r"\b(?:Removed\s+Attempt|Attempt|Trial|Algorithm|Node|Parent|Refine|Explore|Crossover)\s*#\d+\s*(?:\[[^\]]*\]\s*)?:?\s*",
    flags=re.IGNORECASE,
)
_BARE_HASH_ID_RE = re.compile(r"\s*\(#\d+\)|(?<!\w)#\d+\b\s*")

LEAF_EVICTION_PROMPT_TEMPLATE = """[Problem]
{task_description}
Score: {score_definition}
Function interface:
```python
{function_template}
```

{parent_section}

[Evicted Trial]
{outcome_section}
{trial_evidence}{evicted_note}

[Task]
This trial was removed to manage population capacity. A single scalar score alone cannot distinguish whether a core idea failed or its implementation had a flaw.
Update a concise note of still-untested ideas anchored to the surviving parent/ancestor above:
1. Identify what distinct computational change(s) were made in the evicted trial strictly from the full code/diff shown (never assume unimplemented intent, never use algorithm numbers).
2. Keep an idea ONLY if the code reveals a concrete implementation flaw or an untested sub-component bundled with other changes. If the evidence does not identify a flaw, say so; suggest at most one focused change that would test a component cleanly in isolation.
3. If this trial was already retrying an idea from the existing note and still did not beat its generating parent, drop that idea unless the code reveals a distinct concrete flaw.
4. Preserve still-valid entries from the existing note cleanly without rewriting them unnecessarily; drop any idea already implemented in the surviving parent/ancestor's code. Keep at most ~3-4 distinct entries.
5. Do NOT summarize why the surviving algorithm succeeds. Do NOT write general rules, prohibitions ("avoid / never / always"), or rigid prescriptions.

[Output]
Return one valid JSON object with an "experience" string (about 80-130 words; return an empty string if no concrete untested idea remains)."""

CLEANUP_MERGE_PROMPT_TEMPLATE = """[Problem]
{task_description}
Score: {score_definition}

[Surviving Algorithm]
fitness = {target_fitness}
```python
{target_code}
```

[Candidate Notes to Clean / Merge]
{notes_body}

[Task]
Clean and merge the candidate note(s) for the surviving algorithm above:
- Remove any idea or component that is ALREADY implemented in the surviving algorithm's code.
- Deduplicate overlapping directions and preserve up to ~3-4 sharpest distinct untested ideas verbatim where still valid, noting any concrete suspected flaw or isolated test angle.
- Do NOT summarize how the surviving algorithm succeeded, and do NOT add general rules or prohibitions.

[Output]
Return one valid JSON object with an "experience" string (about 80-130 words; return an empty string if all listed ideas are already realized)."""


def strip_stale_node_ids(text: str) -> str:
    cleaned = _BARE_HASH_ID_RE.sub(" ", _STALE_ID_PREFIX_RE.sub("", text))
    cleaned = re.sub(r"\s+([;:,.)])", r"\1", cleaned)
    return clean_text(cleaned).strip(" ;,")


def outcome_summary(base: Program | None, candidate: Program) -> str:
    """实测目标值与相对基准程序的胜负摘要。"""
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
    """紧凑源码差异；无变化、整体重写或 omit_if_truncated=True 且超限时返回空串。"""
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


def build_leaf_eviction_prompt(
    task, tree: TreePopulation, archive: dict[int, Program], receiver_id: int, leaf_id: int
) -> str:
    """构造剪叶反思提示：始终对照亲生生成父代评估真实改动，截断时回退完整叶源码。"""
    receiver = archive[receiver_id]
    leaf = archive[leaf_id]
    gen_parent_id = (
        leaf.parent_id
        if leaf.parent_id is not None and leaf.parent_id != ROOT_ID and leaf.parent_id in archive
        else receiver_id
    )
    gen_parent = archive[gen_parent_id]

    existing_note = clean_text(tree.idea_notes.get(receiver_id, ""))
    existing_line = (
        f"\nExisting note on surviving {'parent' if gen_parent_id == receiver_id else 'ancestor'}: "
        f"{existing_note}"
        if existing_note
        else ""
    )

    if gen_parent_id == receiver_id:
        parent_section = (
            "[Surviving Parent]\n"
            f"fitness = {fitness_text(receiver.fitness)}\n"
            f"```python\n{bare_code(receiver.code)}\n```{existing_line}"
        )
        outcome_section = f"Outcome vs generating parent: {outcome_summary(gen_parent, leaf)}"
        diff_label = "parent"
    else:
        parent_section = (
            "[Surviving Ancestor Receiving This Note]\n"
            f"fitness = {fitness_text(receiver.fitness)}\n"
            f"```python\n{bare_code(receiver.code)}\n```{existing_line}\n\n"
            "[Evicted Trial's Direct Generating Parent (Earlier Removed)]\n"
            f"fitness = {fitness_text(gen_parent.fitness)}\n"
            f"```python\n{bare_code(gen_parent.code)}\n```"
        )
        outcome_section = (
            f"Outcome vs direct generating parent: {outcome_summary(gen_parent, leaf)}\n"
            f"Outcome vs surviving ancestor: {outcome_summary(receiver, leaf)}"
        )
        diff_label = "direct generating parent"

    diff = compact_code_diff(gen_parent.code, leaf.code, omit_if_truncated=True)
    if diff:
        trial_evidence = f"Complete code diff relative to {diff_label}:\n```diff\n{diff}\n```"
    else:
        trial_evidence = (
            f"Complete evicted trial code (shown in full relative to {diff_label}):\n"
            f"```python\n{bare_code(leaf.code)}\n```"
        )

    carried = clean_text(tree.idea_notes.get(leaf_id, ""))
    evicted_note = f"\nEarlier notes carried by the evicted trial: {carried}" if carried else ""
    return LEAF_EVICTION_PROMPT_TEMPLATE.format(
        task_description=task.task_description.strip(),
        score_definition=task.score_definition.strip(),
        function_template=task.function_template.strip(),
        parent_section=parent_section,
        outcome_section=outcome_section,
        trial_evidence=trial_evidence,
        evicted_note=evicted_note,
    )


def build_cleanup_merge_prompt(
    task, archive: dict[int, Program], target_id: int, labeled_notes: list[tuple[str, str]]
) -> str:
    target = archive[target_id]
    blocks = [
        f"Note {chr(65 + idx)} ({source_label}):\n{note}"
        for idx, (source_label, note) in enumerate(labeled_notes)
    ]
    return CLEANUP_MERGE_PROMPT_TEMPLATE.format(
        task_description=task.task_description.strip(),
        score_definition=task.score_definition.strip(),
        target_fitness=fitness_text(target.fitness),
        target_code=bare_code(target.code),
        notes_body="\n\n".join(blocks),
    )


def _fallback_join(notes: list[str]) -> str:
    joined = " ".join(dict.fromkeys(clean_text(n) for n in notes if clean_text(n)))
    if len(joined) <= MAX_FALLBACK_NOTE_CHARS:
        return joined
    return joined[:MAX_FALLBACK_NOTE_CHARS].rsplit(" ", 1)[0]


def _call_reflection(model: Any, prompt: str, timeout: float, *, kind: str) -> tuple[str | None, dict]:
    report: dict[str, Any] = {"role": "reflection", "kind": kind, "status": "success"}
    with getattr(model, "request_context", nullcontext)():
        try:
            reply = model.complete(prompt, temperature=0.3, max_tokens=1024, timeout=timeout)
            content = reply if isinstance(reply, str) else reply.content
            if not isinstance(reply, str):
                report.update(
                    model=reply.model,
                    finish_reason=reply.finish_reason,
                    usage=reply.usage,
                    metadata=reply.metadata,
                )
            parsed = parse_json_object(content)
            raw = parsed.get("experience")
            if not isinstance(raw, str):
                raise TypeError("Missing string field 'experience'")
            note = strip_stale_node_ids(raw)
            report["experience"] = note
            return note, report
        except Exception as exc:  # noqa: BLE001 - reflection errors must not abort search
            report.update(
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                metadata=getattr(model, "last_metadata", {}),
            )
            return None, report


def reflect_reduction(
    task,
    tree: TreePopulation,
    archive: dict[int, Program],
    event: Reduction,
    model: Any | None,
    timeout: float,
) -> tuple[str, dict | None]:
    """叶淘汰触发剪枝反思；路径收缩/单子重连时，若笔记来自被移除祖先则对照新代码清理合并。"""
    if event.kind == "retire_root":
        return "", None

    if event.kind in ("upgrade", "evict_single_child"):
        seen: set[str] = set()
        labeled: list[tuple[str, str]] = []
        has_inherited = False
        for nid in event.removed:
            text = clean_text(tree.idea_notes.get(nid, ""))
            if text and text not in seen:
                seen.add(text)
                labeled.append(("inherited from replaced ancestor", text))
                has_inherited = True
        own_text = clean_text(tree.idea_notes.get(event.target, ""))
        if own_text and own_text not in seen:
            seen.add(own_text)
            labeled.append(("existing on surviving algorithm", own_text))

        if not labeled:
            return "", None
        # 仅保留节点自身笔记（未从祖先继承新笔记）时零开销原样保留
        if not has_inherited:
            return labeled[0][1], None
        raw_texts = [text for _, text in labeled]
        if model is None:
            return _fallback_join(raw_texts), None
        prompt = build_cleanup_merge_prompt(task, archive, event.target, labeled)
        note, report = _call_reflection(model, prompt, timeout, kind=f"{event.kind}_cleanup")
        return (_fallback_join(raw_texts) if note is None else note), report

    # event.kind == "evict_leaf"
    existing = clean_text(tree.idea_notes.get(event.target, ""))
    leaf_id = event.removed[0]
    carried = clean_text(tree.idea_notes.get(leaf_id, ""))
    fallback = _fallback_join([existing, carried])
    if model is None:
        return fallback, None
    prompt = build_leaf_eviction_prompt(task, tree, archive, event.target, leaf_id)
    note, report = _call_reflection(model, prompt, timeout, kind="leaf_eviction")
    return (fallback if note is None else note), report
