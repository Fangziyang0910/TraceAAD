"""反思简化（Reflective Simplification）：区域代码与评价记录的凝练及多轮整合。"""

import ast
from copy import copy
from dataclasses import dataclass
from difflib import unified_diff
import json
import re
from typing import Any

from .models import TreePopulation
from .topology import (
    _strip_docstrings,
    BranchConsolidation,
    ConsolidationPlan,
    PathContraction,
    TopologyConfig,
    apply_consolidation,
    plan_consolidation,
)


TIMEOUT_PREFIX = "Evaluation exceeded"


def _parse_failure(failure: str):
    """The exception message and the traceback blocks that lie in the candidate's own code."""
    lines = failure.strip().splitlines()
    start = len(lines) - 1
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and not line.startswith("Traceback"):
            start = i
    message = re.sub(r"^(?:\w+\.)+(?=\w+(?::|$))", "", "\n".join(lines[start:]))
    blocks, keep = [], False
    for line in lines[:start]:
        if line.lstrip().startswith('File "'):
            keep = "<candidate>" in line or "<string>" in line
            if keep:
                blocks.append([line.strip()])
        elif keep and line.startswith("  "):
            blocks[-1].append(line)
        else:
            keep = False
    return re.sub(r"\d+(?:\.\d+)?", "N", message), blocks


def failure_feedback(failure: str) -> str:
    """Repair also sees the candidate's own traceback lines."""
    message, blocks = _parse_failure(failure)
    return "\n".join([*(line for block in blocks for line in block), message])


MAX_EXPERIENCE_CHARS = 960

_CTRL_REPAIR = str.maketrans({
    "\x08": r"\b",
    "\x0c": r"\f",
    "\r": " ",
    "\t": " ",
})
_STALE_ID_PREFIX_RE = re.compile(
    r"\b(?:Removed\s+Attempt|Attempt|Trial|Algorithm|Node|Parent|Refine|Explore|Crossover)\s*#\d+\s*(?:\[[^\]]*\]\s*)?:?\s*",
    flags=re.IGNORECASE,
)
_BARE_HASH_ID_RE = re.compile(r"\s*\(#\d+\)|(?<!\w)#\d+\b\s*")


def clean_prompt_text(text: str | None) -> str:
    """Restore LaTeX backslash prefixes accidentally decoded as JSON control chars (\\b, \\f) and collapse whitespace."""
    if not text:
        return ""
    repaired = text.translate(_CTRL_REPAIR)
    repaired = "".join(ch for ch in repaired if ch == "\n" or ord(ch) >= 32)
    return " ".join(repaired.split())


def _strip_stale_node_ids(text: str) -> str:
    """Remove transient tree node numbers (#ID) so distilled experience names computations rather than removed IDs."""
    cleaned = _STALE_ID_PREFIX_RE.sub("", text)
    cleaned = _BARE_HASH_ID_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+([;:,.)])", r"\1", cleaned)
    return " ".join(cleaned.split()).strip(" ;,")


def _truncate_at_clause(text: str, limit: int) -> str:
    """Bound experience length cleanly on a semicolon or sentence boundary instead of cutting mid-word."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    best_cut = max(window.rfind("; "), window.rfind(". "), window.rfind(" | "))
    if best_cut >= limit // 2:
        suffix = "." if window[best_cut : best_cut + 2] in {"; ", ". "} else ""
        return window[:best_cut].rstrip(" ;,.") + suffix
    space_cut = window.rfind(" ")
    if space_cut >= limit // 2:
        return window[:space_cut].rstrip(" ;,.") + "..."
    return window


def _bare_code(code: str) -> str:
    """去除模块/函数/类文档字符串后的标准化源码，保证 diff 与展示聚焦实际计算。"""
    try:
        return ast.unparse(_strip_docstrings(code))
    except (SyntaxError, ValueError):
        return code.strip()


def compact_diff(old_code: str, new_code: str) -> str:
    """计算两段标准化源码的紧凑 unified diff。"""
    a = _bare_code(old_code).splitlines()
    b = _bare_code(new_code).splitlines()
    if a == b:
        return "(normalized code unchanged)"
    lines = [
        "..." if line.startswith("@@") else line
        for line in list(unified_diff(a, b, lineterm="", n=2))[2:]
    ]
    return "\n".join(lines)


def compact_failure(failure_text: str | None) -> str:
    """提取执行错误类型与候选程序出错源码行，或超时说明。"""
    text = (failure_text or "").strip()
    if not text:
        return "infeasible (unknown error)"
    if text.startswith(TIMEOUT_PREFIX):
        return "timeout (exceeded evaluation time limit)"
    message, blocks = _parse_failure(text)
    if blocks:
        frame = blocks[-1]
        source = next(
            (line.strip() for line in frame[1:] if line.strip() and set(line.strip()) - set("~^ ")),
            None,
        )
        if source:
            return f"{message} at `{source[:80]}`"
    return message[:120]


def _outcome_word(parent_fit: float | None, child_fit: float | None) -> str:
    if child_fit is None:
        return "failed"
    if parent_fit is None:
        return f"fitness={child_fit:.6g}"
    if child_fit < parent_fit:
        return f"{parent_fit:.6g} -> {child_fit:.6g} (improved)"
    if child_fit == parent_fit:
        return f"{parent_fit:.6g} -> {child_fit:.6g} (tied)"
    return f"{parent_fit:.6g} -> {child_fit:.6g} (regressed)"


VARIATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "evotrace_variation_experience",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "experience": {"type": "string"},
            },
            "required": ["experience"],
        },
    },
}

SELECTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "evotrace_selection_experience",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "experience": {"type": "string"},
            },
            "required": ["experience"],
        },
    },
}


VARIATION_PROMPT_TEMPLATE = """{task_header}

[Path Evidence]
{slice_body}

[Task]
Compress the algorithm-design experience along this transition ({edge_label}) for a researcher continuing the search.
Write a short research note preserving what is worth remembering from this development: the computational ideas, informative outcomes, and relevant earlier experience. A faithful account is enough; an explanation or general principle is optional. Keep any interpretation tentative and leave future design decisions open.

[Output]
Return one JSON object with an "experience" string: a concise paragraph, about 120 words, explaining how the computation developed and what is worth remembering from the intervening experiments. Include informative outcomes and earlier experience. The paragraph will accompany the surviving implementation in later design calls; name computations rather than trial numbers. The system supplies endpoint scores."""


SELECTION_PROMPT_TEMPLATE = """{task_header}

[Branch Experiments]
{slice_body}

[Task]
Compress the algorithm-design experience around {node_label} for a researcher continuing the search.
Write a short research note preserving what is worth remembering from these experiments: the computational ideas, informative outcomes, and relevant earlier experience. A faithful account is enough; an explanation or general principle is optional. Keep any interpretation tentative and leave future design decisions open. These branches were removed to manage the population; their ideas are not thereby disproved.

[Output]
Return one JSON object with an "experience" string: a concise paragraph, about 120 words, relating the computational ideas tried and their informative outcomes, including earlier experience. It will accompany an algorithm in later design calls, preserving experiences its code alone cannot convey. The paragraph must stand on its own without trial numbers."""


def _task_header(
    task_description: str,
    score_definition: str = "Lower fitness is better.",
    function_template: str = "",
) -> str:
    lines = ["[Problem]", task_description.strip(), f"Score: {score_definition.strip()}"]
    if function_template.strip():
        lines.append(f"Function interface:\n```python\n{function_template.strip()}\n```")
    return "\n".join(lines)


def build_variation_prompt(
    tree: TreePopulation,
    contraction: PathContraction,
    *,
    task_description: str,
    score_definition: str = "Lower fitness is better.",
    function_template: str = "",
) -> str:
    """构造单条路径收缩（u -> removed_ids -> v）的区域反思提示词。

    完整纳入：（1）两端保留节点代码与净 diff；（2）中间逐步代码差分与分数；
    （3）中间被删边原有的 variation_experience 与中间被删节点原有的 selection_experience。
    """
    u_id, v_id = contraction.parent_id, contraction.child_id
    v_node = tree.nodes[v_id]
    chain_ids = [u_id, *contraction.removed_ids, v_id]

    parts: list[str] = []
    if u_id == tree.root_id:
        edge_label = f"Initial Origin -> Algorithm {v_id}"
        parts.append("Ancestor: Initial population origin")
    else:
        u_node = tree.nodes[u_id]
        edge_label = f"Algorithm {u_id} -> Algorithm {v_id}"
        parts.append(
            f"Surviving Ancestor (fitness = {u_node.fitness:.6g}):\n"
            f"Code:\n```python\n{_bare_code(u_node.code)}\n```"
        )

    step_lines: list[str] = ["Contracted Steps Along Path:"]
    for idx in range(1, len(chain_ids)):
        prev_id, cur_id = chain_ids[idx - 1], chain_ids[idx]
        cur_node = tree.nodes[cur_id]
        cur_edge = tree.edges[cur_id]
        prev_fit = None if prev_id == tree.root_id else tree.nodes[prev_id].fitness
        op_str = cur_edge.operator or "Init"
        tag = "Surviving Descendant" if cur_id == v_id else "Removed Intermediate"
        step_lines.append(
            f"- Step {idx} [{op_str}] ({tag}): {_outcome_word(prev_fit, cur_node.fitness)}"
        )
        if cur_id != v_id:
            if prev_id != tree.root_id:
                step_lines.append(
                    f"  Step code change:\n```diff\n{compact_diff(tree.nodes[prev_id].code, cur_node.code)}\n```"
                )
            else:
                step_lines.append(f"  Initial evaluated code:\n```python\n{_bare_code(cur_node.code)}\n```")
        if cur_edge.variation_experience:
            step_lines.append(f"  Prior Edge Experience: {clean_prompt_text(cur_edge.variation_experience)}")
        if cur_id != v_id and cur_node.selection_experience:
            step_lines.append(
                f"  Earlier Side-Branch Experience: {clean_prompt_text(cur_node.selection_experience)}"
            )
    parts.append("\n".join(step_lines))

    parts.append(
        f"Surviving Descendant (fitness = {v_node.fitness:.6g}):\n"
        f"Code:\n```python\n{_bare_code(v_node.code)}\n```"
    )
    if u_id != tree.root_id:
        parts.append(
            f"Net Code Diff (Ancestor -> Descendant):\n```diff\n"
            f"{compact_diff(tree.nodes[u_id].code, v_node.code)}\n```"
        )

    return VARIATION_PROMPT_TEMPLATE.format(
        task_header=_task_header(task_description, score_definition, function_template),
        slice_body="\n\n".join(parts),
        edge_label=edge_label,
    )


def _get_node_selection_exp(tree: TreePopulation, node_id: int) -> str:
    if node_id == tree.root_id:
        return tree.root_selection_experience
    return tree.nodes[node_id].selection_experience


def _set_node_selection_exp(tree: TreePopulation, node_id: int, exp: str) -> None:
    if node_id == tree.root_id:
        tree.root_selection_experience = exp
    elif node_id in tree.nodes:
        tree.nodes[node_id].selection_experience = exp


def build_selection_prompt(
    tree: TreePopulation,
    consolidation: BranchConsolidation,
    *,
    task_description: str,
    score_definition: str = "Lower fitness is better.",
    function_template: str = "",
) -> str:
    """提供承接基准、逐个尝试的真实父代差分、结果与完整旧经验。"""
    p_id = consolidation.parent_id
    if p_id == tree.root_id:
        node_label = "the initial population origin"
        parts = ["Anchor: Initial population origin (virtual root)"]
    else:
        anchor_node = tree.nodes[p_id]
        node_label = f"the retained parent algorithm (fitness = {anchor_node.fitness:.6g})"
        parts = [
            f"Retained Parent Baseline (fitness = {anchor_node.fitness:.6g}; reference implementation):\n"
            f"```python\n{_bare_code(anchor_node.code)}\n```"
        ]
    existing = _get_node_selection_exp(tree, p_id)
    if existing:
        parts.append(f"Earlier Branch Experience:\n{clean_prompt_text(existing)}")

    trial_numbers = {nid: idx for idx, nid in enumerate(sorted(consolidation.removed_ids), 1)}
    earlier_notes = []
    external_parents = {}
    for nid in trial_numbers:
        parent_id = tree.edges[nid].parent_id
        if parent_id not in trial_numbers and parent_id not in (p_id, tree.root_id):
            external_parents.setdefault(parent_id, len(external_parents) + 1)
    for parent_id, number in external_parents.items():
        parent = tree.nodes[parent_id]
        parts.append(f"Additional parent {number} (fitness = {parent.fitness:.6g}):\n"
                     f"```python\n{_bare_code(parent.code)}\n```")
    for nid, idx in trial_numbers.items():
        node, edge = tree.nodes[nid], tree.edges[nid]
        actual_parent = tree.nodes.get(edge.parent_id)
        op = edge.operator or "Init"
        parent_desc = "initial origin" if actual_parent is None else (
            "retained parent" if edge.parent_id == p_id else f"intermediate parent (fitness = {actual_parent.fitness:.6g})"
        )
        if edge.parent_id in trial_numbers:
            parent_desc = f"Trial {trial_numbers[edge.parent_id]}"
        elif edge.parent_id in external_parents:
            parent_desc = f"Additional parent {external_parents[edge.parent_id]}"
        rows = [
            f"Removed Trial {idx} [{op}] from {parent_desc}: "
            f"{_outcome_word(actual_parent.fitness if actual_parent else None, node.fitness)}",
        ]
        if node.fitness is None:
            rows.append(f"Execution error: {compact_failure(node.failure)}")
        if actual_parent is not None:
            rows.append(
                f"Code change from its parent:\n```diff\n{compact_diff(actual_parent.code, node.code)}\n```"
            )
        else:
            rows.append(f"Evaluated Code:\n```python\n{_bare_code(node.code)}\n```")
        if edge.variation_experience:
            earlier_notes.append(f"Development leading to Trial {idx}: {clean_prompt_text(edge.variation_experience)}")
        if node.selection_experience:
            earlier_notes.append(f"Side experiments at Trial {idx}: {clean_prompt_text(node.selection_experience)}")
        parts.append("\n".join(rows))
    if earlier_notes:
        parts.append("[Earlier experience carried by these branches]\nThese are prior summaries, separate from the current code comparisons above.\n" + "\n\n".join(earlier_notes))

    return SELECTION_PROMPT_TEMPLATE.format(
        task_header=_task_header(task_description, score_definition, function_template),
        slice_body="\n\n".join(parts),
        node_label=node_label,
    )


def format_variation_experience(
    retained_changes: str,
    intermediate_attempts: str,
    *,
    parent_fit: float | None = None,
    child_fit: float | None = None,
) -> str:
    header = f"[{_outcome_word(parent_fit, child_fit)}] " if child_fit is not None else ""
    ret = _strip_stale_node_ids(clean_prompt_text(retained_changes))
    mid = _strip_stale_node_ids(clean_prompt_text(intermediate_attempts))
    if mid and mid.lower() not in {"none", "none.", "n/a"}:
        text = f"{header}Development: {ret} | Experience: {mid}"
    else:
        text = f"{header}Development: {ret}"
    return text


def format_selection_experience(pruned_attempts: str) -> str:
    raw = _strip_stale_node_ids(clean_prompt_text(pruned_attempts))
    while raw.startswith(("[Eliminated", "[Branch experience]")):
        end = raw.find("]")
        raw = raw[end + 1 :].lstrip() if end != -1 else raw.removeprefix("[Eliminated").lstrip()
    return f"[Branch experience] {raw}"


def fallback_variation_experience(tree: TreePopulation, contraction: PathContraction) -> str:
    """无模型或调用异常时的确定性程序事实兜底，同时保留旧边与被缩中间节点的已有经验。"""
    u_id, v_id = contraction.parent_id, contraction.child_id
    v_node = tree.nodes[v_id]
    u_fit = None if u_id == tree.root_id else tree.nodes[u_id].fitness
    chain_ids = (*contraction.removed_ids, v_id)
    prior_notes: list[str] = []
    for cid in chain_ids:
        ve = tree.edges[cid].variation_experience
        if ve:
            prior_notes.append(clean_prompt_text(ve)[:140])
        if cid != v_id and tree.nodes[cid].selection_experience:
            prior_notes.append(clean_prompt_text(tree.nodes[cid].selection_experience)[:140])
    mids = ", ".join(
        f"({tree.edges[r].operator or 'Init'}:{tree.nodes[r].fitness:.6g})"
        if tree.nodes[r].fitness is not None
        else f"(failed: {compact_failure(tree.nodes[r].failure)[:48]})"
        for r in contraction.removed_ids[:6]
    )
    mid_summary = f"Contracted {len(contraction.removed_ids)} step(s): {mids}"
    if prior_notes:
        mid_summary += " | Prior: " + " ; ".join(prior_notes[:2])
    return _truncate_at_clause(format_variation_experience(
        clean_prompt_text(v_node.design)[:220],
        mid_summary,
        parent_fit=u_fit,
        child_fit=v_node.fitness,
    ), MAX_EXPERIENCE_CHARS)


def fallback_selection_experience(
    tree: TreePopulation,
    consolidation: BranchConsolidation,
) -> str:
    """无模型或调用异常时的确定性旁支账本兜底（原地整合已有账本，避免重复前缀或丢失新尝试）。"""
    p_id = consolidation.parent_id
    existing = clean_prompt_text(_get_node_selection_exp(tree, p_id))
    if existing.startswith(("[Eliminated", "[Branch experience]")):
        end = existing.find("]")
        existing = existing[end + 1 :].lstrip() if end != -1 else existing
    new_items: list[str] = []
    for nid in sorted(consolidation.removed_ids):
        node, edge = tree.nodes[nid], tree.edges[nid]
        outcome = f"score {node.fitness:.6g}" if node.fitness is not None else compact_failure(node.failure)
        label = clean_prompt_text(node.design)[:90] or (edge.operator or "Init")
        new_items.append(f"{label}: {outcome}")
        for note in (edge.variation_experience, node.selection_experience):
            if note:
                cleaned_note = clean_prompt_text(note)
                if cleaned_note.startswith(("[Eliminated", "[Branch experience]")):
                    idx = cleaned_note.find("]")
                    cleaned_note = cleaned_note[idx + 1 :].lstrip() if idx != -1 else cleaned_note
                new_items.append(cleaned_note[:140])
    # Put newest items first if overflow occurs, or keep existing + new within budget cleanly
    new_block = "; ".join(new_items)
    if not existing:
        return _truncate_at_clause(format_selection_experience(new_block), MAX_EXPERIENCE_CHARS)
    combined = f"{existing}; {new_block}"
    if len(combined) + len("[Branch experience] ") <= MAX_EXPERIENCE_CHARS:
        return format_selection_experience(combined)
    # Reserve space so newly removed items are never completely dropped on fallback
    reserved_new = _truncate_at_clause(new_block, MAX_EXPERIENCE_CHARS // 2)
    avail_old = max(120, MAX_EXPERIENCE_CHARS - len("[Branch experience] ") - len(reserved_new) - 4)
    trimmed_old = _truncate_at_clause(existing, avail_old)
    return format_selection_experience(f"{trimmed_old}; {reserved_new}")


@dataclass(frozen=True)
class ReflectionCallRecord:
    kind: str  # "variation" | "selection"
    target_id: int  # edge child_id for variation, node_id (or root_id=0) for selection
    prompt_chars: int
    experience: str
    used_fallback: bool
    usage: dict[str, Any]
    prompt: str = ""
    response: str = ""
    error: str | None = None


def _call_structured_model(
    model: Any,
    prompt: str,
    schema: dict[str, Any],
    *,
    temperature: float = 0.3,
    max_tokens: int = 1536,
    timeout: float = 180.0,
) -> tuple[str, dict[str, Any]]:
    from contextlib import nullcontext

    client = model
    if hasattr(model, "for_role"):
        client = model.for_role(schema)
    elif hasattr(model, "request_options"):
        client = copy(model)
        client.request_options = {**model.request_options, "response_format": schema}
    with getattr(client, "request_context", nullcontext)():
        reply = client.complete(prompt, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    content = reply.content if hasattr(reply, "content") else str(reply)
    usage = dict(getattr(reply, "usage", None) or {})
    return content, usage


def reflect_and_apply(
    tree: TreePopulation,
    plan: ConsolidationPlan,
    *,
    model: Any | None = None,
    task_description: str = "",
    score_definition: str = "Lower fitness is better.",
    function_template: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1536,
    timeout: float = 180.0,
) -> tuple[TreePopulation, list[ReflectionCallRecord]]:
    """对重构方案执行反思简化，写回重连边的 variation_experience 与保留节点（含虚拟根）的 selection_experience。"""
    new_tree = apply_consolidation(tree, plan)
    records: list[ReflectionCallRecord] = []

    # 路径与旁支各保留自己的证据和格式，共用调用、异常处理与记录。
    branches = sorted(plan.branch_consolidations, key=lambda b: (-len(b.removed_ids), b.parent_id))
    for region in (*plan.path_contractions, *branches):
        variation = isinstance(region, PathContraction)
        kind = "variation" if variation else "selection"
        target_id = region.child_id if variation else region.parent_id
        build_prompt = build_variation_prompt if variation else build_selection_prompt
        fallback = fallback_variation_experience if variation else fallback_selection_experience
        schema = VARIATION_SCHEMA if variation else SELECTION_SCHEMA
        prompt, response, usage, error = "", "", {}, None
        used_fallback = True
        if model is not None:
            prompt = build_prompt(
                tree, region, task_description=task_description,
                score_definition=score_definition, function_template=function_template,
            )
            try:
                response, usage = _call_structured_model(
                    model, prompt, schema,
                    temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                )
                parsed = json.loads(response)
                experience = parsed["experience"]
                if not isinstance(experience, str) or not experience.strip():
                    raise ValueError("Reflection experience must be a nonempty string")
                experience = _strip_stale_node_ids(clean_prompt_text(experience))
                if not experience:
                    raise ValueError("Reflection experience has no content after cleanup")
                if variation:
                    parent_fit = None if region.parent_id == tree.root_id else tree.nodes[region.parent_id].fitness
                    exp = f"[{_outcome_word(parent_fit, tree.nodes[region.child_id].fitness)}] {experience}"
                else:
                    exp = format_selection_experience(experience)
                used_fallback = False
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if used_fallback:
            exp = fallback(tree, region)
        if variation:
            new_tree.edges[target_id].variation_experience = exp
        else:
            _set_node_selection_exp(new_tree, target_id, exp)
        records.append(ReflectionCallRecord(
            kind, target_id, len(prompt), exp, used_fallback, usage, prompt, response, error,
        ))

    return new_tree, records


def reflective_simplify(
    tree: TreePopulation,
    *,
    config: TopologyConfig = TopologyConfig(),
    model: Any | None = None,
    task_description: str = "",
    score_definition: str = "Lower fitness is better.",
    function_template: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1536,
    timeout: float = 180.0,
) -> tuple[TreePopulation, ConsolidationPlan, list[ReflectionCallRecord]]:
    """统一选择与结构补全，再按移除区域写回变异经验和选择经验。"""
    plan = plan_consolidation(tree, config)
    new_tree, records = reflect_and_apply(
        tree,
        plan,
        model=model,
        task_description=task_description,
        score_definition=score_definition,
        function_template=function_template,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return new_tree, plan, records
