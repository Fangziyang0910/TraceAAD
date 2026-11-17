"""反思简化（Reflective Simplification）：区域代码与评价记录的凝练及多轮整合。"""

from collections.abc import Sequence
from contextlib import nullcontext
from copy import copy
import json
import re
from typing import Any

from .evaluation import compact_failure
from .models import ROOT_ID, TreePopulation, clean_text, json_schema, outcome_text
from .topology import BranchConsolidation, PathContraction, Region, apply_consolidation, bare_code, compact_diff

MAX_EXPERIENCE_CHARS = 960
VARIATION_SCHEMA = json_schema("evotrace_variation_experience", "experience")
SELECTION_SCHEMA = json_schema("evotrace_selection_experience", "experience")

_STALE_ID_PREFIX_RE = re.compile(
    r"\b(?:Removed\s+Attempt|Attempt|Trial|Algorithm|Node|Parent|Refine|Explore|Crossover)\s*#\d+\s*(?:\[[^\]]*\]\s*)?:?\s*",
    flags=re.IGNORECASE,
)
_BARE_HASH_ID_RE = re.compile(r"\s*\(#\d+\)|(?<!\w)#\d+\b\s*")

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


def _strip_stale_node_ids(text: str) -> str:
    cleaned = _BARE_HASH_ID_RE.sub(" ", _STALE_ID_PREFIX_RE.sub("", text))
    cleaned = re.sub(r"\s+([;:,.)])", r"\1", cleaned)
    return " ".join(cleaned.split()).strip(" ;,")


def _truncate_at_clause(text: str, limit: int) -> str:
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


def _task_header(
    task_description: str, score_definition: str = "Lower fitness is better.", function_template: str = ""
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
    u_id, v_id = contraction.parent_id, contraction.child_id
    v_node = tree.nodes[v_id]
    chain_ids = [u_id, *contraction.removed_ids, v_id]

    parts: list[str] = []
    if u_id == ROOT_ID:
        edge_label = f"Initial Origin -> Algorithm {v_id}"
        parts.append("Ancestor: Initial population origin")
    else:
        u_node = tree.nodes[u_id]
        edge_label = f"Algorithm {u_id} -> Algorithm {v_id}"
        parts.append(
            f"Surviving Ancestor (fitness = {u_node.fitness:.6g}):\n"
            f"Code:\n```python\n{bare_code(u_node.code)}\n```"
        )

    step_lines = ["Contracted Steps Along Path:"]
    for idx in range(1, len(chain_ids)):
        prev_id, cur_id = chain_ids[idx - 1], chain_ids[idx]
        cur_node, cur_edge = tree.nodes[cur_id], tree.edges[cur_id]
        prev_fit = None if prev_id == ROOT_ID else tree.nodes[prev_id].fitness
        op_str = cur_edge.operator or "Init"
        tag = "Surviving Descendant" if cur_id == v_id else "Removed Intermediate"
        step_lines.append(f"- Step {idx} [{op_str}] ({tag}): {outcome_text(prev_fit, cur_node.fitness)}")
        if cur_id != v_id:
            if prev_id != ROOT_ID:
                step_lines.append(
                    f"  Step code change:\n```diff\n{compact_diff(tree.nodes[prev_id].code, cur_node.code)}\n```"
                )
            else:
                step_lines.append(f"  Initial evaluated code:\n```python\n{bare_code(cur_node.code)}\n```")
        if cur_edge.variation_experience:
            step_lines.append(f"  Prior Edge Experience: {cur_edge.variation_experience}")
        if cur_id != v_id and cur_node.selection_experience:
            step_lines.append(f"  Earlier Side-Branch Experience: {cur_node.selection_experience}")
    parts.append("\n".join(step_lines))
    parts.append(
        f"Surviving Descendant (fitness = {v_node.fitness:.6g}):\n"
        f"Code:\n```python\n{bare_code(v_node.code)}\n```"
    )
    if u_id != ROOT_ID:
        parts.append(
            f"Net Code Diff (Ancestor -> Descendant):\n```diff\n"
            f"{compact_diff(tree.nodes[u_id].code, v_node.code)}\n```"
        )
    return VARIATION_PROMPT_TEMPLATE.format(
        task_header=_task_header(task_description, score_definition, function_template),
        slice_body="\n\n".join(parts),
        edge_label=edge_label,
    )


def _get_sel_exp(tree: TreePopulation, node_id: int) -> str:
    return clean_text(tree.root_selection_experience) if node_id == ROOT_ID else tree.nodes[node_id].selection_experience


def build_selection_prompt(
    tree: TreePopulation,
    consolidation: BranchConsolidation,
    *,
    task_description: str,
    score_definition: str = "Lower fitness is better.",
    function_template: str = "",
) -> str:
    p_id = consolidation.parent_id
    if p_id == ROOT_ID:
        node_label = "the initial population origin"
        parts = ["Anchor: Initial population origin (virtual root)"]
    else:
        anchor_node = tree.nodes[p_id]
        node_label = f"the retained parent algorithm (fitness = {anchor_node.fitness:.6g})"
        parts = [
            f"Retained Parent Baseline (fitness = {anchor_node.fitness:.6g}; reference implementation):\n"
            f"```python\n{bare_code(anchor_node.code)}\n```"
        ]
    if existing := _get_sel_exp(tree, p_id):
        parts.append(f"Earlier Branch Experience:\n{existing}")

    trial_numbers = {nid: idx for idx, nid in enumerate(sorted(consolidation.removed_ids), 1)}
    earlier_notes = []
    external_parents: dict[int, int] = {}
    for nid in trial_numbers:
        parent_id = tree.edges[nid].parent_id
        if parent_id not in trial_numbers and parent_id not in (p_id, ROOT_ID):
            external_parents.setdefault(parent_id, len(external_parents) + 1)
    for parent_id, number in external_parents.items():
        parent = tree.nodes[parent_id]
        parts.append(f"Additional parent {number} (fitness = {parent.fitness:.6g}):\n```python\n{bare_code(parent.code)}\n```")
    for nid, idx in trial_numbers.items():
        node, edge = tree.nodes[nid], tree.edges[nid]
        actual_parent = tree.nodes.get(edge.parent_id)
        op = edge.operator or "Init"
        if edge.parent_id in trial_numbers:
            parent_desc = f"Trial {trial_numbers[edge.parent_id]}"
        elif edge.parent_id in external_parents:
            parent_desc = f"Additional parent {external_parents[edge.parent_id]}"
        elif actual_parent is None:
            parent_desc = "initial origin"
        elif edge.parent_id == p_id:
            parent_desc = "retained parent"
        else:
            parent_desc = f"intermediate parent (fitness = {actual_parent.fitness:.6g})"
        rows = [
            f"Removed Trial {idx} [{op}] from {parent_desc}: "
            f"{outcome_text(actual_parent.fitness if actual_parent else None, node.fitness)}"
        ]
        if node.fitness is None:
            rows.append(f"Execution error: {compact_failure(node.failure)}")
        if actual_parent is not None:
            rows.append(f"Code change from its parent:\n```diff\n{compact_diff(actual_parent.code, node.code)}\n```")
        else:
            rows.append(f"Evaluated Code:\n```python\n{bare_code(node.code)}\n```")
        if edge.variation_experience:
            earlier_notes.append(f"Development leading to Trial {idx}: {edge.variation_experience}")
        if node.selection_experience:
            earlier_notes.append(f"Side experiments at Trial {idx}: {node.selection_experience}")
        parts.append("\n".join(rows))
    if earlier_notes:
        parts.append(
            "[Earlier experience carried by these branches]\n"
            "These are prior summaries, separate from the current code comparisons above.\n"
            + "\n\n".join(earlier_notes)
        )
    return SELECTION_PROMPT_TEMPLATE.format(
        task_header=_task_header(task_description, score_definition, function_template),
        slice_body="\n\n".join(parts),
        node_label=node_label,
    )


def format_variation_experience(
    retained_changes: str, intermediate_attempts: str, *, parent_fit: float | None = None, child_fit: float | None = None
) -> str:
    header = f"[{outcome_text(parent_fit, child_fit)}] " if child_fit is not None else ""
    ret = _strip_stale_node_ids(clean_text(retained_changes))
    mid = _strip_stale_node_ids(clean_text(intermediate_attempts))
    if mid and mid.lower() not in {"none", "none.", "n/a"}:
        return f"{header}Development: {ret} | Experience: {mid}"
    return f"{header}Development: {ret}"


def format_selection_experience(pruned_attempts: str) -> str:
    raw = _strip_stale_node_ids(clean_text(pruned_attempts))
    while raw.startswith(("[Eliminated", "[Branch experience]")):
        end = raw.find("]")
        raw = raw[end + 1 :].lstrip() if end != -1 else raw.removeprefix("[Eliminated").lstrip()
    return f"[Branch experience] {raw}"


def fallback_variation_experience(tree: TreePopulation, contraction: PathContraction) -> str:
    u_id, v_id = contraction.parent_id, contraction.child_id
    v_node = tree.nodes[v_id]
    u_fit = None if u_id == ROOT_ID else tree.nodes[u_id].fitness
    prior_notes = []
    for cid in (*contraction.removed_ids, v_id):
        if tree.edges[cid].variation_experience:
            prior_notes.append(tree.edges[cid].variation_experience[:140])
        if cid != v_id and tree.nodes[cid].selection_experience:
            prior_notes.append(tree.nodes[cid].selection_experience[:140])
    mids = ", ".join(
        f"({tree.edges[r].operator or 'Init'}:{tree.nodes[r].fitness:.6g})"
        if tree.nodes[r].fitness is not None else f"(failed: {compact_failure(tree.nodes[r].failure)[:48]})"
        for r in contraction.removed_ids[:6]
    )
    mid_summary = f"Contracted {len(contraction.removed_ids)} step(s): {mids}"
    if prior_notes:
        mid_summary += " | Prior: " + " ; ".join(prior_notes[:2])
    return _truncate_at_clause(
        format_variation_experience(v_node.design[:220], mid_summary, parent_fit=u_fit, child_fit=v_node.fitness),
        MAX_EXPERIENCE_CHARS,
    )


def fallback_selection_experience(tree: TreePopulation, consolidation: BranchConsolidation) -> str:
    existing = _get_sel_exp(tree, consolidation.parent_id)
    if existing.startswith(("[Eliminated", "[Branch experience]")):
        end = existing.find("]")
        existing = existing[end + 1 :].lstrip() if end != -1 else existing
    new_items = []
    for nid in sorted(consolidation.removed_ids):
        node, edge = tree.nodes[nid], tree.edges[nid]
        outcome = f"score {node.fitness:.6g}" if node.fitness is not None else compact_failure(node.failure)
        new_items.append(f"{node.design[:90] or (edge.operator or 'Init')}: {outcome}")
        for note in (edge.variation_experience, node.selection_experience):
            if note:
                if note.startswith(("[Eliminated", "[Branch experience]")):
                    idx = note.find("]")
                    note = note[idx + 1 :].lstrip() if idx != -1 else note
                new_items.append(note[:140])
    new_block = "; ".join(new_items)
    if not existing:
        return _truncate_at_clause(format_selection_experience(new_block), MAX_EXPERIENCE_CHARS)
    combined = f"{existing}; {new_block}"
    if len(combined) + len("[Branch experience] ") <= MAX_EXPERIENCE_CHARS:
        return format_selection_experience(combined)
    reserved_new = _truncate_at_clause(new_block, MAX_EXPERIENCE_CHARS // 2)
    avail_old = max(120, MAX_EXPERIENCE_CHARS - len("[Branch experience] ") - len(reserved_new) - 4)
    return format_selection_experience(f"{_truncate_at_clause(existing, avail_old)}; {reserved_new}")


def _with_schema(model: Any, schema: dict) -> Any:
    if hasattr(model, "for_role"):
        return model.for_role(schema)
    if hasattr(model, "request_options"):
        model = copy(model)
        model.request_options = {**model.request_options, "response_format": schema}
    return model


def reflect_and_apply(
    tree: TreePopulation,
    regions: Sequence[Region],
    *,
    model: Any | None = None,
    task_description: str = "",
    score_definition: str = "Lower fitness is better.",
    function_template: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1536,
    timeout: float = 180.0,
) -> tuple[TreePopulation, list[dict]]:
    """执行拓扑重构并写回重连边的 variation_experience 与保留节点的 selection_experience。"""
    new_tree = apply_consolidation(tree, regions)
    contractions = [r for r in regions if isinstance(r, PathContraction)]
    branches = sorted((r for r in regions if isinstance(r, BranchConsolidation)),
                      key=lambda b: (-len(b.removed_ids), b.parent_id))
    records: list[dict] = []
    for region in (*contractions, *branches):
        variation = isinstance(region, PathContraction)
        target_id = region.child_id if variation else region.parent_id
        usage, error, exp = {}, None, None
        if model is not None:
            prompt = (build_variation_prompt if variation else build_selection_prompt)(
                tree, region,  # type: ignore[arg-type]
                task_description=task_description, score_definition=score_definition, function_template=function_template,
            )
            client = _with_schema(model, VARIATION_SCHEMA if variation else SELECTION_SCHEMA)
            try:
                with getattr(client, "request_context", nullcontext)():
                    reply = client.complete(prompt, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
                content = reply.content if hasattr(reply, "content") else str(reply)
                usage = dict(getattr(reply, "usage", None) or {})
                raw = json.loads(content)["experience"]
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError("Reflection experience must be a nonempty string")
                cleaned = _strip_stale_node_ids(clean_text(raw))
                if not cleaned:
                    raise ValueError("Reflection experience has no content after cleanup")
                if variation:
                    p_fit = None if region.parent_id == ROOT_ID else tree.nodes[region.parent_id].fitness
                    exp = f"[{outcome_text(p_fit, tree.nodes[region.child_id].fitness)}] {cleaned}"
                else:
                    exp = format_selection_experience(cleaned)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if exp is None:
            exp = (fallback_variation_experience(tree, region) if variation
                   else fallback_selection_experience(tree, region))  # type: ignore[arg-type]
        if variation:
            new_tree.edges[target_id].variation_experience = exp
        elif target_id == ROOT_ID:
            new_tree.root_selection_experience = exp
        elif target_id in new_tree.nodes:
            new_tree.nodes[target_id].selection_experience = exp
        if model is not None:
            records.append({
                "role": "reflection",
                "kind": "variation" if variation else "selection",
                "target_id": target_id,
                "status": "fallback" if error else "success",
                "errors": int(error is not None),
                "usage": usage,
                "error": error,
                "experience": exp,
            })
    return new_tree, records
