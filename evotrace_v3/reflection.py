"""反思树简化（Reflective Tree Simplification）：区域代码与评价记录的凝练及多轮整合。"""

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


def clean_prompt_text(text: str | None) -> str:
    """Restore LaTeX backslash prefixes accidentally decoded as JSON control chars (\\b, \\f) and collapse whitespace."""
    if not text:
        return ""
    repaired = text.translate(_CTRL_REPAIR)
    repaired = "".join(ch for ch in repaired if ch == "\n" or ord(ch) >= 32)
    return " ".join(repaired.split())


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


def _outcome_word(parent_fit: float | None, child_fit: float | None, *, include_scores: bool = True) -> str:
    if child_fit is None:
        return "failed"
    if parent_fit is None:
        return f"fitness={child_fit:.6g}" if include_scores else "evaluated (no algorithm parent)"
    outcome = "improved" if child_fit < parent_fit else "tied" if child_fit == parent_fit else "regressed"
    return f"{parent_fit:.6g} -> {child_fit:.6g} ({outcome})" if include_scores else outcome


def _reflection_schema(name: str) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
            "properties": {"experience": {"type": "string"}},
            "required": ["experience"]}}}


VARIATION_SCHEMA = _reflection_schema("evotrace_variation_experience")
SELECTION_SCHEMA = _reflection_schema("evotrace_selection_experience")

VARIATION_PROMPT_TEMPLATE = """{task_header}

[Path Evidence]
{slice_body}

[Task]
Compress the algorithm-design experience along this transition for a researcher continuing the search.
Write a short research note preserving what is worth remembering from this development: the computational ideas, informative outcomes, and relevant earlier experience. A faithful account is enough; an explanation or general principle is optional. Keep any interpretation tentative and leave future design decisions open.

[Output]
Return one JSON object with an "experience" string: a concise paragraph, about 120 words, explaining how the computation developed and what is worth remembering from the intervening experiments. Include informative outcomes and earlier experience. The paragraph will accompany the surviving implementation in later design calls; name computations rather than trial numbers. The system supplies endpoint scores."""

SELECTION_PROMPT_TEMPLATE = """{task_header}

[Branch Experiments]
{slice_body}

[Task]
Compress the algorithm-design experience from these experiments for a researcher continuing the search.
Write a short research note preserving what is worth remembering from these experiments: the computational ideas, informative outcomes, and relevant earlier experience. A faithful account is enough; an explanation or general principle is optional. Keep any interpretation tentative and leave future design decisions open. These branches were removed to manage the population; their ideas are not thereby disproved.

[Output]
Return one JSON object with an "experience" string: a concise paragraph, about 120 words, relating the computational ideas tried and their informative outcomes, including earlier experience. It will accompany an algorithm in later design calls, preserving experiences its code alone cannot convey. The paragraph must stand on its own without trial numbers."""


def comparison_catalog(tree: TreePopulation, region) -> tuple[dict[str, dict], set[str], str | None]:
    """Measured comparisons keep their original endpoints across experience transfers."""
    catalog: dict[str, dict] = {}
    current: set[str] = set()
    variation = isinstance(region, PathContraction)
    ids = (*region.removed_ids, region.child_id, *region.side_ids) if variation else region.removed_ids
    for i in ids:
        node_notes = () if variation and i == region.child_id else tree.nodes[i].selection_observations
        for observation in (*node_notes, *tree.edges[i].variation_observations):
            catalog[observation["comparison_id"]] = dict(observation)
    if not variation:
        prior = (tree.root_selection_observations if region.parent_id == tree.root_id
                 else tree.nodes[region.parent_id].selection_observations)
        for observation in prior:
            catalog[observation["comparison_id"]] = dict(observation)

    def add(parent_id, child_id, operator):
        parent = tree.nodes.get(parent_id)
        child = tree.nodes[child_id]
        key = f"{parent_id}->{child_id}"
        catalog[key] = dict(comparison_id=key, parent_id=parent_id, child_id=child_id,
                            parent_fitness=parent.fitness if parent else None,
                            fitness=child.fitness, operator=operator,
                            failure=compact_failure(child.failure) if child.fitness is None else None)
        current.add(key)
        return key

    for i in ids:
        edge = tree.edges[i]
        add(edge.parent_id, i, "development stage" if edge.contracted else edge.operator or "Init")
    net = (add(region.parent_id, region.child_id, "net development")
           if variation and region.parent_id != tree.root_id else None)
    return catalog, current, net


def _catalog_block(tree: TreePopulation, region) -> str:
    catalog, current, net = comparison_catalog(tree, region)
    lines = ["[Comparison Catalog]", "Identifiers refer to the original compared implementations, not to the current experience owner."]
    for key, item in catalog.items():
        label = _outcome_word(item["parent_fitness"], item["fitness"])
        source = "overall path" if key == net else "current" if key in current else "earlier observation"
        if item["parent_fitness"] is None:
            source += "; independent evaluation, no parent comparison"
        lines.append(f"{key} [{source}; {item['operator']}]: {label}")
        if item.get("failure"):
            lines.append("Execution error: " + item["failure"])
    return "\n".join(lines)


def render_observations(observations: list[dict]) -> str:
    parts = []
    for item in observations:
        result = _outcome_word(item["parent_fitness"], item["fitness"])
        label = (f"Comparison {item['comparison_id']}" if item["parent_fitness"] is not None
                 else f"Evaluation {item['child_id']}; no parent comparison")
        parts.append(f"[{label}: {result}] "
                     + ("Execution error: " + item["failure"] + ". " if item.get("failure") else "") + item["changes"]
                     + (" Possible explanation: " + item["interpretation"] if item.get("interpretation") else ""))
    return "\n".join(parts)


def _bind_experience(parsed: dict, tree: TreePopulation, region) -> tuple[str, list[dict]]:
    note = parsed["experience"]
    if not isinstance(note, str) or not clean_prompt_text(note):
        raise ValueError("Reflection must contain nonempty algorithm-design experience")
    catalog, _, _ = comparison_catalog(tree, region)
    # Region provenance is determined by the tree, not selected by the language model.
    # These measurements do not certify the meaning of the generated experience.
    observations = [{k: v for k, v in item.items() if k not in ("changes", "interpretation")}
                    for item in catalog.values()]
    return clean_prompt_text(note), observations


def _prior_note(note: str, observations: list[dict]) -> str:
    """Keep a carried research note associated with its own original evidence."""
    sources = ", ".join(item["comparison_id"] for item in observations)
    provenance = f"[Supporting comparisons: {sources}] " if sources else ""
    return provenance + clean_prompt_text(note)


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

    完整纳入：（1）两端保留节点代码与净 diff；（2）中间受评实现与逐步分数；
    （3）中间被删边原有的 variation_experience 与中间被删节点原有的 selection_experience。
    """
    u_id, v_id = contraction.parent_id, contraction.child_id
    v_node = tree.nodes[v_id]
    chain_ids = [u_id, *contraction.removed_ids, v_id]

    start_fit = None if u_id == tree.root_id else tree.nodes[u_id].fitness
    overall = f"Overall measured outcome: {_outcome_word(start_fit, v_node.fitness, include_scores=False)}; {_outcome_word(start_fit, v_node.fitness)}"
    parts = ([overall]
             if start_fit is not None else ["The virtual root has no implementation or score; no overall relative outcome exists."])
    if u_id == tree.root_id:
        parts.append("Ancestor: Initial population origin")
    else:
        u_node = tree.nodes[u_id]
        parts.append(
            f"Starting Implementation {u_id}, fitness={u_node.fitness:.6g}:\n"
            f"Code:\n```python\n{_bare_code(u_node.code)}\n```"
        )

    earlier_notes = []
    step_lines: list[str] = ["Contracted Steps Along Path:"]
    for idx in range(1, len(chain_ids)):
        prev_id, cur_id = chain_ids[idx - 1], chain_ids[idx]
        cur_node = tree.nodes[cur_id]
        cur_edge = tree.edges[cur_id]
        prev_fit = None if prev_id == tree.root_id else tree.nodes[prev_id].fitness
        op_str = "Compressed development" if cur_edge.contracted else cur_edge.operator or "Init"
        tag = "Endpoint" if cur_id == v_id else "Intermediate"
        step_lines.append(
            f"- Step {idx} [{op_str}] ({tag}), comparison {prev_id}->{cur_id}: {_outcome_word(prev_fit, cur_node.fitness, include_scores=False)}; {_outcome_word(prev_fit, cur_node.fitness)}"
        )
        if cur_id != v_id:
            step_lines.append(f"  Evaluated implementation:\n```python\n{_bare_code(cur_node.code)}\n```")
        if cur_edge.variation_experience:
            earlier_notes.append(f"Development {prev_id}->{cur_id}: {_prior_note(cur_edge.variation_experience, cur_edge.variation_observations)}")
        if cur_id != v_id and cur_node.selection_experience:
            earlier_notes.append(
                f"Side experiments at implementation {cur_id}: {_prior_note(cur_node.selection_experience, cur_node.selection_observations)}"
            )
    parts.append("\n".join(step_lines))

    if contraction.side_ids:
        chain_index = {cid: idx for idx, cid in enumerate(chain_ids[1:], 1)}
        side_numbers = {nid: idx for idx, nid in enumerate(sorted(contraction.side_ids), 1)}
        side_lines: list[str] = ["Side Attempts from Contracted Intermediates:"]
        side_notes: list[str] = []
        for nid, s_idx in side_numbers.items():
            s_node, s_edge = tree.nodes[nid], tree.edges[nid]
            actual_parent = tree.nodes.get(s_edge.parent_id)
            s_op = "Compressed development" if s_edge.contracted else s_edge.operator or "Init"
            if s_edge.parent_id in chain_index:
                parent_desc = f"Step {chain_index[s_edge.parent_id]}"
            elif s_edge.parent_id in side_numbers:
                parent_desc = f"Side Trial {side_numbers[s_edge.parent_id]}"
            else:
                parent_desc = "initial origin" if actual_parent is None else "parent"
            side_lines.append(
                f"- Side Trial {s_idx} [{s_op}] from {parent_desc}, comparison {s_edge.parent_id}->{nid}: "
                f"{_outcome_word(actual_parent.fitness if actual_parent else None, s_node.fitness)}"
            )
            if s_node.fitness is None:
                side_lines.append(f"  Execution error: {compact_failure(s_node.failure)}")
            if actual_parent is None:
                side_lines.append(f"  Evaluated implementation:\n```python\n{_bare_code(s_node.code)}\n```")
            else:
                side_lines.append(
                    f"  Code change from its parent:\n```diff\n{compact_diff(actual_parent.code, s_node.code)}\n```"
                )
            if s_edge.variation_experience:
                side_notes.append(f"  Development leading to Side Trial {s_idx}: {_prior_note(s_edge.variation_experience, s_edge.variation_observations)}")
            if s_node.selection_experience:
                side_notes.append(f"  Earlier side experiments at Side Trial {s_idx}: {_prior_note(s_node.selection_experience, s_node.selection_observations)}")
        if side_notes:
            earlier_notes.extend(side_notes)
        parts.append("\n".join(side_lines))

    parts.append(
        f"Ending Implementation {v_id}, {_outcome_word(None, v_node.fitness)}:\n"
        f"Code:\n```python\n{_bare_code(v_node.code)}\n```"
    )
    if u_id != tree.root_id:
        parts.append(
            f"{overall}\nNet Code Diff (Ancestor -> Descendant):\n```diff\n"
            f"{compact_diff(tree.nodes[u_id].code, v_node.code)}\n```"
        )

    related = _related_results(tree, set(chain_ids[:-1]), set(chain_ids) | set(contraction.side_ids))
    if related:
        parts.append(related)
    if earlier_notes:
        parts.append("[Earlier Algorithm-Design Experience]\n" + "\n\n".join(earlier_notes))

    return VARIATION_PROMPT_TEMPLATE.format(
        task_header=_task_header(task_description, score_definition, function_template),
        slice_body="\n\n".join(parts) + "\n\n" + _catalog_block(tree, contraction),
    )


def _related_results(tree: TreePopulation, parent_ids: set[int], excluded: set[int]) -> str:
    lines = []
    for parent_id in sorted(parent_ids):
        candidates = [node for i, node in tree.nodes.items()
                      if i not in excluded and node.fitness is not None
                      and tree.edges[i].parent_id == parent_id]
        if not candidates:
            continue
        node = min(candidates, key=lambda n: (n.fitness, n.id))
        origin = "initial population" if parent_id == tree.root_id else f"implementation {parent_id}"
        lines.append(f"- From {origin}: fitness={node.fitness:.6g}; reported design: {clean_prompt_text(node.design)}")
    if not lines:
        return ""
    return ("[Related Results]\nOther evaluated designs from the same starting computations, "
            "provided as context for interpreting the experiments above:\n" + "\n".join(lines))


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
        parts = ["Anchor: Initial population origin (virtual root). These initial implementations have no shared parent implementation."]
    else:
        anchor_node = tree.nodes[p_id]
        parts = [
            f"Parent Baseline {p_id}, fitness={anchor_node.fitness:.6g}:\n"
            f"```python\n{_bare_code(anchor_node.code)}\n```"
        ]
    earlier_notes = []
    existing = _get_node_selection_exp(tree, p_id)
    if existing:
        facts = tree.root_selection_observations if p_id == tree.root_id else tree.nodes[p_id].selection_observations
        earlier_notes.append(f"Earlier experiments at the anchor:\n{_prior_note(existing, facts)}")

    trial_numbers = {nid: idx for idx, nid in enumerate(sorted(consolidation.removed_ids), 1)}
    external_parents = {}
    for nid in trial_numbers:
        parent_id = tree.edges[nid].parent_id
        if parent_id not in trial_numbers and parent_id not in (p_id, tree.root_id):
            external_parents.setdefault(parent_id, len(external_parents) + 1)
    for parent_id, number in external_parents.items():
        parent = tree.nodes[parent_id]
        parts.append(f"Additional parent {number} (implementation {parent_id}, fitness={parent.fitness:.6g}):\n"
                     f"```python\n{_bare_code(parent.code)}\n```")
    for nid, idx in trial_numbers.items():
        node, edge = tree.nodes[nid], tree.edges[nid]
        actual_parent = tree.nodes.get(edge.parent_id)
        op = "Compressed development" if edge.contracted else edge.operator or "Init"
        parent_desc = "initial origin" if actual_parent is None else (
            "anchor parent" if edge.parent_id == p_id else "intermediate parent"
        )
        if edge.parent_id in trial_numbers:
            parent_desc = f"Trial {trial_numbers[edge.parent_id]}"
        elif edge.parent_id in external_parents:
            parent_desc = f"Additional parent {external_parents[edge.parent_id]}"
        rows = [
            f"Trial {idx} [{op}] from {parent_desc}, comparison {edge.parent_id}->{nid}: "
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
            earlier_notes.append(f"Development leading to Trial {idx}: {_prior_note(edge.variation_experience, edge.variation_observations)}")
        if node.selection_experience:
            earlier_notes.append(f"Side experiments at Trial {idx}: {_prior_note(node.selection_experience, node.selection_observations)}")
        parts.append("\n".join(rows))
    related = _related_results(tree, {p_id} | {tree.edges[i].parent_id for i in trial_numbers},
                               {p_id, *trial_numbers, *external_parents})
    if related:
        parts.append(related)
    if earlier_notes:
        parts.append("[Earlier Algorithm-Design Experience]\n" + "\n\n".join(earlier_notes))

    return SELECTION_PROMPT_TEMPLATE.format(
        task_header=_task_header(task_description, score_definition, function_template),
        slice_body="\n\n".join(parts) + "\n\n" + _catalog_block(tree, consolidation),
    )


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
        schema = VARIATION_SCHEMA if variation else SELECTION_SCHEMA
        prompt, response, usage, error = "", "", {}, None
        used_fallback = True
        observations = []
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
                exp, observations = _bind_experience(json.loads(response), tree, region)
                used_fallback = False
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if used_fallback:
            catalog, current, net = comparison_catalog(tree, region)
            # Preserve selected measured facts even if the language call fails.
            ordered = ([net] if net else []) + [key for key in catalog if key in current and key != net]
            ordered += [key for key in catalog if key not in current]
            for key in ordered:
                item = dict(catalog[key])
                item.setdefault("changes", f"Evaluated {item['operator']} computation.")
                item.setdefault("interpretation", "")
                if observations and len(render_observations(observations + [item])) > MAX_EXPERIENCE_CHARS:
                    break
                observations.append(item)
            exp = render_observations(observations)
            # Retain all source facts on outage even when the displayed fallback is short.
            observations = [dict(item) for item in catalog.values()]
            legacy = []
            ids = (*region.removed_ids, region.child_id, *region.side_ids) if variation else region.removed_ids
            for i in ids:
                node_notes = () if variation and i == region.child_id else ((tree.nodes[i].selection_experience, tree.nodes[i].selection_observations),)
                for text, facts in (*node_notes, (tree.edges[i].variation_experience, tree.edges[i].variation_observations)):
                    if text:
                        legacy.append(_prior_note(text, facts))
            if not variation:
                text = _get_node_selection_exp(tree, region.parent_id)
                facts = (tree.root_selection_observations if region.parent_id == tree.root_id
                         else tree.nodes[region.parent_id].selection_observations)
                if text:
                    legacy.append(_prior_note(text, facts))
            if legacy:
                exp += "\nEarlier research notes: " + _truncate_at_clause(
                    "; ".join(dict.fromkeys(legacy)), MAX_EXPERIENCE_CHARS)
        if variation:
            new_tree.edges[target_id].variation_experience = exp
            new_tree.edges[target_id].variation_observations = observations
        else:
            _set_node_selection_exp(new_tree, target_id, exp)
            if target_id == new_tree.root_id:
                new_tree.root_selection_observations = observations
            else:
                new_tree.nodes[target_id].selection_observations = observations
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
