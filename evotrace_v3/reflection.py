"""反思树简化：被淘汰的区域凝练为经验。路径收缩写入边经验，子树剪枝写入节点经验。"""

import ast
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import re

from .models import Edge, TreePopulation
from .topology import _strip_docstrings, PathContraction, SimplificationPlan, apply_simplification


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


def _bare_code(code: str) -> str:
    """去除文档字符串后的标准化源码，使展示聚焦实际计算。"""
    try:
        return ast.unparse(_strip_docstrings(code))
    except (SyntaxError, ValueError):
        return code.strip()


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
        return f"fitness = {child_fit:.6g}"
    outcome = "improved" if child_fit < parent_fit else "tied" if child_fit == parent_fit else "regressed"
    return f"{parent_fit:.6g} -> {child_fit:.6g} ({outcome})"


def _operator_label(tree: TreePopulation, edge: Edge) -> str:
    if edge.contracted:
        return "Compressed development"
    label = edge.operator or "Init"
    if label == "Crossover" and edge.reference_id in tree.nodes:
        label += f" with reference Algorithm {edge.reference_id}"
    return label


HISTORY_NOTE = ("A compressed development stands for removed intermediate algorithms and is summarized by its "
                "stage experience. Experience from attempts at an algorithm summarizes removed attempts that "
                "started from it.")

EXPERIENCE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "evotrace_experience", "strict": True,
    "schema": {"type": "object", "additionalProperties": False,
               "properties": {"experience": {"type": "string"}}, "required": ["experience"]}}}

PROMPT_TEMPLATE = """{problem}

[Algorithms]
{situation} Each record gives the origin and outcome, the design, the failure if any, earlier experience, and the code. {history_note}

{records}

[Task]
Summarize the algorithm-design experience from the algorithms above. It will be stored {placement}.
For each finding, state the starting computation, the change tried, and the measured outcome. Carry forward the relevant earlier experience. Do not turn a single trial into a general rule or prohibition.

[Output]
Return one JSON object with an "experience" string of about 120 words. Name computations, not algorithm numbers: the removed algorithms will not appear later."""


def problem_header(task_description: str, score_definition: str, function_template: str = "") -> str:
    lines = ["[Problem]", task_description.strip(), f"Score: {score_definition.strip()}"]
    if function_template.strip():
        lines.append(f"Function interface:\n```python\n{function_template.strip()}\n```")
    return "\n".join(lines)


def _record(tree: TreePopulation, node_id: int, earlier: list[str], *, retained: bool = False,
            stage: bool = True, attempts: bool = True) -> str:
    """一个算法的记录：来源与结果 → design → 失败原因 → 经验 → 代码。

    stage / attempts 指是否给出它的入边经验与节点经验；给出的经验会被新经验取代，所以同时记入 earlier。
    """
    node, edge = tree.nodes[node_id], tree.edges[node_id]
    name = f"Algorithm {node_id}" + (" (retained)" if retained else "")
    parent = tree.nodes.get(edge.parent_id)
    if parent is None:
        origin = "compressed development from the initial population" if edge.contracted else "initial population"
        head = f"{name} ({origin}): {_outcome_word(None, node.fitness)}"
    else:
        head = (f"{name} <- Algorithm {edge.parent_id} ({_operator_label(tree, edge)}): "
                f"{_outcome_word(parent.fitness, node.fitness)}")
    lines = [head, f"Design: {clean_prompt_text(node.design)}"]
    if node.fitness is None:
        lines.append(f"Failure: {compact_failure(node.failure)}")
    for shown, text, label in ((stage, edge.experience, "Stage experience"),
                               (attempts, node.experience, "Experience from attempts at this algorithm")):
        if shown and text:
            lines.append(f"{label}: {clean_prompt_text(text)}")
            earlier.append(clean_prompt_text(text))
    lines.append(f"```python\n{_bare_code(node.code)}\n```")
    return "\n".join(lines)


def _anchor(tree: TreePopulation, node_id: int, earlier: list[str], *, attempts: bool = False) -> str:
    """区域外仍保留的算法，作为比较基准：design 与代码；只在其节点经验将被取代时给出该经验。"""
    node = tree.nodes[node_id]
    lines = [f"Algorithm {node_id} (retained): {_outcome_word(None, node.fitness)}",
             f"Design: {clean_prompt_text(node.design)}"]
    if attempts and node.experience:
        lines.append(f"Experience from attempts at this algorithm: {clean_prompt_text(node.experience)}")
        earlier.append(clean_prompt_text(node.experience))
    lines.append(f"```python\n{_bare_code(node.code)}\n```")
    return "\n".join(lines)


def build_reflection_prompt(tree: TreePopulation, region, problem: str) -> tuple[str, list[str]]:
    """一个淘汰区域的反思提示，以及区域内将被新经验取代的已有经验。"""
    earlier: list[str] = []
    removed = sorted(region.removed_ids)
    if isinstance(region, PathContraction):
        removed = sorted((*region.removed_ids, *region.branch_ids))
        u, v = region.parent_id, region.child_id
        path = " -> ".join(f"Algorithm {i}" for i in (*region.removed_ids, v))
        anchors = sorted({tree.edges[i].parent_id for i in removed} - set(removed) - {tree.root_id})
        records = [_anchor(tree, i, earlier) for i in anchors]
        records += [_record(tree, i, earlier) for i in removed]
        records.append(_record(tree, v, earlier, retained=True, attempts=False))
        if u == tree.root_id:
            situation = (f"Algorithm {v} developed from the initial population along {path}. "
                         "The intermediate algorithms and their side branches will be removed.")
            placement = f"with Algorithm {v} as the experience of how it developed from the initial population"
        else:
            situation = (f"Algorithm {v} developed from Algorithm {u} along {path}; overall "
                         f"{_outcome_word(tree.nodes[u].fitness, tree.nodes[v].fitness)}. The intermediate "
                         f"algorithms and their side branches will be removed, and Algorithm {u} will lead "
                         f"directly to Algorithm {v}.")
            placement = f"with Algorithm {v} as the experience of how it developed from Algorithm {u}"
    else:
        p = region.parent_id
        anchors = sorted({tree.edges[i].parent_id for i in removed} - set(removed) - {tree.root_id, p})
        records = [] if p == tree.root_id else [_anchor(tree, p, earlier, attempts=True)]
        records += [_anchor(tree, i, earlier) for i in anchors]
        records += [_record(tree, i, earlier) for i in removed]
        if p == tree.root_id:
            situation = ("The algorithms below are initial algorithms and attempts derived from them. "
                         "They will be removed from the population.")
            placement = "as the experience from the initial population"
            if tree.root_experience:
                earlier.append(clean_prompt_text(tree.root_experience))
                records.insert(0, "Earlier experience from the initial population: "
                               + clean_prompt_text(tree.root_experience))
        else:
            situation = (f"The algorithms below are attempts that started from Algorithm {p}, directly or through "
                         f"other removed attempts. They will be removed; Algorithm {p} stays.")
            placement = f"with Algorithm {p} as the experience from attempts that started from it"
    prompt = PROMPT_TEMPLATE.format(problem=problem, situation=situation, history_note=HISTORY_NOTE,
                                    records="\n\n".join(records), placement=placement)
    return prompt, list(dict.fromkeys(earlier))


@dataclass(frozen=True)
class ReflectionCall:
    kind: str  # "edge" | "node"
    target_id: int  # 边经验写入收缩边的子节点；节点经验写入保留节点（虚拟根为 0）
    experience: str
    prompt: str
    response: str = ""
    usage: dict = field(default_factory=dict)
    error: str | None = None


def reflect_and_apply(
    tree: TreePopulation,
    plan: SimplificationPlan,
    *,
    model,
    problem: str,
    temperature: float = 0.3,
    max_tokens: int = 1536,
    timeout: float = 180.0,
) -> tuple[TreePopulation, list[ReflectionCall]]:
    """执行树简化，每个区域调用一次反思，把经验写回收缩边或保留节点。

    所有反思读取同一棵简化前的树。调用失败时沿用区域内已有的经验，淘汰照常进行。
    """
    new_tree = apply_simplification(tree, plan)
    calls = []
    for region in (*plan.path_contractions, *plan.subtree_prunings):
        is_edge = isinstance(region, PathContraction)
        prompt, earlier = build_reflection_prompt(tree, region, problem)
        response, usage, error = "", {}, None
        try:
            with getattr(model, "request_context", nullcontext)():
                reply = model.complete(prompt, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
            response, usage = getattr(reply, "content", reply), dict(getattr(reply, "usage", None) or {})
            experience = clean_prompt_text(json.loads(response)["experience"])
            if not experience:
                raise ValueError("Reflection returned an empty experience")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            experience = " ".join(earlier)
        if is_edge:
            new_tree.edges[region.child_id].experience = experience
        elif region.parent_id == tree.root_id:
            new_tree.root_experience = experience
        else:
            new_tree.nodes[region.parent_id].experience = experience
        calls.append(ReflectionCall("edge" if is_edge else "node",
                                    region.child_id if is_edge else region.parent_id,
                                    experience, prompt, response, usage, error))
    return new_tree, calls
