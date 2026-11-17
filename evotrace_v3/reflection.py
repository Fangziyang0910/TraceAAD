"""反思式淘汰：被移除的路径区域凝练为边经验，子树区域凝练为节点经验。"""

from collections.abc import Sequence
from contextlib import nullcontext
import json

from .evaluation import compact_failure
from .models import Edge, ROOT_ID, TreePopulation, clean_text, json_schema, outcome_text
from .topology import PathContraction, Region, apply_simplification, bare_code

EXPERIENCE_SCHEMA = json_schema("evotrace_experience", "experience")

PROMPT_TEMPLATE = """{problem}

[Algorithms]
{situation}

{records}

[Task]
Record factual observations from the algorithms above, keeping earlier observations that still matter.
State what each instructive attempt actually computed (relative to its starting computation, if modified) and its measured fitness or failure.
Report specific computations and outcomes only, without rules, advice, or speculative reasons.

[Output]
Return one JSON object with an "experience" string of about 120 words. Name computations, not algorithm numbers: these algorithms will not be shown again."""


def problem_header(task_description: str, score_definition: str, function_template: str = "") -> str:
    lines = ["[Problem]", task_description.strip(), f"Score: {score_definition.strip()}"]
    if function_template.strip():
        lines.append(f"Function interface:\n```python\n{function_template.strip()}\n```")
    return "\n".join(lines)


def _origin(edge: Edge) -> str:
    if edge.parent_id == ROOT_ID:
        return "developed over several steps from a design made from scratch" if edge.contracted else "designed from scratch"
    if edge.contracted:
        return f"developed from Algorithm {edge.parent_id} over several steps"
    if edge.operator == "Crossover":
        return f"combining Algorithm {edge.parent_id} with another algorithm"
    return f"modified from Algorithm {edge.parent_id}"


def _record(
    tree: TreePopulation, node_id: int, earlier: list[str], *,
    retained: bool = False, attempts: bool = True, best_kept: float | None = None,
) -> str:
    node, edge = tree.nodes[node_id], tree.edges[node_id]
    parent = tree.nodes.get(edge.parent_id)
    result = outcome_text(None if parent is None else parent.fitness, node.fitness)
    if parent is None and node.fitness is not None and best_kept is not None and node.fitness > best_kept:
        result += f" (worse than best kept {best_kept:.6g})"
    name = f"Algorithm {node_id}" + (" (kept)" if retained else "")
    lines = [f"{name}, {_origin(edge)}: {result}", f"Design: {node.design}"]
    if node.fitness is None:
        lines.append(f"Failure: {compact_failure(node.failure)}")
    if edge.experience:
        lines.append(f"Earlier observations on the steps that led to it: {edge.experience}")
        earlier.append(edge.experience)
    if attempts and node.experience:
        lines.append(f"Earlier observations on attempts made from it: {node.experience}")
        earlier.append(node.experience)
    lines.append(f"```python\n{bare_code(node.code)}\n```")
    return "\n".join(lines)


def _anchor(tree: TreePopulation, node_id: int, earlier: list[str], *, attempts: bool = False) -> str:
    node = tree.nodes[node_id]
    lines = [f"Algorithm {node_id} (kept): {outcome_text(None, node.fitness)}", f"Design: {node.design}"]
    if attempts and node.experience:
        lines.append(f"Earlier observations on attempts made from it: {node.experience}")
        earlier.append(node.experience)
    lines.append(f"```python\n{bare_code(node.code)}\n```")
    return "\n".join(lines)


def build_reflection_prompt(tree: TreePopulation, region: Region, problem: str) -> tuple[str, list[str]]:
    """构造单个淘汰区域的事实反思提示，并返回区域内将被取代的已有经验。"""
    earlier: list[str] = []
    if isinstance(region, PathContraction):
        removed = sorted((*region.removed_ids, *region.branch_ids))
        u, v = region.parent_id, region.child_id
        path = " -> ".join(f"Algorithm {i}" for i in (*region.removed_ids, v))
        anchors = sorted({tree.edges[i].parent_id for i in removed} - set(removed) - {ROOT_ID})
        records = [_anchor(tree, i, earlier) for i in anchors]
        records += [_record(tree, i, earlier) for i in removed]
        records.append(_record(tree, v, earlier, retained=True, attempts=False))
        if u == ROOT_ID:
            situation = (f"Algorithm {v} was developed along {path}, starting from a design made from scratch. "
                         "Side attempts along the way are included.")
        else:
            situation = (f"Algorithm {v} was developed from Algorithm {u} along {path}; overall "
                         f"{outcome_text(tree.nodes[u].fitness, tree.nodes[v].fitness)}. "
                         "Side attempts along the way are included.")
    else:
        p = region.parent_id
        removed = sorted(region.removed_ids)
        removed_set = set(removed)
        best_kept = None
        if p == ROOT_ID:
            removed.sort(key=lambda i: (tree.nodes[i].fitness is None, tree.nodes[i].fitness or 0.0))
            best_kept = min(
                (n.fitness for k, n in tree.nodes.items() if k not in removed_set and n.fitness is not None),
                default=None,
            )
        anchors = sorted({tree.edges[i].parent_id for i in removed} - removed_set - {ROOT_ID, p})
        records = [] if p == ROOT_ID else [_anchor(tree, p, earlier, attempts=True)]
        records += [_anchor(tree, i, earlier) for i in anchors]
        records += [_record(tree, i, earlier, best_kept=best_kept) for i in removed]
        if p == ROOT_ID:
            situation = ("The algorithms below were designed from scratch, together with any attempts made from "
                         "them, listed from best to worst fitness (lower fitness is better).")
            if root_exp := clean_text(tree.root_experience):
                earlier.append(root_exp)
                records.insert(0, "Earlier observations: " + root_exp)
        else:
            situation = (f"The algorithms below are attempts that started from Algorithm {p}, directly or from "
                         "one another.")
    prompt = PROMPT_TEMPLATE.format(problem=problem, situation=situation, records="\n\n".join(records))
    return prompt, list(dict.fromkeys(earlier))


def reflect_and_apply(
    tree: TreePopulation,
    regions: Sequence[Region],
    *,
    model,
    problem: str,
    temperature: float = 0.3,
    max_tokens: int = 1536,
    timeout: float = 180.0,
) -> tuple[TreePopulation, list[dict]]:
    """在同一棵简化前树上对各淘汰区域执行反思，写回边经验或节点经验。"""
    new_tree = apply_simplification(tree, regions)
    records: list[dict] = []
    for region in regions:
        is_edge = isinstance(region, PathContraction)
        prompt, earlier = build_reflection_prompt(tree, region, problem)
        usage, error = {}, None
        try:
            with getattr(model, "request_context", nullcontext)():
                reply = model.complete(prompt, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
            content = reply if isinstance(reply, str) else reply.content
            usage = dict(getattr(reply, "usage", None) or {})
            experience = clean_text(json.loads(content)["experience"])
            if not experience:
                raise ValueError("Reflection returned an empty experience")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            experience = " ".join(earlier)
        if is_edge:
            new_tree.edges[region.child_id].experience = experience
        elif region.parent_id == ROOT_ID:
            new_tree.root_experience = experience
        else:
            new_tree.nodes[region.parent_id].experience = experience
        records.append({
            "role": "reflection",
            "kind": "edge" if is_edge else "node",
            "target_id": region.child_id if is_edge else region.parent_id,
            "status": "error" if error else "success",
            "usage": usage,
            "error": error,
            "experience": experience,
        })
    return new_tree, records
