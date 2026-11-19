"""反思式淘汰：被移除的路径区域凝练为边经验，算法节点下的子树区域凝练为该节点的节点经验；虚拟根下的子树区域直接移除。

反思把原始尝试的生成意图、收缩路径的已有经验与受评实现和结果对照。
子树区的尝试另与起点算法比较；路径区只给每一步相对自身起点的变化。
"""

from collections.abc import Sequence
from contextlib import nullcontext
import json

from .evaluation import compact_failure
from .models import Edge, ROOT_ID, TreePopulation, clean_text, json_schema, outcome_text
from .topology import PathContraction, Region, apply_simplification, bare_code

EXPERIENCE_SCHEMA = json_schema("evotrace_experience", "analysis", "experience")

PROMPT_TEMPLATE = """{problem}

[Algorithms]
{situation}

{records}

[Task]
{question}
Compare what each attempt intended, what its code computed, and how it performed. What would a designer miss from the current implementation and its score alone? Ground each judgment in the outcomes shown and preserve comparisons that matter for the next design decision. When several changes were tested together, report their combined outcome and treat individual contributions as hypotheses unless another comparison isolates them. If an idea still looks worth developing after a poor result, explain why in light of what was actually tested.{merge}

[Output]
Return a JSON object with two strings. In "analysis", briefly compare the attempts. In "experience", state the useful design insight as briefly as it warrants. Describe the computations in words, omitting algorithm numbers and parameter values. If there is nothing to add or revise, return an empty string for "experience"."""

MERGE = (" Earlier experience summarizes previous attempts. Repeating it adds no evidence. "
         "If these attempts change or extend it, write the updated experience, keeping earlier insights "
         "that still apply. An empty experience keeps the earlier notes.")


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
        return f"generated from Algorithm {edge.parent_id} aiming to combine it with another algorithm"
    if edge.operator == "Explore":
        return f"generated from Algorithm {edge.parent_id} aiming at a different core computation"
    return f"generated from Algorithm {edge.parent_id} aiming to improve it"


def _record(
    tree: TreePopulation, node_id: int, earlier: list[str], *,
    role: str = "", base_id: int | None = None, attempts: bool = True,
) -> str:
    """一次尝试：结果行（相对自身起点；子树区另与起点算法比较）、生成前分析、设计、已有经验与源码。"""
    node, edge = tree.nodes[node_id], tree.edges[node_id]
    parent = tree.nodes.get(edge.parent_id)
    head = f"Algorithm {node_id}{f', {role}' if role else ''}, {_origin(edge)}: "
    head += outcome_text(None if parent is None else parent.fitness, node.fitness)
    base = None if base_id is None else tree.nodes[base_id].fitness
    if base is not None and edge.parent_id != base_id and node.fitness is not None:
        word = "better than" if node.fitness < base else "tied with" if node.fitness == base else "worse than"
        head += f"; {word} Algorithm {base_id}"
    lines = [head]
    if not edge.contracted and node.analysis:
        lines.append(f"Analysis before coding: {node.analysis}")
    lines.append(f"Reported design: {node.design}")
    if node.fitness is None:
        lines.append(f"Failure: {compact_failure(node.failure)}")
    if edge.experience:
        lines.append(f"Earlier experience from the steps that led to it: {edge.experience}")
        earlier.append(edge.experience)
    if attempts and node.experience:
        lines.append(f"Earlier experience from attempts made from it: {node.experience}")
        earlier.append(node.experience)
    lines.append(f"```python\n{bare_code(node.code)}\n```")
    return "\n".join(lines)


def _anchor(tree: TreePopulation, node_id: int, earlier: list[str], *, role: str = "", attempts: bool = False) -> str:
    node = tree.nodes[node_id]
    lines = [f"Algorithm {node_id}{f', {role}' if role else ''}: {outcome_text(None, node.fitness)}", f"Reported design: {node.design}"]
    if attempts and node.experience:
        lines.append(f"Earlier experience from attempts made from it: {node.experience}")
        earlier.append(node.experience)
    lines.append(f"```python\n{bare_code(node.code)}\n```")
    return "\n".join(lines)


def build_reflection_prompt(tree: TreePopulation, region: Region, problem: str) -> tuple[str, list[str]]:
    """构造单个淘汰区域的设计反思提示，并返回区域内将被取代的已有经验。"""
    earlier: list[str] = []
    if isinstance(region, PathContraction):
        u, v = region.parent_id, region.child_id
        removed = sorted((*region.removed_ids, *region.branch_ids))
        on_path = set(region.removed_ids)
        path = " -> ".join(f"Algorithm {i}" for i in (*region.removed_ids, v))
        anchors = sorted({tree.edges[i].parent_id for i in removed} - set(removed) - {ROOT_ID})
        records = [_anchor(tree, i, earlier, role="the start of the path" if i == u else "") for i in anchors]
        records += [_record(tree, i, earlier, role="on the path" if i in on_path else "a side attempt") for i in removed]
        records.append(_record(tree, v, earlier, role="the end of the path", attempts=False))
        if u == ROOT_ID:
            situation = (f"Algorithm {v} was developed along {path}, starting from a design made from scratch. "
                         "Side attempts along the way are included.")
        else:
            situation = (f"Algorithm {v} was developed from Algorithm {u} along {path}; overall "
                         f"{outcome_text(tree.nodes[u].fitness, tree.nodes[v].fitness)}. "
                         "Side attempts along the way are included.")
        question = (f"Reflect, as an algorithm designer, on how Algorithm {v} was developed. "
                    "Which ideas shaped it, and did any intermediate or side approach work better than the final design?")
    else:
        p = region.parent_id
        if p == ROOT_ID:
            raise ValueError("Subtree regions under the virtual root are removed without reflection")
        removed = sorted(region.removed_ids)
        anchors = sorted({tree.edges[i].parent_id for i in removed} - set(removed) - {p})
        records = [_anchor(tree, p, earlier, role="the starting algorithm", attempts=True)]
        records += [_anchor(tree, i, earlier) for i in anchors]
        records += [_record(tree, i, earlier, base_id=p) for i in removed]
        situation = f"The algorithms below are attempts that started from Algorithm {p}, directly or from one another."
        question = (f"What did the branches from Algorithm {p} try? "
                    "What do their outcomes show about the implementations tested?")
    prompt = PROMPT_TEMPLATE.format(
        problem=problem, situation=situation, records="\n\n".join(records),
        question=question, merge=MERGE if earlier else "",
    )
    return prompt, list(dict.fromkeys(earlier))


def reflect_and_apply(
    tree: TreePopulation,
    regions: Sequence[Region],
    *,
    model,
    problem: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 180.0,
) -> tuple[TreePopulation, list[dict]]:
    """在同一更新前树状态上对各满块区域执行反思，写回边经验或骨架节点的节点经验。

    虚拟根下的子树区域只移除、不反思，因为虚拟根不是可供后续算子读取的算法个体。
    回复先写 analysis（对照各尝试的意图、代码与结果），再写 experience；analysis 只留存在记录中。
    """
    new_tree = apply_simplification(tree, regions)
    records: list[dict] = []
    for region in regions:
        is_edge = isinstance(region, PathContraction)
        if not is_edge and region.parent_id == ROOT_ID:
            continue
        prompt, earlier = build_reflection_prompt(tree, region, problem)
        usage, error, analysis = {}, None, None
        try:
            with getattr(model, "request_context", nullcontext)():
                reply = model.complete(prompt, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
            content = reply if isinstance(reply, str) else reply.content
            usage = dict(getattr(reply, "usage", None) or {})
            value = json.loads(content)
            if not isinstance(value["experience"], str):
                raise ValueError("Reflection experience must be a string")
            experience = clean_text(value["experience"])
            analysis = clean_text(value.get("analysis"))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            experience = ""
        experience = experience or " ".join(earlier)
        if is_edge:
            new_tree.edges[region.child_id].experience = experience
        else:
            new_tree.nodes[region.parent_id].experience = experience
        records.append({
            "role": "reflection",
            "kind": "edge" if is_edge else "node",
            "target_id": region.child_id if is_edge else region.parent_id,
            "status": "error" if error else "success",
            "usage": usage,
            "error": error,
            "analysis": analysis,
            "experience": experience,
        })
    return new_tree, records
