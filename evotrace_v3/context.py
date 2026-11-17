"""按演化树组织生成提示：当前算法的形成过程与尝试、参考卡片与已有做法卡片。"""

from collections.abc import Sequence

from .evaluation import TIMEOUT_PREFIX, compact_failure, failure_feedback
from .models import Edge, Node, Operator, Program, ROOT_ID, TreePopulation, clean_text, outcome_text
from .task import Task
from .topology import bare_code

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
- "analysis": Briefly explain the computation you choose for this attempt and how it serves the task, using the supplied evidence when relevant.
- "code": Provide the complete implementation, including necessary imports and helper definitions.
- "design": In about 60 words, describe for another algorithm designer the main computation the code performs and its key parameters. Describe the implementation, not expected performance.
Preserve the required function signature and satisfy the task constraints and input/output requirements.
Return only the JSON object, without Markdown fences or additional text."""


def _path_from_root(tree: TreePopulation, node_id: int) -> list[int]:
    path: list[int] = []
    cur = node_id
    while cur != ROOT_ID:
        path.append(cur)
        cur = tree.edges[cur].parent_id
    path.reverse()
    return path


def _origin(edge: Edge) -> str:
    if edge.parent_id == ROOT_ID:
        return "developed over several steps from a design made from scratch" if edge.contracted else "designed from scratch"
    return ("developed over several steps" if edge.contracted
            else "combined with another algorithm" if edge.operator == "Crossover" else "modified")


def _experiences(tree: TreePopulation, node_id: int, *, include: bool, node: bool = True) -> list[str]:
    if not include:
        return []
    lines = []
    if tree.edges[node_id].experience:
        lines.append(f"Experience from the steps that led to it: {tree.edges[node_id].experience}")
    if node and tree.nodes[node_id].experience:
        lines.append(f"Experience from attempts made from it: {tree.nodes[node_id].experience}")
    return lines


def _lineage_blocks(
    tree: TreePopulation, parent_id: int, *, include_experiences: bool, exclude: frozenset[int] = frozenset()
) -> tuple[list[str], set[tuple[str, int]]]:
    parent = tree.nodes[parent_id]
    path = _path_from_root(tree, parent_id)
    rows = ["[Development of the Current Algorithm]"]
    for k, node_id in enumerate(path):
        node, edge = tree.nodes[node_id], tree.edges[node_id]
        prev = tree.nodes.get(edge.parent_id)
        label = "First version" if k == 0 else f"Step {k}"
        head = f"{label} ({_origin(edge)}): {outcome_text(None if prev is None else prev.fitness, node.fitness)}"
        rows.append("\n".join([
            head,
            f"Design: {node.design}",
            *_experiences(tree, node_id, include=include_experiences, node=node_id != parent_id),
        ]))
    blocks = ["\n\n".join(rows)]

    shown = [tree.nodes[cid] for cid, edge in sorted(tree.edges.items())
             if edge.parent_id == parent_id and cid not in exclude]
    parts = []
    if include_experiences and parent.experience:
        parts.append(f"Experience from earlier attempts: {parent.experience}")
    for idx, child in enumerate(shown, 1):
        edge = tree.edges[child.id]
        result = (f"failed - {compact_failure(child.failure)}" if child.fitness is None
                  else outcome_text(parent.fitness, child.fitness))
        parts.append("\n".join([
            f"Attempt {idx} ({_origin(edge)}): {result}",
            f"Design: {child.design}",
            *_experiences(tree, child.id, include=include_experiences),
        ]))
    if parts:
        blocks.append("\n\n".join(["[Attempts from the Current Algorithm]\nIn order of creation.", *parts]))
    seen = {(kind, j) for j in (*path, *(c.id for c in shown)) for kind in ("edge", "node")}
    return blocks, seen


def _reference_card(tree: TreePopulation, node_id: int, seen: set[tuple[str, int]], *, include_experiences: bool) -> str:
    node = tree.nodes[node_id]
    lines = [f"fitness = {node.fitness:.6g}", f"Design: {node.design}"]
    if include_experiences:
        exp = []
        for j in _path_from_root(tree, node_id):
            whose = "it" if j == node_id else "an earlier version"
            if tree.edges[j].experience and ("edge", j) not in seen:
                exp.append(f"- Steps that led to {whose}: {tree.edges[j].experience}")
            if tree.nodes[j].experience and ("node", j) not in seen:
                exp.append(f"- Attempts made from {whose}: {tree.nodes[j].experience}")
            seen.update({("edge", j), ("node", j)})
        if exp:
            lines += ["Experience along its development:", *exp]
    lines.append(f"```python\n{bare_code(node.code)}\n```")
    return "\n".join(lines)


def _current_code(node: Node) -> str:
    return f"[Current Algorithm]\nfitness = {node.fitness:.6g}\n```python\n{bare_code(node.code)}\n```"


def build_evolution_context(
    tree: TreePopulation,
    parent_id: int,
    operator: Operator,
    reference_id: int | None = None,
    *,
    explore_ids: Sequence[int] = (),
    include_experiences: bool = True,
) -> tuple[str, str]:
    """按算子返回 (树上下文文本, 指令文本)。"""
    if operator == "Explore":
        blocks = []
        if include_experiences and (root_exp := clean_text(tree.root_experience)):
            blocks.append("[Experience from Earlier Approaches]\n" + root_exp)
        shown = sorted({i for i in explore_ids if i in tree.nodes}, key=lambda i: (tree.nodes[i].fitness, i))
        if shown:
            cards = [
                f"Approach {k}: fitness = {tree.nodes[i].fitness:.6g}\nDesign: {tree.nodes[i].design}"
                for k, i in enumerate(shown, 1)
            ]
            blocks.append("\n\n".join(["[Existing Approaches]\nFrom best to worst fitness.", *cards]))
        instruction = (
            "Rethink how this task can be solved and implement an alternative whose main computation differs from the existing approaches. "
            "Develop a new way to compute the result, rather than another weighting of the same local features. "
            "The algorithms and experience above are starting points for ideas, not a structure you must preserve. "
            "Consider changes in the information used, the representation, or the procedure that produces the result. "
            "Choose an approach suited to the task and implement its main computation."
        )
        return "\n\n".join(blocks), instruction

    parent = tree.nodes[parent_id]
    if operator == "Refine":
        lineage, _ = _lineage_blocks(tree, parent_id, include_experiences=include_experiences)
        blocks = [*lineage, _current_code(parent)]
        instruction = (
            "Improve the current algorithm by examining how it computes its result and where this computation can be improved. "
            "Use the implementation, outcomes, and development experience to choose a meaningful modification. "
            "You may revise its representation, the information it uses, or its procedure as well as its parameters; "
            "implement a coherent improved algorithm."
        )
    elif operator == "Crossover":
        if reference_id is None or reference_id not in tree.nodes:
            raise ValueError("Crossover requires a valid retained reference_id")
        lineage, seen = _lineage_blocks(
            tree, parent_id, include_experiences=include_experiences, exclude=frozenset({reference_id})
        )
        ref_card = _reference_card(tree, reference_id, seen, include_experiences=include_experiences)
        blocks = [
            *lineage,
            "[Other Algorithm]\nAn algorithm whose ideas may be combined with the current algorithm.\n" + ref_card,
            _current_code(parent),
        ]
        instruction = (
            "Understand the computational ideas in both implementations and develop an algorithm that combines their useful strengths. "
            "Use their development and experimental experience to reason about how the computations fit together. "
            "If no useful combination emerges, improve the current algorithm using what the comparison reveals."
        )
    else:
        raise ValueError(f"Unsupported evolution operator: {operator!r}")
    return "\n\n".join(blocks), instruction


def _render(task: Task, context: str, instruction: str, *, has_scores: bool = False) -> str:
    template = COMMON_TEMPLATE
    if not context:
        template = template.replace("[Context]\n{context}\n\n", "")
    elif has_scores and task.score_definition not in task.task_description:
        context = f"Score definition: {task.score_definition}\n\n{context}"
    return template.format_map({
        "task_description": task.task_description,
        "function_template": task.function_template,
        "context": context,
        "instruction": instruction,
    })


def evolution_prompt(
    task: Task,
    tree: TreePopulation,
    parent_id: int,
    operator: Operator,
    reference_id: int | None = None,
    *,
    explore_ids: Sequence[int] = (),
    include_experiences: bool = True,
) -> str:
    context, instruction = build_evolution_context(
        tree, parent_id, operator, reference_id, explore_ids=explore_ids, include_experiences=include_experiences
    )
    return _render(task, context, instruction, has_scores=True)


def init_prompt(task: Task, archive: dict[int, Program], failure_count: int = 2) -> str:
    context = []
    if 0 in archive:
        example = archive[0]
        text = "[Example Algorithm]\n"
        if design := clean_text(example.design):
            text += f"Design: {design}\n"
        context.append(text + f"```python\n{bare_code(example.code)}\n```")
    valid = sorted((p for p in archive.values() if p.valid and p.operator == "Init"), key=lambda p: p.id)
    if valid:
        algorithms = [
            f"Approach {k}: fitness = {p.evaluation.fitness:.6g}\nDesign: {clean_text(p.design)}"
            for k, p in enumerate(valid, 1)
        ]
        context.append("[Existing Approaches]\n\n" + "\n\n".join(algorithms))
    failed = sorted(
        sorted(
            (p for p in archive.values() if p.parent_id == 0 and p.evaluation is not None and p.evaluation.status != "success"),
            key=lambda p: -p.id,
        )[:failure_count],
        key=lambda p: p.id,
    )
    if failed:
        context.append("[Failed Attempts]\n" + "\n\n".join(
            f"Failed attempt {k}\nDesign: {clean_text(p.design)}\n"
            f"```python\n{bare_code(p.code)}\n```\nFailure: {failure_feedback(p.evaluation.failure)}"
            for k, p in enumerate(failed, 1)
        ))
    instruction = (
        "Develop an effective algorithm with a different main computation from the ideas already represented. "
        "Seek a different way to compute the result, rather than another weighting of the same local features. "
        "You can change the representation, the information used, or the computational procedure. "
        "Implement the chosen approach as a complete algorithm."
    ) if valid else (
        "Design an effective algorithm for this task. Consider how the returned result affects the complete solution "
        "and choose a computational approach suited to that role. The example illustrates an implementation, "
        "not a structure to preserve. Implement the specified function."
    )
    if failed:
        instruction += "\nUse the failed attempts and evaluation feedback to inform your implementation."
    return _render(task, "\n\n".join(context), instruction, has_scores=bool(valid))


def repair_format_prompt(task: Task, response: str, feedback: str) -> str:
    context = f"[Original Response]\n{response}\n\n[Parsing Feedback]\n{feedback}"
    return _render(task, context, (
        "Convert the original response into the JSON object specified in [Output], using the "
        "parsing feedback to correct its structure, required fields, and string escaping.\n"
        "Preserve the existing algorithmic idea and computational logic.\n"
        "Complete any missing or truncated content consistently with that approach and the "
        "task requirements."
    ))


def repair_program_prompt(task: Task, program: Program) -> str:
    failure = program.evaluation.failure
    feedback = (
        f"The evaluation did not finish within the total time limit of {task.time_limit:g} seconds."
        if failure.startswith(TIMEOUT_PREFIX) else failure_feedback(failure)
    )
    context = (
        f"[Current Algorithm]\nDesign: {clean_text(program.design)}\n"
        f"```python\n{bare_code(program.code)}\n```\n\n[Failure Feedback]\n{feedback}"
    )
    instruction = (
        "Trace the reported exception or invalid result to its cause in the code, using the "
        "design to understand the intended algorithm.\n"
        "Correct the faulty computations or data handling so the function executes correctly "
        "and satisfies the task requirements, while retaining the intended core computation."
    ) if program.evaluation.status == "error" else (
        "Make the algorithm finish within the reported time limit. "
        "Reduce the cost of its core computation, using an approximation when needed, "
        "while retaining the intended algorithmic idea."
    )
    return _render(task, context, instruction)
