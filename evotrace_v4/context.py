"""按算子组织生成提示：精化读形成路径，交叉另读参考个体，探索读种群中的不同算法。"""

from collections.abc import Sequence

from .evaluation import TIMEOUT_PREFIX, compact_failure, failure_feedback
from .models import Edge, Node, Operator, Program, ROOT_ID, TreePopulation, clean_text, outcome_text
from .task import Task
from .topology import bare_code

ALGORITHM_ANALYSIS = (
    "Use the structure of the problem to work out a concrete computation. "
    "Which choices would it favor, and why could they improve the final solution? "
    "Draw on earlier results when useful."
)

# 可选材料按渲染后的树上下文字符数取舍。主父代始终保留，端点分词随模型而异。
CONTEXT_CHARS = 90_000

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
- "analysis": {analysis_instruction} Use up to 300 words.
- "code": Provide the complete implementation using only the inputs in [Function interface], including necessary imports and helper definitions.
- "design": In about 120 words, describe for another algorithm designer the main computation the code performs and its key parameters. Describe the implementation, not expected performance.
Keep "design" faithful to "code". Preserve the required function signature and satisfy the task constraints and input/output requirements.
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
    if edge.contracted:
        return "developed over several steps"
    if edge.operator == "Crossover":
        return "aimed at combining with another algorithm"
    if edge.operator == "Explore":
        return "aimed at a different core computation"
    return "aimed at improvement"


def _experiences(
    tree: TreePopulation, node_id: int, *, include: bool, node: bool = True, seen: frozenset[int] | set[int] = frozenset(),
) -> list[str]:
    """个体自身的入边经验与节点经验；已在上文出现过的个体不再重复。"""
    if not include or node_id in seen:
        return []
    lines = []
    if tree.edges[node_id].experience:
        lines.append(f"Local reflection on the steps that led to it: {tree.edges[node_id].experience}")
    if node and tree.nodes[node_id].experience:
        lines.append(f"Local reflection on attempts made from it: {tree.nodes[node_id].experience}")
    return lines


def _lineage_blocks(
    tree: TreePopulation, parent_id: int, *, include_experiences: bool, exclude: int | None = None,
) -> tuple[list[str], set[int]]:
    """形成路径（逐边的生成意图或收缩经验），以及从主父代出发、仍在树上的尝试；返回已展示的个体。"""
    path = _path_from_root(tree, parent_id)
    rows = ["[Development of the Current Algorithm]"]
    for k, node_id in enumerate(path):
        node, edge = tree.nodes[node_id], tree.edges[node_id]
        prev = tree.nodes.get(edge.parent_id)
        label = "First version" if k == 0 else f"Step {k}"
        if node_id == parent_id:
            label += ", the current algorithm"
        lines = [f"{label} ({_origin(edge)}): {outcome_text(None if prev is None else prev.fitness, node.fitness)}"]
        if not edge.contracted and node.analysis:
            lines.append(f"Reported intent when this version was generated (references are local to that step): {node.analysis}")
        lines.append(f"Reported design: {node.design}")
        lines.extend(_experiences(tree, node_id, include=include_experiences, node=node_id != parent_id))
        rows.append("\n".join(lines))
    blocks = ["\n\n".join(rows)]

    parent = tree.nodes[parent_id]
    children = [i for i, edge in sorted(tree.edges.items()) if edge.parent_id == parent_id and i != exclude]
    attempts = []
    if include_experiences and parent.experience:
        attempts.append(f"Local reflection on earlier attempts: {parent.experience}")
    for k, i in enumerate(children, 1):
        child = tree.nodes[i]
        result = (f"failed - {compact_failure(child.failure, child.code)}" if child.fitness is None
                  else outcome_text(parent.fitness, child.fitness))
        attempts.append("\n".join([
            f"Attempt {k} ({_origin(tree.edges[i])}): {result}",
            f"Reported design: {child.design}",
            *_experiences(tree, i, include=include_experiences),
        ]))
    if attempts:
        heading = "[Attempts from the Current Algorithm]" + ("\nIn order of creation." if children else "")
        blocks.append("\n\n".join([heading, *attempts]))
    return blocks, {*path, *children}


def _card(tree: TreePopulation, node_id: int, *, include_experiences: bool, seen: frozenset[int] | set[int] = frozenset()) -> list[str]:
    """个体卡片：设计简述、自身的入边与节点经验，源码放在最后。"""
    return [
        f"Reported design: {tree.nodes[node_id].design}",
        *_experiences(tree, node_id, include=include_experiences, seen=seen),
        f"```python\n{bare_code(tree.nodes[node_id].code)}\n```",
    ]


def _current_code(node: Node) -> str:
    return f"[Current Algorithm]\n{outcome_text(None, node.fitness)}\n```python\n{bare_code(node.code)}\n```"


def build_evolution_context(
    tree: TreePopulation,
    parent_id: int,
    operator: Operator,
    reference_id: int | None = None,
    *,
    explore_ids: Sequence[int] = (),
    include_experiences: bool = True,
) -> tuple[str, str]:
    """按算子返回 (树上下文文本, 指令文本)。

    精化：形成路径、近期尝试及局部反思，空间允许时附上保留前驱的源码，最后是主父代源码。
    交叉：形成路径、近期尝试、参考个体卡片与主父代源码。
    探索：主父代卡片与至多 M 个异质代表卡片，各含设计、成绩、自身经验与源码；不含形成路径与尝试列表。
    可选材料（前驱源码、代表卡片）只在渲染后的树上下文不超过 CONTEXT_CHARS 时放入。
    """
    parent = tree.nodes[parent_id]
    if operator == "Explore":
        current = "\n".join(["[Current Algorithm]", outcome_text(None, parent.fitness),
                             *_card(tree, parent_id, include_experiences=include_experiences)])

        def render(cards: list[tuple[float, int, str]]) -> str:
            if not cards:
                return current
            approaches = [f"Approach {k}: {body}" for k, (_, _, body) in enumerate(sorted(cards), 1)]
            return "\n\n".join([current, "[Other Existing Approaches]\nFrom best to worst fitness.", *approaches])

        cards: list[tuple[float, int, str]] = []
        for i in explore_ids:
            fitness = tree.nodes[i].fitness
            card = (fitness, i, "\n".join([outcome_text(None, fitness),
                                           *_card(tree, i, include_experiences=include_experiences)]))
            if len(render([*cards, card])) <= CONTEXT_CHARS:
                cards.append(card)
        instruction = "Develop a different way to compute the result. Choose an approach that looks promising for this task."
        return render(cards), instruction

    if operator == "Refine":
        blocks, _ = _lineage_blocks(tree, parent_id, include_experiences=include_experiences)
        current = _current_code(parent)
        previous = tree.nodes.get(tree.edges[parent_id].parent_id)
        if previous is not None:
            code = bare_code(previous.code)
            listing = ("Same displayed code as the current algorithm." if code == bare_code(parent.code)
                       else f"```python\n{code}\n```")
            comparison = f"[Previous Retained Implementation]\n{outcome_text(None, previous.fitness)}\n{listing}"
            if len("\n\n".join([*blocks, comparison, current])) <= CONTEXT_CHARS:
                blocks.append(comparison)
        blocks.append(current)
        instruction = "Improve the current algorithm. Choose the modification you judge most promising for this task."
    else:
        blocks, seen = _lineage_blocks(tree, parent_id, include_experiences=include_experiences, exclude=reference_id)
        ref = tree.nodes[reference_id]
        blocks.append("\n".join([
            "[Other Algorithm]",
            "An algorithm whose ideas may be combined with the current algorithm.",
            outcome_text(None, ref.fitness),
            *_card(tree, reference_id, include_experiences=include_experiences, seen=seen),
        ]))
        blocks.append(_current_code(parent))
        instruction = "Develop the current algorithm by combining ideas from the two implementations where they work well together."
    return "\n\n".join(blocks), instruction


def _render(
    task: Task,
    context: str,
    instruction: str,
    *,
    has_scores: bool = False,
    analysis_instruction: str = ALGORITHM_ANALYSIS,
) -> str:
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
        "analysis_instruction": analysis_instruction,
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
    valid = sorted(
        (p for p in archive.values() if p.valid and p.operator == "Init"),
        key=lambda p: (p.evaluation.fitness, p.id),
    )
    if valid:
        algorithms = [
            f"Approach {k}: {outcome_text(None, p.evaluation.fitness)}\nReported design: {clean_text(p.design)}"
            for k, p in enumerate(valid, 1)
        ]
        context.append("[Existing Approaches]\nFrom best to worst fitness.\n\n" + "\n\n".join(algorithms))
    failed_ids = sorted(p.id for p in archive.values()
                        if p.parent_id == ROOT_ID and p.evaluation is not None and not p.valid)
    failed = [archive[i] for i in failed_ids[-failure_count:]] if failure_count else []
    if failed:
        context.append("[Failed Attempts]\n" + "\n\n".join(
            f"Failed attempt {k}\nReported design: {clean_text(p.design)}\n"
            f"```python\n{p.code.rstrip()}\n```\nFailure: {failure_feedback(p.evaluation.failure, p.code)}"
            for k, p in enumerate(failed, 1)
        ))
    if valid:
        instruction = (
            "Develop an effective algorithm with a different main computation from the ideas already represented. "
            "Seek a different way to compute the result, rather than another weighting of the same local features. "
            "You can change the representation, the information used, or the computational procedure. "
            "Implement the chosen approach as a complete algorithm."
        )
    else:
        instruction = (
            "Design an effective algorithm for this task. Consider how the returned result affects the complete solution "
            "and choose a computational approach suited to that role."
        )
        if 0 in archive:
            instruction += " The example illustrates an implementation, not a structure to preserve."
        instruction += " Implement the specified function."
    if failed:
        instruction += "\nUse the failed attempts and evaluation feedback to inform your implementation."
    return _render(task, "\n\n".join(context), instruction, has_scores=bool(valid))


def repair_format_prompt(task: Task, response: str, feedback: str) -> str:
    context = f"[Original Response]\n{response}\n\n[Parsing Feedback]\n{feedback}"
    instruction = (
        "Convert the original response into the JSON object specified in [Output], using the "
        "parsing feedback to correct its structure, required fields, and string escaping.\n"
        "Preserve the existing algorithmic idea and computational logic.\n"
        "Complete any missing or truncated content consistently with that approach and the "
        "task requirements."
    )
    return _render(
        task, context, instruction,
        analysis_instruction="Keep the original rationale if present. Otherwise briefly state the intended computation.",
    )


def repair_program_prompt(task: Task, program: Program) -> str:
    failure = program.evaluation.failure
    feedback = (
        f"The evaluation did not finish within the total time limit of {task.time_limit:g} seconds."
        if failure.startswith(TIMEOUT_PREFIX) else failure_feedback(failure, program.code)
    )
    context = (
        f"[Current Algorithm]\nReported design: {clean_text(program.design)}\n"
        f"```python\n{program.code.rstrip()}\n```\n\n[Failure Feedback]\n{feedback}"
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
    return _render(task, context, instruction, analysis_instruction="Explain the correction for the reported failure.")
