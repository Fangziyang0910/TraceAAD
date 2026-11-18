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

# 端点上下文为 49152 token，生成预留 16384，输入须低于 32768 token。代码约 3–4 字符/token，
# 探索的上下文以 9 万字符为限，放不下的代表卡片按抽样顺序略去，主父代卡片始终保留。
EXPLORE_CONTEXT_CHARS = 90_000

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
    tree: TreePopulation,
    node_id: int,
    *,
    include: bool,
    node: bool = True,
    seen: set[tuple[str, int]] | frozenset = frozenset(),
) -> list[str]:
    """个体自身的入边经验与节点经验；已在上文出现的经验不重复。"""
    if not include:
        return []
    lines = []
    if tree.edges[node_id].experience and ("edge", node_id) not in seen:
        lines.append(f"Experience from the steps that led to it: {tree.edges[node_id].experience}")
    if node and tree.nodes[node_id].experience and ("node", node_id) not in seen:
        lines.append(f"Experience from attempts made from it: {tree.nodes[node_id].experience}")
    return lines


def _attempts_block(
    tree: TreePopulation,
    parent_id: int,
    *,
    include_experiences: bool,
    exclude: frozenset[int] = frozenset(),
) -> tuple[str | None, list[Node]]:
    parent = tree.nodes[parent_id]
    shown = [
        tree.nodes[cid] for cid, edge in sorted(tree.edges.items())
        if edge.parent_id == parent_id and cid not in exclude
    ]
    parts = []
    if include_experiences and parent.experience:
        parts.append(f"Experience from earlier attempts: {parent.experience}")
    for idx, child in enumerate(shown, 1):
        edge = tree.edges[child.id]
        result = (
            f"failed - {compact_failure(child.failure)}" if child.fitness is None
            else outcome_text(parent.fitness, child.fitness)
        )
        parts.append("\n".join([
            f"Attempt {idx} ({_origin(edge)}): {result}",
            f"Design: {child.design}",
            *_experiences(tree, child.id, include=include_experiences),
        ]))
    if not parts:
        return None, shown
    heading = "[Attempts from the Current Algorithm]" + ("\nIn order of creation." if shown else "")
    return "\n\n".join([heading, *parts]), shown


def _lineage_blocks(
    tree: TreePopulation, parent_id: int, *, include_experiences: bool, exclude: frozenset[int] = frozenset()
) -> tuple[list[str], set[tuple[str, int]]]:
    """形成路径（各步设计、成绩变化与路径上的经验），随后是从主父代出发的尝试及其节点经验。"""
    path = _path_from_root(tree, parent_id)
    rows = ["[Development of the Current Algorithm]"]
    for k, node_id in enumerate(path):
        node, edge = tree.nodes[node_id], tree.edges[node_id]
        prev = tree.nodes.get(edge.parent_id)
        label = "First version" if k == 0 else f"Step {k}"
        if node_id == parent_id:
            label += ", the current algorithm"
        head = f"{label} ({_origin(edge)}): {outcome_text(None if prev is None else prev.fitness, node.fitness)}"
        rows.append("\n".join([
            head,
            f"Design: {node.design}",
            *_experiences(tree, node_id, include=include_experiences, node=node_id != parent_id),
        ]))
    blocks = ["\n\n".join(rows)]
    attempts_text, shown = _attempts_block(
        tree, parent_id, include_experiences=include_experiences, exclude=exclude
    )
    if attempts_text is not None:
        blocks.append(attempts_text)
    seen = {(kind, j) for j in (*path, *(c.id for c in shown)) for kind in ("edge", "node")}
    return blocks, seen


def _card(
    tree: TreePopulation,
    node_id: int,
    *,
    include_experiences: bool,
    seen: set[tuple[str, int]] | frozenset = frozenset(),
) -> list[str]:
    """个体卡片：设计简述、自身的入边与节点经验，源码放在最后。"""
    node = tree.nodes[node_id]
    return [
        f"Design: {node.design}",
        *_experiences(tree, node_id, include=include_experiences, seen=seen),
        f"```python\n{bare_code(node.code)}\n```",
    ]


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
    """按算子返回 (树上下文文本, 指令文本)。

    精化：形成路径上各步的设计、成绩变化与经验，主父代的近期尝试与节点经验，最后是主父代源码。
    交叉：精化的上下文，加参考个体卡片（设计、成绩、自身经验与源码），主父代源码仍在最后。
    探索：主父代卡片与至多 M 个异质代表卡片，各含设计、成绩、自身经验与源码；不含形成路径与尝试列表。
    """
    if parent_id not in tree.nodes or tree.nodes[parent_id].fitness is None:
        raise ValueError("Evolution requires a valid retained feasible parent_id")
    parent = tree.nodes[parent_id]

    if operator == "Explore":
        blocks = ["\n".join([
            "[Current Algorithm]",
            f"fitness = {parent.fitness:.6g}",
            *_card(tree, parent_id, include_experiences=include_experiences),
        ])]
        size = len(blocks[0])
        cards: list[tuple[float, int, str]] = []
        for i in dict.fromkeys(explore_ids):
            if i not in tree.nodes or i == parent_id or tree.nodes[i].fitness is None:
                continue
            body = "\n".join(_card(tree, i, include_experiences=include_experiences))
            if size + len(body) > EXPLORE_CONTEXT_CHARS:
                continue
            size += len(body)
            cards.append((tree.nodes[i].fitness, i, body))  # type: ignore[arg-type]
        if cards:
            cards.sort(key=lambda c: (c[0], c[1]))
            blocks.append("\n\n".join([
                "[Other Existing Approaches]\nFrom best to worst fitness.",
                *(f"Approach {k}: fitness = {fit:.6g}\n{body}" for k, (fit, _, body) in enumerate(cards, 1)),
            ]))
        instruction = (
            "Design an algorithm whose main computation differs from every algorithm shown above, "
            "not another weighting of the same local features. "
            "The code shows what each one actually computes, even where a design is inaccurate, "
            "and the experience shows what has already been tried. "
            "The code is not a structure you must preserve. "
            "A different computation may use other information, another representation, or another procedure. "
            "For example, it can estimate the downstream consequences of a choice, construct and improve a tentative "
            "solution, or solve a useful subproblem. "
            "Implement the computation that suits this task."
        )
        return "\n\n".join(blocks), instruction

    if operator == "Refine":
        lineage, _ = _lineage_blocks(tree, parent_id, include_experiences=include_experiences)
        blocks = [*lineage, _current_code(parent)]
        instruction = (
            "Improve the current algorithm by examining how it computes its result and where this computation can be improved. "
            "The development path shows what each step changed and how fitness responded; "
            "the experience and the attempts show what has already been tried along this path. "
            "Use them with the implementation to locate what limits the result and to choose a meaningful modification. "
            "You may revise its representation, the information it uses, or its procedure as well as its parameters; "
            "implement a coherent improved algorithm."
        )
    elif operator == "Crossover":
        if reference_id is None or reference_id not in tree.nodes:
            raise ValueError("Crossover requires a valid retained reference_id")
        lineage, seen = _lineage_blocks(
            tree, parent_id, include_experiences=include_experiences, exclude=frozenset({reference_id})
        )
        ref = tree.nodes[reference_id]
        blocks = [
            *lineage,
            "\n".join([
                "[Other Algorithm]",
                "An algorithm whose ideas may be combined with the current algorithm.",
                f"fitness = {ref.fitness:.6g}",
                *_card(tree, reference_id, include_experiences=include_experiences, seen=seen),
            ]),
            _current_code(parent),
        ]
        instruction = (
            "Understand the computational ideas in both implementations and develop an algorithm that combines their useful strengths. "
            "The development path shows how the current algorithm was formed and what has been tried along it; "
            "the other algorithm's code shows what it actually computes, and its experience shows what was learned about it. "
            "Use them to decide which of its computations can strengthen the current algorithm and how the two fit together. "
            "If no useful combination emerges, improve the current algorithm using what the comparison reveals."
        )
    else:
        raise ValueError(f"Unsupported evolution operator: {operator!r}")
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
            f"Approach {k}: fitness = {p.evaluation.fitness:.6g}\nDesign: {clean_text(p.design)}"
            for k, p in enumerate(valid, 1)
        ]
        context.append("[Existing Approaches]\nFrom best to worst fitness.\n\n" + "\n\n".join(algorithms))
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
    return _render(task, context, instruction, analysis_instruction="Explain the correction for the reported failure.")
