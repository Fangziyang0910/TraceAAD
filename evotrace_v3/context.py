"""按被修改算法的演化路径（Refine、Crossover）、参考算法卡片（Crossover）与代表个体卡片（Explore）组织生成上下文。"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .config import Config
from .models import Edge, Node, Operator, Program, TreePopulation
from .task import Task
from .reflection import (
    HISTORY_NOTE, TIMEOUT_PREFIX, _bare_code, _operator_label, _outcome_word, clean_prompt_text, compact_failure,
    failure_feedback,
)


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
- "design": Give another algorithm designer a concise, factual description of what the code computes and how it produces the result. Describe the implementation, not expected performance.
Preserve the required function signature and satisfy the task constraints and input/output requirements.
Return only the JSON object, without Markdown fences or additional text."""


@dataclass(frozen=True)
class ContextConfig:
    """生成上下文的消融开关；只控制模型输入，不改变树结构或抽样权重。"""

    include_experiences: bool = True  # 设为 False 即为隐藏节点与边经验的消融对照


def _path_from_root(tree: TreePopulation, node_id: int) -> list[int]:
    """返回从虚拟根之后到 node_id 的节点编号列表 [root_child, ..., node_id]。"""
    path: list[int] = []
    cur = node_id
    while cur != tree.root_id:
        path.append(cur)
        cur = tree.edges[cur].parent_id
    path.reverse()
    return path


def _experiences(tree: TreePopulation, node_id: int, *, config: ContextConfig, node: bool = True) -> list[str]:
    """附着在一个个体上的经验：入边上的阶段经验，以及从它出发的尝试凝练成的节点经验。"""
    if not config.include_experiences:
        return []
    lines = []
    if tree.edges[node_id].experience:
        lines.append(f"Stage experience: {clean_prompt_text(tree.edges[node_id].experience)}")
    if node and tree.nodes[node_id].experience:
        lines.append(f"Experience from attempts at this algorithm: {clean_prompt_text(tree.nodes[node_id].experience)}")
    return lines


def _lineage_blocks(
    tree: TreePopulation, parent_id: int, *, config: ContextConfig, exclude: frozenset[int] = frozenset(),
) -> tuple[list[str], list[int], list[int]]:
    """被修改算法的形成轨迹与从它出发的尝试；每条记录依次给出结果、design 与经验，不含代码。"""
    parent = tree.nodes[parent_id]
    path = _path_from_root(tree, parent_id)
    rows = ["[Evolution Path]\nHow the current algorithm was formed, from its earliest retained ancestor to the "
            "current algorithm. " + HISTORY_NOTE]
    for node_id in path:
        node, edge = tree.nodes[node_id], tree.edges[node_id]
        if edge.parent_id == tree.root_id:
            origin = "compressed development from the initial population" if edge.contracted else "initial population"
            head = f"Algorithm {node_id} ({origin}): fitness = {node.fitness:.6g}"
        else:
            head = (f"Algorithm {node_id} <- Algorithm {edge.parent_id} ({_operator_label(tree, edge)}): "
                    f"{_outcome_word(tree.nodes[edge.parent_id].fitness, node.fitness)}")
        # 当前算法的节点经验来自从它出发的尝试，放在尝试段开头。
        rows.append("\n".join([head, f"Design: {clean_prompt_text(node.design)}",
                               *_experiences(tree, node_id, config=config, node=node_id != parent_id)]))
    blocks = ["\n\n".join(rows)]

    shown = [tree.nodes[cid] for cid, edge in sorted(tree.edges.items())
             if edge.parent_id == parent_id and cid not in exclude]
    parts = []
    if config.include_experiences and parent.experience:
        parts.append(f"Experience from earlier attempts at the current algorithm: {clean_prompt_text(parent.experience)}")
    for idx, child in enumerate(shown, 1):
        result = (f"failed - {compact_failure(child.failure)}" if child.fitness is None
                  else _outcome_word(parent.fitness, child.fitness))
        parts.append("\n".join([f"Attempt {idx} ({_operator_label(tree, tree.edges[child.id])}): {result}",
                                f"Design: {clean_prompt_text(child.design)}",
                                *_experiences(tree, child.id, config=config)]))
    if parts:
        blocks.append("\n\n".join(["[Attempts from Current Algorithm]\nEvaluated attempts generated from the "
                                   "current algorithm, in order of creation.", *parts]))
    return blocks, path, [c.id for c in shown]


def _card(tree: TreePopulation, node_id: int, seen: set[tuple[str, int]], *, config: ContextConfig,
          code: bool = False) -> str:
    """一个提供想法的算法：分数、design、其发展路径上尚未出现过的经验，以及按需给出的代码。"""
    node = tree.nodes[node_id]
    lines = [f"Algorithm {node_id}: fitness = {node.fitness:.6g}", f"Design: {clean_prompt_text(node.design)}"]
    if config.include_experiences:
        experience = []
        for j in _path_from_root(tree, node_id):
            if tree.edges[j].experience and ("edge", j) not in seen:
                experience.append(f"- Stage experience leading to Algorithm {j}: "
                                  f"{clean_prompt_text(tree.edges[j].experience)}")
            if tree.nodes[j].experience and ("node", j) not in seen:
                experience.append(f"- Experience from attempts at Algorithm {j}: "
                                  f"{clean_prompt_text(tree.nodes[j].experience)}")
            seen.update({("edge", j), ("node", j)})
        if experience:
            lines += ["Experience along its development:", *experience]
    if code:
        lines.append(f"```python\n{_bare_code(node.code)}\n```")
    return "\n".join(lines)


def _current_algorithm_card(node: Node) -> str:
    return f"Algorithm {node.id}: fitness = {node.fitness:.6g}\n```python\n{_bare_code(node.code)}\n```"


def build_evolution_context(
    tree: TreePopulation,
    parent_id: int,
    operator: Operator,
    reference_id: int | None = None,
    config: ContextConfig = ContextConfig(),
    explore_ids: Sequence[int] = (),
) -> tuple[str, str, bool, dict[str, Any]]:
    """上下文 = 相关经验 + 被修改算法的形成轨迹 + 提供想法的算法卡片；代码只给被重写或被移植的算法。

    顺序为从全局到局部、从过去到现在，需要直接修改的当前算法代码放在最后。
    """
    parent = tree.nodes[parent_id]
    blocks: list[str] = []
    view: dict[str, Any] = {"operator": operator, "parent_id": parent_id}
    if config.include_experiences and tree.root_experience:
        blocks.append("[Initial Population Experience]\nExperience from evaluated initial implementations that are "
                      "no longer in the population.\n" + clean_prompt_text(tree.root_experience))

    if operator == "Refine":
        lineage, path, shown = _lineage_blocks(tree, parent_id, config=config)
        blocks += [*lineage, "[Current Algorithm]\n" + _current_algorithm_card(parent)]
        view.update({"path": path, "attempts": shown})
        instruction = (
            "Improve the current algorithm by examining how it makes decisions and where its computation can be improved. "
            "Use the implementation, outcomes, and development experience to choose a meaningful modification. "
            "You may revise its representation, decision logic, or search procedure as well as its parameters; "
            "implement a coherent improved algorithm."
        )

    elif operator == "Crossover":
        if reference_id is None or reference_id not in tree.nodes:
            raise ValueError("Crossover requires a valid retained reference_id")
        lineage, path, shown = _lineage_blocks(tree, parent_id, config=config, exclude=frozenset({reference_id}))
        seen = {(kind, j) for j in (*path, *shown) for kind in ("edge", "node")}
        reference = _card(tree, reference_id, seen, config=config, code=True)
        blocks += [*lineage,
                   "[Reference Algorithm]\nAn algorithm whose ideas may be combined with the current algorithm. "
                   "Its experience lists only experience along its development that has not appeared above.\n" + reference,
                   "[Current Algorithm]\n" + _current_algorithm_card(parent)]
        view.update({"reference_id": reference_id, "path": path, "attempts": shown})
        instruction = (
            "Understand the computational ideas in both implementations and develop an algorithm that combines their useful strengths. "
            "Use their development and experimental experience to reason about how the computations fit together. "
            "If no useful combination emerges, improve the current algorithm using what the comparison reveals."
        )

    elif operator == "Explore":
        seen: set[tuple[str, int]] = set()
        shown = sorted({parent_id, *explore_ids}, key=lambda i: (tree.nodes[i].fitness, i))
        cards = [_card(tree, i, seen, config=config) for i in shown]
        blocks.append("\n\n".join([
            "[Representative Algorithms]\nRetained algorithms with structurally different implementations, including "
            "the best one, sorted by fitness. Each lists the experience along its development that has not appeared "
            "above. " + HISTORY_NOTE, *cards]))
        view.update({"explore_ids": list(explore_ids), "shown": shown})
        instruction = (
            "Rethink how this task can be solved and implement an alternative whose main computation differs from the existing approaches. "
            "The algorithms and experience above are starting points for ideas, not a structure you must preserve. "
            "Consider changes in the information used, the representation, or the procedure that produces the decision. "
            "Choose an approach suited to the task and implement its main computation."
        )

    else:
        raise ValueError(f"Unsupported evolution operator: {operator!r}")

    return "\n\n".join(blocks), instruction, True, view


class EvoTracePromptBuilder:
    """Build initialization, repair and tree-context prompts without copying programs."""

    def __init__(self, task: Task, config: Config):
        self.task = task
        self.config = config

    def _render(self, context: str, instruction: str, *, has_scores=False) -> str:
        template = COMMON_TEMPLATE
        if not context:
            template = template.replace("[Context]\n{context}\n\n", "")
        elif has_scores and self.task.score_definition not in self.task.task_description:
            context = f"Score definition: {self.task.score_definition}\n\n{context}"
        return template.format_map({
            "task_description": self.task.task_description,
            "function_template": self.task.function_template,
            "context": context,
            "instruction": instruction,
        })

    def initialization(self, archive: dict[int, Program]) -> str:
        context = []
        if 0 in archive:
            example = archive[0]
            text = "[Example Algorithm]\n"
            if design := clean_prompt_text(example.design):
                text += f"Design: {design}\n"
            context.append(text + f"```python\n{_bare_code(example.code)}\n```")
        valid = sorted((p for p in archive.values() if p.valid and p.operator == "Init"),
                       key=lambda p: p.id)
        if valid:
            algorithms = [
                f"Algorithm {p.id}: fitness = {p.evaluation.fitness:.6g}\n"
                f"Design: {clean_prompt_text(p.design)}"
                for p in valid
            ]
            context.append("[Existing Algorithmic Ideas]\n\n"
                           + "\n\n".join(algorithms))
        failed = sorted(
            sorted((p for p in archive.values() if p.parent_id == 0
                    and p.evaluation is not None and p.evaluation.status != "success"),
                   key=lambda p: -p.id)[:self.config.failure_count],
            key=lambda p: p.id,
        )
        if failed:
            context.append("[Failed Attempts]\n" + "\n\n".join(
                f"Algorithm {p.id}\n"
                f"Design: {clean_prompt_text(p.design)}\n"
                f"```python\n{_bare_code(p.code)}\n```\nFailure: {failure_feedback(p.evaluation.failure)}"
                for p in failed
            ))
        instruction = (
            'Develop an effective algorithm with a different main computation from the ideas already represented. '
            'You can change the representation, the information used to make decisions, or the computational procedure. '
            'Implement the chosen approach as a complete algorithm.'
        ) if valid else (
            'Design an effective algorithm for this task. Consider how the returned decision affects the complete solution '
            'and choose a computational approach suited to that role. The example illustrates an implementation, '
            'not a structure to preserve. Implement the specified function.'
        )
        if failed:
            instruction += "\nUse the failed attempts and evaluation feedback to inform your implementation."
        return self._render("\n\n".join(context), instruction, has_scores=bool(valid))

    def repair_format(self, response: str, feedback: str) -> str:
        context = f"[Original Response]\n{response}\n\n[Parsing Feedback]\n{feedback}"
        return self._render(context, (
            'Convert the original response into the JSON object specified in [Output], using the '
            'parsing feedback to correct its structure, required fields, and string escaping.\n'
            'Preserve the existing algorithmic idea and computational logic.\n'
            'Complete any missing or truncated content consistently with that approach and the '
            'task requirements.'
        ))

    def repair_program(self, program: Program) -> str:
        failure = program.evaluation.failure
        if failure.startswith(TIMEOUT_PREFIX):
            feedback = f"The evaluation did not finish within the total time limit of {self.task.time_limit:g} seconds."
        else:
            feedback = failure_feedback(failure)
        context = (f"[Current Algorithm]\nDesign: {clean_prompt_text(program.design)}\n"
                   f"```python\n{_bare_code(program.code)}\n```\n\n"
                   f"[Failure Feedback]\n{feedback}")
        instruction = ((
            'Trace the reported exception or invalid result to its cause in the code, using the '
            'design to understand the intended algorithm.\n'
            'Correct the faulty computations or data handling so the function executes correctly '
            'and satisfies the task requirements, while retaining the intended core decision '
            'mechanism.'
        ) if program.evaluation.status == "error"
                       else (
                           'Make the algorithm finish within the reported time limit. '
                           'Reduce the cost of its core computation, using an approximation when needed, '
                           'while retaining the intended algorithmic idea.'
                       ))
        return self._render(context, instruction)

    def evolution_from_tree(
        self, tree: TreePopulation, parent_id: int, operator: Operator,
        reference_id: int | None = None, context_config: ContextConfig = ContextConfig(),
        explore_ids: Sequence[int] = (),
    ) -> tuple[str, dict[str, Any]]:
        """完整展示上下文，不设长度裁剪；实际输入长度由调用记录中的 usage 给出。"""
        text, instruction, has_scores, view = build_evolution_context(
            tree, parent_id, operator, reference_id, context_config, explore_ids)
        return self._render(text, instruction, has_scores=has_scores), view
