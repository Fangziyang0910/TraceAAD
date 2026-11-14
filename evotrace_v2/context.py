"""为算法生成提供当前实现、相关发展过程与已有经验。"""

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

from .config import Config
from .models import Edge, Node, Operator, Program, TreePopulation
from .task import Task
from .reflection import (
    TIMEOUT_PREFIX, _bare_code, _outcome_word, clean_prompt_text, compact_diff,
    compact_failure, failure_feedback,
)
from .topology import code_distance, identity


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
    """上下文呈现边界；这些限制控制模型输入，不限制树的生长、选择或简化。"""

    max_input_tokens: int = 30000
    # Optional rendering windows, used only when fitting an oversized request.
    max_path_steps: int | None = None
    max_unconsolidated_attempts: int | None = None
    max_unconsolidated_failures: int | None = None
    max_directions: int | None = None
    max_experiences: int | None = None
    include_reference_code: bool = True
    include_experiences: bool = True  # 设为 False 即为“留空两种经验”的消融对照

    def __post_init__(self):
        if type(self.max_input_tokens) is not int or self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be a positive integer")
        for name in ("max_path_steps", "max_unconsolidated_attempts", "max_unconsolidated_failures",
                     "max_directions", "max_experiences"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be None or a nonnegative integer")


def _path_from_root(tree: TreePopulation, node_id: int) -> list[int]:
    """返回从虚拟根之后到 node_id 的节点编号列表 [root_child, ..., node_id]。"""
    path: list[int] = []
    cur = node_id
    while cur != tree.root_id:
        path.append(cur)
        cur = tree.edges[cur].parent_id
    path.reverse()
    return path


def _lowest_common_ancestor(tree: TreePopulation, u_id: int, v_id: int) -> tuple[int, list[int], list[int]]:
    """返回 (lca_id, path_after_lca_to_u, path_after_lca_to_v)，lca_id 可为 root_id(0)。"""
    path_u = [tree.root_id, *_path_from_root(tree, u_id)]
    path_v = [tree.root_id, *_path_from_root(tree, v_id)]
    k = 0
    limit = min(len(path_u), len(path_v))
    while k < limit and path_u[k] == path_v[k]:
        k += 1
    lca = path_u[k - 1]
    return lca, path_u[k:], path_v[k:]


def _route_path(tree: TreePopulation, node_id: int) -> list[int]:
    """Current route only; the incoming entry edge retains the shift's evidence."""
    route = tree.direction_of(node_id)
    path = [node_id]
    parent = tree.edges[node_id].parent_id
    while parent != tree.root_id and tree.direction_of(parent) == route:
        path.append(parent)
        parent = tree.edges[parent].parent_id
    return list(reversed(path))


def _origin_summary(tree, entry_id, expanded_ids, include_experiences):
    parent_id = tree.edges[entry_id].parent_id
    if parent_id == tree.root_id or parent_id in expanded_ids:
        return ""
    parent = tree.nodes[parent_id]
    lines = [f"Route entry {entry_id} originated from Algorithm {parent_id}: "
             f"{clean_prompt_text(parent.design)}",
             "This is the source of an Explore shift, not another local step in the current route."]
    if include_experiences and parent.selection_experience:
        lines.append(f"Experiments at that source: {clean_prompt_text(parent.selection_experience)}")
    return "[Route Origin]\n" + "\n".join(lines)


def _operator_label(tree: TreePopulation, edge: Edge) -> str:
    """生成干净的算子标签：仅当算子本身是 Crossover 时标记，且仅当参考节点仍在树上时展示参考编号。"""
    op_str = edge.operator or "Init"
    if op_str == "Crossover" and edge.reference_id is not None and edge.reference_id in tree.nodes:
        op_str += f" with reference Algorithm {edge.reference_id}"
    return op_str


def _step_card(
    tree: TreePopulation,
    parent_id: int,
    child_id: int,
    label: str,
    *,
    include_experiences: bool = True,
) -> str:
    """渲染树上的一条边 parent_id -> child_id。

    - 单步未收缩边：展示分数变化与源码 diff；
    - 收缩后的阶段边（``edge.contracted`` 或已有 ``variation_experience``）：固定展示端点分数变化与后代 Design，
      仅在 ``include_experiences=True`` 时附加变异经验，使经验开关构成纯净的增量消融。
    """
    edge = tree.edges[child_id]
    child = tree.nodes[child_id]
    header = f"{label} ({_operator_label(tree, edge)})"
    child_design = clean_prompt_text(child.design)

    if parent_id == tree.root_id:
        lines = [f"{header}: fitness = {child.fitness:.6g}", f"Design: {child_design}"]
        if include_experiences and edge.variation_experience:
            lines.append(f"Stage experience: {clean_prompt_text(edge.variation_experience)}")
        return "\n".join(lines)

    parent = tree.nodes[parent_id]
    outcome = _outcome_word(parent.fitness, child.fitness)
    if edge.contracted or bool(edge.variation_experience):
        lines = [f"{header}: {outcome}", f"Resulting design: {child_design}"]
        if include_experiences and edge.variation_experience:
            lines.append(f"Stage experience: {clean_prompt_text(edge.variation_experience)}")
        return "\n".join(lines)

    diff_text = compact_diff(parent.code, child.code)
    return f"{header}: {outcome}\nResulting design: {child_design}\nCode change:\n```diff\n{diff_text}\n```"


def _render_subpath(
    tree: TreePopulation,
    start_id: int,
    step_nodes: list[int],
    heading: str,
    *,
    max_steps: int | None,
    include_experiences: bool,
    selection_exclude: int | None = None,
) -> tuple[str, list[int]]:
    """渲染从 start_id 沿 step_nodes 前进的一段路径。

    默认展示整条保留路径。只有请求装不进输入窗口时才缩短到指定步数，
    此时保留当前入口、近期阶段及可解释差分的实际基准。
    """
    if not step_nodes or (max_steps is not None and max_steps <= 0):
        return "", []
    if max_steps is None:
        max_steps = len(step_nodes)

    full_seq = [start_id, *step_nodes]
    all_step_indices = list(range(1, len(full_seq)))
    rows: list[str] = []

    if start_id != tree.root_id:
        selected_indices = all_step_indices[-max_steps:]
        first_step_idx = 1
    else:
        first_node = tree.nodes[full_seq[1]]
        first_edge = tree.edges[full_seq[1]]
        init_line = (
            f"Retained Path Entry {first_node.id}: fitness = {first_node.fitness:.6g}\n"
            f"Design: {clean_prompt_text(first_node.design)}"
        )
        if include_experiences and first_edge.variation_experience:
            init_line += f"\nStage experience: {clean_prompt_text(first_edge.variation_experience)}"
        if include_experiences and first_node.id != selection_exclude and first_node.selection_experience:
            init_line += f"\nExperiments from this stage: {clean_prompt_text(first_node.selection_experience)}"
        rows.append(init_line)
        tail_steps = all_step_indices[1:]
        keep_tail = max(0, max_steps - 1)
        selected_indices = tail_steps[-keep_tail:] if keep_tail > 0 else []
        first_step_idx = 2

    if selected_indices and selected_indices[0] > first_step_idx:
        omitted = selected_indices[0] - first_step_idx
        rows.append(f"... {omitted} earlier step(s) omitted ...")

    if selected_indices:
        baseline = tree.nodes[full_seq[selected_indices[0] - 1]]
        rows.append(f"Baseline for the following changes (fitness = {baseline.fitness:.6g}):\n"
                    f"```python\n{_bare_code(baseline.code)}\n```")

    shown_ids: list[int] = []
    for idx in selected_indices:
        p_id, c_id = full_seq[idx - 1], full_seq[idx]
        prior_edge = tree.edges[p_id]
        current_edge = tree.edges[c_id]
        if idx != selected_indices[0] and (prior_edge.contracted or prior_edge.variation_experience) and not (
            current_edge.contracted or current_edge.variation_experience
        ):
            rows.append(f"Implementation after the summarized stage:\n```python\n{_bare_code(tree.nodes[p_id].code)}\n```")
        disp_num = idx - 1 if start_id == tree.root_id else idx
        rows.append(_step_card(tree, p_id, c_id, f"Step {disp_num}", include_experiences=include_experiences))
        node = tree.nodes[c_id]
        if include_experiences and c_id != selection_exclude and node.selection_experience:
            rows.append(f"Experiments from this stage: {clean_prompt_text(node.selection_experience)}")
        shown_ids.append(c_id)

    body = "\n\n".join(rows)
    return f"[{heading}]\n{body}", shown_ids


def _direct_attempts_block(
    tree: TreePopulation,
    parent_id: int,
    heading: str,
    *,
    config: ContextConfig,
) -> tuple[str, dict[str, Any]]:
    """Read retained outgoing developments and the experience already attached here."""
    parent = tree.nodes[parent_id]
    parts: list[str] = []

    if config.include_experiences and parent.selection_experience:
        parts.append(f"Experience from earlier experiments: {clean_prompt_text(parent.selection_experience)}")

    direct_kids = [
        tree.nodes[cid]
        for cid, edge in sorted(tree.edges.items())
        if edge.parent_id == parent_id
    ]
    # 按标准化源码与状态去重，保留最新一条
    unique: dict[tuple[Any, ...], Node] = {}
    for child in direct_kids:
        key = (
            tree.edges[child.id].operator,
            tree.edges[child.id].reference_id,
            identity(child.code),
            child.fitness is not None,
            child.fitness,
            compact_failure(child.failure)[:40] if child.fitness is None else "",
        )
        unique[key] = child

    valid_kids = sorted(
        (c for c in unique.values() if c.fitness is not None),
        key=lambda c: -c.id,
    )[: config.max_unconsolidated_attempts]
    failed_kids = sorted(
        (c for c in unique.values() if c.fitness is None),
        key=lambda c: -c.id,
    )[: config.max_unconsolidated_failures]
    shown = sorted([*valid_kids, *failed_kids], key=lambda c: c.id)

    if shown:
        cards: list[str] = []
        for idx, child in enumerate(shown, 1):
            edge = tree.edges[child.id]
            op_str = _operator_label(tree, edge)
            child_design = clean_prompt_text(child.design)
            if child.fitness is None:
                cards.append(
                    f"Active/Recent Child {idx} ({op_str}): failed — {compact_failure(child.failure)}\n"
                    f"Attempted design: {child_design}\n"
                    f"Code change:\n```diff\n{compact_diff(parent.code, child.code)}\n```"
                )
            else:
                cards.append(_step_card(
                    tree, parent_id, child.id, f"Active/Recent Child {idx}",
                    include_experiences=config.include_experiences,
                ))
        parts.append("\n\n".join(cards))

    meta = {
        "has_selection_experience": bool(config.include_experiences and parent.selection_experience),
        "shown_valid_children": [c.id for c in valid_kids],
        "shown_failed_children": [c.id for c in failed_kids],
    }
    if not parts:
        return "", meta
    return f"[{heading}]\n" + "\n\n".join(parts), meta


def _directions_overview_block(
    tree: TreePopulation,
    *,
    config: ContextConfig,
) -> tuple[str, list[dict[str, Any]]]:
    """展示不同方向的入口思想与经验；结构筛选和累计计数保留在元数据中。"""
    children: dict[int, list[int]] = defaultdict(list)
    for cid, edge in sorted(tree.edges.items()):
        if tree.nodes[cid].fitness is not None:
            children[edge.parent_id].append(cid)

    dir_roots = [
        nid
        for nid in sorted(tree.nodes)
        if tree.nodes[nid].fitness is not None
        and (tree.edges[nid].parent_id == tree.root_id or tree.edges[nid].operator == "Explore")
    ]

    records = []
    for r_id in dir_roots:
        stack = [c for c in children.get(r_id, ()) if tree.edges[c].operator != "Explore"]
        members = [r_id]
        while stack:
            cur = stack.pop()
            members.append(cur)
            stack.extend(c for c in children.get(cur, ()) if tree.edges[c].operator != "Explore")
        best_id = min(members, key=lambda i: (tree.nodes[i].fitness, i))  # type: ignore[arg-type]
        latest_activity = max(members)
        total_evals = tree.direction_evaluations[tree.direction_of(r_id)]
        records.append((r_id, members, best_id, latest_activity, total_evals))

    # 先按最近活跃度排序，并用标准化源码 + AST 结构距离（>= 0.12）筛选异构方向代表
    records.sort(key=lambda item: -item[3])
    distinct_recs: list[tuple[int, list[int], int, int, int]] = []
    seen_identities: set[str] = set()
    for rec in records:
        r_code = tree.nodes[rec[0]].code
        ident = identity(r_code)
        if ident in seen_identities:
            continue
        seen_identities.add(ident)
        if any(code_distance(r_code, tree.nodes[prev[0]].code) < 0.12 for prev in distinct_recs):
            continue
        distinct_recs.append(rec)

    chosen = sorted(distinct_recs[: config.max_directions], key=lambda item: item[0])

    rows: list[str] = []
    if config.include_experiences and tree.root_selection_experience:
        rows.append(f"Experience from initial directions: {clean_prompt_text(tree.root_selection_experience)}")

    view: list[dict[str, Any]] = []
    for idx, (r_id, members, best_id, latest, total_evals) in enumerate(chosen, 1):
        r_node = tree.nodes[r_id]
        r_edge = tree.edges[r_id]
        direction_id = tree.direction_of(r_id)
        lines = [
            f"Approach {idx}: {clean_prompt_text(r_node.design)}",
        ]
        if latest != r_id:
            lines.append(f"Recent development: {clean_prompt_text(tree.nodes[latest].design)}")
        if config.include_experiences:
            if r_edge.variation_experience:
                lines.append(f"Development experience: {clean_prompt_text(r_edge.variation_experience)}")
            experiences: list[str] = []
            seen_experiences: set[str] = set()
            for m in sorted(members):
                sel_exp = clean_prompt_text(tree.nodes[m].selection_experience)
                if sel_exp and sel_exp not in seen_experiences:
                    seen_experiences.add(sel_exp)
                    source = "the entry approach" if m == r_id else (
                        "the recent development" if m == latest else f"retained variant {m}"
                    )
                    experiences.append(f"Experiments around {source}: {sel_exp}")
                edge = tree.edges[m]
                if m != r_id and edge.variation_experience:
                    note = clean_prompt_text(edge.variation_experience)
                    if note not in seen_experiences:
                        seen_experiences.add(note)
                        experiences.append(f"Development toward retained variant {m}: {note}")
            if experiences:
                if config.max_experiences is not None:
                    experiences = experiences[-config.max_experiences:] if config.max_experiences else []
                if experiences:
                    lines.append("Experience from related experiments:\n" + "\n\n".join(experiences))
        rows.append("\n\n".join(lines))
        view.append({
            "origin": direction_id,
            "entry": r_id,
            "best": best_id,
            "retained_members": len(members),
            "shown_experiences": len(experiences) if config.include_experiences else 0,
            "evaluations": total_evals,
        })

    if not rows:
        return "", []
    return "[Algorithmic Ideas and Experience]\n" + "\n\n".join(rows), view


def _current_algorithm_card(node: Node, *, show_score: bool) -> str:
    lines = [f"Algorithm {node.id}"]
    if show_score and node.fitness is not None:
        lines.append(f"Score: {node.fitness:.6g}")
    lines.append(f"```python\n{_bare_code(node.code)}\n```")
    return "\n".join(lines)


def build_evolution_context(
    tree: TreePopulation,
    parent_id: int,
    operator: Operator,
    reference_id: int | None = None,
    config: ContextConfig = ContextConfig(),
) -> tuple[str, str, bool, dict[str, Any]]:
    """按算子职责抽取路径级（Refine）、分支级（Crossover）或全树级（Explore）上下文。

    返回 ``(rendered_context, instruction, has_scores, metadata_view)``。
    """
    parent = tree.nodes[parent_id]
    blocks: list[str] = []
    if operator != "Explore":
        blocks.append("[Current Algorithm]\n" + _current_algorithm_card(parent, show_score=True))
    view: dict[str, Any] = {"operator": operator, "parent_id": parent_id}

    if operator == "Refine":
        path = _route_path(tree, parent_id)
        origin = _origin_summary(tree, path[0], set(path), config.include_experiences)
        if origin:
            blocks.append(origin)
        path_text, shown_steps = _render_subpath(
            tree,
            tree.root_id,
            path,
            "Evolution Path",
            max_steps=config.max_path_steps,
            include_experiences=config.include_experiences,
            selection_exclude=parent_id,
        )
        if path_text:
            blocks.append(path_text)
        view.update({"path": path, "shown_steps": shown_steps,
                     "route_entry": path[0], "origin_id": tree.edges[path[0]].parent_id})
        instruction = (
            "Improve the current algorithm by examining how it makes decisions and where its computation can be improved. "
            "Use the implementation, outcomes, and development experience to choose a meaningful modification. "
            "You may revise its representation, decision logic, or search procedure as well as its parameters; "
            "implement a coherent improved algorithm."
        )

    elif operator == "Crossover":
        if reference_id is None or reference_id not in tree.nodes:
            raise ValueError("Crossover requires a valid retained reference_id")
        ref_node = tree.nodes[reference_id]
        lca_id, branch_to_parent, branch_to_ref = _lowest_common_ancestor(tree, parent_id, reference_id)
        same_route = tree.direction_of(parent_id) == tree.direction_of(reference_id)
        branch_start = lca_id
        if not same_route:
            branch_to_parent = _route_path(tree, parent_id)
            branch_to_ref = _route_path(tree, reference_id)
            branch_start = tree.root_id
            expanded = set(branch_to_parent + branch_to_ref)
            if lca_id != tree.root_id:
                blocks.append(f"[Shared Origin]\nAlgorithm {lca_id} is a historical source; "
                              "their ancestry crosses an Explore boundary. Each route's local development is shown separately.")
            for entry in (branch_to_parent[0], branch_to_ref[0]):
                origin = _origin_summary(tree, entry, expanded, config.include_experiences)
                if origin:
                    blocks.append(origin)
                    expanded.add(tree.edges[entry].parent_id)

        if same_route and lca_id != tree.root_id:
            lca_node = tree.nodes[lca_id]
            lca_lines = [
                f"[Common Ancestor]\nAlgorithm {lca_id}: fitness = {lca_node.fitness:.6g}",
                f"Design: {clean_prompt_text(lca_node.design)}",
            ]
            if config.include_experiences and lca_node.selection_experience:
                lca_lines.append(
                    f"Branch experience at ancestor: {clean_prompt_text(lca_node.selection_experience)}"
                )
            blocks.append("\n".join(lca_lines))
        elif lca_id == tree.root_id and config.include_experiences and tree.root_selection_experience:
            blocks.append(
                f"[Origin Context]\nExperience from initial directions: {clean_prompt_text(tree.root_selection_experience)}"
            )

        cur_branch_text, shown_cur = _render_subpath(
            tree,
            branch_start,
            branch_to_parent,
            "Current Branch since Common Ancestor" if same_route else "Current Route Development",
            max_steps=config.max_path_steps,
            include_experiences=config.include_experiences,
            selection_exclude=parent_id,
        )
        if cur_branch_text:
            blocks.append(cur_branch_text)

        ref_branch_text, shown_ref = _render_subpath(
            tree,
            branch_start,
            branch_to_ref,
            "Reference Branch since Common Ancestor" if same_route else "Reference Route Development",
            max_steps=config.max_path_steps,
            include_experiences=config.include_experiences,
            selection_exclude=parent_id,
        )
        if ref_branch_text:
            blocks.append(ref_branch_text)

        ref_lines = [
            f"Algorithm {ref_node.id}",
            f"Score: {ref_node.fitness:.6g}",
            f"Design: {clean_prompt_text(ref_node.design)}",
        ]
        if config.include_reference_code:
            ref_lines.append(f"```python\n{_bare_code(ref_node.code)}\n```")
        blocks.insert(1, "[Reference Algorithm]\n" + "\n".join(ref_lines))

        view.update({
            "reference_id": reference_id,
            "lca_id": lca_id,
            "same_route": same_route,
            "current_route_entry": _route_path(tree, parent_id)[0],
            "reference_route_entry": _route_path(tree, reference_id)[0],
            "shown_current_branch": shown_cur,
            "shown_reference_branch": shown_ref,
            "include_reference_code": config.include_reference_code,
        })
        instruction = (
            "Understand the computational ideas in both implementations and develop an algorithm that combines their useful strengths. "
            "Use their development and experimental experience to reason about how the computations fit together. "
            "The integration may reorganize the current algorithm rather than simply add another scoring term. "
            "If no useful combination emerges, improve the current algorithm using what the comparison reveals."
        )

    elif operator == "Explore":
        dir_text, dir_view = _directions_overview_block(tree, config=config)
        if dir_text:
            blocks.append(dir_text)
        view["directions"] = dir_view
        instruction = (
            "Rethink how this task can be solved and implement an alternative whose main computation differs from the existing approaches. "
            "Develop a new way to compute the decision, rather than another weighting of the same local features. "
            "The existing implementation and experience are starting points for ideas, not a structure you must preserve. "
            "Consider changes in the information used, the representation, or the procedure that produces the decision. "
            "For example, an algorithm can estimate downstream consequences, construct and improve a tentative solution, "
            "or solve useful subproblems; these are possible inspirations, not required methods. "
            "Choose an approach suited to the task and implement its main computation."
        )

    else:
        raise ValueError(f"Unsupported evolution operator: {operator!r}")

    if operator in ("Refine", "Crossover"):
        att_text, att_meta = _direct_attempts_block(
            tree, parent_id, "Attempts from Current Algorithm", config=config,
        )
        if att_text:
            blocks.append(att_text)
        view.update(att_meta)

    has_scores = operator != "Explore" or (
        config.include_experiences and (
            bool(tree.root_selection_experience)
            or any(
                tree.edges[d["entry"]].variation_experience
                or any(tree.nodes[m].selection_experience or tree.edges[m].variation_experience
                       for m in tree.nodes if tree.direction_of(m) == d["origin"])
                for d in view.get("directions", [])
            )
        )
    )
    if operator == "Explore":
        blocks.append("[One Existing Implementation]\n" + _current_algorithm_card(parent, show_score=False))
    return "\n\n".join(blocks), instruction, has_scores, view


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
                f"Algorithm {p.id}\n"
                f"Design: {clean_prompt_text(p.design)}\nScore: {p.evaluation.fitness}"
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
            'Seek a different way to compute the decision, rather than another weighting of the same local features. '
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
        count_tokens=None,
    ) -> tuple[str, dict[str, Any]]:
        # Real serving-tokenizer counts when supplied; UTF-8 bytes are a conservative
        # fallback for standalone clients, explicitly identified in the run record.
        counter = count_tokens or (lambda text: len(text.encode("utf-8")))
        cfg = context_config
        reductions = []
        while True:
            text, instruction, has_scores, view = build_evolution_context(
                tree, parent_id, operator, reference_id, cfg,
            )
            prompt = self._render(text, instruction, has_scores=has_scores)
            size = counter(prompt)
            if size <= cfg.max_input_tokens:
                view["input_window"] = {"size": size, "limit": cfg.max_input_tokens,
                                        "counter": "serving_tokenizer" if count_tokens else "utf8_bytes",
                                        "reductions": reductions}
                return prompt, view
            windows = {
                "max_path_steps": max(len(view.get("shown_steps", [])),
                                      len(view.get("shown_current_branch", [])),
                                      len(view.get("shown_reference_branch", []))),
                "max_unconsolidated_attempts": len(view.get("shown_valid_children", [])),
                "max_unconsolidated_failures": len(view.get("shown_failed_children", [])),
                "max_directions": len(view.get("directions", [])),
                "max_experiences": max((d["shown_experiences"] for d in view.get("directions", [])), default=0),
            }
            trials = []
            for field, count in windows.items():
                if count <= (1 if field in {"max_path_steps", "max_directions"} else 0):
                    continue
                reduced = max(1, count // 2) if field in {"max_path_steps", "max_directions"} else count // 2
                candidate = replace(cfg, **{field: reduced})
                body, inst, scores, _ = build_evolution_context(tree, parent_id, operator, reference_id, candidate)
                candidate_size = counter(self._render(body, inst, has_scores=scores))
                trials.append((candidate_size, field, reduced, candidate))
            if not trials or min(t[0] for t in trials) >= size:
                raise ValueError("Required algorithm context exceeds the configured input window")
            _, field, reduced, cfg = min(trials, key=lambda t: (t[0], t[1]))
            reductions.append({"field": field, "retained": reduced})
