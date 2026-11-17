"""为算法生成提供当前实现、相关发展过程与已有经验。"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .evaluation import TIMEOUT_PREFIX, compact_failure, failure_feedback
from .models import Edge, Node, Operator, Program, ROOT_ID, TreePopulation, clean_text, outcome_text
from .task import Task
from .topology import bare_code, code_distance, compact_diff, identity

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
    max_path_steps: int = 10
    max_unconsolidated_attempts: int = 4
    max_unconsolidated_failures: int = 2
    max_directions: int = 8
    include_reference_code: bool = True
    include_experiences: bool = True


def _path_from_root(tree: TreePopulation, node_id: int) -> list[int]:
    path: list[int] = []
    cur = node_id
    while cur != ROOT_ID:
        path.append(cur)
        cur = tree.edges[cur].parent_id
    path.reverse()
    return path


def _lowest_common_ancestor(tree: TreePopulation, u_id: int, v_id: int) -> tuple[int, list[int], list[int]]:
    path_u = [ROOT_ID, *_path_from_root(tree, u_id)]
    path_v = [ROOT_ID, *_path_from_root(tree, v_id)]
    k = 0
    while k < min(len(path_u), len(path_v)) and path_u[k] == path_v[k]:
        k += 1
    return path_u[k - 1], path_u[k:], path_v[k:]


def _operator_label(tree: TreePopulation, edge: Edge) -> str:
    op_str = edge.operator or "Init"
    if op_str == "Crossover" and edge.reference_id is not None and edge.reference_id in tree.nodes:
        op_str += f" with reference Algorithm {edge.reference_id}"
    return op_str


def _step_card(
    tree: TreePopulation, parent_id: int, child_id: int, label: str, *, include_experiences: bool = True
) -> str:
    edge, child = tree.edges[child_id], tree.nodes[child_id]
    header = f"{label} ({_operator_label(tree, edge)})"
    if parent_id == ROOT_ID:
        lines = [f"{header}: fitness = {child.fitness:.6g}", f"Design: {child.design}"]
        if include_experiences and edge.variation_experience:
            lines.append(f"Stage experience: {edge.variation_experience}")
        return "\n".join(lines)

    parent = tree.nodes[parent_id]
    outcome = outcome_text(parent.fitness, child.fitness)
    if edge.contracted or edge.variation_experience:
        lines = [f"{header}: {outcome}", f"Resulting design: {child.design}"]
        if include_experiences and edge.variation_experience:
            lines.append(f"Stage experience: {edge.variation_experience}")
        return "\n".join(lines)
    return f"{header}: {outcome}\nResulting design: {child.design}\nCode change:\n```diff\n{compact_diff(parent.code, child.code)}\n```"


def _render_subpath(
    tree: TreePopulation, start_id: int, step_nodes: list[int], heading: str, *,
    max_steps: int, include_experiences: bool,
) -> str:
    if not step_nodes or max_steps <= 0:
        return ""
    full_seq = [start_id, *step_nodes]
    all_indices = list(range(1, len(full_seq)))
    rows: list[str] = []
    if start_id != ROOT_ID:
        selected = all_indices[-max_steps:]
        first_idx = 1
    else:
        first_node, first_edge = tree.nodes[full_seq[1]], tree.edges[full_seq[1]]
        init_line = f"Initial Algorithm {first_node.id}: fitness = {first_node.fitness:.6g}\nDesign: {first_node.design}"
        if include_experiences and first_edge.variation_experience:
            init_line += f"\nStage experience: {first_edge.variation_experience}"
        rows.append(init_line)
        keep_tail = max(0, max_steps - 1)
        selected = all_indices[1:][-keep_tail:] if keep_tail > 0 else []
        first_idx = 2

    if selected and selected[0] > first_idx:
        rows.append(f"... {selected[0] - first_idx} earlier step(s) omitted ...")
    if selected:
        baseline = tree.nodes[full_seq[selected[0] - 1]]
        rows.append(
            f"Baseline for the following changes (fitness = {baseline.fitness:.6g}):\n"
            f"```python\n{bare_code(baseline.code)}\n```"
        )
    for idx in selected:
        p_id, c_id = full_seq[idx - 1], full_seq[idx]
        prior_edge, cur_edge = tree.edges[p_id], tree.edges[c_id]
        if idx != selected[0] and (prior_edge.contracted or prior_edge.variation_experience) and not (
            cur_edge.contracted or cur_edge.variation_experience
        ):
            rows.append(f"Implementation after the summarized stage:\n```python\n{bare_code(tree.nodes[p_id].code)}\n```")
        disp_num = idx - 1 if start_id == ROOT_ID else idx
        rows.append(_step_card(tree, p_id, c_id, f"Step {disp_num}", include_experiences=include_experiences))
    return f"[{heading}]\n" + "\n\n".join(rows)


def _direct_attempts_block(tree: TreePopulation, parent_id: int, heading: str, *, config: ContextConfig) -> str:
    parent = tree.nodes[parent_id]
    parts: list[str] = []
    if config.include_experiences and parent.selection_experience:
        parts.append(f"Experience from earlier experiments: {parent.selection_experience}")

    unique: dict[tuple[Any, ...], Node] = {}
    for cid, edge in sorted(tree.edges.items()):
        if edge.parent_id == parent_id:
            child = tree.nodes[cid]
            key = (
                edge.operator, edge.reference_id, identity(child.code),
                child.fitness is not None, child.fitness,
                compact_failure(child.failure)[:40] if child.fitness is None else "",
            )
            unique[key] = child

    valid_kids = sorted((c for c in unique.values() if c.fitness is not None), key=lambda c: -c.id)[: config.max_unconsolidated_attempts]
    failed_kids = sorted((c for c in unique.values() if c.fitness is None), key=lambda c: -c.id)[: config.max_unconsolidated_failures]
    shown = sorted([*valid_kids, *failed_kids], key=lambda c: c.id)
    if shown:
        cards = []
        for idx, child in enumerate(shown, 1):
            edge = tree.edges[child.id]
            if child.fitness is None:
                cards.append(
                    f"Active/Recent Child {idx} ({_operator_label(tree, edge)}): failed — {compact_failure(child.failure)}\n"
                    f"Attempted design: {child.design}\n"
                    f"Code change:\n```diff\n{compact_diff(parent.code, child.code)}\n```"
                )
            else:
                cards.append(_step_card(
                    tree, parent_id, child.id, f"Active/Recent Child {idx}",
                    include_experiences=config.include_experiences,
                ))
        parts.append("\n\n".join(cards))
    return f"[{heading}]\n" + "\n\n".join(parts) if parts else ""


def _directions_overview_block(tree: TreePopulation, *, config: ContextConfig) -> tuple[str, bool]:
    children: dict[int, list[int]] = defaultdict(list)
    for cid, edge in sorted(tree.edges.items()):
        if tree.nodes[cid].fitness is not None:
            children[edge.parent_id].append(cid)

    dir_roots = [
        nid for nid in sorted(tree.nodes)
        if tree.nodes[nid].fitness is not None
        and (tree.edges[nid].parent_id == ROOT_ID or tree.edges[nid].operator == "Explore")
    ]
    records = []
    for r_id in dir_roots:
        stack = [c for c in children.get(r_id, ()) if tree.edges[c].operator != "Explore"]
        members = [r_id]
        while stack:
            cur = stack.pop()
            members.append(cur)
            stack.extend(c for c in children.get(cur, ()) if tree.edges[c].operator != "Explore")
        records.append((r_id, members, max(members)))

    records.sort(key=lambda item: -item[2])
    distinct: list[tuple[int, list[int], int]] = []
    seen_ident: set[str] = set()
    for rec in records:
        r_code = tree.nodes[rec[0]].code
        ident = identity(r_code)
        if ident in seen_ident or any(code_distance(r_code, tree.nodes[p[0]].code) < 0.12 for p in distinct):
            continue
        seen_ident.add(ident)
        distinct.append(rec)

    chosen = sorted(distinct[: config.max_directions], key=lambda item: item[0])
    rows: list[str] = []
    has_exp = False
    if config.include_experiences and (root_exp := clean_text(tree.root_selection_experience)):
        rows.append(f"Experience from initial directions: {root_exp}")
        has_exp = True

    for idx, (r_id, members, latest) in enumerate(chosen, 1):
        r_node, r_edge = tree.nodes[r_id], tree.edges[r_id]
        direction_id = tree.direction_of(r_id)
        lines = [f"Approach {idx}: {r_node.design}"]
        if latest != r_id:
            lines.append(f"Recent development: {tree.nodes[latest].design}")
        if config.include_experiences:
            if r_edge.variation_experience:
                lines.append(f"Development experience: {r_edge.variation_experience}")
                has_exp = True
            if any(tree.nodes[m].selection_experience for m in tree.nodes if tree.direction_of(m) == direction_id):
                has_exp = True
            exps: list[str] = []
            seen_exp: set[str] = set()
            for m in sorted(members):
                sel_exp = tree.nodes[m].selection_experience
                if sel_exp and sel_exp not in seen_exp:
                    seen_exp.add(sel_exp)
                    source = "the entry approach" if m == r_id else (
                        "the recent development" if m == latest else tree.nodes[m].design
                    )
                    exps.append(f"Experiments around {source}: {sel_exp}")
            if exps:
                lines.append("Experience from related experiments:\n" + "\n\n".join(exps))
        rows.append("\n\n".join(lines))
    return ("[Algorithmic Ideas and Experience]\n" + "\n\n".join(rows) if rows else ""), has_exp


def _current_algorithm_card(node: Node, *, show_score: bool) -> str:
    lines = [f"Algorithm {node.id}"]
    if show_score and node.fitness is not None:
        lines.append(f"Score: {node.fitness:.6g}")
    lines.append(f"```python\n{bare_code(node.code)}\n```")
    return "\n".join(lines)


def build_evolution_context(
    tree: TreePopulation,
    parent_id: int,
    operator: Operator,
    reference_id: int | None = None,
    config: ContextConfig = ContextConfig(),
) -> tuple[str, str, bool]:
    """按算子职责抽取路径级（Refine）、分支级（Crossover）或全树级（Explore）上下文。"""
    parent = tree.nodes[parent_id]
    blocks: list[str] = []
    if operator != "Explore":
        blocks.append("[Current Algorithm]\n" + _current_algorithm_card(parent, show_score=True))

    has_scores = True
    if operator == "Refine":
        if path_text := _render_subpath(
            tree, ROOT_ID, _path_from_root(tree, parent_id), "Evolution Path",
            max_steps=config.max_path_steps, include_experiences=config.include_experiences,
        ):
            blocks.append(path_text)
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
        if lca_id != ROOT_ID:
            lca_node = tree.nodes[lca_id]
            lca_lines = [
                f"[Common Ancestor]\nAlgorithm {lca_id}: fitness = {lca_node.fitness:.6g}",
                f"Design: {lca_node.design}",
            ]
            if config.include_experiences and lca_node.selection_experience:
                lca_lines.append(f"Branch experience at ancestor: {lca_node.selection_experience}")
            blocks.append("\n".join(lca_lines))
        elif config.include_experiences and (root_exp := clean_text(tree.root_selection_experience)):
            blocks.append(f"[Origin Context]\nExperience from initial directions: {root_exp}")

        for branch, title in ((branch_to_parent, "Current Branch since Common Ancestor"),
                              (branch_to_ref, "Reference Branch since Common Ancestor")):
            if text := _render_subpath(
                tree, lca_id, branch, title,
                max_steps=config.max_path_steps, include_experiences=config.include_experiences,
            ):
                blocks.append(text)

        ref_lines = [f"Algorithm {ref_node.id}", f"Score: {ref_node.fitness:.6g}", f"Design: {ref_node.design}"]
        if config.include_reference_code:
            ref_lines.append(f"```python\n{bare_code(ref_node.code)}\n```")
        blocks.insert(1, "[Reference Algorithm]\n" + "\n".join(ref_lines))
        instruction = (
            "Understand the computational ideas in both implementations and develop an algorithm that combines their useful strengths. "
            "Use their development and experimental experience to reason about how the computations fit together. "
            "The integration may reorganize the current algorithm rather than simply add another scoring term. "
            "If no useful combination emerges, improve the current algorithm using what the comparison reveals."
        )
    elif operator == "Explore":
        dir_text, has_scores = _directions_overview_block(tree, config=config)
        if dir_text:
            blocks.append(dir_text)
        blocks.append("[One Existing Implementation]\n" + _current_algorithm_card(parent, show_score=False))
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
        if att_text := _direct_attempts_block(tree, parent_id, "Attempts from Current Algorithm", config=config):
            blocks.append(att_text)
    return "\n\n".join(blocks), instruction, has_scores


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
    task: Task, tree: TreePopulation, parent_id: int, operator: Operator,
    reference_id: int | None = None, config: ContextConfig = ContextConfig(),
) -> str:
    text, instruction, has_scores = build_evolution_context(tree, parent_id, operator, reference_id, config)
    return _render(task, text, instruction, has_scores=has_scores)


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
            f"Algorithm {p.id}\nDesign: {clean_text(p.design)}\nScore: {p.evaluation.fitness}"
            for p in valid
        ]
        context.append("[Existing Algorithmic Ideas]\n\n" + "\n\n".join(algorithms))
    failed = sorted(
        sorted(
            (p for p in archive.values() if p.parent_id == 0 and p.evaluation is not None and p.evaluation.status != "success"),
            key=lambda p: -p.id,
        )[:failure_count],
        key=lambda p: p.id,
    )
    if failed:
        context.append("[Failed Attempts]\n" + "\n\n".join(
            f"Algorithm {p.id}\nDesign: {clean_text(p.design)}\n"
            f"```python\n{bare_code(p.code)}\n```\nFailure: {failure_feedback(p.evaluation.failure)}"
            for p in failed
        ))
    instruction = (
        "Develop an effective algorithm with a different main computation from the ideas already represented. "
        "Seek a different way to compute the decision, rather than another weighting of the same local features. "
        "You can change the representation, the information used to make decisions, or the computational procedure. "
        "Implement the chosen approach as a complete algorithm."
    ) if valid else (
        "Design an effective algorithm for this task. Consider how the returned decision affects the complete solution "
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
        "and satisfies the task requirements, while retaining the intended core decision "
        "mechanism."
    ) if program.evaluation.status == "error" else (
        "Make the algorithm finish within the reported time limit. "
        "Reduce the cost of its core computation, using an approximation when needed, "
        "while retaining the intended algorithmic idea."
    )
    return _render(task, context, instruction)
