"""Prompt assembly for TraceAAD V10.1 (design doc sections 2.2 and 5)."""

from __future__ import annotations

from .schema import Node

OPERATOR_INSTRUCTIONS: dict[str, str] = {
    "Refine": (
        "Continue developing the current algorithmic direction. Preserve the core design\n"
        "principle of the current algorithm. Use the historical trajectory as evidence to\n"
        "understand how this direction has developed and what has already been tried,\n"
        "then make a coherent improvement that better realizes or strengthens the current\n"
        "idea. The implementation may change substantially if needed, but do not replace\n"
        "the core algorithmic principle with a different one."
    ),
    "Pivot": (
        "Develop a materially different algorithmic direction from the current node.\n"
        "Treat the current code as a usable starting scaffold, but do not assume that its\n"
        "core design principle should be preserved. Use the historical trajectory as\n"
        "evidence to understand which directions have already been explored, then\n"
        "introduce a different primary algorithmic mechanism. The change must be\n"
        "different at the mechanism level, not merely parameter tuning, coefficient\n"
        "adjustment, or superficial restructuring."
    ),
    "Fuse": (
        "Create a coherent algorithm by combining complementary mechanisms from the\n"
        "current algorithm and the provided reference algorithm. Treat the current\n"
        "algorithm as the target design and the reference algorithm as an external source\n"
        "of design knowledge. Preserve a substantive mechanism from the current\n"
        "algorithm and incorporate a compatible mechanism from the reference algorithm.\n"
        "Integrate them according to their algorithmic roles rather than mechanically\n"
        "copying code, averaging formulas, concatenating logic, or replacing the current\n"
        "algorithm with the reference algorithm."
    ),
}

INIT_INSTRUCTION = (
    "Design a novel algorithm for this task from scratch. Propose one clear design\n"
    "idea and implement it as a complete function that satisfies the Task Contract."
)

OUTPUT_CONTRACT = (
    "Respond with exactly two parts and nothing else:\n"
    "Idea: <one sentence stating the actual algorithmic mechanism you introduce,\n"
    "modify, or combine>\n"
    "```python\n"
    "<the complete function implementation>\n"
    "```"
)


def build_task_contract(task_description: str, template_program: str) -> str:
    return (
        "# Task Contract\n\n"
        f"{task_description.strip()}\n\n"
        "The target function to design is:\n\n"
        f"```python\n{template_program.strip()}\n```\n\n"
        "Objective: maximize the fitness returned by the evaluator (higher is better).\n"
        "Design the algorithm using only information available through the target\n"
        "function interface. Do not assume access to unavailable state or future\n"
        "information."
    )


def _trend(fitness: float, parent_fitness: float) -> str:
    if fitness > parent_fitness:
        return "Improved"
    if fitness < parent_fitness:
        return "Degraded"
    return "Unchanged"


def render_trajectory(ancestors: list[Node], display: int) -> str:
    """Render up to ``display`` generations, oldest displayed first.

    ``ancestors`` is nearest-parent-first and may carry one extra oldest entry
    that is used only to compute the trend of the oldest displayed generation.
    """
    if display <= 0 or not ancestors:
        return ""
    display_nodes = ancestors[:display]
    k = len(display_nodes)
    shown = list(reversed(display_nodes))
    lines = ["# Historical Design Trajectory"]
    for step, (ancestor_idx, node) in enumerate(
        zip(range(k - 1, -1, -1), shown), start=1
    ):
        entry = (
            f"\nStep {step}\n"
            f"Idea: {node.idea}\n"
            f"Fitness: {node.fitness}"
        )
        parent_position = ancestor_idx + 1
        if parent_position < len(ancestors):
            entry += f" ({_trend(node.fitness, ancestors[parent_position].fitness)})"
        lines.append(entry)
    return "\n".join(lines)


def _render_current(current: Node) -> str:
    return (
        "# Current Algorithm\n"
        f"Idea: {current.idea}\n"
        f"Fitness: {current.fitness}\n\n"
        f"```python\n{current.code}\n```"
    )


def _render_reference(donor: Node) -> str:
    return (
        "# Reference Algorithm\n"
        f"Idea: {donor.idea}\n"
        f"Fitness: {donor.fitness}\n\n"
        f"```python\n{donor.code}\n```"
    )


def _render_operator(operator: str, instruction: str) -> str:
    return (
        f"# Improvement Operator\n"
        f"Operator: {operator}\n\n"
        f"Instruction:\n{instruction}"
    )


def _assemble(
    task_contract: str,
    current: Node | None,
    ancestors: list[Node],
    display: int,
    operator: str,
    donor: Node | None,
) -> str:
    parts = [task_contract]
    if current is not None:
        parts.append(_render_current(current))
    trajectory = render_trajectory(ancestors, display)
    if trajectory:
        parts.append(trajectory)
    if donor is not None:
        parts.append(_render_reference(donor))
    instruction = INIT_INSTRUCTION if operator == "Init" else OPERATOR_INSTRUCTIONS[operator]
    parts.append(_render_operator(operator, instruction))
    parts.append(f"# Output\n{OUTPUT_CONTRACT}")
    return "\n\n\n".join(parts)


def build_prompt(
    *,
    task_contract: str,
    current: Node | None,
    ancestors: list[Node],
    operator: str,
    donor: Node | None,
    max_prompt_chars: int,
    max_gens: int,
) -> str:
    """Assemble the generation context; when over budget, drop the oldest
    trajectory generations first. Task Contract, Current Code, Operator
    Instruction and Fuse donor Code are never trimmed."""
    kept = min(max_gens, len(ancestors))
    while True:
        prompt = _assemble(task_contract, current, ancestors, kept, operator, donor)
        if len(prompt) <= max_prompt_chars or kept == 0:
            break
        kept -= 1
    if len(prompt) > max_prompt_chars:
        raise ValueError(
            f"prompt exceeds context budget even without trajectory: "
            f"{len(prompt)} > {max_prompt_chars} chars"
        )
    return prompt
