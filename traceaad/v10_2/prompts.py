"""Prompt assembly for TraceAAD V10.2 (design doc section 4)."""

from __future__ import annotations

import io
import tokenize

from .schema import Node

OPERATOR_INSTRUCTIONS: dict[str, str] = {
    "Refine": (
        "Continue developing the current algorithmic direction. Preserve the core design\n"
        "principle of the current algorithm. Use the historical trajectory as evidence to\n"
        "understand how this direction has developed and what has already been tried,\n"
        "then make a coherent improvement that better realizes or strengthens the current\n"
        "idea. Simplification and removal of auxiliary mechanisms are valid improvements\n"
        "when they better realize the same core design principle. The implementation may\n"
        "change substantially if needed, but do not replace the core algorithmic principle\n"
        "with a different one."
    ),
    "Pivot": (
        "Develop a materially different algorithmic direction from the current node.\n"
        "Treat the current code as a usable starting scaffold, but do not assume that its\n"
        "core design principle should be preserved. Use the historical trajectory as\n"
        "evidence to understand how this lineage has developed and avoid reverting to\n"
        "mechanisms already used along this lineage, then introduce a different primary\n"
        "algorithmic mechanism. Reuse only implementation components that remain useful\n"
        "for the new mechanism. Discard legacy mechanisms that are not necessary for the\n"
        "new direction. The change must be different at the mechanism level, not merely\n"
        "parameter tuning, coefficient\n"
        "adjustment, or superficial restructuring."
    ),
    "Fuse": (
        "Create a coherent algorithm by selectively combining complementary mechanisms\n"
        "from the current algorithm and the reference algorithm. Identify the substantive\n"
        "mechanism worth retaining from the current algorithm and one compatible mechanism\n"
        "worth transferring from the reference algorithm. Integrate them according to\n"
        "their algorithmic roles. The retained target and transferred donor mechanisms\n"
        "must interact substantively in the resulting decision process. Do not preserve\n"
        "all mechanisms from either algorithm. When the reference mechanism overlaps with,\n"
        "supersedes, or makes an existing component unnecessary, replace or remove that\n"
        "component rather than stacking both. Preserve computationally expensive components\n"
        "when they play a distinct and useful algorithmic role. Avoid mechanical code\n"
        "copying, concatenating multiple heuristics, or accumulating several signals that\n"
        "express essentially the same information."
    ),
}

IMPLEMENTATION_PRINCIPLE = (
    "# Implementation Principle\n\n"
    "Prioritize algorithm quality. Complex computation is acceptable when it serves a "
    "distinct and useful algorithmic role. Do not remove effective mechanisms merely to "
    "shorten the implementation. At the same time, avoid redundant accumulation: when "
    "introducing a new mechanism, replace, simplify, or remove existing components that "
    "provide overlapping functionality or are superseded by the new design.\n\n"
    "Express the algorithm through executable code rather than explanatory comments. "
    "Avoid verbose comments and do not preserve comments from previous implementations "
    "unless they are essential for correctness."
)

INIT_INSTRUCTION = (
    "Design a novel algorithm for this task from scratch. Propose one clear design\n"
    "idea and implement it as a complete function that satisfies the Task Contract."
)

OUTPUT_CONTRACT = (
    "Respond with exactly two parts and nothing else:\n"
    "Latest Design Idea: <one sentence stating the actual algorithmic mechanism you introduce,\n"
    "modify, or combine>\n"
    "```python\n"
    "<the complete function implementation>\n"
    "```\n"
    "Keep the implementation concise. Do not include explanatory comments, reasoning notes,\n"
    "design discussion, or commented-out alternatives. Use comments only when strictly\n"
    "necessary to clarify non-obvious implementation constraints.\n"
    "Do not narrate your reasoning inside the code."
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


def strip_comments_for_prompt(code: str) -> str:
    """Return a comment-free prompt view without changing stored code."""
    tokens = []
    stream = io.StringIO(code)
    try:
        for token in tokenize.generate_tokens(stream.readline):
            if token.type == tokenize.COMMENT:
                continue
            tokens.append(token)
        result = tokenize.untokenize(tokens)
    except (tokenize.TokenError, IndentationError):
        return code

    lines = result.splitlines()
    cleaned = []
    blank = False
    for line in lines:
        if line.strip():
            cleaned.append(line.rstrip())
            blank = False
        elif not blank:
            cleaned.append("")
            blank = True
    return "\n".join(cleaned).strip()


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
            f"Latest Design Idea: {node.idea}\n"
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
        f"Latest Design Idea: {current.idea}\n"
        f"Fitness: {current.fitness}\n\n"
        f"```python\n{strip_comments_for_prompt(current.code)}\n```"
    )


def _render_reference(donor: Node) -> str:
    return (
        "# Reference Algorithm\n"
        f"Latest Design Idea: {donor.idea}\n"
        f"Fitness: {donor.fitness}\n\n"
        f"```python\n{strip_comments_for_prompt(donor.code)}\n```"
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
    parts = [task_contract, IMPLEMENTATION_PRINCIPLE]
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
