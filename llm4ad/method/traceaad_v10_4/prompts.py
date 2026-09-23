"""Two-stage design and realization prompts for TraceAAD V10.4."""

from __future__ import annotations

from llm4ad.method.traceaad_v10_3.prompts import (
    build_task_contract,
    strip_comments_for_prompt,
)
from llm4ad.method.traceaad_v10_3.schema import Node


DESIGN_INSTRUCTIONS: dict[str, str] = {
    "Refine": (
        "Improve the current algorithm by continuing to develop its algorithmic "
        "idea. Study the current algorithm and how it was formed, then propose "
        "the development you judge most promising for making the algorithm better."
    ),
    "Pivot": (
        "Study the current algorithm and how it was formed. Identify an important "
        "limitation or unrealized opportunity in the current design, and develop a "
        "stronger algorithmic idea from that insight."
    ),
    "Fuse": (
        "Study the current and reference algorithms and identify the strengths of "
        "their designs. Develop a better algorithmic idea that brings their useful "
        "insights together as one coherent design."
    ),
}

INIT_DESIGN_INSTRUCTION = (
    "Design a competitive algorithm for this task from scratch. Develop a clear "
    "algorithmic idea that can be implemented as the target function."
)


def _result(child_fitness: float, parent_fitness: float) -> str:
    if child_fitness > parent_fitness:
        return "improve"
    if child_fitness < parent_fitness:
        return "regress"
    return "plateau"


def render_formation_history(
    current: Node | None,
    ancestors: list[Node],
    max_events: int,
) -> str:
    """Render the recent edges that formed ``current``, oldest event first.

    ``ancestors`` is nearest-parent-first. Unlike the V10.3 snapshot history,
    the current node's own formation edge is included, matching V9.16.
    """
    lines = ["# Recent Algorithm Improvement History"]
    if current is None or current.parent_id is None or max_events <= 0:
        lines.append("No formation events are shown for this algorithm.")
        return "\n".join(lines)

    lineage = [current, *ancestors]
    by_id = {node.id: node for node in lineage}
    events = [
        node
        for node in lineage
        if node.parent_id is not None and node.parent_id in by_id
    ][:max_events]
    for index, node in enumerate(reversed(events), start=1):
        parent = by_id[node.parent_id]
        lines.extend(
            [
                "",
                f"[History {index}] Formation step",
                f"Idea: {node.idea}",
                (
                    f"Result: {_result(node.fitness, parent.fitness)} "
                    f"(Fitness: {parent.fitness} -> {node.fitness})"
                ),
            ]
        )
    return "\n".join(lines)


def _render_current(current: Node) -> str:
    return (
        "# Current Algorithm\n"
        f"Idea: {current.idea}\n"
        f"Fitness: {current.fitness}\n\n"
        f"```python\n{strip_comments_for_prompt(current.code)}\n```"
    )


def _render_reference(donor: Node) -> str:
    return (
        "# Reference Algorithm\n"
        f"Idea: {donor.idea}\n"
        f"Fitness: {donor.fitness}\n\n"
        f"```python\n{strip_comments_for_prompt(donor.code)}\n```"
    )


def build_idea_prompt(
    *,
    task_contract: str,
    current: Node | None,
    ancestors: list[Node],
    operator: str,
    donor: Node | None,
    max_events: int,
) -> str:
    """Build the complete design context for the first LLM call."""
    parts = [task_contract]
    if current is not None:
        parts.append(_render_current(current))
        parts.append(render_formation_history(current, ancestors, max_events))
    if donor is not None:
        parts.append(_render_reference(donor))
    instruction = (
        INIT_DESIGN_INSTRUCTION
        if operator == "Init"
        else DESIGN_INSTRUCTIONS[operator]
    )
    parts.extend(
        [
            f"# Design Direction\n{instruction}",
            (
                "# Proposed Algorithm Design\n"
                "Describe the algorithmic idea you propose. Make the design clear "
                "enough to guide its implementation. The Python implementation will "
                "be produced separately after this design. Respond with the design, "
                "not the Python implementation."
            ),
        ]
    )
    return "\n\n\n".join(parts)


def build_code_prompt(
    *,
    task_contract: str,
    current: Node | None,
    donor: Node | None,
    idea: str,
) -> str:
    """Build the realization context without trajectory or operator guidance."""
    parts = [task_contract]
    if current is not None:
        parts.append(_render_current(current))
    if donor is not None:
        parts.append(_render_reference(donor))
    parts.extend(
        [
            f"# Proposed Algorithm Design\n{idea}",
            (
                "# Implementation\n"
                "Implement the proposed algorithm design as a complete executable "
                "implementation of the target function, using the current "
                "implementation as the starting program when one is provided.\n\n"
                "Return only the complete Python implementation in a python fenced "
                "code block."
            ),
        ]
    )
    return "\n\n\n".join(parts)


__all__ = [
    "DESIGN_INSTRUCTIONS",
    "build_code_prompt",
    "build_idea_prompt",
    "build_task_contract",
    "render_formation_history",
]
