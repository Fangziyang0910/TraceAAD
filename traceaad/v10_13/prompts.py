"""Concise evolutionary operators and bounded, action-specific contexts."""

from dataclasses import dataclass, field

from .parsing import OUTPUT_FORMAT
from .tree import Node

OPERATOR_INSTRUCTIONS = {
    "Init": (
        "Develop a competitive algorithm for this task using the problem's "
        "structure and your algorithmic knowledge. Use the evaluated examples, "
        "when available, to explore another promising approach."
    ),
    "Refine": (
        "Build on the current algorithm's main idea. Use its computations and "
        "recent design results to find a promising improvement to the decision "
        "rule or its organization, while retaining what makes the approach useful."
    ),
    "Tune": (
        "Preserve the current algorithm's main idea and computational structure. "
        "Calibrate influential parameters, thresholds, weights, or schedules, "
        "considering how they interact and affect decisions across inputs."
    ),
    "Pivot": (
        "Reconsider the current approach using the task structure and your "
        "algorithmic knowledge. Develop a competitive alternative based on a "
        "different main decision idea. A reference, when supplied, is a source "
        "of inspiration for a new approach."
    ),
    "Fuse": (
        "Use the host algorithm as a starting point. Compare its computations "
        "with the donor, adapt ideas that complement the host, and integrate "
        "them into one coherent algorithm aiming to outperform both inputs. "
        "Do not mechanically concatenate the two programs."
    ),
}


@dataclass
class PromptContext:
    prompt: str
    operator: str
    reference_ids: list[int]
    reference_program: Node | None = None
    fallback_reason: str | None = None
    history_ids: list[int] = field(default_factory=list)


class PromptBuilder:
    def __init__(self, llm, task_contract, *, max_tokens, history_depth,
                 lookup, all_nodes):
        self.llm = llm
        self.task_contract = task_contract
        self.max_tokens = max_tokens
        self.history_depth = history_depth
        self.lookup = lookup
        self.all_nodes = all_nodes

    def count(self, text):
        return self.llm.count_prompt_tokens(text)

    def formation_edges(self, current):
        edges = []
        for _ in range(self.history_depth):
            if current is None or current.parent_id is None:
                break
            source = self.lookup(current.parent_id)
            if source is None:
                raise ValueError("missing formation predecessor")
            edges.append((source, current))
            current = source
        return list(reversed(edges))

    def program(self, node, title):
        idea = ' '.join(node.idea.split()) if node.idea else ""
        description = f"Idea: {idea}\n" if idea else ""
        return (f"# {title}\nFitness: {node.fitness}\n{description}"
                f"```python\n{node.code.strip()}\n```")

    def _history_text(self, edges):
        lines = ["# Recent Design History",
                 "Recorded ideas and measured scores along the current program's formation."]
        for source, target in edges:
            lines.append(
                f"{target.operator} | Fitness {source.fitness} -> {target.fitness}\n"
                f"Idea: {' '.join((target.idea or '').split())}"
            )
        return "\n\n".join(lines)

    def _join(self, parts, operator):
        return "\n\n".join(parts + ["# Design Task\n" + OPERATOR_INSTRUCTIONS[operator],
                                    "# Output\n" + OUTPUT_FORMAT])

    def build_initial_context(self):
        roots = sorted((node for node in self.all_nodes() if node.parent_id is None),
                       key=lambda node: node.id)
        retained = list(roots)
        while True:
            parts = [self.task_contract, "Fitness: higher is better."]
            parts += [self.program(node, "Previous Initial Algorithm") for node in retained]
            text = self._join(parts, "Init")
            if self.count(text) <= self.max_tokens:
                return PromptContext(text, "Init", [node.id for node in retained],
                                     fallback_reason="initial_examples_trimmed" if retained != roots else None)
            if not retained:
                raise ValueError("task and output instructions exceed context capacity")
            retained.pop(0)

    def build_initial(self):
        return self.build_initial_context().prompt

    def build_development(self, parent, operator, donor=None):
        # Formation links are useful for local development, not semantic classes.
        edges = self.formation_edges(parent) if operator in ("Refine", "Tune") else []
        head = [self.task_contract, "Fitness: higher is better.",
                self.program(parent, "Host Algorithm" if operator == "Fuse" else "Current Algorithm")]
        while True:
            parts = head + ([self._history_text(edges)] if edges else [])
            if donor is not None and operator in ("Pivot", "Fuse"):
                parts.append(self.program(donor, "Donor Algorithm" if operator == "Fuse" else "Reference Algorithm"))
            text = self._join(parts, operator)
            if self.count(text) <= self.max_tokens:
                if operator == "Fuse" and donor is None:
                    fallback = self.build_development(parent, "Refine")
                    fallback.fallback_reason = "donor_unavailable"
                    return fallback
                used_donor = donor if operator in ("Pivot", "Fuse") else None
                return PromptContext(text, operator, [used_donor.id] if used_donor else [], used_donor,
                                     history_ids=[child.id for _, child in edges])
            if edges:
                edges.pop(0)  # Keep recent steps; never cut a program or task interface.
            elif donor is not None:
                fallback = self.build_development(parent, "Refine" if operator == "Fuse" else operator)
                fallback.fallback_reason = "reference_exceeds_context"
                return fallback
            else:
                raise ValueError("task, current program and output instructions exceed context capacity")

    def build(self, parent, operator, donor=None):
        return self.build_development(parent, operator, donor).prompt
