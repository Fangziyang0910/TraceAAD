"""Prompt construction for the TraceAAD V11.1 search."""

import ast
from dataclasses import dataclass

from .parsing import OUTPUT_FORMAT
from .tree import Node

OPERATOR_INSTRUCTIONS = {
    "Init": (
        "Develop a promising algorithmic idea for the task and implement it "
        "as a competitive algorithm."
    ),
    "Refine": (
        "Build on the current algorithm's main idea. Use its code and design "
        "history to make a focused improvement to its computations or their "
        "organization."
    ),
    "Tune": (
        "Preserve the algorithm's main idea and computational structure. "
        "Calibrate influential parameters according to how they affect its "
        "decisions, aiming to improve performance."
    ),
    "Pivot": (
        "Use the current algorithm as a starting point, but try a different main "
        "decision mechanism with help from the reference ideas. Develop and "
        "implement a coherent competitive alternative."
    ),
    "Fuse": (
        "Starting from the current algorithm, borrow useful design ideas from "
        "the reference algorithms, combine their strengths, and form a coherent "
        "improved algorithm. Do not mechanically concatenate programs."
    ),
}

INIT_REFERENCE_INSTRUCTION = (
    "Use the previously evaluated algorithms and their results to develop "
    "another promising approach."
)

REFERENCE_INTRO = (
    "These reference nodes were evaluated earlier in the search and are listed "
    "from best to worst fitness. Use their recorded ideas as design references."
)

@dataclass
class PromptContext:
    prompt: str
    operator: str
    reference_ids: list[int]
    reference_program: Node | None = None
    fallback_reason: str | None = None


class PromptBuilder:
    """Build initialization, formation-history, and archive-reference prompts."""

    def __init__(self, llm, task_contract, *, max_tokens, history_depth, lookup, all_nodes):
        self.llm = llm
        self.task_contract = task_contract
        self.max_tokens = max_tokens
        self.history_depth = history_depth
        self.lookup = lookup
        self.all_nodes = all_nodes

    def count(self, text):
        return self.llm.count_prompt_tokens(text)

    def program_code(self, node):
        tree = ast.parse(node.code)
        return ast.unparse(tree).strip()

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
        return f"# {title}\nFitness: {node.fitness}\n```python\n{self.program_code(node)}\n```"

    def _history_text(self, edges):
        history = [
            "# Design History of the Current Algorithm",
            "Use the recorded design changes and their results to guide this design.",
        ]
        for index, (source, target) in enumerate(edges, 1):
            history.append(
                f"Step {index} | {target.operator} | "
                f"Fitness: {source.fitness} -> {target.fitness}"
            )
            if target.idea:
                history.append("Idea: " + " ".join(target.idea.split()))
        return "\n".join(history)

    def build_initial(self):
        roots = sorted((node for node in self.all_nodes() if node.parent_id is None),
                       key=lambda node: node.id)
        parts = [self.task_contract, "Fitness: higher is better."]
        instruction = OPERATOR_INSTRUCTIONS["Init"]
        if roots:
            parts.append("# Previous Initial Algorithms\nEarlier evaluated functions, in generation order.")
            parts.extend(self.program(node, "Previous Initial Algorithm") for node in roots)
            instruction += " " + INIT_REFERENCE_INSTRUCTION
        parts.extend(["# Design Task\n" + instruction, "# Output\n" + OUTPUT_FORMAT])
        text = "\n\n\n".join(parts)
        self.check_capacity(text)
        return text

    def _fit_prompt(self, head, tail, middle, render, *, label, empty_ok=False):
        """Join head + middle + tail, dropping items from the end of ``middle``
        (the least important end) until the prompt fits the token budget.

        Returns (text, retained middle items). When even the bare prompt
        exceeds the budget, raises, or returns (None, []) if ``empty_ok``.
        """
        full = list(middle)
        retained = list(middle)
        while True:
            middle_parts = [render(retained)] if retained else []
            text = "\n\n\n".join(head + middle_parts + tail)
            if self.count(text) <= self.max_tokens:
                if len(retained) < len(full):
                    print(f"v111: {label} trimmed to {len(retained)} of {len(full)} "
                          f"(prompt exceeded the {self.max_tokens}-token context budget)", flush=True)
                return text, retained
            if not retained:
                if empty_ok:
                    return None, retained
                raise ValueError("complete prompt exceeds the model context budget")
            retained.pop()

    def build_trajectory(self, parent, operator):
        head = [self.task_contract, "Fitness: higher is better.",
                self.program(parent, "Current Algorithm")]
        tail = ["# Design Task\n" + OPERATOR_INSTRUCTIONS[operator], "# Output\n" + OUTPUT_FORMAT]
        edges = self.formation_edges(parent)
        newest_first = list(reversed(edges))
        text, _ = self._fit_prompt(
            head, tail, newest_first,
            lambda kept: self._history_text(list(reversed(kept))),
            label="formation history steps",
        )
        return text

    def check_capacity(self, text):
        if self.count(text) > self.max_tokens:
            raise ValueError("complete prompt exceeds the model context budget")

    def build_references(self, parent, operator, references=(), *, reference_program=None, require_minimal=True):
        """Assemble the prompt; return (text, retained references in display order).

        Capacity trimming drops reference entries from worst fitness first and
        never touches the task contract or the current algorithm code. When the
        references are exhausted, a required minimal prompt that still exceeds
        the budget raises; otherwise (the Fuse fallback path) the caller gets
        (None, []) and validates capacity on the prompt it falls back to.
        """
        ordered = sorted(references, key=lambda node: (-node.fitness, node.id))
        head = [self.task_contract, "Fitness: higher is better.",
                self.program(parent, "Current Algorithm")]
        tail = []
        if reference_program is not None:
            tail.append(self.program(reference_program, "Reference Algorithm"))
        tail.extend(["# Design Task\n" + OPERATOR_INSTRUCTIONS[operator], "# Output\n" + OUTPUT_FORMAT])
        return self._fit_prompt(head, tail, ordered, self._references_text,
                                label="reference entries", empty_ok=not require_minimal)

    def _references_text(self, references):
        lines = ["# Reference Nodes", REFERENCE_INTRO]
        for index, node in enumerate(references, 1):
            entry = [f"Reference {index} | Fitness: {node.fitness}"]
            if node.idea:
                entry.append("Idea: " + " ".join(node.idea.split()))
            lines.append("\n".join(entry))
        return "\n".join(lines)

    def build_development(self, parent, operator, references=()):
        """Apply the fixed operator/context mapping, including Fuse fallback."""
        if operator in ("Refine", "Tune"):
            return PromptContext(self.build_trajectory(parent, operator), operator, [])

        reference_program = references[0] if operator == "Fuse" and references else None
        fallback_reason = None
        if operator == "Fuse" and reference_program is None:
            fallback_reason = "reference_pool_empty"
        else:
            prompt, retained = self.build_references(
                parent, operator, references, reference_program=reference_program,
                require_minimal=operator != "Fuse",
            )
            if operator != "Fuse" or retained:
                return PromptContext(prompt, operator, [node.id for node in retained],
                                     reference_program)
            fallback_reason = "references_trimmed_to_zero"

        return PromptContext(self.build_trajectory(parent, "Refine"), "Refine", [],
                             fallback_reason=fallback_reason)
