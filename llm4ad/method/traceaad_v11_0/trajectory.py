"""Prompt construction for the TraceAAD V11.0 search."""

import ast

from .core import digest
from .errors import OUTPUT

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
        "Identify a limitation of the current approach in relation to the task, "
        "using the reference ideas where helpful. Develop and implement a "
        "competitive alternative based on a different main idea that addresses "
        "this limitation."
    ),
    "Fuse": (
        "Compare the current algorithm and the reference ideas to identify "
        "useful ideas that can complement one another. Adapt and combine "
        "selected ideas into a coherent algorithm that aims to improve "
        "performance on the task."
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


class TrajectoryBuilder:
    """Formation-path context for Refine, Tune, and informed initialization."""

    def __init__(self, llm, task_contract, *, max_tokens, max_events, lookup, all_nodes):
        self.llm = llm
        self.task_contract = task_contract
        self.max_tokens = max_tokens
        self.max_events = max_events
        self.lookup = lookup
        self.all_nodes = all_nodes
        self._counts = {}

    def count(self, text):
        key = digest(text)
        if key not in self._counts:
            self._counts[key] = self.llm.count_prompt_tokens(text)
        return self._counts[key]

    def function_view(self, node):
        tree = ast.parse(node.code)
        return ast.unparse(tree).strip()

    def formation_edges(self, current):
        edges = []
        for _ in range(self.max_events):
            if current is None or current.parent_id is None:
                break
            source = self.lookup(current.parent_id)
            if source is None:
                raise ValueError("missing formation predecessor")
            edges.append((source, current))
            current = source
        return list(reversed(edges))

    def program(self, node, title):
        return f"# {title}\nFitness: {node.fitness}\n```python\n{self.function_view(node)}\n```"

    def _complete(self, parts, instruction):
        parts.extend(["# Design Task\n" + instruction, "# Output\n" + OUTPUT])
        text = "\n\n\n".join(parts)
        self.check_capacity(text)
        return text

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
        return self._complete(parts, instruction)

    def build(self, parent, operator):
        head = [self.task_contract, "Fitness: higher is better.",
                self.program(parent, "Current Algorithm")]
        tail = ["# Design Task\n" + OPERATOR_INSTRUCTIONS[operator], "# Output\n" + OUTPUT]
        edges = self.formation_edges(parent)
        kept = len(edges)
        while True:
            middle = [self._history_text(edges[-kept:])] if kept else []
            text = "\n\n\n".join(head + middle + tail)
            if self.count(text) <= self.max_tokens:
                if kept < len(edges):
                    print(f"v110: formation history truncated to {kept} of {len(edges)} steps "
                          f"(prompt exceeded the {self.max_tokens}-token context budget)", flush=True)
                return text
            if kept == 0:
                raise ValueError("complete prompt exceeds the model context budget")
            kept -= 1

    def check_capacity(self, text):
        if self.count(text) > self.max_tokens:
            raise ValueError("complete prompt exceeds the model context budget")


class ReferenceContextBuilder(TrajectoryBuilder):
    """Shared archive-reference context for Pivot and Fuse."""

    def build(self, parent, operator, references=(), *, require_minimal=True):
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
        tail = ["# Design Task\n" + OPERATOR_INSTRUCTIONS[operator], "# Output\n" + OUTPUT]
        retained = list(ordered)
        while True:
            middle = [self._references_text(retained)] if retained else []
            text = "\n\n\n".join(head + middle + tail)
            if self.count(text) <= self.max_tokens:
                if len(retained) < len(ordered):
                    print(f"v110: reference context trimmed to {len(retained)} of {len(ordered)} "
                          f"entries (prompt exceeded the {self.max_tokens}-token context budget)",
                          flush=True)
                return text, retained
            if not retained:
                if require_minimal:
                    raise ValueError("complete prompt exceeds the model context budget")
                return None, retained
            retained.pop()

    def _references_text(self, references):
        lines = ["# Reference Nodes", REFERENCE_INTRO]
        for index, node in enumerate(references, 1):
            entry = [f"Reference {index} | Fitness: {node.fitness}"]
            if node.idea:
                entry.append("Idea: " + " ".join(node.idea.split()))
            lines.append("\n".join(entry))
        return "\n".join(lines)
