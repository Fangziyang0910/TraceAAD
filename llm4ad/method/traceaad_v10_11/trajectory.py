"""Prompt construction for the function-level TraceAAD search."""

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
        "Identify a limitation of the current approach in relation to the task. "
        "Develop and implement a competitive alternative based on a different "
        "main idea that addresses this limitation."
    ),
    "Fuse": (
        "Compare the current and reference algorithms to identify useful ideas "
        "that can complement one another. Adapt and combine their computations "
        "into a coherent algorithm that aims to outperform both."
    ),
}

INIT_REFERENCE_INSTRUCTION = (
    "Use the previously evaluated algorithms and their results to develop "
    "another promising approach."
)


class TrajectoryBuilder:
    def __init__(self, llm, task_contract, *, max_tokens, max_events, lookup, all_nodes,
                 include_history_code=False):
        self.llm = llm
        self.task_contract = task_contract
        self.max_tokens = max_tokens
        self.max_events = max_events
        self.lookup = lookup
        self.all_nodes = all_nodes
        self.include_history_code = include_history_code
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

    def build(self, parent, operator, donor=None):
        parts = [self.task_contract, "Fitness: higher is better."]
        if parent is not None:
            parts.append(self.program(parent, "Current Algorithm"))
            edges = self.formation_edges(parent)
            if edges:
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
                    if self.include_history_code:
                        history.append("Code:\n```python\n" + self.function_view(target) + "\n```")
                parts.append("\n\n".join(history))
        if donor is not None:
            parts.append(self.program(donor, "Reference Algorithm"))
        return self._complete(parts, OPERATOR_INSTRUCTIONS[operator])

    def check_capacity(self, text):
        if self.count(text) > self.max_tokens:
            raise ValueError("complete prompt exceeds the model context budget")
