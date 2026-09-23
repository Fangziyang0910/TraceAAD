"""V10.5 launch prompts: one shared context, concise Idea, complete Code."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

from llm4ad.method.traceaad_v10_3.prompts import build_task_contract, strip_comments_for_prompt
from llm4ad.method.traceaad_v10_3.schema import Node

PRINCIPLE = (
    "Improve evaluated solution quality within the task's information and runtime\n"
    "limits. Use the formation history to understand the changes and their observed\n"
    "outcomes; a previously unsuccessful mechanism may be revisited when its\n"
    "limitation is addressed. Choose the computational structure and implementation\n"
    "details that best serve the proposed improvement."
)
INSTRUCTIONS = {
    "Init": (
        "Design a competitive algorithm for this task using the available function\n"
        "interface. Develop one coherent approach and implement it as a complete\n"
        "executable function."
    ),
    "Refine": (
        "Develop the current algorithm's main approach by addressing the limitation you\n"
        "judge most consequential. You may substantially revise formulas, auxiliary\n"
        "mechanisms, or computational structure when this better realizes that approach."
    ),
    "Pivot": (
        "Develop a promising alternative with a different primary decision mechanism\n"
        "from the current algorithm. Use the current implementation and its formation\n"
        "history to identify what limits this approach, while retaining any components\n"
        "that remain useful for the alternative."
    ),
    "Fuse": (
        "Use the current and reference algorithms to identify complementary useful ideas\n"
        "and develop one coherent algorithm that aims to improve on the stronger input.\n"
        "Keep, replace, or recombine components according to their roles in the resulting\n"
        "decision process; neither implementation has to retain its original structure."
    ),
}
OUTPUT = (
    "Return one algorithm in exactly these two parts:\n"
    "Idea: <1-3 concise sentences describing the implemented mechanism and the\n"
    "decision change it is intended to produce>\n"
    "```python\n"
    "<complete executable implementation of the target function, with any required imports and helpers>\n"
    "```\n"
    "Keep the Idea brief, roughly within 100 words. Use code comments for important\n"
    "implementation constraints when useful, without repeating the design discussion."
)
TEMPLATE_HASH = hashlib.sha256(
    (PRINCIPLE + "\n".join(INSTRUCTIONS.values()) + OUTPUT).encode()
).hexdigest()


@dataclass(frozen=True)
class Prompt:
    text: str
    tokens: int
    history_ids: tuple[int, ...]
    omissions: tuple[str, ...]


def formation_events(current: Node | None, ancestors: list[Node], limit: int) -> list[tuple[Node, Node]]:
    """Return child/parent edges, oldest first, including current's formation."""
    if current is None or limit == 0:
        return []
    lineage = [current, *ancestors]
    by_id = {node.id: node for node in lineage}
    edges = [(node, by_id[node.parent_id]) for node in lineage if node.parent_id in by_id]
    return list(reversed(edges[:limit]))


def render_history(events: list[tuple[Node, Node]], omitted: set[int]) -> str:
    lines = ["# Formation History"]
    for step, (child, parent) in enumerate(events, 1):
        trend = "improved" if child.fitness > parent.fitness else "regressed" if child.fitness < parent.fitness else "unchanged"
        lines.append(
            f"Step {step} | {child.operator} | Fitness: {parent.fitness} -> {child.fitness} | {trend}\n"
            f"Idea: {'[Design text omitted due to length.]' if child.id in omitted else child.idea}"
        )
    return "\n\n".join(lines)


class PromptBuilder:
    """Cache exact server-tokenizer counts; only remove whole history events."""

    def __init__(self, llm, task_contract: str, *, max_tokens: int, history_tokens: int, max_events: int, log_count=None):
        self.llm = llm
        self.task_contract = task_contract
        self.max_tokens = max_tokens
        self.history_tokens = history_tokens
        self.max_events = max_events
        self.log_count = log_count
        self._counts: dict[tuple[bool, str], int] = {}
        self._views: dict[int, str] = {}

    def count(self, text: str, *, chat: bool = False) -> int:
        key = (chat, hashlib.sha256(text.encode()).hexdigest())
        if key not in self._counts:
            fn = self.llm.count_prompt_tokens if chat else self.llm.count_tokens
            started = time.time()
            record = {'kind': 'chat' if chat else 'text', 'text_hash': key[1]}
            try:
                self._counts[key] = fn(text)
                record['tokens'] = self._counts[key]
            except Exception as exc:
                record['error'] = str(exc)
                raise
            finally:
                if self.log_count is not None:
                    record.update(ts=started, seconds=time.time() - started)
                    self.log_count(record)
        return self._counts[key]

    def _code(self, node: Node) -> str:
        if node.id not in self._views:
            self._views[node.id] = strip_comments_for_prompt(node.code)
        return self._views[node.id]

    def assemble(self, current, operator, donor, events, omitted) -> str:
        parts = [self.task_contract]
        if current is not None:
            # Non-root Idea belongs to the formation event; do not repeat it.
            idea = f"Idea: {current.idea}\n" if current.parent_id is None else ""
            parts.append(f"# Current Algorithm\n{idea}Fitness: {current.fitness}\n\n```python\n{self._code(current)}\n```")
        if donor is not None:
            parts.append(f"# Reference Algorithm\nIdea: {donor.idea}\nFitness: {donor.fitness}\n\n```python\n{self._code(donor)}\n```")
        if events:
            parts.append(render_history(events, omitted))
        principle = "" if operator == "Init" else PRINCIPLE + "\n\n"
        parts.append(f"# Design Direction\nOperator: {operator}\n\n{principle}{INSTRUCTIONS[operator]}")
        parts.append(f"# Output\n{OUTPUT}")
        return "\n\n\n".join(parts)

    def fits(self, current, operator, donor=None) -> bool:
        return self.count(self.assemble(current, operator, donor, [], set()), chat=True) <= self.max_tokens

    def build(self, current, ancestors, operator, donor=None) -> Prompt:
        events = formation_events(current, ancestors, self.max_events)
        omitted: set[int] = set()
        reasons: list[str] = []
        for edge in events:
            if self.count(render_history([edge], set())) > self.history_tokens:
                omitted.add(edge[0].id)
                reasons.append(f"idea_too_long:{edge[0].id}")
        while events and self.count(render_history(events, omitted)) > self.history_tokens:
            reasons.append(f"history_budget:{events.pop(0)[0].id}")
        while True:
            text = self.assemble(current, operator, donor, events, omitted)
            tokens = self.count(text, chat=True)
            if tokens <= self.max_tokens:
                return Prompt(text, tokens, tuple(n.id for n, _ in events), tuple(reasons))
            if not events:
                raise ValueError("minimum complete prompt exceeds the model context budget")
            reasons.append(f"context_budget:{events.pop(0)[0].id}")
