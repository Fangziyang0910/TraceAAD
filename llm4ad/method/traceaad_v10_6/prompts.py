"""Shared task contracts, design-first generation and grounded implementation ideas."""

from dataclasses import dataclass
import hashlib
import re

from llm4ad.base import TextFunctionProgramConverter
from llm4ad.method.traceaad_v10_5.prompts import PromptBuilder as BaseBuilder, formation_events

HISTORY_GUIDANCE = (
    "The development history describes how the current algorithm was built and how\n"
    "previous versions performed. Use it as context for this algorithm design task."
)
INSTRUCTIONS = {
    'Init': 'Design a competitive algorithm for this task and implement it using the provided\nfunction interface.',
    'Refine': "Build on the current algorithm's main idea to design an improved version for\nthis task. Choose the implementation changes most likely to improve its performance.",
    'Pivot': 'Design an algorithm for this task using a different main idea from the current\nalgorithm. Use the current algorithm as a reference for developing a promising\nnew approach.',
    'Fuse': 'Design an improved algorithm for this task by combining useful ideas from the\ncurrent and reference algorithms. Choose and adapt the parts that work well\ntogether, aiming to outperform both algorithms.',
}
OUTPUT = """First describe your proposed algorithm in one concise paragraph, explaining its
main decision method and key calculations. Then implement it in this format:

Idea: <design idea>
```python
<complete target implementation, including all required imports and helpers>
```"""
SUMMARY_INSTRUCTION = """Explain the algorithm implemented by the final code for the stated task.
The design idea provides the intended approach; the code operations determine
the implemented method. Describe its main decision rule, the calculations that
determine its output, and the important parameters and conditions. Derive each
preference from the computation and the supplied task contract. State directly
established effects as facts and expected performance benefits as hypotheses.
Write approximately 500 words in 2–3 paragraphs, scaled to the implementation's
complexity. Return your implementation idea as: Idea: <implementation idea>"""
COMPARISON = 'Describe the important implementation changes relative to the parent code.'
GENERATION = 'idea_code_then_implementation_idea'
TEMPLATE_HASH = hashlib.sha256((HISTORY_GUIDANCE + str(INSTRUCTIONS) + OUTPUT + SUMMARY_INSTRUCTION + COMPARISON).encode()).hexdigest()


def build_task_contract(evaluation):
    # Match V9.16: shared description and docstring, with the body left to evolve.
    template = TextFunctionProgramConverter.text_to_function(evaluation.template_program)
    template.body = ''
    return (f'# Task Contract\n{evaluation.task_description.strip()}\n\n'
            f'Target interface:\n```python\n{str(template).strip()}\n```\n'
            'Fitness is this task\'s score and higher is better.')


def build_summary_prompt(task_contract, design_idea, code, parent=None):
    parts = [task_contract, '# Design Idea\n' + design_idea]
    if parent is not None:
        parts.append(f'# Parent Code\n```python\n{parent.code}\n```')
    parts.extend([f'# Final Code\n```python\n{code}\n```',
                  '# Implementation Description\n' + SUMMARY_INSTRUCTION])
    if parent is not None:
        parts.append(COMPARISON)
    return '\n\n'.join(parts)


@dataclass(frozen=True)
class Prompt:
    text: str
    tokens: int
    history_ids: tuple[int, ...]
    omissions: tuple[str, ...]
    history_tokens: int
    summaries: dict


class PromptBuilder(BaseBuilder):
    def __init__(self, *args, summary_tokens=1024, lookup=None, **kwargs):
        super().__init__(*args, **kwargs)
        if summary_tokens < 1:
            raise ValueError('summary_tokens must be positive')
        self.summary_tokens = summary_tokens
        self.lookup = lookup or (lambda _: None)
        self._summaries = {}

    def summary(self, node):
        if node.id not in self._summaries:
            raw = node.idea.strip()
            original = self.count(raw) if raw else 0
            view, status = raw, 'present' if raw else 'unavailable'
            if original > self.summary_tokens:
                marker = '[Remaining summary paragraphs omitted.]'
                kept = []
                for paragraph in re.split(r'\n\s*\n', raw):
                    candidate = '\n\n'.join([*kept, paragraph, marker])
                    if self.count(candidate) > self.summary_tokens:
                        break
                    kept.append(paragraph)
                view = '\n\n'.join([*kept, marker]) if kept else ''
                status = 'prefix' if kept else 'oversized'
            self._summaries[node.id] = (view, {'raw_tokens': original,
                'view_tokens': self.count(view) if view else 0, 'status': status})
        return self._summaries[node.id]

    def render_history(self, events):
        lines = ['# Development History', HISTORY_GUIDANCE]
        for index, (child, parent) in enumerate(events, 1):
            donor = self.lookup(child.donor_id) if child.donor_id is not None else None
            summary, _ = self.summary(child)
            text = f'Step {index}\nPrevious version fitness: {parent.fitness}\n'
            if donor is not None:
                text += f'Reference algorithm fitness: {donor.fitness}\n'
            text += f'Resulting version fitness: {child.fitness}\nImplementation Summary: '
            lines.append(text + (summary or '[Summary unavailable; refer to implementation.]'))
        return '\n\n'.join(lines)

    def assemble(self, current, operator, donor, events, omitted=None):
        omitted = omitted or set()
        parts = [self.task_contract]
        for title, node, show_summary in [('Current Algorithm', current, current is not None and current.parent_id is None),
                                           ('Reference Algorithm', donor, True)]:
            if node is None:
                continue
            text = f'# {title}\nFitness: {node.fitness}\n\n```python\n{self._code(node)}\n```'
            if show_summary and node.id not in omitted:
                summary, _ = self.summary(node)
                text += '\nImplementation Summary: ' + (summary or '[Summary unavailable; refer to implementation.]')
            parts.append(text)
        if events:
            parts.append(self.render_history(events))
        parts.append('# Algorithm Design Task\n' + INSTRUCTIONS[operator])
        parts.append('# Output\n' + OUTPUT)
        return '\n\n\n'.join(parts)

    def fits(self, current, operator, donor=None):
        omitted = {n.id for n in [current, donor] if n is not None}
        return self.count(self.assemble(current, operator, donor, [], omitted), chat=True) <= self.max_tokens

    def build(self, current, ancestors, operator, donor=None):
        events = formation_events(current, ancestors, self.max_events)
        reasons, omitted = [], set()
        while events and self.count(self.render_history(events)) > self.history_tokens:
            reasons.append(f'history_budget:{events.pop(0)[0].id}')
        while True:
            text = self.assemble(current, operator, donor, events, omitted)
            tokens = self.count(text, chat=True)
            if tokens <= self.max_tokens:
                break
            if events:
                reasons.append(f'context_budget:{events.pop(0)[0].id}')
                continue
            candidate = next((n for n in [donor, current] if n is not None and
                              n.id not in omitted and (n is donor or n.parent_id is None)), None)
            if candidate is None:
                raise ValueError('minimum complete prompt exceeds the model context budget')
            omitted.add(candidate.id)
            reasons.append(f'context_summary:{candidate.id}')
        summaries = {}
        shown = [edge[0] for edge in events]
        shown += [n for n in [donor, current] if n is not None and n.id not in omitted
                  and (n is donor or n.parent_id is None)]
        for n in shown:
            _, info = self.summary(n)
            summaries[str(n.id)] = info
            if info['status'] in ['prefix', 'oversized']:
                reasons.append(f'summary_{info["status"]}:{n.id}')
        return Prompt(text, tokens, tuple(n.id for n, _ in events), tuple(reasons),
                      self.count(self.render_history(events)) if events else 0, summaries)
