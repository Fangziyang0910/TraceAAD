"""Single-pass proposals with small, operator-specific evidence."""

from dataclasses import dataclass, field

from .parsing import FULL_OUTPUT_FORMAT, OUTPUT_FORMAT

OPERATOR_INSTRUCTIONS = {
    'Init': (
        'Develop a competitive algorithm using the task structure and your algorithmic knowledge. '
        'Use previous examples to explore another promising decision principle, rather than '
        'another formula producing the same choices.'
    ),
    'Refine': (
        'Treat the current algorithm as a working design. Find a promising improvement to how it '
        'represents the remaining problem, compares alternatives, or coordinates its components. '
        'Retain useful computations and consider where the new design would make a better decision. '
        'A focused edit or a coherent rewrite is welcome.'
    ),
    'Tune': (
        'Preserve the main decision method. Calibrate influential parameters, thresholds, weights, '
        'or schedules, including their interactions and dependence on the input or search state. '
        'Choose scales that can change decisions or sampling; a strictly increasing rescaling of an argmax '
        'score leaves its choices unchanged.'
    ),
    'Pivot': (
        'Reconsider a central assumption of the current approach. Use your algorithmic knowledge '
        'to develop a competitive alternative in problem representation, decision rule, or search '
        'organization. Seek a useful alternative decision principle, not cosmetic variation. '
        'Useful components may survive.'
    ),
    'Fuse': (
        'Improve the host using a reference computation that addresses a weakness or supplies '
        'a complementary signal or decision principle. Adapt its scale and interactions so it '
        'can improve actual decisions, aiming to outperform both programs. Do not mechanically '
        'concatenate them. If the reference adds nothing useful, improve the host directly.'
    ),
}

EVIDENCE_NOTE = (
    'Fitness measures the whole program; higher is better. Ideas describe intended changes, '
    'not verified explanations. Use the code and measured outcomes to guide your design.'
)


def brief(text, limit=280):
    text = ' '.join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + ' … [description shortened]'


@dataclass
class PromptContext:
    prompt: str
    operator: str
    reference_ids: list[int]
    reference_program: object | None = None
    fallback_reason: str | None = None
    history_ids: list[int] = field(default_factory=list)
    trial_ids: list[int] = field(default_factory=list)


class PromptBuilder:
    def __init__(self, llm, task_contract, *, max_tokens, history_depth, lookup, all_nodes):
        self.llm, self.task_contract = llm, task_contract
        self.max_tokens, self.history_depth = max_tokens, history_depth
        self.lookup, self.all_nodes = lookup, all_nodes

    def count(self, text):
        return self.llm.count_prompt_tokens(text)

    def formation_edges(self, current):
        edges = []
        for _ in range(self.history_depth):
            if current is None or current.parent_id is None:
                break
            source = self.lookup(current.parent_id)
            if source is None:
                raise ValueError('missing formation predecessor')
            edges.append((source, current))
            current = source
        return list(reversed(edges))

    def idea_view(self, node):
        text = node.idea or node.idea_fields.get('mechanism') or node.idea_fields.get('change', '')
        return 'Idea: ' + (brief(text, 480) if text else '[not recorded]')

    def card(self, node):
        return f'Node {node.id} | measured fitness: {node.fitness}\n{self.idea_view(node)}'

    def program(self, node, title):
        return f'# {title}\n{self.card(node)}\n```python\n{node.code}\n```'

    def _history_text(self, edges):
        lines = ['# Recent Design History — formation, oldest first']
        for source, target in edges:
            lines.append(f'Node {source.id} -> {target.id} | {target.operator} | '
                         f'whole-program fitness {source.fitness} -> {target.fitness}\n'
                         + self.idea_view(target))
        return '\n\n'.join(lines)

    def local_trials(self, parent, operator):
        children = [node for node in self.all_nodes() if node.parent_id == parent.id]
        if operator == 'Tune':
            preferred = [node for node in children if node.operator == 'Tune']
            if preferred:
                children = preferred
        good = [node for node in children if node.fitness > parent.fitness]
        other = [node for node in children if node.fitness <= parent.fitness]
        selected = ([max(good, key=lambda n: (n.fitness, n.id))] if good else [])
        selected += [max(other, key=lambda n: n.id)] if other else []
        return selected

    def _trial_text(self, parent, trials):
        return '# Local trials — measured children of this exact parent\n' + '\n\n'.join(
            f'Node {parent.id} -> {node.id} | {node.operator} | '
            f'whole-program fitness {parent.fitness} -> {node.fitness}\n{self.idea_view(node)}'
            for node in trials)

    def _join(self, parts, operator, parent=None):
        return '\n\n'.join(parts + ['# Design Task\n' + OPERATOR_INSTRUCTIONS[operator],
                                     '# Output\n' + (OUTPUT_FORMAT if parent else FULL_OUTPUT_FORMAT)])

    def build_initial_context(self):
        roots = sorted((n for n in self.all_nodes() if n.parent_id is None), key=lambda n: n.id)
        retained = list(roots)
        while True:
            parts = [self.task_contract, EVIDENCE_NOTE]
            parts += [self.program(node, 'Previous Initial Algorithm') for node in retained]
            prompt = self._join(parts, 'Init')
            if self.count(prompt) <= self.max_tokens:
                return PromptContext(prompt, 'Init', [n.id for n in retained],
                                     fallback_reason='initial_examples_trimmed' if retained != roots else None)
            if not retained:
                raise ValueError('task and output instructions exceed context capacity')
            retained.pop(0)

    def build_initial(self):
        return self.build_initial_context().prompt

    def build_development(self, parent, operator, donor=None, *, references=None):
        references = list(references if references is not None else ([donor] if donor else []))
        reference = references[0] if operator == 'Fuse' and references else None
        edges = self.formation_edges(parent) if operator in ('Refine', 'Tune') else []
        trials = self.local_trials(parent, operator) if operator in ('Refine', 'Tune') else []
        fallback = None
        while True:
            parts = [self.task_contract, EVIDENCE_NOTE,
                     self.program(parent, 'Host Algorithm' if operator == 'Fuse' else 'Current Algorithm')]
            if edges:
                parts.append(self._history_text(edges))
            if trials:
                parts.append(self._trial_text(parent, trials))
            if reference is not None:
                parts.append(self.program(reference, 'Reference Algorithm'))
            if fallback:
                parts.append('Context note: ' + fallback + '. Continue with available evidence.')
            prompt = self._join(parts, operator, parent)
            if self.count(prompt) <= self.max_tokens:
                return PromptContext(prompt, operator, [reference.id] if reference else [], reference,
                                     fallback, [child.id for _, child in edges], [n.id for n in trials])
            if edges:
                edges.pop(0)
            elif trials:
                trials.pop()
                fallback = 'local_trial_records_trimmed'
            elif reference is not None:
                reference = None
                fallback = 'reference_implementation_exceeds_context'
            else:
                raise ValueError('task, parent and output instructions exceed context capacity')

    def build(self, parent, operator, donor=None):
        return self.build_development(parent, operator, donor).prompt
