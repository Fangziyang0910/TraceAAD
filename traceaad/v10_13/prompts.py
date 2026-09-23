"""Compact idea/fitness evidence, optional reads, and full/edit proposals."""

from dataclasses import dataclass, field

from .parsing import FULL_OUTPUT_FORMAT, OUTPUT_FORMAT, code_hash

OPERATOR_INSTRUCTIONS = {
    'Init': (
        'Develop a competitive algorithm using the task structure and your algorithmic knowledge. '
        'Use previous examples, when available, to explore another promising decision hypothesis. '
        'Reuse useful components without making cosmetic difference or novelty the objective.'
    ),
    'Refine': (
        'Treat the current algorithm as a working design. Find a promising improvement to how it '
        'represents the remaining problem, compares alternatives, or coordinates its components. '
        'Retain computations that remain useful. A focused edit is welcome; a coherent rewrite is allowed.'
    ),
    'Tune': (
        'Preserve the main decision method. Calibrate influential parameters, thresholds, weights, '
        'or schedules, including their interactions and dependence on the input or search state. '
        'Prefer changes that affect downstream decisions or sampling over changes that cancel out. '
        'An exact local edit can preserve the surrounding implementation.'
    ),
    'Pivot': (
        'Reconsider a central assumption of the current approach. Use your algorithmic knowledge '
        'to develop a competitive alternative in problem representation, decision rule, or search '
        'organization. Reference ideas are optional inspiration, not templates to copy. '
        'Look for a useful different principle rather than the highest-scoring example. '
        'Useful components may survive; novelty alone is not the objective.'
    ),
    'Fuse': (
        'Use the host as a starting point. Look for a reference computation that addresses a host '
        'weakness or supplies a complementary signal, decision principle, or search organization. '
        'Adapt its scale, assumptions, and interactions into a coherent algorithm aiming to outperform '
        'both inputs. Choose for complementary ideas, not score alone. Overall reference fitness '
        'does not measure the value of each component. '
        'Do not mechanically concatenate programs. If no reference is useful, improve the host '
        'without forced mixing. Read one reference implementation if needed for integration.'
    ),
}

EVIDENCE_NOTE = (
    'Fitness is measured for the whole program; higher is better. Idea fields are author descriptions, '
    'not verified explanations or causal component scores. Missing descriptions mean unknown. '
    'Design changes should reach computations used in the returned result and work across valid inputs.'
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
        if node.idea_fields:
            return '\n'.join(f'{key}: {brief(value)}' for key, value in node.idea_fields.items()
                             if key in ('mechanism', 'change', 'transfer') and value)
        return 'Idea: ' + (brief(node.idea, 480) if node.idea else '[not recorded]')

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
        if parent is not None:
            parts = parts + [f'# Exact edit base\nbase_hash: {code_hash(parent.code)}']
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

    def build_development(self, parent, operator, donor=None, *, references=None,
                          read_reference=None, include_trials=False, allow_context=True):
        references = list(references if references is not None else ([donor] if donor else []))
        edges = self.formation_edges(parent) if operator in ('Refine', 'Tune') else []
        local_operator = operator in ('Refine', 'Tune')
        trials = self.local_trials(parent, operator) if include_trials and local_operator else []
        available_trials = len(trials)
        fallback = None
        while True:
            parts = [self.task_contract, EVIDENCE_NOTE,
                     self.program(parent, 'Host Algorithm' if operator == 'Fuse' else 'Current Algorithm')]
            if edges:
                parts.append(self._history_text(edges))
            if trials:
                parts.append(self._trial_text(parent, trials))
            elif include_trials and local_operator and not available_trials:
                parts.append('# Local trials\nNo evaluated child records are available for this parent.')
            if references:
                parts.append('# Optional reference ideas — unranked, not algorithm classes\n' +
                             '\n\n'.join(self.card(node) for node in references))
            if read_reference is not None:
                parts.append(self.program(read_reference, 'Requested Reference Implementation'))
            if fallback:
                parts.append('Context note: ' + fallback + '. Continue with available evidence.')
            if allow_context and (local_operator or references):
                ids = [node.id for node in references]
                request = (
                    '{"mode":"context","trials":true} retrieves at most two compact Idea/fitness '
                    'records of evaluated children of this exact parent: the best improvement and '
                    'the latest non-improvement, when available. These are whole-program outcomes; '
                    'they do not establish why a change worked. No historical code or diff is returned.'
                    if local_operator else
                    '{"mode":"context","reference_id":ID} retrieves one complete reference '
                    'implementation. Choose an ID from ' + str(ids) +
                    ' if its idea looks useful for the current design; otherwise ignore the references.'
                )
                parts.append('# Optional context read\nIf additional evidence would help, return '
                             'one JSON request instead of a proposal. ' + request +
                             '\nOnly one read round is available; otherwise submit full/edit now.')
            elif not allow_context:
                parts.append('The optional read round is complete. Submit a full/edit proposal now.')
            prompt = self._join(parts, operator, parent)
            if self.count(prompt) <= self.max_tokens:
                return PromptContext(prompt, operator, [n.id for n in references], read_reference,
                                     fallback, [child.id for _, child in edges], [n.id for n in trials])
            if edges:
                edges.pop(0)
            elif trials:
                trials.pop()
                fallback = 'local_trial_records_trimmed'
            elif references:
                references.pop()
                fallback = 'reference_cards_trimmed'
            elif read_reference is not None:
                read_reference = None
                fallback = 'requested_implementation_exceeds_context'
            else:
                raise ValueError('task, parent and output instructions exceed context capacity')

    def build(self, parent, operator, donor=None):
        return self.build_development(parent, operator, donor).prompt
