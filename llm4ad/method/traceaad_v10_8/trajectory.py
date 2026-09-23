"""Deterministic complete-code evidence from one real formation suffix."""

from difflib import unified_diff
import hashlib

from llm4ad.method.traceaad_v10_7.prompts import (
    OUTPUT, GENERATION, TrajectoryBuilder as ViewBuilder,
)

CONTEXT_POLICY = 'verified_formation_suffix_v1'
INSTRUCTIONS = {
    'Init': 'Design one competitive, coherent algorithm for the task.',
    'Refine': 'Improve the current implementation around one main improvement '
              'hypothesis. Use its formation evidence when available. Preserve '
              'unrelated computations unless coherent changes require otherwise.',
    'Pivot': 'Use the current implementation and its formation evidence to '
             'understand its main assumptions. Try another competitive main '
             'decision method. History is a comparison basis, not a blacklist; '
             'reuse useful computations when appropriate.',
    'Fuse': 'Use the donor as a source of useful computations to improve the '
            'current implementation. Produce one coherent algorithm aiming to '
            'outperform the better input; no fixed contribution ratio is required.',
}
HISTORY_NOTE = (
    '# Recent Formation Transitions\n'
    'These are consecutive actual transitions, oldest first, ending at the '
    'current implementation. Diffs run from source to target; full predecessor '
    'code is used when shorter. Fitness describes the whole measured transition, '
    'not the causal contribution of an individual change.'
)
TEMPLATE_HASH = hashlib.sha256(
    (str(INSTRUCTIONS) + HISTORY_NOTE + OUTPUT).encode()
).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class TrajectoryBuilder(ViewBuilder):
    def __init__(self, *args, lookup, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookup = lookup
        self._transitions = {}

    def view_record(self, node):
        code, removed = self.code_view(node)
        return {
            'node_id': node.id, 'raw_code_hash': digest(node.code),
            'view_code_hash': digest(code), 'removed_reference_comments': removed,
        }

    def program(self, node, title):
        code, _ = self.code_view(node)
        return f'# {title}\nFitness: {node.fitness}\n```python\n{code}\n```'

    def transition(self, source, target):
        key = (source.id, target.id)
        if key not in self._transitions:
            if target.parent_id != source.id:
                raise ValueError('formation evidence must be a real direct edge')
            before, _ = self.code_view(source)
            after, _ = self.code_view(target)
            if before == after:
                kind = 'unchanged'
                change = ('Code unchanged.' if source.code == target.code else
                          'Code unchanged in the cleaned prompt view.')
            else:
                # Preserve every hunk, including a missing final newline.
                diff = ''.join(
                    line if line.endswith('\n') else line + '\n\\ No newline at end of file\n'
                    for line in unified_diff(
                        before.splitlines(keepends=True), after.splitlines(keepends=True),
                        fromfile='source.py', tofile='target.py',
                    )
                )
                patch = f'Complete source-to-target diff:\n```diff\n{diff}```'
                predecessor = f'Complete predecessor code:\n```python\n{before}\n```'
                kind, change = (('diff', patch) if self.count(patch) <= self.count(predecessor)
                                else ('predecessor', predecessor))
            text = (f'Executed operator: {target.operator}\n'
                    f'Fitness: {source.fitness} -> {target.fitness}\n')
            if target.donor_id is not None:
                text += ('An additional donor participated in this multi-input transition; '
                         'its code is not included in this transition block.\n')
            text += change
            facts = {
                'source_id': source.id, 'target_id': target.id,
                'source_evaluation_id': source.evaluation_id,
                'target_evaluation_id': target.evaluation_id,
                'operator': target.operator, 'source_fitness': source.fitness,
                'target_fitness': target.fitness, 'historical_donor_id': target.donor_id,
                'representation': kind, 'text_hash': digest(text),
            }
            self._transitions[key] = text, facts
        return self._transitions[key]

    def assemble(self, parent, operator, donor, edges=()):
        parts = [self.task_contract]
        if parent is not None:
            parts.append(self.program(parent, 'Current Implementation'))
        if edges:
            parts.append(self.history_text(edges))
        if donor is not None:
            parts.append(self.program(donor, 'Donor'))
        parts.extend(['# Design Task\n' + INSTRUCTIONS[operator], '# Output\n' + OUTPUT])
        return '\n\n\n'.join(parts)

    def history_text(self, edges):
        return HISTORY_NOTE + '\n\n' + '\n\n'.join(
            self.transition(source, target)[0] for source, target in edges
        )

    def fits(self, current, operator, donor=None):
        return self.count(self.assemble(current, operator, donor), chat=True) <= self.max_tokens

    def build(self, parent, operator, donor=None):
        edges = []
        cursor = parent
        reason = 'initialization' if parent is None else 'root' if parent.parent_id is None else None
        if parent is not None and parent.parent_id is not None and self.max_events == 0:
            reason = 'history_disabled'
        for _ in range(self.max_events):
            if cursor is None or cursor.parent_id is None:
                break
            source = self.lookup(cursor.parent_id)
            if source is None:
                raise ValueError('missing archived formation predecessor')
            edges.insert(0, (source, cursor))
            cursor = source
        else:
            if edges and cursor.parent_id is not None:
                reason = 'edge_limit'
        # Check only complete candidate suffixes. When everything fits, there
        # is one history count and one chat count, regardless of edge depth.
        while edges and self.count(self.history_text(edges)) > self.history_tokens:
            edges.pop(0)
            reason = 'history_budget'
        while True:
            text = self.assemble(parent, operator, donor, edges)
            if self.count(text, chat=True) <= self.max_tokens:
                break
            if not edges:
                raise ValueError('minimum complete prompt exceeds the model context budget')
            edges.pop(0)
            reason = 'context_budget'
        shown = {n.id: n for edge in edges for n in edge}
        for node in (parent, donor):
            if node is not None:
                shown[node.id] = node
        return text, {
            'history_ids': [target.id for _, target in edges],
            'history_edge_count': len(edges),
            'history_tokens': self.count(self.history_text(edges)) if edges else 0,
            'history_stop_reason': reason,
            'history_fallback_reason': reason if not edges else None,
            'evidence_relations': [self.transition(*edge)[1] for edge in edges],
            'context_node_ids': list(shown),
            'context_code_views': [self.view_record(n) for n in shown.values()],
            'context_best_fitness': max((n.fitness for n in shown.values()), default=None),
        }
