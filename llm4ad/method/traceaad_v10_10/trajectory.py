"""Unified short formation path for normal search generation."""

import ast

from llm4ad.method.traceaad_v10_8.trajectory import TrajectoryBuilder as BaseBuilder, digest
from .errors import OUTPUT

GENERATION = 'target_function_idea500_template_rebuild_one_repair_v3'

CONTEXT_POLICY = 'unified_short_formation_path_compact_code_v2'
INITIALIZATION_POLICY = 'sequential_informed_v1'
INSTRUCTIONS = {
    'Init': (
        'Study the task and the previously evaluated initial algorithms when available. '
        'Design a competitive candidate around a promising decision mechanism.'
    ),
    'Refine': (
        'Build on the current decision method. Use its code and formation history to '
        'identify the most valuable next improvement and implement it as one coherent refinement.'
    ),
    'Tune': (
        'Preserve the current decision method and computational structure. Calibrate a '
        'small coherent set of influential coefficients, thresholds, exponents, or schedules '
        'to improve its performance.'
    ),
    'Pivot': (
        'Use the task, current algorithm, and formation history to develop a competitive '
        'alternative built around a different primary decision mechanism.'
    ),
    'Fuse': (
        'Identify a host limitation that donor computations can address, then adapt and '
        'integrate the relevant computations into one coherent host-centered algorithm '
        'that aims to outperform both inputs.'
    ),
}
HISTORY_TITLE = '# Formation History of the Current Algorithm — oldest to newest'
HISTORY_NOTE = (
    'Use the sequence of design ideas and observed fitness changes to understand '
    'how the current algorithm developed and guide this design.'
)
TEMPLATE_HASH = digest(str(INSTRUCTIONS) + HISTORY_TITLE + HISTORY_NOTE + OUTPUT)


class TrajectoryBuilder(BaseBuilder):
    def __init__(self, *args, all_nodes, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_nodes = all_nodes

    def formation_edges(self, current):
        """Recent formation edges of the current node, oldest first."""
        edges = []
        for _ in range(self.max_events):
            if current is None or current.parent_id is None:
                break
            source = self.lookup(current.parent_id)
            if source is None:
                raise ValueError('missing archived formation predecessor')
            edges.append((source, current))
            current = source
        return list(reversed(edges))

    def function_view(self, node):
        try:
            tree = ast.parse(node.code)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f'cannot parse target function for node {node.id}') from exc
        functions = [item for item in tree.body
                     if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1:
            raise ValueError(f'cannot extract target function for node {node.id}')
        # ast.unparse removes generated comments while retaining executable behavior.
        return ast.unparse(functions[0]).strip()

    def render_history(self, edges):
        blocks = []
        for step, (source, target) in enumerate(edges, start=1):
            lines = [f'Step {step} | {target.operator} | '
                     f'Fitness: {source.fitness} -> {target.fitness}']
            if target.idea:
                # A step without a description keeps its operator and scores;
                # no description is invented for it.
                lines.append('Idea: ' + ' '.join(target.idea.split()))
            blocks.append('\n'.join(lines))
        return '\n\n'.join(blocks)

    def program(self, node, title):
        return f'# {title}\nFitness: {node.fitness}\n```python\n{self.function_view(node)}\n```'

    def assemble(self, parent, operator, donor, history=''):
        parts = [self.task_contract]
        if parent is not None:
            title = 'Host Algorithm' if operator == 'Fuse' else 'Current Algorithm'
            parts.append(self.program(parent, title))
        if history:
            parts.append(f'{HISTORY_TITLE}\n{HISTORY_NOTE}\n\n{history}')
        if donor is not None:
            parts.append(self.program(donor, 'Donor Algorithm'))
        parts.extend(['# Design Task\n' + INSTRUCTIONS[operator], '# Output\n' + OUTPUT])
        return '\n\n\n'.join(parts)

    def build_initial(self):
        # Sequential informed initialization: the first root sees only the task;
        # each later root sees every previously evaluated root, raw code and fitness.
        roots = sorted((n for n in self.all_nodes() if n.parent_id is None), key=lambda n: n.id)
        parts = [self.task_contract]
        if roots:
            parts.append('# Previous Initial Algorithms\n'
                         'Complete previously evaluated programs, in generation order. '
                         'Fitness: higher is better.')
            parts.extend(f'# Previous Initial Algorithm\n'
                         f'Fitness: {n.fitness}\n'
                         f'```python\n{self.function_view(n)}\n```' for n in roots)
        parts.extend(['# Design Task\n' + INSTRUCTIONS['Init'], '# Output\n' + OUTPUT])
        text = '\n\n\n'.join(parts)
        self.check_capacity(text)
        return text, {}

    def build(self, parent, operator, donor=None):
        if operator == 'Init':
            return self.build_initial()
        edges = self.formation_edges(parent)
        text = self.assemble(parent, operator, donor, self.render_history(edges))
        self.check_capacity(text)
        scores = [n.fitness for n in (parent, donor) if n is not None]
        scores += [fitness for s, t in edges for fitness in (s.fitness, t.fitness)]
        return text, {
            'history_edges': [[s.id, t.id] for s, t in edges],
            'context_node_ids': [n.id for n in (parent, donor) if n is not None],
            'context_best_fitness': max(scores, default=None),
        }

    def check_capacity(self, text):
        if self.count(text, chat=True) > self.max_tokens:
            raise ValueError('complete prompt exceeds the model context budget')
