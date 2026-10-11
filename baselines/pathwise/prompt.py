from __future__ import annotations

import copy
import json
from pathlib import Path
import random

from .graph import PathWiseAction, PathWiseNode
from core import Function

_PROMPTS = Path(__file__).with_name('prompts')
_PHRASES = json.loads((_PROMPTS / 'perturbations.json').read_text())


class PathWisePrompt:
    POLICY_PHRASES = _PHRASES['explorative_phrases']
    WORLD_MODEL_PHRASES = _PHRASES['explorative_phrases_world_model']

    def __init__(self, task_prompt: str, template_function: Function):
        self.task_prompt = task_prompt
        self.template_function = copy.deepcopy(template_function)
        self.func_name = template_function.name
        self.func_desc = (template_function.docstring or '').replace('"""', '').strip()

    def _messages(self, role, **values):
        common = dict(problem_desc=self.task_prompt, func_desc=self.func_desc,
                      func_name=self.func_name, fitness_goal='lower', metric_plural='objective values',
                      metric_name='objective value', metric_name_cap='Objective value')
        return [{'role': 'system', 'content': (_PROMPTS / f'system_{role}.txt').read_text()},
                {'role': 'user', 'content': (_PROMPTS / f'user_{role}.txt').read_text().format(**(common | values))}]

    def function_signature(self, version=2):
        ret = f' -> {self.template_function.return_type}' if self.template_function.return_type else ''
        return f'def {self.func_name}_v{version}({self.template_function.args}){ret}:'

    def blank_function(self, version=2):
        return self.function_signature(version) + '\n    pass'

    @staticmethod
    def perturbation_phrase(phrases, prob):
        return random.choice(phrases) if phrases and random.random() < prob else ''

    @staticmethod
    def _node_code(node, version=None):
        function = copy.deepcopy(node.function)
        function.docstring = ''
        if version is not None:
            function.name = f'{function.name}_v{version}'
        return str(function).strip()

    @classmethod
    def format_state(cls, nodes, *, shuffle=True):
        if not nodes:
            return 'Empty state'
        state = list(nodes)
        if shuffle:
            random.shuffle(state)
        text = (f'I have {len(state)} existing heuristics listed below. For each heuristic, I provide its ID, '
                'description, the heuristics used to derive it with their descriptions and objective values, '
                'its derivation logic, its full code, and its own objective value:\n\n')
        for node in state:
            text += f'ID: {node.node_id}\nDescription: {node.description}\n'
            if node.parents:
                text += 'Heuristics used to derive this candidate:\n'
                text += '\n'.join(f'- {p.node_id} (objective value: {p.score:.3f}): {p.description}' for p in node.parents) + '\n'
            text += f'Derivation logic: {node.rationale}\nCode:\n{cls._node_code(node)}\nObjective value: {node.score:.3f}\n\n'
        return text

    def initialization_prompt(self, index, external_knowledge=''):
        messages = self._messages('initialization', function_signature=self.function_signature())
        if external_knowledge:
            messages[-1]['content'] += '\n' + external_knowledge
        return messages

    def policy_prompt(self, state, policy_reflection, perturbation=''):
        return self._messages('policy', current_state=self.format_state(state),
            reflection_section=f'\nReflection:\n{policy_reflection}\n' if policy_reflection else '',
            explorative_nudge=f' {perturbation}' if perturbation else '',
            parent_constraint=f"\nYou must choose parent ID(s) only from the {len(state)} heuristics listed above ({', '.join(n.node_id for n in state)}).")

    def world_model_prompt(self, action, parent_nodes, world_model_reflection, perturbation=''):
        parents = '\n\n'.join(
            f"ID: {node.node_id} algorithm's description, its corresponding code and its objective value are:\n"
            f'{node.description}\n{self._node_code(node, 1)}\nObjective value: {node.score:.5f}' for node in parent_nodes)
        comparison = 'the provided algorithm' if len(parent_nodes) == 1 else 'all algorithms'
        messages = self._messages('world_model', num_parents=len(parent_nodes), parent_heuristics=parents,
            directive=action.rationale + (f' {perturbation}' if perturbation else ''),
            reflection_section=f'\nReflection:\n{world_model_reflection}\n' if world_model_reflection else '',
            fitness_comparison=f'The new algorithm should have an objective value lower than {comparison}.',
            func_signature_constraint='')
        if not world_model_reflection:
            messages[-1]['content'] = messages[-1]['content'].replace('directive and reflection', 'directive')
        return messages

    def policy_critic_prompt(self, state, actions, action_summaries):
        action_data = []
        by_id = {n.node_id: n for n in state}
        for i, (action, summary) in enumerate(zip(actions, action_summaries)):
            parents = ', '.join(f'{p} (objective value: {by_id[p].score:.5f})' if p in by_id else p for p in action.parents)
            label = 'Parent ID' if len(action.parents) == 1 else 'Parent IDs'
            action_data.append(f'Action {i}\n{label}: {parents}\nDirective: {action.rationale}\n{summary}')
        return self._messages('policy_critic', current_state=self.format_state(state, shuffle=False),
                              action_data='\n\n'.join(action_data))

    def world_model_critic_prompt(self, best, worst):
        return self._messages('world_model_critic', worst_description=worst.description,
            worst_fitness=worst.score, worst_code=self._node_code(worst, 0),
            best_description=best.description, best_fitness=best.score, best_code=self._node_code(best, 1))
