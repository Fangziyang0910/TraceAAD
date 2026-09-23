from __future__ import annotations

import copy
from typing import List

from llm4ad.base import Function


class ReEvoPrompt:
    _GENERATOR_SYSTEM = (
        'You are an expert in the domain of optimization heuristics. '
        'Your task is to design heuristics that can effectively solve optimization problems. '
        'Your response outputs Python code and nothing else. Format your code as a Python code string: "```python ...```"'
    )
    _REFLECTOR_SYSTEM = (
        'You are an expert in the domain of optimization heuristics. '
        'Your task is to give hints to design better heuristics.'
    )

    @classmethod
    def get_pop_init_prompt(cls, task_prompt: str, seed_function: Function) -> str:
        seed_function = copy.deepcopy(seed_function)
        func_name = seed_function.name
        prompt = f'''{task_prompt}
{seed_function}
Refer to the format of a trivial design above. Be very creative and give `{func_name}_v2`. Output code only and enclose your code with Python code block: ```python ...```.
'''
        return '\n'.join([cls._GENERATOR_SYSTEM, prompt])

    @classmethod
    def _worse_better(cls, indivs: List[Function]) -> tuple[Function, Function]:
        assert len(indivs) == 2
        indivs = copy.deepcopy(indivs)
        indivs.sort(key=lambda function: function.score)
        return indivs[0], indivs[1]

    @classmethod
    def get_short_term_reflection_prompt(cls, task_prompt: str, indivs: List[Function]) -> str:
        worse, better = cls._worse_better(indivs)
        prompt = f'''Below are two {better.name} functions for {task_prompt}

You are provided with two code versions below, where the second version performs better than the first one.

[Worse code]
{worse}

[Better code]
{better}

You respond with some hints for designing better heuristics, based on the two code versions and using less than 20 words.'''
        return '\n'.join([cls._REFLECTOR_SYSTEM, prompt])

    @classmethod
    def get_crossover_prompt(cls, task_prompt: str, short_term_reflection_prompt: str, indivs: List[Function]) -> str:
        worse, better = cls._worse_better(indivs)
        func_name = worse.name
        worse.name = f'{worse.name}_v0'
        better.name = f'{better.name}_v1'
        prompt = f'''{task_prompt}

[Worse code]
{worse}

[Better code]
{better}

[Reflection]
{short_term_reflection_prompt}

[Improved code]
Please write an improved function `{func_name}_v2`, according to the reflection. Output code only and enclose your code with Python code block: ```python ...```.
'''
        return '\n'.join([cls._GENERATOR_SYSTEM, prompt])

    @classmethod
    def get_long_term_reflection_prompt(
            cls,
            task_prompt: str,
            prior_long_term_reflection: str,
            new_short_term_reflections: List[str],
    ) -> str:
        new_short_term_reflections = '\n'.join(new_short_term_reflections)
        prompt = f'''Below is your prior long-term reflection on designing heuristics for {task_prompt}
{prior_long_term_reflection}

Below are some newly gained insights.
{new_short_term_reflections}

Write constructive hints for designing better heuristics, based on prior reflections and new insights and using less than 50 words.'''
        return '\n'.join([cls._REFLECTOR_SYSTEM, prompt])

    @classmethod
    def get_elist_mutation_prompt(cls, task_prompt: str, long_term_reflection_prompt: str, elite_function: Function) -> str:
        elite_function = copy.deepcopy(elite_function)
        func_name = elite_function.name
        elite_function.name = f'{elite_function.name}_v1'
        prompt = f'''{task_prompt}

[Prior reflection]
{long_term_reflection_prompt}

[Code]
{elite_function}

[Improved code]
Please write a mutated function `{func_name}_v2`, according to the reflection. Output code only and enclose your code with Python code block: ```python ...```.
'''
        return '\n'.join([cls._GENERATOR_SYSTEM, prompt])
