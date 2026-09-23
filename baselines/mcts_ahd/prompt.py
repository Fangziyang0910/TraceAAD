from __future__ import annotations

import copy
from typing import List, Dict

from core import *


class MAPrompt:
    @classmethod
    @classmethod
    def get_system_prompt(cls) -> str:
        return ''


    @classmethod
    def _template_stub(cls, template_function: Function):
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        return temp_func

    @classmethod
    def _function_contract(cls, template_function: Function) -> str:
        inputs = [
            arg.strip().split(':', 1)[0].split('=', 1)[0].strip()
            for arg in template_function.args.split(',')
            if arg.strip()
        ]
        if len(inputs) == 0:
            joined_inputs = "no input"
        elif len(inputs) == 1:
            joined_inputs = f"'{inputs[0]}'"
        else:
            joined_inputs = ", ".join(f"'{arg}'" for arg in inputs)
        if template_function.name.startswith('select_next_node'):
            output_name = 'next_node'
        elif template_function.name.startswith('priority'):
            output_name = 'priority'
        elif template_function.name.startswith('heuristics'):
            output_name = 'heuristics_matrix'
        elif template_function.name.startswith('crossover'):
            output_name = 'offsprings'
        elif template_function.name.startswith('utility'):
            output_name = 'utility_value'
        else:
            output_name = 'result'
        return (
            f"implement it in Python as a function named '{template_function.name}'.\n"
            f"This function should accept {len(inputs)} input(s): {joined_inputs}. "
            f"The function should return 1 output(s): '{output_name}'. "
            "Use the following function signature:\n"
        )

    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function):
        # template
        temp_func = cls._template_stub(template_function)
        # create prompt content
        prompt_content = f'''{task_prompt}
First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside a brace outside the code implementation.
Next, {cls._function_contract(template_function)}
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = cls._template_stub(template_function)
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f"No.{i + 1} algorithm's description, its corresponding code and its objective value are:\n{indi.algorithm}\n{str(indi)}\nObjective value: {str(-indi.score)}\n\n"
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:

{indivs_prompt}
Please create a new algorithm that has a totally different form from the given algorithms. Try generating codes with different structures, flows or algorithms. The new algorithm should have a relatively low objective value.
First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside a brace outside the code implementation.
Next, {cls._function_contract(template_function)}
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = cls._template_stub(template_function)
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f"No.{i + 1} algorithm's description, its corresponding code and its objective value are:\n{indi.algorithm}\n{str(indi)}\nObjective value: {str(-indi.score)}\n\n"
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes and objective values as follows:

{indivs_prompt}
Please create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have a objective value lower than both algorithms.
Firstly, list the common ideas in the No.{1} algorithm that may give good performances. Secondly, based on the common idea, describe the design idea based on the No.{len(indivs)} algorithm and main steps of your algorithm in one sentence. The description must be inside a brace. Thirdly, {cls._function_contract(template_function)}
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = cls._template_stub(template_function)

        # create prmpt content
        prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows.

Algorithm's description:
{indi.algorithm}
Code:
{str(indi)}
Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.
First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. The description must be inside a brace outside the code implementation.
Next, {cls._function_contract(template_function)}
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = cls._template_stub(template_function)
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows.

Algorithm's description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm.
First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. The description must be inside a brace outside the code implementation.
Next, {cls._function_contract(template_function)}
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_s1(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = cls._template_stub(template_function)
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f"No.{i + 1} algorithm's description, its corresponding code and its objective value are:\n{indi.algorithm}\n{str(indi)}\nObjective value: {str(-indi.score)}\n\n"
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes and objective values as follows:

{indivs_prompt}
Please help me create a new algorithm that is inspired by all the above algorithms with its objective value lower than any of them.
Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. The description must be inside a brace. Thirdly, {cls._function_contract(template_function)}
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content
