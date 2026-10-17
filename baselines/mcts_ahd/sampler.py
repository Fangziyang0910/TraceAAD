from __future__ import annotations

import re
from typing import Tuple

from core import LLM, Function, Program
from .prompt import MAPrompt
from core.code import function_from_source, parse_program


class MASampler:
    def __init__(self, llm: LLM, template_program: str | Program, profiler=None):
        self.llm = llm
        self._template_program = template_program
        self._profiler = profiler

    def get_thought_and_function(
            self,
            task_description: str,
            prompt: str,
            *,
            operator: str | None = None,
            sample_order: int | None = None,
    ) -> Tuple[str, Function]:
        # MCTS-AHD ee9c4f42, Evolution._get_alg: greedy extraction through last return.
        template = parse_program(self._template_program) if isinstance(self._template_program, str) else self._template_program
        name = template.functions[0].name
        output = ('next_node' if name.startswith('select_next_node') else
                  'priority' if name.startswith('priority') else
                  'heuristics_matrix' if name.startswith('heuristics') else
                  'offsprings' if name.startswith('crossover') else
                  'utility_value' if name.startswith('utility') else 'result')
        for attempt in range(5):
            response = self.llm.draw_sample(prompt)
            match = re.search(r'\{(.*?)\}', response, re.DOTALL)
            if match is None:
                return None, None
            thought = match.group(1)
            if not thought:
                token = 'python' if 'python' in response else 'import' if 'import' in response else 'def'
                descriptions = re.findall(r'^.*?(?=' + token + ')', response, re.DOTALL)
                thought = descriptions[0] if descriptions else ''
            codes = re.findall(r'import.*return', response, re.DOTALL)
            if not codes:
                codes = re.findall(r'def.*return', response, re.DOTALL)
            self._log_llm_call(stage='generate', operator=operator, sample_order=sample_order,
                               prompt=prompt, response=response, attempt=attempt,
                               function_parse_success=bool(codes))
            if thought and codes:
                break
        else:
            return None, None
        code = codes[0] + ' ' + output
        function = function_from_source(code, name)
        if function is None:
            return thought, None
        prompt2 = MAPrompt.adapter(task_description, template.functions[0]).get_prompt_refine(code, thought)
        describe = self.llm.draw_sample(prompt2)
        self._log_llm_call(
            stage='thought_alignment',
            operator=operator,
            sample_order=sample_order,
            prompt=prompt2,
            response=describe,
            parsed_thought=thought,
            function_parse_success=True,
        )
        return describe, function

    def _log_llm_call(self, **payload):
        logger = getattr(self._profiler, 'log_llm_call', None)
        if callable(logger):
            logger(llm=self.llm, **payload)
