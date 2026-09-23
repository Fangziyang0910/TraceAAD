from __future__ import annotations

from typing import Tuple

from .sampling import sample_thought_and_function, trim_braced_thought
from core import LLM, Function, Program


class EoHSampler:
    def __init__(self, llm: LLM, template_program: str | Program, profiler=None):
        self.llm = llm
        self._template_program = template_program
        self._profiler = profiler

    def get_thought_and_function(self, prompt: str, *, operator=None, sample_order=None) -> Tuple[str, Function]:
        return sample_thought_and_function(
            self.llm,
            prompt,
            self._template_program,
            profiler=self._profiler,
            operator=operator,
            sample_order=sample_order,
        )

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        return trim_braced_thought(response)
