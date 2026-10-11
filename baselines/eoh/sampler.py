from __future__ import annotations

import re
from typing import Any, Callable, Tuple

from core import LLM, Function, Program

from baselines.shared.sampling import SampleTrimmer


def log_sampler_llm_call(profiler: Any, **payload) -> None:
    logger = getattr(profiler, "log_llm_call", None)
    if callable(logger):
        logger(**payload)


def trim_braced_thought(response: str) -> str | None:
    try:
        bracketed_texts = re.findall(r"\{.*?\}", response)
        return bracketed_texts[0]
    except Exception:
        pass
    # Fallback: 当 LLM 不输出花括号格式的思想时（如关闭思考的模型直接输出
    # 纯代码），用函数 docstring 作为思想的替代，避免样本被误拒。
    try:
        doc_match = re.search(r'"""(.*?)"""', response, re.S)
        if doc_match:
            return doc_match.group(1).strip()
    except Exception:
        pass
    return None


def sample_thought_and_function(
        llm: Any,
        prompt: str,
        template_program: Any,
        *,
        profiler: Any = None,
        operator: str | None = None,
        sample_order: int | None = None,
        stage: str = "generate",
        attach_entire_code: bool = False,
        postprocess: Callable[[Any], Any] | None = None,
):
    response = llm.draw_sample(prompt)
    thought = trim_braced_thought(response)
    code = response
    function = SampleTrimmer.sample_to_function(code, template_program)

    if function is not None and attach_entire_code:
        function.entire_code = str(SampleTrimmer.sample_to_program(code, template_program))
    if function is not None and postprocess is not None:
        function = postprocess(function)

    log_sampler_llm_call(
        profiler,
        llm=llm,
        stage=stage,
        operator=operator,
        sample_order=sample_order,
        prompt=prompt,
        response=response,
        parsed_thought=thought,
        thought_parse_success=thought is not None,
        function_parse_success=function is not None,
    )
    return thought, function


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
