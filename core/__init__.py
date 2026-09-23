"""Shared core from the LLM4AD platform (kept with attribution):

- ``core.code``: code representation (Function/Program/TextFunctionProgramConverter).
- ``core.evaluate``: secure multiprocess evaluation (Evaluation/SecureEvaluator).
- ``core.llm``: LLM client interface (LLM/OpenAIAPI).

Methods and tasks live in the top-level ``traceaad``, ``baselines`` and
``benchmarks`` packages.
"""

from .code import (
    Function,
    Program,
    TextFunctionProgramConverter
)
from .evaluate import (
    Evaluation,
    EvaluationOutcome,
    InvalidEvaluationResult,
    SecureEvaluator,
    set_kill_with_parent,
)
from .llm import LLM, OpenAIAPI, TokenizationError
