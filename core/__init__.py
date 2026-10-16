"""Program representations, model requests and isolated evaluation."""

from .code import Function, Program, parse_function, parse_program, replace_function
from .evaluate import (
    Evaluation,
    EvaluationOutcome,
    InvalidEvaluationResult,
    SecureEvaluator,
    set_kill_with_parent,
)
from .llm import LLM, OpenAIAPI, TokenizationError
