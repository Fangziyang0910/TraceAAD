"""Small deterministic stand-ins shared by the V10.13 method and experiment tests."""

from core import Evaluation


class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(
            template_program="def score(x):\n    pass",
            task_description="Return a numeric score.",
            safe_evaluate=False,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1)


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def count_prompt_tokens(self, text):
        return len(text.split())

    def draw_sample_with_details(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {
            "content": next(self.responses),
            "finish_reason": "stop",
            "usage": {},
            "model": "test",
        }


def response(value):
    return f"Idea: return {value}\n```python\ndef score(x):\n    return {value}\n```"
