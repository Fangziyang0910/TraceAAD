"""V21 output protocol: think in analysis, implement in code, then describe that code in design."""

from dataclasses import replace

from traceaad_base.prompts import PromptBuilder as BasePromptBuilder

from .representation import bare

COMMON_TEMPLATE = """{task_description}

Implement the following Python function:
```python
{function_template}
```

[Context]
{context}

[Instruction]
{instruction}

[Output]
Return exactly one valid JSON object with three string fields, in this order:
- "analysis": Before writing the code, in 2–3 concise sentences, using at most 100 English words or 200 Chinese characters, explain the approach you will implement and why it suits the task.
- "code": Provide the complete implementation of the required function, including necessary imports and helper definitions. Do not include comments or docstrings.
- "design": After writing the code, describe what the code actually computes in one self-contained paragraph of 2–3 concise sentences, using at most 100 English words or 200 Chinese characters. State its representation, main computations, and how their results determine the output. Include the formulas or parameter values needed to understand the core mechanism. If the code differs from the analysis, describe the code.
Preserve the required function signature and satisfy the task constraints and input/output requirements.
Return only the JSON object, without Markdown fences or additional text."""

CANDIDATE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "traceaad_candidate",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "code": {"type": "string"},
                "design": {"type": "string"},
            },
            "required": ["analysis", "code", "design"],
            "additionalProperties": False,
        },
    },
}


class PromptBuilder(BasePromptBuilder):
    """Share current Base initialization and repair; retain mechanism 2's output protocol."""

    common_template = COMMON_TEMPLATE

    def initialization(self, archive):
        return super().initialization({i: replace(p, code=bare(p.code)) for i, p in archive.items()})
