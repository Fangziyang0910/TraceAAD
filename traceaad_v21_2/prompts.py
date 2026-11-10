"""V21.2 output protocol: design reasoning set by the operator instruction, then code, then a description of that code."""

from traceaad_v21_1.prompts import CANDIDATE_RESPONSE_FORMAT, PromptBuilder as V21PromptBuilder

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
- "analysis": Before writing the code, briefly explain your design reasoning according to [Instruction], in 2–4 sentences and at most 120 words.
- "code": Provide the complete implementation of the required function, including necessary imports and helper definitions. Do not include comments or docstrings.
- "design": After writing the code, describe what the code actually computes in one self-contained paragraph of 2–3 concise sentences, using at most 100 English words or 200 Chinese characters. State its representation, main computations, and how their results determine the output. Include the formulas or parameter values needed to understand the core mechanism. If the code differs from the analysis, describe the code.
Preserve the required function signature and satisfy the task constraints and input/output requirements.
Return only the JSON object, without Markdown fences or additional text."""

__all__ = ["CANDIDATE_RESPONSE_FORMAT", "COMMON_TEMPLATE", "PromptBuilder"]


class PromptBuilder(V21PromptBuilder):
    """Initialization and repair as in V21.1; only the analysis field changes."""

    common_template = COMMON_TEMPLATE
