"""One generation call; history may inform a change without becoming a separate reasoning task."""

from traceaad_v21_1.prompts import CANDIDATE_RESPONSE_FORMAT, PromptBuilder as V21PromptBuilder

EVOLUTION_TEMPLATE = """{task_description}

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
- "analysis": Before writing the code, briefly state the computation you will change or introduce. Available evolution records may inform your choice. Use 2–3 concise sentences and at most 100 English words or 200 Chinese characters.
- "code": Provide the complete implementation of the required function, including necessary imports and helper definitions. Do not include comments or docstrings.
- "design": After writing the code, describe what the code actually computes in one self-contained paragraph of 2–3 concise sentences, using at most 100 English words or 200 Chinese characters. State its representation, main computations, and how their results determine the output. Include the formulas or parameter values needed to understand the core mechanism. If the code differs from the analysis, describe the code.
Preserve the required function signature and satisfy the task constraints and input/output requirements.
Return only the JSON object, without Markdown fences or additional text."""


class PromptBuilder(V21PromptBuilder):
    """Initialization and repair retain their existing prompts; only evolution uses this template."""

    def evolution_render(self, context, instruction, *, has_scores):
        return self._render(context, instruction, has_scores=has_scores, template=EVOLUTION_TEMPLATE)
