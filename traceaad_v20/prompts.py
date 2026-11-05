"""V20 prompt texts: Base prompts without word caps and with short failure feedback."""

import re

from traceaad_base.prompts import PromptBuilder as BasePromptBuilder

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
- "analysis": Briefly explain the algorithmic rationale of the proposed change, the evidence that motivates it, and its expected effect on solving the task.
- "design": Describe the final algorithm implemented in "code" in one short self-contained paragraph. State the solution representation, the main solving principle, and how the procedure determines the returned result.
- "code": Provide the complete implementation of the required function, including necessary imports and helper definitions. Do not include comments or docstrings.
First determine a complete executable algorithm; then make "analysis" and "design" accurately describe the final "code".
Preserve the required function signature and satisfy the task constraints and input/output requirements.
Return only the JSON object, without Markdown fences or additional text."""

TIMEOUT_REPAIR = """Reduce runtime so the full evaluation finishes and completes successfully.
Identify the core computation and its contribution to the returned result.
First correct non-termination and eliminate redundant work, reusing intermediate results where possible.
Preserve a working version of the core procedure by reducing candidate sets, iteration counts, or search ranges when needed.
If the core method remains too costly, use a smaller search or cheaper approximation of the same decision information, keeping that estimate in the final decision."""


def failure_summary(failure: str) -> str:
    """Exception type and candidate line only."""
    if not failure:
        return ""
    lower = failure.lower()
    if "time" in lower and ("exceed" in lower or "timeout" in lower):
        return "timeout"
    kind = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)", failure)
    location = re.findall(r'(?:<candidate>|<string>).*?line (\d+)', failure)
    return (kind[-1] if kind else "execution_error") + (
        " at candidate line " + location[-1] if location else "")


class PromptBuilder(BasePromptBuilder):
    common_template = COMMON_TEMPLATE
    timeout_repair = TIMEOUT_REPAIR
    summarize = staticmethod(failure_summary)

    def repair_feedback(self, failure: str) -> str:
        return failure_summary(failure or "")
