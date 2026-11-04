"""V20 prompt texts: Base prompts without word caps and with short failure feedback."""

import re

from traceaad_base.prompts import COMMON_TEMPLATE as BASE_TEMPLATE, PromptBuilder as BasePromptBuilder

COMMON_TEMPLATE = BASE_TEMPLATE.replace(
    '- "analysis": In 2–3 concise sentences, using at most 100 English words or 200 Chinese characters, explain',
    '- "analysis": Briefly explain',
).replace(
    'in one self-contained paragraph of 2–3 concise sentences, using at most 100 English words or 200 Chinese characters.',
    'in one short self-contained paragraph.',
)

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
