"""V21 material selection, with conditional experience used within the operator's task."""

from traceaad_v21_1.context import context as v21_context
from traceaad_v21_1.prompts import PromptBuilder as V21PromptBuilder

from .notes import experience_block

COMMON = (
    "Read the implemented computations and measured results together. "
    "Use design descriptions to locate the intended computation, and trace its role in the actual code."
)
REFINE = (
    "Find one limitation in what the current code computes and change the computation that addresses it, "
    "keeping its main mechanism. The change may add, replace, or remove a computation. "
    "Develop the computation using what earlier attempts revealed. "
    "When revisiting an idea, explain the changed condition, implementation, or supporting computation. "
    "In analysis, first identify the current computational problem and the change that addresses it. "
)
CROSSOVER = (
    "Adapt one useful computation from the reference into the current method, adjusting representations as needed. "
    "Choose an integration that contributes to the current decision, or develop your best candidate "
    "from the available computations. "
    "In analysis, first name the reference computation and the current decision it will change. "
)
EXPLORE = (
    "Introduce a different computation that determines the returned result. Options include decision rules "
    "on different information, estimating consequences before deciding, iteratively improving a complete "
    "solution, decomposition, solving small parts exactly, and population or restart search. "
    "Implement the computation in full for evaluation and further development. In analysis, state the new computation."
)
FAILURES = "Use execution feedback to locate and correct the reported implementation issue."
EXPERIENCE = (
    "Use applicable experience to develop the selected computation. Explain how its conditions match "
    "the current code or planned modification, including representation, scale, decision position and operation order."
)
OPERATORS = {"Refine": REFINE, "Explore": EXPLORE, "Crossover": CROSSOVER}


class PromptBuilder(V21PromptBuilder):
    """Keep the generation protocol and express the output instructions as positive actions."""

    common_template = """{task_description}

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
- "code": Provide the complete implementation of the required function, including necessary imports and helper definitions. Write executable Python in "code" and place explanations in "analysis" and "design".
- "design": After writing the code, describe what the code actually computes in one self-contained paragraph of 2–3 concise sentences, using at most 100 English words or 200 Chinese characters. State its representation, main computations, and how their results determine the output. Include the formulas or parameter values needed to understand the core mechanism. If the code differs from the analysis, describe the code.
Preserve the required function signature and satisfy the task constraints and input/output requirements.
Return the JSON object directly."""


def context(archive, parent, operator, reference, config, notes):
    text, _, view = v21_context(archive, parent, operator, reference, config)
    if operator == "Explore":
        return text, EXPLORE, view
    parts = [COMMON, OPERATORS[operator]]
    if view["failed"]:
        parts.append(FAILURES)
    if notes:
        parts.append(EXPERIENCE)
        text = experience_block(notes) + "\n\n" + text
    return text, "\n".join(parts), view
