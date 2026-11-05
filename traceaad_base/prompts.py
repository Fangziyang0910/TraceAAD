"""Model-visible prompts and finite history windows from the Base specification."""

import re

from .config import Config
from .models import Program
from .task import Task

TIMEOUT_PREFIX = "Evaluation exceeded"


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
- "analysis": In 2–3 concise sentences, using at most 100 English words or 200 Chinese characters, explain the approach you will implement and why it suits the task. When modifying an algorithm, state which computations you retain, replace, or add, the relevant evidence from the provided context, and how the change affects the returned result.
- "design": Describe the decision procedure implemented in "code" in one self-contained paragraph of 2–3 concise sentences, using at most 100 English words or 200 Chinese characters. State its representation, main computations, and how their results determine the output. Include the formulas or parameter values needed to understand the core mechanism.
- "code": Provide the complete implementation of the required function, including necessary imports and helper definitions. Do not include comments or docstrings.
Keep "analysis" and "design" consistent with the implementation in "code".
Preserve the required function signature and satisfy the task constraints and input/output requirements.
Return only the JSON object, without Markdown fences or additional text."""

INIT = """Design an effective algorithm for the task described above and implement the specified Python function."""

INIT_WITH_HISTORY = """Design an effective algorithm for the task and implement the specified Python function.
Use the existing algorithms and their scores to identify the decision information they already use.
Develop a solution procedure that obtains useful decision information through a different main computation."""

INIT_FAILURES = """Use the failed designs, their analyses, and evaluation feedback to guide how you implement your design."""

ANCESTORS = """Follow the parent links, designs, and analyses to understand how the current algorithm developed.
Use the evaluated outcomes of those changes to guide the requested design."""

OFFSPRING = """Use the previous offspring analyses, designs, and evaluated outcomes to identify which changes to the current algorithm have been tried.
Develop a further computational change beyond the designs already tried, aiming for better solution quality."""

FAILURES = """Use the failed offspring analyses and designs with their error or timeout feedback to identify issues relevant to your proposed change, and address them in the new code."""

REFINE = """Identify the current algorithm's core decision mechanism from its code, including the intermediate computations that determine the returned result.
Improve solution quality by refining those computations, their supporting steps, or their parameters, while preserving the core mechanism's role in the decision."""

EXPLORE = """Design an alternative algorithm that improves solution quality through a different main decision mechanism.
Draw on solution methods for the underlying problem to choose an internal representation and computations.
Implement that mechanism using the available inputs and return the result required by the function interface."""

CROSSOVER = """Improve solution quality by integrating a computation from the reference design that supplies information complementary to the current algorithm.
Implement the selected computation using its stated representation and formulas, and combine its result with the current decision mechanism so both contribute to the returned result.
Align their input representations and numerical scales so the selected computations work together."""

FORMAT_REPAIR = """Convert the original response into the JSON object specified in [Output], using the parsing feedback to correct its structure, required fields, and string escaping.
Preserve the existing algorithmic idea and computational logic.
Complete any missing or truncated content consistently with that approach and the task requirements."""

ERROR_REPAIR = """Trace the reported exception or invalid result to its cause in the code, using the design to understand the intended algorithm.
Correct the faulty computations or data handling so the function executes correctly and satisfies the task requirements, while retaining the intended core decision mechanism."""

TIMEOUT_REPAIR = """Reduce runtime so the full evaluation finishes within the reported time limit.
Identify the core computation and its contribution to the returned result.
First correct non-termination and eliminate redundant work, reusing intermediate results where possible.
Preserve a working version of the core procedure by reducing candidate sets, iteration counts, or search ranges when needed.
If the core method remains too costly, use a smaller search or cheaper approximation of the same decision information, keeping that estimate in the final decision."""

REPAIR_DESIGN = """In "analysis", state only what went wrong and what you changed to correct it; the earlier analysis is kept.
Update "design" to describe the repaired algorithm as a self-contained procedure consistent with "code"."""


def _parse_failure(failure: str):
    """The exception message and the traceback blocks that lie in the candidate's own code."""
    lines = failure.strip().splitlines()
    start = len(lines) - 1
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and not line.startswith("Traceback"):
            start = i
    message = re.sub(r"^(?:\w+\.)+(?=\w+(?::|$))", "", "\n".join(lines[start:]))
    blocks, keep = [], False
    for line in lines[:start]:
        if line.lstrip().startswith('File "'):
            keep = "<candidate>" in line or "<string>" in line
            if keep:
                blocks.append([line.strip()])
        elif keep and line.startswith("  "):
            blocks[-1].append(line)
        else:
            keep = False
    return re.sub(r"\d+(?:\.\d+)?", "N", message), blocks


def failure_summary(failure: str) -> str:
    """The exception and where it occurred in the candidate; no limits, sizes or evaluator frames."""
    if failure.startswith(TIMEOUT_PREFIX):
        return "The evaluation did not finish within the allowed time."
    message, blocks = _parse_failure(failure)
    return "\n".join([message, *(blocks[-1][:1] if blocks else [])])


def failure_feedback(failure: str) -> str:
    """Repair also sees the candidate's own traceback lines."""
    message, blocks = _parse_failure(failure)
    return "\n".join([*(line for block in blocks for line in block), message])


class PromptBuilder:
    common_template = COMMON_TEMPLATE
    timeout_repair = TIMEOUT_REPAIR
    summarize = staticmethod(failure_summary)

    def __init__(self, task: Task, config: Config):
        self.task = task
        self.config = config

    def _render(self, context: str, instruction: str, *, has_scores=False) -> str:
        template = self.common_template
        if not context:
            template = template.replace("[Context]\n{context}\n\n", "")
        elif has_scores and self.task.score_definition not in self.task.task_description:
            context = f"Score definition: {self.task.score_definition}\n\n{context}"
        return template.format_map({
            "task_description": self.task.task_description,
            "function_template": self.task.function_template,
            "context": context,
            "instruction": instruction,
        })

    def _failures(self, archive: dict[int, Program], parent_id: int) -> list[Program]:
        failed = [p for p in archive.values() if p.parent_id == parent_id
                  and p.evaluation is not None and p.evaluation.status != "success"]
        return sorted(sorted(failed, key=lambda p: -p.id)[:self.config.failure_count],
                      key=lambda p: p.id)

    @staticmethod
    def _origin(program: Program, *, operator: bool = True) -> list[str]:
        lines = [f"Algorithm {program.id}", f"Parent: Algorithm {program.parent_id}"]
        if operator:
            lines.append(f"Operator: {program.operator}")
        if program.reference_id is not None:
            lines.append(f"Reference: Algorithm {program.reference_id}")
        if program.analysis:
            lines.append(f"Analysis: {program.analysis}")
        return lines

    @classmethod
    def _record(cls, program: Program, *, operator: bool = True) -> str:
        return "\n".join([*cls._origin(program, operator=operator),
                          f"Design: {program.design}",
                          f"Score: {program.evaluation.fitness}"])

    @classmethod
    def _failure_block(cls, program: Program, *, operator: bool) -> str:
        lines = cls._origin(program, operator=operator)
        lines.extend([f"Design: {program.design}",
                      f"Failure: {cls.summarize(program.evaluation.failure)}"])
        return "\n".join(lines)

    def initialization(self, archive: dict[int, Program]) -> str:
        context = []
        if 0 in archive:
            example = archive[0]
            text = "[Example Algorithm]\n"
            if example.design:
                text += f"Design: {example.design}\n"
            context.append(text + f"```python\n{example.code}\n```")
        valid = sorted((p for p in archive.values() if p.valid and p.operator == "Init"),
                       key=lambda p: (p.evaluation.fitness, p.id))
        if valid:
            algorithms = [f"{self._record(p, operator=False)}\n```python\n{p.code}\n```"
                          for p in valid]
            context.append("[Existing Algorithms]\nListed best first. Lower scores are better.\n\n"
                           + "\n\n".join(algorithms))
        failed = self._failures(archive, 0)
        if failed:
            context.append("[Failed Attempts]\n" + "\n\n".join(
                self._failure_block(p, operator=False) for p in failed))
        instruction = INIT_WITH_HISTORY if valid else INIT
        if failed:
            instruction += "\n" + INIT_FAILURES
        return self._render("\n\n".join(context), instruction, has_scores=bool(valid))

    def histories(self, archive: dict[int, Program], parent: Program):
        ancestors = []
        current = parent.parent_id
        while current and current in archive and len(ancestors) < self.config.ancestor_count:
            ancestor = archive[current]
            ancestors.append(ancestor)
            current = ancestor.parent_id
        ancestors.reverse()
        children = [p for p in archive.values() if p.parent_id == parent.id and p.valid]
        selected = []
        if children and self.config.offspring_count:
            best = min(children, key=lambda p: (p.evaluation.fitness, -p.id))
            selected = [best] + sorted((p for p in children if p.id != best.id),
                                       key=lambda p: -p.id)[:self.config.offspring_count - 1]
            selected.sort(key=lambda p: p.id)
        return ancestors, selected, self._failures(archive, parent.id)

    def evolution(self, archive: dict[int, Program], parent: Program,
                  operator: str, reference: Program | None = None) -> str:
        ancestors, children, failures = self.histories(archive, parent)
        context = []

        def records(programs):
            return "\n\n".join(self._record(p) for p in programs)

        if ancestors:
            context.append("[Ancestor History]\nAncestors of the current algorithm, "
                           "ordered from earlier to later.\n\n" + records(ancestors))
        if children:
            context.append("[Evaluated Offspring]\nPrevious algorithms generated directly "
                           "from the current algorithm, ordered from earlier to later.\n\n"
                           + records(children))
        if failures:
            context.append("[Failed Offspring]\nFailed attempts generated directly from "
                           "the current algorithm.\n\n" + "\n\n".join(
                               self._failure_block(p, operator=True) for p in failures))
        if operator == "Crossover":
            if reference is None:
                raise ValueError("Crossover requires a reference")
            context.append(f"[Reference Design]\nScore: {reference.evaluation.fitness}\n"
                           f"Design: {reference.design}")
        context.append(f"[Current Algorithm]\n{self._record(parent)}\n"
                       f"```python\n{parent.code}\n```")
        instruction = {"Refine": REFINE, "Explore": EXPLORE, "Crossover": CROSSOVER}[operator]
        for available, guidance in ((ancestors, ANCESTORS), (children, OFFSPRING), (failures, FAILURES)):
            if available:
                instruction += "\n" + guidance
        return self._render("\n\n".join(context), instruction, has_scores=True)

    def repair_format(self, response: str, feedback: str) -> str:
        context = f"[Original Response]\n{response}\n\n[Parsing Feedback]\n{feedback}"
        return self._render(context, FORMAT_REPAIR)

    def repair_feedback(self, failure: str) -> str:
        if not failure.startswith(TIMEOUT_PREFIX):
            return failure_feedback(failure)
        per_instance = self.task.metadata.get("seconds_per_instance")
        allowed = (f"{per_instance:g} seconds per problem instance" if per_instance
                   else f"{self.task.time_limit:g} seconds in total")
        return f"The evaluation did not finish within the allowed time of {allowed}."

    def repair_program(self, program: Program) -> str:
        origin = "\n".join(self._origin(program))
        context = (f"[Current Algorithm]\n{origin}\nDesign: {program.design}\n"
                   f"```python\n{program.code}\n```\n\n"
                   f"[Failure Feedback]\n{self.repair_feedback(program.evaluation.failure)}")
        instruction = {"error": ERROR_REPAIR, "timeout": self.timeout_repair}[program.evaluation.status]
        return self._render(context, instruction + "\n" + REPAIR_DESIGN)
