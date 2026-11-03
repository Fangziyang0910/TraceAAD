"""Model-visible prompts and finite history windows from the Base specification."""

from .config import Config
from .models import Program
from .task import Task


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
- "analysis": In 2–3 concise sentences, using at most 100 English words or 200 Chinese characters, explain the algorithmic rationale of the proposed change, the evidence that motivates it, and its expected effect on solving the task.
- "design": Describe the final algorithm implemented in "code" in one self-contained paragraph of 2–3 concise sentences, using at most 100 English words or 200 Chinese characters. State the solution representation, the main solving principle, and how the procedure determines the returned result.
- "code": Provide the complete implementation of the required function, including necessary imports and helper definitions. Do not include comments or docstrings.
First determine a complete executable algorithm; then make "analysis" and "design" accurately describe the final "code".
Preserve the required function signature and satisfy the task constraints and input/output requirements.
Return only the JSON object, without Markdown fences or additional text."""

INIT = """Design an effective algorithm for the task described above and implement the specified Python function."""

INIT_WITH_HISTORY = """Design an effective algorithm for the task and implement the specified Python function.
Study the existing algorithms and their scores to understand the solution principles already represented.
Choose a distinct and well-motivated algorithmic principle for this initial candidate."""

INIT_FAILURES = """Use the failed designs, their analyses, and evaluation feedback to guide how you implement your design."""

ANCESTORS = """Read the ancestor sequence as the development of an algorithmic idea.
Use the changes and their evaluated outcomes to decide which direction should be continued, revised, or redirected."""

OFFSPRING = """Read the previous offspring as trials of alternative developments of the current idea.
Use their analyses, designs, and outcomes to develop a meaningful next variant rather than repeating an already tested direction."""

FAILURES = """Use the failed offspring and their feedback to understand which proposed directions were not executable or not suitable for the task.
Account for that evidence when choosing and implementing the next algorithm."""

REFINE = """Develop the current algorithm as a continuation of its solution principle.
Use its implementation, history, and outcomes to identify a specific weakness or opportunity, then make one coherent improvement to the construction method, evaluation rule, search process, or parameters.
Implement a complete algorithm whose decisions remain grounded in the refined principle."""

EXPLORE = """Design an alternative solution method based on a different algorithmic principle.
Choose a representation and a construction or search process suited to the underlying problem; the method may reason through auxiliary solutions, future consequences, or a different search space.
Implement the complete method using the available inputs and return the result required by the function interface."""

CROSSOVER = """Study the current and reference algorithms as two sources of algorithmic ideas.
Identify complementary principles and synthesize them into one coherent solution method, explaining the role of each source and how they work together in the final decisions.
Implement the complete synthesized algorithm using the available inputs and the required function interface."""

FORMAT_REPAIR = """Convert the original response into the JSON object specified in [Output], using the parsing feedback to correct its structure, required fields, and string escaping.
Preserve the existing algorithmic idea and computational logic.
Complete any missing or truncated content consistently with that approach and the task requirements."""

ERROR_REPAIR = """Trace the reported exception or invalid result to its cause in the code, using the design to understand the intended algorithm.
Correct the faulty computations or data handling so the function executes correctly and satisfies the task requirements, while retaining the intended core decision mechanism."""

TIMEOUT_REPAIR = """Reduce runtime so the full evaluation finishes within the reported total time limit.
Identify the core computation and its contribution to the returned result.
First correct non-termination and eliminate redundant work, reusing intermediate results where possible.
Preserve a working version of the core procedure by reducing candidate sets, iteration counts, or search ranges when needed.
If the core method remains too costly, use a smaller search or cheaper approximation of the same decision information, keeping that estimate in the final decision."""

REPAIR_DESIGN = """Update "analysis" to retain the original design rationale and describe the computational changes implemented after repair.
Update "design" to describe the repaired algorithm as a self-contained procedure consistent with "code"."""


def failure_summary(failure: str) -> str:
    """Keep the exception and a useful location, without copying its full stack."""
    lines = failure.strip().splitlines()
    if not lines or not any(line.startswith("Traceback") for line in lines):
        return failure.strip()
    frames = [line.strip() for line in lines if line.lstrip().startswith('File "')]
    message_start = len(lines) - 1
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and not line.startswith("Traceback"):
            message_start = i
    message = "\n".join(lines[message_start:])
    candidate_frames = [line for line in frames if "<candidate>" in line or "<string>" in line]
    location = (candidate_frames or frames)[-1:]
    return "\n".join([message, *location])


class PromptBuilder:
    def __init__(self, task: Task, config: Config):
        self.task = task
        self.config = config

    def _render(self, context: str, instruction: str, *, has_scores=False) -> str:
        template = COMMON_TEMPLATE
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
                      f"Failure: {failure_summary(program.evaluation.failure)}"])
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
            context.append(
                f"[Reference Algorithm]\nAlgorithm {reference.id}\n"
                f"Analysis: {reference.analysis}\n"
                f"Design: {reference.design}\n"
                f"Score: {reference.evaluation.fitness}\n"
                f"```python\n{reference.code}\n```"
            )
        context.append(f"[Current Algorithm]\n{self._record(parent)}\n"
                       f"```python\n{parent.code}\n```")
        instruction = {"Refine": REFINE, "Explore": EXPLORE, "Crossover": CROSSOVER}[operator]
        for available, guidance in ((ancestors, ANCESTORS), (children, OFFSPRING), (failures, FAILURES)):
            if available:
                instruction += "\n" + guidance
        return self._render("\n\n".join(context), instruction, has_scores=True)

    def repair_format(self, response: str, feedback: str) -> str:
        context = f"[Original Response]\n{response}\n\n[Parsing Feedback]\n{feedback}"
        return self._render(context, FORMAT_REPAIR + "\n" + REPAIR_DESIGN)

    def repair_program(self, program: Program) -> str:
        origin = "\n".join(self._origin(program))
        context = (f"[Current Algorithm]\n{origin}\nDesign: {program.design}\n"
                   f"```python\n{program.code}\n```\n\n"
                   f"[Failure Feedback]\n{program.evaluation.failure}")
        instruction = {"error": ERROR_REPAIR, "timeout": TIMEOUT_REPAIR}[program.evaluation.status]
        return self._render(context, instruction + "\n" + REPAIR_DESIGN)
