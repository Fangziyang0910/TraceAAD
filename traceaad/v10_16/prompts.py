"""Version-specific goals and context; measured history is shared."""

from traceaad.common.prompts import PromptBuilder as MeasuredPrompts, ContextTooLong

REFINE = """[Your Task: Refine]
Continue this line of development: write an algorithm that scores better than every version and attempt shown above."""

REFINE_ROOT = """[Your Task: Refine]
Continue developing the current algorithm: write a version that scores better than it and than every attempt shown above."""

EXPLORE = """[Your Task: Explore]
Write an algorithm that scores better than the best found so far by changing how the current algorithm makes its decisions, not by tuning it."""

CROSSOVER = """[Your Task: Crossover]
Combine the two lines of development: write an algorithm that scores better than every version and attempt shown above, bringing into the current algorithm what the reference algorithm does well."""

ANALYSIS = {
    "Refine": "what limits this line of development so far, and what change should take it past every version and attempt",
    "Crossover": "what the reference algorithm does well that the current algorithm lacks, and how to combine them",
    "Repair": "what caused the failure, and how to fix it",
    "Explore": "what the current algorithm cannot capture, and what different computation should capture it",
    "Init": None,
}

class PromptBuilder(MeasuredPrompts):
    UNCHANGED_ROOT = "an initial design"
    REFINE = REFINE
    REFINE_ROOT = REFINE_ROOT
    CROSSOVER = CROSSOVER
    ANALYSIS = ANALYSIS

    def _current(self, node):
        return f"[Current Algorithm]\n{self.measured(node)}\n```python\n{node['code'].rstrip()}\n```"
