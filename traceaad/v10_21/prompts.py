"""V10.21 context: the problem, the interface and the score; the rest is algorithm improvement.

The prompts say nothing about time, as in the related methods: the evaluator
enforces its limits for every method, and the model's job is a better
algorithm, not a model of the evaluator. Every earlier statement of time
steered what the model wrote: V10.20's timings and Deepen made programs aim
at the limit, "Programs must be efficient" kept them cheap scoring rules, and
a stated per-instance budget made the first programs prune the candidates
(22% of TSP initial programs, 10% without it), code that later edits often
broke. A timed-out program is reported as timed out, nothing more.

One line states what the score is and that lower is better: without it,
JSSP Refine read the best programs' negative scores (5-7% below the
reference) as failures and removed their lookahead simulation. Measurements
show the score only. The fixed tasks' description keeps the problem and the
solver settings, without the evaluator's bookkeeping. Steps are Refine,
Explore and Crossover; none prescribes a direction. Code comments stay banned.
"""

import copy

from benchmarks.tasks import FIXED_TASKS
from traceaad.common.history import score_text
from traceaad.v10_17.prompts import PromptBuilder as V1017Prompts
from traceaad.v10_20.prompts import corrected_description

CVRP_COMPUTATION = ("Within the evaluation time limit, the function\n"
                    "may perform any computation on the complete instance before returning its prior.")
# The interface ends with the contract; no statement about how much to compute.
COMPUTATION = (" Within the time limit, the function may perform any computation on its inputs.", "")
# Said once, in the score line.
OP_OBJECTIVE = "\nThe evaluation objective is the negative mean collected prize; lower is better."
FIXED_SCORE = ("the mean deviation of the final objective from a reference value, in percent of the reference; "
               "a negative score beats the reference")


REPAIR = """[Your Task: Repair]
The program failed during evaluation. Fix it, keeping the algorithm it was meant to implement."""


class PromptBuilder(V1017Prompts):
    ACTIONS = ("Refine", "Explore", "Crossover")
    REPAIR = REPAIR

    def __init__(self, llm, task, evaluation, programs, attempts, config):
        evaluation = copy.copy(evaluation)
        if task in FIXED_TASKS:
            # The problem and the solver's settings; the evaluator's bookkeeping
            # (seeding, copies, reference handling) is left out.
            settings = ", ".join(f"{name}={value}" for name, value in evaluation.outer_settings.items())
            evaluation.task_description = evaluation.DESCRIPTION + (
                f"\nThe fixed solver settings are {settings}." if settings else "")
            evaluation.design_notes = ""
            evaluation.score_meaning = FIXED_SCORE
        else:
            evaluation.task_description = (evaluation.task_description.replace(CVRP_COMPUTATION, "")
                                           .replace(OP_OBJECTIVE, "").rstrip())
        super().__init__(llm, task, evaluation, programs, attempts, config)
        self.common[2] = self.common[2].replace(*COMPUTATION)

    def task_text(self, description):
        return corrected_description(self.task, description)

    def evaluation_text(self, meaning, higher):
        return f"Score: {meaning}. {'Higher' if higher else 'Lower'} is better."

    def measured(self, program):
        return f"Score: {score_text(program['score'])}"

    def elapsed(self, seconds):
        return ""

    def failure(self, program):
        if program["failure"]["kind"] == "timeout":
            return "timed out"
        return super().failure(program)

    def error_text(self, failed):
        if failed["failure"]["kind"] == "timeout":
            return "Timed out."
        return super().error_text(failed)
