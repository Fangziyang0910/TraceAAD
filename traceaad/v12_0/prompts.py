"""Match earlier Designs and build the implementation prompt with the selected program."""

import copy
import re

from benchmarks.tasks import FIXED_TASKS
from traceaad.common.history import score_text
from traceaad.common.prompts import ContextTooLong, idea_view, PromptBuilder as MeasuredPrompts

CALLED_ONCE = "The function is called once per instance, before the ant colony starts."
MANY_TIMES = re.compile(r"Use efficient NumPy operations because the function is evaluated many\s+times\.")
SERVICE = ("At a customer, the vehicle waits until the time window opens if it arrives early, then spends the "
           "customer's service duration before moving on. The current time passed to the function already includes "
           "the waiting and service durations of the customers served so far; the service durations themselves are "
           "not passed to the function.")


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

MATCH = """Each design below describes an algorithm for the same task.

[Design A]
{design}

[Earlier Designs in This Search]
{listed}

Which earlier designs are built on the same main computation as Design A?
End your reply with one line: "Same:" followed by their numbers separated by commas, or "Same: none"."""

PROPOSAL = "[Your Proposal]\n{proposal}"

EARLIER = """[An Earlier Program for the Same Computation]
An evaluated program from this search, built on the same main computation as your Design.
{measured}
Design: {design}
```python
{code}
```"""

IMPLEMENT = """[Your Task: Implement]
Implement your Design in the current algorithm."""


def corrected_description(task, description):
    """The task text with the facts about computation corrected."""
    if task in ("cvrp_aco", "op_aco"):
        description, replaced = MANY_TIMES.subn(CALLED_ONCE, description)
        if replaced != 1 and CALLED_ONCE not in description:
            raise ValueError(f"{task}: the task text no longer has the expected computation sentence")
    if task == "vrptw_construct":
        description += "\n\n" + SERVICE
    return description


def match_prompt(design, ranked):
    """The match request for a proposal's Design and the ranked earlier programs (also used offline)."""
    listed = "\n".join(f"[{program['id']}] {idea_view(program['idea'])}" for program, _ in ranked)
    return MATCH.format(design=idea_view(design), listed=listed)


class PromptBuilder(MeasuredPrompts):
    ACTIONS = ("Refine", "Explore", "Crossover")
    REPAIR = REPAIR
    UNCHANGED_ROOT = "an initial design"
    PROPOSAL = PROPOSAL
    EARLIER = EARLIER
    IMPLEMENT = IMPLEMENT

    def match(self, design, ranked):
        prompt = match_prompt(design, ranked)
        return prompt, self.count(prompt)

    def implement_format(self):
        return ("[Output Format]\nReply with one Python code block:\n"
                "Code:\n```python\n<the complete program>\n```\n" + self.CLOSING)

    def implement(self, explore_prompt, parent, proposal, earlier):
        """The implementation request; drops the earlier program, then the Explore material, if too long."""
        head = explore_prompt[:explore_prompt.rindex("[Output Format]")].rstrip()
        proposal_block = self.PROPOSAL.format(proposal=proposal) if proposal else ""
        earlier_block = (self.EARLIER.format(measured=self.measured(earlier), design=idea_view(earlier["idea"]),
                                             code=earlier["code"].rstrip()) if earlier is not None else "")
        tail = [self.IMPLEMENT, self.implement_format()]
        trims = []
        prompt = self._render([head, proposal_block, earlier_block, *tail])
        tokens = self.count(prompt)
        if tokens > self.config.max_input_tokens and earlier_block:
            trims.append("earlier_program")
            prompt = self._render([head, proposal_block, *tail])
            tokens = self.count(prompt)
        if tokens > self.config.max_input_tokens:
            trims.append("explore_material")
            prompt = self._render(self.common + [self._current(parent), proposal_block, *tail])
            tokens = self.count(prompt)
        if tokens > self.config.max_input_tokens:
            raise ContextTooLong("minimum implementation prompt exceeds context")
        return {"prompt": prompt, "input_tokens": tokens, "trims": trims}

    def _current(self, node):
        return f"[Current Algorithm]\n{self.measured(node)}\n```python\n{node['code'].rstrip()}\n```"

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

    def failure(self, program):
        if program["failure"]["kind"] == "timeout":
            return "timed out"
        return super().failure(program)

    def error_text(self, failed):
        if failed["failure"]["kind"] == "timeout":
            return "Timed out."
        return super().error_text(failed)
