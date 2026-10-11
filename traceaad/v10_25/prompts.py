"""V10.17 context roles with explicit changes and grounded modification prompts."""

from traceaad.common.history import code_diff
from traceaad.common.prompts import KNOWN, idea_view
from traceaad.v10_21.prompts import PromptBuilder as ProtocolPrompts


class PromptBuilder(ProtocolPrompts):
    FORMATION_INTRO = (
        "The steps that produced the current algorithm, oldest first. "
        "Each step gives its outcome, Design and code changes. "
        "Design describes the proposed algorithm; scores report the results of whole programs. "
        "When a proposal failed and was repaired, the step shows the failure, both Designs, "
        "and separate diffs for the proposal and its repair.")
    CROSSOVER = f"""[Your Task: Crossover]
Adapt a computation from the reference algorithm that could improve the current algorithm. Keep the current algorithm's core idea and write a version that scores better than it. {KNOWN}"""
    ANALYSIS = {
        **ProtocolPrompts.ANALYSIS,
        "Refine": ("what limits the current algorithm, using its code and any available code changes "
                   "and evaluated outcomes, and what change could improve it while retaining its core idea"),
        "Crossover": ("which computation in the reference could help the current algorithm, considering "
                      "the code and any available changes and outcomes, and how to adapt it to work "
                      "with the current computation while retaining the current algorithm's core idea"),
        "Repair": ("what caused the failure, and how to fix it while retaining "
                   "the intended decision computation"),
    }

    def _edge(self, parent, child, failed, index, *, latest=False, subject="current"):
        text = super()._edge(parent, child, failed, index, latest=latest, subject=subject)
        if failed is None:
            return text
        # Keep one history step so trimming cannot separate a failed proposal from its repair.
        heading = text.split("\n  Code diff (previous → current):", 1)[0]
        return (heading + f"\n  Failed proposal's Design: {idea_view(failed.get('idea'))}"
                f"\n  Code diff (program {parent['id']} → failed program {failed['id']}):\n```diff\n"
                f"{code_diff(parent['code'], failed['code'])}\n```"
                f"\n  Repair diff (failed program {failed['id']} → program {child['id']}):\n```diff\n"
                f"{code_diff(failed['code'], child['code'])}\n```")
