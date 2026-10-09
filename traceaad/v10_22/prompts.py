"""V10.22 context: Explore reasons from the output the function should produce; Refine from the score.

The model proposes what its own analysis of the current algorithm suggests.
Asked what the algorithm "cannot capture", it names the next better-known
estimate for one term of the same rule (an MST or 1-tree bound instead of a
nearest-neighbour completion), whatever the search has already tried: in the
TSP runs of 2026-10-09, 52% of Explores proposed that bound and 4.8% beat their
start; listing the failed Explores of the whole search left the share at 57%
(docs/03-现象与检验/2026-10-09-Explore的先验与精确决策.md). The structures
that set the strongest results approximate the exact output instead: planning
the remaining route and taking its first step (TSP 5.70 against 5.81-5.94),
constructing complete solutions and returning their edges (CVRP). Explore's
analysis therefore starts from the exact output and from what the current
algorithm computes in its place.

The same prior drives every step: after the first 300 attempts, 70%-89% of the
analyses of Explore, Refine and Crossover named computational cost as what
limits the current algorithm, and those changes beat their start less often
(Refine 8.7% against 16%). The prompts still say nothing about time; Refine's
analysis asks what limits the current algorithm's score, and Explore's starts
from the output.

Explore decides from the current algorithm and the attempts made from it. The
list of changes that set a new best is dropped: removing it lowered the share
of repeated proposals (57% to 46%), and the goal no longer measures against
the search's best. The goal is stated for the current algorithm, as in Refine
and Crossover. A further initial design is asked for a different core idea,
not to beat the ones shown.
"""

from traceaad.common.prompts import KNOWN
from traceaad.v10_21.prompts import PromptBuilder as V1021Prompts

ANOTHER_INITIAL = """[Your Task: Design a Different Algorithm]
Write an algorithm for this task built on a core idea different from those of the algorithms above, and make it score as well as possible."""

EXPLORE = f"""[Your Task: Explore]
Write a version of the current algorithm that scores better than it by changing what it computes to produce its output, not by tuning it. {KNOWN}"""

REFINE_ANALYSIS = ("what limits the current algorithm's score, and what change keeps its core idea and makes it "
                   "score better")

EXPLORE_ANALYSIS = ("which output would lead to the best final result for the inputs the function receives, and what "
                    "computing that output exactly would require; what the current algorithm computes in its place; "
                    "and which different computation would come closer to that output")


class PromptBuilder(V1021Prompts):
    ANOTHER_INITIAL = ANOTHER_INITIAL
    EXPLORE = EXPLORE
    ANALYSIS = {**V1021Prompts.ANALYSIS, "Refine": REFINE_ANALYSIS, "Explore": EXPLORE_ANALYSIS}

    def _progress_section(self, shown):
        return "", []
