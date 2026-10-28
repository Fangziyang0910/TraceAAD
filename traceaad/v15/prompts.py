"""V15 prompts. With information exposure, only the program being edited is shown as code.

Formation steps keep their Design and outcome but not their code diff; a
Crossover reference and earlier initial algorithms appear by Design and score.
The reply writes its Design after its code, so the Design describes the code
that was written rather than a plan.
"""

from traceaad.prompts import idea_view
from traceaad.v14.prompts import PromptBuilderV14

RECORDS_INFO = "Scores and failures are evaluation results. Design is a description of a program's code, written when the program was generated. Code is shown only for the algorithm you are asked to change; other programs are described by their Design and evaluated outcome. A repaired program inherits the failed program's Design."
FORMATION_INTRO_INFO = "The steps that produced the current algorithm, oldest first. Each step gives its evaluated outcome and the Design of the program it produced. If a step required repair, the first failure is stated separately."
CROSSOVER_INFO = "[Your Task: Crossover]\nCombine useful computations from the current algorithm and the reference design to improve solution quality. The reference is described, not shown: implement the parts you take from it in the current code, so that they work with its existing computations."
ANOTHER_INITIAL_INFO = "[Your Task: Design a Better Algorithm]\nUse the designs and scores above to design another algorithm that aims for better solutions."
DESIGN_INFO = "describe the code you returned: its main computations, naming the quantities, formulas and parameter values that decide its output, and, when modifying an algorithm, the main change; at most 200 words"


class PromptBuilderV15(PromptBuilderV14):
    def __init__(self, *args, info=True):
        super().__init__(*args)
        self.info = info
        if info:
            self.records = RECORDS_INFO
            self.formation_intro = FORMATION_INTRO_INFO
            self.crossover = CROSSOVER_INFO
            self.another_initial = ANOTHER_INITIAL_INFO

    def _change(self, parent, child):
        return "" if self.info else super()._change(parent, child)

    def _reference(self, reference):
        if not self.info:
            return super()._reference(reference)
        return f"[Reference Design]\nAnother evaluated algorithm from this search; its code is not shown.\n{self.measured(reference)}\nDesign: {idea_view(reference.idea)}"

    def _root(self, node, number):
        if not self.info:
            return super()._root(node, number)
        return f"Algorithm {number} · {self.measured(node)} · Design: {idea_view(node.idea)}"

    def _format(self, analysis, *, design=True, edit=False):
        if not (self.info and design):
            return super()._format(analysis, design=design)
        fields = [
            f'"analysis": explain {analysis}.',
            '"code": the complete Python program. Write no comments or docstrings in the code.',
            f'"design": {DESIGN_INFO}.',
        ]
        return "[Output Format]\nReply with one JSON object with three string fields, in this order:\n" + "\n".join(fields)
