"""V21.3: V21.2 with every evolution step shown through its full actual code change."""

from traceaad_v21_2.search import TraceAAD as V21_2

from .context import context

PROTOCOL = "trajectory_design_reasoning_full_diff_v1"


class TraceAAD(V21_2):
    method = "traceaad_v21_3"
    generation_protocol = PROTOCOL

    def evolution_prompt(self, attempt, parent, reference):
        text, instruction, scores, view = context(self.archive, parent, attempt.operator, reference, self.config,
                                                  reference_code=self.reference_code)
        attempt.record["context"] = view
        return self.prompts.evolution_render(text, instruction, has_scores=scores)
