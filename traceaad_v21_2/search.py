"""V21.2: current Base selection with trajectory-guided design reasoning."""

from traceaad_base.search import TraceAAD as Base

from .context import context
from .prompts import CANDIDATE_RESPONSE_FORMAT, PromptBuilder

PROTOCOL = "trajectory_design_reasoning_v1"


class TraceAAD(Base):
    method = "traceaad_v21_2"
    prompt_type = PromptBuilder
    generation_protocol = PROTOCOL
    mechanisms = (2,)

    def __init__(self, task, model, *, reference_code=False, **options):
        """`reference_code` also shows the reference's full source; it is the control for the default."""
        self.reference_code = reference_code
        if reference_code:
            self.generation_protocol = PROTOCOL + "+reference_code"
        if hasattr(model, "request_options"):
            model.request_options = {**model.request_options, "response_format": CANDIDATE_RESPONSE_FORMAT}
        super().__init__(task, model, **options)

    def evolution_prompt(self, attempt, parent, reference):
        text, instruction, scores, view = context(self.archive, parent, attempt.operator, reference, self.config,
                                                  reference_code=self.reference_code)
        attempt.record["context"] = view
        return self.prompts.evolution_render(text, instruction, has_scores=scores)
