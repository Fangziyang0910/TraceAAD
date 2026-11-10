"""V21.3: compact operator-specific evolution, integration and exploration records."""

from traceaad_base.search import TraceAAD as Base
from .context import context
from .prompts import CANDIDATE_RESPONSE_FORMAT, PromptBuilder

PROTOCOL = "operator_trajectory_context_v1"


class TraceAAD(Base):
    method = "traceaad_v21_3"
    prompt_type = PromptBuilder
    generation_protocol = PROTOCOL
    mechanisms = (2,)

    def __init__(self, task, model, *, reference_code=False, **options):
        self.reference_code = reference_code
        if reference_code:
            self.generation_protocol = type(self).generation_protocol + "+reference_code"
        if hasattr(model, "request_options"):
            model.request_options = {**model.request_options, "response_format": CANDIDATE_RESPONSE_FORMAT}
        super().__init__(task, model, **options)

    def evolution_prompt(self, attempt, parent, reference):
        text, instruction, scores, view = context(self.archive, parent, attempt.operator, reference, self.config,
                                                  reference_code=self.reference_code)
        attempt.record["context"] = view
        return self.prompts.evolution_render(text, instruction, has_scores=scores)
