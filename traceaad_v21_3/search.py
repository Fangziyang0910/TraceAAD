"""V21.3: current Base selection with V21.2's design reasoning, shown through layered evidence."""

from traceaad_base.search import TraceAAD as Base
from traceaad_v21_2.prompts import CANDIDATE_RESPONSE_FORMAT, PromptBuilder

from .context import context

PROTOCOL = "trajectory_design_reasoning_layered_v1"


class TraceAAD(Base):
    method = "traceaad_v21_3"
    prompt_type = PromptBuilder
    generation_protocol = PROTOCOL
    mechanisms = (2,)

    def __init__(self, task, model, **options):
        if hasattr(model, "request_options"):
            model.request_options = {**model.request_options, "response_format": CANDIDATE_RESPONSE_FORMAT}
        super().__init__(task, model, **options)

    def evolution_prompt(self, attempt, parent, reference):
        text, instruction, scores, view = context(self.archive, parent, attempt.operator, reference, self.config)
        attempt.record["context"] = view
        return self.prompts.evolution_render(text, instruction, has_scores=scores)
