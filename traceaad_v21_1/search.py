"""V21.1: current Base selection with mechanism-2 generation."""

from traceaad_base.search import TraceAAD as Base

from .context import context
from .prompts import CANDIDATE_RESPONSE_FORMAT, PromptBuilder


class TraceAAD(Base):
    method = "traceaad_v21_1"
    prompt_type = PromptBuilder
    generation_protocol = "trajectory_algorithm_context_v1"
    mechanisms = (2,)

    def __init__(self, task, model, **options):
        if hasattr(model, "request_options"):
            model.request_options = {**model.request_options, "response_format": CANDIDATE_RESPONSE_FORMAT}
        super().__init__(task, model, **options)

    def evolution_prompt(self, attempt, parent, reference):
        text, instruction, view = context(self.archive, parent, attempt.operator, reference, self.config)
        attempt.record["context"] = view
        return self.prompts._render(text, instruction, has_scores=attempt.operator != "Explore")
