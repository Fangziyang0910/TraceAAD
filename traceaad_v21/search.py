"""V21 search: Base's selection, evaluation and repair with the trajectory context in generation."""

from traceaad_base.search import TraceAAD as Base

from .context import context, reference
from .prompts import CANDIDATE_RESPONSE_FORMAT, PromptBuilder


class TraceAAD(Base):
    method = "traceaad_v21"
    prompt_type = PromptBuilder

    def __init__(self, task, model, **options):
        if hasattr(model, "request_options"):
            model.request_options = {**model.request_options, "response_format": CANDIDATE_RESPONSE_FORMAT}
        super().__init__(task, model, **options)

    def reference(self, parent):
        return reference(self.archive, parent, self.rng)

    def evolution_prompt(self, attempt, parent, reference):
        text, instruction, view = context(self.archive, parent, attempt.operator, reference, self.config)
        attempt.record["context"] = view
        return self.prompts._render(text, instruction, has_scores=attempt.operator != "Explore")
