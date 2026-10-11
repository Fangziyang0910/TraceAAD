"""V10.17 search with current evaluation and explicit failed/repair changes."""

from traceaad.v10_21.traceaad import TraceAADV1021
from .config import Config
from .prompts import PromptBuilder


class TraceAADV1025(TraceAADV1021):
    METHOD = "v1025"
    Config = Config
    PromptBuilder = PromptBuilder

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.identity["mechanism"] = "v1025-v1017-base-2"
        saved = self.facts.state or {}
        if saved and saved.get("identity") != self.identity:
            raise ValueError("V10.25 resume requires the same configuration, task, protocol and mechanism")
