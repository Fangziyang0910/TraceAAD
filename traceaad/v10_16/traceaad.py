"""V10.16: its mechanism, using shared search execution."""

from traceaad.common.search import Search
from .config import Config
from .prompts import PromptBuilder


class TraceAADV1016(Search):
    METHOD = "v1016"
    Config = Config
    PromptBuilder = PromptBuilder
