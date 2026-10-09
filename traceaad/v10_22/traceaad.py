"""V10.22: V10.21's search; Explore reasons from the exact output of the function."""

from traceaad.v10_21.traceaad import TraceAADV1021
from .config import Config
from .prompts import PromptBuilder


class TraceAADV1022(TraceAADV1021):
    METHOD = "v1022"
    Config = Config
    PromptBuilder = PromptBuilder
