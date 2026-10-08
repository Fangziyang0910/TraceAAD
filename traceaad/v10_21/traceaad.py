"""V10.21: V10.20's search; only the prompts change (prompts.py)."""

from traceaad.v10_20.traceaad import TraceAADV1020
from .config import Config
from .prompts import PromptBuilder


class TraceAADV1021(TraceAADV1020):
    METHOD = "v1021"
    Config = Config
    PromptBuilder = PromptBuilder
