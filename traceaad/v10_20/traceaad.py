"""V10.20: V10.17's search with the Deepen step among the sampled operators.

Deepen is sampled like Refine, Crossover and Explore and starts from the same
experience-weighted parent. Like an Explore proposal, a Deepen proposal brings
new computation into the algorithm, so it opens an exploration: a random eighth
of them get three Refine steps from the best version reached (V10.17's rule).
Its prompt is in prompts.py. Repair and final selection are V10.17's.
"""

from traceaad.v10_17.traceaad import TraceAADV1017
from .config import Config
from .prompts import PromptBuilder


class TraceAADV1020(TraceAADV1017):
    METHOD = "v1020"
    EXPLORING = ("Explore", "Deepen")
    Config = Config
    PromptBuilder = PromptBuilder
