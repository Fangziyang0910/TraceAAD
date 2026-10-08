"""V10.21: V10.17's Refine, Explore and Crossover with concise prompts and per-instance execution.

V10.20's Deepen prescribed one direction of change (keep the rule, add a search
guided by it). V10.21 leaves the direction to the model: no step asks for more
search or more time, and only Explore proposals open explorations (V10.17).
"""

from traceaad.v10_17.traceaad import TraceAADV1017
from traceaad.common.instance_evaluation import InstanceProgramEvaluator
from .config import Config
from .prompts import PromptBuilder


class TraceAADV1021(TraceAADV1017):
    METHOD = "v1021"
    EXPLORING = ("Explore",)
    Config = Config
    PromptBuilder = PromptBuilder
    Evaluator = InstanceProgramEvaluator

    def _evaluation_options(self):
        return {"timeout_seconds": self.config.eval_timeout_seconds,
                "n_workers": self.config.eval_workers}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training.scheduler_socket = self.config.scheduler_socket
        if self.selection:
            self.selection.scheduler_socket = self.config.scheduler_socket

    def _freeze(self):
        """Recheck the training top programs and freeze the best valid result."""
        if self.selection:
            return super()._freeze()
        ranking = self._ranking()
        results = []
        for node_id in ranking[:self.config.final_candidates]:
            node = self.archive[node_id]
            outcome = self.training.evaluate(node["code"], node["key"], **self._evaluation_options())
            results.append({"node_id": node_id, **{k: v for k, v in outcome.items() if k != "evaluations"}})
            self._save(evaluations=outcome["evaluations"])
        self.progress.finalists = [r["node_id"] for r in results]
        self.progress.selection_results = results
        scored = [r for r in results if r["fitness"] is not None]
        if not scored:
            self.progress.phase = "selection_failed" if ranking else "no_valid_root"
            self._save()
            return
        chosen = min(scored, key=lambda r: (r["fitness"], ranking.index(r["node_id"])))["node_id"]
        self._finish(self.archive[chosen], "training")
