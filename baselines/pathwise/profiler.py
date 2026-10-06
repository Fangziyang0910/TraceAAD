from __future__ import annotations

import json
import os

from .graph import PathWiseAction, PathWiseEdge, PathWiseNode
from .population import Population
from pathlib import Path
from traceaad.common.storage import append_jsonl
from baselines.profiler import ProfilerBase


class PathWiseProfiler(ProfilerBase):
    def __init__(self, run_dir=None, **kwargs):
        super().__init__(run_dir, **kwargs)
        self._cur_gen = 0
        if self._log_dir:
            self._event_dir = os.path.join(self._log_dir, "pathwise")
            os.makedirs(self._event_dir, exist_ok=True)

    def register_population(self, pop: Population):
        if not self._log_dir:
            return
        with self._artifact_lock:
            if self._num_samples == 0 or pop.generation == self._cur_gen:
                return
            nodes_json = []
            for node in pop.nodes:
                nodes_json.append({
                    "node_id": node.node_id,
                    "description": node.description,
                    "rationale": node.rationale,
                    "function_key": self._sources.add(str(node.function)),
                    "score": node.score,
                    "parents": [parent.__dict__ for parent in node.parents],
                })
            append_jsonl(Path(self._log_dir).parent / "events.jsonl",
                         {"kind": "population", "generation": pop.generation, "members": nodes_json})
            self._cur_gen = pop.generation

    def register_entailment_step(
            self,
            *,
            outer_iteration: int,
            inner_step: int,
            actions: list[PathWiseAction],
            selected_node: PathWiseNode,
            edge: PathWiseEdge,
            policy_reflection: str,
            world_model_reflection: str,
    ):
        self._append_event("entailment_steps.jsonl", {
            "outer_iteration": outer_iteration,
            "inner_step": inner_step,
            "actions": [action.__dict__ for action in actions],
            "selected_node": {
                "node_id": selected_node.node_id,
                "description": selected_node.description,
                "rationale": selected_node.rationale,
                "score": selected_node.score,
            },
            "edge": edge.__dict__,
            "policy_reflection": policy_reflection,
            "world_model_reflection": world_model_reflection,
        })
        self.log_method_event(
            method="pathwise",
            event="entailment_step",
            outer_iteration=outer_iteration,
            inner_step=inner_step,
            actions=[action.__dict__ for action in actions],
            selected_node_id=selected_node.node_id,
            selected_score=selected_node.score,
            edge=edge.__dict__,
            policy_reflection=policy_reflection,
            world_model_reflection=world_model_reflection,
        )
        self.log_method_state(
            method="pathwise",
            phase="entailment_graph",
            outer_iteration=outer_iteration,
            inner_step=inner_step,
            selected_node_id=selected_node.node_id,
            selected_score=selected_node.score,
        )

    def _append_event(self, filename: str, content: dict):
        if not self._log_dir:
            return
        path = os.path.join(self._event_dir, filename)
        with open(path, "a") as jsonl_file:
            jsonl_file.write(json.dumps(content) + "\n")
