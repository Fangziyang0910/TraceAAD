from __future__ import annotations

import json
import os
from threading import Lock
from typing import Optional

from .graph import PathWiseAction, PathWiseEdge, PathWiseNode
from .population import Population
from core import Function
from baselines.profiler import ProfilerBase


class PathWiseProfiler(ProfilerBase):
    def __init__(
            self,
            log_dir: Optional[str] = None,
            *,
            initial_num_samples=0,
            log_style="complex",
            create_random_path=True,
            **kwargs,
    ):
        super().__init__(
            log_dir=log_dir,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs,
        )
        self._cur_gen = 0
        self._pop_lock = Lock()
        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, "population")
            self._event_dir = os.path.join(self._log_dir, "pathwise")
            os.makedirs(self._ckpt_dir, exist_ok=True)
            os.makedirs(self._event_dir, exist_ok=True)

    def register_population(self, pop: Population):
        if not self._log_dir:
            return
        try:
            self._pop_lock.acquire()
            if self._num_samples == 0 or pop.generation == self._cur_gen:
                return
            nodes_json = []
            for node in pop.nodes:
                nodes_json.append({
                    "node_id": node.node_id,
                    "description": node.description,
                    "rationale": node.rationale,
                    "function": str(node.function),
                    "score": node.score,
                    "parents": [parent.__dict__ for parent in node.parents],
                })
            path = os.path.join(self._ckpt_dir, f"pop_{pop.generation}.json")
            with open(path, "w") as json_file:
                json.dump(nodes_json, json_file, indent=4)
            self._cur_gen = pop.generation
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()

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

    def _write_json(self, function: Function, program="", *, record_type="history", record_sep=200):
        assert record_type in ["history", "best"]
        if not self._log_dir:
            return

        sample_order = self._num_samples
        content = {
            "sample_order": sample_order,
            "algorithm": function.algorithm,
            "function": str(function),
            "operator": function.operator,
            "score": function.score,
            "program": program,
        }

        if record_type == "history":
            lower_bound = ((sample_order - 1) // record_sep) * record_sep
            upper_bound = lower_bound + record_sep
            filename = f"samples_{lower_bound + 1}~{upper_bound}.json"
        else:
            filename = "samples_best.json"

        path = os.path.join(self._samples_json_dir, filename)
        try:
            with open(path, "r") as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(content)
        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4)
