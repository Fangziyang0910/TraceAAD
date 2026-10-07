"""Evaluate the evolved component in its fixed outer solver on content-checked data."""

from pathlib import Path
import numpy as np

from core import Evaluation
from . import dataset, fssp, graph, mdmkp, set_cover
from .templates import DESCRIPTIONS, FUNCTION_NAMES, TEMPLATES

SOLVERS = {"fssp_gls": fssp.solve, "mdmkp_search": mdmkp.solve,
           "graph_colouring": graph.solve, "set_cover_construct": set_cover.solve}


class AHDEvaluation(Evaluation):
    def __init__(self, task, split="train", timeout_seconds=60, data_root=None, limit=None,
                 iterations=10, local_passes=3, search_steps=32, repair_steps=100,
                 reduction_attempts=4, improvement_steps=16, settings=None, **kwargs):
        if task not in dataset.TASKS:
            raise ValueError(f"unknown AHD task: {task}")
        super().__init__(template_program=TEMPLATES[task], task_description=DESCRIPTIONS[task],
                         timeout_seconds=timeout_seconds, **kwargs)
        self.task, self.suite_protocol = task, dataset.PROTOCOL
        self._data_root = Path(data_root) if data_root is not None else dataset.DATA_ROOT
        rows = dataset.records(task, split, self._data_root)
        self.split = rows[0]['split']
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            rows = rows[:limit]
        self._rows = rows
        self._datasets = [{k: r[k] for k in ("id", "group", "scale", "dimensions", "content", "sha256", "reference", "reference_kind")}
                          for r in rows]
        self._instances = [dataset.load(r, self._data_root) for r in rows]
        self.n_instance = len(rows)
        self.problem_size = max(r["scale"] for r in rows)
        self.score_meaning = "the mean relative deviation from the stored reference, in percent (lower is better)"
        self.instance_description = dataset.describe(task, self.split, self._data_root) if limit is None else f"{len(rows)} fixed {self.split} instances"
        defaults = {"fssp_gls": {"iterations": iterations, "local_passes": local_passes},
                    "mdmkp_search": {"steps": search_steps},
                    "graph_colouring": {"repair_steps": repair_steps, "reduction_attempts": reduction_attempts},
                    "set_cover_construct": {"improvement_steps": improvement_steps}}[task]
        settings = dict(defaults if settings is None else settings)
        if set(settings) != set(defaults):
            raise ValueError("outer-solver settings do not match this task")
        if any(type(v) is not int or v < 1 for v in settings.values()):
            raise ValueError("outer-solver iteration counts must be positive integers")
        self.outer_settings = settings
        limit_text = f"{timeout_seconds:g} seconds" if timeout_seconds is not None else "no configured limit"
        direction = "(reference - profit)" if task == "mdmkp_search" else "(objective - reference)"
        references = ", ".join(sorted({r["reference_kind"] for r in rows}))
        self.design_notes = (
            f"Evaluation uses {self.instance_description}. The WHOLE dataset evaluation has {limit_text}. "
            f"The fixed outer-solver settings are {settings}. The target is called repeatedly inside this solver; "
            "its return is consumed exactly as specified in the template. The program is executed afresh "
            "for each instance, so candidate globals are reset between instances. Inputs passed to the "
            "target are copies; changing them does not change solver state. Scores are averaged equally "
            f"over instances: 100 * {direction} / abs(reference). Lower is better. "
            f"References for this split are: {references}. They are hidden from the candidate and are "
            "not guaranteed integer optima. Negative deviations mean the reference was improved. "
            "Generated reference bounds/baselines differ from published standard references, so their "
            "deviations must be reported separately. Score computation does not clip negative deviations. Invalid return "
            "contracts or errors fail the evaluation; score quality is always computed from the final valid solution.")
        # Baselines read task_description directly; TraceAAD also reads design_notes.
        self.task_description += "\n\n" + self.design_notes
        if task == "fssp_gls":
            # Compile/cache fixed kernels before candidate timing; all candidates use identical kernels.
            tiny = np.ones((2, 2), dtype=float)
            fssp.neh(tiny)
            fssp.best_neighbor(np.arange(2), tiny, np.arange(2))

    def evaluate_program(self, program_str, callable_func):
        values = []
        for row, original in zip(self._rows, self._instances):
            namespace = {}
            exec(program_str, namespace)
            function = namespace[FUNCTION_NAMES[self.task]]
            data = {k: v.copy() for k, v in original.items()}
            _, objective = SOLVERS[self.task](data, function, **self.outer_settings)
            reference = row["reference"]
            delta = reference - objective if self.task == "mdmkp_search" else objective - reference
            values.append(100.0 * delta / abs(reference))
        result = float(np.mean(values))
        if not np.isfinite(result):
            raise ValueError("outer solver produced a non-finite objective")
        return result
