"""CO-Bench problems as TraceAAD tasks: the program is a complete ``solve`` for one instance.

Problems, instances, feasibility checks and best-known normalization come
unchanged from CO-Bench (Sun et al., 2025; https://huggingface.co/datasets/CO-Bench/CO-Bench).
Each instance is solved in its own forked process under a wall-clock limit
(CO-Bench uses 10 s); an instance over the limit scores 0, as in CO-Bench.
A solve that raises or returns an infeasible solution fails the evaluation
with the error, so the search records it as a failed program and can repair it.
The score is CO-Bench's: the mean over cases of the mean normalized score
(best known / found for minimization, found / best known for maximization), higher is better.

Instances follow CO-Bench's development/test split (``get_dev``); ``limit``
takes a fixed, deterministic subset that cycles over the cases so that every
case size is represented.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time

from core import Evaluation

DATA_ROOT = Path(os.environ.get("COBENCH_DATA", "/home/fang/code/LLM4AD/data/CO-Bench"))

COBENCH_TASKS = {
    "cob_period_vrp": "Vehicle routing: period routing",
    "cob_crew": "Crew scheduling",
    "cob_container": "Container loading",
    "cob_jssp": "Job shop scheduling",
    "cob_steiner": "Euclidean Steiner problem",
    "cob_set_cover": "Set covering",
    "cob_graph_colour": "Graph colouring",
    "cob_cwl": "Capacitated warehouse location",
    "cob_set_partition": "Set partitioning",
    "cob_tsp": "Travelling salesman problem",
    "cob_flow_shop": "Flow shop scheduling",
    "cob_mdmkp": "Multi-Demand Multidimensional Knapsack problem",
}


def _load_config(problem):
    path = DATA_ROOT / problem / "config.py"
    spec = importlib.util.spec_from_file_location(f"cobench_{abs(hash(problem))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = path.read_text(encoding="utf-8")
    solve = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "solve")
    template = "\n".join(source.splitlines()[solve.lineno - 1:solve.end_lineno])
    return module, template


def _cases(problem):
    return sorted(f for f in os.listdir(DATA_ROOT / problem)
                  if not f.endswith(".py") and f != "__pycache__" and (DATA_ROOT / problem / f).is_file())


def _run_one(conn, solve, eval_func, instance):
    os.setpgrp()
    try:
        solution = solve(**instance)
        solution = {str(k): v for k, v in solution.items()}
        value = eval_func(**instance, **solution)
        conn.send(("ok", float(value)))
    except BaseException as exc:  # reported to the search as the program's error
        conn.send(("error", f"{type(exc).__name__}: {exc}"[:600]))
    finally:
        conn.close()


class COBenchEvaluation(Evaluation):
    def __init__(self, task, split="dev", limit=None, instance_seconds=10.0, workers=2,
                 timeout_seconds=None, **kwargs):
        if task not in COBENCH_TASKS:
            raise ValueError(f"unknown CO-Bench task {task}")
        self.task, self.problem = task, COBENCH_TASKS[task]
        self.split, self.limit = split, limit
        self.instance_seconds, self.workers = float(instance_seconds), int(workers)
        module, template = _load_config(self.problem)
        self._module = module
        dev = module.get_dev() if hasattr(module, "get_dev") else None
        selected = {}
        for case in _cases(self.problem):
            if dev is not None and case not in dev and split == "dev":
                continue
            instances = module.load_data(str(DATA_ROOT / self.problem / case))
            if dev is None:  # no published split: alternate instances
                chosen = [i for i in range(len(instances)) if (i % 2 == 0) == (split == "dev")]
            else:
                dev_ids = dev.get(case)
                if dev_ids is not None and len(dev_ids) == 0:
                    dev_ids = [0]  # CO-Bench treats an empty list as the first instance
                ids = set(dev_ids or [])
                chosen = ([i for i in (dev_ids or []) if i < len(instances)] if split == "dev"
                          else [i for i in range(len(instances)) if i not in ids])
            if chosen:
                selected[case] = (instances, chosen)
        order = []  # cycle over cases so that a limited subset covers every size
        depth = 0
        while any(depth < len(ids) for _, ids in selected.values()):
            order += [(case, ids[depth]) for case, (_, ids) in selected.items() if depth < len(ids)]
            depth += 1
        if limit is not None:
            order = order[:limit]
        self.order = order
        self._instances = {case: instances for case, (instances, _) in selected.items()}
        # What the protocol identity fingerprints: the problem and the exact instances evaluated.
        self._datasets = {"problem": self.problem, "instances": [list(item) for item in order]}
        if timeout_seconds is None:
            rounds = math.ceil(len(order) / self.workers)
            timeout_seconds = rounds * (self.instance_seconds + 3) + 30
        super().__init__(template_program=template, task_description=module.DESCRIPTION.strip(),
                         timeout_seconds=timeout_seconds, **kwargs)
        self.design_notes = (
            f"The program is evaluated on {len(order)} instances. `solve` is called once per instance in a "
            f"separate process; each call must return within {self.instance_seconds:g} seconds of wall-clock "
            f"time, otherwise that instance scores 0. A solution that violates a constraint, or an error, "
            f"makes the whole evaluation fail. The score is the mean over instances of the solution quality "
            f"relative to the best known solution (1 means equal to the best known); higher is better.")

    def instance_list(self):
        return [(case, index, self._instances[case][index]) for case, index in self.order]

    def evaluate_program(self, program_str, callable_func, **kwargs):
        ctx = mp.get_context("fork")
        eval_func = self._module.eval_func
        pending = list(self.instance_list())
        running, raw = {}, {}
        while pending or running:
            while pending and len(running) < self.workers:
                case, index, instance = pending.pop(0)
                receive, send = ctx.Pipe(duplex=False)
                process = ctx.Process(target=_run_one, args=(send, callable_func, eval_func, instance))
                process.start()
                send.close()
                running[(case, index)] = (process, receive, time.monotonic())
            time.sleep(0.02)
            for name, (process, receive, started) in list(running.items()):
                outcome = None
                if receive.poll():
                    try:
                        outcome = receive.recv()
                    except EOFError:
                        outcome = ("error", "the solve process ended without a result")
                elif not process.is_alive():
                    outcome = ("error", f"the solve process ended with exit code {process.exitcode}")
                elif time.monotonic() - started > self.instance_seconds:
                    outcome = ("timeout", None)
                if outcome is None:
                    continue
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                process.join(1)
                receive.close()
                del running[name]
                if outcome[0] == "error":
                    for other, _, _ in running.values():
                        try:
                            os.killpg(other.pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
                    raise RuntimeError(f"instance {name[0]}#{name[1]}: {outcome[1]}")
                raw[name] = outcome[1] if outcome[0] == "ok" else "timeout"
        if all(value == "timeout" for value in raw.values()):
            raise RuntimeError(f"every instance exceeded the {self.instance_seconds:g} s limit")
        return self.normalize(raw)

    def normalize(self, raw):
        """CO-Bench's normalization applied to the selected instances, then its average."""
        results = {}
        for case in {case for case, _ in raw}:
            scores = [None] * len(self._instances[case])
            for (c, index), value in raw.items():
                if c == case:
                    scores[index] = value
            results[case] = (scores, None)
        if hasattr(self._module, "norm_score"):
            results = self._module.norm_score(results)
        per_case = []
        for case, (scores, _) in results.items():
            picked = [scores[index] for c, index in raw if c == case]
            per_case.append(sum(v if isinstance(v, (int, float)) else 0.0 for v in picked) / len(picked))
        return sum(per_case) / len(per_case)
