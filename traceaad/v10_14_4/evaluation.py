"""Fixed training states, seeded isolated execution, and explicit protocol IDs."""

import copy
import hashlib
import inspect
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from core import Evaluation


def fingerprint(value):
    """Stable content identity, including arrays (never their memory addresses)."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"sequence:{len(item)}".encode())
            for entry in item:
                visit(entry)
        elif isinstance(item, np.generic):
            visit(item.item())
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(json.dumps(item, allow_nan=False).encode())
            digest.update(b"\x00")
        else:
            raise TypeError(f"unfingerprintable protocol value: {type(item).__name__}")
    visit(value)
    return digest.hexdigest()


def protocol_identity(evaluation, seeds, probes, role):
    module = inspect.getmodule(type(evaluation))
    source_file = getattr(module, "__file__", None)
    settings = {key: value for key, value in vars(evaluation).items()
                if key == "_datasets" or not key.startswith("_")}
    source_files = [Path(__file__), Path(__file__).parents[2] / "core" / "evaluate.py"]
    if source_file:
        source_files.append(Path(source_file))
        if "/benchmarks/" in str(source_file):
            source_files.extend(Path(source_file).parent.glob("*.py"))
    code_hashes = {str(p.resolve().relative_to(Path(__file__).resolve().parents[2]))
                  if p.resolve().is_relative_to(Path(__file__).resolve().parents[2]) else str(p.resolve()):
                  hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(set(source_files))}
    environment = fingerprint({"python": sys.version, "numpy": np.__version__,
                               "sources": code_hashes, "settings": settings})
    return fingerprint({"environment": environment, "seeds": seeds,
                        "probes": probes, "role": role}), environment


def training_probes(evaluation, task):
    """Common inputs derived only from this evaluator's training data.

    These are a small decision panel, not a complete algorithm descriptor.
    Unsupported generic evaluators use a single unpartitioned quality frontier.
    """
    data = getattr(evaluation, "_datasets", None)
    if data is None or task is None:
        return []
    probes = []
    instances = list(data.values())[:4] if isinstance(data, dict) else list(data)[:4]
    for index, instance in enumerate(instances):
        if task in {"cvrp_aco", "op_aco"}:
            captured = []

            def capture(*args):
                captured.append(copy.deepcopy(args))
                return np.ones_like(args[0] if task == "cvrp_aco" else args[1])

            built = evaluation._build_prior(instance, capture)
            probes.append({"kind": "matrix", "args": captured[0],
                           "mask_depot": task == "op_aco", "states": aco_states(evaluation, task, built),
                           "scene": f"training instance {index}: fixed feasible ACO transition states, initial pheromone"})
        elif task == "tsp_construct":
            _, distances = instance
            current, visited = 0, {0}
            stages = {0, 1, len(distances)//2, max(0, len(distances)-6), len(distances)-2}
            for step in range(len(distances) - 1):
                eligible = np.array([int(n) for n in np.argsort(distances[current]) if n not in visited])
                if step in stages:
                    probes.append({"kind": "node", "args": (current, 0, eligible, distances.copy()),
                                   "scene": f"training instance {index}, step {step}, remaining {len(eligible)}, current {current}"})
                current = int(eligible[0])
                visited.add(current)
        elif task == "vrptw_construct":
            _, distances, demands, capacity, services, windows = instance
            current, load, now, visited = 0, 0., 0., {0}
            for step in range(3):
                eligible = np.array([n for n in range(1, len(demands)) if n not in visited
                    and load + demands[n] <= capacity
                    and max(now + distances[current, n], windows[n, 0]) <= windows[n, 1]
                    and max(now + distances[current, n], windows[n, 0]) + services[n]
                    + distances[n, 0] <= windows[0, 1]], dtype=int)
                if not len(eligible):
                    break
                probes.append({"kind": "node", "args": (current, 0, eligible, capacity-load, now,
                    demands.copy(), distances.copy(), windows.copy()),
                    "scene": f"training instance {index}, step {step}, capacity {capacity-load:g}, time {now:g}"})
                next_node = int(min(eligible, key=lambda n: distances[current, n]))
                now = max(now + distances[current, next_node], windows[next_node, 0]) + services[next_node]
                load += demands[next_node]
                current = next_node
                visited.add(current)
        elif task == "online_bin_packing":
            capacity = int(instance["capacity"])
            items = instance["items"]
            bins = np.full(len(items), capacity, dtype=int)
            stages = {0, 7, min(31, len(items)-1), len(items)//2, len(items)-1}
            for step, item in enumerate(items):
                feasible = np.flatnonzero(bins >= item)
                if not len(feasible):
                    break
                if step in stages:
                    probes.append({"kind": "ranking", "args": (int(item), bins[feasible].copy()),
                        "bin_indices": feasible.copy(), "capacity": capacity,
                        "scene": f"training instance {index}, capacity {capacity}, step {step}/{len(items)}, item {int(item)}, feasible bins {len(feasible)}"})
                best = feasible[np.argmin(bins[feasible] - item)]
                bins[best] -= int(item)
    return probes


def describe_decision(value, probe):
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError("nonfinite common-state output")
    if probe["kind"] == "node":
        if array.size != 1 or float(array.item()) != int(array.item()):
            raise ValueError("node output must be one integer")
        node = int(array.item())
        args = probe["args"]
        if node not in args[2] and not (len(args) == 8 and args[0] != 0 and node == 0):
            raise ValueError("infeasible common-state node")
        return [node]
    expected = np.asarray(probe["args"][1]).shape if probe["kind"] == "ranking" else (
        np.asarray(probe["args"][0] if not probe.get("mask_depot") else probe["args"][1]).shape)
    if array.shape != expected:
        raise ValueError("wrong common-state output shape")
    if array.ndim == 1:
        chosen = int(np.argmax(array))
        item, bins = probe["args"]
        return [chosen, int(bins[chosen]), int(bins[chosen] - item),
                int(np.count_nonzero(array == array[chosen]))]
    prior = np.maximum(array.astype(float) + 1e-9, 1e-9)
    distributions = []
    for state in probe["states"]:
        row = prior[state["current"]]
        if probe.get("mask_depot"):
            row = np.append(row, 1.)  # actual OP dummy sink heuristic
        weights = (np.asarray(state["pheromone"]) ** state["alpha"]
                   * row ** state["beta"] * np.asarray(state["mask"]))
        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            raise ValueError("invalid diagnostic ACO transition weights")
        distributions.extend((weights / total).tolist())
    return distributions


def aco_states(evaluation, task, built):
    """Capture feasibility from the real solver on a fixed baseline trajectory.

    With positive flat priors and unit initial pheromones, nonzero probability
    is exactly the solver's visit/capacity/length mask. We retain its exponent
    and pheromone values; candidate priors are later passed through that same
    weight/normalization rule. These states are not candidate rollout traces.
    """
    class Recorder:
        def __init__(self):
            self.current, self.states = 0, []
            self.rng = np.random.default_rng(20260927)
            self.solver = None

        def choice(self, size, p):
            solver = self.solver
            self.states.append({"current": self.current, "mask": (p > 0).astype(float),
                "pheromone": solver.pheromone[self.current].copy(),
                "alpha": solver.alpha, "beta": solver.beta})
            chosen = int(self.rng.choice(size, p=p))
            self.current = chosen
            return chosen

    recorder = Recorder()
    if task == "cvrp_aco":
        from benchmarks.cvrp_aco.evaluation import ACO
        distances, demands, prior = built
        solver = ACO(distances, demands, prior, evaluation.capacity, n_ants=1, rng=recorder)
        recorder.solver = solver
        solver._generate_paths()
    else:
        from benchmarks.op_aco.evaluation import ACO
        prizes, distances, prior = built
        solver = ACO(prizes, distances, evaluation.max_len, prior, n_ants=1, rng=recorder)
        recorder.solver = solver
        solver._gen_sol()
    n = len(recorder.states)
    return [recorder.states[i] for i in sorted({0, min(1, n-1), n//2, n-1})] if n else []


def decision_scene(value, probe, decision):
    result = {"scene": probe["scene"], "scope": "finite training probe; no global equivalence claim"}
    if probe["kind"] == "ranking":
        chosen, capacity, residual, ties = decision
        bins = np.asarray(probe["args"][1]).copy()
        bins[chosen] = residual
        capacities, counts = np.unique(bins, return_counts=True)
        result.update(selected_index=chosen,
                      selected_original_bin=int(probe["bin_indices"][chosen]),
                      selected_capacity=capacity, residual_after=residual, top_score_ties=ties,
                      after_residual_histogram=list(zip(capacities.tolist(), counts.tolist())),
                      after_ordered_state_sha256=fingerprint(bins))
    elif probe["kind"] == "matrix":
        n = len(probe["args"][1]) + int(probe.get("mask_depot", False))
        result["transitions"] = []
        for state, probabilities in zip(probe["states"], np.asarray(decision).reshape(-1, n)):
            top = np.argsort(-probabilities, kind="stable")[:3]
            positive = probabilities[probabilities > 0]
            result["transitions"].append({"current": state["current"],
                "feasible_count": len(positive), "top_indices": top.tolist(),
                "top_probabilities": probabilities[top].tolist(),
                "entropy": float(-np.sum(positive * np.log(positive)))})
    else:
        result["decision"] = decision
    return result


class SeededEvaluation(Evaluation):
    """Evaluate score first, then probes in a fresh namespace; probes cannot bias fitness."""

    def __init__(self, inner, probes=(), *, diagnostic_only=False):
        super().__init__(template_program=inner.template_program,
                         task_description=inner.task_description,
                         timeout_seconds=inner.timeout_seconds,
                         safe_evaluate=inner.safe_evaluate,
                         daemon_eval_process=inner.daemon_eval_process, fork_proc=inner.fork_proc)
        self.inner = inner
        self.probes = probes
        self.diagnostic_only = diagnostic_only
        if diagnostic_only:
            self.safe_evaluate = True
            self.timeout_seconds = min(5., inner.timeout_seconds or 5.)

    def evaluate_program(self, program_str, callable_func, *, seed=730241, include_probes=True, source=None):
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            # The engine submits the trusted template to SecureEvaluator and
            # passes the candidate as data, so candidate module code executes
            # once, AFTER seeding, inside the isolated process.
            program_str = source if source is not None else program_str
            random.seed(seed)
            np.random.seed(seed)
            namespace = {}
            exec(program_str, namespace)
            function = namespace[callable_func.__name__]
            evaluator = copy.copy(self.inner)
            if hasattr(evaluator, "aco_seed"):
                evaluator.aco_seed += seed
            score = 0. if self.diagnostic_only else evaluator.evaluate_program(program_str, function)
            if score is None or not np.isfinite(float(score)):
                raise ValueError("task returned no finite score")
            profile, scenes, probe_error = [], [], None
            started = time.monotonic()
            calls = 0
            if include_probes and self.probes:
                try:
                    for i, probe in enumerate(self.probes):
                        random.seed(seed + i)
                        np.random.seed((seed + i) % 2**32)
                        probe_namespace = {}
                        exec(program_str, probe_namespace)
                        calls += 1
                        value = probe_namespace[callable_func.__name__](*copy.deepcopy(probe["args"]))
                        decision = describe_decision(value, probe)
                        profile.extend(decision)
                        scenes.append(decision_scene(value, probe, decision))
                except Exception as exc:
                    profile, scenes, probe_error = [], [], f"{type(exc).__name__}: {exc}"
            return {"score": float(score), "profile": profile, "scenes": scenes,
                    "probe_error": probe_error, "probe_seconds": time.monotonic()-started,
                    "probe_calls": calls}
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
