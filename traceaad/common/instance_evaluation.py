"""Bounded instance parallelism with a separate deadline and probe per instance.

Two limits apply to each instance. The function budget bounds the CPU time
spent inside the candidate function, summed over its calls: it is the computation
the task grants the candidate, and the fixed solver's time does not count,
since that time varies with the core and load but not with the candidate.
CPU time rather than wall time, so that waiting while other processes use
the core does not count.
The instance limit bounds the whole instance (process start, solver and
function) and only stops programs that hang.
"""

import copy
import math
import multiprocessing
import os
from multiprocessing.connection import wait
import statistics
import time

import numpy as np

from core import SecureEvaluator
from core.evaluate import EVALUATION_SEED, EvaluationOutcome, _stop_eval_process
from core.scheduling import SchedulerSession
from benchmarks.tasks import FUNCTION_SECONDS
from .config import REVISION
from .evaluation import SeededEvaluation, clean_traceback, failing_line, fingerprint, protocol_identity

EXECUTION = "isolated-instances-v1"


def _positive(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def validate_execution(timeout_seconds, n_workers, function_seconds=None):
    if not _positive(timeout_seconds):
        raise ValueError("timeout_seconds must be a finite positive per-instance limit")
    if function_seconds is not None and not _positive(function_seconds):
        raise ValueError("function_seconds must be a finite positive per-instance budget")
    if type(n_workers) is not int or n_workers < 1:
        raise ValueError("n_workers must be a positive integer")


def _instance_view(evaluation, index, timeout):
    """Reuse initialized data and fixed solvers, including the original ACO stream."""
    view = copy.copy(evaluation)
    if hasattr(evaluation, "_datasets"):
        view._datasets = evaluation._datasets[index:index + 1]
    if hasattr(evaluation, "_rows"):
        view._rows = evaluation._rows[index:index + 1]
        view._instances = evaluation._instances[index:index + 1]
    if hasattr(evaluation, "aco_seed"):
        # ACO uses aco_seed + index; the one-instance view enumerates from zero.
        view.aco_seed += index
    view.n_instance = 1
    view.n_workers = 1
    view.timeout_seconds = timeout
    return view


class _ResultPipe:
    """The safe evaluator's queue interface, without a queue feeder thread."""

    def __init__(self, connection):
        self.connection = connection

    def put(self, outcome):
        self.connection.send((outcome, time.monotonic()))
        self.connection.close()


def _run_instance(evaluator, code, function_name, sender, seed, cpu, scheduler_socket):
    # A forked child must not keep the parent's allocation connection alive.
    if scheduler_socket is not None:
        scheduler_socket.close()
    output = _ResultPipe(sender)
    try:
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})
    except OSError as exc:
        output.put(EvaluationOutcome(None, 'runtime_error', type(exc).__name__, str(exc)))
        return
    evaluator._evaluate_in_safe_process_with_details(code, function_name, output, seed=seed)


class InstanceProgramEvaluator:
    """Evaluate one program with runtime limits supplied on each call.

    ``function_seconds`` is each instance's budget inside the candidate function;
    ``timeout_seconds`` bounds each whole instance, including the fixed solver
    and program initialization. Waiting for a worker is excluded.
    Instance i starts from candidate seed (seed + i) mod 2**32, with fresh globals;
    worker count and completion order do not select a different random stream.
    ``seconds`` is wall time; ``instance_seconds`` sums the instance durations.
    """

    def __init__(self, evaluation, seeds=(730241,), role="search", *, measure_calls=True,
                 timeout_seconds=10, n_workers=1, scheduler_socket=None, function_seconds=FUNCTION_SECONDS):
        validate_execution(timeout_seconds, n_workers, function_seconds)
        if function_seconds is not None and not measure_calls:
            raise ValueError("a function budget needs the call measurement")
        self.evaluation, self.seeds, self.role = evaluation, tuple(seeds), role
        self.measure_calls, self.function_seconds = measure_calls, function_seconds
        self.timeout_seconds, self.n_workers = timeout_seconds, n_workers
        self.scheduler_socket = scheduler_socket
        self.n_instance = int(getattr(evaluation, "n_instance", 1))
        if self.n_instance < 1:
            raise ValueError("evaluation must contain at least one instance")
        if self.n_instance > 1 and not hasattr(evaluation, "_datasets"):
            raise ValueError("instance evaluation requires an indexed dataset")
        if hasattr(evaluation, "_datasets") and not isinstance(evaluation._datasets, (list, tuple, np.ndarray)):
            raise ValueError("instance evaluation requires a list of independent instances")
        if hasattr(evaluation, "_datasets") and len(evaluation._datasets) != self.n_instance:
            raise ValueError("n_instance must match the initialized dataset")
        _, self.environment = protocol_identity(evaluation, self.seeds, role, measure_calls)
        self.protocol = self.protocol_for(timeout_seconds)

    def protocol_for(self, timeout_seconds):
        return fingerprint({"environment": self.environment, "seeds": self.seeds,
                            "role": self.role, "measure_calls": self.measure_calls,
                            "execution": EXECUTION, "revision": REVISION,
                            "timeout_seconds": float(timeout_seconds),
                            "function_seconds": self.function_seconds, "function_clock": "process_cpu",
                            "timeout_scope": "instance", "candidate_seed": "seed+index:uint32"})

    def _instances(self, code, seed, timeout, workers):
        budget = self.function_seconds
        context = multiprocessing.get_context("spawn" if self.evaluation.fork_proc is False else "fork")
        active, completed, next_index = {}, {}, 0
        cpus, released = {}, []
        session, peak_workers, queue_seconds = None, 0, 0.0
        count = self.n_instance
        function_name = SecureEvaluator(self.evaluation)._target_function_name()

        def finish(index, outcome, finished):
            process, receiver, seeded, started = active.pop(index)
            measured = seeded.measured(until=finished, pid=process.pid)
            _stop_eval_process(process)
            receiver.close()
            if process.is_alive():  # the bounded joins above can expire on a loaded host
                process.kill()
                process.join()
            try:
                process.close()
            except ValueError:
                # Evaluations in several threads: another thread's Process.start()
                # can reap this child first, and poll() then reports it as running.
                pass
            value = outcome.result
            valid = isinstance(value, dict) and isinstance(value.get("score"), (int, float)) and math.isfinite(value["score"])
            row = {"instance_index": index, "seed": ((EVALUATION_SEED if seed is None else seed) + index) % 2**32,
                   "valid": valid, "score": value["score"] if valid else None,
                   "failure_kind": outcome.failure_kind, "error_type": outcome.error_type,
                   "error": outcome.error, "traceback": outcome.traceback,
                   "seconds": max(0., finished - started), "cpu_seconds": outcome.cpu_seconds, **measured}
            completed[index] = row
            if session is not None:
                row['cpu_id'] = cpus.pop(index)
                released.append(row['cpu_id'])
            return row

        try:
            if self.scheduler_socket:
                session = SchedulerSession(self.scheduler_socket, 'cpu', f'{os.getpid()}:{self.role}')
                workers = min(workers, session.status()['cpu']['capacity'])
            while next_index < count or active:
                if session is not None:
                    assigned = session.poll(min(workers, count - next_index + len(active)), released)
                    released.clear()
                else:
                    assigned = [None] * min(count - next_index, workers - len(active))
                for cpu in assigned:
                    index = next_index
                    view = _instance_view(self.evaluation, index, timeout)
                    seeded = SeededEvaluation(view, self.measure_calls)
                    evaluator = SecureEvaluator(seeded)
                    receiver, sender = context.Pipe(duplex=False)
                    process = context.Process(target=_run_instance,
                        args=(evaluator, code, function_name, sender,
                              ((EVALUATION_SEED if seed is None else seed) + index) % 2**32,
                              cpu, session.sock if session else None))
                    started = time.monotonic()
                    try:
                        process.start()
                    except BaseException:
                        receiver.close()
                        sender.close()
                        raise
                    sender.close()
                    active[index] = (process, receiver, seeded, started)
                    cpus[index] = cpu
                    peak_workers = max(peak_workers, len(active))
                    next_index += 1
                if not active:
                    waiting = time.monotonic()
                    time.sleep(0.05)
                    queue_seconds += time.monotonic() - waiting
                    continue
                deadline = min(r[3] + timeout for r in active.values())
                # The function budget is read from the probe, so it is polled.
                poll = 0.05 if session or budget is not None else timeout
                wait([handle for p, receiver, _, _ in active.values() for handle in (receiver, p.sentinel)],
                     timeout=min(poll, max(0., deadline - time.monotonic())))
                failed = False
                for index, (process, receiver, seeded, started) in list(active.items()):
                    outcome, finished = None, time.monotonic()
                    if receiver.poll():
                        try:
                            outcome, finished = receiver.recv()
                        except EOFError:
                            pass
                    inside = seeded.measured(until=finished, pid=process.pid)["function_cpu_seconds"]
                    if budget is not None and inside > budget:
                        outcome = EvaluationOutcome(None, "timeout", "FunctionBudgetExceeded",
                                                    f"instance {index} used more than {budget:g}s of CPU inside the function")
                    elif finished - started >= timeout:
                        outcome = EvaluationOutcome(None, "timeout", "TimeoutError",
                                                    f"instance {index} exceeded {timeout:g}s")
                    elif outcome is None and not process.is_alive():
                        outcome = EvaluationOutcome(None, "runtime_error", "WorkerExit",
                                                    f"instance {index} worker exited with code {process.exitcode}")
                    if outcome is not None:
                        failed |= not finish(index, outcome, finished)["valid"]
                if failed:
                    break
        finally:
            # Failure, interruption, or preparation error releases every running
            # instance, including candidate-created descendants.
            for index in list(active):
                finish(index, EvaluationOutcome(None, "cancelled", "Cancelled",
                                                "another instance failed or evaluation was interrupted"), time.monotonic())
            if session is not None:
                session.close()
        return [completed[i] for i in sorted(completed)], {
            'socket': self.scheduler_socket, 'peak_workers': peak_workers, 'queue_seconds': queue_seconds}

    def evaluate(self, code, source_key, *, timeout_seconds=None, n_workers=None):
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        workers = self.n_workers if n_workers is None else n_workers
        validate_execution(timeout, workers)
        workers = min(workers, self.n_instance)
        records = []
        for seed in self.seeds:
            started = time.monotonic()
            instances, scheduling = self._instances(code, seed, timeout, workers)
            elapsed = time.monotonic() - started
            failed = next((r for r in instances if not r["valid"] and r["failure_kind"] != "cancelled"), None)
            measured = {"calls": sum(r["calls"] for r in instances) if self.measure_calls else None,
                        "function_seconds": sum(r["function_seconds"] for r in instances) if self.measure_calls else None,
                        "function_cpu_seconds": sum(r["function_cpu_seconds"] for r in instances) if self.measure_calls else None,
                        "call_running": any(r["call_running"] for r in instances)}
            instance_seconds = sum(r["seconds"] for r in instances)
            record = {"key": source_key, "role": self.role, "protocol": self.protocol_for(timeout),
                      "seed": seed, "valid": failed is None,
                      "score": statistics.fmean(r["score"] for r in instances) if failed is None else None,
                      "failure_kind": failed["failure_kind"] if failed else None,
                      "error_type": failed["error_type"] if failed else None,
                      "error": failed["error"] if failed else None,
                      "traceback": failed["traceback"] if failed else None,
                      "seconds": elapsed, "instance_seconds": instance_seconds,
                      "cpu_seconds": sum(r["cpu_seconds"] or 0. for r in instances),
                      "timeout_seconds": timeout, "function_seconds_limit": self.function_seconds,
                      "timeout_scope": "instance", "n_workers": workers,
                      "n_instances": self.n_instance,
                      "instances": instances, **measured}
            if self.scheduler_socket:
                record['scheduler'] = scheduling
                record['n_workers'] = scheduling['peak_workers']
                record['worker_limit'] = workers
            records.append(record)
            if failed:
                kind = "timeout" if failed["failure_kind"] == "timeout" else "invalid_output" if failed["failure_kind"] == "invalid_result" else "runtime_error"
                error = clean_traceback(failed["traceback"] or failed["error"] or kind, code)
                return {"fitness": None, "failure": {"kind": kind, "error": error,
                        "instance_index": failed["instance_index"], "line": failing_line(error, code),
                        "error_type": failed["error_type"], "seconds": failed["seconds"],
                        "timeout_seconds": timeout, "function_seconds_limit": self.function_seconds,
                        **{name: failed[name] for name in ("calls", "function_seconds", "function_cpu_seconds", "call_running")}},
                        "seconds": elapsed, "instance_seconds": instance_seconds, **measured, "evaluations": records}
        return {"fitness": statistics.fmean(r["score"] for r in records), "failure": None,
                "seconds": statistics.fmean(r["seconds"] for r in records),
                "instance_seconds": statistics.fmean(r["instance_seconds"] for r in records),
                "calls": round(statistics.fmean(r["calls"] for r in records)) if self.measure_calls else None,
                "function_seconds": statistics.fmean(r["function_seconds"] for r in records) if self.measure_calls else None,
                "function_cpu_seconds": statistics.fmean(r["function_cpu_seconds"] for r in records) if self.measure_calls else None,
                "call_running": False, "evaluations": records}
