"""Whole-task fallback for experiments outside the current CO suite."""

import math
import statistics
import time

from core import SecureEvaluator
from core.evaluate import EVALUATION_SEED
from traceaad.evaluation import SeededEvaluation, clean_traceback, failing_line


class WholeTaskEvaluator:
    """Run all configured seeds and return fitness, failure and measurements.

    This whole-task evaluator serves direct evaluations and tasks outside the
    fixed CO suite. Fixed-suite search and held-out use InstanceProgramEvaluator.
    """

    def __init__(self, evaluation, seeds=(EVALUATION_SEED,), role="search"):
        self.seeded = SeededEvaluation(evaluation)
        self.evaluator = SecureEvaluator(self.seeded)
        self.seeds, self.role = tuple(seeds), role

    def evaluate(self, code, source_key):
        records = []
        for seed in self.seeds:
            started = time.monotonic()
            self.seeded.reset()
            outcome = self.evaluator.evaluate_program_with_details(code, seed=seed)
            elapsed = time.monotonic() - started
            measured = self.seeded.measured(until=started + elapsed)
            if (
                outcome.failure_kind == "timeout"
                and self.seeded.timeout_seconds is not None
                and measured["function_seconds"] is not None
            ):
                measured["function_seconds"] = min(
                    measured["function_seconds"], self.seeded.timeout_seconds
                )
            value = outcome.result
            valid = (
                isinstance(value, dict)
                and type(value.get("score")) in (int, float)
                and math.isfinite(value["score"])
            )
            records.append(
                {
                    "key": source_key,
                    "role": self.role,
                    "seed": seed,
                    "valid": valid,
                    "score": value["score"] if valid else None,
                    "failure_kind": outcome.failure_kind,
                    "error_type": outcome.error_type,
                    "error": outcome.error,
                    "traceback": outcome.traceback,
                    "seconds": elapsed,
                    "cpu_seconds": outcome.cpu_seconds,
                    **measured,
                }
            )
            if not valid:
                kind = (
                    "timeout"
                    if outcome.failure_kind == "timeout"
                    else "invalid_output"
                    if outcome.failure_kind == "invalid_result"
                    else "runtime_error"
                )
                error = clean_traceback(outcome.traceback or outcome.error or kind, code)
                return {
                    "fitness": None,
                    "failure": {
                        "kind": kind,
                        "error": error,
                        "line": failing_line(error, code),
                        "seconds": elapsed,
                        **measured,
                    },
                    "seconds": elapsed,
                    **measured,
                    "evaluations": records,
                }
        return {
            "fitness": statistics.fmean(r["score"] for r in records),
            "failure": None,
            "seconds": statistics.fmean(r["seconds"] for r in records),
            "calls": round(statistics.fmean(r["calls"] for r in records)),
            "function_seconds": statistics.fmean(r["function_seconds"] for r in records),
            "call_running": False,
            "evaluations": records,
        }
