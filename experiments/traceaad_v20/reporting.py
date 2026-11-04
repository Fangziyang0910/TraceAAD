"""Rebuild public experiment views from the search's own records.

The search writes its own records under ``directory/native``. This adapter only
writes derived files in ``directory``; it never participates in search/resume.
Publish after construction, after each finished attempt, and with SearchResult
after ``run()`` returns. Each attempt contributes at most one candidate event;
its budget position includes every Evaluate, including repair.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

from experiments.infra.storage import read_object, write_json


BUDGET_AXIS = "评价次数"
BUDGET_BASIS = "evaluate_calls"


def _timestamp(path, fallback):
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo("Asia/Shanghai")).isoformat()
    except FileNotFoundError:
        return fallback


def _beijing(seconds):
    return datetime.fromtimestamp(seconds, ZoneInfo("Asia/Shanghai")).isoformat()


def _replace_text(path, content):
    """Replace the inode so cached Programs/TrainingHistory readers restart."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as out:
            temporary = Path(out.name)
            out.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _jsonl(records):
    # Keep kind first: batch_status.journal_tail recognizes it before parsing.
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")) + "\n" for row in records)


def _program(program):
    record = program.to_dict()
    record.pop("code")
    evaluation = record["evaluation"] or {}
    return {
        **record,
        "action": program.operator,
        "idea": program.design or "",
        "status": {"success": "valid", "error": "eval_failed", "timeout": "timeout"}.get(
            evaluation.get("status"), "example"),
        "valid": program.valid,
        "fitness": evaluation.get("fitness"),
        "reason": evaluation.get("failure"),
    }


def _delivery_failure(attempt):
    last = attempt.calls[-1] if attempt.calls else {}
    if last.get("format_failure"):
        return "invalid_output", last["format_failure"]
    return "delivery_failed", last.get("error") or "No evaluated program was produced"


def _token_usage(calls):
    """Sum reported usage, including partial usage, without estimating tokens.

    Totals cover only reported fields. Calls with partial or absent usage remain
    explicit; raw provider usage is also preserved in each attempt's call list.
    """
    totals = dict.fromkeys(("calls", "calls_with_usage", "calls_without_usage",
                           "calls_with_partial_usage", "calls_with_reasoning_usage",
                           "prompt_tokens", "completion_tokens", "total_tokens",
                           "reasoning_tokens", "cached_prompt_tokens"), 0)
    for call in calls:
        usage = call.get("usage") or {}
        counts = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                  if type(usage.get(key)) is int and usage[key] >= 0}
        full = "prompt_tokens" in counts and "completion_tokens" in counts
        if full:
            counts.setdefault("total_tokens", counts["prompt_tokens"] + counts["completion_tokens"])
        totals["calls"] += 1
        totals["calls_with_usage" if counts else "calls_without_usage"] += 1
        totals["calls_with_partial_usage"] += bool(counts) and not full
        for key, count in counts.items():
            totals[key] += count
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if type(reasoning) is int and reasoning >= 0:
            totals["calls_with_reasoning_usage"] += 1
            totals["reasoning_tokens"] += reasoning
        cached = (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens", usage.get("prompt_cache_hit_tokens"))
        if type(cached) is int and cached >= 0:
            totals["cached_prompt_tokens"] += cached
    return totals


class Reporter:
    """Public records for ``TraceAAD(run_dir=directory / 'native')``.

    ``metadata`` is the full public run_config supplied by the experiment runner.
    All fields are retained except the budget and its axis/basis, which follow
    the independent search configuration. ``publish`` returns the public summary.
    """

    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.metadata = deepcopy(metadata)
        self.directory.mkdir(parents=True, exist_ok=True)
        previous = read_object(self.directory / "summary.json")
        fallback = previous.get("started_at") or datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        self.started_at = self.metadata.get("started_at") or _timestamp(
            self.directory / "native" / "run_config.json", fallback)

    def _progress(self, used, attempts, phase):
        # Legacy readers call the budget counter "attempts". Actual generation
        # attempts live in generation_attempts here and summary.attempts.
        return {"attempts": used, "generation_attempts": attempts, "phase": phase,
                "started_at": self.started_at, "budget_axis": BUDGET_AXIS,
                "budget_basis": BUDGET_BASIS}

    def publish(self, search, result=None):
        native = self.directory / "native"
        metadata = {**self.metadata, "budget": search.config.evaluation_budget,
                    "budget_axis": BUDGET_AXIS, "budget_basis": BUDGET_BASIS}
        programs = {key: _program(value) for key, value in sorted(search.archive.items())}
        calls = [call for attempt in search.attempts for call in attempt.calls]
        evaluations = [record for attempt in search.attempts for record in attempt.evaluations]
        valid = sum(program["valid"] for program in programs.values())
        program_count = sum(key != 0 for key in programs)
        best = programs.get(search.best.id) if search.best is not None else None
        initialized = sum(p["valid"] and p["action"] == "Init" for p in programs.values())
        phase = "initialization" if initialized < search.config.initial_size else "evolution"
        status = "running"
        if result is not None:
            # These are existing public terminal states, understood by remote
            # sync/watch. Preserve the native status separately, including failed.
            status = "finished" if result.status == "finished" else "no_valid_root"
            phase = "finished" if status == "finished" else "selection_failed"

        events = []
        if 0 in programs:
            events.append({"kind": "program", "program": programs[0], "ts": self.started_at})
        used, last_at = 0, self.started_at
        for count, attempt in enumerate(search.attempts, 1):
            used = max([used, *(record["id"] for record in attempt.evaluations)])
            last_at = _beijing(attempt.finished_at) if attempt.finished_at else last_at
            program = programs.get(attempt.program_id)
            if program is None:
                outcome, reason = _delivery_failure(attempt)
            else:
                outcome, reason = program["status"], program["reason"]
            events.append({
                "kind": "candidate" if program is not None else "delivery_failed",
                "candidate_id": attempt.id,
                "node_id": attempt.program_id,
                "budget_used": used,
                "x_label": BUDGET_AXIS,
                "budget_axis": BUDGET_AXIS,
                "budget_basis": BUDGET_BASIS,
                "operator": attempt.operator,
                "status": outcome,
                "valid": bool(program and program["valid"]),
                "fitness": program["fitness"] if program else None,
                "program": program,
                "attempt": {**asdict(attempt), "action": attempt.operator,
                            "idea": program["idea"] if program else "", "reason": reason,
                            "model_calls": attempt.model_calls},
                "ts": last_at,
                "progress": self._progress(
                    used, count, "initialization" if attempt.operator == "Init" else "evolution"),
            })
        finished_at = _timestamp(native / "result.json", last_at) if result is not None else None
        events.append({"kind": "progress", "ts": finished_at or last_at,
                       "progress": self._progress(search.evaluations_used, len(search.attempts), phase)})
        auxiliary = {}
        for attempt in search.attempts:
            for request in attempt.record.get("aux", []):
                row = auxiliary.setdefault(request["role"], {"requests": 0, "errors": 0, "tokens": {}})
                # An embedding row already sums that attempt's embedding requests.
                row["requests"] += request.get("requests", 1)
                row["errors"] += request.get("errors", int(request["status"] != "success"))
                for key, value in (request.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        row["tokens"][key] = row["tokens"].get(key, 0) + value
        summary = {"auxiliary":auxiliary,
            "status": status,
            "native_status": result.status if result is not None else "running",
            "source": "native/",
            "phase": phase,
            "method": metadata.get("method"),
            "budget": metadata["budget"],
            "budget_used": search.evaluations_used,
            "budget_axis": BUDGET_AXIS,
            "budget_basis": BUDGET_BASIS,
            # TrainingHistory also consults this counter if an interrupted
            # Evaluate consumed budget without producing an individual.
            "method_sample_count": search.evaluations_used,
            "evaluation_calls": search.evaluations_used,
            "evaluation_outcomes": dict(Counter(record["status"] for record in evaluations)),
            "attempts": len(search.attempts),
            "candidate_count": len(search.attempts),
            "valid_candidate_count": valid,
            "program_count": program_count,
            "num_nodes": valid,
            "failed_programs": program_count - valid,
            "repair_attempts": sum(attempt.repair_used for attempt in search.attempts),
            "format_failures": sum(bool(call.get("format_failure")) for call in calls),
            "model_calls": search.model_calls,
            "token_usage": _token_usage(calls),
            "best": best,
            "started_at": self.started_at,
            "stop_reason": result.stop_reason if result is not None else None,
        }
        if result is not None:
            summary.update(finished_at=finished_at, final_selection="training",
                           selection_evaluations=0)

        write_json(self.directory / "run_config.json", metadata)
        _replace_text(self.directory / "programs.jsonl", _jsonl(
            {"id": program.id, "code": program.code}
            for _, program in sorted(search.archive.items())))
        _replace_text(self.directory / "events.jsonl", _jsonl(events))
        if result is not None:
            write_json(self.directory / "selection.json", {
                "criterion": "training", "selected_node": best["id"] if best else None,
                "finalists": [], "results": [],
            })
        else:
            (self.directory / "selection.json").unlink(missing_ok=True)
        if result is not None and best is not None:
            _replace_text(self.directory / "best_program.py", search.archive[best["id"]].code)
        else:
            (self.directory / "best_program.py").unlink(missing_ok=True)
        # Publish the terminal marker after its selected source and identity.
        write_json(self.directory / "summary.json", summary)
        return summary
