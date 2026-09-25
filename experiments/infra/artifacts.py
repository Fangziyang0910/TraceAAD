"""Shared helpers for locating finished-run summaries and scored samples.

Supports the profiler layout (`logs/run_summary.json` + `logs/samples/`),
TraceAAD V8/V9 layout (`logs/summary.json` + `artifacts/candidates.jsonl`),
and the V9.8 explicit `best_program.py` artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_run_summary(
    run_dir: Path,
    *,
    require_finished: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    for relative in ("logs/run_summary.json", "logs/summary.json"):
        path = run_dir / relative
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if require_finished and (
            summary.get("status") != "finished" or summary.get("search_aborted")
        ):
            raise RuntimeError(f"run is not a completed search: {run_dir}")
        return summary
    raise RuntimeError(f"run is not finished: missing summary under {run_dir}/logs/")


def load_scored_samples(
    run_dir: Path,
    *,
    max_sample_order: int | None = None,
) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    records: list[dict[str, Any]] = []

    samples_dir = run_dir / "logs" / "samples"
    if samples_dir.is_dir():
        for path in sorted(samples_dir.glob("samples_*.json")):
            if path.name == "samples_best.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Cannot read sample artifact {path}: {error}") from error
            if not isinstance(data, list):
                continue
            records.extend(_filter_scored(data, max_sample_order=max_sample_order))

    best_history_path = run_dir / "best_history.jsonl"
    if not records and best_history_path.exists():
        records.extend(
            _load_best_history(best_history_path, max_sample_order=max_sample_order)
        )

    candidates_path = run_dir / "artifacts" / "candidates.jsonl"
    if not records and candidates_path.exists():
        with candidates_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Cannot parse {candidates_path}: {error}") from error
                records.extend(_filter_scored([record], max_sample_order=max_sample_order))

    # Some completed TraceAAD runs (including the local Qwen3.8 V9.7 batch,
    # V9.8, and V9.14) write the selected final program explicitly and keep its
    # score and response order in the completed run summary, without the legacy
    # candidates JSONL stream. Treat this as a single auditable scored sample.
    # V9.14 identifies the best algorithm by tree node id instead of response
    # order.
    if not records:
        best_program_path = run_dir / "best_program.py"
        summary_path = run_dir / "logs" / "summary.json"
        if best_program_path.exists() and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            score = summary.get("best_score")
            sample_order = summary.get(
                "best_response_order",
                summary.get(
                    "best_sample_order",
                    summary.get("best_algorithm_id", summary.get("best_node_id")),
                ),
            )
            if isinstance(score, (int, float)) and isinstance(sample_order, int):
                if max_sample_order is None or sample_order <= max_sample_order:
                    records.append(
                        {
                            "program": best_program_path.read_text(encoding="utf-8"),
                            "score": score,
                            "sample_order": sample_order,
                        }
                    )

    # V10.x runs keep the search-selected best node directly in
    # logs/run_summary.json (tree node id + fitness + code + operator),
    # without any legacy sample stream.
    if not records:
        summary_path = run_dir / "logs" / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            best = summary.get("best")
            if isinstance(best, dict) and isinstance(best.get("code"), str):
                score = best.get("fitness")
                sample_order = best.get("evaluation_id")
                if not isinstance(sample_order, int):
                    sample_order = best.get("node_id")
                if not isinstance(sample_order, int) and isinstance(best.get("id"), int):
                    events_path = run_dir / "events.jsonl"
                    if events_path.exists():
                        for line in events_path.read_text(encoding="utf-8").splitlines():
                            event = json.loads(line)
                            if event.get("node_id") == best["id"]:
                                sample_order = event.get("evaluation_id")
                                break
                if isinstance(score, (int, float)) and isinstance(sample_order, int):
                    if max_sample_order is None or sample_order <= max_sample_order:
                        records.append(
                            {
                                "program": best["code"],
                                "score": float(score),
                                "sample_order": sample_order,
                                "operator": best.get(
                                    "operator", best.get("origin_operator")
                                ),
                            }
                        )

    return records


def _load_best_history(
    path: Path,
    *,
    max_sample_order: int | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            score = record.get("fitness", record.get("score"))
            if not isinstance(score, (int, float)):
                continue
            program = record.get("program")
            if not isinstance(program, str):
                continue
            sample_order = record.get(
                "eval_count",
                record.get("slot", record.get("sample_order", record.get("order"))),
            )
            if not isinstance(sample_order, int):
                continue
            if max_sample_order is not None and sample_order > max_sample_order:
                continue
            out.append(
                {
                    "program": program,
                    "score": float(score),
                    "sample_order": sample_order,
                }
            )
    return out


def _filter_scored(
    data: list[Any],
    *,
    max_sample_order: int | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in data:
        if not isinstance(record, dict):
            continue
        # TraceAAD V9.7 records the same facts as ``child_fitness``/``order``.
        score = record.get("score", record.get("child_fitness"))
        if not isinstance(score, (int, float)):
            continue
        if "program" not in record and "code" not in record:
            continue
        sample_order = record.get("sample_order", record.get("order"))
        if max_sample_order is not None:
            if not isinstance(sample_order, int) or sample_order > max_sample_order:
                continue
        if "program" not in record and isinstance(record.get("code"), str):
            record = {**record, "program": record["code"]}
        out.append({**record, "score": score, "sample_order": sample_order})
    return out


def pick_best_sample(
    run_dir: Path,
    *,
    max_sample_order: int | None = None,
    sample_order: int | None = None,
    allow_incomplete: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    load_run_summary(run_dir, require_finished=not allow_incomplete)
    records = load_scored_samples(run_dir, max_sample_order=max_sample_order)
    if not records:
        raise RuntimeError(f"no valid samples found under {run_dir}")
    if sample_order is not None:
        for record in records:
            if record.get("sample_order") == sample_order:
                return record, records
        raise RuntimeError(f"sample_order={sample_order} not found among {len(records)} valid samples")
    return max(records, key=lambda record: float(record["score"])), records
