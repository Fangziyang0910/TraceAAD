"""Convert saved maximization fitness to minimized objectives, without re-evaluation.

Run with ``python -m experiments.infra.migrations.minimize_results --root experiments_result``.
Original numeric artifacts are backed up before replacement. Model calls and source
files are immutable evidence and are never rewritten. A receipt makes each conversion
idempotent and checks candidate counts, incumbent identities and held-out values.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import time

from traceaad.common.storage import RESULT_FORMAT, append_jsonl, normalize_live_record, read_json, rows, write_json

NATIVE_MIN = {"tsp_construct", "cvrp_aco", "online_bin_packing", "vrptw_construct"}
TASKS = NATIVE_MIN | {"op_aco"}
ALIASES = dict(zip(("tsp", "cvrp", "obp", "vrptw", "op"),
                  ("tsp_construct", "cvrp_aco", "online_bin_packing", "vrptw_construct", "op_aco")))
CANONICAL = {"run_config.json", "events.jsonl", "resume.json", "summary.json",
             "selection.json", "heldout.json"}
PRESERVE = {"programs.jsonl", "calls.jsonl", "calls.jsonl.gz", "prompt_audit.jsonl"}
SCORE_FIELDS = {"eval_score", "mean_eval_score", "train_artifact_score", "train_recomputed_score",
                "mean_train_recomputed_score", "probe_score", "official_score", "profile_score",
                "combined_score", "perf", "perfs", "performance", "best_perf",
                "best_generated_algo_perf", "archived"}
NATIVE_FIELDS = {"score", "first_score", "best_score", "search_best_score", "start_score",
                 "parent_score", "selected_score", "objective", "eval_objective",
                 "mean_eval_objective", "selected_training_objective", "selection_objective"}
TEXT_FIELDS = {"prompt", "response", "code", "program", "idea", "algorithm", "description",
               "task_description", "template_program", "error", "traceback", "design", "messages"}


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def negate(value):
    if numeric(value):
        return -value if math.isfinite(value) else None
    if isinstance(value, list):
        return [negate(v) for v in value]
    if isinstance(value, dict):
        # Standard deviations, sample counts and improvements do not change sign.
        return {k: (v if k in {"std", "sd", "n", "count", "sem", "variance"} else negate(v))
                for k, v in value.items()}
    return value


def native_to_min(value, task):
    if numeric(value):
        # These original metrics are positive distances/counts. Historical files
        # contain both the metric and its old negative-fitness representation.
        return (abs(value) if task in NATIVE_MIN else -value) if math.isfinite(value) else None
    if isinstance(value, list):
        return [native_to_min(v, task) for v in value]
    if isinstance(value, dict):
        return {k: native_to_min(v, task) for k, v in value.items()}
    return value


def infer_task(path, value=None):
    if isinstance(value, dict) and value.get("task") in TASKS:
        return value["task"]
    for part in path.parts[::-1]:
        if part in TASKS or part.startswith("cob_"):
            return part
        for short, task in (("vrptw", "vrptw_construct"), ("cvrp", "cvrp_aco"),
                            ("tsp", "tsp_construct"), ("obp", "online_bin_packing"), ("op", "op_aco")):
            if re.search(rf"(?:^|[_\-]){short}(?:[_\-.]|$)", part):
                return task
    return None


def transform(value, task=None, *, canonical=False, context=()):
    if isinstance(value, list):
        return [transform(v, task, canonical=canonical, context=context) for v in value]
    if not isinstance(value, dict):
        return value
    task = value.get("task", task)
    task = ALIASES.get(str(task).lower(), task)
    result = {}
    for key, item in value.items():
        child_task = (key if key in TASKS or key.startswith("cob_") else
                      ALIASES.get(key.lower(), task))
        if key in TEXT_FIELDS and isinstance(item, (str, list)):
            result[key] = item
        elif "fitness" in key and not any(s in key for s in ("gain", "rank", "std", "tolerance", "delta")):
            result[key] = negate(item)
        elif key in SCORE_FIELDS:
            result[key] = negate(item)
        elif key == "scores":
            result[key] = negate(item)
        elif key in NATIVE_FIELDS and numeric(item):
            # Per-seed evaluation scores and baseline method-event scores used
            # the internal maximized fitness; program/exploration scores used
            # the task metric. Program scores are reconciled below.
            internal = canonical and ("evaluations" in context or "members" in context or "data" in context)
            result[key] = -item if internal else native_to_min(item, task) if task else item
        elif key in {"training", "selection"} and numeric(item) and "diagnosis_v1016_develop" in context:
            result[key] = -item
        elif key == "higher" and isinstance(item, bool):
            result[key] = False
        elif key == "result_format":
            result[key] = RESULT_FORMAT
        else:
            result[key] = transform(item, child_task, canonical=canonical, context=(*context, key))
    if canonical and "fitness" in result and "score" in result:
        result["score"] = result["fitness"]
    return result


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def backup(path, root, *, live=False):
    archive = root / ".archive/minimize_20261007"
    target = (archive / "live_completion" if live else archive) / path.relative_to(root)
    target = target.with_name(target.name + ".gz")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with path.open("rb") as source, gzip.open(target, "wb", compresslevel=1) as dest:
            shutil.copyfileobj(source, dest)
    return str(target.relative_to(root))


def convert_file(path, root, task=None, *, canonical=False, live=False):
    """Stage a file, preserving byte boundaries, then atomically replace it."""
    task = infer_task(path, None) or task
    context = ("diagnosis_v1016_develop",) if "diagnosis_v1016_develop" in path.parts else ()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as h:
        temporary = Path(h.name)
    changed, rows, frontier, new_frontier = False, 0, [], []
    incumbent = new_incumbent = None
    old_offsets, new_offsets = {0: 0}, {0: 0}
    try:
        if path.suffix == ".jsonl":
            with path.open("rb") as source, temporary.open("wb") as dest:
                while line := source.readline():
                    if not line.endswith(b"\n"):
                        raise ValueError(f"incomplete journal; stop the writer before migration: {path}")
                    before = json.loads(line)
                    after = (normalize_live_record(before, task) if live else
                             transform(before, task, canonical=canonical, context=context))
                    changed |= before != after
                    dest.write((json.dumps(after, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode())
                    rows += 1
                    old_offsets[rows], new_offsets[rows] = source.tell(), dest.tell()
                    if canonical and before.get("kind") == "candidate" and before.get("fitness") is not None:
                        old, new = before["fitness"], after["fitness"]
                        if new != (abs(old) if live else -old):
                            raise ValueError(f"fitness was not negated: {path} row {rows}")
                        reference = abs(old) if live else old
                        if incumbent is None or (reference < incumbent if live else reference > incumbent):
                            incumbent = reference
                            frontier.append(before["candidate_id"])
                        if new_incumbent is None or new < new_incumbent:
                            new_incumbent = new
                            new_frontier.append(after["candidate_id"])
            if frontier != new_frontier:
                raise ValueError(f"incumbent identities changed: {path}")
        else:
            before = json.loads(path.read_text())
            task = infer_task(path, before) or task
            after = (normalize_live_record(before, task) if live else
                     transform(before, task, canonical=canonical, context=context))
            if path.name == "run_config.json":
                after.update(objective="min", score_direction="min", native_objective=before.get(
                    "native_objective", "min" if task in NATIVE_MIN else "max"))
                if canonical:
                    after["result_format"] = RESULT_FORMAT
            changed = before != after
            temporary.write_text(json.dumps(after, ensure_ascii=False, allow_nan=False) + "\n")
        if not changed:
            return None
        original_hash = digest(path)
        archive = backup(path, root, live=live)
        if original_hash != digest(path):
            raise ValueError(f"file changed during migration: {path}")
        stat = path.stat()
        temporary.replace(path)
        import os
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        return {"path": str(path.relative_to(root)), "before_sha256": original_hash,
                "after_sha256": digest(path), "backup": archive, "rows": rows,
                "frontiers": len(frontier), "offsets": (old_offsets, new_offsets)}
    finally:
        temporary.unlink(missing_ok=True)


def convert_run(run, root):
    config = read_json(run / "run_config.json", {})
    if config.get("result_format") == RESULT_FORMAT:
        return None
    canonical = config.get("result_format") == "traceaad-results-v1"
    if not canonical:
        return None
    receipt_path = run / ".minimize/receipt.json"
    receipt = read_json(receipt_path, {"files": [], "objective": "min"})
    done = {r["path"] for r in receipt["files"]}
    task = config["task"]
    # Recover an interrupted checkpoint update before making its backup.
    if "event_offsets" in receipt and str((run / "resume.json").relative_to(root)) not in done:
        checkpoint = read_json(run / "resume.json", {})
        boundary = checkpoint.get("files", {}).get("events.jsonl")
        old, new = receipt["event_offsets"]
        if boundary not in old.values() and boundary in new.values():
            row = next(i for i, size in new.items() if size == boundary)
            checkpoint["files"]["events.jsonl"] = old[row]
            write_json(run / "resume.json", checkpoint)
    for name in ("events.jsonl", "summary.json", "selection.json", "heldout.json", "resume.json", "run_config.json"):
        path = run / name
        relative = str(path.relative_to(root))
        if not path.exists() or relative in done:
            continue
        checkpoint_original = digest(path) if name == "resume.json" else None
        checkpoint_backup = backup(path, root) if name == "resume.json" else None
        result = convert_file(path, root, task, canonical=True)
        if result:
            offsets = result.pop("offsets")
            if name == "events.jsonl":
                receipt["event_offsets"] = offsets
            receipt["files"].append(result)
            write_json(receipt_path, receipt)
        if name == "resume.json" and "event_offsets" in receipt:
            checkpoint = read_json(path)
            if "events.jsonl" in checkpoint.get("files", {}):
                old, new = receipt.get("event_offsets", ({}, {}))
                boundary = checkpoint["files"]["events.jsonl"]
                row = next((i for i, size in old.items() if size == boundary), None)
                if row is None:
                    raise ValueError(f"checkpoint is not at a complete event boundary: {run}")
                checkpoint["files"]["events.jsonl"] = new[row]
                write_json(path, checkpoint)
                if result is None:
                    result = {"path": relative, "before_sha256": checkpoint_original,
                              "backup": checkpoint_backup, "rows": 0, "frontiers": 0}
                    receipt["files"].append(result)
                result["after_sha256"] = digest(path)
                write_json(receipt_path, receipt)
    receipt.pop("event_offsets", None)
    receipt["complete"] = True
    write_json(receipt_path, receipt)
    cached = run / ".cache/history.json"
    cached.unlink(missing_ok=True)
    return {"run": str(run.relative_to(root)), "files": len(receipt["files"]),
            "frontiers": sum(r["frontiers"] for r in receipt["files"])}


def complete_live_run(run, root):
    """Finalize a legacy writer only after its terminal files stop changing."""
    marker = run / ".minimize/live.json"
    info = json.loads(marker.read_text())
    summary = json.loads((run / "summary.json").read_text())
    if summary.get("status") != "finished":
        return None
    receipt_path = run / ".minimize/live_receipt.json"
    receipt = read_json(receipt_path, {"files": [], "objective": "min"})
    for name in ("events.jsonl", "summary.json", "selection.json", "heldout.json", "resume.json"):
        path = run / name
        if not path.exists():
            continue
        # Read raw JSON here: the ordinary reader already normalizes marked runs.
        checkpoint = json.loads(path.read_text()) if name == "resume.json" else None
        original_hash = digest(path)
        original_backup = backup(path, root, live=True) if checkpoint is not None else None
        result = convert_file(path, root, info["task"], canonical=True, live=True)
        if result:
            offsets = result.pop("offsets")
            if name == "events.jsonl":
                receipt["event_offsets"] = offsets
            receipt["files"].append(result)
            write_json(receipt_path, receipt)
        if checkpoint and "event_offsets" in receipt:
            old, new = receipt["event_offsets"]
            boundary = checkpoint.get("files", {}).get("events.jsonl")
            row = next((i for i, size in old.items() if size == boundary), None)
            if row is None:
                raise ValueError(f"live checkpoint is not at an event boundary: {run}")
            updated = normalize_live_record(checkpoint, info["task"])
            updated["files"]["events.jsonl"] = new[row]
            write_json(path, updated)
            if result is None:
                result = {"path": str(path.relative_to(root)), "before_sha256": original_hash,
                          "backup": original_backup, "rows": 0, "frontiers": 0}
                receipt["files"].append(result)
            result["after_sha256"] = digest(path)
    receipt.pop("event_offsets", None)
    receipt["complete"] = True
    write_json(receipt_path, receipt)
    marker.unlink()
    (run / ".cache/history.json").unlink(missing_ok=True)
    # The old CLI may have derived diagnostics from the mixed snapshot. Rebuild
    # them from the completed minimized facts, preserving its published copy.
    if "init_attempts" in summary and (run / "run_config.json").exists():
        if (run / "diagnostics.json").exists():
            backup(run / "diagnostics.json", root, live=True)
        from experiments.infra.diagnose_search import diagnose
        diagnose(run)
    return {"run": str(run.relative_to(root)), "files": len(receipt["files"])}


def watch_live(root):
    pending = {p.parent.parent: None for p in root.rglob(".minimize/live.json")}
    print(f"waiting for {len(pending)} legacy writers; their read views already minimize", flush=True)
    while pending:
        for run in list(pending):
            summary = run / "summary.json"
            if not summary.exists() or json.loads(summary.read_text()).get("status") != "finished":
                continue
            names = ("events.jsonl", "summary.json", "resume.json", "selection.json", "heldout.json", "diagnostics.json")
            stamp = tuple((p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else None
                          for p in (run / n for n in names))
            if stamp != pending[run]:
                pending[run] = stamp
                continue
            result = complete_live_run(run, root)
            print(json.dumps(result), flush=True)
            del pending[run]
            report_path = root / ".archive/minimize_20261007/report.json"
            report = read_json(report_path, {})
            report.setdefault("live_completed", []).append(result)
            report["live_runs"] = [str(p.relative_to(root)) for p in pending]
            write_json(report_path, report)
        if pending:
            time.sleep(15)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments_result"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auxiliary", action="store_true", help="also migrate diagnostic and historical numeric artifacts")
    parser.add_argument("--watch-live", action="store_true", help="finish marked legacy writers after their terminal publication")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.watch_live:
        watch_live(root)
        return
    runs = sorted(p.parent for p in root.rglob("run_config.json") if ".archive" not in p.parts)
    pending = [r for r in runs if read_json(r / "run_config.json", {}).get("result_format") == "traceaad-results-v1"]
    if args.dry_run:
        print(json.dumps({"runs": len(pending), "root": str(root), "auxiliary": args.auxiliary}))
        return
    report_path = root / ".archive/minimize_20261007/report.json"
    report = read_json(report_path, {"objective": "min", "runs": [], "auxiliary": []})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(lambda r: convert_run(r, root), pending), 1):
            if result:
                report["runs"].append(result)
            if i % 25 == 0:
                write_json(report_path, report)
                print(f"runs {i}/{len(pending)}", flush=True)
    # The per-run receipts also cover workers completed before an interrupted
    # batch report, so restart does not lose their verification records.
    report["runs"] = [{"run": str(r.relative_to(root)), "files": len(receipt["files"]),
                       "frontiers": sum(f["frontiers"] for f in receipt["files"])}
                      for r in runs if (receipt := read_json(r / ".minimize/receipt.json", {})).get("complete")]
    write_json(report_path, report)
    if args.auxiliary:
        journal = report_path.with_name("auxiliary.jsonl")
        records = {r["path"]: r for r in [*report["auxiliary"], *rows(journal)]}
        report["auxiliary"] = list(records.values())
        completed = set(records)
        canonical_runs = {r for r in runs if read_json(r / "run_config.json", {}).get("result_format") == RESULT_FORMAT}
        paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl"}
                       and not any(s in p.parts for s in (".archive", ".cache", ".conversion", ".minimize"))
                       and p.name not in PRESERVE
                       and not (p.parent in canonical_runs and p.name in CANONICAL)
                       and str(p.relative_to(root)) not in completed)
        for i, path in enumerate(paths, 1):
            result = convert_file(path, root)
            if result:
                result.pop("offsets")
                report["auxiliary"].append(result)
                append_jsonl(journal, result)
            if i % 250 == 0:
                print(f"auxiliary {i}/{len(paths)}", flush=True)
        write_json(report_path, report)
    print(json.dumps({"runs": len(report["runs"]), "auxiliary_files": len(report["auxiliary"]),
                      "report": str(report_path)}))


if __name__ == "__main__":
    main()
