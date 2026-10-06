"""Compact, read-only status for canonical batch manifests, locally or over SSH."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time

from experiments.infra.monitor_results import SCALES, heldout_identity, scale_of_split
from traceaad.common.storage import Programs, read_json as result_json
from experiments.infra.monitor_timing import batch_timing, search_timing


TAIL_BYTES = 1_048_576
PROGRESS_RECORD = re.compile(rb'^\s*\{\s*"kind"\s*:\s*"(?:progress|candidate)"')


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def journal_tail(path):
    """Read at most 1 MiB; discard incomplete records without a full-file fallback."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - TAIL_BYTES))
            data = stream.read(TAIL_BYTES)
        lines = data.splitlines(keepends=True)
        if size > TAIL_BYTES:
            lines = lines[1:]
    except OSError:
        return {}, {}, None
    checkpoint, candidate = {}, {}
    for line in reversed(lines):
        if not line.endswith(b"\n") or not PROGRESS_RECORD.match(line):
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        if record.get("progress") and not checkpoint:
            checkpoint = record.get("progress") or {}
        if record.get("kind") == "candidate" and not candidate:
            candidate = record
        if checkpoint and candidate:
            break
    return checkpoint, candidate, path.stat().st_mtime


def run_path(root, row):
    """Resolve a copied manifest by its task/name, not the original host's path."""
    task = row["task"]
    name = row.get("run_name") or Path(row["run_dir"]).name
    if any(Path(part).name != part or part in {".", ".."} for part in (task, name)):
        raise ValueError("task and run name must be single path components")
    return root / task / name


def expected_splits(task):
    if task == "online_bin_packing":
        return [f"eval_{size * 1000}_{capacity}" for size, capacity in
                ((1, 100), (1, 500), (5, 100), (5, 500), (10, 100), (10, 500))]
    prefix = "test" if task in {"cvrp_aco", "op_aco"} else "eval"
    return [f"{prefix}_{size}" for size in SCALES[task]]


def run_status(root, row, budget, now):
    directory = run_path(root, row)
    summary_path = directory / "summary.json"
    summary = read_json(summary_path)
    checkpoint, candidate, modified = journal_tail(directory / "events.jsonl") if summary.get("status") != "finished" else ({}, {}, None)
    phase = checkpoint.get("phase") or summary.get("phase")
    raw = summary.get("status")
    resumed = modified is not None and summary_path.exists() and modified > summary_path.stat().st_mtime
    if raw == "finished":
        status = "finished"
    elif raw and raw != "running" and not resumed:
        status = "blocked"
    elif checkpoint or candidate:
        status = "recorded"  # Journal activity alone does not prove a process is alive.
    else:
        status = "unknown"
    counts = [value for value in (checkpoint.get("attempts"), candidate.get("budget_used"), summary.get("budget_used"))
              if isinstance(value, int) and value >= 0]
    used = max(counts) if counts else None
    snapshot = {"completed": used, "elapsed": checkpoint.get("elapsed") if checkpoint.get("attempts") == used else None,
                "started_at": checkpoint.get("started_at"), "phase": phase,
                "completed_at": datetime.fromtimestamp(modified, timezone.utc).isoformat() if modified else None}
    timing_row = {"budget_used": used or 0, "budget": summary.get("budget", budget),
                  "status": "running" if status == "recorded" else status,
                  "updated_at": snapshot["completed_at"]}
    timing = search_timing(timing_row, summary, snapshot, unit="候选", now=now)
    if timing["state"] == "stale" and status == "recorded":
        status = "stale"
    best = summary.get("best") or {}
    key = best.get("key")
    program = directory / "best_program.py"
    code = program.read_text(encoding="utf-8") if program.exists() else Programs(directory).get(key)
    frozen = bool(status == "finished" and key and code
                  and hashlib.sha256(code.encode()).hexdigest() == key)
    heldout = {"valid": [], "failed": [], "missing": [], "mismatched": [], "unverified": []}
    results = {r["scale"]: r for r in result_json(directory / "heldout.json", []) if not r["variant"]}
    for split in expected_splits(row["task"]):
        result = results.get(str(scale_of_split(row["task"], split)))
        if result is None:
            heldout["missing"].append(split)
            continue
        verification = heldout_identity(directory, result)
        if verification == "legacy":
            heldout["unverified"].append(split)
        elif verification != "verified":
            heldout["mismatched"].append(split)
        elif isinstance(result.get("fitness"), (int, float)) and math.isfinite(result["fitness"]):
            heldout["valid"].append(split)
        else:
            heldout["failed"].append(split)
    return {"task": row["task"], "name": directory.name, "status": status, "phase": phase,
            "budget_used": used, "budget": timing_row["budget"], "frozen": frozen,
            "log_age_seconds": round(max(0, now - modified)) if modified else None,
            "timing": timing, "heldout": heldout}


def collect(manifest_path):
    manifest = read_json(manifest_path)
    if not isinstance(manifest.get("plan"), list) or not manifest["plan"]:
        raise ValueError(f"missing or invalid batch plan: {manifest_path}")
    if any(row.get("task") not in SCALES for row in manifest["plan"]):
        raise ValueError("unsupported task in batch plan")
    now = time.time()
    rows = [run_status(manifest_path.parent, row, manifest.get("budget_per_run", 1000), now)
            for row in manifest["plan"]]
    counts = Counter(row["status"] for row in rows)
    return {"batch": manifest.get("batch", manifest_path.stem),
            "observed_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "manifest": str(manifest_path), "runs": rows,
            "summary": {"runs": len(rows), **dict(counts),
                        "heldout_valid": sum(len(r["heldout"]["valid"]) for r in rows),
                        "heldout_failed": sum(len(r["heldout"]["failed"]) for r in rows),
                        "heldout_mismatched": sum(len(r["heldout"]["mismatched"]) for r in rows),
                        "heldout_unverified": sum(len(r["heldout"]["unverified"]) for r in rows),
                        "ready_for_heldout": sum(r["frozen"] and bool(r["heldout"]["missing"]) for r in rows)},
            "search_timing": batch_timing([{**r, "status": "running" if r["status"] == "recorded" else r["status"]} for r in rows])}


def change_key(payload):
    """Elapsed time and ETA alone should not wake the caller."""
    return [(r["name"], r["status"], r["phase"], r["budget_used"], r["frozen"], r["heldout"])
            for r in payload["runs"]]


def observe(manifest, wait_seconds=0, interval=10):
    payload = collect(manifest)
    if not wait_seconds:
        return payload
    before = change_key(payload)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
        payload = collect(manifest)
        if change_key(payload) != before:
            payload["wait_result"] = "changed"
            return payload
    payload["wait_result"] = "timeout"
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--json", action="store_true", help="structured output, including every run")
    parser.add_argument("--details", action="store_true", help="show every run instead of just items needing attention")
    parser.add_argument("--wait-seconds", type=float, default=0, help="wait up to 60 seconds for a meaningful change")
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--ssh", help="SSH host alias; executes this reader via stdin in one connection")
    parser.add_argument("--repo", type=Path, help="absolute remote repository path; required with --ssh")
    args = parser.parse_args(argv)
    if not math.isfinite(args.wait_seconds) or not 0 <= args.wait_seconds <= 60 or not math.isfinite(args.interval) or args.interval < 1:
        parser.error("wait-seconds must be between 0 and 60; interval must be at least 1")
    if args.ssh:
        if args.ssh.startswith("-") or not args.repo or not args.repo.is_absolute():
            parser.error("--ssh requires a host alias and an absolute --repo")
        forwarded = ["--manifest", str(args.manifest), "--wait-seconds", str(args.wait_seconds), "--interval", str(args.interval)]
        forwarded += [flag for flag, enabled in (("--json", args.json), ("--details", args.details)) if enabled]
        command = f"cd {shlex.quote(str(args.repo))} && " + shlex.join([".venv/bin/python", "-", *forwarded])
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.ssh, command],
                                input=Path(__file__).read_text(encoding="utf-8"), text=True,
                                timeout=args.wait_seconds + 60, check=False)
        return result.returncode
    payload = observe(args.manifest, args.wait_seconds, args.interval)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    else:
        print(f"{payload['batch']} | {payload['observed_at']}")
        print(json.dumps(payload["summary"], ensure_ascii=False))
        eta = payload["search_timing"]
        print(f"Search ETA: {round(eta['eta_seconds'] / 60)} min" if eta["eta_seconds"] is not None else f"Search ETA: {eta['state']}")
        if payload.get("wait_result"):
            print("Wait:", payload["wait_result"])
        for row in payload["runs"]:
            h = row["heldout"]
            attention = row["status"] in {"blocked", "stale", "unknown"} or h["failed"] or h["mismatched"] or h["unverified"] or (row["frozen"] and h["missing"]) or (row["status"] == "finished" and not row["frozen"])
            if args.details or attention:
                used = row['budget_used'] if row['budget_used'] is not None else '?'
                print(f"{row['name']}: {row['status']}/{row['phase']} {used}/{row['budget']} frozen={row['frozen']} "
                      f"heldout={len(h['valid'])} missing={len(h['missing'])} failed={len(h['failed'])} mismatched={len(h['mismatched'])} unverified={len(h['unverified'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
