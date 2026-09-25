"""Run the three-arm E32 initialization comparison on server3."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from experiments.infra.base import BACKENDS as BACKEND_PROFILES, free_slots
from experiments.infra.launcher import check_backends

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BATCH = "init_compare_20260926"
SCHEDULE = HERE / "schedule.json"
STATE = HERE / "batch_state.json"
BACKENDS = ("server3", "server3b")
TASKS = ("tsp_construct", "online_bin_packing", "cvrp_aco", "op_aco", "vrptw_construct")
MODES = ("independent", "sequential", "hybrid")
REPEATS = 4
MAX_ATTEMPTS = 3
PROTOCOL_FILES = (
    "experiments/traceaad_initialization/run.py",
    "experiments/traceaad_initialization/heldout.py",
    "experiments/infra/base.py",
    "experiments/infra/runner.py",
    "benchmarks/generated_data_config.py",
    "traceaad/v10_13/traceaad.py",
    "traceaad/v10_13/prompts.py",
    "traceaad/v10_13/parsing.py",
    "traceaad/v10_13/selection.py",
    "traceaad/v10_13/tree.py",
)


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def protocol_digest():
    digest = hashlib.sha256()
    for relative in PROTOCOL_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def jobs():
    rng = random.Random(20260926)
    rows = []
    for repeat in range(1, REPEATS + 1):
        tasks = list(TASKS)
        rng.shuffle(tasks)
        for task in tasks:
            modes = list(MODES)
            rng.shuffle(modes)
            for mode in modes:
                endpoint_index = (repeat + TASKS.index(task) + MODES.index(mode)) % 2
                rows.append({
                    "task": task, "mode": mode, "repeat": repeat,
                    "seed": repeat - 1, "backend": BACKENDS[endpoint_index],
                    "run_name": f"{BATCH}_{task}_{mode}_r{repeat}",
                    "session": f"initcmp_{task}_{mode}_r{repeat}",
                })
    return rows


def validate(rows):
    expected = {(task, mode, repeat) for task in TASKS for mode in MODES
                for repeat in range(1, REPEATS + 1)}
    if len(rows) != len(expected) or {
        (r["task"], r["mode"], r["repeat"]) for r in rows
    } != expected:
        raise ValueError("schedule must contain 5 tasks x 3 arms x 4 repeats")
    if Counter(r["backend"] for r in rows) != {"server3": 30, "server3b": 30}:
        raise ValueError("schedule must contain 30 runs per endpoint")
    for task in TASKS:
        for mode in MODES:
            counts = Counter(r["backend"] for r in rows
                             if r["task"] == task and r["mode"] == mode)
            if counts != {"server3": 2, "server3b": 2}:
                raise ValueError(f"endpoint imbalance for {task} {mode}: {counts}")
    for field in ("run_name", "session"):
        if len({r[field] for r in rows}) != len(rows):
            raise ValueError(f"duplicate {field}")


def load_schedule():
    rows = jobs()
    validate(rows)
    if SCHEDULE.exists():
        if json.loads(SCHEDULE.read_text(encoding="utf-8")) != rows:
            raise ValueError(f"stored schedule differs from code: {SCHEDULE}")
    else:
        SCHEDULE.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return rows


def run_dir(row):
    return RESULTS / row["task"] / row["run_name"]


def summary_status(row):
    path = run_dir(row) / "logs" / "run_summary.json"
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("status")
            if isinstance(value, str):
                return value
        except (OSError, json.JSONDecodeError):
            pass
    return None


def session_alive(row):
    return subprocess.run(["tmux", "has-session", "-t", f"={row['session']}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          check=False).returncode == 0


def save_state(state):
    state["updated_at"] = timestamp()
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(STATE)


def load_state(rows):
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        if state.get("batch") != BATCH or set(state.get("jobs", {})) != {
            row["run_name"] for row in rows
        }:
            raise ValueError("state does not match this formal schedule")
        if state.get("protocol_digest") != protocol_digest():
            raise RuntimeError("formal protocol code changed after batch creation")
        return state
    state = {"batch": BATCH, "created_at": timestamp(),
             "protocol_digest": protocol_digest(), "jobs": {
        row["run_name"]: {"attempts": 0, "status": "queued", "last_error": None}
        for row in rows
    }}
    save_state(state)
    return state


def refresh(rows, state):
    for row in rows:
        record = state["jobs"][row["run_name"]]
        status = summary_status(row)
        if status in {"finished", "incomplete_initialization", "call_cap", "generation_stall"}:
            record["status"] = status
        elif session_alive(row):
            record["status"] = "running"
        elif record["attempts"] >= MAX_ATTEMPTS:
            record["status"] = "blocked"
            record["last_error"] = status or "no terminal summary"
        elif run_dir(row).exists() and not (run_dir(row) / "tree_state.json").exists():
            record["status"] = "blocked"
            record["last_error"] = "existing directory has no resumable checkpoint"
        elif record["attempts"] > 0:
            record["status"] = "queued"
            record["last_error"] = status or "session ended without terminal summary"


def check_resume_config(row):
    run = run_dir(row)
    if not run.exists():
        return
    if not (run / "tree_state.json").exists():
        raise RuntimeError(f"existing run lacks checkpoint: {run}")
    config_path = run / "run_config.json"
    if not config_path.exists():
        raise RuntimeError(f"existing run lacks config: {run}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    params = config.get("method_params", {})
    expected = {"budget": 32, "budget_basis": "evaluator_calls",
                "n_roots": 8, "history_depth": 3, "max_input_tokens": 24320,
                "max_calls": 96, "init_mode": row["mode"],
                "init_only": False, "output_tokens": 8192}
    for field, value in expected.items():
        if params.get(field) != value:
            raise RuntimeError(f"resume config mismatch for {run}: {field}")
    for field, value in (("task", row["task"]), ("seed", row["seed"]),
                         ("repeat", row["repeat"]), ("backend", row["backend"]),
                         ("method", "initialization_v1013")):
        if config.get(field) != value:
            raise RuntimeError(f"resume config mismatch for {run}: {field}")
    llm = config.get("llm", {})
    profile = BACKEND_PROFILES[row["backend"]]
    for field, value in (("base_url", profile.base_url), ("model", profile.model),
                         ("max_tokens", 8192), ("enable_thinking", False),
                         ("temperature", 1.0), ("top_p", 0.95), ("top_k", 20)):
        if llm.get(field) != value:
            raise RuntimeError(f"resume LLM config mismatch for {run}: {field}")
    task_eval = config.get("task_eval", {})
    if task_eval.get("split") != "train":
        raise RuntimeError(f"resume task split mismatch for {run}")
    if row["task"] in {"cvrp_aco", "op_aco"} and task_eval.get("n_workers") != 2:
        raise RuntimeError(f"resume ACO worker count mismatch for {run}")


def launch(row):
    check_resume_config(row)
    if session_alive(row):
        raise RuntimeError(f"session already exists: {row['session']}")
    command = [
        "uv", "run", "python", "-u", "-m", "experiments.traceaad_initialization.run",
        "--task", row["task"], "--backend", row["backend"],
        "--init-mode", row["mode"], "--seed", str(row["seed"]),
        "--repeat", str(row["repeat"]), "--run-name", row["run_name"],
        "--budget", "32", "--output-tokens", "8192", "--eval-workers", "2",
    ]
    subprocess.run(["tmux", "new-session", "-d", "-s", row["session"],
                    "-c", str(ROOT), *command], cwd=ROOT, check=True)


def start_available(rows, state):
    queued = [r for r in rows if state["jobs"][r["run_name"]]["status"] == "queued"]
    if not queued:
        return
    check_backends(BACKENDS)
    capacity = free_slots()
    active = Counter(r["backend"] for r in rows
                     if state["jobs"][r["run_name"]]["status"] == "running")
    for row in queued:
        backend = row["backend"]
        if min(9 - active[backend], capacity.get(backend, 0)) <= 0:
            continue
        record = state["jobs"][row["run_name"]]
        try:
            launch(row)
        except Exception as exc:
            record["last_error"] = str(exc)
            record["status"] = "blocked"
            print(f"blocked {row['run_name']}: {exc}", flush=True)
            save_state(state)
            continue
        record["attempts"] += 1
        record["status"] = "running"
        record["started_at"] = record.get("started_at") or timestamp()
        active[backend] += 1
        capacity[backend] -= 1
        save_state(state)
        print(f"started {row['run_name']} on {backend}", flush=True)


def run_batch(*, watch, interval):
    rows = load_schedule()
    lock_path = STATE.with_suffix(".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = load_state(rows)
        while True:
            if state["protocol_digest"] != protocol_digest():
                raise RuntimeError("formal protocol code changed during the batch")
            refresh(rows, state)
            try:
                start_available(rows, state)
            except Exception as exc:
                print(f"endpoint check failed: {exc}", flush=True)
            save_state(state)
            counts = Counter(state["jobs"][r["run_name"]]["status"] for r in rows)
            print(f"{timestamp()} {dict(counts)}", flush=True)
            terminal = sum(counts[s] for s in
                           ("finished", "incomplete_initialization", "call_cap",
                            "generation_stall", "blocked"))
            if terminal == len(rows) and watch:
                subprocess.run(["uv", "run", "python", "-u", "-m",
                                "experiments.traceaad_initialization.heldout"],
                               cwd=ROOT, check=True)
                return
            if not watch:
                return
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error("--interval must be positive")
    rows = load_schedule()
    if args.dry_run:
        print(json.dumps({"batch": BATCH, "runs": len(rows), "evaluation_budget": 32,
                          "backends": Counter(r["backend"] for r in rows)},
                         ensure_ascii=False, indent=2))
        return
    run_batch(watch=args.watch, interval=args.interval)


if __name__ == "__main__":
    main()
