"""Resume only the eight restored ACO runs from an explicitly verified runtime."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.traceaad_v10_12.freeze import runtime_environment, verify_runtime
from traceaad.v10_12.storage import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    identity = verify_runtime(runtime)
    receipt = json.loads(args.receipt.read_text())
    assert receipt["applied"]
    results = ROOT / "experiments/traceaad_v10_12/results"
    manifest_path = results / "batch_20260921_v1012.json"
    manifest = json.loads(manifest_path.read_text())
    rows = {row["run_name"]: row for row in manifest["plan"]}
    for record in receipt["runs"]:
        row = rows[record["run_name"]]
        assert row["status"] == "paused"
        assert subprocess.run(["tmux", "has-session", "-t", "=" + row["session"]],
                              capture_output=True).returncode != 0
        run = results / row["task"] / row["run_name"]
        state = json.loads((run / "tree_state.json").read_text())
        assert state["budget_used"] == record["restored_budget"]
    # Verify that the finally selected runtime uses the same search/evaluator
    # files as the original batch, independently of the fixed launch code.
    original = json.loads((results / "runtime_20260921_v1012/runtime_manifest.json").read_text())
    new = json.loads((runtime / "runtime_manifest.json").read_text())
    for path, digest in original["files"].items():
        if path.startswith(("llm4ad/method/traceaad_v10_12/", "llm4ad/task/", "llm4ad/base/")):
            assert new["files"][path] == digest, path
    now = datetime.now().astimezone().isoformat()
    manifest["recovery"] = dict(at=now, cutoff_exclusive=receipt["cutoff_exclusive"],
                                backup=receipt["backup"], receipt=str(args.receipt.resolve()),
                                runtime=str(runtime), source_identity=identity,
                                discarded_calls=sum(r["discarded_calls"] for r in receipt["runs"]))
    for row in manifest["plan"]:
        if row["task"] not in ("cvrp_aco", "op_aco"):
            state = json.loads((results / row["task"] / row["run_name"] / "tree_state.json").read_text())
            assert state["budget_used"] == 1000
            row["status"] = "finished"
    atomic_json(manifest_path, manifest)
    for record in receipt["runs"]:
        row = rows[record["run_name"]]
        run = results / row["task"] / row["run_name"]
        config = json.loads((run / "run_config.json").read_text())
        params = config["method_params"]
        command = [sys.executable, "-m", "experiments.traceaad_v10_12.run",
                   "--task", row["task"], "--backend", row["backend"],
                   "--repeat", str(row["repeat"]), "--seed", str(row["seed"]),
                   "--run-name", row["run_name"],
                   "--eval-workers", str(config["task_eval"]["n_workers"])]
        for key in ("budget", "n_roots", "traj_gens", "n_profile_cards", "profile_card_tau",
                    "output_tokens", "max_input_tokens"):
            command.extend(["--" + key.replace("_", "-"), str(params[key])])
        if params.get("history_code"):
            command.append("--history-code")
        if config["llm"].get("enable_thinking"):
            command.append("--thinking")
        routing = dict(ts=now, operation="restore_pre_incident_and_bind_frozen_runtime",
                       backend=row["backend"], seed=row["seed"],
                       original_budget=record["original_budget"], budget_used=record["restored_budget"],
                       runtime=str(runtime), source_identity=identity, backup=receipt["backup"])
        with (run / "routing_history.jsonl").open("a") as handle:
            handle.write(json.dumps(routing) + "\n")
        subprocess.run(["tmux", "new-session", "-d", "-s", row["session"],
                        "-c", str(runtime), "-e", f"PYTHONPATH={runtime}", *command],
                       cwd=runtime, env=runtime_environment(runtime), check=True)
        row.update(status="running", resumed_at=datetime.now().astimezone().isoformat(),
                   runtime=str(runtime), source_identity=identity,
                   restored_budget=record["restored_budget"])
        atomic_json(manifest_path, manifest)
        print(row["task"], row["repeat"], "resumed from", record["restored_budget"], flush=True)


if __name__ == "__main__":
    main()
