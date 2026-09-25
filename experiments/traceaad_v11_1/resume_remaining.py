"""Resume the explicitly paused V11.1 runs from their settled checkpoints."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"


def resume(batch_path: Path) -> None:
    payload = json.loads(batch_path.read_text())
    rows = [row for row in payload["plan"] if row.get("status") == "paused"]
    if not 0 < len(rows) <= 8:
        raise ValueError(f"expected 1 to 8 paused runs, found {len(rows)}")
    now = datetime.now().astimezone().isoformat()
    for row in rows:
        run = RESULTS / row["task"] / row["run_name"]
        state = json.loads((run / "tree_state.json").read_text())
        config = json.loads((run / "run_config.json").read_text())
        if state["budget_used"] >= config["method_params"]["budget"]:
            raise ValueError(f"paused run already exhausted: {row['run_name']}")
        if subprocess.run(["tmux", "has-session", "-t", "=" + row["session"]],
                          capture_output=True).returncode == 0:
            raise RuntimeError(f"session already exists: {row['session']}")
        params = config["method_params"]
        command = ["uv", "run", "python", "-m", "experiments.traceaad_v11_1.run",
                   "--task", row["task"], "--backend", row["backend"],
                   "--repeat", str(row["repeat"]), "--seed", str(row["seed"]),
                   "--run-name", row["run_name"]]
        if "n_workers" in config["task_eval"]:
            command += ["--eval-workers", str(config["task_eval"]["n_workers"])]
        for key in ("budget", "n_roots", "history_depth", "max_input_tokens",
                    "output_tokens", "n_references"):
            command += ["--" + key.replace("_", "-"), str(params[key])]
        if config["llm"].get("enable_thinking"):
            command.append("--thinking")
        with (run / "routing_history.jsonl").open("a") as handle:
            handle.write(json.dumps({"ts": now, "operation": "resume_from_checkpoint",
                                     "backend": row["backend"],
                                     "budget_used": state["budget_used"]}) + "\n")
        subprocess.run(["tmux", "new-session", "-d", "-s", row["session"], "-c",
                        str(ROOT), *command], cwd=ROOT, check=True)
        row.update(status="running", resumed_at=datetime.now().astimezone().isoformat(),
                   resumed_budget=state["budget_used"])
        batch_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(row["task"], row["repeat"], "resumed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    resume(args.batch)
