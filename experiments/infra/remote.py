"""Use server3 as a disposable executor: push the code, pull results, prune what is safely local.

server3 keeps no history. ``push`` mirrors this working tree there (the
remote .venv and experiments_result stay), ``pull`` copies one batch's runs,
manifest and launch logs here, and ``prune`` deletes a batch's runs on server3
once they are complete and the local copy is byte-identical. ``watch`` pulls
and prunes every interval while a batch runs.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

HOST = "B3-server3"
REMOTE = "/home/fzy/code/LLM4AD/TraceAAD"
ROOT = Path(__file__).resolve().parents[2]
CODE_EXCLUDES = (".git/", ".venv/", "experiments_result/", "__pycache__/", ".pytest_cache/", ".cache/", ".env", ".vscode/",
                 ".claude/", "*.pyc")
TERMINAL = {"finished", "search_complete", "selection_failed", "no_valid_root"}


def run(*command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def push():
    # --delete removes code deleted here; excluded paths (the remote .venv and results) are kept.
    changes = run("rsync", "-az", "--delete", "--itemize-changes", *(f"--exclude={p}" for p in CODE_EXCLUDES),
                  f"{ROOT}/", f"{HOST}:{REMOTE}/")
    if any(line.split()[-1] in {"pyproject.toml", "uv.lock"} for line in changes.splitlines() if line.strip()):
        run("ssh", HOST, f"cd {REMOTE} && ~/.local/bin/uv sync --frozen")
    print(f"pushed {len(changes.splitlines())} changes")


def results(experiment):
    return f"{HOST}:{REMOTE}/experiments_result/{experiment}/", ROOT / "experiments_result" / experiment


def pull(experiment, batch):
    remote, local = results(experiment)
    local.mkdir(parents=True, exist_ok=True)
    run("rsync", "-az", "--prune-empty-dirs", "--exclude=.cache/", f"--include=batch_{batch}.json",
        f"--include=launch_logs/{batch}_*", "--include=*/", f"--include=*/{batch}_*/***", "--exclude=*",
        remote, f"{local}/")


def prune(experiment, batch, heldout=True):
    """Delete complete runs on server3 whose local copy is identical; the manifest and
    launch logs go with the last run."""
    remote, local = results(experiment)
    runs = sorted(path for path in local.glob(f"*/{batch}_*") if path.is_dir())
    left = []
    for path in runs:
        relative = path.relative_to(local).as_posix()
        summary = path / "summary.json"
        status = json.loads(summary.read_text()).get("status") if summary.exists() else None
        complete = status in TERMINAL and (not heldout or (path / "heldout.json").exists())
        differs = run("rsync", "-anc", "--itemize-changes", "--exclude=.cache/", f"{remote}{relative}/", f"{path}/")
        if not complete or differs.strip():
            left.append(relative)
            continue
        run("ssh", HOST, f"rm -rf {REMOTE}/experiments_result/{experiment}/{relative}")
        print(f"pruned {relative}")
    if runs and not left:
        run("ssh", HOST, f"cd {REMOTE}/experiments_result/{experiment} && "
                         f"rm -f batch_{batch}.json launch_logs/{batch}_*")
    return left


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("push", "pull", "prune", "watch"))
    parser.add_argument("--experiment", default="traceaad_v10_21")
    parser.add_argument("--batch")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--no-heldout", action="store_true", help="prune finished runs without a held-out result")
    args = parser.parse_args(argv)
    if args.action == "push":
        return push()
    if not args.batch:
        parser.error("--batch is required")
    while True:
        pull(args.experiment, args.batch)
        if args.action == "pull":
            return
        left = prune(args.experiment, args.batch, heldout=not args.no_heldout)
        if args.action == "prune":
            return
        print(time.strftime("%F %T"), f"{len(left)} runs on server3", flush=True)
        # Done once the batch has arrived here and nothing of it is left there.
        if not left and any(path.is_dir() for path in results(args.experiment)[1].glob(f"*/{args.batch}_*")):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
