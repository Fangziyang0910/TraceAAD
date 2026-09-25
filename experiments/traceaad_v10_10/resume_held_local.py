"""Resume the held formal CVRP route after one of the three active routes ends."""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from experiments.infra.launcher import get_summary_status


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/traceaad_v10_10/results"
HELD_TASK = "cvrp_aco"
HELD_NAME = "20260911_v1010_formal_cvrp_v1010_rep1"
HELD_SESSION = "v1010f_cvrp_r1"
ACTIVE = (
    ("cvrp_aco", "20260911_v1010_formal_cvrp_v1010_rep2", "v1010f_cvrp_r2"),
    ("cvrp_aco", "20260911_v1010_formal_cvrp_v1010_rep3", "v1010f_cvrp_r3"),
    ("op_aco", "20260911_v1010_formal_op_v1010_rep1", "v1010f_op_r1"),
)


def alive(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def launch_held() -> None:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "experiments.traceaad_v10_10.run",
        "--task",
        HELD_TASK,
        "--backend",
        "server3b",
        "--repeat",
        "1",
        "--seed",
        "0",
        "--run-name",
        HELD_NAME,
    ]
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            HELD_SESSION,
            "-c",
            str(ROOT),
            shlex.join(command),
        ],
        check=True,
    )
    print(f"resumed {HELD_NAME} in {HELD_SESSION}", flush=True)


def main() -> None:
    held_dir = RESULTS / HELD_TASK / HELD_NAME
    while True:
        if alive(HELD_SESSION):
            return
        if get_summary_status(held_dir) in ("finished", "blocked"):
            return
        completed = [
            (task, name)
            for task, name, session in ACTIVE
            if get_summary_status(RESULTS / task / name) in ("finished", "blocked")
        ]
        if completed:
            launch_held()
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
