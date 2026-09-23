from __future__ import annotations

import os
import time
from pathlib import Path

from core import Evaluation, SecureEvaluator

TEMPLATE = """def score():
    return 1.0
"""


def _hang_and_report(path: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
        handle.flush()
    while True:
        time.sleep(1)


class NestedHangEvaluation(Evaluation):
    def __init__(self, pid_file: Path) -> None:
        super().__init__(
            template_program=TEMPLATE,
            timeout_seconds=2,
            safe_evaluate=True,
            fork_proc=True,
            daemon_eval_process=False,
        )
        self._pid_file = str(pid_file)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        import multiprocessing

        context = multiprocessing.get_context("fork")
        workers = [
            context.Process(target=_hang_and_report, args=(self._pid_file,))
            for _ in range(3)
        ]
        pids: list[str] = []
        for worker in workers:
            worker.start()
            if worker.pid is not None:
                pids.append(str(worker.pid))
        Path(self._pid_file).write_text("\n".join(pids) + "\n", encoding="utf-8")
        time.sleep(30)
        return 1.0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_timeout_kills_nested_spawn_workers(tmp_path: Path) -> None:
    pid_file = tmp_path / "workers.txt"
    outcome = SecureEvaluator(NestedHangEvaluation(pid_file)).evaluate_program_with_details(
        TEMPLATE
    )
    assert outcome.failure_kind == "timeout"

    deadline = time.time() + 5
    pids: list[int] = []
    while time.time() < deadline:
        if pid_file.is_file():
            pids = [
                int(line)
                for line in pid_file.read_text(encoding="utf-8").split()
                if line.strip()
            ]
            if pids:
                break
        time.sleep(0.05)
    assert pids, "nested workers never started"

    leftover_deadline = time.time() + 5
    leftover = [pid for pid in pids if _alive(pid)]
    while leftover and time.time() < leftover_deadline:
        time.sleep(0.05)
        leftover = [pid for pid in pids if _alive(pid)]
    assert leftover == []
