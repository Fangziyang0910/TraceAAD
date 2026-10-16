import json
import multiprocessing
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.scheduling import SchedulerError, SchedulerServer, SchedulerSession, scheduler_status
from tests.support import TinyEvaluation
from traceaad.instance_evaluation import InstanceProgramEvaluator


@pytest.fixture
def daemon():
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) < 2:
        pytest.skip("two CPU IDs needed for concurrency checks")
    with tempfile.TemporaryDirectory(prefix="trace-sched-") as directory:
        path = str(Path(directory) / "s")
        endpoints = [
            {"base_url": "http://one/v1", "model": "test", "slots": 2},
            {"base_url": "http://two/v1", "model": "test", "slots": 1},
        ]
        with SchedulerServer(path, cpus, endpoints) as server:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
            thread.start()
            try:
                yield (path, server)
            finally:
                server.shutdown()
                thread.join(timeout=3)


def eventually(check):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.02)
    assert check()


class Instances(TinyEvaluation):
    def __init__(self):
        super().__init__()
        self.n_instance = 4
        self._datasets = list(range(4))
        self.fork_proc = False

    def evaluate_program(self, source, function):
        return sum((function(i) for i in self._datasets)) / len(self._datasets)


def _evaluate_instances(path, directory, job, sender):
    code = f"import json, os, time\ndef score(x):\n    started = time.monotonic()\n    time.sleep(.15)\n    ended = time.monotonic()\n    with open({directory!r} + '/{job}-' + str(x), 'w') as f:\n        json.dump([started, ended, list(os.sched_getaffinity(0))], f)\n    return x\n"
    try:
        result = InstanceProgramEvaluator(Instances(), scheduler_socket=path).evaluate(
            code, "x", timeout_seconds=5, n_workers=4
        )
        sender.send(result)
    finally:
        sender.close()


def test_evaluations_share_cpu_bound_and_pin_each_instance(daemon, tmp_path):
    path, server = daemon
    context = multiprocessing.get_context("spawn")
    jobs = []
    for job in ("a", "b"):
        receiver, sender = context.Pipe(False)
        process = context.Process(
            target=_evaluate_instances, args=(path, str(tmp_path), job, sender)
        )
        process.start()
        sender.close()
        jobs.append((process, receiver))
    try:
        for process, receiver in jobs:
            assert receiver.poll(25)
            result = receiver.recv()
            assert result["fitness"] == 1.5, result
            assert result["evaluations"][0]["scheduler"]["peak_workers"] <= 2
            assert {r["cpu_id"] for r in result["evaluations"][0]["instances"]} <= set(server.cpus)
            process.join(5)
            assert process.exitcode == 0
        intervals = [json.loads(p.read_text()) for p in tmp_path.iterdir()]
        assert len(intervals) == 8
        for start, _, affinity in intervals:
            assert len(affinity) == 1 and affinity[0] in server.cpus
            running = [row for row in intervals if row[0] <= start < row[1]]
            assert len(running) <= 2
            assert len({row[2][0] for row in running}) == len(running)
    finally:
        for process, receiver in jobs:
            if process.is_alive():
                process.kill()
            process.join(5)
            process.close()
            receiver.close()
    eventually(lambda: scheduler_status(path)["cpu"]["used"] == 0)


def test_waiting_for_cpu_does_not_use_instance_budget(daemon):
    path, _ = daemon
    with SchedulerSession(path, "cpu") as holder:
        held = holder.poll(2)
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(
                InstanceProgramEvaluator(Instances(), scheduler_socket=path).evaluate,
                "def score(x): return x",
                "x",
                timeout_seconds=2,
                n_workers=2,
            )
            eventually(lambda: scheduler_status(path)["cpu"]["waiting"] > 0)
            time.sleep(2.1)
            holder.poll(0, held)
            result = future.result(timeout=15)
        assert result["fitness"] == 1.5
        assert result["evaluations"][0]["scheduler"]["queue_seconds"] >= 2


def test_scheduler_disconnect_stops_running_instances(daemon, tmp_path):
    path, server = daemon
    marker = tmp_path / "worker"
    task = Instances()
    task.n_instance, task._datasets = (1, [0])
    code = f"import os, time\ndef score(x):\n    with open({str(marker)!r}, 'w') as f:\n        f.write(str(os.getpid()))\n    time.sleep(30)\n    return 1\n"
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(
            InstanceProgramEvaluator(task, scheduler_socket=path).evaluate,
            code,
            "x",
            timeout_seconds=60,
        )
        eventually(lambda: marker.exists() and marker.read_text())
        pid = int(marker.read_text())
        server.server_close()
        with pytest.raises(SchedulerError):
            future.result(timeout=10)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    eventually(lambda: server.pools["cpu"].status()["used"] == 0)
