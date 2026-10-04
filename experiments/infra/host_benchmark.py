"""How evaluation time and throughput change with the number of concurrent evaluations on this host.

usage: PYTHONPATH=. uv run python -m experiments.infra.host_benchmark TASK PROGRAM LEVELS...

Each level starts that many evaluations of the same program on the task's training
evaluator at once (no time limit, BLAS/OpenMP single-threaded as in formal runs; ACO
tasks keep their four evaluation workers) and reports the wall time of each, the
throughput, and the highest package temperature seen when ``sensors`` is available.
It only measures the host; it does not change how runs are scheduled.
"""
import experiments  # noqa: F401
import json
import multiprocessing
import os
import re
import statistics
import subprocess
import sys
import threading
import time

from experiments.infra.base import build_task


def evaluate(task, code, queue):
    evaluation, _ = build_task(task, 4)
    evaluation.timeout_seconds = None
    namespace = {}
    exec(code, namespace)
    function = namespace[str(evaluation.template_program).split("def ")[1].split("(")[0]]
    started = time.monotonic()
    score = evaluation.evaluate_program(code, function)
    queue.put((time.monotonic() - started, float(score)))


def package_temperature():
    try:
        text = subprocess.run(["sensors"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = re.findall(r"Package id \d+:\s+\+([\d.]+)", text)
    return max(map(float, found)) if found else None


def level(task, code, count):
    queue = multiprocessing.get_context("fork").Queue()
    temps, done = [], threading.Event()

    def watch():
        while not done.is_set():
            t = package_temperature()
            if t is not None:
                temps.append(t)
            done.wait(2)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    started = time.monotonic()
    processes = [multiprocessing.get_context("fork").Process(target=evaluate, args=(task, code, queue))
                 for _ in range(count)]
    for p in processes:
        p.start()
    results = [queue.get() for _ in processes]
    for p in processes:
        p.join()
    wall = time.monotonic() - started
    done.set()
    watcher.join()
    seconds = sorted(r[0] for r in results)
    return {"concurrent": count, "median_s": statistics.median(seconds), "min_s": seconds[0],
            "max_s": seconds[-1], "evals_per_min": 60 * count / wall,
            "max_package_C": max(temps) if temps else None, "score": results[0][1]}


def main():
    task, program, *levels = sys.argv[1:]
    code = open(program).read()
    for count in map(int, levels):
        print(json.dumps(level(task, code, count)), flush=True)


if __name__ == "__main__":
    main()
