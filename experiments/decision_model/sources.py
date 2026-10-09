"""Two additional independent V10.22 traces per task, at most 12,000 candidates."""

import experiments  # noqa: F401
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

from benchmarks.tasks import CO_TASKS, training_task
from experiments.infra.scheduled_llm import ScheduledLLM
from experiments.infra.base import sampling_controls
from traceaad.common.config import REVISION
from traceaad.common.storage import write_json
from traceaad.v10_22 import Config, TraceAADV1022


def run_source(root, task, seed, socket):
    name = f"20261009_decision_sources_{task}_rep{seed + 1}"
    work = root / task / name
    if (work / "summary.json").exists():
        summary = json.loads((work / "summary.json").read_text())
        if summary["status"] == "finished":
            return
    work.mkdir(parents=True, exist_ok=True)
    evaluation, task_kwargs = training_task(task, condition="traceaad")
    llm = ScheduledLLM(socket, max_tokens=8192, label="decision-source:" + task)
    config = Config(budget=1000, seed=seed, scheduler_socket=socket, eval_workers=4)
    write_json(work / "run_config.json", {"result_format": "traceaad-results-v2", "method": "v1022",
        "task": task, "seed": seed, "run_name": name, "budget": 1000, "objective": "min",
        "revision": REVISION, "task_eval": task_kwargs, "method_params": asdict(config),
        "llm": {"model": llm.model, "sampling": sampling_controls(False), "max_tokens": 8192,
                "enable_thinking": False},
        "evaluation_execution": {"protocol": "isolated-instances-v1", "function_seconds": 2,
                                 "timeout_seconds": 20, "timeout_scope": "instance", "n_workers": 4}})
    try:
        method = TraceAADV1022(evaluation=evaluation, llm=llm, run_dir=work, config=config, task=task)
        method.run()
        print(task, seed, "complete", flush=True)
    finally:
        llm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scheduler-socket", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(task, seed) for seed in range(2) for task in CO_TASKS]
    with ThreadPoolExecutor(args.workers) as pool:
        for future in [pool.submit(run_source, args.output, task, seed, args.scheduler_socket) for task, seed in jobs]:
            future.result()
    write_json(args.output / "complete.json", {"runs": 12, "max_candidates": 12000})


if __name__ == "__main__":
    main()
