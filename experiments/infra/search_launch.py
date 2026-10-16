"""Build host launch plans."""

from experiments.infra.base import HOSTED_BACKENDS, RESULTS_ROOT, TASK_SHORT


def build_plan(
    previous,
    batch,
    experiment,
    module,
    budget=1000,
    *,
    eval_workers=4,
    eval_timeout_seconds=None,
    scheduler_socket=None,
):
    source = previous["plan"]
    plan = []
    for item in source:
        task, repeat, backend, seed = (item[k] for k in ("task", "repeat", "backend", "seed"))
        if seed != repeat - 1:
            raise ValueError("prior batch does not use repeat - 1 seeds")
        short = TASK_SHORT[task]
        name = f"{batch}_{short}_{experiment}_rep{repeat}"
        tag = experiment.removeprefix("traceaad_")
        session = f"{batch}_{short}_{tag}_r{repeat}"
        command = [
            "uv",
            "run",
            "python",
            "-m",
            module,
            "--task",
            task,
            "--backend",
            backend,
            "--repeat",
            str(repeat),
            "--seed",
            str(seed),
            "--run-name",
            name,
            f"--budget={budget}",
            f"--eval-workers={eval_workers}",
            "--experiment",
            experiment,
        ]
        if eval_timeout_seconds is not None:
            command.append(f"--eval-timeout-seconds={eval_timeout_seconds:g}")
        if scheduler_socket:
            command.extend(["--scheduler-socket", str(scheduler_socket)])
        plan.append(
            {
                "task": task,
                "repeat": repeat,
                "seed": seed,
                "backend": backend,
                "session": session,
                "run_name": name,
                "run_dir": str(RESULTS_ROOT / experiment / task / name),
                "command": command,
                "status": "planned",
            }
        )
        if scheduler_socket and backend not in HOSTED_BACKENDS:
            plan[-1]["backend"] = "scheduled"
    if len({item["run_name"] for item in plan}) != len(plan):
        raise ValueError("duplicate run identities")
    return plan
