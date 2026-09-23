"""Run one CALM (w/o GRPO) experiment at the unified 1000-eval budget."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from baselines.calm import CALM, CALMProfiler
from baselines.calm.task_config import get_task_hyperparams, resolve_task_key

from experiments.infra.base import (
    ALL_TASKS,
    BACKENDS,
    EXPERIMENTS_ROOT,
    TASKS,
    TaskName,
    build_llm_client,
    build_task,
    llm_payload,
    resolve_backend,
    resolve_run_dir as resolve_run_dir_file,
    run_in_tmux_log,
    set_random_seed,
    write_run_config as write_run_config_file,
)

FAIR_MAX_SAMPLE_NUMS = 1000
METHOD_LABEL = "CALM (w/o GRPO)"


@dataclass(frozen=True, slots=True)
class RunSpec:
    task: TaskName
    backend: str
    base_url: str
    model: str
    no_proxy: str
    max_sample_nums: int = FAIR_MAX_SAMPLE_NUMS
    num_evaluators: int = 4
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = EXPERIMENTS_ROOT

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / self.task / "calm"


def make_run_spec(
    *,
    task: TaskName,
    backend: str = "local",
    base_url: str | None = None,
    model: str | None = None,
    no_proxy: str | None = None,
    max_sample_nums: int = FAIR_MAX_SAMPLE_NUMS,
    num_evaluators: int = 4,
    eval_workers: int | None = None,
    output_tokens: int = 16384,
    seed: int = 0,
    repeat: int | None = None,
    run_name: str | None = None,
    experiments_root: Path = EXPERIMENTS_ROOT,
) -> RunSpec:
    profile = resolve_backend(backend, base_url, model, no_proxy)
    spec = RunSpec(
        task=task,
        backend=backend,
        base_url=profile.base_url,
        model=profile.model,
        no_proxy=profile.no_proxy,
        max_sample_nums=max_sample_nums,
        num_evaluators=num_evaluators,
        eval_workers=eval_workers,
        output_tokens=output_tokens,
        seed=seed,
        repeat=repeat,
        run_name=run_name,
        experiments_root=experiments_root.resolve(),
    )
    if spec.max_sample_nums <= 0:
        raise ValueError("max_sample_nums must be positive")
    if spec.num_evaluators <= 0:
        raise ValueError("num_evaluators must be positive")
    if spec.eval_workers is not None and spec.eval_workers <= 0:
        raise ValueError("eval_workers must be positive")
    if spec.output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    return spec


def build_method(spec: RunSpec, log_dir: Path) -> CALM:
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    hp = get_task_hyperparams(spec.task)
    llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.output_tokens,
        temperature=float(hp.generation_temperature),
        top_p=float(hp.generation_top_p),
    )
    return CALM(
        llm=llm,
        evaluation=evaluation,
        profiler=CALMProfiler(
            log_dir=str(log_dir),
            log_style="complex",
            create_random_path=False,
        ),
        max_sample_nums=spec.max_sample_nums,
        seed=spec.seed,
        task_key=spec.task,
        num_evaluators=spec.num_evaluators,
        debug_mode=False,
    )


def write_run_config(spec: RunSpec, run_dir: Path, run_name: str) -> None:
    evaluation, task_config = build_task(spec.task, spec.eval_workers)
    task_key = resolve_task_key(evaluation)
    hp = get_task_hyperparams(task_key)
    write_run_config_file(
        run_dir,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "run_name": run_name,
            "task": spec.task,
            "method": "calm",
            "method_label": METHOD_LABEL,
            "enable_grpo": False,
            "repeat": spec.repeat,
            "backend": spec.backend,
            "seed": spec.seed,
            "llm": llm_payload(
                base_url=spec.base_url,
                model=spec.model,
                no_proxy=spec.no_proxy,
                max_tokens=spec.output_tokens,
                temperature=float(hp.generation_temperature),
                top_p=float(hp.generation_top_p),
            ),
            "task_eval": task_config,
            "method_params": {
                "max_sample_nums": spec.max_sample_nums,
                "num_evaluators": spec.num_evaluators,
                "task_hyperparams": asdict(hp),
                "budget_basis": (
                    "Fair comparison budget max_sample_nums=1000; "
                    "search hyperparameters from CALM local.yaml per task; "
                    "GRPO disabled (search framework only)."
                ),
            },
        },
    )


def resolve_run_dir(spec: RunSpec) -> tuple[Path, str]:
    """Create (or return) the run directory for a spec (legacy convenience API)."""
    return resolve_run_dir_file(spec.experiment_root, spec.run_name)


def run_experiment(spec: RunSpec) -> Path:
    run_dir, run_name = resolve_run_dir(spec)
    log_dir = run_dir / "logs"
    write_run_config(spec, run_dir, run_name)
    print(f"run_dir={run_dir}")
    run_in_tmux_log(
        run_dir,
        log_dir,
        [
            f"log_dir={log_dir}",
            f"llm={spec.model} @ {spec.base_url}",
            f"method={METHOD_LABEL}, budget={spec.max_sample_nums}, "
            f"evaluators={spec.num_evaluators}",
        ],
        lambda: build_method(spec, log_dir).run(),
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one CALM (w/o GRPO) experiment at 1000-eval budget."
    )
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--max-sample-nums", type=int, default=FAIR_MAX_SAMPLE_NUMS)
    parser.add_argument("--num-evaluators", type=int, default=4)
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument("--output-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--run-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(
        make_run_spec(
            task=args.task,
            backend=args.backend,
            base_url=args.base_url,
            model=args.model,
            no_proxy=args.no_proxy,
            max_sample_nums=args.max_sample_nums,
            num_evaluators=args.num_evaluators,
            eval_workers=args.eval_workers,
            output_tokens=args.output_tokens,
            seed=args.seed,
            repeat=args.repeat,
            run_name=args.run_name,
        )
    )


if __name__ == "__main__":
    main()
