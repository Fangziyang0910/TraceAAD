"""Run one CALM (w/o GRPO) experiment at the unified 1000-eval budget."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from baselines.calm import CALM, CALMProfiler
from baselines.calm.task_config import get_task_hyperparams, resolve_task_key

from benchmarks.tasks import ALL_TASKS
from experiments.infra.base import (
    BACKENDS,
    DEFAULT_BACKEND,
    RESULTS_ROOT,
    build_llm_client,
    build_task,
    resolve_backend,
    set_random_seed,
)
from traceaad.storage import write_json
from experiments.infra.runner import baseline_run_config, run_baseline_experiment

FAIR_MAX_SAMPLE_NUMS = 1000
METHOD_LABEL = "CALM (w/o GRPO)"


@dataclass(slots=True)
class RunSpec:
    task: str
    backend: str = DEFAULT_BACKEND
    base_url: str | None = None
    model: str | None = None
    no_proxy: str | None = None
    max_sample_nums: int = FAIR_MAX_SAMPLE_NUMS
    # Every method evaluates one candidate at a time, so methods load the host
    # alike; the evaluator count only sets how many evaluations run concurrently.
    num_evaluators: int = 1
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = RESULTS_ROOT

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / "calm" / self.task

    def __post_init__(self):
        profile = resolve_backend(self.backend, self.base_url, self.model, self.no_proxy)
        self.base_url, self.model, self.no_proxy = profile.base_url, profile.model, profile.no_proxy
        self.experiments_root = self.experiments_root.resolve()
        if self.max_sample_nums <= 0:
            raise ValueError("max_sample_nums must be positive")
        if self.num_evaluators <= 0:
            raise ValueError("num_evaluators must be positive")
        if self.eval_workers is not None and self.eval_workers <= 0:
            raise ValueError("eval_workers must be positive")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")


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
            run_dir=log_dir.parent,
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
    write_json(
        run_dir / "run_config.json",
        baseline_run_config(spec, run_dir, run_name, "calm", {
                "max_sample_nums": spec.max_sample_nums,
                "num_evaluators": spec.num_evaluators,
                "task_hyperparams": asdict(hp),
                "budget_basis": (
                    "Fair comparison budget max_sample_nums=1000; "
                    "search hyperparameters from CALM local.yaml per task; "
                    "GRPO disabled (search framework only)."
                ),
            }, task_config=task_config,
            llm_options={"temperature": float(hp.generation_temperature),
                         "top_p": float(hp.generation_top_p)},
            extra={"method_label": METHOD_LABEL, "enable_grpo": False}),
    )


def run_experiment(spec: RunSpec) -> Path:
    return run_baseline_experiment(
        spec, write_config=write_run_config, build_method=build_method,
        description=(f"method={METHOD_LABEL}, budget={spec.max_sample_nums}, "
                     f"evaluators={spec.num_evaluators}"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one CALM (w/o GRPO) experiment at 1000-eval budget."
    )
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default=DEFAULT_BACKEND)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--max-sample-nums", type=int, default=FAIR_MAX_SAMPLE_NUMS)
    parser.add_argument("--num-evaluators", type=int, default=1)
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument("--output-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--run-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(
        RunSpec(
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
