"""Run one paper-aligned ReEvo experiment at the unified 1000-eval budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from baselines.reevo import ReEvo, ReEvoProfiler

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

# Paper table / original cfg/config.yaml used max_fe=100 for sample-efficiency
# claims. Fair comparison across methods in this repo uses a unified budget.
PAPER_MAX_SAMPLE_NUMS = 100
FAIR_MAX_SAMPLE_NUMS = 1000
PAPER_POP_SIZE = 10
PAPER_INIT_POP_SIZE = 30
PAPER_MUTATION_RATE = 0.5


@dataclass(slots=True)
class RunSpec:
    task: str
    backend: str = DEFAULT_BACKEND
    base_url: str | None = None
    model: str | None = None
    no_proxy: str | None = None
    max_sample_nums: int = FAIR_MAX_SAMPLE_NUMS
    pop_size: int = PAPER_POP_SIZE
    init_pop_size: int = PAPER_INIT_POP_SIZE
    mutation_rate: float = PAPER_MUTATION_RATE
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = RESULTS_ROOT

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / "reevo" / self.task

    def __post_init__(self):
        profile = resolve_backend(self.backend, self.base_url, self.model, self.no_proxy)
        self.base_url, self.model, self.no_proxy = profile.base_url, profile.model, profile.no_proxy
        self.experiments_root = self.experiments_root.resolve()
        if self.max_sample_nums <= 0:
            raise ValueError("max_sample_nums must be positive")
        if self.pop_size <= 0:
            raise ValueError("pop_size must be positive")
        if self.init_pop_size <= 0:
            raise ValueError("init_pop_size must be positive")
        if not 0.0 < self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in (0, 1]")
        if self.eval_workers is not None and self.eval_workers <= 0:
            raise ValueError("eval_workers must be positive")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")


def build_method(spec: RunSpec, log_dir: Path) -> ReEvo:
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.output_tokens,
    )
    return ReEvo(
        llm=llm,
        evaluation=evaluation,
        profiler=ReEvoProfiler(
            run_dir=log_dir.parent,
        ),
        max_sample_nums=spec.max_sample_nums,
        pop_size=spec.pop_size,
        init_pop_size=spec.init_pop_size,
        mutation_rate=spec.mutation_rate,
        num_samplers=1,
        num_evaluators=1,
        max_consecutive_sample_failures=20,
        multi_thread_or_process_eval="thread",
        debug_mode=False,
    )


def write_run_config(spec: RunSpec, run_dir: Path, run_name: str) -> None:
    write_json(
        run_dir / "run_config.json",
        baseline_run_config(spec, run_dir, run_name, "reevo", {
                "max_sample_nums": spec.max_sample_nums,
                "population_size": spec.pop_size,
                "init_pop_size": spec.init_pop_size,
                "mutation_rate": spec.mutation_rate,
                "num_samplers": 1,
                "num_evaluators": 1,
                "sampling_adaptation": "shared model sampling profile for every call",
                "budget_basis": "shared fixed budget; original ReEvo search hyperparameters",
            }),
    )


def run_experiment(spec: RunSpec) -> Path:
    return run_baseline_experiment(
        spec, write_config=write_run_config, build_method=build_method,
        description=(f"reevo=pop={spec.pop_size}, init={spec.init_pop_size}, "
                     f"budget={spec.max_sample_nums}, mutation_rate={spec.mutation_rate}"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one paper-aligned ReEvo experiment.")
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default=DEFAULT_BACKEND)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--max-sample-nums", type=int, default=FAIR_MAX_SAMPLE_NUMS)
    parser.add_argument("--pop-size", type=int, default=PAPER_POP_SIZE)
    parser.add_argument("--init-pop-size", type=int, default=PAPER_INIT_POP_SIZE)
    parser.add_argument("--mutation-rate", type=float, default=PAPER_MUTATION_RATE)
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
            pop_size=args.pop_size,
            init_pop_size=args.init_pop_size,
            mutation_rate=args.mutation_rate,
            eval_workers=args.eval_workers,
            output_tokens=args.output_tokens,
            seed=args.seed,
            repeat=args.repeat,
            run_name=args.run_name,
        )
    )


if __name__ == "__main__":
    main()
