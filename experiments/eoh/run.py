"""Run one official-source EoH experiment at the unified 1000-eval budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from baselines.eoh import EoH, EoHProfiler

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

FORMAL_BUDGET = 1000
PAPER_GENERATIONS = 20
OPERATORS = ("e1", "e2", "m1", "m2")


def official_population_size(task: str) -> int:
    return 5


@dataclass(slots=True)
class RunSpec:
    task: str
    backend: str = DEFAULT_BACKEND
    base_url: str | None = None
    model: str | None = None
    no_proxy: str | None = None
    generations: int = PAPER_GENERATIONS
    budget: int | None = None
    pop_size: int | None = None
    parents: int = 2
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = RESULTS_ROOT

    @property
    def effective_pop_size(self) -> int:
        return self.pop_size or official_population_size(self.task)

    @property
    def effective_budget(self) -> int:
        return self.budget if self.budget is not None else FORMAL_BUDGET

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / "eoh" / self.task

    def __post_init__(self):
        profile = resolve_backend(self.backend, self.base_url, self.model, self.no_proxy)
        self.base_url, self.model, self.no_proxy = profile.base_url, profile.model, profile.no_proxy
        self.experiments_root = self.experiments_root.resolve()
        if self.generations <= 0:
            raise ValueError("generations must be positive")
        if self.effective_budget <= 0:
            raise ValueError("budget must be positive")
        if self.effective_pop_size <= 0:
            raise ValueError("pop_size must be positive")
        if not 2 <= self.parents <= self.effective_pop_size:
            raise ValueError("parents must be between 2 and pop_size")
        if self.eval_workers is not None and self.eval_workers <= 0:
            raise ValueError("eval_workers must be positive")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")


def build_method(spec: RunSpec, log_dir: Path) -> EoH:
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.output_tokens,
    )
    return EoH(
        llm=llm,
        evaluation=evaluation,
        profiler=EoHProfiler(
            run_dir=log_dir.parent,
        ),
        max_generations=spec.generations,
        max_sample_nums=spec.effective_budget,
        pop_size=spec.effective_pop_size,
        selection_num=spec.parents,
        operators=list(OPERATORS),
        operator_weights=[1.0] * len(OPERATORS),
        use_m3_operator=False,
        num_samplers=1,
        num_evaluators=1,
        max_consecutive_sample_failures=20,
        multi_thread_or_process_eval="thread",
        debug_mode=False,
    )


def write_run_config(spec: RunSpec, run_dir: Path, run_name: str) -> None:
    write_json(
        run_dir / "run_config.json",
        baseline_run_config(spec, run_dir, run_name, "eoh", {
                "paper_generations": spec.generations,
                "max_sample_nums": spec.effective_budget,
                "population_size": spec.effective_pop_size,
                "initialization_samples": 2 * spec.effective_pop_size,
                "selection_num": spec.parents,
                "operators": list(OPERATORS),
                "operator_weights": [1.0] * len(OPERATORS),
                "m3_enabled": False,
                "num_samplers": 1,
                "num_evaluators": 1,
                "budget_basis": (
                    "all formal task comparisons use a fixed 1000-evaluation budget"
                ),
            }),
    )


def run_experiment(spec: RunSpec) -> Path:
    return run_baseline_experiment(
        spec, write_config=write_run_config, build_method=build_method,
        description=(f"eoh=pop={spec.effective_pop_size}, parents={spec.parents}, "
                     f"budget={spec.effective_budget}, operators={','.join(OPERATORS)}"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one official-source EoH experiment.")
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default=DEFAULT_BACKEND)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--generations", type=int, default=PAPER_GENERATIONS)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--parents", type=int, default=2)
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
            generations=args.generations,
            budget=args.budget,
            pop_size=args.pop_size,
            parents=args.parents,
            eval_workers=args.eval_workers,
            output_tokens=args.output_tokens,
            seed=args.seed,
            repeat=args.repeat,
            run_name=args.run_name,
        )
    )


if __name__ == "__main__":
    main()
