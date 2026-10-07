"""Run one FunSearch experiment at the unified 1000-sample budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from baselines.funsearch import FunSearch, FunSearchProfiler
from baselines.funsearch.config import ProgramsDatabaseConfig

from experiments.infra.base import (
    ALL_TASKS,
    BACKENDS,
    RESULTS_ROOT,
    TaskName,
    build_llm_client,
    build_task,
    resolve_backend,
    resolve_run_dir as resolve_run_dir_file,
    set_random_seed,
    write_run_config as write_run_config_file,
)
from experiments.infra.runner import baseline_run_config, run_baseline_experiment

FORMAL_BUDGET = 1000
# google-deepmind/funsearch implementation/config.py
PAPER_SAMPLES_PER_PROMPT = 4
PAPER_DATABASE = ProgramsDatabaseConfig()


@dataclass(frozen=True, slots=True)
class RunSpec:
    task: TaskName
    backend: str
    base_url: str
    model: str
    no_proxy: str
    budget: int = FORMAL_BUDGET
    samples_per_prompt: int = PAPER_SAMPLES_PER_PROMPT
    num_islands: int = PAPER_DATABASE.num_islands
    functions_per_prompt: int = PAPER_DATABASE.functions_per_prompt
    reset_period: float = PAPER_DATABASE.reset_period
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = RESULTS_ROOT

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / "funsearch" / self.task

    @property
    def database_config(self) -> ProgramsDatabaseConfig:
        return ProgramsDatabaseConfig(
            functions_per_prompt=self.functions_per_prompt,
            num_islands=self.num_islands,
            reset_period=self.reset_period,
            cluster_sampling_temperature_init=PAPER_DATABASE.cluster_sampling_temperature_init,
            cluster_sampling_temperature_period=PAPER_DATABASE.cluster_sampling_temperature_period,
        )


def make_run_spec(
    *,
    task: TaskName,
    backend: str = "local",
    base_url: str | None = None,
    model: str | None = None,
    no_proxy: str | None = None,
    budget: int = FORMAL_BUDGET,
    samples_per_prompt: int = PAPER_SAMPLES_PER_PROMPT,
    num_islands: int = PAPER_DATABASE.num_islands,
    functions_per_prompt: int = PAPER_DATABASE.functions_per_prompt,
    reset_period: float = PAPER_DATABASE.reset_period,
    eval_workers: int | None = None,
    output_tokens: int = 16384,
    seed: int = 0,
    repeat: int | None = None,
    run_name: str | None = None,
    experiments_root: Path = RESULTS_ROOT,
) -> RunSpec:
    profile = resolve_backend(backend, base_url, model, no_proxy)
    spec = RunSpec(
        task=task, backend=backend, base_url=profile.base_url, model=profile.model,
        no_proxy=profile.no_proxy, budget=budget, samples_per_prompt=samples_per_prompt,
        num_islands=num_islands, functions_per_prompt=functions_per_prompt,
        reset_period=reset_period, eval_workers=eval_workers, output_tokens=output_tokens,
        seed=seed, repeat=repeat, run_name=run_name, experiments_root=experiments_root.resolve(),
    )
    if spec.budget <= 0:
        raise ValueError("budget must be positive")
    if spec.samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")
    if spec.num_islands < 2:
        raise ValueError("num_islands must be at least 2")
    if spec.functions_per_prompt <= 0:
        raise ValueError("functions_per_prompt must be positive")
    if spec.reset_period <= 0:
        raise ValueError("reset_period must be positive")
    if spec.eval_workers is not None and spec.eval_workers <= 0:
        raise ValueError("eval_workers must be positive")
    if spec.output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    return spec


def build_method(spec: RunSpec, log_dir: Path) -> FunSearch:
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.output_tokens,
    )
    return FunSearch(
        llm=llm,
        evaluation=evaluation,
        profiler=FunSearchProfiler(run_dir=log_dir.parent),
        max_sample_nums=spec.budget,
        samples_per_prompt=spec.samples_per_prompt,
        database_config=spec.database_config,
        max_consecutive_sample_failures=20,
        debug_mode=False,
    )


def write_run_config(spec: RunSpec, run_dir: Path, run_name: str) -> None:
    config = spec.database_config
    write_run_config_file(
        run_dir,
        baseline_run_config(spec, run_dir, run_name, "funsearch", {
            "max_sample_nums": spec.budget,
            "samples_per_prompt": spec.samples_per_prompt,
            "num_islands": config.num_islands,
            "functions_per_prompt": config.functions_per_prompt,
            "reset_period_seconds": config.reset_period,
            "cluster_sampling_temperature_init": config.cluster_sampling_temperature_init,
            "cluster_sampling_temperature_period": config.cluster_sampling_temperature_period,
            "num_samplers": 1,
            "num_evaluators": 1,
            "budget_basis": "every model sample counts, including samples whose code cannot be parsed",
        }),
    )


def resolve_run_dir(spec: RunSpec) -> tuple[Path, str]:
    return resolve_run_dir_file(spec.experiment_root, spec.run_name)


def run_experiment(spec: RunSpec) -> Path:
    return run_baseline_experiment(
        spec, write_config=write_run_config, build_method=build_method,
        description=(f"funsearch=islands={spec.num_islands}, functions_per_prompt={spec.functions_per_prompt}, "
                     f"samples_per_prompt={spec.samples_per_prompt}, budget={spec.budget}"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one FunSearch experiment.")
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--budget", type=int, default=FORMAL_BUDGET)
    parser.add_argument("--samples-per-prompt", type=int, default=PAPER_SAMPLES_PER_PROMPT)
    parser.add_argument("--num-islands", type=int, default=PAPER_DATABASE.num_islands)
    parser.add_argument("--functions-per-prompt", type=int, default=PAPER_DATABASE.functions_per_prompt)
    parser.add_argument("--reset-period", type=float, default=PAPER_DATABASE.reset_period,
                        help="seconds between resets of the weaker half of the islands")
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument("--output-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--run-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(make_run_spec(
        task=args.task, backend=args.backend, base_url=args.base_url, model=args.model,
        no_proxy=args.no_proxy, budget=args.budget, samples_per_prompt=args.samples_per_prompt,
        num_islands=args.num_islands, functions_per_prompt=args.functions_per_prompt,
        reset_period=args.reset_period, eval_workers=args.eval_workers,
        output_tokens=args.output_tokens, seed=args.seed, repeat=args.repeat, run_name=args.run_name,
    ))


if __name__ == "__main__":
    main()
