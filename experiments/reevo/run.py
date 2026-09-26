"""Run one paper-aligned ReEvo experiment at the unified 1000-eval budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from baselines.reevo import ReEvo, ReEvoProfiler

from experiments.infra.base import (
    ALL_TASKS,
    BACKENDS,
    RESULTS_ROOT,
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

# Paper table / original cfg/config.yaml used max_fe=100 for sample-efficiency
# claims. Fair comparison across methods in this repo uses a unified budget.
PAPER_MAX_SAMPLE_NUMS = 100
FAIR_MAX_SAMPLE_NUMS = 1000
PAPER_POP_SIZE = 10
PAPER_INIT_POP_SIZE = 30
PAPER_MUTATION_RATE = 0.5


@dataclass(frozen=True, slots=True)
class RunSpec:
    task: TaskName
    backend: str
    base_url: str
    model: str
    no_proxy: str
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


def make_run_spec(
    *,
    task: TaskName,
    backend: str = "local",
    base_url: str | None = None,
    model: str | None = None,
    no_proxy: str | None = None,
    max_sample_nums: int = FAIR_MAX_SAMPLE_NUMS,
    pop_size: int = PAPER_POP_SIZE,
    init_pop_size: int = PAPER_INIT_POP_SIZE,
    mutation_rate: float = PAPER_MUTATION_RATE,
    eval_workers: int | None = None,
    output_tokens: int = 16384,
    seed: int = 0,
    repeat: int | None = None,
    run_name: str | None = None,
    experiments_root: Path = RESULTS_ROOT,
) -> RunSpec:
    profile = resolve_backend(backend, base_url, model, no_proxy)
    spec = RunSpec(
        task=task,
        backend=backend,
        base_url=profile.base_url,
        model=profile.model,
        no_proxy=profile.no_proxy,
        max_sample_nums=max_sample_nums,
        pop_size=pop_size,
        init_pop_size=init_pop_size,
        mutation_rate=mutation_rate,
        eval_workers=eval_workers,
        output_tokens=output_tokens,
        seed=seed,
        repeat=repeat,
        run_name=run_name,
        experiments_root=experiments_root.resolve(),
    )
    if spec.max_sample_nums <= 0:
        raise ValueError("max_sample_nums must be positive")
    if spec.pop_size <= 0:
        raise ValueError("pop_size must be positive")
    if spec.init_pop_size <= 0:
        raise ValueError("init_pop_size must be positive")
    if not 0.0 < spec.mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be in (0, 1]")
    if spec.eval_workers is not None and spec.eval_workers <= 0:
        raise ValueError("eval_workers must be positive")
    if spec.output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    return spec


def build_method(spec: RunSpec, log_dir: Path) -> ReEvo:
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.output_tokens,
        temperature=1.0,
    )
    return ReEvo(
        llm=llm,
        evaluation=evaluation,
        profiler=ReEvoProfiler(
            log_dir=str(log_dir),
            log_style="complex",
            create_random_path=False,
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
    _, task_config = build_task(spec.task, spec.eval_workers)
    write_run_config_file(
        run_dir,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "run_name": run_name,
            "task": spec.task,
            "method": "reevo",
            "repeat": spec.repeat,
            "backend": spec.backend,
            "seed": spec.seed,
            "llm": llm_payload(
                base_url=spec.base_url,
                model=spec.model,
                no_proxy=spec.no_proxy,
                max_tokens=spec.output_tokens,
                temperature=1.0,
            ),
            "task_eval": task_config,
            "method_params": {
                "max_sample_nums": spec.max_sample_nums,
                "population_size": spec.pop_size,
                "init_pop_size": spec.init_pop_size,
                "mutation_rate": spec.mutation_rate,
                "num_samplers": 1,
                "num_evaluators": 1,
                "init_temperature_offset": 0.3,
                "budget_basis": (
                    "Fair comparison budget max_sample_nums=1000 (paper default "
                    "max_fe=100 preserved as PAPER_MAX_SAMPLE_NUMS); other "
                    "hyperparams follow paper/original cfg: pop_size=10, "
                    "init_pop_size=30, mutation_rate=0.5, temperature=1; "
                    "initialization uses temperature+0.3"
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
            "reevo="
            f"pop={spec.pop_size}, init={spec.init_pop_size}, "
            f"budget={spec.max_sample_nums}, mutation_rate={spec.mutation_rate}",
        ],
        lambda: build_method(spec, log_dir).run(),
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one paper-aligned ReEvo experiment.")
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
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
        make_run_spec(
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
