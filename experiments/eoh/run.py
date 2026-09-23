"""Run one paper-aligned EoH experiment at the unified 1000-eval budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from baselines.eoh import EoH, EoHProfiler

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

FORMAL_BUDGET = 1000
PAPER_GENERATIONS = 20
OPERATORS = ("e1", "e2", "m1", "m2")


def paper_population_size(task: TaskName) -> int:
    return 20 if task == "online_bin_packing" else 10


@dataclass(frozen=True, slots=True)
class RunSpec:
    task: TaskName
    backend: str
    base_url: str
    model: str
    no_proxy: str
    generations: int = PAPER_GENERATIONS
    budget: int | None = None
    pop_size: int | None = None
    parents: int = 5
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = EXPERIMENTS_ROOT

    @property
    def effective_pop_size(self) -> int:
        return self.pop_size or paper_population_size(self.task)

    @property
    def effective_budget(self) -> int:
        return self.budget if self.budget is not None else FORMAL_BUDGET

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / self.task / "eoh"


def make_run_spec(
    *,
    task: TaskName,
    backend: str = "local",
    base_url: str | None = None,
    model: str | None = None,
    no_proxy: str | None = None,
    generations: int = PAPER_GENERATIONS,
    budget: int | None = None,
    pop_size: int | None = None,
    parents: int = 5,
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
        generations=generations,
        budget=budget,
        pop_size=pop_size,
        parents=parents,
        eval_workers=eval_workers,
        output_tokens=output_tokens,
        seed=seed,
        repeat=repeat,
        run_name=run_name,
        experiments_root=experiments_root.resolve(),
    )
    if spec.generations <= 0:
        raise ValueError("generations must be positive")
    if spec.effective_budget <= 0:
        raise ValueError("budget must be positive")
    if spec.effective_pop_size <= 0:
        raise ValueError("pop_size must be positive")
    if not 2 <= spec.parents <= spec.effective_pop_size:
        raise ValueError("parents must be between 2 and pop_size")
    if spec.eval_workers is not None and spec.eval_workers <= 0:
        raise ValueError("eval_workers must be positive")
    if spec.output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    return spec


def build_method(spec: RunSpec, log_dir: Path) -> EoH:
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.output_tokens,
        temperature=1.0,
    )
    return EoH(
        llm=llm,
        evaluation=evaluation,
        profiler=EoHProfiler(
            log_dir=str(log_dir),
            log_style="complex",
            create_random_path=False,
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
    _, task_config = build_task(spec.task, spec.eval_workers)
    write_run_config_file(
        run_dir,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "run_name": run_name,
            "task": spec.task,
            "method": "eoh",
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
            "eoh="
            f"pop={spec.effective_pop_size}, parents={spec.parents}, "
            f"budget={spec.effective_budget}, operators={','.join(OPERATORS)}",
        ],
        lambda: build_method(spec, log_dir).run(),
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one paper-aligned EoH experiment.")
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--generations", type=int, default=PAPER_GENERATIONS)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--parents", type=int, default=5)
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
