"""Run one paper-aligned ShinkaEvolve experiment at the unified 1000-eval budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from baselines.shinka_evo import ShinkaEvo, ShinkaEvoProfiler

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

# Paper Circle Packing table used generations=150. Fair comparison across
# methods in this repo uses a unified evaluation budget of 1000.
PAPER_MAX_SAMPLE_NUMS = 150
PAPER_NUM_GENERATIONS = 150
FAIR_MAX_SAMPLE_NUMS = 1000
FAIR_NUM_GENERATIONS = 1000
PAPER_NUM_ISLANDS = 2
PAPER_ARCHIVE_SIZE = 40
PAPER_ELITE_SELECTION_RATIO = 0.3
PAPER_NUM_ARCHIVE_INSPIRATIONS = 4
PAPER_NUM_TOP_K_INSPIRATIONS = 2
PAPER_MIGRATION_INTERVAL = 10
PAPER_MIGRATION_RATE = 0.0
PAPER_PARENT_SELECTION_STRATEGY = "weighted"
PAPER_PARENT_SELECTION_LAMBDA = 10.0
PAPER_PATCH_TYPES = ("diff", "full", "cross")
PAPER_PATCH_TYPE_PROBS = (0.45, 0.45, 0.1)
PAPER_MAX_PATCH_RESAMPLES = 3
PAPER_META_REC_INTERVAL = 10
PAPER_LLM_UCB_EXPLORATION_COEF = 1.0


@dataclass(frozen=True, slots=True)
class RunSpec:
    task: TaskName
    backend: str
    base_url: str
    model: str
    no_proxy: str
    max_sample_nums: int = FAIR_MAX_SAMPLE_NUMS
    num_generations: int = FAIR_NUM_GENERATIONS
    num_islands: int = PAPER_NUM_ISLANDS
    archive_size: int = PAPER_ARCHIVE_SIZE
    elite_selection_ratio: float = PAPER_ELITE_SELECTION_RATIO
    num_archive_inspirations: int = PAPER_NUM_ARCHIVE_INSPIRATIONS
    num_top_k_inspirations: int = PAPER_NUM_TOP_K_INSPIRATIONS
    migration_interval: int = PAPER_MIGRATION_INTERVAL
    migration_rate: float = PAPER_MIGRATION_RATE
    parent_selection_strategy: str = PAPER_PARENT_SELECTION_STRATEGY
    parent_selection_lambda: float = PAPER_PARENT_SELECTION_LAMBDA
    patch_types: tuple[str, ...] = PAPER_PATCH_TYPES
    patch_type_probs: tuple[float, ...] = PAPER_PATCH_TYPE_PROBS
    max_patch_resamples: int = PAPER_MAX_PATCH_RESAMPLES
    meta_rec_interval: int = PAPER_META_REC_INTERVAL
    llm_ucb_exploration_coef: float = PAPER_LLM_UCB_EXPLORATION_COEF
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = RESULTS_ROOT

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / "shinka_evo" / self.task


def make_run_spec(
    *,
    task: TaskName,
    backend: str = "local",
    base_url: str | None = None,
    model: str | None = None,
    no_proxy: str | None = None,
    max_sample_nums: int = FAIR_MAX_SAMPLE_NUMS,
    num_generations: int = FAIR_NUM_GENERATIONS,
    num_islands: int = PAPER_NUM_ISLANDS,
    archive_size: int = PAPER_ARCHIVE_SIZE,
    elite_selection_ratio: float = PAPER_ELITE_SELECTION_RATIO,
    num_archive_inspirations: int = PAPER_NUM_ARCHIVE_INSPIRATIONS,
    num_top_k_inspirations: int = PAPER_NUM_TOP_K_INSPIRATIONS,
    migration_interval: int = PAPER_MIGRATION_INTERVAL,
    migration_rate: float = PAPER_MIGRATION_RATE,
    parent_selection_strategy: str = PAPER_PARENT_SELECTION_STRATEGY,
    parent_selection_lambda: float = PAPER_PARENT_SELECTION_LAMBDA,
    patch_types: tuple[str, ...] = PAPER_PATCH_TYPES,
    patch_type_probs: tuple[float, ...] = PAPER_PATCH_TYPE_PROBS,
    max_patch_resamples: int = PAPER_MAX_PATCH_RESAMPLES,
    meta_rec_interval: int = PAPER_META_REC_INTERVAL,
    llm_ucb_exploration_coef: float = PAPER_LLM_UCB_EXPLORATION_COEF,
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
        num_generations=num_generations,
        num_islands=num_islands,
        archive_size=archive_size,
        elite_selection_ratio=elite_selection_ratio,
        num_archive_inspirations=num_archive_inspirations,
        num_top_k_inspirations=num_top_k_inspirations,
        migration_interval=migration_interval,
        migration_rate=migration_rate,
        parent_selection_strategy=parent_selection_strategy,
        parent_selection_lambda=parent_selection_lambda,
        patch_types=patch_types,
        patch_type_probs=patch_type_probs,
        max_patch_resamples=max_patch_resamples,
        meta_rec_interval=meta_rec_interval,
        llm_ucb_exploration_coef=llm_ucb_exploration_coef,
        eval_workers=eval_workers,
        output_tokens=output_tokens,
        seed=seed,
        repeat=repeat,
        run_name=run_name,
        experiments_root=experiments_root.resolve(),
    )
    if spec.max_sample_nums <= 0:
        raise ValueError("max_sample_nums must be positive")
    if spec.num_generations <= 0:
        raise ValueError("num_generations must be positive")
    if spec.num_islands <= 0:
        raise ValueError("num_islands must be positive")
    if spec.archive_size <= 0:
        raise ValueError("archive_size must be positive")
    if not 0.0 < spec.elite_selection_ratio <= 1.0:
        raise ValueError("elite_selection_ratio must be in (0, 1]")
    if spec.eval_workers is not None and spec.eval_workers <= 0:
        raise ValueError("eval_workers must be positive")
    if spec.output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    return spec


def build_method(spec: RunSpec, log_dir: Path) -> ShinkaEvo:
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.output_tokens,
        temperature=1.0,
    )
    # Paper uses a separate meta model; with a single Qwen endpoint we reuse it
    # so meta-scratchpad updates remain active.
    meta_llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.output_tokens,
        temperature=1.0,
    )
    return ShinkaEvo(
        llm=llm,
        evaluation=evaluation,
        profiler=ShinkaEvoProfiler(
            log_dir=str(log_dir),
            log_style="complex",
            create_random_path=False,
        ),
        max_sample_nums=spec.max_sample_nums,
        num_generations=spec.num_generations,
        num_islands=spec.num_islands,
        archive_size=spec.archive_size,
        patch_types=spec.patch_types,
        patch_type_probs=spec.patch_type_probs,
        parent_selection_strategy=spec.parent_selection_strategy,
        meta_llm=meta_llm,
        novelty_llm=None,
        embedding_fn=None,
        max_patch_resamples=spec.max_patch_resamples,
        elite_selection_ratio=spec.elite_selection_ratio,
        num_archive_inspirations=spec.num_archive_inspirations,
        num_top_k_inspirations=spec.num_top_k_inspirations,
        parent_selection_lambda=spec.parent_selection_lambda,
        migration_interval=spec.migration_interval,
        migration_rate=spec.migration_rate,
        island_elitism=True,
        meta_rec_interval=spec.meta_rec_interval,
        sample_single_meta_rec=True,
        use_text_feedback=False,
        random_seed=spec.seed,
        llm_ucb_exploration_coef=spec.llm_ucb_exploration_coef,
        llm_ucb_epsilon=0.2,
        llm_ucb_auto_decay=0.95,
        max_consecutive_sample_failures=20,
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
            "method": "shinka_evo",
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
                "num_generations": spec.num_generations,
                "num_islands": spec.num_islands,
                "archive_size": spec.archive_size,
                "elite_selection_ratio": spec.elite_selection_ratio,
                "num_archive_inspirations": spec.num_archive_inspirations,
                "num_top_k_inspirations": spec.num_top_k_inspirations,
                "migration_interval": spec.migration_interval,
                "migration_rate": spec.migration_rate,
                "parent_selection_strategy": spec.parent_selection_strategy,
                "parent_selection_lambda": spec.parent_selection_lambda,
                "patch_types": list(spec.patch_types),
                "patch_type_probs": list(spec.patch_type_probs),
                "max_patch_resamples": spec.max_patch_resamples,
                "meta_rec_interval": spec.meta_rec_interval,
                "meta_llm_enabled": True,
                "novelty_rejection_enabled": False,
                "embedding_enabled": False,
                "llm_ucb_exploration_coef": spec.llm_ucb_exploration_coef,
                "budget_basis": (
                    "Fair comparison budget max_sample_nums=num_generations=1000 "
                    "(paper Circle Packing used 150). Other hyperparams follow "
                    "Circle Packing table: archive_size=40, islands=2, "
                    "inspirations=4/2, migration_rate=0, patch probs [0.45,0.45,0.1], "
                    "meta interval=10, UCB exploration=1.0; novelty disabled; "
                    "meta LLM reuses the same Qwen endpoint"
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
            "shinka_evo="
            f"gens={spec.num_generations}, budget={spec.max_sample_nums}, "
            f"archive={spec.archive_size}, islands={spec.num_islands}, "
            f"insp={spec.num_archive_inspirations}/{spec.num_top_k_inspirations}",
        ],
        lambda: build_method(spec, log_dir).run(),
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one paper-aligned ShinkaEvolve experiment."
    )
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--max-sample-nums", type=int, default=FAIR_MAX_SAMPLE_NUMS)
    parser.add_argument("--num-generations", type=int, default=FAIR_NUM_GENERATIONS)
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
            num_generations=args.num_generations,
            eval_workers=args.eval_workers,
            output_tokens=args.output_tokens,
            seed=args.seed,
            repeat=args.repeat,
            run_name=args.run_name,
        )
    )


if __name__ == "__main__":
    main()
