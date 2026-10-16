"""Run one paper-aligned ShinkaEvolve experiment at the unified 1000-eval budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .embedding import CodeEmbedding, embedding_settings
from baselines.shinka_evo import ShinkaEvo, ShinkaEvoProfiler

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


@dataclass(slots=True)
class RunSpec:
    task: str
    backend: str = DEFAULT_BACKEND
    base_url: str | None = None
    model: str | None = None
    no_proxy: str | None = None
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

    def __post_init__(self):
        profile = resolve_backend(self.backend, self.base_url, self.model, self.no_proxy)
        self.base_url, self.model, self.no_proxy = profile.base_url, profile.model, profile.no_proxy
        self.experiments_root = self.experiments_root.resolve()
        if self.max_sample_nums <= 0:
            raise ValueError("max_sample_nums must be positive")
        if self.num_generations <= 0:
            raise ValueError("num_generations must be positive")
        if self.num_islands <= 0:
            raise ValueError("num_islands must be positive")
        if self.archive_size <= 0:
            raise ValueError("archive_size must be positive")
        if not 0.0 < self.elite_selection_ratio <= 1.0:
            raise ValueError("elite_selection_ratio must be in (0, 1]")
        if self.eval_workers is not None and self.eval_workers <= 0:
            raise ValueError("eval_workers must be positive")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")


def build_method(spec: RunSpec, log_dir: Path) -> ShinkaEvo:
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.output_tokens,
    )
    # Paper uses a separate meta model; with a single generation endpoint we reuse it
    # so meta-scratchpad updates remain active.
    meta_llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.output_tokens,
    )
    return ShinkaEvo(
        llm=llm,
        evaluation=evaluation,
        profiler=ShinkaEvoProfiler(
            run_dir=log_dir.parent,
        ),
        max_sample_nums=spec.max_sample_nums,
        num_generations=spec.num_generations,
        num_islands=spec.num_islands,
        archive_size=spec.archive_size,
        patch_types=spec.patch_types,
        patch_type_probs=spec.patch_type_probs,
        parent_selection_strategy=spec.parent_selection_strategy,
        meta_llm=meta_llm,
        novelty_llm=llm,
        embedding_fn=CodeEmbedding(),
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
    write_json(
        run_dir / "run_config.json",
        baseline_run_config(spec, run_dir, run_name, "shinka_evo", {
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
                "novelty_rejection_enabled": True,
                "embedding_enabled": True,
                "embedding": embedding_settings(),
                "llm_ucb_exploration_coef": spec.llm_ucb_exploration_coef,
                "budget_basis": (
                    "Fair comparison budget max_sample_nums=num_generations=1000 "
                    "(paper Circle Packing used 150). Other hyperparams follow "
                    "Circle Packing table: archive_size=40, islands=2, "
                    "inspirations=4/2, migration_rate=0, patch probs [0.45,0.45,0.1], "
                    "meta interval=10, UCB exploration=1.0; embedding and novelty enabled; "
                    "meta and novelty LLMs reuse the selected generation model; initialization, rejected and evaluated candidates count toward the shared budget"
                ),
            }),
    )


def run_experiment(spec: RunSpec) -> Path:
    return run_baseline_experiment(
        spec, write_config=write_run_config, build_method=build_method,
        description=(f"shinka_evo=gens={spec.num_generations}, budget={spec.max_sample_nums}, "
                     f"archive={spec.archive_size}, islands={spec.num_islands}, "
                     f"insp={spec.num_archive_inspirations}/{spec.num_top_k_inspirations}"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one paper-aligned ShinkaEvolve experiment."
    )
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default=DEFAULT_BACKEND)
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
        RunSpec(
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
