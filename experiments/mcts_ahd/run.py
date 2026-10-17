"""Run one MCTS-AHD experiment at the unified 1000-eval budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from baselines.mcts_ahd import MAProfiler, MCTS_AHD

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

# Paper-aligned defaults (MCTS-AHD icml2025: N_I=4, lambda_0=0.1, alpha=0.5).
MAX_SAMPLE_NUMS = 1000
INIT_SIZE = 4
POP_SIZE = 10
SELECTION_NUM = 2
NUM_SAMPLERS = 4
# Every method evaluates one candidate at a time, so methods load the host
# alike; the evaluator count only sets how many evaluations run concurrently.
NUM_EVALUATORS = 1
ALPHA = 0.5
LAMBDA_0 = 0.1
MAX_CONSECUTIVE_SAMPLE_FAILURES = 20
EVAL_EXECUTOR = "thread"


@dataclass(slots=True)
class RunSpec:
    task: str
    backend: str = DEFAULT_BACKEND
    base_url: str | None = None
    model: str | None = None
    no_proxy: str | None = None
    max_sample_nums: int = MAX_SAMPLE_NUMS
    init_size: int = INIT_SIZE
    pop_size: int = POP_SIZE
    selection_num: int = SELECTION_NUM
    num_samplers: int = NUM_SAMPLERS
    num_evaluators: int = NUM_EVALUATORS
    alpha: float = ALPHA
    lambda_0: float = LAMBDA_0
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = RESULTS_ROOT

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / "mcts_ahd" / self.task

    def __post_init__(self):
        profile = resolve_backend(self.backend, self.base_url, self.model, self.no_proxy)
        self.base_url, self.model, self.no_proxy = profile.base_url, profile.model, profile.no_proxy
        self.experiments_root = self.experiments_root.resolve()
        if self.max_sample_nums <= 0:
            raise ValueError("max_sample_nums must be positive")
        if self.init_size <= 0:
            raise ValueError("init_size must be positive")
        if self.pop_size <= 0:
            raise ValueError("pop_size must be positive")
        if not 1 <= self.selection_num <= self.pop_size:
            raise ValueError("selection_num must be between 1 and pop_size")
        if self.num_samplers <= 0:
            raise ValueError("num_samplers must be positive")
        if self.num_evaluators <= 0:
            raise ValueError("num_evaluators must be positive")
        if self.eval_workers is not None and self.eval_workers <= 0:
            raise ValueError("eval_workers must be positive")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")


def build_method(spec: RunSpec, log_dir: Path) -> MCTS_AHD:
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.output_tokens,
    )
    return MCTS_AHD(
        llm=llm,
        evaluation=evaluation,
        profiler=MAProfiler(run_dir=log_dir.parent),
        max_sample_nums=spec.max_sample_nums,
        init_size=spec.init_size,
        pop_size=spec.pop_size,
        selection_num=spec.selection_num,
        num_samplers=spec.num_samplers,
        num_evaluators=spec.num_evaluators,
        alpha=spec.alpha,
        lambda_0=spec.lambda_0,
        max_consecutive_sample_failures=MAX_CONSECUTIVE_SAMPLE_FAILURES,
        multi_thread_or_process_eval=EVAL_EXECUTOR,
        debug_mode=False,
    )


def write_run_config(spec: RunSpec, run_dir: Path, run_name: str) -> None:
    write_json(
        run_dir / "run_config.json",
        baseline_run_config(spec, run_dir, run_name, "mcts_ahd", {
                "max_sample_nums": spec.max_sample_nums,
                "init_size": spec.init_size,
                "pop_size": spec.pop_size,
                "selection_num": spec.selection_num,
                "num_samplers": spec.num_samplers,
                "num_evaluators": spec.num_evaluators,
                "alpha": spec.alpha,
                "lambda_0": spec.lambda_0,
                "max_consecutive_sample_failures": MAX_CONSECUTIVE_SAMPLE_FAILURES,
                "eval_executor": EVAL_EXECUTOR,
                "debug": False,
            }, extra={"timestamp": run_name}),
    )


def run_experiment(spec: RunSpec) -> Path:
    return run_baseline_experiment(
        spec, write_config=write_run_config, build_method=build_method,
        description=(f"mcts_ahd=init={spec.init_size}, pop={spec.pop_size}, "
                     f"budget={spec.max_sample_nums}, "
                     f"lambda_0={spec.lambda_0}, alpha={spec.alpha}"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one MCTS-AHD experiment at 1000-eval budget."
    )
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default=DEFAULT_BACKEND)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--max-sample-nums", type=int, default=MAX_SAMPLE_NUMS)
    parser.add_argument("--init-size", type=int, default=INIT_SIZE)
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--selection-num", type=int, default=SELECTION_NUM)
    parser.add_argument("--num-samplers", type=int, default=NUM_SAMPLERS)
    parser.add_argument("--num-evaluators", type=int, default=NUM_EVALUATORS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--lambda-0", type=float, default=LAMBDA_0)
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
            init_size=args.init_size,
            pop_size=args.pop_size,
            selection_num=args.selection_num,
            num_samplers=args.num_samplers,
            num_evaluators=args.num_evaluators,
            alpha=args.alpha,
            lambda_0=args.lambda_0,
            eval_workers=args.eval_workers,
            output_tokens=args.output_tokens,
            seed=args.seed,
            repeat=args.repeat,
            run_name=args.run_name,
        )
    )


if __name__ == "__main__":
    main()
