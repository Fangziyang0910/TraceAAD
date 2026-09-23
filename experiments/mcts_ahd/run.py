"""Run one MCTS-AHD experiment at the unified 1000-eval budget."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from baselines.mcts_ahd import MAProfiler, MCTS_AHD

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
    write_run_config as write_run_config_file,
)

# Paper-aligned defaults (MCTS-AHD icml2025: N_I=4, lambda_0=0.1, alpha=0.5).
MAX_SAMPLE_NUMS = 1000
INIT_SIZE = 4
POP_SIZE = 10
SELECTION_NUM = 2
NUM_SAMPLERS = 4
NUM_EVALUATORS = 4
ALPHA = 0.5
LAMBDA_0 = 0.1
MAX_CONSECUTIVE_SAMPLE_FAILURES = 20
EVAL_EXECUTOR = "thread"


@dataclass(frozen=True, slots=True)
class RunSpec:
    task: TaskName
    backend: str
    base_url: str
    model: str
    no_proxy: str
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
    experiments_root: Path = EXPERIMENTS_ROOT

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / self.task / "mcts_ahd"


def make_run_spec(
    *,
    task: TaskName,
    backend: str = "local",
    base_url: str | None = None,
    model: str | None = None,
    no_proxy: str | None = None,
    max_sample_nums: int = MAX_SAMPLE_NUMS,
    init_size: int = INIT_SIZE,
    pop_size: int = POP_SIZE,
    selection_num: int = SELECTION_NUM,
    num_samplers: int = NUM_SAMPLERS,
    num_evaluators: int = NUM_EVALUATORS,
    alpha: float = ALPHA,
    lambda_0: float = LAMBDA_0,
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
        init_size=init_size,
        pop_size=pop_size,
        selection_num=selection_num,
        num_samplers=num_samplers,
        num_evaluators=num_evaluators,
        alpha=alpha,
        lambda_0=lambda_0,
        eval_workers=eval_workers,
        output_tokens=output_tokens,
        seed=seed,
        repeat=repeat,
        run_name=run_name,
        experiments_root=experiments_root.resolve(),
    )
    if spec.max_sample_nums <= 0:
        raise ValueError("max_sample_nums must be positive")
    if spec.init_size <= 0:
        raise ValueError("init_size must be positive")
    if spec.pop_size <= 0:
        raise ValueError("pop_size must be positive")
    if not 1 <= spec.selection_num <= spec.pop_size:
        raise ValueError("selection_num must be between 1 and pop_size")
    if spec.num_samplers <= 0:
        raise ValueError("num_samplers must be positive")
    if spec.num_evaluators <= 0:
        raise ValueError("num_evaluators must be positive")
    if spec.eval_workers is not None and spec.eval_workers <= 0:
        raise ValueError("eval_workers must be positive")
    if spec.output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    return spec


def build_method(spec: RunSpec, log_dir: Path) -> MCTS_AHD:
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.output_tokens,
        temperature=1.0,
    )
    return MCTS_AHD(
        llm=llm,
        evaluation=evaluation,
        profiler=MAProfiler(log_dir=str(log_dir), log_style="complex", create_random_path=False),
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
    _, task_config = build_task(spec.task, spec.eval_workers)
    write_run_config_file(
        run_dir,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "task": spec.task,
            "method": "mcts_ahd",
            "timestamp": run_name,
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
            f"mcts_ahd=init={spec.init_size}, pop={spec.pop_size}, "
            f"budget={spec.max_sample_nums}, "
            f"lambda_0={spec.lambda_0}, alpha={spec.alpha}",
        ],
        lambda: build_method(spec, log_dir).run(),
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one MCTS-AHD experiment at 1000-eval budget."
    )
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
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
        make_run_spec(
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
