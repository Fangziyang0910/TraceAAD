"""Run only EvoTrace initialization, returning the evaluated tree."""

from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config
from .task import Task
from .search import EvoTrace

from .models import TreePopulation


@dataclass
class InitializationResult:
    population: TreePopulation
    stop_reason: str
    evaluations: int
    model_calls: int


def initialize(task: Task, model, *, config: Config | None = None,
               run_dir: str | Path | None = None) -> InitializationResult:
    """Stop after initial_size feasible algorithms; retain evaluated failures as leaves."""
    config = config or Config()
    if config.initial_size < 1:
        raise ValueError("initial_size must be positive")
    if run_dir is not None and (Path(run_dir) / "run_config.json").exists():
        raise FileExistsError("Initialization requires a new run directory")

    search = EvoTrace(task, model, config=config, run_dir=run_dir)
    stop_reason = "initialized"
    while sum(p.valid and p.operator == "Init" for p in search.archive.values()) < config.initial_size:
        if search.step() is None:
            stop_reason = search.stop_reason
            break
    result = InitializationResult(
        search.tree, stop_reason, search.evaluations_used, search.model_calls,
    )
    search.store.write("initialization.json", asdict(result))
    return result
