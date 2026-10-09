"""A fixed block policy, with the existing candidate and evaluation protocol."""

from dataclasses import dataclass

from benchmarks.tasks import INSTANCE_SECONDS
from traceaad.common.config import SearchConfig
from traceaad.common.instance_evaluation import validate_execution

EXPERIMENT = "traceaad_v10_24"


@dataclass
class Config(SearchConfig):
    block_size: int = 4
    frontier_groups: int = 8
    eval_timeout_seconds: float = float(INSTANCE_SECONDS)
    eval_workers: int = 1
    scheduler_socket: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if any(type(v) is not int or v < 1 for v in (self.block_size, self.frontier_groups)):
            raise ValueError("block_size and frontier_groups must be positive integers")
        validate_execution(self.eval_timeout_seconds, self.eval_workers)
