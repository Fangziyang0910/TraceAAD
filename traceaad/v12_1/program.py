"""One parsed model-generated program and everything learned about it."""

from dataclasses import dataclass, field
from datetime import datetime

from .canonical import key


@dataclass
class Program:
    id: int
    code: str
    idea: str
    analysis: str | None
    action: str
    generation: int
    parent_id: int | None = None
    reference_id: int | None = None
    repair_of: int | None = None
    depth: int = 0
    status: str = "pending"
    fitness: float | None = None
    failure: dict | None = None
    eval_seconds: float | None = None
    eval_wall_seconds: float | None = None
    calls: int | None = None
    function_seconds: float | None = None
    request_id: int | None = None
    call_ids: list[int] = field(default_factory=list)
    request: dict = field(default_factory=dict)
    link: dict | None = None
    exploration: dict | None = None
    verdict: dict | None = None
    created_at: str = field(
        default_factory=lambda: (
            datetime.now().astimezone().isoformat(timespec="seconds")
        )
    )

    @property
    def valid(self):
        return self.status == "valid"

    @property
    def score(self):
        return self.fitness

    @property
    def key(self):
        """Artifact integrity hash; never used to identify another individual."""
        return key(self.code)

    @property
    def repaired(self):
        return self.repair_of is not None
