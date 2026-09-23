"""Heuristic archive records for CALM (w/o GRPO)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(eq=False)
class HeuristicRecord:
    code: str
    idea: str
    name: str
    parent_prompt_type: str = 'seed'
    perf: Optional[float] = None
    perfs: Optional[np.ndarray] = None
    birth: int = 0
    response: str = ''
    code_key: Optional[str] = None
    last_used_epoch: int = 0
    born_from_revisit: bool = False
    function: Any = None
    extras: dict = field(default_factory=dict)

    @property
    def perf_str(self) -> str:
        assert self.perf is not None
        return str(np.floor(1000 * abs(self.perf)) / 1000)

    @property
    def sid(self) -> str:
        assert self.perf is not None
        return (
            f"{self.parent_prompt_type}(Perf={str(np.floor(1000 * abs(self.perf)) / 1000)}, "
            f"Step={self.birth}))"
        )

    def __eq__(self, other):
        # Match reference_code/CALM HeuristicPolicy: equality by stripped code.
        return isinstance(other, HeuristicRecord) and self.code.strip() == other.code.strip()

    def __hash__(self):
        # Use stripped code so hash is consistent with __eq__.
        return hash(self.code.strip())
