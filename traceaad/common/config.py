"""Shared search parameters; method-specific parameters stay in each version."""

from dataclasses import dataclass

REVISION = "research-protocol-v4-20261008"


@dataclass
class SearchConfig:
    budget: int = 1000
    roots: int = 8
    init_attempt_limit: int = 16
    history_depth: int = 8
    root_tokens: int = 8000
    max_input_tokens: int = 24320
    output_tokens: int = 8192
    final_candidates: int = 5
    evaluation_seeds: tuple[int, ...] = (730241,)
    seed: int = 0

    def __post_init__(self):
        if min(self.budget, self.roots, self.init_attempt_limit, self.history_depth,
               self.root_tokens, self.max_input_tokens, self.output_tokens, self.final_candidates) < 1:
            raise ValueError("search limits must be positive")
        if not self.evaluation_seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in self.evaluation_seeds):
            raise ValueError("evaluation seeds must be nonempty uint32 integers")
