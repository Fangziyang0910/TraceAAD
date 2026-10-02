"""Frozen V10.16 search protocol."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    budget: int = 1000
    roots: int = 8
    init_attempt_limit: int = 16
    history_depth: int = 8
    attempts_shown: int = 8  # most recent attempts from the current algorithm
    progress_shown: int = 8  # most recent improvements of the search best (Explore)
    root_tokens: int = 8000
    max_input_tokens: int = 24320
    output_tokens: int = 8192
    final_candidates: int = 5
    evaluation_seeds: tuple[int, ...] = (730241,)
    seed: int = 0

    def __post_init__(self):
        for name in ("budget", "roots", "init_attempt_limit", "history_depth", "attempts_shown",
                     "progress_shown", "root_tokens", "max_input_tokens", "output_tokens",
                     "final_candidates"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.roots != 8 or self.init_attempt_limit != 16 or self.history_depth != 8:
            raise ValueError("V10.16 fixes 8 roots, 16 initialization attempts and 8 history steps")
        if self.attempts_shown != 8 or self.progress_shown != 8:
            raise ValueError("V10.16 shows up to 8 attempts and 8 search improvements")
        if self.final_candidates != 5:
            raise ValueError("V10.16 selects from five training finalists")
        if not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("evaluation seeds must be nonempty and unique")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.evaluation_seeds):
            raise ValueError("evaluation seeds must be uint32 integers")
