"""Frozen V10.18 search protocol."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    budget: int = 1000
    roots: int = 8
    init_attempt_limit: int = 16
    history_depth: int = 8
    attempts_shown: int = 8  # most recent attempts from the current algorithm
    progress_shown: int = 8  # most recent improvements of the search best (Explore)
    # Shares of the operator draw. An Explore draw develops the open exploration when
    # there is one and proposes a new design otherwise, so proposals slow down while
    # development is under way and the two share one budget.
    refine_share: float = 0.35
    explore_share: float = 0.40
    crossover_share: float = 0.25
    # A new design is developed until it shows whether it works: at least
    # ``development_min`` steps, then while it keeps improving (it ends after
    # ``development_stall`` steps in a row without a better version), at most
    # ``development_max`` steps. It joins ordinary competition, and development ends,
    # once its best version is among the ``final_candidates`` best programs of the search.
    development_min: int = 2
    development_stall: int = 2
    development_max: int = 8
    root_tokens: int = 8000
    max_input_tokens: int = 24320
    output_tokens: int = 8192
    final_candidates: int = 5
    evaluation_seeds: tuple[int, ...] = (730241,)
    seed: int = 0

    def __post_init__(self):
        for name in ("budget", "roots", "init_attempt_limit", "history_depth", "attempts_shown",
                     "progress_shown", "development_min", "development_stall", "development_max",
                     "root_tokens", "max_input_tokens", "output_tokens", "final_candidates"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.roots != 8 or self.init_attempt_limit != 16 or self.history_depth != 8:
            raise ValueError("V10.18 fixes 8 roots, 16 initialization attempts and 8 history steps")
        if self.attempts_shown != 8 or self.progress_shown != 8:
            raise ValueError("V10.18 shows up to 8 attempts and 8 search improvements")
        shares = (self.refine_share, self.explore_share, self.crossover_share)
        if any(not 0 < s < 1 for s in shares) or abs(sum(shares) - 1) > 1e-9:
            raise ValueError("operator shares must be positive and sum to one")
        if not self.development_min <= self.development_max:
            raise ValueError("development_min must not exceed development_max")
        if self.final_candidates != 5:
            raise ValueError("V10.18 selects from five training finalists")
        if not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("evaluation seeds must be nonempty and unique")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.evaluation_seeds):
            raise ValueError("evaluation seeds must be uint32 integers")
