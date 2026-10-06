"""Frozen V10.19 search protocol."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    budget: int = 1000
    roots: int = 8
    init_attempt_limit: int = 16
    history_depth: int = 8
    attempts_shown: int = 8  # most recent attempts from the current algorithm
    progress_shown: int = 8  # most recent improvements of the search best (Explore)
    # Shares of the operator draw. An Explore draw continues the open change when
    # there is one and proposes a new change otherwise, so proposals slow down while
    # a change is being made to work and the two share one budget.
    refine_share: float = 0.40
    explore_share: float = 0.35
    crossover_share: float = 0.25
    # A change an Explore proposal makes is judged against the algorithm it was made
    # to: when its first version scores worse than that algorithm, up to
    # ``development_steps`` steps try to make it work there; the change ends as soon as
    # a version scores better than that algorithm.
    development_steps: int = 3
    root_tokens: int = 8000
    max_input_tokens: int = 24320
    output_tokens: int = 8192
    final_candidates: int = 5
    evaluation_seeds: tuple[int, ...] = (730241,)
    seed: int = 0

    def __post_init__(self):
        for name in ("budget", "roots", "init_attempt_limit", "history_depth", "attempts_shown",
                     "progress_shown", "development_steps", "root_tokens", "max_input_tokens",
                     "output_tokens", "final_candidates"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.roots != 8 or self.init_attempt_limit != 16 or self.history_depth != 8:
            raise ValueError("V10.19 fixes 8 roots, 16 initialization attempts and 8 history steps")
        if self.attempts_shown != 8 or self.progress_shown != 8:
            raise ValueError("V10.19 shows up to 8 attempts and 8 search improvements")
        shares = (self.refine_share, self.explore_share, self.crossover_share)
        if any(not 0 < s < 1 for s in shares) or abs(sum(shares) - 1) > 1e-9:
            raise ValueError("operator shares must be positive and sum to one")
        if self.final_candidates != 5:
            raise ValueError("V10.19 selects from five training finalists")
        if not self.evaluation_seeds or len(set(self.evaluation_seeds)) != len(self.evaluation_seeds):
            raise ValueError("evaluation seeds must be nonempty and unique")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.evaluation_seeds):
            raise ValueError("evaluation seeds must be uint32 integers")
