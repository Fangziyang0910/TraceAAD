"""The continuation tree is distinct from the ancestry of generated programs."""

from dataclasses import dataclass, field


@dataclass
class DevelopmentState:
    id: int
    task_id: int
    program_id: int
    address: str = ""
    retry_from: int | None = None
    attempt_count: int = 0
    result_id: int | None = None
    preferred: int | None = None
    alternative: int | None = None

    @property
    def expanded(self):
        return self.result_id is not None


@dataclass
class DevelopmentTask:
    id: int
    root_program_id: int
    root_state_id: int
    best_program_id: int
    best_curve: list[float]
    age: int = 0
    depth_limit: int = 2
    discrepancy_round: int = 0
    stack: list = field(default_factory=list)
    status: str = "YOUNG"
    comparison: dict = field(default_factory=dict)


def next_state(task, states, config):
    """Resume the exact-discrepancy DFS; returning a state does not expand it."""
    while True:
        while task.stack:
            state_id, remaining = task.stack.pop()
            state = states[state_id]
            depth = len(state.address)
            if remaining < 0 or remaining > task.depth_limit - depth - 1:
                continue
            if not state.expanded:
                task.stack.append((state_id, remaining))
                return state
            if depth == task.depth_limit - 1:
                continue
            successors = []
            if state.preferred is not None:
                successors.append((state.preferred, remaining))
            if remaining > 0 and state.alternative is not None:
                successors.append((state.alternative, remaining - 1))
            # LIFO: alternatives first puts the discrepancy closest to the root.
            task.stack.extend(successors if config.early_first else reversed(successors))
        limit = min(task.depth_limit - 1, task.depth_limit.bit_length() - 1) if config.lds else 0
        task.discrepancy_round += 1
        if task.discrepancy_round > limit:
            task.depth_limit += config.depth_increment
            task.discrepancy_round = 0
        task.stack.append((task.root_state_id, task.discrepancy_round))
