from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from llm4ad.base import LLM


@dataclass
class BanditUpdate:
    arm: int
    reward: float | None
    baseline: float | None
    shifted_reward: float


class ShinkaLLMBandit:
    """Small reward-only Asymmetric UCB adapter for LLM4AD LLM objects."""

    def __init__(
            self,
            llms: Sequence[LLM],
            *,
            exploration_coef: float = 1.0,
            epsilon: float = 0.2,
            auto_decay: float | None = 0.95,
            seed: int | None = None,
    ):
        if not llms:
            raise ValueError("At least one LLM is required.")
        self.llms = list(llms)
        self.exploration_coef = float(exploration_coef)
        self.epsilon = float(epsilon)
        self.auto_decay = auto_decay
        self.rng = random.Random(seed)
        n = len(self.llms)
        self.n_submitted = np.zeros(n, dtype=float)
        self.n_completed = np.zeros(n, dtype=float)
        self.total_reward = np.zeros(n, dtype=float)
        self.last_probs = np.ones(n, dtype=float) / n
        self.history: list[BanditUpdate] = []

    @property
    def is_fixed(self) -> bool:
        return len(self.llms) == 1

    def select(self) -> tuple[int, LLM, dict[str, Any]]:
        if self.is_fixed:
            self.n_submitted[0] += 1.0
            self.last_probs = np.array([1.0])
            return 0, self.llms[0], {"probabilities": [1.0], "ucb_scores": [0.0]}
        scores = self._ucb_scores()
        probs = self._selection_probabilities(scores)
        arm = self.rng.choices(list(range(len(self.llms))), weights=probs.tolist(), k=1)[0]
        self.n_submitted[arm] += 1.0
        self.last_probs = probs
        return arm, self.llms[arm], {"probabilities": probs.tolist(), "ucb_scores": scores.tolist()}

    def _selection_probabilities(self, scores: np.ndarray) -> np.ndarray:
        n = len(self.llms)
        unseen = np.where((self.n_submitted <= 0.0) & (self.n_completed <= 0.0))[0]
        probs = np.zeros(n, dtype=float)
        if unseen.size > 0:
            probs[unseen] = 1.0 / unseen.size
            return probs
        max_score = float(np.max(scores))
        winners = np.where(scores == max_score)[0]
        if winners.size == n:
            probs[:] = 1.0 / n
        elif winners.size > 0:
            probs[winners] = (1.0 - self.epsilon) / winners.size
            others = [idx for idx in range(n) if idx not in set(winners.tolist())]
            if others:
                probs[others] = self.epsilon / len(others)
        else:
            probs[:] = 1.0 / n
        return probs

    def update(self, arm: int, reward: float | None, baseline: float | None) -> BanditUpdate:
        raw_reward = None if reward is None else float(reward)
        if raw_reward is None:
            shifted = 0.0
        else:
            shifted = raw_reward - float(baseline or 0.0)
            shifted = max(shifted, 0.0)
        self.n_completed[arm] += 1.0
        self.total_reward[arm] += shifted
        update = BanditUpdate(arm=arm, reward=raw_reward, baseline=baseline, shifted_reward=shifted)
        self.history.append(update)
        if self.auto_decay is not None and len(self.llms) > 1:
            self.total_reward *= float(self.auto_decay)
            self.n_completed *= float(self.auto_decay)
        return update

    def _ucb_scores(self) -> np.ndarray:
        n = len(self.llms)
        total = max(float(np.sum(np.maximum(self.n_submitted, self.n_completed))), 1.0)
        scores = np.zeros(n, dtype=float)
        for idx in range(n):
            visits = max(self.n_completed[idx], 1e-9)
            if self.n_submitted[idx] == 0 and self.n_completed[idx] == 0:
                scores[idx] = float("inf")
                continue
            mean = self.total_reward[idx] / visits
            exploration = self.exploration_coef * math.sqrt(math.log(total + 1.0) / visits)
            scores[idx] = mean + exploration
        return scores
