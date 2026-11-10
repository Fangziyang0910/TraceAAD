"""Base-anchored allocation corrected by recent, quality-matched champion yield."""

import math
import random
from dataclasses import asdict, dataclass

import numpy as np

from traceaad_base import Config
from traceaad_base.selection import SELECTION_RULE, boltzmann_weights, identity

OPERATORS = ("Refine", "Explore", "Crossover")
MOMENTUM_DECAY = 0.5
HISTORY_WINDOW_CAP = 256
SHARED_NEIGHBORS = 32
SHARED_CAP = 16.0
SHRINKAGE = 16.0
MULTIPLIER_CAP = 4.0
UPDATE_RULE = "base_anchored_champion_yield_v3"
TRAJECTORY_RULE = "continuous_origin_momentum_drawdown_v1"
MATCHING_RULE = "quality_state_nearest32_linear_kernel_v1"


@dataclass(frozen=True)
class Action:
    parent_id: int
    operator: str
    reference_id: int | None = None


@dataclass(frozen=True)
class Node:
    id: int
    fitness: float
    operator: str
    parent_id: int | None
    reference_id: int | None
    family: int
    birth_change: float
    momentum: float
    ancestor_best: float


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: int
    action: Action
    parent_fitness: float
    reference_fitness: float | None
    champion_before: float
    evaluations_before: int
    parent_origin: str
    parent_birth_change: float
    parent_momentum: float
    parent_ancestor_best: float
    champion_gain: float
    first_champion_gain: float
    evaluations: int
    model_calls: int
    first_result: dict
    final_result: dict

    @property
    def quality_state(self) -> tuple[float, ...]:
        if self.action.operator == "Crossover":
            return (self.parent_fitness, float(self.reference_fitness), self.champion_before)
        return (self.parent_fitness, self.champion_before)

    @property
    def trajectory(self) -> tuple[str, float, float, float]:
        return (
            self.parent_origin,
            self.parent_birth_change,
            self.parent_momentum,
            self.parent_fitness - self.parent_ancestor_best,
        )


@dataclass(frozen=True)
class Estimate:
    value: float
    eta: float
    shared_weight: float
    local_weight: float
    matching_radius: float
    window_breakthroughs: int


def _sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _squash(value: float, radius: float) -> float:
    if radius == 0.0:
        return _sign(value)
    return value / (radius + abs(value))


def _trajectory_weight(
    candidate: tuple[str, float, float, float],
    record: tuple[str, float, float, float],
    radius: float,
) -> float:
    origin_g, birth_g, momentum_g, drawdown_g = candidate
    origin_i, birth_i, momentum_i, drawdown_i = record
    distance = (
        float(origin_g != origin_i)
        + abs(_squash(birth_g, radius) - _squash(birth_i, radius)) / 2.0
        + abs(momentum_g - momentum_i) / 2.0
        + abs(_squash(drawdown_g, radius) - _squash(drawdown_i, radius))
    ) / 4.0
    return 1.0 / (1.0 + distance)


class Selector:
    """Estimate champion yield from comparable history and correct Base allocation."""

    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.sources: dict[str, int] = {}
        self.history: list[AttemptRecord] = []
        self.processed: set[int] = set()
        self.champion = math.inf
        self.evaluations_used = 0

    @property
    def settings(self) -> dict:
        return {
            "update_rule": UPDATE_RULE,
            "trajectory_rule": TRAJECTORY_RULE,
            "matching_rule": MATCHING_RULE,
            "base_selection_rule": SELECTION_RULE,
            "momentum_decay": MOMENTUM_DECAY,
            "history_window_cap": HISTORY_WINDOW_CAP,
            "shared_neighbors": SHARED_NEIGHBORS,
            "shared_cap": SHARED_CAP,
            "shrinkage": SHRINKAGE,
            "multiplier_cap": MULTIPLIER_CAP,
        }

    def _node(
        self,
        pid: int,
        fitness: float,
        operator: str,
        parent_id: int | None,
        reference_id: int | None,
    ) -> Node:
        if operator == "Init":
            return Node(pid, fitness, operator, None, None, pid, 0.0, 0.0, fitness)
        parent = self.nodes[parent_id]
        birth = parent.fitness - fitness
        momentum = MOMENTUM_DECAY * _sign(birth) + (1.0 - MOMENTUM_DECAY) * parent.momentum
        reference = self.nodes.get(reference_id)
        ancestor_best = min(
            fitness,
            parent.ancestor_best,
            reference.ancestor_best if reference is not None else math.inf,
        )
        return Node(
            pid,
            fitness,
            operator,
            parent_id,
            reference_id,
            parent.family if operator == "Refine" else pid,
            birth,
            momentum,
            ancestor_best,
        )

    @staticmethod
    def actions_for(parent: Node, reference: Node | None) -> list[Action]:
        actions = [Action(parent.id, operator) for operator in OPERATORS[:2]]
        if reference is not None:
            actions.append(Action(parent.id, "Crossover", reference.id))
        return actions

    def actions(self, reference_ids: dict[int, int | None]) -> list[Action]:
        return [
            action
            for node in sorted(self.nodes.values(), key=lambda node: node.id)
            for action in self.actions_for(node, self.nodes.get(reference_ids[node.id]))
        ]

    def quality_state(self, action: Action) -> tuple[float, ...]:
        parent = self.nodes[action.parent_id]
        if action.operator == "Crossover":
            return (parent.fitness, self.nodes[action.reference_id].fitness, self.champion)
        return (parent.fitness, self.champion)

    @staticmethod
    def _trajectory_of(node: Node) -> tuple[str, float, float, float]:
        return (node.operator, node.birth_change, node.momentum, node.fitness - node.ancestor_best)

    def _estimates(
        self, actions: list[Action], *, evaluation_budget: int, first_only: bool
    ) -> list[Estimate]:
        window = max(1, min(int(evaluation_budget), HISTORY_WINDOW_CAP))
        results: list[Estimate | None] = [None] * len(actions)
        by_operator: dict[str, list[int]] = {operator: [] for operator in OPERATORS}
        for index, action in enumerate(actions):
            by_operator[action.operator].append(index)

        for operator, indices in by_operator.items():
            if not indices:
                continue
            pool = [
                record
                for record in self.history
                if record.action.operator == operator
                and record.evaluations >= 1
                and (self.evaluations_used - record.evaluations_before) < window
            ]
            breakthroughs = sum(
                1
                for record in pool
                if (record.first_champion_gain if first_only else record.champion_gain) > 0.0
            )
            if not pool:
                for index in indices:
                    results[index] = Estimate(0.0, 0.0, 0.0, 0.0, 0.0, 0)
                continue

            history_x = np.array([record.quality_state for record in pool], dtype=np.float64)
            yields = np.array(
                [
                    (record.first_champion_gain if first_only else record.champion_gain)
                    for record in pool
                ],
                dtype=np.float64,
            )
            costs = np.array(
                [1.0 if first_only else float(record.evaluations) for record in pool],
                dtype=np.float64,
            )
            pairwise = (
                np.max(np.abs(history_x[:, None, :] - history_x[None, :, :]), axis=2)
                if len(pool) > 1
                else np.zeros((len(pool), len(pool)))
            )
            local_by_action: dict[Action, list[int]] = {}
            for position, record in enumerate(pool):
                local_by_action.setdefault(record.action, []).append(position)
            candidate_x = np.array(
                [self.quality_state(actions[index]) for index in indices], dtype=np.float64
            )
            distances = np.max(np.abs(candidate_x[:, None, :] - history_x[None, :, :]), axis=2)
            radius_cache: dict[tuple[int, ...], float] = {}

            for row, index in enumerate(indices):
                action = actions[index]
                order = np.argsort(distances[row], kind="stable")
                local_idx = local_by_action.get(action, [])
                local_set = set(local_idx)
                shared_idx = [int(i) for i in order if int(i) not in local_set][:SHARED_NEIGHBORS]
                basis = (
                    shared_idx
                    if shared_idx
                    else [int(i) for i in order if int(i) in local_set][:SHARED_NEIGHBORS]
                )
                if not basis and not local_idx:
                    results[index] = Estimate(0.0, 0.0, 0.0, 0.0, 0.0, breakthroughs)
                    continue
                key = tuple(basis)
                if key not in radius_cache:
                    if len(basis) < 2:
                        radius_cache[key] = 0.0
                    else:
                        sub = pairwise[np.ix_(basis, basis)]
                        upper = sub[np.triu_indices(len(basis), k=1)]
                        positive = upper[upper > 0.0]
                        radius_cache[key] = (
                            2.0 * float(np.median(positive)) if len(positive) else 0.0
                        )
                radius = radius_cache[key]
                candidate_traj = self._trajectory_of(self.nodes[action.parent_id])

                n_shared = y_shared = c_shared = 0.0
                for position in shared_idx:
                    if radius > 0.0:
                        quality = max(1.0 - float(distances[row, position]) / radius, 0.0)
                    else:
                        quality = float(distances[row, position] == 0.0)
                    if quality == 0.0:
                        continue
                    record = pool[position]
                    recency = max(
                        1.0 - (self.evaluations_used - record.evaluations_before) / window, 0.0
                    )
                    w = (
                        quality
                        * _trajectory_weight(candidate_traj, record.trajectory, radius)
                        * recency
                    )
                    n_shared += w
                    y_shared += w * float(yields[position])
                    c_shared += w * float(costs[position])
                n_local = y_local = c_local = 0.0
                for position in local_idx:
                    record = pool[position]
                    recency = max(
                        1.0 - (self.evaluations_used - record.evaluations_before) / window, 0.0
                    )
                    w = _trajectory_weight(candidate_traj, record.trajectory, radius) * recency
                    n_local += w
                    y_local += w * float(yields[position])
                    c_local += w * float(costs[position])
                share_scale = min(1.0, SHARED_CAP / n_shared) if n_shared > 0.0 else 0.0
                denominator = c_local + share_scale * c_shared
                value = (
                    (y_local + share_scale * y_shared) / denominator if denominator > 0.0 else 0.0
                )
                effective = n_local + min(n_shared, SHARED_CAP)
                eta = effective / (SHRINKAGE + effective)
                results[index] = Estimate(
                    float(value),
                    float(eta),
                    float(n_shared),
                    float(n_local),
                    float(radius),
                    int(breakthroughs),
                )
        return [
            result if result is not None else Estimate(0.0, 0.0, 0.0, 0.0, 0.0, 0)
            for result in results
        ]

    def estimate_action(
        self, action: Action, config: Config | None = None, *, first_only: bool = False
    ) -> Estimate:
        budget = (config or Config()).evaluation_budget
        return self._estimates([action], evaluation_budget=budget, first_only=first_only)[0]

    def allocation(self, config: Config | None = None, *, first_only: bool = False,
                   reference_ids: dict[int, int | None]):
        """Return actions, Base probabilities, estimates, final probabilities, diagnostics."""
        config = config or Config()
        if not self.nodes:
            raise ValueError("No valid parent candidates")
        if config.crossover_weight > 0:
            # Initialization leaves distinct valid programs, so every parent has a reference in the quality pool.
            missing = sorted(pid for pid in self.nodes if reference_ids.get(pid) is None)
            if missing:
                raise RuntimeError(f"No reference for Crossover from Algorithms {missing}")
        actions = self.actions(reference_ids)
        quality = boltzmann_weights(
            {node.id: node.fitness for node in self.nodes.values()}, config.selection_temperature
        )
        operator_weights = {
            "Refine": config.refine_weight,
            "Explore": config.explore_weight,
            "Crossover": config.crossover_weight,
        }
        base_weights = [
            quality[action.parent_id] * operator_weights[action.operator]
            for action in actions
        ]
        base_sum = sum(base_weights)
        beta = (
            [weight / base_sum for weight in base_weights]
            if base_sum > 0.0
            else [1.0 / len(actions)] * len(actions)
        )

        estimates = self._estimates(
            actions, evaluation_budget=config.evaluation_budget, first_only=first_only
        )
        m = sum(b * estimate.eta for b, estimate in zip(beta, estimates))
        z = sum(b * estimate.eta * estimate.value for b, estimate in zip(beta, estimates))
        mean_value = z / m if m > 0.0 else 0.0
        multipliers: list[float] = []
        uncapped: list[float] = []
        if z > 0.0 and mean_value > 0.0:
            for estimate in estimates:
                ratio = estimate.value / mean_value
                uncapped.append(float(ratio))
                multipliers.append(float(min(ratio, MULTIPLIER_CAP)))
            unnormalized = [
                b * ((1.0 - estimate.eta) + estimate.eta * multiplier)
                for b, estimate, multiplier in zip(beta, estimates, multipliers)
            ]
            total = sum(unnormalized)
            probs = [value / total for value in unnormalized] if total > 0.0 else list(beta)
        else:
            multipliers = [0.0] * len(actions)
            uncapped = [0.0] * len(actions)
            probs = list(beta)
        diagnostics = {
            "m": float(m),
            "z": float(z),
            "mean_value": float(mean_value),
            "multipliers": multipliers,
            "uncapped_multipliers": uncapped,
        }
        return actions, beta, estimates, probs, diagnostics

    def choose(
        self, remaining_evaluations: int, rng: random.Random, config: Config | None = None,
        *, reference_ids: dict[int, int | None],
    ) -> tuple[Action, dict]:
        if remaining_evaluations < 1:
            raise ValueError("No evaluation budget remains")
        config = config or Config()
        first_only = remaining_evaluations == 1
        actions, beta, estimates, probs, diagnostics = self.allocation(
            config, first_only=first_only, reference_ids=reference_ids
        )
        index = rng.choices(range(len(actions)), weights=probs, k=1)[0]
        chosen = actions[index]
        estimate = estimates[index]
        parent = self.nodes[chosen.parent_id]
        record = {
            "parent_id": chosen.parent_id,
            "operator": chosen.operator,
            "reference_id": chosen.reference_id,
            "action_group": asdict(chosen),
            "parent_fitness": parent.fitness,
            "reference_fitness": (
                self.nodes[chosen.reference_id].fitness if chosen.reference_id is not None else None
            ),
            "champion_before_attempt": self.champion,
            "evaluations_used_before_attempt": self.evaluations_used,
            "parent_origin": parent.operator,
            "parent_birth_change": parent.birth_change,
            "parent_momentum": parent.momentum,
            "parent_ancestor_best": parent.ancestor_best,
            "matching_radius": estimate.matching_radius,
            "shared_weight": estimate.shared_weight,
            "local_weight": estimate.local_weight,
            "value": estimate.value,
            "eta": estimate.eta,
            "base_probability": beta[index],
            "selection_probability": probs[index],
            "probability": probs[index],
            "mean_value": diagnostics["mean_value"],
            "multiplier": diagnostics["multipliers"][index],
            "uncapped_multiplier": diagnostics["uncapped_multipliers"][index],
            "window_breakthroughs": estimate.window_breakthroughs,
            "remaining_evaluations": remaining_evaluations,
            "valuation_view": "first_result" if first_only else "full_attempt",
        }
        return chosen, record

    @staticmethod
    def _candidate_result(
        status: str | None, fitness: float | None, code: str | None, sources: dict[str, int]
    ) -> dict:
        valid = (
            status == "success"
            and fitness is not None
            and math.isfinite(fitness)
            and code is not None
        )
        source = identity(code) if valid else None
        return {
            "status": status,
            "fitness": float(fitness) if (fitness is not None and math.isfinite(fitness)) else None,
            "source_identity": source,
            "usable_at_attempt_start": bool(valid and source not in sources),
        }

    def observe(
        self,
        attempt_id: int,
        record: dict | None,
        program,
        *,
        evaluations: int = 0,
        model_calls: int = 0,
        first_result: dict | None = None,
        final_result: dict | None = None,
    ) -> None:
        """Observe one finished attempt; Init builds the pool without operator history."""
        if attempt_id in self.processed:
            return
        evaluations = int(evaluations)
        computed_final = self._candidate_result(
            program.evaluation.status
            if (program is not None and program.evaluation is not None)
            else None,
            program.evaluation.fitness
            if (program is not None and program.evaluation is not None)
            else None,
            program.code if program is not None else None,
            self.sources,
        )
        if final_result is None:
            final_result = computed_final
        elif final_result != computed_final:
            raise ValueError(f"Incompatible V22.1 final_result at attempt {attempt_id}")

        if record is not None:
            action = Action(record["parent_id"], record["operator"], record["reference_id"])
            parent = self.nodes.get(action.parent_id)
            if parent is None:
                raise ValueError(f"Incompatible V22.1 parent at attempt {attempt_id}")

            def close(name: str, expected: float) -> None:
                if not math.isclose(
                    float(record[name]), float(expected), rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"Incompatible V22.1 {name} at attempt {attempt_id}")

            close("parent_fitness", parent.fitness)
            expected_reference = (
                self.nodes[action.reference_id].fitness if action.reference_id is not None else None
            )
            if record.get("reference_fitness") != expected_reference and not (
                expected_reference is not None
                and record.get("reference_fitness") is not None
                and math.isclose(
                    float(record["reference_fitness"]),
                    float(expected_reference),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"Incompatible V22.1 reference_fitness at attempt {attempt_id}")
            close("champion_before_attempt", self.champion)
            if int(record["evaluations_used_before_attempt"]) != self.evaluations_used:
                raise ValueError(
                    f"Incompatible V22.1 evaluations_used_before_attempt at attempt {attempt_id}"
                )
            if record["parent_origin"] != parent.operator:
                raise ValueError(f"Incompatible V22.1 parent_origin at attempt {attempt_id}")
            close("parent_birth_change", parent.birth_change)
            close("parent_momentum", parent.momentum)
            close("parent_ancestor_best", parent.ancestor_best)
            champion_before = float(record["champion_before_attempt"])

            if first_result is None:
                if evaluations == 0:
                    first_result = {
                        "status": None,
                        "fitness": None,
                        "source_identity": None,
                        "usable_at_attempt_start": False,
                    }
                elif evaluations == 1:
                    first_result = dict(final_result)
                else:
                    raise ValueError(
                        f"Multi-evaluation attempt {attempt_id} requires recorded first_result"
                    )

            def usable(result: dict) -> bool:
                return bool(
                    result.get("usable_at_attempt_start")
                    and result.get("status") == "success"
                    and result.get("fitness") is not None
                    and math.isfinite(result["fitness"])
                )

            first_gain = (
                max(champion_before - float(first_result["fitness"]), 0.0)
                if usable(first_result)
                else 0.0
            )
            final_gain = (
                max(champion_before - float(final_result["fitness"]), 0.0)
                if usable(final_result)
                else 0.0
            )
            for name, value in (("champion_gain", final_gain), ("first_champion_gain", first_gain)):
                if name in record and not math.isclose(
                    float(record[name]), float(value), rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"Incompatible V22.1 {name} at attempt {attempt_id}")
            record.update(
                champion_gain=float(final_gain),
                first_champion_gain=float(first_gain),
                evaluations=evaluations,
                model_calls=int(model_calls),
                first_result=first_result,
                final_result=final_result,
                usable=bool(usable(final_result)),
            )
            self.history.append(
                AttemptRecord(
                    attempt_id=int(attempt_id),
                    action=action,
                    parent_fitness=float(record["parent_fitness"]),
                    reference_fitness=(
                        None
                        if record.get("reference_fitness") is None
                        else float(record["reference_fitness"])
                    ),
                    champion_before=champion_before,
                    evaluations_before=int(record["evaluations_used_before_attempt"]),
                    parent_origin=str(record["parent_origin"]),
                    parent_birth_change=float(record["parent_birth_change"]),
                    parent_momentum=float(record["parent_momentum"]),
                    parent_ancestor_best=float(record["parent_ancestor_best"]),
                    champion_gain=float(final_gain),
                    first_champion_gain=float(first_gain),
                    evaluations=evaluations,
                    model_calls=int(model_calls),
                    first_result=first_result,
                    final_result=final_result,
                )
            )

        if final_result["status"] == "success" and final_result["fitness"] is not None:
            self.champion = min(self.champion, float(final_result["fitness"]))
        if final_result["usable_at_attempt_start"]:
            node = self._node(
                program.id,
                float(final_result["fitness"]),
                program.operator,
                program.parent_id,
                program.reference_id,
            )
            self.nodes[node.id] = node
            self.sources[final_result["source_identity"]] = node.id
        self.evaluations_used += evaluations
        self.processed.add(attempt_id)
