"""Trajectory-conditioned responses and two-attempt allocation; no model calls here."""

from collections import ChainMap
from dataclasses import dataclass, field, replace
import math

import numpy as np
from scipy.special import gammaln, stdtr, stdtrit

from traceaad_base.selection import identity


OPERATORS = ("Refine", "Explore", "Crossover")
DIMENSION = 9
SHORTLIST = 8
_points, _weights = np.polynomial.legendre.leggauss(16)
QUANTILES, QUADRATURE_WEIGHTS = (_points + 1.) / 2., _weights / 2.


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
    ancestor_momentum: float
    momentum: float
    ancestor_best: float


@dataclass
class Group:
    n: int = 0
    u: int = 0
    k: int = 0
    sx: np.ndarray = field(default_factory=lambda: np.zeros(DIMENSION, dtype=np.float64))
    sy: float = 0.


_EMPTY = Group()


class ResponseModel:
    """One operator's smoothed gate and conjugate regression with group intercepts."""

    def __init__(self):
        self.N = self.U = 0
        self.m = np.zeros(DIMENSION, dtype=np.float64)
        self.V = np.diag([1.] + [.25] * (DIMENSION - 1))
        self.A, self.B = 3., 2.
        self.groups = {}

    def fork(self):
        """Share untouched groups; an observation always replaces its own group."""
        model = ResponseModel.__new__(ResponseModel)
        model.N, model.U = self.N, self.U
        model.m, model.V = self.m.copy(), self.V.copy()
        model.A, model.B = self.A, self.B
        model.groups = ChainMap({}, self.groups)
        return model

    def predict(self, actions, x):
        groups = [self.groups.get(action, _EMPTY) for action in actions]
        n = np.array([g.n for g in groups], dtype=np.float64)
        u = np.array([g.u for g in groups], dtype=np.float64)
        other = (1. + self.U - u) / (2. + self.N - n)
        pi = (4. * other + u) / (4. + n)
        c = 1. / (1. + np.array([g.k for g in groups], dtype=np.float64))
        z = x - c[:, None] * np.array([g.sx for g in groups])
        mu = z @ self.m + c * np.array([g.sy for g in groups])
        H = 1. + c + np.einsum("ij,jk,ik->i", z, self.V, z)
        return pi, mu, np.sqrt((self.B / self.A) * H), 2. * self.A

    def observe_gate(self, action, usable):
        group = self.groups.get(action, _EMPTY)
        self.N += 1
        self.U += int(usable)
        self.groups[action] = replace(group, n=group.n + 1, u=group.u + int(usable))

    def observe_usable(self, action, x, y):
        group = self.groups.get(action, _EMPTY)
        c = 1. / (group.k + 1.)
        z = np.asarray(x, dtype=np.float64) - c * group.sx
        mu = z @ self.m + c * group.sy
        H = 1. + c + z @ self.V @ z
        e, v = y - mu, self.V @ z
        self.m = self.m + v * e / H
        self.V = self.V - np.outer(v, v) / H
        self.V = (self.V + self.V.T) / 2.
        self.A += .5
        self.B += e * e / (2. * H)
        self.groups[action] = replace(group, k=group.k + 1, sx=group.sx + x, sy=group.sy + y)


def squash(z):
    return z / (1. + abs(z))


def expected_improvement(pi, mu, scale, df, gap):
    d = (mu - gap) / scale
    density = np.exp(gammaln((df + 1.) / 2.) - gammaln(df / 2.)
                     - .5 * np.log(df * np.pi) - (df + 1.) / 2. * np.log1p(d * d / df))
    return pi * np.maximum(0., (mu - gap) * stdtr(df, d) + scale * (df + d * d) / (df - 1.) * density)


class Selector:
    def __init__(self):
        self.nodes = {}
        self.sources = {}
        self.models = {operator: ResponseModel() for operator in OPERATORS}
        self.processed = set()
        self.f0 = self.s0 = None
        self.champion = math.inf
        self.max_id = 0
        self.pending = []

    def _node(self, pid, fitness, operator, parent_id, reference_id):
        if operator == "Init":
            return Node(pid, fitness, operator, None, None, pid, 0., 0., 0., fitness)
        parent = self.nodes[parent_id]
        delta = parent.fitness - fitness
        ancestor = parent.momentum
        momentum = .5 * squash(delta / self.s0) + .5 * ancestor if self.s0 is not None else 0.
        reference = self.nodes.get(reference_id)
        threshold = min(fitness, parent.ancestor_best,
                        reference.ancestor_best if reference is not None else math.inf)
        return Node(pid, fitness, operator, parent_id, reference_id,
                    parent.family if operator == "Refine" else pid, delta, ancestor, momentum, threshold)

    def initialize(self):
        """Freeze the initial champion once initialization is complete."""
        if self.f0 is None and self.nodes:
            self.f0 = min(node.fitness for node in self.nodes.values())
            self._freeze_scale()

    def _freeze_scale(self):
        if self.s0 is not None or self.f0 is None:
            return
        fitness = [node.fitness for node in self.nodes.values()]
        differences = [abs(a - b) for i, a in enumerate(fitness) for b in fitness[i + 1:] if a != b]
        if not differences:
            return
        self.s0 = float(np.median(differences))
        # Before a scale exists only raw transitions are retained. Rebuild their
        # deterministic ancestry features, then replay regression, never the gate.
        for pid, node in list(self.nodes.items()):
            self.nodes[pid] = self._node(pid, node.fitness, node.operator, node.parent_id, node.reference_id)
        pending, self.pending = self.pending, []
        for action, record, fitness in pending:
            x = self.features(action, self.nodes)
            y = (record["parent_fitness"] - fitness) / self.s0
            self.models[action.operator].observe_usable(action, x, y)

    def features(self, action, nodes):
        p = nodes[action.parent_id]
        return np.array([
            1., squash((self.f0 - p.fitness) / self.s0), squash(p.birth_change / self.s0),
            p.ancestor_momentum, squash((p.fitness - p.ancestor_best) / self.s0),
            *(float(p.operator == operator) for operator in OPERATORS),
            squash((p.fitness - nodes[action.reference_id].fitness) / self.s0)
            if action.operator == "Crossover" else 0.,
        ], dtype=np.float64)

    def family_champions(self):
        champions = {}
        for node in self.nodes.values():
            old = champions.get(node.family)
            if old is None or (node.fitness, node.id) < (old.fitness, old.id):
                champions[node.family] = node
        # Every parent can use the first or second distinct family champion.
        return sorted(champions.values(), key=lambda node: (node.fitness, node.id))[:2]

    @staticmethod
    def actions_for(parent, references):
        actions = [Action(parent.id, operator) for operator in OPERATORS[:2]]
        reference = min((node for node in references if node.family != parent.family),
                        key=lambda node: (node.fitness, node.id), default=None)
        if reference is not None:
            actions.append(Action(parent.id, "Crossover", reference.id))
        return actions

    def forecasts(self, actions, nodes, models, champion):
        result = np.empty((len(actions), 5), dtype=np.float64)
        for operator in OPERATORS:
            indices = [i for i, action in enumerate(actions) if action.operator == operator]
            if not indices:
                continue
            selected = [actions[i] for i in indices]
            x = np.array([self.features(action, nodes) for action in selected])
            pi, mu, scale, df = models[operator].predict(selected, x)
            gap = np.array([(nodes[action.parent_id].fitness - champion) / self.s0 for action in selected])
            result[indices] = np.column_stack((pi, mu, scale, np.full(len(indices), df),
                                               expected_improvement(pi, mu, scale, df, gap)))
        return result

    def second_actions(self, base, action, references, child=None):
        nodes = self.nodes if child is None else ChainMap({child.id: child}, self.nodes)
        references = references if child is None else [*references, child]
        actions = []
        for original in dict.fromkeys([*base, action]):
            if original.operator == "Crossover":
                candidates = self.actions_for(nodes[original.parent_id], references)
                actions.extend(a for a in candidates if a.operator == "Crossover")
            else:
                actions.append(original)
        if child is not None:
            actions.extend(self.actions_for(child, references))
        return list(dict.fromkeys(actions)), nodes

    def two_step(self, action, prediction, shortlist, references):
        pi, mu, scale, df, immediate = prediction
        x = self.features(action, self.nodes)
        shadow = self.models[action.operator].fork()
        shadow.observe_gate(action, False)
        models = {**self.models, action.operator: shadow}
        actions, nodes = self.second_actions(shortlist, action, references)
        j0 = self.forecasts(actions, nodes, models, self.champion)[:, 4].max()
        following = 0.
        for y, weight in zip(mu + scale * stdtrit(df, QUANTILES), QUADRATURE_WEIGHTS):
            shadow = self.models[action.operator].fork()
            shadow.observe_gate(action, True)
            shadow.observe_usable(action, x, y)
            models = {**self.models, action.operator: shadow}
            fitness = self.nodes[action.parent_id].fitness - self.s0 * y
            child = self._node(self.max_id + 1, fitness, action.operator, action.parent_id, action.reference_id)
            actions, nodes = self.second_actions(shortlist, action, references, child)
            j1 = self.forecasts(actions, nodes, models, min(self.champion, fitness))[:, 4].max()
            following += weight * j1
        return float(immediate + (1. - pi) * j0 + pi * following)

    def choose(self, remaining_evaluations, rng):
        if remaining_evaluations < 1:
            raise ValueError("No evaluation budget remains")
        self.initialize()
        if not self.nodes:
            raise ValueError("No valid parent candidates")
        references = self.family_champions()
        actions = [action for node in sorted(self.nodes.values(), key=lambda node: node.id)
                   for action in self.actions_for(node, references)]
        if self.s0 is None:
            chosen = rng.choice(actions)
            prediction, value, x = [None] * 5, None, None
        else:
            forecasts = self.forecasts(actions, self.nodes, self.models, self.champion)
            values = forecasts[:, 4].copy()
            if remaining_evaluations >= 4:
                shortlist = []
                for operator in OPERATORS:
                    indices = [i for i, action in enumerate(actions) if action.operator == operator]
                    indices.sort(key=lambda i: (-forecasts[i, 4], actions[i].parent_id,
                                                actions[i].reference_id or -1))
                    shortlist.extend(actions[i] for i in indices[:SHORTLIST])
                values = np.array([self.two_step(action, prediction, shortlist, references)
                                   for action, prediction in zip(actions, forecasts)])
            index = rng.choice(np.flatnonzero(values == values.max()).tolist())
            chosen, prediction, value = actions[index], forecasts[index], float(values[index])
            x = self.features(chosen, self.nodes).tolist()
        record = {"parent_id": chosen.parent_id, "operator": chosen.operator,
                  "reference_id": chosen.reference_id, "x": x,
                  "parent_fitness": self.nodes[chosen.parent_id].fitness, "f0": self.f0, "s0": self.s0,
                  "champion": self.champion, "I": prediction[4], "Q": value, "pi": prediction[0],
                  "remaining_evaluations": remaining_evaluations}
        return chosen, record

    def observe(self, attempt_id, record, program):
        """Observe one final attempt result; initialization has no selection record."""
        if attempt_id in self.processed:
            return
        if program is not None:
            self.max_id = max(self.max_id, program.id)
        valid = (program is not None and program.valid and program.evaluation.fitness is not None
                 and math.isfinite(program.evaluation.fitness))
        if valid:
            self.champion = min(self.champion, program.evaluation.fitness)
        source = identity(program.code) if valid else None
        usable = valid and source not in self.sources
        if record is not None:
            action = Action(record["parent_id"], record["operator"], record["reference_id"])
            model = self.models[action.operator]
            model.observe_gate(action, usable)
            if usable:
                if self.s0 is None:
                    self.pending.append((action, record, program.evaluation.fitness))
                else:
                    y = (record["parent_fitness"] - program.evaluation.fitness) / self.s0
                    model.observe_usable(action, np.asarray(record["x"], dtype=np.float64), y)
            record["usable"] = bool(usable)
        if usable:
            node = self._node(program.id, program.evaluation.fitness, program.operator,
                              program.parent_id, program.reference_id)
            self.nodes[node.id] = node
            self.sources[source] = node.id
            self._freeze_scale()
        self.processed.add(attempt_id)
