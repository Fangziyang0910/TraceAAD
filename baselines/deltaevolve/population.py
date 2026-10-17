"""Bounded island MAP-Elites populations and a global fitness archive."""

import math

import numpy as np


class Population:
    def __init__(self, rng, *, num_islands=3, population_size=40, archive_size=20,
                 migration_interval=10, migration_rate=0.1):
        if min(num_islands, population_size, archive_size, migration_interval) < 1:
            raise ValueError("Population sizes and migration interval must be positive")
        if not 0 <= migration_rate <= 1:
            raise ValueError("migration_rate must be in [0, 1]")
        self.rng = rng
        self.population_size, self.archive_size = population_size, archive_size
        self.migration_interval, self.migration_rate = migration_interval, migration_rate
        self.nodes = {}
        self.islands = [{} for _ in range(num_islands)]
        self.archive_ids = []
        self.projection = None

    def cell(self, embedding):
        vector = np.asarray(embedding, dtype=float)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("Code embedding must be a finite nonempty vector")
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Code embedding must have nonzero finite norm")
        if self.projection is None:
            self.projection = np.random.default_rng(0).normal(size=(2, vector.size))
            self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)
        if vector.size != self.projection.shape[1]:
            raise ValueError("Code embedding dimension changed within the run")
        descriptors = np.tanh(self.projection @ (vector / norm) * math.sqrt(vector.size))
        return tuple(np.minimum(9, ((descriptors + 1) * 5).astype(int)).tolist())

    def key(self, identity):
        node = self.nodes[identity]
        return node["fitness"], identity

    def _insert(self, identity, island):
        cells = self.islands[island]
        cell = self.nodes[identity]["cell"]
        old = cells.get(cell)
        if old is None or self.key(identity) < self.key(old):
            cells[cell] = identity
        if len(cells) > self.population_size:
            worst = max(cells, key=lambda c: self.key(cells[c]))
            del cells[worst]

    def add(self, node, island):
        if node["id"] in self.nodes:
            raise ValueError("Duplicate DeltaEvolve node ID")
        if not math.isfinite(node["fitness"]):
            raise ValueError("Only valid finite-score nodes enter the population")
        self.nodes[node["id"]] = node
        self._insert(node["id"], island)
        self.archive_ids = sorted(self.archive_ids + [node["id"]], key=self.key)[:self.archive_size]

    def select(self, top_k=3, diverse_m=2):
        island = self.rng.randrange(len(self.islands))
        candidates = sorted(self.islands[island].values(), key=self.key)
        draw = self.rng.random()
        if draw < 0.1:
            elites = self.archive_ids[:max(1, math.ceil(len(self.archive_ids) * 0.1))]
            parent_id = self.rng.choice(elites)
        elif draw < 0.3:
            parent_id = self.rng.choice(candidates)
        else:
            parent_id = self.rng.choices(candidates, weights=[1 / (i + 1) for i in range(len(candidates))])[0]
        top = [i for i in self.archive_ids if i != parent_id and self.nodes[i]["parent_id"] is not None][:top_k]
        # Merge occupied cells across islands; each cell contributes its best program.
        cells = {}
        for members in self.islands:
            for cell, identity in members.items():
                if cell not in cells or self.key(identity) < self.key(cells[cell]):
                    cells[cell] = identity
        origin = self.nodes[parent_id]["cell"]
        distant = sorted(cells, key=lambda c: (-sum((a-b)**2 for a, b in zip(c, origin)), self.key(cells[c])))
        diverse = [cells[c] for c in distant if cells[c] not in {parent_id, *top}
                   and self.nodes[cells[c]]["parent_id"] is not None][:diverse_m]
        return island, self.nodes[parent_id], [self.nodes[i] for i in top], [self.nodes[i] for i in diverse]

    def migrate(self, iteration):
        if iteration % self.migration_interval or self.migration_rate == 0:
            return []
        outgoing = [sorted(set(cells.values()), key=self.key)[:math.ceil(len(cells) * self.migration_rate)]
                    for cells in self.islands]
        transfers = []
        for source, identities in enumerate(outgoing):
            target = (source + 1) % len(self.islands)
            for identity in identities:
                self._insert(identity, target)
                transfers.append({"node_id": identity, "source_island": source, "target_island": target,
                                  "retained": identity in self.islands[target].values()})
        return transfers
