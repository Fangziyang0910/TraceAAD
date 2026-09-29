"""Behavior regions for independent discovery and regional parent selection."""

import math


def distance(left, right):
    if not left or len(left) != len(right):
        return None
    # Discrete choices use disagreement; ACO probabilities need a continuous distance.
    if any(isinstance(x, float) or isinstance(y, float) for x, y in zip(left, right)):
        return sum(abs(float(x)-float(y)) for x, y in zip(left, right)) / len(left)
    return sum(a != b for a, b in zip(left, right)) / len(left)


class Frontier:
    def __init__(self, config, anchors, state=None):
        self.config, self.anchors = config, anchors
        self.regions = state or []

    def freeze(self):
        unique = {}
        for anchor in self.anchors.values():
            if anchor["profile"]:
                unique.setdefault(anchor["artifact_id"], anchor)
        pool = sorted(unique.values(), key=lambda a: (-a["fitness"], a["id"]))
        refs = pool[:1]
        while len(refs) < min(self.config.initial_regions, self.config.regions) and pool:
            remaining = [a for a in pool if a not in refs]
            if not remaining:
                break
            candidate = max(remaining, key=lambda a: (
                min(distance(a["profile"], r["profile"]) for r in refs), a["fitness"], -a["id"]))
            if min(distance(candidate["profile"], r["profile"]) for r in refs) < self.config.min_behavior_distance:
                break
            if any(candidate["profile"] == r["profile"] for r in refs):
                break
            refs.append(candidate)
        self.regions = [{"reference": a["profile"], "champion": None,
                         "stagnation": 0, "checkpoint": None} for a in refs]
        if not self.regions:
            self.regions = [{"reference": [], "champion": None,
                             "stagnation": 0, "checkpoint": None}]
        for anchor in self.anchors.values():
            self.admit(anchor)

    def region_for(self, anchor):
        if not self.regions:
            return None
        if not anchor["profile"]:
            # Missing descriptor is not evidence of a new behavior. It stays
            # in the source region, or the first quality region for a root.
            return anchor.get("origin_region") or 0
        scores = [distance(anchor["profile"], r["reference"]) for r in self.regions]
        return min(range(len(scores)), key=lambda i: scores[i] if scores[i] is not None else math.inf)

    def admit_region(self, anchor):
        if not anchor["profile"] or len(self.regions) >= self.config.regions:
            return False
        distances = [distance(anchor["profile"], region["reference"]) for region in self.regions]
        if any(value is None for value in distances) or min(distances) < self.config.min_behavior_distance:
            return False
        self.regions.append({"reference": anchor["profile"], "champion": None,
                             "stagnation": 0, "checkpoint": None})
        self.admit(anchor)
        return True

    def admit(self, anchor):
        index = self.region_for(anchor)
        if index is None:
            return
        region = self.regions[index]
        champion = self.anchors.get(region["champion"])
        if champion is None or anchor["fitness"] > champion["fitness"]:
            region["champion"] = anchor["id"]
            if region["checkpoint"] is None or anchor["fitness"] >= region["checkpoint"] + self.config.delta:
                region["checkpoint"], region["stagnation"] = anchor["fitness"], 0

    def main_weights(self):
        active = [i for i, r in enumerate(self.regions) if r["champion"] is not None]
        scores = [self.anchors[self.regions[i]["champion"]]["fitness"] for i in active]
        weights = []
        for i, score in zip(active, scores):
            # Mid-ranks for ties, independent of node multiplicity.
            rank = 1 + sum(q > score for q in scores) + (sum(q == score for q in scores)-1)/2
            f = self.regions[i]["stagnation"]
            weights.append(rank**-1 * (1 + min(f, 24)/24) / math.sqrt(1 + self.regions[i].get("main_used", 0)))
        return active, weights

    def main(self, rng):
        ids, weights = self.main_weights()
        return rng.choices(ids, weights=weights)[0]
