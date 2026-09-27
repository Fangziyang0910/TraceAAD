"""Frozen common-state representatives, quality champions and bounded trials."""

import math


def distance(left, right):
    if not left or len(left) != len(right):
        return None
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
        while len(refs) < self.config.regions and pool:
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
        self.regions = [{"reference": a["profile"], "champion": None, "challenger": None,
                         "stagnation": 0, "trial_used": 0, "checkpoint": None,
                         "tried": []} for a in refs]
        if not self.regions:
            self.regions = [{"reference": [], "champion": None, "challenger": None,
                             "stagnation": 0, "trial_used": 0, "checkpoint": None, "tried": []}]
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
            if region["challenger"] == anchor["id"]:
                region["challenger"] = None
            if champion:
                self._challenge(index, champion)
        else:
            self._challenge(index, anchor)
        challenger = self.anchors.get(region["challenger"])
        if challenger and not self._eligible(index, challenger):
            region["challenger"] = None

    def _eligible(self, index, anchor):
        region = self.regions[index]
        champion = self.anchors[region["champion"]]
        gap = self.config.challenger_gap * max(abs(champion["fitness"]), 1.)
        different = distance(champion["profile"], anchor["profile"])
        return (anchor["artifact_id"] != champion["artifact_id"]
                and anchor["artifact_id"] not in region["tried"]
                and anchor["fitness"] >= champion["fitness"] - gap
                and different is not None and different >= self.config.min_behavior_distance
                and different > 0)

    def _challenge(self, index, anchor):
        if not self._eligible(index, anchor):
            return
        region = self.regions[index]
        current = self.anchors.get(region["challenger"])
        if current is None or (anchor["fitness"], -anchor["id"]) > (current["fitness"], -current["id"]):
            region["challenger"] = anchor["id"]

    def main_weights(self):
        active = [i for i, r in enumerate(self.regions) if r["champion"] is not None]
        scores = [self.anchors[self.regions[i]["champion"]]["fitness"] for i in active]
        weights = []
        for i, score in zip(active, scores):
            # Mid-ranks for ties, independent of node multiplicity.
            rank = 1 + sum(q > score for q in scores) + (sum(q == score for q in scores)-1)/2
            f = self.regions[i]["stagnation"]
            weights.append(rank**-2 * max(.25, (1 + f/8)**-.5))
        return active, weights

    def reactivate(self, anchor, comparison_id):
        """New exact-background evidence permits one new competition, not a ticket."""
        index = self.region_for(anchor)
        if index is None:
            return False
        region = self.regions[index]
        key = anchor["artifact_id"]
        evidence = region.setdefault("reactivation_evidence", {})
        if evidence.get(key) == comparison_id:
            return False
        evidence[key] = comparison_id
        if key in region["tried"]:
            region["tried"].remove(key)
        self._challenge(index, anchor)
        return True

    def main(self, rng):
        ids, weights = self.main_weights()
        return rng.choices(ids, weights=weights)[0]

    def trial(self, rng):
        active = [i for i, r in enumerate(self.regions) if r["champion"] is not None]
        return rng.choices(active, weights=[1/(1+self.regions[i]["trial_used"]) for i in active])[0]
