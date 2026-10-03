"""Scale-free quality/count competition with shared syntax opportunity accounts.

Syntax identity removes only comments and formatting, retaining literals,
docstrings, names and statement order. It is not a behavioral equivalence claim.
Evaluation, artifacts and provenance retain the exact source.
"""

import ast
from bisect import bisect_left, bisect_right
import hashlib
import math


def syntax_id(code):
    tree = ast.dump(ast.parse(code), include_attributes=False)
    return hashlib.sha256(tree.encode()).hexdigest()


def opportunity_key(anchor):
    return anchor.get("syntax_id", anchor["artifact_id"])


def unique_archive(anchors):
    unique = {}
    for anchor in anchors.values():
        unique.setdefault(opportunity_key(anchor), anchor)
    return list(unique.values())


def select_parent(anchors, counts, constant=1., policy="rank_count"):
    pool = unique_archive(anchors)
    if not pool:
        raise ValueError("cannot select without an executable candidate")
    if policy not in {"rank_count", "raw_count"}:
        raise ValueError("unknown parent policy")
    qualities = sorted(a["fitness"] for a in pool)

    def count(a):
        return counts.get(opportunity_key(a), 0)

    def quality(a):
        if policy == "raw_count":
            return a["fitness"]
        lo, hi = bisect_left(qualities, a["fitness"]), bisect_right(qualities, a["fitness"])
        return (lo + hi) / (2 * len(pool))

    def score(a):
        return quality(a) + constant / math.sqrt(count(a) + 1)

    parent = max(pool, key=lambda a: (score(a), -count(a), -a["id"]))
    return parent, {
        "policy": policy, "eligible_syntax_sources": len(pool),
        "eligible_unique_sources": len({a['artifact_id'] for a in anchors.values()}),
        "opportunity_key": opportunity_key(parent), "parent_count_before": count(parent),
        "quality": parent["fitness"], "selection_quality": quality(parent),
        "bonus": constant / math.sqrt(count(parent) + 1),
        "selection_score": score(parent), "exploration_constant": constant,
        "quality_min": qualities[0], "quality_max": qualities[-1],
        "quality_scale": "empirical mid-CDF" if policy == "rank_count" else "raw fitness",
    }
