"""V9.14 raw-quality/count baseline, with one opportunity account per source.

Historical source: b5bd6bbf:llm4ad/method/traceaad_v9_14/selection.py.
No normalization: c=1 is a frozen baseline, not a cross-task calibration claim.
"""

import math


def unique_archive(anchors):
    unique = {}
    for anchor in anchors.values():
        unique.setdefault(anchor["artifact_id"], anchor)
    return list(unique.values())


def select_parent(anchors, counts, constant=1.):
    pool = unique_archive(anchors)
    if not pool:
        raise ValueError("cannot select without an executable candidate")

    def count(a):
        return counts.get(a["artifact_id"], 0)

    def score(a):
        return a["fitness"] + constant / math.sqrt(count(a) + 1)

    parent = max(pool, key=lambda a: (score(a), -count(a), -a["id"]))
    return parent, {
        "policy": "v9.14_raw_quality_plus_inverse_sqrt_count",
        "eligible_unique_sources": len(pool), "parent_count_before": count(parent),
        "quality": parent["fitness"], "bonus": constant / math.sqrt(count(parent) + 1),
        "selection_score": score(parent), "exploration_constant": constant,
        "quality_min": min(a["fitness"] for a in pool),
        "quality_max": max(a["fitness"] for a in pool),
        "quality_scale": "raw evaluator fitness, higher is better; no normalization",
    }
