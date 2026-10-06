"""The formation path of a program: the generation events that produced it, failed versions included."""

import difflib


def path(node, programs):
    """Programs from the initial one to ``node``, each produced from the previous one."""
    items = [node]
    seen = {node["id"]}
    while items[-1]["parent_id"] is not None:
        parent = programs[items[-1]["parent_id"]]
        if parent["id"] in seen:
            raise ValueError("cyclic formation path")
        items.append(parent)
        seen.add(parent["id"])
    return list(reversed(items))


def score_text(score):
    return format(score, ".6g")


def verdict(parent, child, higher_is_better):
    delta = child - parent if higher_is_better else parent - child
    tolerance = 1e-9 * max(1.0, abs(parent))
    return "improved" if delta > tolerance else "worse" if delta < -tolerance else "same score"


def code_diff(before, after):
    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(), n=2, lineterm=""))[2:]
    return "\n".join(lines)
