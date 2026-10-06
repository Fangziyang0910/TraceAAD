"""The formation path of a program: the generation events that produced it, failed versions included."""

import difflib


def path(node, programs):
    """Programs from the initial one to ``node``, each produced from the previous one."""
    items = [node]
    while items[-1]["parent_id"] is not None:
        parent = programs[items[-1]["parent_id"]]
        items.append(parent)
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


def final_attempt(attempt, attempts):
    """One Repair, when present, determines the result of its original attempt."""
    return next((a for a in attempts.values() if a.get("repair_of") == attempt["id"]), attempt)
