"""Trajectory views and instructions for one generation; nothing here asks the model."""

from traceaad_base.prompts import failure_summary
from traceaad_base.selection import identity
from .representation import bare, changes

COMMON = ("Results and changed computations are measured facts. Designs and proposed changes are descriptions, "
          "and do not establish why a result occurred.")
REFINE = ("Find one limitation in what the current code computes and change the computation that addresses it, "
          "keeping its main mechanism. The change may add, replace, or remove a computation. "
          "Do not repeat a modification already tried. In analysis, say which computation you change and why.")
EXPLORE = ("Make the function reach its result through a computation the current code does not perform. Options include "
           "decision rules on different information, estimating consequences before deciding, iteratively improving a "
           "complete solution, decomposition, solving small parts exactly, and population or restart search. "
           "Implement that computation in full; it does not need to beat the current code yet. "
           "In analysis, state the new computation.")
CROSSOVER = ("Adapt one useful computation from the reference into the current method, adjusting representations as "
             "needed. If no coherent adaptation exists, say so and implement your best candidate. "
             "In analysis, name what you take and the decision it changes.")
OPERATORS = {"Refine": REFINE, "Explore": EXPLORE, "Crossover": CROSSOVER}
FAILURES = ("Use execution feedback to address the reported implementation issue; it does not establish that every "
            "realization of the idea is unsuitable.")
RECENT_ANCESTORS = 3


def ancestors(archive, parent, count):
    """The formation path: the nearest records plus the farther points where the score moved most."""
    path, current = [], parent.parent_id
    while current not in (None, 0):
        program = archive[current]
        path.append(program)
        current = program.parent_id

    def movement(program):
        before = archive.get(program.parent_id)
        return abs(program.evaluation.fitness - before.evaluation.fitness) if before and before.valid else -1.

    far = path[RECENT_ANCESTORS:]
    chosen = sorted(far, key=lambda p: -movement(p))[:count - RECENT_ANCESTORS]  # A stable sort keeps nearer ties first.
    keep = {p.id for p in path[:RECENT_ANCESTORS]} | {p.id for p in chosen}
    return [p for p in reversed(path) if p.id in keep], list(reversed(path))


def attempts(archive, parent, config):
    """Direct attempts shown for this parent; equal source with equal feedback counts once."""
    unique = {}
    for program in sorted(archive.values(), key=lambda p: p.id):
        if program.parent_id == parent.id and program.evaluation is not None:
            key = (identity(program.code), program.evaluation.status, program.evaluation.fitness,
                   failure_summary(program.evaluation.failure or ""))
            unique[key] = program
    valid = [p for p in unique.values() if p.valid]
    failed = [p for p in unique.values() if not p.valid]
    selected = []
    if valid:
        best = min(valid, key=lambda p: (p.evaluation.fitness, -p.id))
        selected = [best] + sorted((p for p in valid if p is not best), key=lambda p: -p.id)[:config.offspring_count - 1]
    failures = sorted(failed, key=lambda p: -p.id)[:config.failure_count]
    return sorted(selected, key=lambda p: p.id), sorted(failures, key=lambda p: p.id)


def _changed(parent, program):
    if identity(parent.code) == identity(program.code):
        return f"Changed computation: none; the code is identical to Algorithm {parent.id}."
    return changes(parent.code, program.code)


def _result(parent, program):
    if parent is not None and parent.valid:
        a, b = parent.evaluation.fitness, program.evaluation.fitness
        outcome = "improved" if b < a else "tied" if b == a else "regressed"
        return f"Result relative to Algorithm {parent.id}: {a} -> {b}; {outcome}"
    return f"Score: {program.evaluation.fitness}"


def card(archive, program, *, change=False):
    """Relation and measured result first, then what the generator said it did, then what actually changed."""
    parent = archive.get(program.parent_id)
    lines = [f"Algorithm {program.id}", f"Parent: Algorithm {program.parent_id}", f"Operator: {program.operator}"]
    if program.reference_id is not None:
        lines.append(f"Reference: Algorithm {program.reference_id}")
    if program.valid:
        lines.append(_result(parent, program))
    else:
        lines.append(f"Result: {program.evaluation.status}; {failure_summary(program.evaluation.failure or '')}")
    lines.append(f"Design: {program.design}")
    lines.append(f"Proposed change: {program.analysis}")
    if change and parent is not None and program.operator != "Init":
        changed = _changed(parent, program)
        if changed:
            lines.append(changed)
    return "\n".join(lines)


def current_card(program, *, score=True):
    """The code is what the algorithm computes; descriptions would frame how it is read."""
    lines = [f"Algorithm {program.id}"] + ([f"Score: {program.evaluation.fitness}"] if score else [])
    return "\n".join(lines + [f"```python\n{bare(program.code)}\n```"])


def reference_card(archive, program):
    """A source of computation, with the evidence of its last change when it has one."""
    lines = [f"Algorithm {program.id}", f"Score: {program.evaluation.fitness}", f"Design: {program.design}"]
    parent = archive.get(program.parent_id)
    if parent is not None and parent.valid and program.operator != "Init":
        lines.append(_result(parent, program))
        changed = _changed(parent, program)
        if changed:
            lines.append(changed)
    return "\n".join(lines + [f"```python\n{bare(program.code)}\n```"])


def context(archive, parent, operator, chosen, config):
    """Return the [Context] text, the [Instruction] text and a record of what was shown; each operator gets what its decision needs."""
    if operator == "Explore":
        view = {"ancestor_path": [], "ancestors": [], "children": [], "failed": []}
        return "[Current Algorithm]\n" + current_card(parent, score=False), EXPLORE, view
    formation, path = ancestors(archive, parent, config.ancestor_count) if operator == "Refine" else ([], [])
    children, failures = attempts(archive, parent, config)
    blocks = []
    if formation:
        order = [p.id for p in path]
        rows, previous = [], -1
        for program in formation:
            index = order.index(program.id)
            omitted = index - previous - 1
            rows.append(card(archive, program)
                        + (f"\nOmitted intermediate nodes before this record: {omitted}" if omitted else ""))
            previous = index
        blocks.append("[Ancestor History]\nAncestors of the current algorithm, ordered from earlier to later.\n\n"
                      + "\n\n".join(rows))
    direct = [p for p in archive.values() if p.parent_id == parent.id and p.evaluation is not None]
    if children:
        total = sum(p.valid for p in direct)
        blocks.append(f"[Evaluated Offspring]\nDirect attempts from Algorithm {parent.id}: {total} valid; "
                      f"showing {len(children)}.\n\n" + "\n\n".join(card(archive, p, change=True) for p in children))
    if failures:
        total = sum(not p.valid for p in direct)
        blocks.append(f"[Failed Offspring]\nFailed direct attempts from Algorithm {parent.id}: {total}; "
                      f"showing {len(failures)}.\n\n" + "\n\n".join(card(archive, p) for p in failures))
    if chosen is not None:
        blocks.append("[Reference Algorithm]\n" + reference_card(archive, chosen))
    blocks.append("[Current Algorithm]\n" + current_card(parent))
    parts = [COMMON, OPERATORS[operator]] + ([FAILURES] if failures else [])
    view = {"ancestor_path": [p.id for p in path], "ancestors": [p.id for p in formation],
            "children": [p.id for p in children], "failed": [p.id for p in failures]}
    return "\n\n".join(blocks), "\n".join(parts), view
