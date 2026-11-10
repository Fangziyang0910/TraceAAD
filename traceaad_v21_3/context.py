"""V21.3 trajectory views: V21.2 with every evolution step shown through its actual code change.

A record of a step (a program with a parent) gives its decision, its full diff against the parent and its result,
whether the program is valid or failed, an ancestor, an offspring or a formation step of a reference. A record of a
program gives its design and score. The current algorithm gives its full code. Selection, ordering and instructions
are those of V21.2.
"""

from traceaad_v21_1.context import FAILURES, ancestors
from traceaad_v21_2.context import (COMMON, CROSSOVER, EXPLORE, OPERATORS, REFINE, _result, current_card, decision,
                                    explorations, receiver_attempts, refine_attempts, reference_steps)
from traceaad_v21_1.representation import bare
from .representation import changes

__all__ = ["COMMON", "CROSSOVER", "EXPLORE", "REFINE", "FAILURES", "card", "context", "decision", "reference_card"]


def card(archive, program):
    """Parent, the stated decision, the actual change, then the measured result; an initial program has no change."""
    parent = archive.get(program.parent_id)
    lines = [f"Algorithm {program.id}", f"Parent: Algorithm {program.parent_id}", f"Operator: {program.operator}"]
    if program.reference_id is not None:
        lines.append(f"Reference: Algorithm {program.reference_id}")
    if program.operator == "Init":
        lines.append(f"Design: {program.design}")
    else:
        if decision(program):
            lines.append(f"Decision: {decision(program)}")
        lines.append(changes(parent, program))
    lines.append(_result(parent, program))
    return "\n".join(lines)


def reference_card(archive, reference, *, code):
    lines = [f"Algorithm {reference.id}", f"Score: {reference.evaluation.fitness}", f"Design: {reference.design}"]
    if code:
        lines.append(f"```python\n{bare(reference.code)}\n```")
    blocks = ["\n".join(lines)]
    steps, _ = reference_steps(archive, reference)
    if steps:
        blocks.append(f"Formation steps of Algorithm {reference.id} whose added lines remain in its code, most such lines "
                      f"first, ordered from earlier to later. A remaining line shows where code came from, not that "
                      f"it drives the result.\n\n" + "\n\n".join(card(archive, p) for p in steps))
    return "\n\n".join(blocks), [p.id for p in steps]


def context(archive, parent, operator, chosen, config, *, reference_code=False):
    """Return the [Context] text, the [Instruction] text, whether scores are shown, and a record of what was shown."""
    view = {"ancestor_path": [], "ancestors": [], "children": [], "failed": [], "reference_steps": [], "explorations": []}
    if operator == "Explore":
        rows, total, shown = explorations(archive, parent)
        blocks = []
        if rows:
            blocks.append(f"[Earlier Explorations]\nDistinct initial and Explore algorithms in this search: {total}; "
                          f"showing {len(rows)}, the most developed and the most recent, ordered by creation.\n\n"
                          + "\n\n".join(rows))
        blocks.append("[Current Algorithm]\n" + current_card(parent, score=False))
        view["explorations"] = shown
        return "\n\n".join(blocks), EXPLORE, bool(rows), view
    blocks = []
    if operator == "Refine":
        formation, path = ancestors(archive, parent, config.ancestor_count)
        children, failures = refine_attempts(archive, parent, config)
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
        view.update(ancestor_path=[p.id for p in path], ancestors=[p.id for p in formation])
    else:
        children, failures = receiver_attempts(archive, parent, config)
    direct = [p for p in archive.values() if p.parent_id == parent.id and p.evaluation is not None]
    if children:
        total = sum(p.valid for p in direct)
        shown = ("the best improvement, if any, and the most recent" if operator == "Refine"
                 else "earlier Crossover attempts first, then the most recent")
        blocks.append(f"[Evaluated Offspring]\nDirect attempts from Algorithm {parent.id}: {total} valid; "
                      f"showing {len(children)}, {shown}, ordered by creation.\n\n"
                      + "\n\n".join(card(archive, p) for p in children))
    if failures:
        total = sum(not p.valid for p in direct)
        blocks.append(f"[Failed Offspring]\nFailed direct attempts from Algorithm {parent.id}: {total}; "
                      f"showing {len(failures)}.\n\n" + "\n\n".join(card(archive, p) for p in failures))
    if chosen is not None:
        text, steps = reference_card(archive, chosen, code=reference_code)
        blocks.append("[Reference Algorithm]\n" + text)
        view["reference_steps"] = steps
    blocks.append("[Current Algorithm]\n" + current_card(parent))
    parts = [COMMON, OPERATORS[operator]] + ([FAILURES] if failures else [])
    view.update(children=[p.id for p in children], failed=[p.id for p in failures])
    return "\n\n".join(blocks), "\n".join(parts), True, view
