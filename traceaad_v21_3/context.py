"""Only Refine reads history: actual changes and outcomes from the same parent and operator."""

from traceaad_base.prompts import EXPLORE, REFINE, failure_summary
from traceaad_base.selection import identity
from traceaad_v21_1.representation import bare

from .representation import changes, failure

CROSSOVER = ("Consider whether the reference offers a useful computation that complements the current algorithm. "
             "When appropriate, adapt and integrate that computation into the current decision procedure, aligning "
             "its inputs and numerical scales as needed. Otherwise, improve the current algorithm directly.")


def card(parent, program):
    """A same-parent Refine attempt: measured result and complete code difference, without model descriptions."""
    lines = [f"Algorithm {program.id}", f"Parent: Algorithm {parent.id}", "Operator: Refine"]
    if program.valid:
        a, b = parent.evaluation.fitness, program.evaluation.fitness
        outcome = "improved" if b < a else "tied" if b == a else "regressed"
        lines.append(f"Result relative to Algorithm {parent.id}: {a} -> {b}; {outcome}")
    else:
        lines.append(failure(program.evaluation))
    lines.append(changes(parent, program))
    return "\n".join(lines)


def refine_attempts(archive, parent, config):
    """Keep distinct same-parent Refine results; prefer an improvement, then the latest attempts."""
    records = [p for p in archive.values()
               if p.parent_id == parent.id and p.operator == "Refine" and p.evaluation is not None]
    unique = {}
    for program in sorted(records, key=lambda p: p.id):
        evaluation = program.evaluation
        key = (identity(program.code), evaluation.status, evaluation.fitness,
               failure_summary(evaluation.failure or ""))
        unique[key] = program
    valid = [p for p in unique.values() if p.valid]
    best = min(valid, key=lambda p: (p.evaluation.fitness, -p.id), default=None)
    first = [best] if best is not None and best.evaluation.fitness < parent.evaluation.fitness else []
    latest = sorted((p for p in valid if p not in first), key=lambda p: -p.id)
    children = sorted((first + latest)[:config.offspring_count], key=lambda p: p.id)
    failed = sorted((p for p in unique.values() if not p.valid), key=lambda p: -p.id)[:config.failure_count]
    return children, sorted(failed, key=lambda p: p.id), len(records)


def current_card(program, *, score=True):
    lines = [f"Algorithm {program.id}"]
    if score:
        lines.append(f"Score: {program.evaluation.fitness}")
    lines.append(f"```python\n{bare(program.code)}\n```")
    return "\n".join(lines)


def context(archive, parent, operator, chosen, config, *, reference_code=False):
    """Return rendered material, instruction, score visibility and the actual selected records."""
    blocks, view = [], {}
    if operator == "Refine":
        children, failed, total = refine_attempts(archive, parent, config)
        shown = sorted(children + failed, key=lambda p: p.id)
        if shown:
            blocks.append(f"[Earlier Refine Attempts]\nEvaluated Refine attempts from Algorithm {parent.id}: "
                          f"{total}; showing {len(children)} valid and {len(failed)} failed attempts, ordered by creation. "
                          "An improvement, when available, and the most recent distinct results are shown.\n\n"
                          + "\n\n".join(card(parent, p) for p in shown))
        view = {"children": [p.id for p in children], "failed": [p.id for p in failed]}
        instruction = REFINE
    elif operator == "Crossover":
        if chosen is None:
            raise ValueError("Crossover requires a reference")
        reference = [f"Algorithm {chosen.id}", f"Score: {chosen.evaluation.fitness}", f"Design: {chosen.design}"]
        if reference_code:
            reference.append(f"```python\n{bare(chosen.code)}\n```")
        blocks.append("[Reference Algorithm]\n" + "\n".join(reference))
        view = {"reference_code": reference_code}
        instruction = CROSSOVER
    elif operator == "Explore":
        instruction = EXPLORE
    else:
        raise ValueError(f"Unknown evolution operator: {operator}")
    scores = operator != "Explore"
    blocks.append("[Current Algorithm]\n" + current_card(parent, score=scores))
    return "\n\n".join(blocks), instruction, scores, view
