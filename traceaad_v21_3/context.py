"""Refine reads a compact formation path and same-parent attempts; history contains measured changes and results."""

from traceaad_base.prompts import EXPLORE, failure_summary
from traceaad_base.selection import identity
from traceaad_v21_1.representation import bare

from .representation import changes, failure

REFINE = ("Improve the current algorithm while retaining its useful computations. "
          "Use its evolution history and previous attempts to inform what to change next. "
          "The recorded changes and results are observations, not explanations of their causes.")
CROSSOVER = ("Consider whether the reference offers a useful computation that complements the current algorithm. "
             "When appropriate, adapt and integrate that computation into the current decision procedure, aligning "
             "its inputs and numerical scales as needed. Otherwise, improve the current algorithm directly.")


def card(parent, program, label):
    """One labelled transition, without repeated node metadata or model descriptions."""
    if program.operator == "Crossover":
        label += f" (Crossover with reference Algorithm {program.reference_id})"
    if program.valid:
        a, b = parent.evaluation.fitness, program.evaluation.fitness
        outcome = "improved" if b < a else "tied" if b == a else "regressed"
        result = f"{a} -> {b}; {outcome}"
    else:
        result = failure(program.evaluation)
    return f"{label}: {result}\n{changes(parent, program)}"


def formation(archive, parent, count):
    """The actual path through the current program; keep its last three changes and earlier large score changes."""
    path, current = [], parent
    while current.id != 0:
        path.append(current)
        if current.parent_id in (None, 0):
            break
        current = archive[current.parent_id]
    path.reverse()
    if count <= 0 or len(path) < 2:
        return [], path
    steps = path[1:]
    recent = steps[-min(3, count):]
    earlier = steps[:-len(recent)]
    earlier = sorted(earlier, key=lambda p: (-abs(p.evaluation.fitness
                                                  - archive[p.parent_id].evaluation.fitness), -p.id))
    keep = {p.id for p in recent + earlier[:count - len(recent)]}
    return [p for p in steps if p.id in keep], path


def attempts(archive, parent, config):
    """Keep distinct same-parent results from every operator; prefer an improvement, then recent attempts."""
    records = [p for p in archive.values()
               if p.parent_id == parent.id and p.evaluation is not None]
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
        selected, path = formation(archive, parent, config.ancestor_count)
        if selected:
            rows = [f"Initial: fitness = {path[0].evaluation.fitness}"]
            positions = {p.id: i for i, p in enumerate(path)}
            previous = 0
            for program in selected:
                position = positions[program.id]
                if position > previous + 1:
                    rows.append(f"... {position - previous - 1} intermediate steps omitted.")
                rows.append(card(archive[program.parent_id], program, f"Step {position}"))
                previous = position
            blocks.append("[Evolution Path]\nEarlier -> later, ending at the current algorithm. "
                          f"Showing {len(selected)} of {len(path) - 1} changes: the latest up to three and "
                          "earlier changes with the largest absolute fitness differences.\n\n" + "\n\n".join(rows))
        children, failed, total = attempts(archive, parent, config)
        shown = sorted(children + failed, key=lambda p: p.id)
        if shown:
            blocks.append(f"[Attempts from Current Algorithm]\nAll attempts below start from the current code. "
                          f"Evaluated attempts: "
                          f"{total}; showing {len(children)} valid and {len(failed)} failed attempts, ordered by creation. "
                          "An improvement, when available, and the most recent distinct results are shown.\n\n"
                          + "\n\n".join(card(parent, p, f"Attempt {i}"
                                              + (f" ({p.operator})" if p.operator != "Crossover" else ""))
                                          for i, p in enumerate(shown, 1)))
        view = {"formation_path": [p.id for p in path], "formation_steps": [p.id for p in selected],
                "children": [p.id for p in children], "failed": [p.id for p in failed]}
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
