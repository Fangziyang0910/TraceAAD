"""Operator-specific trajectory views: modification, formation and integration, exploration and development."""

from traceaad_base.prompts import failure_summary
from traceaad_base.selection import identity

from .representation import bare, changes, failure

REFINE = ("Improve the current algorithm while retaining its useful computations. "
          "Use its evolution history and previous attempts, when available, to inform what to change next. "
          "The recorded changes and results are observations, not explanations of their causes.")
CROSSOVER = ("Use the reference algorithm's evolution and earlier integration attempts, when available, to judge "
             "what may be worth borrowing and how to adapt it to the current algorithm. Integrate a useful "
             "computation when appropriate; otherwise improve the current algorithm directly.")
EXPLORE = ("Develop an algorithm with a different main computation from the current code. "
           "Use earlier explorations and their later development, when available, to choose the next direction. "
           "A previously tried idea may be revisited through a different implementation.")
EXPLORATIONS = 6


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
    """Keep the most recent actual changes on the main-parent path, including the final change."""
    path, current = [], parent
    while current.id != 0:
        path.append(current)
        if current.parent_id in (None, 0):
            break
        current = archive[current.parent_id]
    path.reverse()
    if count <= 0 or len(path) < 2:
        return [], path
    return path[1:][-count:], path


def recent_records(records, config):
    """The most recent distinct successful and failed observations, without selecting a model example."""
    unique = {}
    for program in sorted(records, key=lambda p: p.id):
        evaluation = program.evaluation
        key = (program.parent_id, program.reference_id, identity(program.code), evaluation.status, evaluation.fitness,
               failure_summary(evaluation.failure or ""))
        unique[key] = program
    valid = [p for p in unique.values() if p.valid]
    children = sorted(sorted(valid, key=lambda p: -p.id)[:config.offspring_count], key=lambda p: p.id)
    failed = sorted((p for p in unique.values() if not p.valid), key=lambda p: -p.id)[:config.failure_count]
    return children, sorted(failed, key=lambda p: p.id), len(records)


def attempts(archive, parent, config):
    return recent_records([p for p in archive.values()
                           if p.parent_id == parent.id and p.evaluation is not None], config)


def integrations(archive, parent, reference, config):
    """Past integrations from this receiver or using this reference; no claim of computational equivalence."""
    return recent_records([p for p in archive.values() if p.operator == "Crossover" and p.evaluation is not None
                           and (p.parent_id == parent.id or p.reference_id == reference.id)], config)


def path_block(archive, program, count, heading):
    selected, path = formation(archive, program, count)
    rows = []
    if selected:
        rows.append(f"Initial: fitness = {path[0].evaluation.fitness}")
        positions = {p.id: i for i, p in enumerate(path)}
        previous = 0
        for node in selected:
            position = positions[node.id]
            if position > previous + 1:
                rows.append(f"... {position - previous - 1} intermediate steps omitted.")
            rows.append(card(archive[node.parent_id], node, f"Step {position}"))
            previous = position
    text = (f"[{heading}]\nEarlier -> later; showing the most recent {len(selected)} of {len(path) - 1} changes.\n\n"
            + "\n\n".join(rows)) if rows else ""
    return text, selected, path


def explorations(archive):
    """Recent search activity, with a direction's best observed implementation and the actual path to it."""
    roots, kids = [], {}
    for program in sorted(archive.values(), key=lambda p: p.id):
        if program.valid and program.operator in ("Init", "Explore"):
            roots.append(program)
        if program.evaluation is not None:
            kids.setdefault(program.parent_id, []).append(program)
    records = []
    for root in roots:
        stack, below = [p for p in kids.get(root.id, []) if p.operator != "Explore"], []
        while stack:
            node = stack.pop()
            below.append(node)
            stack.extend(p for p in kids.get(node.id, []) if p.operator != "Explore")
        best = min([root] + [p for p in below if p.valid], key=lambda p: (p.evaluation.fitness, p.id))
        path, node = [], best
        while node.id != root.id:
            path.append(node)
            node = archive[node.parent_id]
        path.reverse()
        records.append((root, below, best, path))
    recent = sorted(records, key=lambda r: max([r[0].id] + [p.id for p in r[1]]), reverse=True)
    distinct = {}
    for record in recent:
        distinct.setdefault(identity(record[0].code), record)
    records = list(distinct.values())[:EXPLORATIONS]
    rows, view = [], []
    for i, (root, below, best, path) in enumerate(sorted(records, key=lambda r: r[0].id), 1):
        row = [f"Direction {i} ({root.operator}): fitness at creation = {root.evaluation.fitness}",
               f"Design: {root.design}",
               f"Development: {len(below)} evaluated Refine/Crossover attempts; "
               f"best observed fitness on this branch, including the origin = {best.evaluation.fitness}"]
        if path:
            steps = [p.operator + (f" with reference Algorithm {p.reference_id}" if p.operator == "Crossover" else "")
                     for p in path]
            row.extend([f"Best later design: {best.design}", "Path to that result: " + " -> ".join(steps)])
        rows.append("\n".join(row))
        view.append({"origin": root.id, "development_count": len(below), "best": best.id,
                     "best_path": [root.id] + [p.id for p in path]})
    return rows, len(distinct), view


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
        history, selected, path = path_block(archive, parent, config.ancestor_count, "Evolution Path")
        if history:
            blocks.append(history)
        children, failed, total = attempts(archive, parent, config)
        shown = sorted(children + failed, key=lambda p: p.id)
        if shown:
            blocks.append(f"[Attempts from Current Algorithm]\nAll attempts below start from the current code. "
                          f"Evaluated attempts: "
                          f"{total}; showing {len(children)} valid and {len(failed)} failed attempts, ordered by creation. "
                          "The most recent distinct results are shown.\n\n"
                          + "\n\n".join(card(parent, p, f"Attempt {i}"
                                              + (f" ({p.operator})" if p.operator != "Crossover" else ""))
                                          for i, p in enumerate(shown, 1)))
        view = {"formation_path": [p.id for p in path], "formation_steps": [p.id for p in selected],
                "children": [p.id for p in children], "failed": [p.id for p in failed]}
        instruction = REFINE
    elif operator == "Crossover":
        if chosen is None:
            raise ValueError("Crossover requires a reference")
        history, selected, path = path_block(archive, chosen, config.ancestor_count, "Reference Evolution Path")
        if history:
            blocks.append(history)
        valid, failed, total = integrations(archive, parent, chosen, config)
        shown = sorted(valid + failed, key=lambda p: p.id)
        if shown:
            rows = []
            for i, program in enumerate(shown, 1):
                receiver = archive[program.parent_id]
                row = [card(receiver, program, f"Integration {i} from Algorithm {receiver.id}")]
                if receiver.id != parent.id:
                    row.append(f"Receiver design: {receiver.design}")
                if program.reference_id != chosen.id:
                    row.append(f"Reference design: {archive[program.reference_id].design}")
                rows.append("\n".join(row))
            blocks.append(f"[Earlier Integrations]\n{total} evaluated Crossovers from the current algorithm or using "
                          f"the reference; showing the most recent {len(shown)} distinct results. "
                          "Each change and result is relative to its own receiver.\n\n" + "\n\n".join(rows))
        reference = [f"Algorithm {chosen.id}", f"Score: {chosen.evaluation.fitness}", f"Design: {chosen.design}"]
        if reference_code:
            reference.append(f"```python\n{bare(chosen.code)}\n```")
        blocks.append("[Reference Algorithm]\n" + "\n".join(reference))
        view = {"reference_path": [p.id for p in path], "reference_steps": [p.id for p in selected],
                "integrations": [p.id for p in shown], "reference_code": reference_code}
        instruction = CROSSOVER
    elif operator == "Explore":
        rows, total, shown = explorations(archive)
        if rows:
            blocks.append(f"[Earlier Explorations]\n{total} distinct initial/Explore programs; showing {len(rows)} "
                          "with the most recent creation or later development, in creation order. Development "
                          "follows main-parent Refine/Crossover links; a later Explore starts a separate record.\n\n"
                          + "\n\n".join(rows))
        view = {"explorations": shown}
        instruction = EXPLORE
    else:
        raise ValueError(f"Unknown evolution operator: {operator}")
    scores = operator != "Explore" or bool(view["explorations"])
    blocks.append("[Current Algorithm]\n" + current_card(parent, score=operator != "Explore"))
    return "\n\n".join(blocks), instruction, scores, view
