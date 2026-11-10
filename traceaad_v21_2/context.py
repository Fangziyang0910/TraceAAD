"""V21.2 trajectory views: each operator's material serves the design decision its analysis makes; nothing here asks the model."""

import re

from traceaad_base.prompts import failure_summary
from traceaad_base.selection import identity
from traceaad_v21_1.context import FAILURES, ancestors, attempts
from traceaad_v21_1.representation import bare, changes

COMMON = ("Results and changed computations are measured facts. Designs and decisions are descriptions, "
          "and do not establish why a result occurred.")
REFINE = ("Improve the current algorithm by changing one computation in it. "
          "Find a specific limitation or improvement opportunity in what the current code computes. "
          "When earlier attempts from this algorithm are shown, use their actual changes and results to judge what "
          "has been tried and what is still worth changing; otherwise, reason from the current computation and the "
          "task. "
          "Make one targeted change that addresses it by adding, replacing, or removing a computation. Make it "
          "different in substance from the attempts shown, and keep the rest of the main mechanism. "
          "In analysis, state the opportunity and, when attempts are shown, how they inform your choice; "
          "end with one sentence that states the change you make.")
CROSSOVER = ("Improve the current algorithm with decision information from the reference algorithm. "
             "Use the reference algorithm and the formation records of its computations to judge which computation "
             "supplies information the current algorithm lacks and what that computation depends on. "
             "When earlier attempts from the current algorithm are shown, use their changes and results. "
             "Redesign the chosen computation for the current algorithm as needed, for example as a cheaper "
             "approximation of the same information. Align input representations and numerical scales, and combine "
             "its result with the current decision mechanism so that both contribute to the returned result. "
             "In analysis, state what you take, what decision information it adds, and how you integrate it; "
             "end with one sentence that states the change you make.")
EXPLORE = ("Develop an algorithm that reaches its result through a computation the current code does not perform. "
           "Options include decision rules on different information, estimating consequences before deciding, "
           "iteratively improving a complete solution, decomposition, solving small parts exactly, and population "
           "or restart search. "
           "When earlier explorations are shown, use their designs, results, and later development to judge which "
           "different computation is still worth trying. A weak or little-developed implementation leaves other "
           "realizations of its idea open. "
           "Implement the new computation in full; it does not need to beat the current code yet. "
           "In analysis, state what the current code relies on, which new computation you introduce, and why it is "
           "worth trying; end with one sentence that states the change you make.")
OPERATORS = {"Refine": REFINE, "Explore": EXPLORE, "Crossover": CROSSOVER}
REFERENCE_STEPS = 4
EXPLORATIONS = 6


def decision(program):
    """The last sentence of the original analysis: the change the generator said it would make."""
    text = (program.analysis or "").split("\nRepair:")[0].strip()
    sentences = [s.strip() for s in re.split(r"(?<![Ee]\.g\.)(?<![Ii]\.e\.)(?<=[.!?。！？])\s+", text) if s.strip()]
    return sentences[-1] if sentences else ""


def _result(parent, program):
    if not program.valid:
        return f"Result: {program.evaluation.status}; {failure_summary(program.evaluation.failure or '')}"
    if parent is not None and parent.valid:
        a, b = parent.evaluation.fitness, program.evaluation.fitness
        outcome = "improved" if b < a else "tied" if b == a else "regressed"
        return f"Result relative to Algorithm {parent.id}: {a} -> {b}; {outcome}"
    return f"Score: {program.evaluation.fitness}"


def _changed(parent, program):
    if identity(parent.code) == identity(program.code):
        return f"Changed computation: none; the code is identical to Algorithm {parent.id}."
    return changes(parent.code, program.code)


def card(archive, program, *, change=False):
    """Parent, the stated decision, the actual change, then the measured result; full reasoning stays in the record."""
    parent = archive.get(program.parent_id)
    lines = [f"Algorithm {program.id}", f"Parent: Algorithm {program.parent_id}", f"Operator: {program.operator}"]
    if program.reference_id is not None:
        lines.append(f"Reference: Algorithm {program.reference_id}")
    if program.operator == "Init":
        lines.append(f"Design: {program.design}")
    elif decision(program):
        lines.append(f"Decision: {decision(program)}")
    if change and parent is not None and program.valid and program.operator != "Init":
        changed = _changed(parent, program)
        if changed:
            lines.append(changed)
    lines.append(_result(parent, program))
    return "\n".join(lines)


def current_card(program, *, score=True):
    lines = [f"Algorithm {program.id}"] + ([f"Score: {program.evaluation.fitness}"] if score else [])
    return "\n".join(lines + [f"```python\n{bare(program.code)}\n```"])


def _lines(code):
    return [line.strip() for line in bare(code).splitlines() if line.strip()]


def reference_steps(archive, reference):
    """Formation steps whose introduced lines the reference still executes, most surviving lines first."""
    path, current = [], reference
    while current is not None and current.operator not in (None, "Init"):
        parent = archive.get(current.parent_id)
        if parent is None or not parent.valid:
            break
        path.append(current)
        current = parent
    kept = set(_lines(reference.code))

    def surviving(program):
        return (set(_lines(program.code)) - set(_lines(archive[program.parent_id].code))) & kept

    def movement(program):
        return abs(program.evaluation.fitness - archive[program.parent_id].evaluation.fitness)

    ranked = sorted((p for p in path if surviving(p)), key=lambda p: (-len(surviving(p)), -movement(p), -p.id))
    chosen = sorted(ranked[:REFERENCE_STEPS], key=lambda p: p.id)
    return chosen, surviving


def reference_card(archive, reference, *, code):
    lines = [f"Algorithm {reference.id}", f"Score: {reference.evaluation.fitness}", f"Design: {reference.design}"]
    if code:
        lines.append(f"```python\n{bare(reference.code)}\n```")
    blocks = ["\n".join(lines)]
    steps, surviving = reference_steps(archive, reference)
    if steps:
        rows = []
        for program in steps:
            text = card(archive, program, change=True)
            if "most of the code was replaced" in text:
                still = surviving(program)
                ordered = [line for line in _lines(reference.code) if line in still]
                text += (f"\nIntroduced lines that Algorithm {reference.id} still executes:\n"
                         + "\n".join(dict.fromkeys(ordered)))
            rows.append(text)
        blocks.append(f"Formation records of computations that Algorithm {reference.id} still executes, "
                      f"ordered from earlier to later:\n\n" + "\n\n".join(rows))
    return "\n\n".join(blocks), [p.id for p in steps]


def _distinct_valid(archive, parent):
    unique = {}
    for program in sorted((p for p in archive.values() if p.parent_id == parent.id and p.valid), key=lambda p: p.id):
        unique[(identity(program.code), program.evaluation.fitness)] = program
    return list(unique.values())


def refine_attempts(archive, parent, config):
    """The best attempt when it improved on the parent, then the latest; a least-bad regression is not singled out."""
    _, failures = attempts(archive, parent, config)
    valid = _distinct_valid(archive, parent)
    best = min(valid, key=lambda p: (p.evaluation.fitness, -p.id), default=None)
    first = [best] if best is not None and best.evaluation.fitness < parent.evaluation.fitness else []
    latest = sorted((p for p in valid if p not in first), key=lambda p: -p.id)
    return sorted((first + latest)[:config.offspring_count], key=lambda p: p.id), failures


def receiver_attempts(archive, parent, config):
    """Earlier attempts from the receiver; earlier integrations first, then the latest others."""
    _, failures = attempts(archive, parent, config)
    valid = sorted(_distinct_valid(archive, parent), key=lambda p: (p.operator != "Crossover", -p.id))
    return sorted(valid[:config.offspring_count], key=lambda p: p.id), failures


def explorations(archive, parent):
    """Initial and Explore algorithms of this search with their later development; facts only, no verdicts."""
    seen, roots = set(), []
    for program in sorted(archive.values(), key=lambda p: p.id):
        if program.valid and program.operator in ("Init", "Explore") and program.id != parent.id:
            key = identity(program.code)
            if key not in seen:
                seen.add(key)
                roots.append(program)
    kids = {}
    for program in archive.values():
        if program.evaluation is not None:
            kids.setdefault(program.parent_id, []).append(program)

    def development(program):
        """Descendants through Refine and Crossover; a later Explore starts its own record."""
        stack, below = [p for p in kids.get(program.id, []) if p.operator != "Explore"], []
        while stack:
            node = stack.pop()
            below.append(node)
            stack.extend(p for p in kids.get(node.id, []) if p.operator != "Explore")
        return below

    stats = {p.id: development(p) for p in roots}
    developed = sorted(roots, key=lambda p: (-len(stats[p.id]), p.id))[:EXPLORATIONS // 2]
    rest = sorted((p for p in roots if p not in developed), key=lambda p: -p.id)[:EXPLORATIONS - len(developed)]
    shown = sorted(developed + rest, key=lambda p: p.id)
    rows = []
    for program in shown:
        earlier = [p.evaluation.fitness for p in archive.values() if p.valid and p.id < program.id]
        then = f" (best in the search then: {min(earlier)})" if earlier else ""
        below = stats[program.id]
        scores = [p.evaluation.fitness for p in below if p.valid]
        best = f"; best score among them: {min(scores)}" if scores else ""
        rows.append("\n".join([f"Algorithm {program.id}", f"Operator: {program.operator}",
                               f"Design: {program.design}",
                               f"Score at creation: {program.evaluation.fitness}{then}",
                               f"Development: {len(below)} later Refine or Crossover attempts from it and its "
                               f"descendants{best}"]))
    return rows, len(roots), [p.id for p in shown]


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
                      + "\n\n".join(card(archive, p, change=True) for p in children))
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
