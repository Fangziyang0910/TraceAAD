"""V21.3 trajectory views: layered evidence on V21.2's operator material.

Every node has three layers: its design (what the program computes, written after its code), its actual change
from its parent (a diff, measured), and its full code. A material's role decides how far it is shown:
  current and reference algorithms   design + full code
  offspring, valid or failed         design + change (+ where a failure was raised)
  nearest three ancestors            design + change
  farther ancestors                  design
  earlier explorations               design + scores + later development (V21.2)
Each record also states its parent, operator and measured result. Nothing derived from `analysis` is shown.
"""

from traceaad_v21_1.context import FAILURES, RECENT_ANCESTORS, ancestors
from traceaad_v21_2.context import EXPLORE, REFINE, explorations, receiver_attempts, refine_attempts
from traceaad_v21_1.representation import bare
from .representation import changes, failure

COMMON = ("Code differences and evaluation results are observed facts. Designs describe implementations but may be "
          "inaccurate. A score change alone does not establish why the result changed.")
CROSSOVER = ("Improve the current algorithm with decision information from the reference algorithm. "
             "Use the reference algorithm to judge which computation supplies information the current algorithm "
             "lacks and what that computation depends on. "
             "When earlier attempts from the current algorithm are shown, use their changes and results. "
             "Redesign the chosen computation for the current algorithm as needed, for example as a cheaper "
             "approximation of the same information. Align input representations and numerical scales, and combine "
             "its result with the current decision mechanism so that both contribute to the returned result. "
             "If no computation of the reference fits the current algorithm, say so in analysis and implement the "
             "most coherent adaptation you can. "
             "In analysis, state what you take, what decision information it adds, and how you integrate it; "
             "end with one sentence that states the change you make.")
OPERATORS = {"Refine": REFINE, "Explore": EXPLORE, "Crossover": CROSSOVER}

__all__ = ["COMMON", "CROSSOVER", "EXPLORE", "REFINE", "FAILURES", "card", "context", "program_card"]


def _result(parent, program):
    if not program.valid:
        return failure(program.evaluation)
    if parent is not None and parent.valid:
        a, b = parent.evaluation.fitness, program.evaluation.fitness
        outcome = "improved" if b < a else "tied" if b == a else "regressed"
        return f"Result relative to Algorithm {parent.id}: {a} -> {b}; {outcome}"
    return f"Score: {program.evaluation.fitness}"


def card(archive, program, *, change=True):
    """A step: parent and operator, its design, its actual change (an initial program has none), its result."""
    parent = archive.get(program.parent_id)
    lines = [f"Algorithm {program.id}", f"Parent: Algorithm {program.parent_id}", f"Operator: {program.operator}"]
    if program.reference_id is not None:
        lines.append(f"Reference: Algorithm {program.reference_id}")
    lines.append(f"Design: {program.design}")
    if change and program.operator != "Init":
        lines.append(changes(parent, program))
    lines.append(_result(parent, program))
    return "\n".join(lines)


def program_card(program, *, score=True):
    """A program shown in full: its design and normalized code."""
    lines = [f"Algorithm {program.id}"] + ([f"Score: {program.evaluation.fitness}"] if score else [])
    return "\n".join(lines + [f"Design: {program.design}", f"```python\n{bare(program.code)}\n```"])


def context(archive, parent, operator, chosen, config):
    """Return the [Context] text, the [Instruction] text, whether scores are shown, and a record of what was shown."""
    view = {"ancestor_path": [], "ancestors": [], "ancestor_changes": [], "children": [], "failed": [],
            "explorations": []}
    if operator == "Explore":
        rows, total, shown = explorations(archive, parent)
        blocks = []
        if rows:
            blocks.append(f"[Earlier Explorations]\nDistinct initial and Explore algorithms in this search: {total}; "
                          f"showing {len(rows)}, the most developed and the most recent, ordered by creation.\n\n"
                          + "\n\n".join(rows))
        blocks.append("[Current Algorithm]\n" + program_card(parent, score=False))
        view["explorations"] = shown
        return "\n\n".join(blocks), EXPLORE, bool(rows), view
    blocks = []
    if operator == "Refine":
        formation, path = ancestors(archive, parent, config.ancestor_count)
        children, failures = refine_attempts(archive, parent, config)
        if formation:
            near = {p.id for p in path[-RECENT_ANCESTORS:]}
            order = [p.id for p in path]
            rows, previous = [], -1
            for program in formation:
                index = order.index(program.id)
                omitted = index - previous - 1
                rows.append(card(archive, program, change=program.id in near)
                            + (f"\nOmitted intermediate nodes before this record: {omitted}" if omitted else ""))
                previous = index
            blocks.append(f"[Ancestor History]\nAncestors of the current algorithm, ordered from earlier to later; "
                          f"the nearest {RECENT_ANCESTORS} show their code change; farther ones show only their design, as "
                          f"background to how the code formed, not as evidence of what changed.\n\n" + "\n\n".join(rows))
        view.update(ancestor_path=[p.id for p in path], ancestors=[p.id for p in formation],
                    ancestor_changes=[p.id for p in formation if p.id in near and p.operator != "Init"]
                    if formation else [])
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
        blocks.append("[Reference Algorithm]\n" + program_card(chosen))
    blocks.append("[Current Algorithm]\n" + program_card(parent))
    parts = [COMMON, OPERATORS[operator]] + ([FAILURES] if failures else [])
    view.update(children=[p.id for p in children], failed=[p.id for p in failures])
    return "\n\n".join(blocks), "\n".join(parts), True, view
