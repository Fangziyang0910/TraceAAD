"""Focused development relations with adjacent implementation and result evidence."""

from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from functools import lru_cache

from traceaad_base.models import Program
from traceaad_base.prompts import failure_feedback, failure_summary
from traceaad_base.selection import identity
from traceaad_v21_1.representation import bare

from .notes import DISPLAY_LIMIT, program_query, ranked_notes


@dataclass(frozen=True)
class Event:
    """One attempt's final evaluated implementation; IDs are attempt IDs."""

    id: int
    parent: Program | None
    child: Program
    best_before: float | None = None
    experience_ids: tuple[int, ...] = ()

    @property
    def key(self):
        parent = self.parent
        before = parent.evaluation if parent else None
        after = self.child.evaluation
        return (identity(parent.code) if parent else None, identity(self.child.code),
                before.status if before else None, before.fitness if before else None,
                after.status, after.fitness, after.failure)


def observe(attempt, archive):
    if not attempt.evaluations or attempt.program_id is None:
        return None
    return Event(attempt.id, archive.get(attempt.parent_id), archive[attempt.program_id],
                 attempt.record.get("best_before"), tuple(attempt.record.get("experience_ids", ())))


def _change(event):
    """A complete diff or the complete evaluated source, never a rewrite summary."""
    child = event.child
    if not child.valid:
        return "source", child.code
    new = bare(child.code)
    if event.parent is None:
        return "source", new
    old = bare(event.parent.code)
    if old == new:
        return "unchanged", new
    diff = "\n".join(unified_diff(old.splitlines(), new.splitlines(),
                                  fromfile="parent", tofile="evaluated child", lineterm=""))
    rewrite = SequenceMatcher(None, old.splitlines(), new.splitlines(), autojunk=False).ratio() < .5
    return ("source", new) if rewrite or len(diff) >= len(new) else ("diff", diff)


def change_sites(event):
    """Positions in this exact parent, not a semantic classification of algorithms."""
    if event.parent is None:
        return frozenset()
    return _sites(bare(event.parent.code), bare(event.child.code))


@lru_cache(maxsize=4096)
def _sites(before, after):
    before, after = before.splitlines(), after.splitlines()
    matcher = SequenceMatcher(None, before, after, autojunk=False)
    if matcher.ratio() < .5:
        return frozenset()  # A rewrite does not identify one comparable parent computation.
    return frozenset(i for tag, a, b, _, _ in matcher.get_opcodes() if tag != "equal"
                     for i in range(a, max(a + 1, b)))


def same_question(a, b):
    if a.parent is None or b.parent is None:
        return False
    same_start = identity(a.parent.code) == identity(b.parent.code)
    return same_start and bool(change_sites(a) & change_sites(b))


def connected(a, b):
    return (a.id == b.id or same_question(a, b)
            or (a.parent is not None and a.parent.id == b.child.id)
            or (b.parent is not None and b.parent.id == a.child.id))


@dataclass(frozen=True)
class Question:
    anchor_id: int
    kind: str
    event_ids: tuple[int, ...]
    background_ids: tuple[int, ...] = ()

    def describe(self, events):
        anchor = events[self.anchor_id]
        return {"anchor_attempt": self.anchor_id, "parent_program": anchor.parent.id,
                "kind": self.kind, "events": list(self.event_ids),
                "background_events": list(self.background_ids)}


def questions(events, unreviewed):
    """Propose bounded comparisons around actual edits, with no model or text retrieval."""
    ordered = sorted(events.values(), key=lambda e: e.id)
    by_program = {e.child.id: e for e in ordered}
    by_start, by_parent = {}, {}
    for event in ordered:
        if event.parent is not None:
            by_start.setdefault(identity(event.parent.code), []).append(event)
            by_parent.setdefault(event.parent.id, []).append(event)
    proposals = {}
    for i in unreviewed:
        anchor = events[i]
        if anchor.parent is None or identity(anchor.parent.code) == identity(anchor.child.code):
            continue
        peers = [e for e in by_start[identity(anchor.parent.code)] if e.id != i and e.key != anchor.key
                 and identity(e.child.code) != identity(anchor.child.code) and same_question(anchor, e)]
        origin = by_program.get(anchor.parent.id)
        following = [e for e in by_parent.get(anchor.child.id, [])
                     if e.child.operator in ("Refine", "Crossover")
                     and identity(e.parent.code) != identity(e.child.code)]
        core, kind = {i}, "single_change"
        if peers:
            core.add(peers[-1].id)
            kind = "same_start"
        elif origin is not None and origin.child.operator != "Init":
            core.add(origin.id)
            kind = "continuation"
        if following:
            core.add(following[-1].id)
            kind = "continuation" if kind == "single_change" else kind
        background = ((origin.id,) if origin is not None and origin.id not in core else ())
        question = Question(i, kind, tuple(sorted(core)), background)
        proposals[question.event_ids] = question
    return sorted(proposals.values(), key=lambda q: (len(q.event_ids) > 1, q.anchor_id), reverse=True)


def related_notes(events, question, notes):
    core = [events[i] for i in question.event_ids]
    exposed = {i for event in core for i in event.experience_ids}
    query = "\n".join(program_query(event.child) + "\n" + program_query(event.parent) for event in core)
    related = []
    for score, note in ranked_notes(notes, query):
        if any(i not in events for i in note["evidence_ids"]):
            raise ValueError(f"Missing source events for experience {note['id']}")
        linked = any(connected(events[i], event) for i in note["evidence_ids"] for event in core)
        shown = note["id"] in exposed
        if linked or shown or score > 0:
            overlap = len(set(note["evidence_ids"]) & set(question.event_ids))
            related.append(((overlap, score, linked, shown, note["id"]), note))
    return [note for _, note in sorted(related, key=lambda row: row[0], reverse=True)]


def render(events, event_ids, new_ids, source_ids, question=None):
    """Starting code, actual edits and results stay together under stable Program/Attempt IDs."""
    ordered = [events[i] for i in sorted(event_ids)]
    parents = {e.parent.id for e in ordered if e.parent is not None}
    printed, groups, represented = {}, {}, {}
    for event in ordered:
        groups.setdefault(event.parent.id if event.parent else None, []).append(event)

    def implementation(program, *, exact=False):
        code = program.code if exact else bare(program.code)
        if code in printed:
            prior = printed[code]
            return (f"Program {program.id}: complete implementation shown above." if prior == program.id
                    else f"Program {program.id}: identical implementation to Program {prior}, shown above.")
        printed[code] = program.id
        return f"Program {program.id} implementation:\n```python\n{code}\n```"

    selected = set(event_ids)
    rows = ["[Trajectory Evidence]"]
    if question is not None:
        anchor = events[question.anchor_id]
        rows.append(f"[One Computational Question]\nExamine the computation changed by Attempt {anchor.id} "
                    f"from Program {anchor.parent.id}. Establish the change and result within each case, "
                    "then compare cases that share the computational condition. The packet includes "
                    "local development and candidate experience sources from across the search.")
    rows.extend([f"New events: {sorted(selected & set(new_ids))}",
                 f"Experience source events: {sorted(selected & set(source_ids))}"])
    for parent_id, group in groups.items():
        parent = group[0].parent
        rows.append(f"[Same Starting Point: Program {parent_id}]" if parent is not None else "[Initialization]")
        if parent is not None:
            rows.append(implementation(parent))
        for event in group:
            child = event.child
            before = event.parent
            if event.key in represented:
                rows.append(f"Attempt {event.id}: Program {before.id if before else None} -> Program {child.id}; "
                            f"same implementation and feedback as Attempt {represented[event.key]}. "
                            "These occurrences share one observation.")
                continue
            represented[event.key] = event.id
            lines = [f"Attempt {event.id}: Program {before.id if before else None} -> Program {child.id}; "
                     f"operator {child.operator}"]
            if event.experience_ids:
                lines.append(f"Experience available during generation: {list(event.experience_ids)}")
            if child.reference_id is not None:
                lines.append(f"Reference: Program {child.reference_id}")
            mode, changed = _change(event)
            if mode == "diff":
                lines.append(f"Actual change:\n```diff\n{changed}\n```")
            elif mode == "unchanged":
                lines.append("Actual change: none; the evaluated computation is unchanged.")
            else:
                lines.append("Actual change: complete evaluated implementation follows.")
            if child.valid:
                if before is not None and before.valid:
                    a, b = before.evaluation.fitness, child.evaluation.fitness
                    result = "improved" if b < a else "tied" if b == a else "regressed"
                    lines.append(f"Measured result for Attempt {event.id}: parent {a} -> child {b}; {result}")
                else:
                    lines.append(f"Measured score for Attempt {event.id}: {child.evaluation.fitness}")
                if event.best_before is not None and child.evaluation.fitness < event.best_before:
                    lines.append("This attempt set a new search best at that time.")
            else:
                failure = child.evaluation.failure or ""
                feedback = (failure_summary(failure) if child.evaluation.status == "timeout"
                            else failure_feedback(failure))
                lines.append(f"Execution result for Attempt {event.id}: {child.evaluation.status}\n{feedback}")
            if mode == "source" or child.id in parents:
                lines.append(implementation(child, exact=not child.valid))
            duplicates = [e.id for e in events.values() if e.id != event.id and e.key == event.key]
            if duplicates:
                lines.append(f"Equivalent observations also recorded at Attempts {sorted(duplicates)}; "
                             "these occurrences share one observation.")
            rows.append("\n".join(lines))
    return "\n\n".join(rows)


@dataclass(frozen=True)
class Material:
    prompt: str
    event_ids: tuple[int, ...]
    new_event_ids: tuple[int, ...]
    deferred_ids: tuple[int, ...]
    input_size: int
    related_ids: tuple[int, ...] = ()
    focus: dict | None = None
    reason: str | None = None


def build(events, unreviewed, notes, prompt_for, measure, limit):
    """Start with a focused question, then add complete candidate experience sources."""
    proposals = questions(events, unreviewed)
    if not proposals:
        return Material("", (), (), tuple(unreviewed), 0, reason="no_new_computational_question")
    for question in proposals:
        selected = set(question.event_ids) | set(question.background_ids)
        text = render(events, selected, unreviewed, (), question)
        prompt = prompt_for([], text)
        size = measure(prompt)
        if size <= limit:
            break
    else:
        return Material("", (), (), tuple(unreviewed), size, reason="focused_relation_exceeds_input_limit")
    shown_notes, sources = [], set()
    for note in related_notes(events, question, notes):
        if len(shown_notes) == DISPLAY_LIMIT:
            break
        candidate_ids = selected | set(note["evidence_ids"])
        candidate_sources = sources | set(note["evidence_ids"])
        candidate_notes = shown_notes + [note]
        candidate = prompt_for(candidate_notes, render(events, candidate_ids, unreviewed,
                                                       candidate_sources, question))
        candidate_size = measure(candidate)
        if candidate_size <= limit:
            selected, sources, shown_notes = candidate_ids, candidate_sources, candidate_notes
            prompt, size = candidate, candidate_size
    read = tuple(i for i in unreviewed if i in selected)
    deferred = tuple(i for i in unreviewed if i not in selected)
    return Material(prompt, tuple(sorted(selected)), read, deferred, size,
                    tuple(n["id"] for n in shown_notes), question.describe(events))
