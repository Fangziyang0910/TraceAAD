"""Complete implementation fragments for reflection, independent of model requests."""

from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff

from traceaad_base.models import Program
from traceaad_base.prompts import failure_feedback, failure_summary
from traceaad_v21.representation import bare, identity


@dataclass(frozen=True)
class Event:
    """One attempt's final evaluated implementation; IDs are attempt IDs."""

    id: int
    parent: Program | None
    child: Program
    best_before: float | None = None

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
                 attempt.record.get("best_before"))


def fragment(events, event_id):
    """The new edge, its parent's formation edge, and two earlier same-start alternatives."""
    event = events[event_id]
    chosen = {event_id}
    if event.parent is None:
        return chosen
    origin = next((e for e in events.values() if e.child.id == event.parent.id), None)
    if origin is not None:
        chosen.add(origin.id)
    earlier = [e for e in events.values() if e.id < event_id and e.parent is not None
               and e.parent.id == event.parent.id]
    valid = [e for e in earlier if e.child.valid]
    best = min(valid, key=lambda e: (e.child.evaluation.fitness, e.id), default=None)
    if best is not None:
        chosen.add(best.id)
    recent = [e for e in earlier if e.key != event.key and (best is None or e.key != best.key)]
    if recent:
        chosen.add(max(recent, key=lambda e: e.id).id)
    return chosen


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


def render(events, event_ids, new_ids, source_ids):
    """Deduplicate implementations and equal observations while preserving every parent link."""
    ordered = [events[i] for i in sorted(event_ids)]
    codes = {}

    def register(code):
        return codes.setdefault(code, f"Code {len(codes) + 1}")

    # A child used as another event's parent is shown in full once.
    for event in ordered:
        if event.parent is not None:
            register(bare(event.parent.code))
    changes = {e.id: _change(e) for e in ordered}
    for event in ordered:
        kind, text = changes[event.id]
        if kind in ("source", "unchanged"):
            register(text)

    groups = {}
    for event in ordered:
        groups.setdefault(event.key, []).append(event)
    rows = []
    for group in groups.values():
        event = group[0]
        child, parent = event.child, event.parent
        ids = [e.id for e in group]
        lines = [f"Observation events: {ids}; occurrences: {len(group)}"]
        for member in group:
            p = member.parent.id if member.parent else None
            c = member.child
            relation = f"Event {member.id}: Algorithm {p} -> Algorithm {c.id}; operator {c.operator}"
            if c.reference_id is not None:
                relation += f"; reference Algorithm {c.reference_id}"
            if (c.valid and member.best_before is not None
                    and c.evaluation.fitness < member.best_before):
                relation += "; this attempt set a new search best"
            lines.append(relation)
        if child.valid:
            if parent is not None and parent.valid:
                a, b = parent.evaluation.fitness, child.evaluation.fitness
                outcome = "improved" if b < a else "tied" if b == a else "regressed"
                lines.append(f"Measured result: parent {a} -> child {b}; {outcome}")
            else:
                lines.append(f"Measured score: {child.evaluation.fitness}")
        else:
            failure = child.evaluation.failure or ""
            feedback = (failure_summary(failure) if child.evaluation.status == "timeout"
                        else failure_feedback(failure))
            lines.append(f"Execution result: {child.evaluation.status}\n{feedback}")
        if parent is not None:
            lines.append(f"Parent implementation: {codes[bare(parent.code)]}")
        kind, text = changes[event.id]
        source = child.code if not child.valid else bare(child.code)
        if source in codes:
            lines.append(f"Evaluated implementation: {codes[source]}"
                         + ("; identical to parent" if kind == "unchanged" else ""))
        if kind == "diff":
            lines.append(f"Complete change from parent:\n```diff\n{text}\n```")
        lines.extend([f"Generator design (description): {child.design}",
                      f"Proposed change (description): {child.analysis}"])
        rows.append("\n".join(lines))

    ids = set(event_ids)
    index = (f"New events: {sorted(ids & set(new_ids))}\n"
             f"Experience source events: {sorted(ids & set(source_ids))}\n"
             f"Event order: {sorted(ids)}. Follow the explicit Algorithm parent -> child links; "
             "equal observations are grouped, and their occurrence count is not independent support.")
    implementations = "\n\n".join(f"{label}\n```python\n{code}\n```" for code, label in codes.items())
    return ("[Trajectory Evidence]\n" + index + "\n\n[Implementations]\n" + implementations
            + "\n\n[Changes and Results]\n" + "\n\n".join(rows))


@dataclass(frozen=True)
class Material:
    prompt: str
    event_ids: tuple[int, ...]
    new_event_ids: tuple[int, ...]
    deferred_ids: tuple[int, ...]
    input_size: int
    reason: str | None = None


def build(events, unreviewed, notes, prompt_for, measure, limit):
    """Pack whole fragments, with all active representative sources as required material."""
    sources = {i for note in notes for i in note["evidence_ids"]}
    missing = sources - events.keys()
    if missing:
        raise ValueError(f"Experience source events are missing: {sorted(missing)}")
    selected = set(sources)

    def prompt(ids):
        return prompt_for(render(events, ids, unreviewed, sources))

    # The usual cadence needs a single tokenizer call, not one call per edge.
    if len(unreviewed) <= 10:
        complete = selected | {i for event_id in unreviewed for i in fragment(events, event_id)}
        complete_text = prompt(complete)
        complete_size = measure(complete_text)
        if complete_size <= limit:
            return Material(complete_text, tuple(sorted(complete)), tuple(unreviewed), (), complete_size)

    text = prompt(selected)
    size = measure(text)
    if size > limit:
        return Material(text, (), (), tuple(unreviewed), size, "experience_sources_exceed_input_limit")
    # Give the oldest retained fact a turn, then current-cadence evidence, then the rest.
    # No event is dropped, including fragments that cannot fit this request.
    order = dict.fromkeys(unreviewed[:1] + unreviewed[-10:] + unreviewed[1:-10])
    for event_id in order:
        # Related context is one hop from this entry, not a recursive tree expansion.
        candidate = selected | fragment(events, event_id)
        proposed = prompt(candidate)
        proposed_size = measure(proposed)
        if proposed_size <= limit:
            selected, text, size = candidate, proposed, proposed_size
    # A newly shown edge may also be complete evidence inside another fragment.
    read = tuple(i for i in unreviewed if i in selected)
    deferred = tuple(i for i in unreviewed if i not in selected)
    reason = None if read else "no_complete_new_fragment_fits"
    return Material(text, tuple(sorted(selected)), read, deferred, size, reason)
