"""Conditional experience, atomic local updates, and the reflection interface."""

from collections import Counter
from copy import deepcopy
import json
from keyword import kwlist
from math import log, sqrt
import re

INTERVAL = 10
DISPLAY_LIMIT = 6
UPDATE_LIMIT = 6
REPRESENTATIVES = 3
CONTEXT_TOKENS = 49152
CHAT_RESERVE = 512
REFLECTION_TOKENS = 4096
FIELDS = ("id", "idea", "when", "lesson", "evidence_ids")

REFLECTION_TEMPLATE = """{task_description}

This search is designing the following Python function:
```python
{function_template}
```
{score}
[Experience Candidates]
{notes}

{evidence}

[Instruction]
You maintain design experience for designers modifying other programs in this search.
Resolve the computational question introduced by the focal edits. Connect each relevant change to its starting Program, evaluated Program and measured result. Use the task's calling semantics to trace how the changed computation reaches the decision. Explain the conditions under which the observed comparison guides a similar computation elsewhere.

Compare the experience candidates with these edits and their results. Cases from different branches can refine the same experience when they share a computational condition. Read each case through its own starting implementation, change and result, then identify the shared condition and the differences that matter. An experience listed during generation is a candidate connection; locate its computation in the evaluated code.

An entry records the computational problem, applicable conditions, the relationship learned and its design implication. It may capture an implementation fact, a performance tradeoff, or a computation that benefits from a supporting adjustment. Choose 1–{representatives} shown Attempt events that preserve the relationship, including the cases needed for a cross-branch comparison. Tie explanations to the implemented operations and measured changes. Write the experience using self-contained computational terms.

Add a reusable understanding as a new entry with a null ID. Refine or merge a shown entry when the cases develop its conditions or lesson, keeping its stable ID. Archive a shown entry when its useful content has been incorporated into another entry or the evidence establishes a replacement. The system retains entries, cumulative sources and previous versions, and selects a bounded subset for each generation. For observations that leave the experience unchanged, return empty update and archive lists.

[Output]
Return exactly one valid JSON object, in this order:
- "analysis": In 2-4 concise sentences, identify the Attempt changes and measured directions, then explain the relationship and computational conditions they reveal.
- "updates": Entries to create or revise, each with "id" (integer or null), "idea" (a short computational title), "when" (computational conditions), "lesson" (what was learned and its design implication), and "evidence_ids" (representative event IDs). Each content field uses 1-2 short sentences.
- "archive_ids": IDs of shown entries to archive. Unmentioned entries remain unchanged.
Use Attempt IDs in evidence_ids. Return the JSON object directly."""

REFLECTION_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "traceaad_conditional_experience", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "analysis": {"type": "string"},
                "updates": {
                    "type": "array", "maxItems": UPDATE_LIMIT,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "id": {"type": ["integer", "null"]},
                            "idea": {"type": "string"}, "when": {"type": "string"},
                            "lesson": {"type": "string"},
                            "evidence_ids": {"type": "array", "minItems": 1,
                                             "maxItems": REPRESENTATIVES,
                                             "items": {"type": "integer"}},
                        },
                        "required": list(FIELDS),
                    },
                },
                "archive_ids": {"type": "array", "maxItems": DISPLAY_LIMIT,
                                "items": {"type": "integer"}},
            },
            "required": ["analysis", "updates", "archive_ids"],
        },
    },
}


_COMMON_WORDS = frozenset(word.casefold() for word in kwlist) | frozenset(
    "the and for from with this that these those when where which what how into over under "
    "before after each any all its use uses using used can may has have had are was were "
    "will would should then than also through their them they both same other one two "
    "code function program implementation computation value result change return solve numpy".split())


def _terms(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {word for word in re.findall(r"[^\W\d_]+", text.casefold())
            if len(word) > 2 and word not in _COMMON_WORDS}


def ranked_notes(notes, text):
    """Rank short experience by IDF-weighted word overlap with the current computation."""
    documents = [_terms(" ".join(note.get(k, "") for k in ("idea", "when", "lesson"))) for note in notes]
    counts = Counter(term for terms in documents for term in terms)
    weights = {term: 1 + log((1 + len(notes)) / (1 + count)) for term, count in counts.items()}
    query = _terms(text) & weights.keys()
    query_norm = sqrt(sum(weights[t] ** 2 for t in sorted(query)))
    ranked = []
    for note, terms in zip(notes, documents, strict=True):
        norm = query_norm * sqrt(sum(weights[t] ** 2 for t in sorted(terms)))
        score = sum(weights[t] ** 2 for t in sorted(query & terms)) / norm if norm else 0.
        ranked.append((score, note))
    return sorted(ranked, key=lambda row: (row[0], row[1]["id"]), reverse=True)


def select_notes(notes, text, *, cursor):
    """Five matches and one rotating entry keep retained experience available over time."""
    if len(notes) <= DISPLAY_LIMIT:
        return notes
    rotating = sorted(notes, key=lambda n: n["id"])[cursor % len(notes)]
    matches = [n for _, n in ranked_notes(notes, text) if n["id"] != rotating["id"]]
    return matches[:DISPLAY_LIMIT - 1] + [rotating]


def program_query(program):
    """Descriptions provide recall terms; evaluated code supplies the implementation context."""
    return "\n".join((program.code, program.design or "", program.analysis or ""))


class Experience:
    """Retained knowledge and archived revisions, independent of the display budget."""

    def __init__(self, snapshot=None):
        self.data = deepcopy(snapshot) if snapshot is not None else {"next_id": 1, "entries": []}

    @property
    def active(self):
        """All retained, unarchived knowledge, available for display selection and reflection."""
        return [{k: deepcopy(entry[k]) for k in FIELDS} for entry in self.data["entries"] if entry["active"]]

    def snapshot(self):
        return deepcopy(self.data)

    def apply(self, response, visible_events, *, editable_ids):
        """Validate the complete transaction before changing entries or cumulative sources."""
        value = json.loads(response)
        if (not isinstance(value, dict) or set(value) != {"analysis", "updates", "archive_ids"}
                or not isinstance(value["analysis"], str) or not isinstance(value["updates"], list)
                or not isinstance(value["archive_ids"], list) or len(value["updates"]) > UPDATE_LIMIT
                or len(value["archive_ids"]) > DISPLAY_LIMIT):
            raise ValueError("Expected analysis, updates and archive_ids")
        state = self.snapshot()
        entries = {entry["id"]: entry for entry in state["entries"]}
        active = {i for i, entry in entries.items() if entry["active"]}
        archive_ids = value["archive_ids"]
        if (any(type(i) is not int or i not in active or i not in editable_ids for i in archive_ids)
                or len(set(archive_ids)) != len(archive_ids)):
            raise ValueError("Archive IDs must name distinct related active entries")
        for i in archive_ids:
            entries[i]["active"] = False
        seen, applied = set(), []
        for raw in value["updates"]:
            if not isinstance(raw, dict) or set(raw) != set(FIELDS):
                raise ValueError("Each update needs id, idea, when, lesson and evidence_ids")
            i = raw["id"]
            if i is not None and (type(i) is not int or i not in active or i not in editable_ids
                                  or i in archive_ids or i in seen):
                raise ValueError("An update must name one related active entry, once, without archiving it")
            if any(not isinstance(raw[k], str) or not raw[k].strip() for k in ("idea", "when", "lesson")):
                raise ValueError("Experience content and computational conditions must be nonempty")
            sources = raw["evidence_ids"]
            if (not isinstance(sources, list) or not 1 <= len(sources) <= REPRESENTATIVES
                    or any(type(e) is not int or e not in visible_events for e in sources)
                    or len(set(sources)) != len(sources)):
                raise ValueError("Sources must name distinct shown Attempt events")
            if len({visible_events[e].key for e in sources}) != len(sources):
                raise ValueError("Sources must represent distinct observations")
            if i is None:
                i = state["next_id"]
                state["next_id"] += 1
            seen.add(i)
            update = {"id": i, **{k: raw[k].strip() for k in ("idea", "when", "lesson")},
                      "evidence_ids": list(sources)}
            previous = entries.get(i, {}).get("source_ids", [])
            entries[i] = {**update, "active": True, "source_ids": list(dict.fromkeys(previous + sources))}
            applied.append(update)
        state["entries"] = list(entries.values())
        self.data = state
        return {"analysis": value["analysis"].strip(), "updates": applied, "archive_ids": list(archive_ids)}


def reflection_prompt(task, notes, evidence):
    score = ("" if task.score_definition in task.task_description
             else f"\nScore definition: {task.score_definition}\n")
    listed = json.dumps(notes, ensure_ascii=False, indent=2)
    return REFLECTION_TEMPLATE.format(task_description=task.task_description,
                                      function_template=task.function_template, score=score,
                                      notes=listed, evidence=evidence, representatives=REPRESENTATIVES)


def experience_block(notes):
    rows = [f"Computation: {n['idea']}\nApplicable computational conditions: {n['when']}\n"
            f"Experience: {n['lesson']}" for n in notes]
    return "[Design Experience]\n" + "\n\n".join(rows)
