"""Conditional experience, atomic local updates, and the reflection interface."""

from copy import deepcopy
import json

from traceaad_v21.context import COMMON

from .evidence import is_relation

INTERVAL = 10
CAPACITY = 6
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
[Current Experience]
{notes}

{evidence}

[Instruction]
You maintain design experience for designers modifying other programs in this search.
Resolve the one computational question in this packet. First connect each relevant actual change to its starting Program, evaluated Program and measured result. Then state which relationship changed your understanding and what it implies for a similar computation elsewhere.

{common}

An entry preserves a reliable relation: in this computation, what changed, what the related attempts actually showed, and what to retain, adjust or check in the next implementation. A static computational fact may need only one event; a same-start comparison or a change that works only after a supporting adjustment needs the corresponding related attempts. Select the smallest connected evidence relation that explains the lesson, using at most {representatives} events. Performance differences alone do not establish their cause. When later results improve, check whether the computation under discussion remained in the code.
Use the task's calling semantics to locate the computation's effect. Separate what the candidate function controls from what its caller already guarantees. Results can support a conditional implementation comparison without a complete causal explanation. If the packet does not establish a new relation, keep it as an observation and return no experience update.

Only the related entries shown above and their source relations are under review. Other active entries stay unchanged. Use null for a new ID; existing IDs are stable. evidence_ids are Attempt IDs, not Program IDs. Preserve observations before extracting a design implication; content must stand alone without local Program labels. Unchanged entries, earlier versions and cumulative sources are retained by the system.
Active entries: {active_count}. Capacity: {capacity}. Update or merge the related entries when this question changes their lesson; archive a related entry only if the new relation removes its design use. If no slot is available and no related entry can be revised or archived, record the understanding in analysis without creating a new entry. A weaker current implementation does not erase its implementation lesson.

[Output]
Return exactly one valid JSON object, in this order:
- "analysis": In 2-4 concise sentences, first identify the specific Attempt changes and measured directions used to answer this one question, then explain what relationship changed your understanding. Do not give an overall ranking of the programs.
- "updates": Entries to create or revise, each with "id" (integer or null), "idea" (a short computational title), "when" (computational conditions), "lesson" (what was learned and its design implication), and "evidence_ids" (representative event IDs). Each content field uses 1-2 short sentences.
- "archive_ids": IDs of active entries to archive. Unmentioned entries remain unchanged.
Return only the JSON object, without Markdown fences or additional text."""

REFLECTION_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "traceaad_conditional_experience", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "analysis": {"type": "string"},
                "updates": {
                    "type": "array", "maxItems": CAPACITY,
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
                "archive_ids": {"type": "array", "maxItems": CAPACITY,
                                "items": {"type": "integer"}},
            },
            "required": ["analysis", "updates", "archive_ids"],
        },
    },
}


class Experience:
    """Active and archived entries; old versions remain in committed reflection snapshots."""

    def __init__(self, snapshot=None):
        self.data = deepcopy(snapshot) if snapshot is not None else {"next_id": 1, "entries": []}

    @property
    def active(self):
        return [{k: deepcopy(entry[k]) for k in FIELDS} for entry in self.data["entries"] if entry["active"]]

    def snapshot(self):
        return deepcopy(self.data)

    def apply(self, response, visible_events, *, editable_ids):
        """Validate the complete transaction before changing entries or cumulative sources."""
        value = json.loads(response)
        if (not isinstance(value, dict) or set(value) != {"analysis", "updates", "archive_ids"}
                or not isinstance(value["analysis"], str) or not isinstance(value["updates"], list)
                or not isinstance(value["archive_ids"], list) or len(value["updates"]) > CAPACITY):
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
                    or len(set(sources)) != len(sources) or not is_relation(visible_events, sources)):
                raise ValueError("Sources must form one connected shown evidence relation")
            if i is None:
                i = state["next_id"]
                state["next_id"] += 1
            seen.add(i)
            update = {"id": i, **{k: raw[k].strip() for k in ("idea", "when", "lesson")},
                      "evidence_ids": list(sources)}
            previous = entries.get(i, {}).get("source_ids", [])
            entries[i] = {**update, "active": True, "source_ids": list(dict.fromkeys(previous + sources))}
            applied.append(update)
        if sum(entry["active"] for entry in entries.values()) > CAPACITY:
            raise ValueError("Updates would exceed six active entries")
        state["entries"] = list(entries.values())
        self.data = state
        return {"analysis": value["analysis"].strip(), "updates": applied, "archive_ids": list(archive_ids)}


def reflection_prompt(task, notes, evidence, *, active_count):
    score = ("" if task.score_definition in task.task_description
             else f"\nScore definition: {task.score_definition}\n")
    listed = json.dumps(notes, ensure_ascii=False, indent=2) if notes else "No experience yet."
    return REFLECTION_TEMPLATE.format(task_description=task.task_description,
                                      function_template=task.function_template, score=score,
                                      notes=listed, evidence=evidence, common=COMMON,
                                      capacity=CAPACITY, representatives=REPRESENTATIVES,
                                      active_count=active_count)


def experience_block(notes):
    rows = [f"Computation: {n['idea']}\nApplicable computational conditions: {n['when']}\n"
            f"Experience: {n['lesson']}" for n in notes]
    return "[Design Experience]\n" + "\n\n".join(rows)
