"""Conditional experience, atomic local updates, and the reflection interface."""

from copy import deepcopy
import json

from traceaad_v21.context import COMMON

INTERVAL = 10
CAPACITY = 6
REPRESENTATIVES = 1
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
Compare actual implementations, same-start attempts and connected changes to identify how a computation works under specific computational conditions. State what the observed results support and how that understanding affects a later implementation.

{common}

Each entry covers one computational question, its applicable conditions, and a concrete design implication. Conditions name an actual computation, dependency, or decision rule in the shown code, rather than just an algorithm family. They may describe the current code or a computation a designer plans to introduce. For a performance comparison, state the actual implementation change and its measured effect under those conditions, rather than ranking different algorithm structures in general.
Distinguish a computation that did not affect decisions, an execution failure, and a performance change after valid execution. When a later change improves the result, check whether the computation being discussed was retained. Record a concrete fact or conditional comparison when that is all the evidence shows; do not invent a cause.

When new evidence bears on an existing entry, read its source implementations and update that entry. Otherwise leave it unchanged. Use null for a new ID; existing IDs are stable. Each update names one representative parent-child event from the evidence shown here, the one that best explains the current computational condition and lesson. These are attempt IDs, not Algorithm IDs. Unchanged entries and all earlier sources are preserved by the system.
Keep at most {capacity} active entries after the updates and archives. Merge entries about the same computational conditions, or archive entries no longer useful for design. A currently weaker implementation does not erase a useful implementation lesson. If there is no new understanding, return no updates and no archives.

[Output]
Return exactly one valid JSON object, in this order:
- "analysis": In 2-4 concise sentences, explain which relationships in this evidence changed your understanding. Do not review every event separately.
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

    def apply(self, response, visible_ids):
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
        if (any(type(i) is not int or i not in active for i in archive_ids)
                or len(set(archive_ids)) != len(archive_ids)):
            raise ValueError("Archive IDs must name distinct active entries")
        for i in archive_ids:
            entries[i]["active"] = False
        seen, applied = set(), []
        for raw in value["updates"]:
            if not isinstance(raw, dict) or set(raw) != set(FIELDS):
                raise ValueError("Each update needs id, idea, when, lesson and evidence_ids")
            i = raw["id"]
            if i is not None and (type(i) is not int or i not in active or i in archive_ids or i in seen):
                raise ValueError("An update must name one active entry, once, without archiving it")
            if any(not isinstance(raw[k], str) or not raw[k].strip() for k in ("idea", "when", "lesson")):
                raise ValueError("Experience content and computational conditions must be nonempty")
            sources = raw["evidence_ids"]
            if (not isinstance(sources, list) or not 1 <= len(sources) <= REPRESENTATIVES
                    or any(type(e) is not int or e not in visible_ids for e in sources)
                    or len(set(sources)) != len(sources)):
                raise ValueError("Representative sources must be distinct events shown in this reflection")
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


def reflection_prompt(task, notes, evidence):
    score = ("" if task.score_definition in task.task_description
             else f"\nScore definition: {task.score_definition}\n")
    listed = json.dumps(notes, ensure_ascii=False, indent=2) if notes else "No experience yet."
    return REFLECTION_TEMPLATE.format(task_description=task.task_description,
                                      function_template=task.function_template, score=score,
                                      notes=listed, evidence=evidence, common=COMMON,
                                      capacity=CAPACITY, representatives=REPRESENTATIVES)


def experience_block(notes):
    rows = [f"Computation: {n['idea']}\nApplicable computational conditions: {n['when']}\n"
            f"Experience: {n['lesson']}" for n in notes]
    return "[Design Experience]\n" + "\n\n".join(rows)
