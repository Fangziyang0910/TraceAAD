"""Mechanism 3: the reflection prompt, its response format and the notes block shown to generation.

A note records how an idea is realized: what it computes and what its realizations taught.
Which approach suits a given program is left to that program and its own trajectory.
"""

import json

from traceaad_v21.context import COMMON, card

INTERVAL = 10
CAPACITY = 6

REFLECTION_TEMPLATE = """{task_description}

This search is designing the following Python function:
```python
{function_template}
```

{score}[Design Notes]
{notes}

[Recent Attempts]
The latest {count} attempts of the search, in order.

{cards}

[Instruction]
You keep the design notes of this search. Before each change, a designer working on one program of the search reads these notes, without seeing the attempts above. The designer wants to realize ideas well on their own program; which approach suits that program is decided by the program and its results, not by the notes.

{common}

Each note covers one idea. "idea" states what it computes, in terms that apply beyond the program where it was tried. "lesson" states what its realizations here showed: what the code actually did, what limited it or made it work, and how it can be realized better.

Revise a note when the new attempts bear on its idea; add a note only for an idea the notes do not cover. Keep at most {capacity} notes: merge related notes, or drop the one least useful for later design; a note is not less useful because it is old. Fewer notes are better than weak ones. If the attempts add nothing, keep the notes unchanged.

[Output]
Return exactly one valid JSON object with two fields, in this order:
- "analysis": In 2-4 concise sentences, for the ideas in these attempts, state what limited or helped each realization and how it could be realized better.
- "notes": The full updated notes, an array of at most {capacity} objects with string fields "idea" and "lesson", each in 1-2 short sentences.
Return only the JSON object, without Markdown fences or additional text."""

REFLECTION_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "traceaad_design_notes",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "notes": {
                    "type": "array", "maxItems": CAPACITY,
                    "items": {
                        "type": "object",
                        "properties": {"idea": {"type": "string"}, "lesson": {"type": "string"}},
                        "required": ["idea", "lesson"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["analysis", "notes"],
            "additionalProperties": False,
        },
    },
}

NOTES_HEADER = ("Notes written from attempts across this search, including programs other than the current one. "
                "They describe how ideas were realized and what limited them; they do not decide which approach "
                "suits the current program. Use them where they help this change.")

NEW_BEST = "This attempt set a new search best."


def attempt_card(archive, program, best_before):
    """Mechanism 2's card; an attempt that set a new search best says so."""
    text = card(archive, program, change=program.valid)
    if not (program.valid and program.evaluation.fitness < best_before):
        return text
    lines = text.split("\n")
    at = next(i for i, line in enumerate(lines) if line.startswith(("Result", "Score")))
    lines.insert(at + 1, NEW_BEST)
    return "\n".join(lines)


def reflection_prompt(task, notes, cards):
    score = ("" if task.score_definition in task.task_description
             else f"Score definition: {task.score_definition}\n\n")
    listed = "\n".join(f"{i}. Idea: {n['idea']}\n   Lesson: {n['lesson']}" for i, n in enumerate(notes, 1))
    return REFLECTION_TEMPLATE.format(
        task_description=task.task_description, function_template=task.function_template, score=score,
        notes=listed or "No notes yet.", count=len(cards), cards="\n\n".join(cards), common=COMMON,
        capacity=CAPACITY)


def parse(response):
    """Return (analysis, notes), or None when the reply is not the requested object."""
    try:
        data = json.loads(response)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("analysis"), str):
        return None
    notes = data.get("notes")
    if not isinstance(notes, list):
        return None
    kept = []
    for note in notes:
        if not (isinstance(note, dict) and isinstance(note.get("idea"), str) and isinstance(note.get("lesson"), str)):
            return None
        if note["idea"].strip() or note["lesson"].strip():
            kept.append({"idea": note["idea"].strip(), "lesson": note["lesson"].strip()})
    return data["analysis"].strip(), kept


def notes_block(notes):
    return "[Design Notes]\n" + NOTES_HEADER + "\n" + "\n".join(f"- {n['idea']}: {n['lesson']}" for n in notes)
