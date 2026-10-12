"""Find the earlier programs of a search that implement the computation an Explore proposal names.

Shared by the search and by the offline replay of base runs (E0).
"""

import re

import numpy as np

# The proposal step ends where code would start; the implementation step writes the code.
STOPS = ("```", "\nCode:")


def cut_proposal(text):
    """The Analysis and Design of an Explore reply, without any code."""
    text = text or ""
    ends = [i for i in (text.find(stop) for stop in STOPS) if i >= 0]
    return (text[:min(ends)] if ends else text).strip()


def normal(text):
    return " ".join((text or "").split())


def rank_candidates(design, programs, exclude_id, vector, limit):
    """The ``limit`` valid programs whose Designs are closest to ``design``, as (program, cosine).

    ``programs`` are the programs evaluated so far; the current program is excluded.
    A Design text shared by several programs is listed once, for its best-scoring program.
    """
    listed = {}
    valid = (p for p in programs.values() if p["valid"] and p["id"] != exclude_id)
    for program in sorted(valid, key=lambda p: (p["fitness"], p["id"])):
        text = normal(program.get("idea"))
        if text and text not in listed:
            listed[text] = program
    if not listed:
        return []

    def unit(text):
        v = np.asarray(vector(text), dtype=float)
        norm = np.linalg.norm(v)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("embedding returned an empty vector")
        return v / norm

    query = unit(normal(design))
    scored = [(program, float(unit(text) @ query)) for text, program in listed.items()]
    scored.sort(key=lambda item: (-item[1], item[0]["id"]))
    return scored[:limit]


def parse_match(text, labels):
    """Listed program ids on the last "Same:" line of the match reply; no such line names nothing."""
    answers = re.findall(r"^[\s*`\"']*same\s*:(.*)$", text or "", re.IGNORECASE | re.MULTILINE)
    if not answers:
        return []
    named = {int(n) for n in re.findall(r"\d+", answers[-1])}
    return sorted(named & set(labels))


def choose_linked(programs):
    """The best-scoring matched program (lower fitness is better; ties go to the earliest)."""
    return min(programs, key=lambda p: (p["fitness"], p["id"]))
