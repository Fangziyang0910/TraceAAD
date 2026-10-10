"""Observable quality groups and bounded control declarations; no semantic oracle."""

import hashlib
import re

AXES = (
    "information: which visible information is lost, misrepresented or repeatedly processed?",
    "output: how could the returned choice, ordering, sampling contrast or perturbation affect the solver?",
    "state: how should visible state, exposed phase or permitted randomness affect this output?",
    "prediction: could lookahead, completion, local improvement or decomposition improve this decision?",
    "organization: could shared, batched or incremental computation realize a useful output effect?",
)
SOURCES = ("Refine", "Crossover", "Explore")


def stable_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def score_vector(program, evaluations):
    # Freeze the original search observation; final rechecks are not online evidence.
    rows = {}
    for e in evaluations:
        if e["key"] == program["key"] and e["role"] == "search" and e["valid"]:
            rows.setdefault((e["protocol"], e["seed"]), e)
    if not rows:
        raise ValueError(f"program {program['id']} has no search observations")
    return tuple((protocol, seed, tuple((i["instance_index"], i["score"])
                 for i in sorted(e["instances"], key=lambda i: i["instance_index"])))
                 for (protocol, seed), e in sorted(rows.items()))


def values(vector):
    return [score for _, _, instances in vector for _, score in instances]


def frontier(programs, evaluations, roots, units, count, seed, excluded=()):
    ids = set(roots) | {u["champion_id"] for u in units if u["champion_id"] is not None}
    groups, sources = {}, set()
    for pid in sorted(ids - set(excluded)):
        p = programs[pid]
        if p["key"] in sources:
            continue
        sources.add(p["key"])
        vector = score_vector(p, evaluations)
        groups.setdefault(vector, []).append(pid)
    ranked = sorted(groups.items(), key=lambda g: (programs[g[1][0]]["fitness"], stable_key(seed, g[0])))
    result, rank, previous = [], 0, None
    for position, (vector, members) in enumerate(ranked, 1):
        fitness = programs[members[0]]["fitness"]
        if fitness != previous:
            if position > count:
                break
            rank = position
        result.append({"key": stable_key(seed, vector), "vector": vector, "members": members, "rank": rank})
        previous = fitness
    return result


def declarations(text):
    """Only named fields before Design/Code may control the request."""
    if not isinstance(text, str):
        return {}
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if "<think>" in text.lower():
        return {}
    analysis = re.split(r"(?im)^\s*(?:Design|Idea|Code|Edits)\s*:|```|^<<<<<<< SEARCH", text, maxsplit=1)[0]
    fields = {}
    for match in re.finditer(r"(?im)^[ \t]*(Base|Question|Change|Effect|Evidence|Status)[ \t]*:[ \t]*([^\r\n]*)", analysis):
        fields[match[1].lower().replace(" ", "_")] = match[2].strip()
    fields["analysis"] = analysis.strip()
    return fields
