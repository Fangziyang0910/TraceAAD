"""Global experience from long-term reflection; retrieval shares corpus statistics across both pools."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
import re
import unicodedata


def tokens(text):
    text = unicodedata.normalize("NFKC", text).lower()
    words = re.findall(r"[a-z_][a-z_0-9]*|[0-9]+", text)
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        words.extend(run)
        words.extend(run[i:i + 2] for i in range(len(run) - 1))
    return words


@dataclass
class Experience:
    id: int
    polarity: str
    content: dict
    version: int
    provided_count: int = 0
    outcomes: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class ExperiencePool:
    """Positive and negative pools rewritten as a whole by long-term reflection."""

    def __init__(self, records=(), versions=()):
        self.entries = {e["id"]: Experience(**deepcopy(e)) for e in records}
        self.versions = deepcopy(list(versions))

    @property
    def current(self):
        return [self.entries[i] for i in (self.versions[-1]["entry_ids"] if self.versions else [])]

    @property
    def reflected_attempt_ids(self):
        return {i for version in self.versions for i in version["reflection_attempt_ids"]}

    def replace(self, lessons, origin):
        """Install the updated pools; earlier versions stay for analysis."""
        version, ids, seen = len(self.versions) + 1, [], set()
        for lesson in lessons:
            key = json.dumps([lesson["polarity"], lesson["content"]], sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            entry = Experience(max(self.entries, default=0) + 1, lesson["polarity"], deepcopy(lesson["content"]), version)
            self.entries[entry.id] = entry
            ids.append(entry.id)
        self.versions.append({"version": version, **deepcopy(origin), "entry_ids": ids})
        return ids

    def record_usage(self, provided_ids, *, attempt_id, improved):
        for eid in set(provided_ids):
            entry = self.entries[eid]
            if any(o["attempt_id"] == attempt_id for o in entry.outcomes):
                continue
            entry.provided_count += 1
            entry.outcomes.append({"attempt_id": attempt_id, "improved": improved})

    def retrieve(self, query, k, rng=None):
        selected = []
        query_words = set(tokens(query))
        entries = self.current
        if not query_words or not k or not entries:
            return selected
        docs = [tokens(e.content["summary"] + " " + e.content["applicable_when"]) for e in entries]
        avg = sum(map(len, docs)) / len(docs)
        frequencies = Counter(word for doc in docs for word in set(doc))
        scores = []
        for entry, doc in zip(entries, docs):
            counts, score = Counter(doc), 0.0
            for word in sorted(query_words & counts.keys()):
                df = frequencies[word]
                idf = math.log(1 + (len(docs) - df + .5) / (df + .5))
                tf = counts[word]
                score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * len(doc) / max(avg, 1)))
            if score > 0:
                scores.append((score, entry))
        for polarity in ("positive", "negative"):
            pool_scores = [(score, entry) for score, entry in scores if entry.polarity == polarity]
            for score, entry in sorted(pool_scores, key=lambda x: (-x[0], -x[1].id))[:k]:
                selected.append({**entry.to_dict(), "retrieval_score": score})
        return selected

    def to_list(self):
        return [self.entries[k].to_dict() for k in sorted(self.entries)]
