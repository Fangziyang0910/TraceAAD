"""Shared positive/negative experience with deterministic bilingual BM25."""

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
    origins: list[dict]
    score: int = 0
    active: bool = True
    provided_count: int = 0
    outcomes: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def content_key(polarity, content):
    return polarity + ":" + json.dumps(content, sort_keys=True, ensure_ascii=False)


class ExperiencePool:
    def __init__(self, records=()):
        self.entries = {e["id"]: Experience(**deepcopy(e)) for e in records}
        self._keys = {content_key(e.polarity, e.content): e.id
                      for e in self.entries.values() if e.active}

    def add(self, lessons, origin):
        ids = []
        for lesson in lessons:
            polarity, content = lesson["polarity"], deepcopy(lesson["content"])
            key = content_key(polarity, content)
            eid = self._keys.get(key)
            entry = self.entries.get(eid)
            if entry is None or not entry.active:
                eid = max(self.entries, default=0) + 1
                entry = Experience(eid, polarity, content, [])
                self.entries[eid] = entry
                self._keys[key] = eid
            if origin not in entry.origins:
                entry.origins.append(deepcopy(origin))
            ids.append(eid)
        return list(dict.fromkeys(ids))

    def feedback(self, provided_ids, *, attempt_id, improved):
        for eid in set(provided_ids):
            entry = self.entries[eid]
            if any(o["attempt_id"] == attempt_id for o in entry.outcomes):
                continue
            entry.provided_count += 1
            entry.outcomes.append({"attempt_id": attempt_id, "improved": bool(improved)})
            entry.score += 1 if improved else -1
            if entry.score < 0:
                entry.active = False

    def retrieve(self, query, k, rng):
        selected = []
        query_words = set(tokens(query))
        for polarity in ("positive", "negative"):
            entries = [e for e in self.entries.values() if e.active and e.polarity == polarity]
            docs = [tokens(" ".join([e.content["summary"], *e.content["recommendations"],
                                    e.content["applicable_when"]])) for e in entries]
            avg = sum(map(len, docs)) / max(1, len(docs))
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
                    scores.append((score, rng.random(), entry))
            for score, _, entry in sorted(scores, key=lambda x: (x[0], x[1]), reverse=True)[:k]:
                selected.append({**entry.to_dict(), "retrieval_score": score})
        return selected

    def to_list(self):
        return [self.entries[k].to_dict() for k in sorted(self.entries)]
