"""Version 1: a quality-filtered crossover reference with a dissimilar formation trajectory."""

from difflib import unified_diff
import hashlib
import json
import time

import numpy as np

from traceaad_base.selection import reference_candidates, sources
from traceaad_v21_3.representation import bare


RULE = "trajectory_embedding_diverse_v1"
PATH_STEPS = 3
DIFF_LINES = 24


def settings_for(embedding):
    return {"rule": RULE, "path_steps": PATH_STEPS, "diff_lines": DIFF_LINES,
            "embedding": dict(embedding.settings)}


def trajectory_text(archive, program):
    """Use source changes and observed outcomes; no additional summary model call."""
    text = ["Current algorithm", program.design or "", "Current code", bare(program.code)]
    steps, child = [], program
    while len(steps) < PATH_STEPS:
        parent = archive.get(child.parent_id)
        if parent is None or not parent.valid:
            break
        rows = list(unified_diff(bare(parent.code).splitlines(), bare(child.code).splitlines(),
                                 lineterm="", n=1))[2:]
        before, after = parent.evaluation.fitness, child.evaluation.fitness
        outcome = "improved" if after < before else "degraded" if after > before else "unchanged"
        step = [f"Operation: {child.operator}", f"Observed result: {outcome}",
                "Actual code change", *rows[:DIFF_LINES]]
        if len(rows) > DIFF_LINES:
            step.append("Remaining diff lines omitted.")
        reference = archive.get(child.reference_id)
        if reference is not None:
            step.extend(["Crossover reference design", reference.design or ""])
        steps.append("\n".join(step))
        child = parent
    text.extend(["Recent formation steps, oldest to newest", *reversed(steps)])
    if not steps:
        text.append("Initial program; no recorded formation step.")
    return "\n".join(text)


class TrajectoryReferences:
    def __init__(self, embedding, store):
        self.embedding = embedding
        self.store = store
        self.settings = settings_for(embedding)
        self.cache = {}
        self.keys = {}
        self.records = {}
        self.dimension = None

    def _vector(self, archive, program):
        text = trajectory_text(archive, program)
        key = hashlib.sha256(json.dumps([self.settings, text], sort_keys=True,
                                       ensure_ascii=False).encode()).hexdigest()
        self.keys[program.id] = key
        if key in self.cache:
            return self.cache[key]
        path = f"reference_embeddings/{key}.json"
        saved = self.store.read(path)
        if saved is None:
            started = time.monotonic()
            self.embedding.last_usage = {}
            try:
                vector = self.embedding(text)
            except Exception as error:
                self.store.append("reference_embedding_calls.jsonl", {
                    "key": key, "program_id": program.id, "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "usage": dict(self.embedding.last_usage),
                    "seconds": time.monotonic() - started,
                })
                raise
            self.store.append("reference_embedding_calls.jsonl", {
                "key": key, "program_id": program.id, "status": "success",
                "usage": dict(self.embedding.last_usage), "seconds": time.monotonic() - started,
            })
        else:
            vector = saved["vector"]
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError("Reference embedding must be a finite nonzero vector")
        if self.dimension is not None and self.dimension != vector.size:
            raise ValueError("Reference embedding dimension changed")
        if saved is None:
            vector = vector / np.max(np.abs(vector))
            vector = vector / np.linalg.norm(vector)
        elif not np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError("Cached reference embedding is not normalized")
        self.dimension = vector.size
        if saved is None:
            self.store.write(path, {"key": key, "program_id": program.id,
                                    "input": text, "vector": vector.tolist()})
        self.cache[key] = vector
        return vector

    def select(self, archive):
        candidates = sources(archive)
        result = dict.fromkeys(p.id for p in candidates)
        self.records = {pid: {"rule": RULE, "reference_id": None} for pid in result}
        if len(candidates) < 2:
            return result
        eligible_ids = set(reference_candidates({p.id: p.evaluation.fitness for p in candidates}))
        eligible = [i for i, p in enumerate(candidates) if p.id in eligible_ids]
        vectors = np.stack([self._vector(archive, p) for p in candidates])
        similarities = vectors @ vectors[eligible].T
        reference_positions = {candidates[i].id: j for j, i in enumerate(eligible)}
        for i, parent in enumerate(candidates):
            if parent.id in reference_positions:
                similarities[i, reference_positions[parent.id]] = np.inf
            j = int(np.argmin(similarities[i]))
            if not np.isfinite(similarities[i, j]):
                continue
            reference = candidates[eligible[j]]
            result[parent.id] = reference.id
            self.records[parent.id] = {
                "rule": RULE, "reference_id": reference.id,
                "cosine_similarity": float(np.clip(similarities[i, j], -1.0, 1.0)),
                "parent_embedding_key": self.keys[parent.id],
                "reference_embedding_key": self.keys[reference.id],
            }
        return result
