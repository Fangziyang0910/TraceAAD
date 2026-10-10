"""V10.23 with measured request values guiding half of ordinary action choices."""

import experiments  # noqa: F401
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import random
import time
from urllib.request import ProxyHandler, Request, build_opener

from experiments.infra.search_run import main as run_search
from traceaad.common.selection import choose_reference
from traceaad.common.config import REVISION
from traceaad.common.storage import append_jsonl
from traceaad.v10_23 import Config as BaseConfig, TraceAADV1023
from .data import ACTIONS, request_state
from .collect import prompt_identity


def check_model_conditions(metadata):
    if metadata["prompt_policy"] != "v1023" or metadata["revision"] != REVISION:
        raise ValueError("decision model generation or evaluation conditions differ")
    if metadata["prompt_sources_sha256"] != prompt_identity("v1023"):
        raise ValueError("decision model prompt sources differ from the running generator")


@dataclass
class Config(BaseConfig):
    decision_url: str = field(default_factory=lambda: os.environ.get("TRACEAAD_DECISION_URL", ""))
    decision_token_file: str = field(default_factory=lambda: os.environ.get("TRACEAAD_DECISION_TOKEN_FILE", ""))
    decision_fraction: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if not self.decision_url or not self.decision_token_file or not 0 <= self.decision_fraction <= 1:
            raise ValueError("configure decision URL/token file and a mixture in [0,1]")


class TraceAADDecision(TraceAADV1023):
    METHOD = "decision_v1"
    Config = Config

    def _choose_parent(self, eligible):
        selected = getattr(self, "_decision_parent", None)
        return selected if selected is not None else super()._choose_parent(eligible)

    def _api(self, path, value=None):
        token = Path(self.config.decision_token_file).read_text().strip()
        headers = {"Authorization": "Bearer " + token, "Content-Type": "application/json"}
        body = json.dumps(value).encode() if value is not None else None
        request = Request(self.config.decision_url.rstrip("/") + path, data=body, headers=headers)
        with build_opener(ProxyHandler({})).open(request, timeout=120) as response:
            return json.load(response)

    def _action_from_model(self, parent):
        metadata = getattr(self, "_decision_metadata", None)
        if metadata is None:
            metadata = self._api("/metadata")
            self._decision_metadata = metadata
        check_model_conditions(metadata)
        horizon = metadata["decision_horizon"]
        if horizon not in (1, 2):
            raise ValueError("ordinary selection requires a first-candidate or automatic-repair model")
        # Prediction and generation must see the same reference without drawing twice.
        rng = random.Random()
        rng.setstate(self.reference_rng.getstate())
        reference, _ = choose_reference(parent, self.archive, rng)
        actions = [a for a in ACTIONS if a != "Crossover" or reference is not None]
        states = [request_state(self.task, self.prompts.common[0], self.template, self.programs,
            self.attempts_table, parent["id"], reference["id"] if reference else None,
            action, self.attempts, self.config.budget, horizon=horizon,
            score_scale=metadata["score_scales"][self.task], prompt_policy="v1023")
            for action in actions]
        choices = self._api("/score", {"records": states})["choices"]
        if {c["action"] for c in choices} != set(actions) or any(
                not math.isfinite(c["expected_gain"]) or c["expected_gain"] < 0 for c in choices):
            raise ValueError("invalid outcome model response")
        if any(not math.isfinite(c["gain_per_candidate"]) or c["gain_per_candidate"] < 0 or
               not 1 <= c["expected_candidate_cost"] <= horizon for c in choices):
            raise ValueError("invalid request cost or value")
        best = max(c["gain_per_candidate"] for c in choices)
        return self.action_rng.choice([c["action"] for c in choices if c["gain_per_candidate"] == best]), choices

    def _ordinary_search(self, sampled=None):
        if sampled is not None:
            return super()._ordinary_search(sampled)
        eligible = [p for p in self.archive.values() if p["id"] not in self.progress.too_long]
        if not eligible:
            return super()._ordinary_search()
        self._decision_parent = super()._choose_parent(eligible)
        parent = self._decision_parent[0]
        started = time.monotonic()
        record = {"before_attempt": self.attempts + 1, "parent_id": parent["id"], "guided": False}
        try:
            if self.config.decision_fraction > 0 and self.action_rng.random() < self.config.decision_fraction:
                try:
                    sampled, choices = self._action_from_model(parent)
                    record.update(guided=True, choices=choices)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    record["fallback_error"] = str(exc)
            if sampled is None:
                sampled = self.action_rng.choices(list(self.config.operators), list(self.config.operators.values()))[0]
            record.update(selected=sampled, decision_seconds=time.monotonic() - started)
            append_jsonl(self.run_dir / "decisions.jsonl", record)
            return super()._ordinary_search(sampled)
        finally:
            self._decision_parent = None


def main(argv=None):
    return run_search(TraceAADDecision, Config, "traceaad_decision_v1",
                      "V10.23 with request value guidance; unchanged generation and evaluation", argv)


if __name__ == "__main__":
    main()
