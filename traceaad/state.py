"""Program individuals, search progress, and an atomic journal checkpoint."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import random
import time

from .program import Program
from .storage import (
    append_jsonl,
    committed_rows,
    committed_size,
    rows,
    detach_link,
    read_json,
    seal_calls,
    write_json,
)


def _tuple_tree(value):
    return tuple(_tuple_tree(v) for v in value) if isinstance(value, list) else value


@dataclass
class Progress:
    phase: str = "roots"
    generations: int = 0
    init_attempts: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    service_failures: int = 0
    selected_id: int | None = None
    repair_id: int | None = None
    pending_id: int | None = None
    started_at: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    )
    elapsed: float = 0.0

    @classmethod
    def restore(cls, saved, seed):
        progress = cls(**{k: saved[k] for k in cls.__dataclass_fields__ if k in saved})
        progress.rng = {
            name: random.Random(f"{seed}:{name}") for name in ("parent", "action", "reference")
        }
        for name, state in saved.get("rng", {}).items():
            progress.rng[name].setstate(_tuple_tree(state))
        progress._clock = time.monotonic()
        return progress

    @property
    def seconds(self):
        return self.elapsed + time.monotonic() - self._clock

    def snapshot(self):
        return {
            **asdict(self),
            "elapsed": self.seconds,
            "rng": {name: rng.getstate() for name, rng in self.rng.items()},
        }

    def record_calls(self, calls, request, output_limit):
        for call in calls:
            self.model_calls += 1
            call["request_id"] = self.model_calls
            call["stage"] = request["action"].lower()
            usage = call["usage"]
            self.input_tokens += (
                usage.get("prompt_tokens", request["input_tokens"]) or request["input_tokens"]
            )
            self.output_tokens += (
                usage.get("completion_tokens", output_limit if not call["error"] else 0) or 0
            )
            self.service_failures += bool(call["error"])


class Facts:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.checkpoint_path = self.run_dir / "resume.json"
        self.source_path = self.run_dir / "programs.jsonl"
        self.sources = {
            row["id"]: row["code"]
            for row in rows(self.source_path, committed_size(self.run_dir, "programs.jsonl"))
        }
        self.programs = {}
        self.evaluations = []
        checkpoint = read_json(self.checkpoint_path, {})
        self.state = checkpoint.get("state")
        self.files = checkpoint.get("files", {})
        for row in committed_rows(self.run_dir):
            self._apply(row)

    def _apply(self, row):
        if metadata := row.get("program"):
            data = {k: v for k, v in metadata.items() if k in Program.__dataclass_fields__}
            if data["id"] in self.programs:
                self.programs[data["id"]].__dict__.update(data)
            else:
                self.programs[data["id"]] = Program(**data, code=self.sources[data["id"]])
        self.evaluations.extend(row.get("evaluations", []))

    @property
    def valid(self):
        return {pid: p for pid, p in self.programs.items() if p.valid}

    def commit(
        self, state, *, program=None, new=False, evaluations=(), calls=(), delivery_error=None
    ):
        for name, size in self.files.items():
            path = self.run_dir / name
            if path.exists() and path.stat().st_size > size:
                detach_link(path)
                with path.open("r+b") as handle:
                    handle.truncate(size)
        row = {"kind": "candidate" if new else "delivery_failed" if delivery_error else "progress"}
        call_path = self.run_dir / "calls.jsonl"
        if call_path.with_suffix(".jsonl.gz").exists():
            call_path = call_path.with_suffix(".jsonl.gz")
        for call in calls:
            self.files[call_path.name] = append_jsonl(call_path, call)
        if program is not None:
            if new:
                # Every individual stores its source, including identical generated code.
                self.files["programs.jsonl"] = append_jsonl(
                    self.source_path, {"id": program.id, "code": program.code}
                )
            row["program"] = program.metadata()
        if new or delivery_error:
            row.update(
                candidate_id=state["generations"],
                budget_used=state["generations"],
                x_label="候选尝试",
                operator=program.action if program else delivery_error["action"],
                status=program.status if program else "delivery_failed",
                fitness=program.fitness if program and program.valid else None,
                valid=bool(program and program.valid),
                node_id=program.id if program else None,
            )
        if delivery_error:
            row["error"] = delivery_error["error"]
            row["parent_id"] = delivery_error["parent_id"]
            row["call_ids"] = [c["request_id"] for c in calls]
        if evaluations:
            row["evaluations"] = list(evaluations)
        row["progress"] = {
            k: state[k]
            for k in (
                "phase",
                "elapsed",
                "started_at",
                "model_calls",
                "input_tokens",
                "output_tokens",
            )
        }
        row["progress"]["attempts"] = state["generations"]
        row["ts"] = datetime.now().astimezone().isoformat(timespec="seconds")
        self.files["events.jsonl"] = append_jsonl(self.path, row)
        write_json(self.checkpoint_path, {"state": state, "files": self.files})
        if program is not None:
            self.programs[program.id] = program
            self.sources[program.id] = program.code
        self.evaluations.extend(evaluations)
        self.state = state

    def save_summary(self, summary):
        write_json(self.summary_path, summary)
        if summary["status"] in {"finished", "no_valid_root"}:
            seal_calls(self.run_dir)
            self.files = read_json(self.checkpoint_path, {}).get("files", self.files)

    def load_summary(self):
        return read_json(self.summary_path)
