"""Program individuals, search progress, and an atomic journal checkpoint."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .program import Program
from .storage import (
    Programs,
    append_jsonl,
    committed_rows,
    detach_link,
    read_json,
    seal_calls,
    write_json,
)

INDIVIDUAL_FORMAT = "program-individual-v1"


@dataclass
class Progress:
    phase: str = "roots"
    generations: int = 0
    init_attempts: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    service_failures: int = 0
    finalists: list[int] = field(default_factory=list)
    selection_results: list[dict] = field(default_factory=list)
    selected_id: int | None = None
    repair_id: int | None = None
    pending_id: int | None = None
    started_at: str = field(
        default_factory=lambda: (
            datetime.now().astimezone().isoformat(timespec="seconds")
        )
    )
    elapsed: float = 0.0


class Facts:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.checkpoint_path = self.run_dir / "resume.json"
        self.sources = Programs(run_dir)
        self.programs, self.explorations = {}, {}
        self.evaluations = []
        checkpoint = read_json(self.checkpoint_path, {})
        self.state = checkpoint.get("state")
        self.files = checkpoint.get("files", {})
        if (
            self.path.exists()
            and self.path.stat().st_size
            and (
                not self.state
                or self.state.get("individual_format") != INDIVIDUAL_FORMAT
            )
        ):
            raise ValueError(
                "this run uses the old attempt/program format; start a new V12.1 run"
            )
        for row in committed_rows(self.run_dir):
            self._apply(row)

    def _apply(self, row):
        if metadata := row.get("program"):
            data = {
                k: v for k, v in metadata.items() if k in Program.__dataclass_fields__
            }
            if data["id"] in self.programs:
                self.programs[data["id"]].__dict__.update(data)
            else:
                self.programs[data["id"]] = Program(
                    **data, code=self.sources.get(metadata["key"])
                )
        self.evaluations.extend(row.get("evaluations", []))
        if exploration := row.get("exploration"):
            self.explorations[exploration["id"]] = exploration

    @property
    def valid(self):
        return {pid: p for pid, p in self.programs.items() if p.valid}

    def commit(
        self,
        state,
        *,
        program=None,
        updated_program=None,
        evaluations=(),
        calls=(),
        exploration=None,
        delivery_error=None,
    ):
        for name, size in self.files.items():
            path = self.run_dir / name
            if path.exists() and path.stat().st_size > size:
                detach_link(path)
                with path.open("r+b") as handle:
                    handle.truncate(size)
        row = {
            "kind": "candidate"
            if program
            else "delivery_failed"
            if delivery_error
            else "progress"
        }
        call_path = self.run_dir / "calls.jsonl"
        if call_path.with_suffix(".jsonl.gz").exists():
            call_path = call_path.with_suffix(".jsonl.gz")
        for call in calls:
            self.files[call_path.name] = append_jsonl(call_path, call)
        individual = program or updated_program
        if individual is not None:
            if program:
                # Every individual stores its source, including identical generated code.
                self.files["programs.jsonl"] = append_jsonl(
                    self.sources.path,
                    {"id": program.id, "key": program.key, "code": program.code},
                )
            row["program"] = {
                **{k: v for k, v in asdict(individual).items() if k != "code"},
                "key": individual.key,
                "score": individual.score,
                "valid": individual.valid,
                "repaired": individual.repaired,
            }
        if program or delivery_error:
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
        if exploration:
            row["exploration"] = exploration
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
        self._apply(row)
        self.state = state

    def save_summary(self, summary):
        write_json(self.summary_path, summary)
        if summary["status"] in {
            "finished",
            "search_complete",
            "selection_failed",
            "no_valid_root",
        }:
            seal_calls(self.run_dir)
            self.files = read_json(self.checkpoint_path, {}).get("files", self.files)

    def load_summary(self):
        return read_json(self.summary_path)
