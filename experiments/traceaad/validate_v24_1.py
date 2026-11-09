"""Replay real archives for packing, or compare V24/V24.1 reflection on frozen windows.

Packing calls only the serving tokenizer. Live reflection requires --live and the
shared scheduler; it never evaluates a candidate or updates the source archive.
"""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from zoneinfo import ZoneInfo

from core.llm import OpenAIAPI
from traceaad_base import Config, Task
from traceaad_base.models import Attempt, Program
from traceaad_base.storage import write_json
from traceaad_v24 import TraceAAD as V24
from traceaad_v24.notes import REFLECTION_FORMAT as V24_FORMAT
from traceaad_v24_1 import TraceAAD as V241
from traceaad_v24_1.evidence import observe
from traceaad_v24_1.notes import REFLECTION_FORMAT as V241_FORMAT

from .scheduling import ScheduledModel, api_key, load_environment


class CounterCache:
    """Persist exact chat counts, with the model and template bound to the cache."""

    def __init__(self, url, model, path):
        self.path = path
        self.settings = {"url": url, "model": model, "enable_thinking": False}
        self.counts = {}
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["settings"] != self.settings:
                raise ValueError("Tokenizer cache settings changed")
            self.counts = saved["counts"]
        self.client = OpenAIAPI(base_url=url, model=model, api_key=api_key(url), timeout=10,
                                extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    def __call__(self, prompt):
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key not in self.counts:
            self.counts[key] = self.client.count_prompt_tokens(prompt)
        return self.counts[key]

    def save(self):
        write_json(self.path, {"settings": self.settings, "counts": self.counts})


class NoGeneration:
    def complete(self, *_args, **_settings):
        raise AssertionError("This probe replays generation; it must never request a candidate")


class StubReflection:
    def __init__(self, counter):
        self.count_prompt_tokens = counter
        self.requests = 0

    def complete(self, prompt, **_settings):
        self.requests += 1
        new = json.loads(re.search(r"New events: (\[[^\n]*\])", prompt)[1])
        active = re.search(r"\[Current Experience\]\n(.*?)\n\n\[Trajectory Evidence\]", prompt, re.S)[1]
        notes = [] if active.startswith("No experience") else json.loads(active)
        active_count = int(re.search(r"Active entries: (\d+)", prompt)[1])
        if active_count >= 6 and not notes:
            return json.dumps({"analysis": "No related entry or free slot in this focused packet.",
                               "updates": [], "archive_ids": []})
        target = notes[0]["id"] if notes else None
        update = {"id": target, "idea": "Replay-only computation",
                  "when": "A fixed simulated condition used only to stress material capacity.",
                  "lesson": "This is a stub update, not a learned algorithmic claim.",
                  "evidence_ids": new[:1]}
        return json.dumps({"analysis": "Packing stress probe.", "updates": [update], "archive_ids": []})


def read_run(directory):
    native = directory / "native"
    config = json.loads((native / "run_config.json").read_text())
    rows = [json.loads(line) for line in (native / "attempts.jsonl").read_bytes().split(b"\n")[:-1]]
    spec = config["task"]
    task = Task(spec["task_description"], spec["template_program"], lambda *_: None, spec["time_limit"],
                entrypoint=spec["entrypoint"], example_code=spec["example_code"],
                example_design=spec["example_design"], score_definition=spec["score_definition"],
                evaluation_seed=spec["evaluation_seed"], executes_source=spec["executes_source"],
                metadata=spec["metadata"])
    return task, Config(**config["config"]), rows


def replay(directory, counter, output):
    task, config, rows = read_run(directory)
    reflector = StubReflection(counter)
    run = V241(task, NoGeneration(), config=config, reflection_model=reflector)
    for row in rows:
        attempt = Attempt(**deepcopy(row["attempt"]))
        attempt.record.pop("reflection", None)
        if attempt.operator != "Init":
            attempt.record["best_before"] = run.best.evaluation.fitness
        if row["program"] is not None:
            program = Program.from_dict(row["program"])
            run.archive[program.id] = program
        run.attempts.append(attempt)
        run.finish(attempt)
    cycles = [(a.id, a.record["reflection"]) for a in run.attempts if "reflection" in a.record]
    successful = [(i, r) for i, r in cycles if r["status"] == "success"]
    lags = [i - statistics.mean(r["reviewed_event_ids"]) for i, r in successful]
    result = {
        "source": str(directory), "source_attempts_sha256": hashlib.sha256(
            (directory / "native/attempts.jsonl").read_bytes()).hexdigest(),
        "cycles": len(cycles), "requests": reflector.requests,
        "scope": "One computational question per cadence; full event coverage is not the objective.",
        "focused_questions": [r["focus"] for _, r in successful],
        "statuses": dict(Counter(r["status"] + ":" + r.get("reason", "") for _, r in cycles)),
        "read_total": sum(len(r["reviewed_event_ids"]) for _, r in successful),
        "new_events": sum(a.operator != "Init" and bool(a.evaluations) for a in run.attempts),
        "final_unreviewed": list(run.unreviewed_events),
        "last_success_attempt": successful[-1][0] if successful else None,
        "input_tokens_median": statistics.median(r["input_size"] for _, r in successful) if successful else None,
        "input_tokens_max": max((r["input_size"] for _, r in successful), default=None),
        "lag_median": statistics.median(lags) if lags else None,
        "lag_max": max(lags, default=None), "reflection_settings": run.reflection_settings,
    }
    write_json(output / "summary.json", result)
    write_json(output / "cycles.json", [{"at": i, **{k: v for k, v in r.items() if k != "snapshot"}}
                                       for i, r in cycles])
    counter.save()
    return result


def frozen_window(run, rows, cutoff):
    eligible = []
    for row in rows:
        if row["attempt"]["id"] > cutoff:
            break
        attempt = Attempt(**deepcopy(row["attempt"]))
        attempt.record.pop("reflection", None)
        if attempt.operator != "Init":
            attempt.record["best_before"] = run.best.evaluation.fitness
        if row["program"] is not None:
            program = Program.from_dict(row["program"])
            run.archive[program.id] = program
        run.attempts.append(attempt)
        event = observe(attempt, run.archive)
        if hasattr(run, "events") and event is not None:
            run.events[event.id] = event
        if attempt.operator != "Init" and event is not None:
            eligible.append(attempt.id)
    if len(eligible) < 10:
        raise ValueError("A live window needs ten evaluated non-initialization attempts")
    return eligible[-10:]


def live_case(directory, cutoff, scheduler, counter, output, repeats, versions):
    task, config, rows = read_run(directory)
    results = []
    model = ScheduledModel(scheduler, "v24_1_validation")
    for repeat in range(repeats):
        for version, method, schema in (("v24", V24, V24_FORMAT), ("v24_1", V241, V241_FORMAT)):
            if version not in versions:
                continue
            path = output / f"{version}_r{repeat + 1}"
            if path.exists():
                raise FileExistsError(path)
            role = model.for_role(schema)
            options = {"reflection_counter": counter} if version == "v24_1" else {}
            run = method(task, NoGeneration(), config=(replace(config, max_tokens=4096)
                         if version == "v24" else config), reflection_model=role, run_dir=path, **options)
            ids = frozen_window(run, rows, cutoff)
            if version == "v24":
                run.pending = ids
            else:
                run.unreviewed_events = ids
            attempt = run.attempts[-1]
            started = time.monotonic()
            run._reflect(attempt)
            record = attempt.record["reflection"]
            write_json(path / "reflection.json", record)
            calls = run.store.rows("calls.jsonl")
            result = {"version": version, "repeat": repeat + 1, "source": str(directory),
                      "cutoff": cutoff, "window": ids, "status": record["status"],
                      "seconds": time.monotonic() - started, "analysis": record.get("analysis"),
                      "notes": run.notes if version == "v24" else run.experience.active,
                      "shown_event_ids": record.get("shown_event_ids", ids),
                      "usage": calls[-1].get("usage", {}) if calls else {},
                      "input_tokens_counted": counter(calls[-1]["prompt"]) if calls else None,
                      "error": record.get("error"), "reason": record.get("reason")}
            write_json(path / "summary.json", result)
            results.append(result)
            counter.save()
            print(json.dumps({"case": output.name, "version": version, "repeat": repeat + 1,
                              "status": result["status"], "usage": result["usage"]}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-url", default="http://222.201.145.6:8000/v1")
    parser.add_argument("--tokenizer-model", default="qwen3.8-27b-awq")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--pack-root", type=Path)
    parser.add_argument("--case", action="append", default=[], help="RUN_DIR:ATTEMPT_CUTOFF")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--versions", nargs="+", choices=("v24", "v24_1"), default=("v24", "v24_1"))
    parser.add_argument("--scheduler-socket", default="/tmp/traceaad-1003/scheduler.sock")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.case and not args.live:
        parser.error("Model reflection requires --live")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    load_environment(args.env_file)
    counter = CounterCache(args.tokenizer_url, args.tokenizer_model, args.output / "token_counts.json")
    result = {"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "tokenizer": counter.settings, "packing": [], "reflection": [],
              "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in ("traceaad_v24_1", "traceaad_v24") for p in Path(folder).glob("*.py")}}
    try:
        if args.pack_root:
            for task in ("tsp_construct", "cvrp_aco", "fssp_gls", "graph_colouring", "jssp_construct", "op_aco"):
                sources = [p for p in (args.pack_root / task).iterdir()
                           if p.name.endswith("_rep1") and (p / "native/run_config.json").exists()]
                if len(sources) != 1:
                    raise ValueError(f"Expected one rep1 archive for {task}: {sources}")
                packed = replay(sources[0], counter, args.output / task)
                result["packing"].append(packed)
                write_json(args.output / "summary.json", result)
                print(json.dumps({"task": task, "cycles": packed["cycles"], "requests": packed["requests"],
                                  "unreviewed": len(packed["final_unreviewed"])}), flush=True)
        for index, case in enumerate(args.case):
            directory, cutoff = case.rsplit(":", 1)
            result["reflection"].extend(live_case(Path(directory), int(cutoff), args.scheduler_socket,
                                                 counter, args.output / f"case{index + 1}", args.repeats,
                                                 args.versions))
            write_json(args.output / "summary.json", result)
    finally:
        counter.save()
        counter.client.close()
        write_json(args.output / "summary.json", result)


if __name__ == "__main__":
    main()
