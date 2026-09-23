"""Restore a validated settled prefix from backed-up journals, with exact RNG.

No LLM or evaluator calls. Derive the earlier RNG state from the saved final
state, then replay EVERY intervening scheduling decision using the unchanged
search implementation. Apply only after all eight runs pass validation.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from traceaad.v10_12.traceaad import TraceAADV1012
from traceaad.v10_12.storage import RunStorage, atomic_json
from traceaad.v10_12.tree import Node, SearchTree


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rng_tuple(state):
    return state[0], tuple(state[1]), state[2]


class NoPrompt:
    def count(self, text):
        return 0

    def build(self, *args, **kwargs):
        return ""

    def build_initial(self):
        return ""


def restore_one(source, cutoff):
    final = json.loads((source / "tree_state.json").read_text())
    config = json.loads((source / "run_config.json").read_text())
    events = [e for e in read(source / "events.jsonl")
              if e["candidate_id"] <= final["completed_attempts"]]
    assert [e["candidate_id"] for e in events] == list(range(1, len(events) + 1))
    kept = [e for e in events if e["ts"] < cutoff]
    assert kept and len(kept) < len(events)
    last = kept[-1]
    tail = events[len(kept):]
    nodes = read(source / "nodes.jsonl")
    by_id = {n["id"]: n for n in nodes}
    initial_nodes = [n for n in nodes if n["evaluation_id"] <= last["budget_used"]]
    counts = Counter(e["parent_id"] for e in kept
                     if e.get("selection", {}).get("parent_count_before") is not None)
    # This engine uses only Random.random/choices; each consumes one random()
    # per selection. Repairs consume none. Tail populations all have >=2 cards.
    assert all(e["parent_id"] is not None for e in tail)
    tail_draws = sum(2 + int(e["donor_id"] is not None) + 2
                     for e in tail if e["repair_of"] is None)
    rng = random.Random(config["seed"])
    target_final = rng_tuple(final["rng_state"])
    final_draws = None
    for i in range(100000):
        if rng.getstate() == target_final:
            final_draws = i
            break
        rng.random()
    assert final_draws is not None and final_draws >= tail_draws
    rng = random.Random(config["seed"])
    for _ in range(final_draws - tail_draws):
        rng.random()
    recovered_rng = rng.getstate()
    engine = TraceAADV1012.__new__(TraceAADV1012)
    engine.rng = rng
    engine.tree = SearchTree()
    for node in initial_nodes:
        engine.tree.add_raw(Node(**node))
    engine.parent_selection_counts = dict(counts)
    engine.n_roots = final["mechanism"]["n_roots"]
    engine.n_profile_cards = final["mechanism"]["n_profile_cards"]
    engine.profile_card_tau = final["mechanism"]["profile_card_tau"]
    assert engine.n_profile_cards == 2
    engine.completed_attempts = last["candidate_id"]
    engine.storage = RunStorage(source)
    engine.storage.last_event = last
    engine.builder = NoPrompt()
    engine.task_contract = "Scheduling replay only; no prompt is submitted."
    fields = ("candidate_id", "operator", "requested_operator", "parent_id",
              "donor_id", "reference_ids", "repair_of")
    for event in tail:
        actual = engine._schedule()
        for key in fields:
            assert getattr(actual, key) == event.get(key), (
                source.name, event["candidate_id"], key, getattr(actual, key), event.get(key))
        if actual.selection:
            assert actual.selection["parent_count_before"] == event["selection"]["parent_count_before"]
            assert abs(actual.selection["parent_probability"] - event["selection"]["parent_probability"]) < 1e-12
        if event["node_id"] is not None:
            engine.tree.add_raw(Node(**by_id[event["node_id"]]))
        engine.completed_attempts = event["candidate_id"]
        engine.storage.last_event = event
    assert engine.rng.getstate() == target_final
    assert engine.parent_selection_counts == {int(k): v for k, v in final["parent_selection_counts"].items()}
    assert [asdict(n) for n in engine.tree.all_nodes()] == final["nodes"]
    invalid_streak = 0
    for event in reversed(kept):
        if event["status"] != "invalid_output":
            break
        invalid_streak += 1
    restored = dict(final, rng_state=list(recovered_rng), nodes=initial_nodes,
                    parent_selection_counts=dict(counts),
                    step_counter=sum(e["parent_id"] is not None for e in kept),
                    budget_used=last["budget_used"], completed_attempts=last["candidate_id"],
                    invalid_streak=invalid_streak, last_event=last)
    prefixes = {}
    for name in ("events.jsonl", "nodes.jsonl", "llm_calls.jsonl"):
        data = (source / name).read_bytes().splitlines(keepends=True)
        limit_key = "evaluation_id" if name == "nodes.jsonl" else "candidate_id"
        limit = last["budget_used"] if name == "nodes.jsonl" else last["candidate_id"]
        selected = [line for line in data if json.loads(line)[limit_key] <= limit]
        assert data[:len(selected)] == selected, (source, name, "not a contiguous prefix")
        prefixes[name] = b"".join(selected)
    receipt = dict(task=config["task"], repeat=config["repeat"], run_name=source.name,
                   original_budget=final["budget_used"], restored_budget=restored["budget_used"],
                   discarded_calls=final["budget_used"]-restored["budget_used"],
                   last_kept_candidate=last["candidate_id"], last_kept_time=last["ts"],
                   replayed_candidates=len(tail), final_rng_draws=final_draws,
                   restored_rng_draws=final_draws-tail_draws,
                   rng_and_all_tail_selections_verified=True,
                   best_fitness=max(n["fitness"] for n in initial_nodes),
                   prefix_sha256={name: hashlib.sha256(data).hexdigest() for name, data in prefixes.items()})
    return restored, prefixes, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-09-22T15:54:00")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    sources = sorted(args.backup.glob("*/*/tree_state.json"))
    assert len(sources) == 8
    staged = []
    for state in sources:
        restored, prefixes, receipt = restore_one(state.parent, args.cutoff)
        destination = args.stage / receipt["task"] / receipt["run_name"]
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / "tree_state.json", restored)
        for name, data in prefixes.items():
            (destination / name).write_bytes(data)
        staged.append((destination, receipt))
        print(receipt["run_name"], receipt["original_budget"], "->", receipt["restored_budget"],
              "verified tail", receipt["replayed_candidates"], flush=True)
    if args.apply:
        results = ROOT / "experiments/traceaad_v10_12/results"
        manifest = json.loads((results / "batch_20260921_v1012.json").read_text())
        rows = {row["run_name"]: row for row in manifest["plan"]}
        assert all(rows[r["run_name"]]["status"] == "paused" for _, r in staged)
        for _, receipt in staged:
            row = rows[receipt["run_name"]]
            assert subprocess.run(
                ["tmux", "has-session", "-t", "=" + row["session"]],
                capture_output=True).returncode != 0
            destination = results / receipt["task"] / receipt["run_name"]
            source = args.backup / receipt["task"] / receipt["run_name"]
            for name in ("events.jsonl", "nodes.jsonl", "llm_calls.jsonl", "tree_state.json"):
                assert (destination / name).read_bytes() == (source / name).read_bytes(), (
                    receipt["run_name"], name, "live run changed after backup")
        for stage, receipt in staged:
            destination = results / receipt["task"] / receipt["run_name"]
            for name in ("events.jsonl", "nodes.jsonl", "llm_calls.jsonl", "tree_state.json"):
                temporary = destination / (name + ".recovery.tmp")
                temporary.write_bytes((stage / name).read_bytes())
                temporary.replace(destination / name)
            # The pre-recovery final summary remains in the complete backup.
            (destination / "logs/run_summary.json").unlink(missing_ok=True)
            with (destination / "tmux_run.log").open("a") as log:
                log.write(f"\nAUTHORIZED RECOVERY {datetime.now().isoformat()}: "
                          f"restored E{receipt['restored_budget']} before {args.cutoff}; "
                          f"original records preserved in {args.backup}\n")
    atomic_json(args.stage / "recovery_receipt.json", {
        "created_at": datetime.now().astimezone().isoformat(), "cutoff_exclusive": args.cutoff,
        "backup": str(args.backup), "applied": args.apply,
        "rng_method": "Locate final seeded Random state; rewind by known tail draw count; "
                      "replay every tail selection and verify final RNG, nodes and parent counters.",
        "runs": [r for _, r in staged],
    })


if __name__ == "__main__":
    main()
