"""Convert historical evidence once; ordinary result readers use only the new format."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

from traceaad.common.storage import Programs, RESULT_FORMAT as CURRENT_RESULT_FORMAT, append_jsonl, read_json, rows, write_json
from .legacy_facts import Facts as LegacyFacts
from .legacy_history import TrainingHistory as LegacyHistory
from .legacy_results import load_batch_heldout, load_selection, selection_identity

RESULT_FORMAT = "traceaad-results-v1"  # intermediate layout of the original maximized evidence

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments_result"
ARCHIVE = RESULTS / ".archive/storage_20261006"
TASK_MIN = {"tsp_construct", "cvrp_aco", "online_bin_packing", "vrptw_construct"}
CANONICAL = {"events.jsonl", "programs.jsonl", "calls.jsonl", "calls.jsonl.gz", "resume.json",
             "summary.json", "heldout.json", "run_config.json", "selection.json", "best_program.py",
             "diagnostics.json", ".cache", ".conversion", ".convert"}
PROGRAM_FIELDS = {"id", "key", "fitness", "score", "valid", "failure", "action", "idea",
                  "parent_id", "depth", "reference_id", "attempt_id", "repaired", "calls",
                  "function_seconds", "eval_seconds"}


def read_old(path):
    try:
        return read_json(path, {})
    except (OSError, ValueError):
        return {}


def archive_run(run_dir, receipt):
    """Keep original bytes in a compressed cold archive, outside the working layout."""
    relative = run_dir.relative_to(RESULTS)
    output = ARCHIVE / relative.parent / (relative.name + ".tar.gz")
    if output.exists():
        return str(output.relative_to(RESULTS))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        for item in run_dir.iterdir():
            if item.name in {".cache", ".convert", ".conversion"}:
                continue
            if item.name in CANONICAL and item.name not in receipt["original_names"]:
                continue
            if item.name == "run_config.json":
                archive.add(run_dir / ".conversion/config.json", arcname=item.name)
            elif item.name == "selection.json" and (run_dir / ".conversion/selection.json").exists():
                archive.add(run_dir / ".conversion/selection.json", arcname=item.name)
            elif item.name == "logs":
                archive.add(item, arcname="logs", filter=lambda info: None if
                            info.name.endswith("monitor_cache.json") else info)
            else:
                archive.add(item, arcname=item.name)
    temporary.replace(output)
    return str(output.relative_to(RESULTS))


def program_meta(program, sources, identifier, fitness, valid, minimize):
    code = program.get("code") or program.get("program") or ""
    if not code:
        return None
    source_key = sources.add(code)
    score = (-fitness if minimize else fitness) if fitness is not None else None
    return {**{k: v for k, v in program.items() if k in PROGRAM_FIELDS},
            "id": identifier, "key": source_key, "fitness": fitness, "score": score,
            "valid": valid, "failure": program.get("failure"),
            "idea": program.get("idea", program.get("algorithm", "")),
            "action": program.get("action", program.get("operator", program.get("origin_operator", program.get("created_by", "unknown")))),
            "reference_id": program.get("reference_id", program.get("donor_id")),
            "parent_id": program.get("parent_id"), "depth": program.get("depth", 0)}


def active_names():
    names = set()
    for process in Path('/proc').glob('[0-9]*'):
        try:
            args = (process / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if '-m' in args and '--run-name' in args and any(
                part.startswith('experiments.traceaad_') for part in args):
            names.add(args[args.index('--run-name') + 1])
    return names


def convert_run(run_dir, heldout=None, active=False):
    run_dir = Path(run_dir).resolve()
    if active:
        raise ValueError("stop or finish the legacy writer before importing minimized results")
    current = read_json(run_dir / "run_config.json", {})
    if current.get("score_direction") == "min" and current.get("result_format") != CURRENT_RESULT_FORMAT:
        raise ValueError("restore the original maximized archive before legacy import")
    work = run_dir / ".convert"
    stash = run_dir / ".conversion"
    stash.mkdir(exist_ok=True)
    receipt_path = stash / "receipt.json"
    receipt = read_json(receipt_path, {})
    if receipt.get("sealed"):
        return receipt
    if not receipt:
        receipt = {"original_names": [p.name for p in run_dir.iterdir() if not p.name.startswith(".")],
                   "run": str(run_dir.relative_to(RESULTS))}
        shutil.copy2(run_dir / "run_config.json", stash / "config.json")
        if (run_dir / "selection.json").exists():
            shutil.copy2(run_dir / "selection.json", stash / "selection.json")
    # A live old search creates its selection and export only at the end.
    if (run_dir / "selection.json").exists() and not (stash / "selection.json").exists():
        shutil.copy2(run_dir / "selection.json", stash / "selection.json")
    receipt["original_names"] = sorted(set(receipt["original_names"]) | {
        p.name for p in run_dir.iterdir() if not p.name.startswith(".") and
        (p.name not in CANONICAL or p.name in {"selection.json", "best_program.py"})})
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    config = read_json(stash / "config.json")
    task = config.get("task", run_dir.parent.name)
    minimize = task in TASK_MIN
    history = LegacyHistory(run_dir, minimize)
    curve = history.read()[0]
    progress = history.progress_snapshot()
    history_available = bool(progress)
    if not history_available:
        # Some remote evaluation inputs retain only the final summary/source.
        # Import those facts without inventing a candidate trajectory.
        progress = {"budget_used": 0, "x_label": config.get("budget_axis", "候选历史缺失"),
                    "valid_nodes": 0, "candidate_count": 0, "valid_candidate_count": 0}
    clock = history.timing_snapshot()
    source_path = Path(history.identity[0]) if history.identity[0] != "samples" else None
    source_mtime = source_path.stat().st_mtime_ns if source_path and source_path.exists() else max(
        (p.stat().st_mtime_ns for p in run_dir.glob("logs/samples/samples_*.json")), default=0)
    receipt["source_mtime_ns"] = source_mtime
    source = progress.get("source")
    events = sorted(history.streams.get(source, []), key=lambda e: e["evaluation"])
    native = config.get("method") in {"v1015", "v1016", "v1017", "v1018", "v1019"}
    facts = LegacyFacts(run_dir) if native else None
    sources = Programs(work)
    programs = {}
    metadata_by_candidate = {}
    evaluation_by_key = {}
    if facts:
        for evaluation in facts.evaluations:
            if evaluation["role"] == "search":
                evaluation_by_key.setdefault(evaluation.get("key"), []).append(evaluation)
        for pid, p in facts.programs.items():
            metadata = program_meta(p, sources, pid, p.get("fitness"), p["valid"], minimize)
            if metadata:
                programs[pid] = metadata
    elif (run_dir / "search.jsonl").exists():
        # Earlier trees kept all sources in a final snapshot rather than node events.
        with (run_dir / "search.jsonl").open("rb") as handle:
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break
                if not any(name in raw[-200:] for name in (b'tree_state.json', b'checkpoints/latest.json')):
                    continue
                record = json.loads(raw)
                data = record.get("data") or {}
                tree = data.get("tree", data) if isinstance(data, dict) else {}
                nodes = tree.get("nodes", tree.get("algorithms", []))
                for node in nodes.values() if isinstance(nodes, dict) else nodes:
                    fitness = node.get("fitness")
                    metadata = program_meta(node, sources, node["id"], fitness, fitness is not None, minimize)
                    if metadata:
                        programs[node["id"]] = metadata
    emitted = set()
    for position, event in enumerate(events, 1):
        aid = event.get("candidate")
        aid = position if aid is None else aid
        if isinstance(aid, str) and aid.isdigit():
            aid = int(aid)
        attempt = dict(facts.attempts.get(aid, {})) if facts else {}
        pid = attempt.get("program_id") if facts else event.get("node_id")
        if isinstance(pid, str) and pid.isdigit():
            pid = int(pid)
        if pid is None and aid in programs:
            pid = aid
        metadata = programs.get(pid)
        if metadata is None:
            p = history.node(aid)
            if not p or not (p.get("code") or p.get("program")):
                p = history.node(event.get("node_id"), by_node=True)
            if p:
                pid = event.get("node_id") if event.get("node_id") is not None else aid
                metadata = program_meta(p, sources, pid, event["fitness"], event["valid"], minimize)
                if metadata:
                    programs[pid] = metadata
        attempt.update(id=aid, action=attempt.get("action", event["operator"]),
                       status=event["status"], program_id=pid,
                       idea=attempt.get("idea", event.get("idea", "")),
                       parent_id=attempt.get("parent_id", event.get("parent_id")),
                       repair_of=attempt.get("repair_of", event.get("repair_of")))
        for k in ("code", "raw_code", "completed_code", "calls", "prompt"):
            attempt.pop(k, None)
        attempt["call_ids"] = [attempt["request_id"]] if attempt.get("request_id") is not None else []
        row = {"kind": "candidate", "candidate_id": aid, "budget_used": event["evaluation"],
               "x_label": progress["x_label"], "operator": event["operator"],
               "status": event["status"], "fitness": event["fitness"], "valid": event["valid"],
               "node_id": pid, "attempt": attempt, "ts": event.get("created_at")}
        if metadata and pid not in emitted:
            row["program"] = metadata
            if facts:
                p = facts.programs.get(pid)
                row["evaluations"] = evaluation_by_key.get(p.get("key"), []) if p else []
            emitted.add(pid)
        append_jsonl(work / "events.jsonl", row)
        metadata_by_candidate[str(aid)] = metadata
    for pid, metadata in programs.items():
        if pid not in emitted:
            append_jsonl(work / "events.jsonl", {"kind": "program", "program": metadata})
    if facts:
        for exploration in facts.explorations.values():
            append_jsonl(work / "events.jsonl", {"kind": "exploration", "exploration": exploration})
        selection_evaluations = [e for e in facts.evaluations if e["role"] != "search"]
        if selection_evaluations:
            append_jsonl(work / "events.jsonl", {"kind": "evaluation", "evaluations": selection_evaluations})
    # Raw requests and replies are preserved independently of the search projection.
    call_count = 0
    journal = run_dir / "search.jsonl"
    if journal.exists():
        with journal.open("rb") as handle, gzip.open(work / "calls.jsonl.gz", "wb", compresslevel=1) as output:
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break
                if raw.startswith((b'{"kind":"state",', b'{"kind":"node",', b'{"kind":"artifact",',
                                   b'{"kind":"evaluation",', b'{"kind":"attempt",', b'{"kind":"request",')):
                    continue
                if (b'"call"' not in raw[:120] and b'llm_calls.jsonl' not in raw[-200:]
                        and b'decisions.jsonl' not in raw[-200:] and b'events.jsonl' not in raw[-200:]):
                    continue
                record = json.loads(raw)
                data = record.get("data") if record.get("source") else record
                if not isinstance(data, dict) or not any(k in data for k in ("response", "prompt", "exact_prompt", "exact_response")):
                    continue
                output.write((json.dumps({**{k: v for k, v in data.items() if k not in {
                    "kind", "source", "exact_prompt", "exact_response", "llm_output", "current_code"}},
                    "request_id": data.get("request_id", data.get("call_id", call_count + 1)),
                    "prompt": data.get("prompt", data.get("exact_prompt", "")),
                    "response": data.get("response", data.get("exact_response", data.get("llm_output", "")))}, ensure_ascii=False,
                    separators=(",", ":")) + "\n").encode())
                call_count += 1
    else:
        call_file = run_dir / "logs/llm_calls.jsonl"
        if call_file.exists():
            with call_file.open("rb") as src, gzip.open(work / "calls.jsonl.gz", "wb", compresslevel=1) as dst:
                for raw in src:
                    if raw.endswith(b"\n"):
                        dst.write(raw)
                        call_count += 1
    original = read_old(run_dir / "logs/run_summary.json")
    exported = run_dir / "best_program.py"
    best_old = original.get("best") if isinstance(original.get("best"), dict) else {}
    selected = None
    if original.get("status") == "finished":
        old_id = best_old.get("id", best_old.get("node_id"))
        selected = programs.get(old_id)
        if selected is None and best_old.get("code"):
            selected = program_meta(best_old, sources, old_id if old_id is not None else "selected",
                                    best_old["fitness"], True, minimize)
        if selected is None:
            order = original.get("best_sample_order", original.get("best_node_id", original.get("best_algorithm_id")))
            selected = metadata_by_candidate.get(str(order))
        if selected is None and exported.exists():
            code = exported.read_text(encoding="utf-8")
            fitness = best_old.get("fitness", original.get("best_score"))
            selected = program_meta({"code": code, "idea": best_old.get("idea", "")}, sources,
                                    old_id if old_id is not None else "selected", fitness, True, minimize)
        if selected is None and history.node("@best_program"):
            p = history.node("@best_program")
            selected = program_meta(p, sources, "selected", original.get("best_score"), True, minimize)
        if selected is None and curve:
            point = next(p for p in reversed(curve) if p["kind"] in {"initial", "breakthrough"})
            selected = metadata_by_candidate.get(str(point.get("candidate"))) or programs.get(point.get("node_id"))
    if selected and selected["id"] not in emitted:
        append_jsonl(work / "events.jsonl", {"kind": "program", "program": selected})
    best = selected if original.get("status") == "finished" else None
    if best is None and curve:
        point = next(p for p in reversed(curve) if p["kind"] in {"initial", "breakthrough"})
        best = metadata_by_candidate.get(str(point.get("candidate"))) or programs.get(point.get("node_id"))
    summary = {**{k: v for k, v in original.items() if k not in {"best", "best_score", "diagnostics"}},
               "result_format": RESULT_FORMAT, "method": config.get("method"),
               "status": original.get("status", "unknown"), "phase": original.get("phase", clock.get("phase")),
               "budget": original.get("budget", original.get("budget_slots", config.get("method_params", {}).get("budget", config.get("method_params", {}).get("max_sample_nums", 0)))),
               "budget_used": progress["budget_used"], "budget_axis": progress["x_label"],
               "num_nodes": progress["valid_nodes"], "best": best, "candidate_count": progress["candidate_count"],
               "valid_candidate_count": progress["valid_candidate_count"]}
    if not history_available:
        summary.update(history_available=False, budget_used=original.get("budget_used", 0),
                       num_nodes=original.get("num_nodes", 0),
                       candidate_count=original.get("candidate_count"),
                       valid_candidate_count=original.get("valid_candidate_count"))
    if best_old.get("selection_fitness") is not None and best:
        summary["best"] = {**best, "selection_fitness": best_old["selection_fitness"]}
    write_json(work / "summary.json", summary)
    state = dict(facts.state or {}) if facts else {}
    state.update(phase=summary["phase"] or ("finished" if summary["status"] == "finished" else "search"),
                 attempts=len(events), selected_id=selected["id"] if selected else None)
    append_jsonl(work / "events.jsonl", {"kind": "progress", "ts": clock.get("completed_at"),
        "progress": {"attempts": progress["budget_used"], "phase": state["phase"],
                     "elapsed": clock.get("elapsed"), "started_at": clock.get("started_at")}})
    selection = read_old(run_dir / "selection.json")
    selection_node, selection_key = selection_identity(selection)
    if selection:
        results = [{**{k: v for k, v in row.items() if k not in {"anchor_id", "outcome"}},
                    "node_id": row.get("node_id", row.get("anchor_id")),
                    "fitness": row.get("fitness", (row.get("outcome") or {}).get("fitness"))}
                   for row in selection.get("results", [])]
        write_json(work / "selection.json", {"selected_node": selection_node,
            "selected_key": selection_key, "results": results,
            "selection_protocol": selection.get("selection_protocol"),
            "finalists": [r["node_id"] for r in results]})
    if heldout is not None:
        write_json(work / "heldout.json", heldout)
    config.update(result_format=RESULT_FORMAT, budget_axis=progress["x_label"], budget=summary["budget"],
                  objective="min" if minimize else "max")
    write_json(work / "run_config.json", config)
    files = {p.name: p.stat().st_size for p in work.iterdir() if p.name in {
        "events.jsonl", "programs.jsonl", "calls.jsonl.gz"}}
    write_json(work / "resume.json", {"state": state, "files": files})
    # Normalize the intermediate v1 numbers before the current reader sees them.
    from .minimize_results import convert_run as minimize_run
    minimize_run(work, work)
    # Compare the curve, counts and source identities before replacing any data.
    from experiments.infra.monitor_history import TrainingHistory
    normalized = TrainingHistory(work, minimize)
    normalized_curve = normalized.read()[0]
    old_points = [(p["evaluation"], -p["fitness"], p["kind"]) for p in curve]
    new_points = [(p["evaluation"], p["fitness"], p["kind"]) for p in normalized_curve]
    if old_points != new_points:
        raise ValueError(f"curve changed during conversion: {run_dir}")
    converted = normalized.progress_snapshot()
    if any(converted[k] != progress[k] for k in ("budget_used", "candidate_count", "valid_candidate_count", "valid_nodes")):
        raise ValueError(f"candidate counts changed during conversion: {run_dir}")
    receipt.setdefault("before_bytes", sum(p.stat().st_size for p in run_dir.rglob('*') if p.is_file()
                       and '.convert' not in p.parts and '.conversion' not in p.parts))
    receipt.update(candidates=len(events), budget_used=progress["budget_used"], programs=len(sources.offsets),
                   calls=call_count)
    if not receipt.get("archive") and not active:
        receipt["archive"] = archive_run(run_dir, receipt)
    if source_mtime:
        os.utime(work / "events.jsonl", ns=(source_mtime, source_mtime))
    original_summary = run_dir / "logs/run_summary.json"
    if original_summary.exists():
        summary_mtime = original_summary.stat().st_mtime_ns
        os.utime(work / "summary.json", ns=(summary_mtime, summary_mtime))
    for p in work.iterdir():
        if p.is_file():
            p.replace(run_dir / p.name)
    shutil.rmtree(work)
    if active:
        # The old process still owns its terminal summary. Keep that publication
        # visible immediately until the final conversion after the process exits.
        summary_path = run_dir / "summary.json"
        summary_path.unlink()
        summary_path.symlink_to("logs/run_summary.json")
    receipt["converted_at"] = datetime.now(timezone.utc).isoformat()
    receipt["after_bytes"] = sum((run_dir/name).stat().st_size for name in CANONICAL if (run_dir/name).is_file())
    write_json(receipt_path, receipt)
    return receipt


def heldout_for_batch(batch):
    old = load_batch_heldout(batch)
    output, orphans = {}, []
    for variant, tasks in old.items():
        for task, data in tasks.items():
            for name, scores in data["runs"].items():
                run = batch / task / name
                entries = []
                for scale in set(map(str,scores)) | set(data.get("verification", {}).get(name, {})):
                    fitness = scores.get(int(scale) if scale.isdigit() else scale)
                    entry = {"variant": variant, "scale": scale, "task": task, "fitness": fitness,
                             "verification": data.get("verification", {}).get(name, {}).get(scale, "legacy")}
                    if not variant:
                        split = (f"eval_{int(scale.split('k_')[0])*1000}_{scale.split('k_')[1]}"
                                 if task == "online_bin_packing" else
                                 ("test_" if task in {"cvrp_aco", "op_aco"} else "eval_") + scale)
                        native = read_old(run / f"heldout_{split}.json")
                        if not native and scale == "50" and task not in {"cvrp_aco", "op_aco"}:
                            native = read_old(run / "heldout_eval.json")
                        if native:
                            entry = {**native, **entry}
                    if entry["verification"] == "verified" and not entry.get("key"):
                        node, key = selection_identity(read_old(run / "selection.json"))
                        entry.update(node_id=node, key=key)
                    entries.append(entry)
                if (run / "run_config.json").exists():
                    output[run] = output.get(run, []) + entries
                else:
                    orphans.extend({**e, "run_name": name} for e in entries)
    return output, orphans


def cleanup_run(run_dir, receipt, active=False):
    if active:
        return
    if not receipt.get("archive"):
        receipt["archive"] = archive_run(run_dir, receipt)
    for name in receipt["original_names"]:
        if name in CANONICAL:
            continue
        p = run_dir / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    receipt["sealed"] = True
    write_json(run_dir / ".conversion/receipt.json", receipt)
    for name in ("config.json", "selection.json"):
        (run_dir / ".conversion" / name).unlink(missing_ok=True)


def write_report():
    receipts = [read_json(p) for p in RESULTS.rglob(".conversion/receipt.json")]
    write_json(ARCHIVE / "report.json", {"format": RESULT_FORMAT,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": sorted(receipts, key=lambda r: r["run"])})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--batch")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args(argv)
    batches = [RESULTS / args.batch] if args.batch else sorted(
        p for p in RESULTS.iterdir() if p.is_dir() and any(p.glob('*/*/run_config.json')))
    active = active_names()
    if args.watch:
        pending = [p.parent for batch in batches for p in batch.glob('*/*/run_config.json')
                   if p.parent.name in active or not read_json(p.parent / '.conversion/receipt.json', {}).get('sealed')
                   and not read_json(p.parent / '.conversion/receipt.json', {}).get('archive')]
        while pending:
            active = active_names()
            for run in list(pending):
                receipt = convert_run(run, read_json(run / "heldout.json", []), run.name in active)
                if run.name not in active:
                    cleanup_run(run, receipt)
                    pending.remove(run)
                print(run.name, 'active' if run.name in active else 'sealed', flush=True)
            write_report()
            if pending:
                time.sleep(args.interval)
        return
    for batch in batches:
        results, orphans = heldout_for_batch(batch)
        if orphans:
            write_json(batch / "orphan_heldout.json", orphans)
        runs = sorted(p.parent for p in batch.glob('*/*/run_config.json'))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(convert_run, run, results.get(run, []), run.name in active): run for run in runs}
            for job in as_completed(jobs):
                receipt = job.result()
                if args.cleanup:
                    cleanup_run(jobs[job], receipt, jobs[job].name in active)
        print(batch.name, len(runs), 'converted', flush=True)
        write_report()


if __name__ == "__main__":
    main()
