"""Check or resume stopped baselines using saved checkpoints and current journals."""

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

from baselines.shared.observability import close_llm, shutdown_executor
from core.code import function_from_source
from baselines.shared.checkpoint import FIELDS, read_checkpoint, restore_checkpoint
from experiments.infra.runner import baseline_run_lock
from traceaad.storage import append_jsonl, read_json, rows, write_json
from experiments.infra.result_storage import Programs

METHODS = ("eoh", "reevo", "funsearch", "mcts_ahd", "calm", "pathwise", "shinka_evo", "deltaevolve",
           "openevolve", "hsevo", "refineevo")
# Search evaluations differ from candidates; a pending evaluation may remain at the budget limit.
PENDING_EVALUATION = ("deltaevolve", "openevolve")


def history(directory):
    directory = Path(directory)
    events = list(rows(directory / "events.jsonl"))
    calls, seen = [], set()
    for path in (directory / "calls.jsonl.gz", directory / "calls.jsonl"):
        for call in rows(path):
            identity = call.get("transport_id")
            if identity and identity in seen:
                continue
            if identity:
                seen.add(identity)
            calls.append(call)
    state = read_json(directory / "resume.json", {}).get("method_state", {})
    config = read_json(directory / "run_config.json")
    if not config or config["method"] not in METHODS:
        raise ValueError("Not a supported baseline run")
    if config.get("status") == "archived":
        raise ValueError("Archived experiments cannot be resumed")
    candidates = [r for r in events if r.get("kind") == "candidate"]
    checkpoint = read_checkpoint(directory) if config["method"] in FIELDS else None
    used = max(
        [state.get("sample_count", state.get("sample_order", 0)), checkpoint["budget_used"] if checkpoint else 0]
        + [r.get("budget_used", 0) for r in candidates]
        + [r.get("data", {}).get("sample_order", 0) for r in events if r.get("data", {}).get("counts_budget") is True]
    )
    for call in calls:
        if call.get("error"):
            continue
        stage = call.get("stage")
        consumed = stage == "generate" and config["method"] in ("eoh", "reevo", "funsearch")
        if config["method"] in ("shinka_evo", "deltaevolve", "openevolve"):
            consumed = stage == "patch"
        if config["method"] == "mcts_ahd":
            consumed = (
                stage == "thought_alignment"
                or stage == "generate"
                and (not call.get("function_parse_success") or call.get("parsed_thought") is None)
            )
        if consumed:
            used = max(used, call.get("sample_order") or 0)
    if (not candidates and not checkpoint) or not 0 <= used <= config["budget"] or used == 0 and not checkpoint:
        raise ValueError("Missing history or inconsistent candidate budget")
    return config, events, calls, state, candidates, used


def build(config, directory):
    module = importlib.import_module("experiments." + config["method"] + ".run")
    params = dict(config["method_params"])
    aliases = {
        "population_size": "pop_size",
        "selection_num": "parents",
        "paper_generations": "generations",
        "reset_period_seconds": "reset_period",
    }
    if config["method"] != "eoh":
        aliases.pop("selection_num")
    params.update({aliases[k]: v for k, v in list(params.items()) if k in aliases})
    params.update(
        task=config["task"],
        backend=config["backend"],
        seed=config["seed"],
        repeat=config["repeat"],
        base_url=config["llm"]["base_url"],
        model=config["llm"]["model"],
        no_proxy=config["llm"]["no_proxy"],
        output_tokens=config["llm"]["max_tokens"],
        budget=config["budget"],
        max_sample_nums=config["budget"],
    )
    accepted = module.RunSpec.__dataclass_fields__
    spec = module.RunSpec(**{k: v for k, v in params.items() if k in accepted})
    # Validate constructor defaults against the original condition before writing logs.
    with tempfile.TemporaryDirectory() as temporary:
        module.write_run_config(spec, Path(temporary), config["run_name"])
        generated = read_json(Path(temporary) / "run_config.json")
    for key in ("task_eval", "method_params", "llm"):
        a, b = dict(config[key]), dict(generated[key])
        if key == "method_params":
            a.pop("scheduler_socket", None)
            b.pop("scheduler_socket", None)
        if a != b:
            raise ValueError(f"Original {key} differs from the current implementation")
    return module.build_method(spec, Path(directory) / "logs")


def recover(method, directory):
    config, events, calls, state, candidates, used = history(directory)
    directory = Path(directory)
    sources = Programs(directory)
    name = config["method"]
    template = getattr(method, "_template_program", None)
    functions = {}
    for row in candidates:
        if name in FIELDS:
            continue
        if not row.get("valid"):
            continue
        code = sources.get(row["program"]["id"])
        function = function_from_source(code, template.functions[0].name)
        if function is None:
            raise ValueError(f"Cannot restore candidate {row['candidate_id']}")
        function.score = row["fitness"]
        function.algorithm = row["attempt"]["idea"]
        function.operator = row["operator"]
        function.sample_time = row["attempt"]["sample_time"]
        function.evaluate_time = row["attempt"]["evaluate_time"]
        functions[row["candidate_id"]] = function
    if not functions and name not in FIELDS:
        raise ValueError("No valid historical programs")
    report = dict(
        method=config["method"],
        task=config["task"],
        budget_used=used,
        budget=config["budget"],
        remaining=config["budget"] - used,
        recovery="journal_boundary",
        rng="reseeded_not_restored",
        valid_programs=sum(r.get("valid", False) for r in candidates),
    )
    method._tot_sample_nums = used
    method._resume_mode = True
    profiler = method._profiler
    profiler._num_samples = max((r["candidate_id"] for r in candidates), default=0)
    profiler._budget_used = used
    profiler._resume_budget_from_method = True
    profiler._evaluate_success_program_num = sum(r.get("valid", False) for r in candidates)
    profiler._evaluate_failed_program_num = len(candidates) - profiler._evaluate_success_program_num
    if name in PENDING_EVALUATION:
        method._evaluation_count = sum(r['attempt'].get('evaluate_time') is not None for r in candidates)
    profiler._canonical_best = copy.deepcopy(
        min((r["program"] for r in candidates if r.get("valid")), key=lambda r: r["fitness"], default=None)
    )
    profiler._token_usage = read_json(directory / "token_usage.json", profiler._token_usage)
    profiler._llm_call_count = profiler._token_usage["calls"]
    profiler._recorded_transport = {r["transport_id"] for r in calls if r.get("transport_id")}
    profiler._method_event_count = sum(r.get("kind") == "method" for r in events)
    profiler._error_count = sum(1 for _ in rows(directory / "logs/errors.jsonl"))
    profiler._resume_elapsed = max((r.get("progress", {}).get("elapsed", 0) for r in candidates), default=0)
    if used >= config["budget"] and name not in PENDING_EVALUATION:
        report["boundary"] = "final_selection_only"
        return report

    if name in FIELDS:
        checkpoint = restore_checkpoint(method, directory, name)
        if not 0 <= checkpoint["budget_used"] <= used or checkpoint["candidate_count"] > profiler._num_samples:
            raise ValueError("Checkpoint budget or candidate count exceeds the retained history")
        method._resume_boundary = checkpoint["boundary"]
        pending = used - checkpoint["budget_used"]
        if name == "deltaevolve":
            method._iteration = max(method._iteration, used - 1)
        if name == "pathwise" and pending:
            if method._initialization_complete:
                method._outer_iteration += 1  # Pending node IDs remain in the journal; do not reuse them.
            else:
                method._initialization_index += pending
        if name == "pathwise":
            profiler._cur_gen = method._population.generation
        report.update(
            recovery="search_checkpoint",
            boundary=checkpoint["boundary"],
            checkpoint_budget=checkpoint["budget_used"],
            pending_budget=pending,
            rng="restored_at_checkpoint",
        )
        return report

    def population():
        snapshots = [(i, r) for i, r in enumerate(events) if r.get("kind") == "population"]
        if not snapshots:
            raise ValueError("No recorded population")
        index, snapshot = snapshots[-1]
        members = []
        for member in snapshot["members"]:
            member_source = member["function"]
            function = function_from_source(member_source, template.functions[0].name)
            if function is None:
                raise ValueError("Cannot parse saved population source")
            function.score, function.algorithm = member["score"], member["algorithm"]
            members.append(function)
        method._population._population = members
        method._population._generation = snapshot["generation"]
        profiler._cur_gen = snapshot["generation"]
        return index

    if name == "eoh":
        index = population()
        for row in events[index + 1 :]:
            if row.get("kind") == "candidate" and row["candidate_id"] in functions:
                method._population.register_function(functions[row["candidate_id"]], increment_generation=False)
        method._population._generation = state.get("generation", method._population.generation)
        report["boundary"] = "next_operator"
    elif name == "reevo":
        population()
        method._elite_function = copy.deepcopy(min(functions.values(), key=lambda f: f.score))
        seeds = [
            functions[r["candidate_id"]]
            for r in candidates
            if r["operator"] == "seed" and r["candidate_id"] in functions
        ]
        if len(seeds) != 1:
            raise ValueError("Missing unique ReEvo seed")
        method._seed_function = seeds[0]
        reflections = [r for r in calls if r.get("stage") == "long_term_reflection" and not r.get("error")]
        if reflections:
            method._long_term_reflection_str = reflections[-1]["response"]
        report["boundary"] = "new_generation_from_last_population_and_current_elite"
    elif name == "funsearch":
        recover_database(method, events, state, candidates, functions)
        report["boundary"] = "new_prompt_batch"
    else:
        index = population()
        tree, accepted = recover_tree(method, events, candidates, functions, calls=calls)
        for row in events[index + 1 :]:
            if row.get("kind") == "candidate" and row["candidate_id"] in accepted:
                method._population.register_function(functions[row["candidate_id"]])
        nodes = method.population_management([node.individual for node in accepted.values()], method._pop_size)
        method._recovered_tree = tree, nodes
        report.update(boundary="new_tree_iteration", tree_nodes=len(accepted))
    return report


def recover_tree(method, events, candidates, functions, *, calls=()):
    from baselines.mcts_ahd.mcts import MCTS, MCTSNode

    tree = MCTS("Root", method.alpha, method.lambda_0)
    accepted = {}
    first_expand = next((i for i, r in enumerate(events) if r.get("data", {}).get("event") == "expand"), len(events))
    # Initialization registers root children but does not emit expand events.
    for row in events[:first_expand]:
        if row.get("kind") == "candidate" and row["candidate_id"] in functions:
            function = functions[row["candidate_id"]]
            if any(n.raw_info.score == function.score or n.code == str(function) for n in tree.root.children):
                continue
            if len(tree.root.children) >= method._init_pop_size:
                break
            accepted[row["candidate_id"]] = method._add_root_child(tree, function)
    for row in events[first_expand:]:
        data = row.get("data", {})
        if data.get("event") != "expand" or data.get("status") != "expanded":
            continue
        identity = row.get("profiler_sample_order")
        function = functions.get(identity)
        if function is None or function.score != data["child_score"]:
            raise ValueError("MCTS expansion cannot be matched to its candidate")
        parents = (
            [tree.root]
            if data["root_parent"]
            else [
                n
                for n in accepted.values()
                if n.raw_info.score == data["parent_score"]
                and n.depth == data["parent_depth"]
                and n.visits == data["parent_visits"] - 1
                and len(n.children) == data["parent_children_count"] - 1
            ]
        )
        if len(parents) > 1:
            prompts = [
                r["prompt"]
                for r in calls
                if r.get("stage") == "generate" and not r.get("error") and r.get("sample_order") == data["sample_order"]
            ]
            if len(prompts) == 1:
                parents = [n for n in parents if n.individual.body.strip() in prompts[0]]
        if len(parents) != 1:
            raise ValueError(
                f"MCTS parent identity is missing or ambiguous: candidate={identity}, "
                f"score={data['parent_score']}, depth={data['parent_depth']}, "
                f"visits={data['parent_visits']}, matches={len(parents)}"
            )
        parent = parents[0]
        node = MCTSNode(
            function.algorithm,
            str(function),
            function.score,
            individual=function,
            raw_info=function,
            parent=parent,
            depth=parent.depth + 1,
            visit=1,
            Q=-function.score,
        )
        if data["operator"] == "e1":
            node.subtree.append(node)
        parent.add_child(node)
        tree.backpropagate(node)
        accepted[identity] = node
    if len(tree.root.children) < method._selection_num:
        raise ValueError("MCTS initialization is incomplete")
    return tree, accepted


def recover_database(method, events, state, candidates, functions):
    db = method._database
    if state["island_resets"] != 0:
        raise ValueError("Island resets require a saved database; journal recovery cannot reconstruct founders")
    profiles = {
        row["profiler_sample_order"]: row["data"]["instance_scores"]
        for row in events
        if row.get("data", {}).get("event") == "evaluation_profile"
    }
    db._clock = lambda: 0.0
    db._last_reset_time = 0.0

    def register(identity, island):
        function = functions.get(identity)
        if function is None:
            return
        vector = profiles.get(identity)
        required = bool(getattr(method._profiler._evaluation, "_instance_execution", None))
        if required and not vector or vector and abs(-sum(vector) / len(vector) - function.score) > 1e-9:
            raise ValueError(f"Missing or inconsistent instance scores for program {identity}")
        function.instance_scores = tuple(vector or ())
        db.register_program(function, island, -function.score)

    seeds = [row for row in candidates if row["operator"] == "seed"]
    if len(seeds) != 1:
        raise ValueError("Missing unique FunSearch seed")
    register(seeds[0]["candidate_id"], None)
    for row in events:
        data = row.get("data", {})
        if data.get("event") == "database_register" and data.get("status") == "registered":
            register(row["profiler_sample_order"], data["island_id"])
    if [island.num_programs for island in db.islands] != state["island_program_counts"]:
        raise ValueError("Recovered island membership differs from recorded counts")
    if [-v for v in db.best_score_per_island] != state["island_best_scores"]:
        raise ValueError("Recovered island best scores differ")
    db._clock = time.time
    elapsed = max(row.get("progress", {}).get("elapsed", 0) for row in candidates)
    db._last_reset_time = time.time() - elapsed


def close_recovery_method(method):
    shutdown_executor(getattr(method, "_evaluation_executor", None))
    providers = getattr(method, "_all_llms", None) or getattr(method, "_llms", None)
    if callable(providers):
        llms = providers()
    else:
        llms = [getattr(method, "_llm", None), getattr(getattr(method, "_sampler", None), "llm", None)]
    seen = set()
    for llm in llms:
        if id(llm) not in seen:
            close_llm(llm)
            seen.add(id(llm))
    embedding = getattr(method, "_embedding_fn", None)
    close_llm(embedding)


def resume_manifest(path, *, execute=False):
    """Use the batch's existing sessions and logs, on its execution machine."""
    path = path.resolve()
    manifest = read_json(path)
    if not manifest or manifest.get("method") not in METHODS or manifest.get("status") == "archived":
        raise ValueError("Not a resumable baseline batch")
    failed = False
    for plan in manifest["plan"]:
        directory = path.parent / plan["task"] / plan["run_name"]
        try:
            config, *_ = history(directory)
            if (config["method"], config["task"], config["run_name"]) != (
                manifest["method"],
                plan["task"],
                plan["run_name"],
            ):
                raise ValueError("Run identity differs from the manifest")
            summary = read_json(directory / "summary.json", {})
            if summary.get("status") == "finished" and (directory / "selection.json").exists():
                print(f"{plan['run_name']}: finalized, skipped", flush=True)
                continue
            command = [sys.executable, "-u", "-m", "experiments.infra.resume_baseline", "--run-dir", str(directory)]
            if not execute:
                subprocess.run(command, check=True)
                continue
            command.append("--execute")
            session = plan["session"]
            if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
                print(f"{plan['run_name']}: session exists, skipped", flush=True)
                continue
            log = directory / "logs/resume_search.log"
            log.parent.mkdir(exist_ok=True)
            root = Path(__file__).resolve().parents[2]
            shell = f"cd {shlex.quote(str(root))} && exec {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1"
            subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], check=True)
            print(f"{plan['run_name']}: launched; log={log}", flush=True)
        except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
            failed = True
            print(f"{plan['run_name']}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    if failed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--manifest", type=Path, help="check or resume all runs in a batch; execute in tmux")
    parser.add_argument("--execute", action="store_true", help="resume model calls; default only checks recovery")
    args = parser.parse_args()
    if args.manifest:
        resume_manifest(args.manifest, execute=args.execute)
        return
    directory = args.run_dir.resolve()
    config, *_ = history(directory)
    from benchmarks.tasks import validate_task_interface

    validate_task_interface(config["task"], config)
    if args.execute:
        with baseline_run_lock(directory):
            from core.scheduling import scheduler_status

            socket = config["method_params"].get("scheduler_socket")
            if socket:
                scheduler_status(socket)
                os.environ["TRACEAAD_SCHEDULER_SOCKET"] = socket
            if (
                read_json(directory / "summary.json", {}).get("status") == "finished"
                and (directory / "selection.json").exists()
            ):
                raise ValueError("Run is already finalized; do not repeat final selection")
            method = build(config, directory)
            try:
                report = recover(method, directory)
                report["resumed_at"] = datetime.now(timezone.utc).isoformat()
                append_jsonl(directory / "recovery.jsonl", report)
                summary = read_json(directory / "summary.json", {})
                write_json(
                    directory / "summary.json",
                    {**summary, "status": "running", "writer_lock": True, "recovery": report},
                )
                os.environ["TRACEAAD_EVALUATION_LOG"] = str(directory / "evaluations.jsonl")
                if execution := getattr(method._profiler._evaluation, "_instance_execution", None):
                    execution["record_path"] = os.environ["TRACEAAD_EVALUATION_LOG"]
                if (report["remaining"] == 0 and getattr(method, "_pending_node", None) is None
                        and getattr(method, "_pending_evaluation", None) is None):
                    counts = ({'search_evaluations': method._evaluation_count}
                              if config['method'] in PENDING_EVALUATION else {})
                    method._profiler.write_run_summary(status="finished", recovery=report, **counts)
                else:
                    method.run()
            finally:
                close_recovery_method(method)
    else:
        with tempfile.TemporaryDirectory() as temporary:
            method = build(config, temporary)
            try:
                report = recover(method, directory)
            finally:
                close_recovery_method(method)
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
