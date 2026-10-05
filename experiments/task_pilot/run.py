"""One short V10.18 search on a candidate task, then the selected program on the full test split.

    uv run python -m experiments.task_pilot.run --task cob_jssp --backend server3 --seed 0 --run-name r1
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import sys

from benchmarks.co_bench import COBENCH_TASKS, COBenchEvaluation
from core import SecureEvaluator
from experiments.infra.base import RESULTS_ROOT, build_llm_client, resolve_backend, set_random_seed, write_run_config
from traceaad.v10_13.storage import write_json
from traceaad.v10_18 import Config, TraceAADV1018

EXPERIMENT = "task_pilot"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(COBENCH_TASKS), required=True)
    parser.add_argument("--backend", default="server3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--limit", type=int, default=8, help="development instances used by the search")
    parser.add_argument("--workers", type=int, default=2, help="instances solved in parallel")
    parser.add_argument("--output-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    run_dir = RESULTS_ROOT / EXPERIMENT / args.task / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    search = COBenchEvaluation(args.task, split="dev", limit=args.limit, workers=args.workers)
    selection = COBenchEvaluation(args.task, split="test", limit=args.limit, workers=args.workers)
    profile = resolve_backend(args.backend, None, None, None)
    config = Config(budget=args.budget, output_tokens=args.output_tokens, seed=args.seed)
    if not (run_dir / "run_config.json").exists():
        write_run_config(run_dir, {
            "created_at": datetime.now().isoformat(timespec="seconds"), "task": args.task,
            "problem": search.problem, "method": "v1018", "backend": args.backend, "seed": args.seed,
            "model": profile.model, "base_url": profile.base_url, "config": asdict(config),
            "search_instances": search.order, "selection_instances": selection.order,
            "instance_seconds": search.instance_seconds, "workers": args.workers,
            "search_timeout": search.timeout_seconds})
    set_random_seed(args.seed)
    llm = build_llm_client(base_url=profile.base_url, model=profile.model, no_proxy=profile.no_proxy,
                           max_tokens=args.output_tokens)
    llm._client = llm._client.with_options(max_retries=0)
    try:
        summary = TraceAADV1018(evaluation=search, selection_evaluation=selection, llm=llm, run_dir=run_dir,
                                config=config, task=args.task).run()
    finally:
        llm.close()
    print(json.dumps({"status": summary["status"], "best": (summary.get("best") or {}).get("score")}))
    if summary["status"] != "finished" or (run_dir / "heldout_test.json").exists():
        return
    code = (run_dir / "best_program.py").read_text(encoding="utf-8")
    test = COBenchEvaluation(args.task, split="test", workers=4)
    outcome = SecureEvaluator(test).evaluate_program_with_details(code)
    score = outcome.result["score"] if isinstance(outcome.result, dict) else outcome.result
    write_json(run_dir / "heldout_test.json", {
        "task": args.task, "problem": test.problem, "instances": len(test.order), "score": score,
        "failure": outcome.failure_kind, "error": outcome.error})
    print(json.dumps({"test": score, "failure": outcome.failure_kind}))


if __name__ == "__main__":
    sys.exit(main())
