"""Command-line entry points; benchmark imports occur only for --task."""

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import sys

from .config import Config
from .llm import OpenAIClient
from .search import TraceAAD
from .task import Task


SCORES = {
    "tsp_construct": "Mean tour length; lower is better.",
    "cvrp_aco": "Mean total route length; lower is better.",
    "op_aco": "Negative mean collected prize; lower is better.",
    "online_bin_packing": "Mean number of used bins; lower is better.",
    "vrptw_construct": "Mean total route distance; lower is better.",
}
TASK_NAMES = tuple(SCORES) + ("fssp_gls", "graph_colouring", "jssp_construct")


def load_task(*, name=None, factory=None, options=None) -> Task:
    if name:
        registry = importlib.import_module("benchmarks.tasks")
        evaluator, settings = registry.training_task(name, condition="traceaad")
        task = Task.from_evaluator(evaluator, metadata={"name": name, "settings": settings})
        if name in SCORES:
            task.score_definition = SCORES[name]
        return task
    module, separator, attribute = (factory or "").partition(":")
    if not separator or not module or not attribute:
        raise ValueError("Use --task-factory module:function")
    value = getattr(importlib.import_module(module), attribute)(**(options or {}))
    return value if isinstance(value, Task) else Task.from_evaluator(value)


def _parser():
    parser = argparse.ArgumentParser(description="Independent TraceAAD V19.1")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run an offline example without a model service")
    demo.add_argument("--run-dir", required=True)
    demo.add_argument("--budget", type=int, default=9)
    demo.add_argument("--quiet", action="store_true")

    run = commands.add_parser("run", help="Search with an OpenAI-compatible model endpoint")
    task = run.add_mutually_exclusive_group(required=True)
    task.add_argument("--task", choices=TASK_NAMES, help="Use a task from this research repository")
    task.add_argument("--task-factory", help="External zero-argument task factory: module:function")
    run.add_argument("--task-options", default="{}", help="JSON keyword arguments for the task factory")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--base-url", default=os.getenv("TRACEAAD_BASE_URL", os.getenv("OPENAI_BASE_URL")))
    run.add_argument("--model", default=os.getenv("TRACEAAD_MODEL", os.getenv("OPENAI_MODEL")))
    run.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the key")
    run.add_argument("--budget", dest="evaluation_budget", type=int, default=None)
    run.add_argument("--initial-size", type=int, default=None)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--temperature", type=float, default=None)
    run.add_argument("--max-tokens", type=int, default=None)
    run.add_argument("--model-timeout", type=float, default=None)
    run.add_argument("--no-example", action="store_true")
    run.add_argument("--quiet", action="store_true")
    return parser


def _config(args):
    if args.resume:
        state = json.loads((Path(args.run_dir) / "state.json").read_text())
        values = state["config"]
    else:
        values = asdict(Config())
    for key in values:
        value = getattr(args, key, None)
        if value is not None:
            values[key] = value
    return Config(**values)


def _execute(task, model, config, directory, *, resume=False, quiet=False):
    with TraceAAD(task, model, config=config, run_dir=directory, resume=resume) as search:
        def progress(attempt):
            program = search.archive.get(attempt.program_id)
            outcome = program.evaluation.status if program is not None else "unevaluated"
            best = search.best.evaluation.fitness if search.best is not None else None
            print(f"attempt={attempt.id} operator={attempt.operator} outcome={outcome} "
                  f"evaluations={search.evaluations_used}/{config.evaluation_budget} "
                  f"model_calls={search.model_calls} best={best}", file=sys.stderr, flush=True)

        result = search.run(on_attempt=None if quiet else progress)
    print(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if result.status == "finished" else 1


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import DemoModel, make_task

            return _execute(make_task(), DemoModel(), Config(evaluation_budget=args.budget),
                            args.run_dir, quiet=args.quiet)
        if not args.base_url or not args.model:
            parser.error("Provide --base-url and --model, or set TRACEAAD_BASE_URL and TRACEAAD_MODEL")
        options = json.loads(args.task_options)
        if not isinstance(options, dict):
            parser.error("--task-options must be a JSON object")
        if args.task and options:
            parser.error("--task-options applies to --task-factory")
        task = load_task(name=args.task, factory=args.task_factory, options=options)
        if args.no_example:
            task.example_code = task.example_design = None
        client = OpenAIClient(args.base_url, args.model, api_key=os.getenv(args.api_key_env, ""))
        return _execute(task, client, _config(args), args.run_dir,
                        resume=args.resume, quiet=args.quiet)
    except (ValueError, OSError, ImportError, AttributeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
