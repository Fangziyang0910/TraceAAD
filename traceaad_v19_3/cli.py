"""Standalone demo and configurable OpenAI-compatible search entry."""

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from .config import Config
from .llm import OpenAIClient
from .search import METHOD, SCHEMA, TraceAAD
from .task import Task


TASK_NAMES = ("tsp_construct", "cvrp_aco", "fssp_gls", "graph_colouring", "jssp_construct",
              "op_aco", "online_bin_packing", "vrptw_construct")
SAMPLING = {"top_p": .95, "chat_template_kwargs": {"enable_thinking": False}}


def load_task(*, name=None, factory=None, options=None):
    if name:
        module = importlib.import_module("benchmarks.tasks")
        evaluator, settings = module.training_task(name, condition="traceaad")
        root = Path(module.__file__).resolve().parents[1]
        sources = [root / "benchmarks/tasks.py", root / "core/evaluate.py",
                   *(root / "benchmarks" / name).glob("*.py")]
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}
        return Task.from_evaluator(evaluator, metadata={"name": name, "settings": settings,
                                                       "source_hashes": hashes})
    module, separator, attribute = (factory or "").partition(":")
    if not separator or not module or not attribute:
        raise ValueError("Use --task-factory module:function")
    value = getattr(importlib.import_module(module), attribute)(**(options or {}))
    return value if isinstance(value, Task) else Task.from_evaluator(value)


def add_config_arguments(parser):
    for name in ("evaluation_budget", "initial_size", "seed", "max_tokens", "reflection_max_tokens",
                 "history_steps", "alternative_count", "experience_k", "max_lessons", "material_tokens",
                 "context_window", "protocol_tokens", "max_unevaluated"):
        parser.add_argument("--budget" if name == "evaluation_budget" else "--" + name.replace("_", "-"),
                            dest=name, type=int, default=None)
    for name in ("temperature", "model_timeout", "parent_floor"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=None)
    parser.add_argument("--reflection-mode", choices=("trajectory", "local", "siblings", "none"), default=None)
    parser.add_argument("--tokenizer", default=None)


def config_from_args(args, run_dir=None):
    if args.resume:
        state = json.loads((Path(run_dir or args.run_dir) / "state.json").read_text())
        if state.get("method") != METHOD or state.get("schema_version") != SCHEMA:
            raise ValueError("Use a V19.3 checkpoint")
        values = dict(state["config"])
    else:
        values = asdict(Config())
    for name in values:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    return Config(**values)


def _execute(task, model, config, directory, *, resume=False, quiet=False):
    with TraceAAD(task, model, config=config, run_dir=directory, resume=resume) as search:
        def progress(attempt):
            print(json.dumps({"attempt": attempt.id, "evaluations": search.evaluations_used,
                "model_calls": search.model_calls, "reflection": attempt.reflection_status,
                "best": search.best.evaluation.fitness if search.best else None}), file=sys.stderr, flush=True)
        result = search.run(on_attempt=None if quiet else progress)
    print(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if result.status == "finished" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent TraceAAD V19.3")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run scripted responses through the complete offline pipeline")
    demo.add_argument("--run-dir", required=True)
    demo.add_argument("--budget", type=int, default=12)
    demo.add_argument("--quiet", action="store_true")
    run = commands.add_parser("run")
    group = run.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", choices=TASK_NAMES)
    group.add_argument("--task-factory")
    run.add_argument("--task-options", default="{}")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--base-url", default=os.getenv("TRACEAAD_BASE_URL", os.getenv("OPENAI_BASE_URL")))
    run.add_argument("--model", default=os.getenv("TRACEAAD_MODEL", os.getenv("OPENAI_MODEL")))
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--request-options", default=json.dumps(SAMPLING))
    run.add_argument("--no-example", action="store_true")
    run.add_argument("--quiet", action="store_true")
    add_config_arguments(run)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import DemoModel, make_task
            return _execute(make_task(), DemoModel(), Config(evaluation_budget=args.budget),
                            args.run_dir, quiet=args.quiet)
        if not args.base_url or not args.model:
            parser.error("Provide --base-url and --model")
        options, request_options = json.loads(args.task_options), json.loads(args.request_options)
        if not isinstance(options, dict) or not isinstance(request_options, dict):
            raise ValueError("Options must be JSON objects")
        if args.task and options:
            raise ValueError("--task-options applies to --task-factory")
        task = load_task(name=args.task, factory=args.task_factory, options=options)
        if args.no_example:
            task.example_code = task.example_design = None
        model = OpenAIClient(args.base_url, args.model, api_key=os.getenv(args.api_key_env, ""),
                            request_options=request_options)
        return _execute(task, model, config_from_args(args), args.run_dir, resume=args.resume, quiet=args.quiet)
    except (ValueError, OSError, ImportError, AttributeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
