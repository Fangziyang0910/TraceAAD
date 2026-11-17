"""Run EvoTrace with an external task and an OpenAI-compatible endpoint."""

import argparse
from dataclasses import asdict, fields
import importlib
import json
import os
from pathlib import Path
import sys

from .allocation import AllocationConfig
from .llm import OpenAIClient
from .models import Config
from .search import EvoTrace
from .task import Task
from .topology import TopologyConfig


def load_task(*, name=None, factory=None, options=None) -> Task:
    """The optional repository benchmark registry is loaded only for --task."""
    if name:
        registry = importlib.import_module("benchmarks.tasks")
        evaluator, settings = registry.training_task(name, condition="traceaad")
        task = Task.from_evaluator(evaluator, metadata={"name": name, "settings": settings})
        scores = {
            "tsp_construct": "Mean tour length; lower is better.",
            "cvrp_aco": "Mean total route length; lower is better.",
            "op_aco": "Negative mean collected prize; lower is better.",
            "online_bin_packing": "Mean number of used bins; lower is better.",
            "vrptw_construct": "Mean total route distance; lower is better.",
        }
        if name in scores:
            task.score_definition = scores[name]
        return task
    module, separator, attribute = (factory or "").partition(":")
    if not separator or not module or not attribute:
        raise ValueError("Use --task-factory module:function")
    value = getattr(importlib.import_module(module), attribute)(**(options or {}))
    return value if isinstance(value, Task) else Task.from_evaluator(value)


def _merge(cls, base: dict, overrides: dict, label: str):
    try:
        return cls(**{**base, **overrides})
    except TypeError as exc:
        raise ValueError(f"Incompatible {label}; resume with its original frozen source") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent EvoTrace v3 search")
    task_args = parser.add_mutually_exclusive_group(required=True)
    task_args.add_argument("--task", help="Task name in the optional research benchmark registry")
    task_args.add_argument("--task-factory", help="External task factory: module:function")
    parser.add_argument("--task-options", default="{}")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--budget", dest="evaluation_budget", type=int)
    parser.add_argument("--initial-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--model-timeout", type=float)
    parser.add_argument("--rank-half-life", type=float)
    parser.add_argument("--max-representatives", type=int)
    parser.add_argument("--cover-distance", type=float)
    parser.add_argument("--region-size", type=int)
    parser.add_argument("--no-example", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.base_url or not args.model:
            parser.error("Provide --base-url and --model, or set OPENAI_BASE_URL and OPENAI_MODEL")
        options = json.loads(args.task_options)
        if not isinstance(options, dict) or (args.task and options):
            parser.error("--task-options must be a JSON object for --task-factory")
        directory = Path(args.run_dir)
        if not args.resume and (directory / "run_config.json").exists():
            parser.error("Run already exists; use --resume")
        saved = json.loads((directory / "run_config.json").read_text()) if args.resume else {}
        cfg_args = {f.name: getattr(args, f.name, None) for f in fields(Config)
                    if getattr(args, f.name, None) is not None}
        top_args = {f.name: getattr(args, f.name) for f in fields(TopologyConfig)
                    if getattr(args, f.name) is not None}
        alloc_args = {"rank_half_life": args.rank_half_life} if args.rank_half_life is not None else {}
        config = _merge(Config, saved.get("config", {}), cfg_args, "config")
        topology = _merge(TopologyConfig, saved.get("topology", {}), top_args, "topology")
        allocation = _merge(AllocationConfig, saved.get("allocation", {}), alloc_args, "allocation")
        task = load_task(name=args.task, factory=args.task_factory, options=options)
        if args.no_example:
            task.example_code = task.example_design = None
        model = OpenAIClient(args.base_url, args.model, api_key=os.getenv(args.api_key_env, ""))
        search = EvoTrace(
            task, model, config=config, topology_config=topology, allocation_config=allocation,
            include_experiences=saved.get("include_experiences", True),
            run_dir=directory, resume=args.resume,
        )
        result = search.run()
        print(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False, indent=2))
        return 0 if result.status == "finished" else 1
    except (ValueError, OSError, ImportError, AttributeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
