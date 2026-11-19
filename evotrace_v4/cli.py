"""Run EvoTrace v4 with an external task and an OpenAI-compatible endpoint."""

import argparse
from dataclasses import asdict, fields
import importlib
import json
import os
from pathlib import Path

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
            "fssp_gls": "Mean makespan; lower is better.",
            "online_bin_packing": "Mean number of used bins; lower is better.",
            "vrptw_construct": "Mean total route distance; lower is better.",
        }
        if name in scores:
            task.score_definition = scores[name]
        return task
    module, _, attribute = factory.partition(":")
    value = getattr(importlib.import_module(module), attribute)(**(options or {}))
    return value if isinstance(value, Task) else Task.from_evaluator(value)


def _settings(cls, saved: dict, args):
    """保存的设置，再叠加命令行显式给出的同名参数。"""
    given = {f.name: getattr(args, f.name, None) for f in fields(cls)}
    return cls(**{**saved, **{name: value for name, value in given.items() if value is not None}})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent EvoTrace v4 search")
    task_args = parser.add_mutually_exclusive_group(required=True)
    task_args.add_argument("--task", help="Task name in the optional research benchmark registry")
    task_args.add_argument("--task-factory", help="External task factory: module:function")
    parser.add_argument("--task-options", default="{}", help="JSON keyword arguments for --task-factory")
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
    parser.add_argument("--explore-references", type=int)
    parser.add_argument("--max-representatives", type=int)
    parser.add_argument("--cover-distance", type=float)
    parser.add_argument("--region-size", type=int)
    parser.add_argument("--no-example", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(args.run_dir)
    if not args.resume and (directory / "run_config.json").exists():
        parser.error("Run already exists; use --resume")
    saved = json.loads((directory / "run_config.json").read_text()) if args.resume else {}
    task = load_task(name=args.task, factory=args.task_factory, options=json.loads(args.task_options))
    if args.no_example:
        task.example_code = task.example_design = None
    search = EvoTrace(
        task, OpenAIClient(args.base_url, args.model, api_key=os.getenv(args.api_key_env, "")),
        config=_settings(Config, saved.get("config", {}), args),
        topology_config=_settings(TopologyConfig, saved.get("topology", {}), args),
        allocation_config=_settings(AllocationConfig, saved.get("allocation", {}), args),
        include_experiences=saved.get("include_experiences", True),
        run_dir=directory, resume=args.resume,
    )
    result = search.run()
    print(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if result.status == "finished" else 1
