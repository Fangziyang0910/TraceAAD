"""Run EvoTrace with an external task and an OpenAI-compatible endpoint."""

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import sys

from .config import Config
from .context import ContextConfig
from .allocation import AllocationConfig
from .llm import OpenAIClient
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
    parser.add_argument("--quality-ratio", type=float,
                        help="Best/worst sampling weight ratio at equal effort (default: 10)")
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
        values = dict(saved["config"]) if args.resume else asdict(Config())
        for key in values:
            value = getattr(args, key, None)
            if value is not None:
                values[key] = value
        settings = {}
        if args.resume:
            if saved.get("method") != "evotrace_v3":
                raise ValueError("This is not a V3 run; resume it with its original frozen source")
            if saved.get("selection_rule") != EvoTrace.selection_rule:
                raise ValueError("Incompatible selection_rule; resume with its original frozen source")
            settings = {
                "topology_config": TopologyConfig(**saved["topology"]),
                "allocation_config": AllocationConfig(**saved["allocation"]),
                "context_config": ContextConfig(**saved["context"]),
                "reflection_temperature": saved["reflection_temperature"],
                "reflection_max_tokens": saved["reflection_max_tokens"],
            }
            if not saved["reflection_enabled"]:
                settings["reflection_model"] = None
        if args.quality_ratio is not None:
            allocation = dict(saved["allocation"]) if args.resume else {}
            allocation["quality_ratio"] = args.quality_ratio
            settings["allocation_config"] = AllocationConfig(**allocation)
        task = load_task(name=args.task, factory=args.task_factory, options=options)
        if args.no_example:
            task.example_code = task.example_design = None
        model = OpenAIClient(args.base_url, args.model, api_key=os.getenv(args.api_key_env, ""))
        search = EvoTrace(task, model, config=Config(**values), run_dir=directory,
                          resume=args.resume, **settings)
        result = search.run()
        print(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False, indent=2))
        return 0 if result.status == "finished" else 1
    except (ValueError, OSError, ImportError, AttributeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
