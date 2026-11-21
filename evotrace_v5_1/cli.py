"""Run EvoTrace v5.1 with an external task and an OpenAI-compatible endpoint."""

import argparse
import importlib
import json
import os
from dataclasses import asdict, fields
from pathlib import Path

from . import Config, EvoTrace, OpenAIClient, Task

CONFIG_NAME = "run_config.json"
SCORES = {
    "tsp_construct": "Mean tour length; lower is better.",
    "cvrp_aco": "Mean total route length; lower is better.",
    "op_aco": "Negative mean collected prize; lower is better.",
    "fssp_gls": "Mean makespan; lower is better.",
    "online_bin_packing": "Mean number of used bins; lower is better.",
    "vrptw_construct": "Mean total route distance; lower is better.",
}


def load_task(*, name=None, factory=None, options=None) -> Task:
    """The optional repository benchmark registry is loaded only for --task."""
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


def _resolve_run(args, parser) -> tuple[Config, str, bool, bool, bool]:
    """解析新跑与续跑参数；续跑时沿用已存配置，仅允许调大 evaluation_budget。"""
    config_path = Path(args.run_dir) / CONFIG_NAME
    if args.resume != config_path.exists():
        parser.error(
            f"No saved run to resume in {args.run_dir}"
            if args.resume
            else "Run already exists; use --resume"
        )
    names = {f.name for f in fields(Config)}
    given = {n: v for n in names if (v := getattr(args, n, None)) is not None}
    if not args.resume:
        model = args.model or os.getenv("OPENAI_MODEL")
        if not args.base_url or not model:
            parser.error("Set --base-url and --model (or OPENAI_BASE_URL and OPENAI_MODEL)")
        reflection = True if args.reflection is None else args.reflection
        return Config(**given), model, reflection, reflection, args.no_example

    saved = json.loads(config_path.read_text())
    saved_cfg = saved.get("config", {})
    if set(saved_cfg) - names:
        parser.error(
            "The saved settings do not match this version; resume with the source that created the run"
        )
    for k, v in given.items():
        if k != "evaluation_budget" and k in saved_cfg and saved_cfg[k] != v:
            parser.error(f"Resume requires the original {k.replace('_', '-')} ({saved_cfg[k]!r})")
    reflection = saved.get("reflection", True)
    if args.reflection is not None and args.reflection != reflection:
        parser.error("Resume requires the original reflection setting")
    saved_task = saved.get("task") or {}
    if args.no_example and saved_task.get("example_code") is not None:
        parser.error("Resume requires the original example setting")
    saved_model = saved.get("model")
    if args.model and saved_model and args.model != saved_model:
        parser.error(f"Resume requires the original model ({saved_model!r})")
    model = args.model or saved_model or os.getenv("OPENAI_MODEL")
    if not args.base_url or not model:
        parser.error("Set --base-url and --model (or OPENAI_BASE_URL and OPENAI_MODEL)")
    budget = {"evaluation_budget": args.evaluation_budget} if args.evaluation_budget is not None else {}
    no_example = args.no_example or ("example_code" in saved_task and saved_task["example_code"] is None)
    return (
        Config(**{**saved_cfg, **budget}),
        model,
        reflection,
        saved.get("include_experiences", reflection),
        no_example,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent EvoTrace v5.1 search")
    task_args = parser.add_mutually_exclusive_group(required=True)
    task_args.add_argument("--task", help="Task name in the optional research benchmark registry")
    task_args.add_argument("--task-factory", help="External task factory: module:function")
    parser.add_argument(
        "--task-options", default="{}", help="JSON keyword arguments for --task-factory"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue with the saved settings; only --budget may change",
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument(
        "--model", help="Model name (default: OPENAI_MODEL, or the saved name on resume)"
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--budget", dest="evaluation_budget", type=int)
    parser.add_argument("--initial-size", type=int)
    parser.add_argument("--population-capacity", type=int)
    parser.add_argument("--selection-lambda", type=float)
    parser.add_argument("--explore-references", type=int)
    parser.add_argument("--reflection-mask-prob", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--model-timeout", type=float)
    parser.add_argument(
        "--reflection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Second-Chance reflection on leaf eviction (default: enabled)",
    )
    parser.add_argument("--no-example", action="store_true")
    args = parser.parse_args(argv)

    config, model_name, reflection, include_experiences, no_example = _resolve_run(args, parser)
    task = load_task(
        name=args.task, factory=args.task_factory, options=json.loads(args.task_options)
    )
    if no_example:
        task.example_code = task.example_design = None
    search = EvoTrace(
        task,
        OpenAIClient(args.base_url, model_name, api_key=os.getenv(args.api_key_env, "")),
        config=config,
        include_experiences=include_experiences,
        reflection=reflection,
        run_dir=Path(args.run_dir),
        resume=args.resume,
    )
    result = search.run()
    print(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False, indent=2))
    return 0 if result.status == "finished" else 1
