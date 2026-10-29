import argparse

from traceaad.v18 import Config


FIELDS = {
    "roots": int, "init_attempt_limit": int, "history_depth": int, "attempts_shown": int,
    "first_checkpoint": int, "checkpoint_multiplier": int, "initial_depth_limit": int,
    "depth_increment": int, "stagnation_window": int, "context_tokens": int,
    "same_age": bool, "lds": bool, "early_first": bool, "diverge": bool,
    "objective": str, "max_seconds": float,
}


def add_arguments(parser):
    for name, kind in FIELDS.items():
        options = {"action": argparse.BooleanOptionalAction} if kind is bool else {"type": kind}
        parser.add_argument("--" + name.replace("_", "-"), default=getattr(Config, name), **options)


def values(args):
    return {name: getattr(args, name) for name in FIELDS}


def command_arguments(args):
    result = []
    for name, kind in FIELDS.items():
        value = getattr(args, name)
        option = name.replace("_", "-")
        if kind is bool:
            result.append("--" + ("" if value else "no-") + option)
        elif value is not None:
            result.append(f"--{option}={value}")
    return result + [f"--output-tokens={args.output_tokens}"]
