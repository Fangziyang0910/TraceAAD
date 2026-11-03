import argparse

from traceaad.v18_1 import Config


BUDGET_BASIS = (
    "Program-generation replies including initialization, parse failures, duplicates and repair; "
    "additional BirthCheck reviews are counted separately and do not consume the generation budget. "
    "API failures without a generation reply do not consume that budget; "
    "all model requests, token usage and costs are recorded, including BirthCheck and failed requests."
)

FIELDS = {
    "roots": int,
    "init_attempt_limit": int,
    "history_depth": int,
    "attempts_shown": int,
    "progress_shown": int,
    "same_age": bool,
    "age_fraction": float,
    "protection_attempts": int,
    "max_refinement_age": int,
    "birth_output_tokens": int,
    "birth_context_tokens": int,
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
