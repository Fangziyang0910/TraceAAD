from traceaad.v17 import Config


def flag(text):
    if text.lower() not in ("true", "false"):
        raise ValueError(f"expected True or False, got {text}")
    return text.lower() == "true"


FIELDS = dict(roots=int, init_attempt_limit=int, history_depth=int, revisit=flag, diverge=flag,
              stagnation_window=int, mature_steps=int, map_size=int, map_distance=float, diverge_shown=int,
              allocation=str, exposure=str, projection=flag, changes=flag, foreign_reference=flag,
              progress_rule=flag, imports=flag)


def add_arguments(parser):
    for name, kind in FIELDS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=getattr(Config, name))


def values(args):
    return {name: getattr(args, name) for name in FIELDS}


def command_arguments(args):
    return [f'--{name.replace("_", "-")}={getattr(args, name)}' for name in FIELDS] + [
        f"--output-tokens={args.output_tokens}"
    ]
