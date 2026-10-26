from traceaad.v13 import Config

FIELDS = dict(roots=int, init_attempt_limit=int, history_depth=int, prompt_tokens=int,
              max_model_calls=int, open_floor=float, route_floor=float, recent_window=int,
              max_routes=int, library_capacity=int, retrieval_size=int, revision_interval=int,
              library_path=str)


def add_arguments(parser):
    for name, kind in FIELDS.items():
        parser.add_argument('--' + name.replace('_', '-'), type=kind, default=getattr(Config, name))


def values(args):
    return {name: getattr(args, name) for name in FIELDS}


def command_arguments(args):
    return [f'--{name.replace("_", "-")}={getattr(args, name)}'
            for name in FIELDS if getattr(args, name) is not None] + [f'--output-tokens={args.output_tokens}']
