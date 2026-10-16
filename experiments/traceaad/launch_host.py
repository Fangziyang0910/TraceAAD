"""Plan or launch six tasks × three repetitions of TraceAAD on this host."""

import argparse

from experiments.infra.search_host_launch import main as launch
from traceaad import Config
from traceaad.config import EXPERIMENT

BACKENDS = ("deepseek", "deepseek")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--backends",
        nargs=2,
        default=BACKENDS,
        help="alternated across runs, e.g. server3 server3b",
    )
    args, rest = parser.parse_known_args(argv)
    return launch(
        Config,
        EXPERIMENT,
        "experiments.traceaad.run",
        "traceaad",
        "local",
        tuple(args.backends),
        rest,
    )


if __name__ == "__main__":
    main()
