"""Plan or launch six tasks × three repetitions of one V12.1 arm on this host."""

import argparse

from experiments.infra.search_host_launch import main as launch
from traceaad.v12_1 import Config, DrawConfig, ResampleConfig
from traceaad.v12_1.config import EXPERIMENT

BACKENDS = ("deepseek", "deepseek")
ARMS = {"verified": (Config, "experiments.traceaad_v12_1.run"),
        "draw": (DrawConfig, "experiments.traceaad_v12_1.run_draw"),
        "resample": (ResampleConfig, "experiments.traceaad_v12_1.run_resample")}


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arm", choices=tuple(ARMS), default="verified")
    parser.add_argument("--backends", nargs=2, default=BACKENDS, help="alternated across runs, e.g. server3 server3b")
    args, rest = parser.parse_known_args(argv)
    config_class, module = ARMS[args.arm]
    return launch(config_class, EXPERIMENT, module, f"v12_1-{args.arm}", "local", tuple(args.backends), rest)


if __name__ == "__main__":
    main()
