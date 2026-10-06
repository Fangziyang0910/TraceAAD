"""Run V10.18 with the shared experiment protocol."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v10_18 import Config, TraceAADV1018
from traceaad.v10_18.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT)


def main(argv=None):
    return run(TraceAADV1018, Config, EXPERIMENT, "V10.18: every new Explore design is developed from its best version until it shows whether it works (at least 2 steps, while improving, at most 8, until it ranks among the five best); Explore draws develop the open design before proposing; Refine/Explore/Crossover 0.35/0.40/0.25", argv)


if __name__ == "__main__":
    main()
