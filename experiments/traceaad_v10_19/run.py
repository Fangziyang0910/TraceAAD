"""Run V10.19 with the shared experiment protocol."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v10_19 import Config, TraceAADV1019
from traceaad.v10_19.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT)


def main(argv=None):
    return run(TraceAADV1019, Config, EXPERIMENT, "V10.19: an Explore proposal is a change to its algorithm (Explore keeps the parts the change does not replace); a change behind its algorithm gets up to 3 Develop steps that see the algorithm and end once a version beats it; Explore draws continue the open change before proposing; a change ends in its best version in the experience and attempt list of its algorithm; Refine/Explore/Crossover 0.40/0.35/0.25", argv)


if __name__ == "__main__":
    main()
