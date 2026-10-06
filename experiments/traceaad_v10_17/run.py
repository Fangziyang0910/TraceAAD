"""Run V10.17 with the shared experiment protocol."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v10_17 import Config, TraceAADV1017
from traceaad.v10_17.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT)


def main(argv=None):
    return run(TraceAADV1017, Config, EXPERIMENT, "V10.17: V10.16 plus three Refine steps for a random eighth of new Explore programs (from the best reached); goals name what the step does for the given program", argv)


if __name__ == "__main__":
    main()
