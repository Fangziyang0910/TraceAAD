"""Run V10.22 with the shared experiment protocol."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v10_22 import Config, TraceAADV1022
from traceaad.v10_22.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT, Config)


def main(argv=None):
    return run(TraceAADV1022, Config, EXPERIMENT, "V10.22: V10.21 with Explore reasoning from the exact output", argv)


if __name__ == "__main__":
    main()
