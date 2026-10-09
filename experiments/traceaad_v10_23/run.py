"""Run V10.23 with the shared experiment protocol."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v10_23 import Config, TraceAADV1023
from traceaad.v10_23.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT, Config)


def main(argv=None):
    return run(TraceAADV1023, Config, EXPERIMENT, "V10.23: steps defined by the kind of change, each with its own material", argv)


if __name__ == "__main__":
    main()
