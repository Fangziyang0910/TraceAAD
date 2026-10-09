"""Run V10.24 with the shared experiment protocol."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v10_24 import Config, TraceAADV1024
from traceaad.v10_24.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT, Config)


def main(argv=None):
    return run(TraceAADV1024, Config, EXPERIMENT, "V10.24: solver-effect requests with finite development commitments", argv)


if __name__ == "__main__":
    main()
