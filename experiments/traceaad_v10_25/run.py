"""Run the V10.17-based mechanism under the shared six-task protocol."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v10_25 import Config, TraceAADV1025
from traceaad.v10_25.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT, Config)


def main(argv=None):
    return run(TraceAADV1025, Config, EXPERIMENT, "V10.25: V10.17 search with explicit repair history", argv)


if __name__ == "__main__":
    main()
