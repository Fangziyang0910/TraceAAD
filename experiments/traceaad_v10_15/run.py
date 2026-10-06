"""Run V10.15 with the shared experiment protocol."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v10_15 import Config, TraceAADV1015
from traceaad.v10_15.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT)


def main(argv=None):
    return run(TraceAADV1015, Config, EXPERIMENT, "V10.15: quality allocation; formation history in generation only", argv)


if __name__ == "__main__":
    main()
