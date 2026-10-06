"""Run V10.16 with the shared experiment protocol."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v10_16 import Config, TraceAADV1016
from traceaad.v10_16.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT)


def main(argv=None):
    return run(TraceAADV1016, Config, EXPERIMENT, "V10.16: programs and generation events; experience-weighted starting points; measured outcomes in context", argv)


if __name__ == "__main__":
    main()
