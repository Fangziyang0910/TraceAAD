"""Run V10.21 with the shared experiment protocol."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v10_21 import Config, TraceAADV1021
from traceaad.v10_21.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT, Config)


def main(argv=None):
    return run(TraceAADV1021, Config, EXPERIMENT, "V10.21: Refine/Explore/Crossover 0.45/0.30/0.25; concise prompts and per-instance evaluation", argv)


if __name__ == "__main__":
    main()
