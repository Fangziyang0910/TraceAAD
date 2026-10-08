"""Run V10.21 with the shared experiment protocol."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v10_21 import Config, TraceAADV1021
from traceaad.v10_21.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT)


def main(argv=None):
    return run(TraceAADV1021, Config, EXPERIMENT, "V10.21: V10.20 with time stated in the unit a program spends it (the per-instance budget in the evaluation section; time per instance, calls per instance and time per call in every measurement) and no ban on code comments (stored programs are canonical source); Refine/Explore/Crossover/Deepen 0.40/0.25/0.20/0.15", argv)


if __name__ == "__main__":
    main()
