"""Run V10.20 with the shared experiment protocol."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v10_20 import Config, TraceAADV1020
from traceaad.v10_20.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT)


def main(argv=None):
    return run(TraceAADV1020, Config, EXPERIMENT, "V10.20: V10.17 plus true facts about computation (ACO functions are called once per instance; VRPTW service durations; evaluation time against the limit; training and test sets with their limits) and a Deepen step that keeps the current decision rule and uses it to guide a search spending more of the time limit, deciding from the current algorithm, its measured cost and earlier Deepen attempts (no formation path); Explore and Deepen proposals both open explorations (1/8 get three Refine steps); Refine/Explore/Crossover/Deepen 0.40/0.25/0.20/0.15", argv)


if __name__ == "__main__":
    main()
