"""Locations of archived initialization studies for offline analysis."""

from experiments.infra.base import RESULTS_ROOT

RESULTS = RESULTS_ROOT / "traceaad_initialization"
SCHEDULE = RESULTS / "schedule.json"


def run_dir(row):
    return RESULTS / row["task"] / row["run_name"]
