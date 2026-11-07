"""V22 = V21 plus mechanism 1."""

from traceaad_v21.search import TraceAAD as V21
from .runtime import TrajectoryAllocation


class TraceAAD(TrajectoryAllocation, V21):
    method = "traceaad_v22"
