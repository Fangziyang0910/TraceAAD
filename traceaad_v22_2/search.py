"""V22.2 = V21.3 plus the trajectory allocation and reference mechanism."""

from traceaad_v21_3.search import TraceAAD as V21_3

from .runtime import TrajectoryAllocation


class TraceAAD(TrajectoryAllocation, V21_3):
    method = "traceaad_v22_2"
    mechanisms = (1, 2)
