"""V22.2 = V21.1 plus the current trajectory allocation and reference mechanism."""

from traceaad_v21_1.search import TraceAAD as V21_1

from .runtime import TrajectoryAllocation


class TraceAAD(TrajectoryAllocation, V21_1):
    method = "traceaad_v22_2"
    mechanisms = (1, 2)
