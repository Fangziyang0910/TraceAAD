"""Base generation with the same trajectory allocation as V22.1."""

from traceaad_base.search import TraceAAD as Base
from traceaad_v22_1.runtime import TrajectoryAllocation


class TraceAAD(TrajectoryAllocation, Base):
    method = "traceaad_v23_1"
