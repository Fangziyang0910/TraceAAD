"""Base generation with the same trajectory allocation as V22."""

from traceaad_base.search import TraceAAD as Base
from traceaad_v22.runtime import TrajectoryAllocation


class TraceAAD(TrajectoryAllocation, Base):
    method = "traceaad_v23"
