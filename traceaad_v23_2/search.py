"""Current Base generation with trajectory allocation and trajectory reference selection."""

from traceaad_base.search import TraceAAD as Base
from traceaad_v22_2.runtime import TrajectoryAllocation


class TraceAAD(TrajectoryAllocation, Base):
    method = "traceaad_v23_2"
    mechanisms = (1,)
