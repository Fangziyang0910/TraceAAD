"""Current Base generation with the same trajectory allocation as V22.1."""

import json
from pathlib import Path

from traceaad_base.search import TraceAAD as Base
from traceaad_v22_1.runtime import TrajectoryAllocation


class TraceAAD(TrajectoryAllocation, Base):
    method = "traceaad_v23_1"

    def __init__(self, task, model, **options):
        super().__init__(task, model, **options)
        if not options.get("resume") and self.store.path is not None:
            saved = self.store.read("run_config.json")
            self.store.write("run_config.json", {
                **saved, "generation_protocol": self.generation_protocol,
            })

    def _validate_resume(self, task, config, directory):
        super()._validate_resume(task, config, directory)
        saved = json.loads((Path(directory) / "run_config.json").read_text())
        if saved.get("generation_protocol") != self.generation_protocol:
            raise ValueError(f"V23.1 resume requires {self.generation_protocol}; "
                             "use the original source snapshot for older runs")
