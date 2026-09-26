"""V10.13 training monitor."""

from __future__ import annotations

import sys

from core.training_monitor import main
from experiments.infra.base import RESULTS_ROOT


if __name__ == "__main__":
    root = RESULTS_ROOT / "traceaad_v10_13"
    if "--results-dir" not in sys.argv:
        sys.argv[1:1] = ["--results-dir", str(root)]
    main()
