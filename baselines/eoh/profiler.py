from baselines.profiler import ProfilerBase


class EoHProfiler(ProfilerBase):
    def __init__(self, run_dir=None, **kwargs):
        super().__init__(run_dir, **kwargs)
        self._cur_gen = 0
