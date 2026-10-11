"""timing.py -- Tiny stage timer so latency is measured, not guessed."""

import time
from contextlib import contextmanager


class Timings:
    def __init__(self):
        self.stages = {}      # stage name -> seconds
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start)

    def mark(self, name: str):
        """Record seconds elapsed since this Timings object was created."""
        self.stages[name] = time.perf_counter() - self._t0

    def summary(self) -> str:
        return "  ".join(f"{k}={v:.2f}s" for k, v in self.stages.items())
