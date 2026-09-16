"""Whole-segment translation draft stabilization, never target-prefix slicing."""
from dataclasses import dataclass, replace
import math
import time


@dataclass(frozen=True)
class DraftState:
    source_start: int
    source_end: int
    source_text: str
    target_text: str
    source_revision: int
    observations: int
    changed_at: float
    is_final: bool
    committed: bool = False


class DraftCommitter:
    def __init__(self, stability_count=2, delay_ms=700, *, clock=time.monotonic):
        if type(stability_count) is not int or stability_count < 1 or delay_ms < 0:
            raise ValueError("Invalid draft commit limits")
        self.stability_count = stability_count
        self.delay_ms = delay_ms
        self.clock = clock
        self.state = None
        self._last_time = None

    def _now(self, now=None):
        value = float(self.clock() if now is None else now)
        if not math.isfinite(value):
            raise ValueError("Local draft time must be finite")
        self._last_time = value if self._last_time is None else max(value, self._last_time)
        return self._last_time

    def propose(self, source_start, source_end, source_text, target_text, *,
                source_revision=0, is_final=False):
        if not 0 <= source_start < source_end or source_end - source_start != len(source_text):
            raise ValueError("Draft must carry its exact whole source span")
        if not source_text.strip() or not target_text.strip() or source_revision < 0:
            raise ValueError("Draft source, target and revision must be valid")
        now = self._now()
        old = self.state
        identity = (source_start, source_end, source_text, source_revision)
        if old is not None and old.committed:
            raise ValueError("Reset a committed draft before proposing another segment")
        same = old is not None and identity == (
            old.source_start, old.source_end, old.source_text, old.source_revision)
        unchanged = same and old.target_text == target_text
        self.state = DraftState(source_start, source_end, source_text, target_text,
                                source_revision, old.observations + 1 if unchanged else 1,
                                old.changed_at if unchanged else now, bool(is_final))
        return self.state

    def ready(self, now=None):
        current = self._now(now)
        return bool(self.state is not None and not self.state.committed and (
            self.state.observations >= self.stability_count
            or (current - self.state.changed_at) * 1000 + 1e-9 >= self.delay_ms))

    def mark_committed(self):
        if not self.ready():
            raise ValueError("Draft has not stabilized")
        self.state = replace(self.state, committed=True)
        return self.state

    def reset(self):
        self.state = None
