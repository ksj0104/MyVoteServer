"""Bounded original-frame receipt times on one server's monotonic clock.

Source timestamps are lookup keys only. Never infer a shared capture/server
clock or interpolate time across a gap. No PCM is retained here.
"""
from collections import deque
from dataclasses import dataclass
import math

from .audio import AudioFrame


INGRESS_SCOPE = "server_application_audio_ingress_to_translation_acceptance"


@dataclass(frozen=True)
class _Receipt:
    track: str
    epoch: str
    start_ns: int
    end_ns: int
    sequence: int
    received_at: float
    continuity: int


class IngressTimeline:
    def __init__(self, max_frames: int = 8192):
        if type(max_frames) is not int or not 1 <= max_frames <= 200000:
            raise ValueError("Ingress history must retain 1..200000 frames")
        self.max_frames = max_frames
        self._frames: deque[_Receipt] = deque()
        self._last: _Receipt | None = None
        self._continuity = 0
        self._gap_pending = False
        self.observed = self.evicted = self.lookup_misses = 0

    def gap(self):
        # Old in-flight clauses still refer to their original receipt times.
        self._gap_pending = True

    def observe(self, frame: AudioFrame, received_at: float):
        if (isinstance(received_at, bool) or not isinstance(received_at, (int, float))
                or not math.isfinite(received_at)):
            raise ValueError("Receipt must use a finite server monotonic time")
        last = self._last
        if last is not None:
            if frame.track_id != last.track or frame.start_time_ns < last.end_ns:
                raise ValueError("Ingress cannot change track or replay source audio")
            if received_at < last.received_at:
                raise ValueError("Server receipt clock moved backwards")
            if (self._gap_pending or frame.capture_epoch != last.epoch
                    or frame.sequence != last.sequence + 1 or frame.start_time_ns != last.end_ns):
                self._continuity += 1
        entry = _Receipt(frame.track_id, frame.capture_epoch, frame.start_time_ns,
                         frame.end_time_ns, frame.sequence, received_at, self._continuity)
        if len(self._frames) == self.max_frames:
            self._frames.popleft()
            self.evicted += 1
        self._frames.append(entry)
        self._last = entry
        self._gap_pending = False
        self.observed += 1

    def anchor(self, track: str, epoch: str, start_ns: int, end_ns: int) -> float | None:
        """Receipt of the frame containing the last word's end, if covered.

        A boundary end belongs to the preceding frame. Require the entire
        clause to remain in one retained, continuous source interval.
        """
        if (type(start_ns) is not int or type(end_ns) is not int
                or start_ns < 0 or end_ns <= start_ns):
            self.lookup_misses += 1
            return None
        endpoint = None
        for item in reversed(self._frames):
            if endpoint is None:
                if item.start_ns < end_ns <= item.end_ns and item.track == track and item.epoch == epoch:
                    endpoint = item
                else:
                    continue
            if item.continuity != endpoint.continuity:
                break
            if item.start_ns <= start_ns:
                return endpoint.received_at
        self.lookup_misses += 1
        return None

    def summary(self):
        return dict(scope=INGRESS_SCOPE, observed_frames=self.observed,
                    retained_frames=len(self._frames), evicted_frames=self.evicted,
                    lookup_misses=self.lookup_misses, max_frames=self.max_frames)
