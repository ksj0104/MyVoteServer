"""Bounded client-local source timestamp age diagnostics.

Only this computer's source and receipt clocks are subtracted. Native endpoint
timestamps are not calibrated acoustic presentation times. No server/GPU clock,
screen refresh estimate, PCM, or caption text is retained by this module.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import time
from typing import Callable

from .audio import AudioFrame


KINDS = ("source.first", "source.stable", "translation.first", "translation.completed")
SOURCE_MODES = ("native_wasapi_loopback", "paced_wav", "injected_test_source")
MAX_INTEGER = 2**63 - 1
_READ_CLOCK = object()


def _integer(value, *, minimum=0):
    return type(value) is int and minimum <= value <= MAX_INTEGER


def _identifier(value, maximum=512):
    return isinstance(value, str) and 0 < len(value) <= maximum and value.strip() and value.isprintable()


@dataclass
class _Range:
    track_id: str
    capture_epoch: str
    origin_ns: int
    start_ns: int
    end_ns: int
    sequence: int


class ClientLatencyTracker:
    """Record the first nonempty receipt for each metric/segment/revision.

    ``clock`` and ``now_ns`` must use integer nanoseconds in the selected source
    domain: Windows QPC for native, perf_counter_ns for paced WAV. A supplied
    callable is an explicit clock injection, not automatically verified timing.
    Register an origin and each actually supplied AudioFrame before captions.
    Only wholly contained, uniquely identified contiguous ranges are measured.
    """

    def __init__(self, source_mode: str, *, clock: Callable[[], int] | None = None,
                 max_samples: int = 10000, max_identities: int = 10000,
                 max_timeline_ranges: int = 256):
        if source_mode not in SOURCE_MODES:
            raise ValueError("Unknown source mode")
        if clock is not None and not callable(clock):
            raise ValueError("clock must be callable")
        for name, value, maximum in (("max_samples", max_samples, 10000),
                                     ("max_identities", max_identities, 10000),
                                     ("max_timeline_ranges", max_timeline_ranges, 10000)):
            if not _integer(value, minimum=1) or value > maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        self.source_mode = source_mode
        self.clock_kind = {"native_wasapi_loopback": "windows_qpc_ns",
                           "paced_wav": "perf_counter_ns",
                           "injected_test_source": "unmeasured"}[source_mode]
        self._clock = clock
        self._injected_clock = clock is not None
        self.max_samples = max_samples
        self.max_identities = max_identities
        self.max_timeline_ranges = max_timeline_ranges
        self._origins: dict[str, int] = {}
        self._ranges: list[_Range] = []
        self._tails: dict[str, _Range] = {}
        self._invalid_tracks: set[str] = set()
        self._timeline_exhausted = False
        self._seen: dict[tuple[str, str, int], set[str]] = {}
        self._samples: list[tuple[str, int]] = []
        self._aggregates = {kind: {"count": 0, "nonnegative_count": 0, "negative_count": 0,
                                  "nonnegative_sum_ns": 0, "min_ns": None, "max_ns": None}
                            for kind in KINDS}
        self._counts = Counter()
        self._problems = Counter()
        self._last_clock_ns: int | None = None

    def _drop(self, reason):
        self._counts["unmeasured_caption_events"] += 1
        self._problems[reason] += 1
        return None

    def _set_origin(self, origin_ns, track_id):
        if not _integer(origin_ns) or not _identifier(track_id, 256):
            raise ValueError("Origin must be nonnegative integer ns and track ID must be bounded text")
        if track_id not in self._origins and len(self._origins) >= self.max_timeline_ranges:
            raise ValueError("Origin track limit reached")
        previous = self._origins.get(track_id)
        if previous is not None and previous != origin_ns:
            self._tails.pop(track_id, None)
            self._counts["origin_changes"] += 1
        self._origins[track_id] = origin_ns

    def set_native_origin(self, qpc_origin_100ns: int, *, track_id: str = "system") -> None:
        if self.source_mode != "native_wasapi_loopback" or not _integer(qpc_origin_100ns):
            raise ValueError("Native origin requires native mode and integer QPC 100-ns units")
        self._set_origin(qpc_origin_100ns * 100, track_id)

    def set_wav_origin(self, perf_counter_origin_ns: int, *, track_id: str = "system") -> None:
        if self.source_mode != "paced_wav":
            raise ValueError("WAV origin requires paced_wav mode")
        self._set_origin(perf_counter_origin_ns, track_id)

    def now_ns(self) -> int | None:
        """Read the selected local clock. Clock failures stay outside text flow."""
        if self.source_mode == "injected_test_source":
            return None
        try:
            if self._clock is None:
                if self.source_mode == "native_wasapi_loopback":
                    from .capture import WindowsQpcClock
                    qpc = WindowsQpcClock()
                    self._clock = lambda: qpc.now_100ns() * 100
                else:
                    self._clock = time.perf_counter_ns
            value = self._clock()
            if not _integer(value):
                raise ValueError("Invalid clock value")
            return value
        except Exception:
            self._problems["clock_read_failed"] += 1
            return None

    def note_gap(self, track_id: str, reason: str = "source_gap") -> None:
        """End a continuity range; a later frame opens a new range.

        A delayed caption entirely inside the old range remains measurable.
        Arbitrary reason strings are not retained as unbounded counter keys.
        """
        if not _identifier(track_id, 256) or not _identifier(reason, 256):
            self._problems["invalid_gap"] += 1
            return
        self._tails.pop(track_id, None)
        self._counts["source_gaps"] += 1

    def note_error(self, *, count: int = 1) -> None:
        """Count rejected/failed result events; do not supply caption metrics."""
        if not _integer(count, minimum=1):
            raise ValueError("Error count must be a positive integer")
        self._counts["error_events_unmeasured"] += count

    def observe_frame(self, frame: AudioFrame) -> bool:
        """Register supplied source coverage without retaining any PCM."""
        self._counts["frame_events"] += 1
        if (not isinstance(frame, AudioFrame) or not _identifier(frame.track_id, 256)
                or not _identifier(frame.capture_epoch, 256)
                or not _integer(frame.start_time_ns) or not _integer(frame.end_time_ns)
                or not _integer(frame.sequence)):
            self._problems["invalid_frame"] += 1
            return False
        if self.source_mode == "injected_test_source":
            self._counts["injected_frames_unmeasured"] += 1
            return False
        origin = self._origins.get(frame.track_id)
        if origin is None:
            self._problems["frame_origin_missing"] += 1
            return False
        if self._timeline_exhausted or frame.track_id in self._invalid_tracks:
            self._counts["frames_timeline_unavailable"] += 1
            return False
        previous = self._tails.get(frame.track_id)
        if previous is not None and previous.capture_epoch == frame.capture_epoch and previous.origin_ns == origin:
            if frame.start_time_ns < previous.end_ns or frame.sequence <= previous.sequence:
                # Once a track contradicts its own epoch, later labels cannot
                # safely distinguish a replayed timestamp from the old audio.
                self._invalid_tracks.add(frame.track_id)
                self._problems["backwards_or_duplicate_frame"] += 1
                return False
            if frame.start_time_ns == previous.end_ns and frame.sequence == previous.sequence + 1:
                previous.end_ns = frame.end_time_ns
                previous.sequence = frame.sequence
                self._counts["accepted_frames"] += 1
                return True
            self._problems["implicit_source_gap"] += 1
        if len(self._ranges) >= self.max_timeline_ranges:
            # No eviction: reusing an overlapping old epoch must never become
            # falsely unique merely because its history fell out of a buffer.
            self._timeline_exhausted = True
            self._problems["timeline_capacity_exhausted"] += 1
            return False
        current = _Range(frame.track_id, frame.capture_epoch, origin,
                         frame.start_time_ns, frame.end_time_ns, frame.sequence)
        self._ranges.append(current)
        self._tails[frame.track_id] = current
        self._counts["accepted_frames"] += 1
        return True

    def observe_caption(self, segment_id: str, source_revision: int, start_ns: int,
                        end_ns: int, kind: str, text_nonempty: bool | str, *,
                        track_id: str = "system", capture_epoch: str | None = None,
                        now_ns: int | None | object = _READ_CLOCK) -> dict | None:
        """Return one first-receipt diagnostic, or None with a counted reason.

        Pass only projection-accepted, successful events. For a stable source
        event call source.first and source.stable; for completion call
        translation.first and translation.completed, reusing one receipt time.
        A string is stripped only to check nonempty text and is never retained.
        A bool must already represent ``bool(validated_text.strip())``.
        """
        self._counts["caption_events"] += 1
        if kind not in KINDS:
            return self._drop("unsupported_metric_kind")
        if (not _identifier(segment_id) or not _identifier(track_id, 256)
                or not _integer(source_revision, minimum=1)
                or not _integer(start_ns) or not _integer(end_ns) or end_ns <= start_ns
                or capture_epoch is not None and not _identifier(capture_epoch, 256)):
            return self._drop("invalid_caption_identity_or_interval")
        if isinstance(text_nonempty, str):
            if len(text_nonempty) > 24000:
                return self._drop("invalid_text_nonempty")
            nonempty = bool(text_nonempty.strip())
        elif type(text_nonempty) is bool:
            nonempty = text_nonempty
        else:
            return self._drop("invalid_text_nonempty")
        if not nonempty:
            self._counts["empty_text_events"] += 1
            return None
        key = (track_id, segment_id, source_revision)
        seen = self._seen.get(key)
        if seen is None:
            if len(self._seen) >= self.max_identities:
                return self._drop("identity_capacity_exhausted")
            seen = self._seen[key] = set()
        if kind in seen:
            self._counts["duplicate_first_events"] += 1
            return None
        # Consume the first text even when its timestamp is invalid/missing.
        # Measuring a later repeat would misrepresent the first receipt time.
        seen.add(kind)
        if self.source_mode == "injected_test_source":
            return self._drop("injected_source_unmeasured")
        if self._timeline_exhausted or track_id in self._invalid_tracks:
            return self._drop("source_timeline_invalid")
        matches = [item for item in self._ranges if item.track_id == track_id
                   and (capture_epoch is None or capture_epoch == item.capture_epoch)
                   and item.start_ns <= start_ns and end_ns <= item.end_ns]
        if not matches:
            return self._drop("source_interval_not_in_one_known_range")
        if len(matches) != 1:
            return self._drop("ambiguous_capture_epoch")
        if any(item is not matches[0] and item.track_id == track_id
               and (capture_epoch is None or item.capture_epoch == capture_epoch)
               and item.start_ns < end_ns and start_ns < item.end_ns for item in self._ranges):
            return self._drop("ambiguous_capture_epoch")
        received = self.now_ns() if now_ns is _READ_CLOCK else now_ns
        if not _integer(received):
            return self._drop("receipt_clock_unavailable_or_invalid")
        if self._last_clock_ns is not None and received < self._last_clock_ns:
            return self._drop("receipt_clock_went_backwards")
        self._last_clock_ns = received
        source = matches[0]
        signed_ns = received - source.origin_ns - end_ns
        aggregate = self._aggregates[kind]
        aggregate["count"] += 1
        aggregate["min_ns"] = signed_ns if aggregate["min_ns"] is None else min(aggregate["min_ns"], signed_ns)
        aggregate["max_ns"] = signed_ns if aggregate["max_ns"] is None else max(aggregate["max_ns"], signed_ns)
        if signed_ns < 0:
            aggregate["negative_count"] += 1
            self._problems["negative_source_timestamp_age"] += 1
        else:
            aggregate["nonnegative_count"] += 1
            aggregate["nonnegative_sum_ns"] += signed_ns
        if len(self._samples) < self.max_samples:
            self._samples.append((kind, signed_ns))
        else:
            self._counts["quantile_samples_not_retained"] += 1
        self._counts["measured_events"] += 1
        return {
            "schema": "myvote.client_latency_sample", "schema_version": 1,
            "kind": kind, "segment_id": segment_id, "source_revision": source_revision,
            "track_id": track_id, "capture_epoch": source.capture_epoch,
            "start_ns": start_ns, "end_ns": end_ns,
            "source_mode": self.source_mode, "clock_kind": self.clock_kind,
            "client_received_clock_ns": received, "timeline_origin_clock_ns": source.origin_ns,
            "signed_source_timestamp_to_client_result_ms": signed_ns / 1e6,
            "source_timestamp_to_client_result_ms": signed_ns / 1e6 if signed_ns >= 0 else None,
            "status": "nonnegative_diagnostic" if signed_ns >= 0 else "negative_timestamp_diagnostic",
            "diagnostic_only": True,
        }

    def summary(self) -> dict:
        metrics = {}
        for kind, aggregate in self._aggregates.items():
            values = sorted(value for sample_kind, value in self._samples if sample_kind == kind and value >= 0)
            def quantile(q):
                if not values:
                    return None
                position = (len(values) - 1) * q
                left, right = math.floor(position), math.ceil(position)
                return (values[left] + (values[right] - values[left]) * (position - left)) / 1e6
            metrics[kind] = {
                "count": aggregate["count"], "nonnegative_count": aggregate["nonnegative_count"],
                "negative_count": aggregate["negative_count"],
                "signed_min_ms": None if aggregate["min_ns"] is None else aggregate["min_ns"] / 1e6,
                "signed_max_ms": None if aggregate["max_ns"] is None else aggregate["max_ns"] / 1e6,
                "nonnegative_mean_ms": (aggregate["nonnegative_sum_ns"] / aggregate["nonnegative_count"] / 1e6
                                        if aggregate["nonnegative_count"] else None),
                "retained_nonnegative_quantile_samples": len(values),
                "nonnegative_p50_ms": quantile(.5), "nonnegative_p95_ms": quantile(.95),
            }
        return {
            "schema": "myvote.client_latency", "schema_version": 1,
            "source_mode": self.source_mode, "clock_kind": self.clock_kind,
            "clock_provider": "injected_callable" if self._injected_clock else "source_mode_default",
            "metric": "source_timestamp_to_client_result_ms", "diagnostic_only": True,
            "reference_point": "end of caption source interval; not utterance onset or final acoustic endpoint",
            "counts": dict(self._counts), "problems": dict(self._problems), "metrics": metrics,
            "memory": {"retained_samples": len(self._samples), "max_samples": self.max_samples,
                       "retained_segment_revisions": len(self._seen), "max_identities": self.max_identities,
                       "timeline_ranges": len(self._ranges), "max_timeline_ranges": self.max_timeline_ranges,
                       "timeline_capacity_exhausted": self._timeline_exhausted},
            "quantile_scope": "First bounded samples across all four kinds; no eviction or claim of representative full-session quantiles",
            "assumptions": [
                "Origins and receipt times share this client computer's source clock; no server absolute time is subtracted",
                "Native origin is CaptureFormat.qpc_origin_100ns; normalized timestamps remain relative to that origin",
                "WAV pacing and receipt must both use perf_counter_ns and the identical registered origin",
                "Only projection-accepted successful results and known supplied source intervals are measured",
            ],
            "limitations": [
                "Native endpoint timestamps are uncalibrated diagnostics, including known possible negative offsets",
                "Negative values remain signed diagnostics and are excluded from nonnegative means and quantiles",
                "Injected sources are unmeasured; injected clock callables require caller-verified units and domain",
                "Receipt time excludes subsequent Dispatcher, WPF rendering, physical scanout and acoustic presentation",
                "No model, GPU, server processing or cross-computer clock subtraction is inferred",
            ],
        }
