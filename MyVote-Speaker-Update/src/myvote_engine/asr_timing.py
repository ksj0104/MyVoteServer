"""Source-caption timing on one process clock, without PCM or word text.

The measured call includes the transcriber adapter, not just model inference.
Dispatch wait includes executor and optional shared ASR scheduler admission.
These diagnostics do not change source times, agreement or deadline policy.
"""
from __future__ import annotations

from dataclasses import dataclass


def _milliseconds(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or end < start:
        return None
    return (end - start) * 1000


@dataclass
class AsrWindowTiming:
    window_id: int
    window_start_ns: int
    window_end_ns: int
    queued_at: float | None
    processing_started_at: float
    dispatch_started_at: float | None = None
    call_started_at: float | None = None
    call_finished_at: float | None = None
    await_finished_at: float | None = None
    cached_from_window_id: int | None = None
    cached_hypothesis: AsrWindowTiming | None = None

    def metrics(self, observed_at: float) -> dict:
        return {
            "schema_version": 1,
            "clock_scope": "server_process_monotonic",
            "window_id": self.window_id,
            "window_start_ns": self.window_start_ns,
            "window_end_ns": self.window_end_ns,
            "queue_ms": _milliseconds(self.queued_at, self.processing_started_at),
            "pre_dispatch_ms": _milliseconds(self.processing_started_at, self.dispatch_started_at),
            "dispatch_wait_ms": _milliseconds(self.dispatch_started_at, self.call_started_at),
            "transcriber_call_ms": _milliseconds(self.call_started_at, self.call_finished_at),
            "resume_wait_ms": _milliseconds(self.call_finished_at, self.await_finished_at),
            "processing_ms": _milliseconds(self.processing_started_at, observed_at),
            "cached_from_window_id": self.cached_from_window_id,
            "cached_hypothesis": (self.cached_hypothesis.metrics(self.cached_hypothesis.await_finished_at)
                if self.cached_hypothesis is not None else None),
        }


@dataclass(frozen=True)
class WordCommitTiming:
    window: AsrWindowTiming
    ready_at: float
    reason: str

    def caption_metrics(self, *, first_word: WordCommitTiming, emitting_window: AsrWindowTiming,
                        caption_end_ns: int, source_created_at: float,
                        ingress_anchor: float | None) -> dict:
        # Only a validated same-process ingress anchor can yield an ingress age.
        # Missing history and future receipts are not replaced by a new budget.
        window = self.window
        return {
            "schema_version": 1,
            "clock_scope": "server_process_monotonic",
            "emitting_window_id": emitting_window.window_id,
            "emitting_window_end_ns": emitting_window.window_end_ns,
            "commit_window": window.metrics(self.ready_at),
            "commit_reason": self.reason,
            "source_playhead_lag_ms": (emitting_window.window_end_ns - caption_end_ns) / 1e6,
            "ingress_to_commit_window_queue_ms": _milliseconds(ingress_anchor, window.queued_at),
            "ingress_to_latest_word_commit_ms": _milliseconds(ingress_anchor, self.ready_at),
            "ingress_to_source_created_ms": _milliseconds(ingress_anchor, source_created_at),
            "post_asr_to_commit_ms": _milliseconds(window.await_finished_at, self.ready_at),
            "first_word_commit_to_source_ms": _milliseconds(first_word.ready_at, source_created_at),
            "latest_word_commit_to_source_ms": _milliseconds(self.ready_at, source_created_at),
        }
