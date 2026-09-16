"""Finalize retained audio normally when both PCM and transcription go idle.

This optional session adapter handles an open RPC whose client stops sending
PCM without sending a stop message. It never promotes provisional words itself:
the original builder checkpoint, ASR finalization and stabilizer do that work.
Per-speaker semantic inactivity while audio continues belongs to the router.
"""

import asyncio
from collections import OrderedDict
import hashlib
import math
import time


def install_transcript_idle(streaming_module, *, inactivity_flush_s=2):
    if (type(inactivity_flush_s) not in (int, float) or not math.isfinite(inactivity_flush_s)
            or not 0 <= inactivity_flush_s <= 120):
        raise ValueError("Transcription inactivity must be finite seconds in [0, 120]")
    metadata = {"semantic_transcript_idle": "normal-asr-finalization-v1",
                "semantic_transcript_idle_ms": int(inactivity_flush_s * 1000),
                "semantic_transcript_idle_scope": "no-pcm-and-no-transcription-change"}
    original = streaming_module.StreamingSession
    if getattr(original, "_myvote_transcript_idle_v1", False):
        if original._myvote_transcript_idle_s != inactivity_flush_s:
            raise ValueError("Transcription inactivity already installed with a different interval")
        return metadata
    if not inactivity_flush_s:
        return metadata

    class TranscriptIdleSession(original):
        _myvote_transcript_idle_v1 = True
        _myvote_transcript_idle_s = inactivity_flush_s
        _myvote_transcript_idle_original = original

        def __init__(self, *args, **kwargs):
            self._transcript_idle_task = None
            self._transcript_idle_wake = asyncio.Event()
            self._transcript_idle_stopping = False
            self._transcript_idle_processing = 0
            self._transcript_idle_feeding = 0
            self._transcript_idle_input_at = None
            self._transcript_idle_text_at = None
            self._transcript_idle_signatures = OrderedDict()
            super().__init__(*args, **kwargs)

        def _wake_transcript_idle(self):
            self._transcript_idle_wake.set()
            if (self._transcript_idle_stopping or self._closed or self._finishing
                    or self._transcript_idle_input_at is None or self._transcript_idle_text_at is None):
                return
            if self._transcript_idle_task is None or self._transcript_idle_task.done():
                self._transcript_idle_task = asyncio.create_task(
                    self._transcript_idle_loop(), name="myvote-transcript-inactivity")

        async def _stop_transcript_idle(self, *, permanent=False):
            if permanent:
                self._transcript_idle_stopping = True
            task, self._transcript_idle_task = self._transcript_idle_task, None
            if task is not None and task is not asyncio.current_task():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self._transcript_idle_wake.set()

        async def feed(self, frame, speech_probability):
            received_at = time.monotonic()
            self._transcript_idle_feeding += 1
            try:
                result = await super().feed(frame, speech_probability)
                self._transcript_idle_input_at = max(self._transcript_idle_input_at or received_at,
                                                     received_at)
                return result
            finally:
                self._transcript_idle_feeding -= 1
                self._wake_transcript_idle()

        async def _emit(self, kind, segment_id=None, **data):
            observed_at = time.monotonic()
            if kind in ("audio.gap", "asr.skipped_overload"):
                self._transcript_idle_input_at = self._transcript_idle_text_at = None
                self._transcript_idle_signatures.clear()
                await self._stop_transcript_idle()
            result = await super()._emit(kind, segment_id, **data)
            text = data.get("text") if kind == "transcript.updated" else None
            if isinstance(text, str) and text.strip():
                signature = hashlib.sha256(text.strip().encode("utf-8")).digest()
                if self._transcript_idle_signatures.get(segment_id) != signature:
                    self._transcript_idle_signatures[segment_id] = signature
                    self._transcript_idle_signatures.move_to_end(segment_id)
                    while len(self._transcript_idle_signatures) > 32:
                        self._transcript_idle_signatures.popitem(last=False)
                    self._transcript_idle_text_at = observed_at
                    self._wake_transcript_idle()
            return result

        async def _process_window(self, window, *, queued_at=None):
            self._transcript_idle_processing += 1
            try:
                return await super()._process_window(window, queued_at=queued_at)
            finally:
                self._transcript_idle_processing -= 1
                self._wake_transcript_idle()

        async def _transcript_idle_loop(self):
            try:
                while not (self._transcript_idle_stopping or self._closed or self._finishing):
                    self._transcript_idle_wake.clear()
                    if (self._transcript_idle_input_at is None or self._transcript_idle_text_at is None
                            or getattr(self.builder, "_segment_id", None) is None):
                        return
                    busy = (self._transcript_idle_feeding or self._transcript_idle_processing
                            or getattr(self, "_boundary_processing", False) or self._pending)
                    remaining = max(self._transcript_idle_input_at,
                                    self._transcript_idle_text_at) + inactivity_flush_s - time.monotonic()
                    if remaining <= 0 and not busy:
                        # No await between final snapshot and reset: a later
                        # PCM frame starts a fresh builder segment, never changes
                        # the retained immutable window being enqueued here.
                        windows = self.builder.flush(reason="transcript_inactivity")
                        self.builder.reset()
                        self._transcript_idle_text_at = None
                        for window in windows:
                            await self._enqueue(window)
                        self.counts["transcript_idle_checkpoints"] = (
                            self.counts.get("transcript_idle_checkpoints", 0) + len(windows))
                        return
                    try:
                        if remaining <= 0:
                            # ASR completion/feed/queue progress wakes us. Do not
                            # poll a native inference call or cancel it on age.
                            await self._transcript_idle_wake.wait()
                        else:
                            await asyncio.wait_for(self._transcript_idle_wake.wait(), timeout=remaining)
                    except TimeoutError:
                        pass
            except asyncio.CancelledError:
                raise
            except Exception:
                # Do not log raw text or arbitrary sink/provider error messages.
                self.counts["transcript_idle_errors"] = self.counts.get("transcript_idle_errors", 0) + 1
                await self._emit("pipeline.failed", error="transcript_idle_checkpoint_failed")

        async def finish(self, *args, **kwargs):
            await self._stop_transcript_idle(permanent=True)
            return await super().finish(*args, **kwargs)

        async def close(self):
            await self._stop_transcript_idle(permanent=True)
            return await super().close()

    streaming_module.StreamingSession = TranscriptIdleSession
    return metadata
