"""Experimental neural + time-frequency separation, independent of captions.

This callable pipeline applies the frequency algorithm to actual model outputs.
It produces two anonymous audio estimates, never speaker IDs or clean-speech
evidence. Run its bounded synchronous call on a dedicated worker. It is not
automatically enabled in the real-time gateway: that needs separate speech
quality, channel continuity and multi-speaker caption validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import TYPE_CHECKING, Iterable

from .separation import LocalConvTasNetSeparation, SpeechSeparation, SAMPLE_RATE, NS

if TYPE_CHECKING:
    from .frequency_separation import FrequencyMaskDiagnostics


@dataclass(frozen=True)
class FrequencySeparationResult:
    sources: tuple[tuple[float, ...], tuple[float, ...]]
    window_start_ns: int
    window_end_ns: int
    model_identity: str
    algorithm_identity: str
    unmodeled_tail_samples: int
    model_runtime_ms: float
    frequency_runtime_ms: float
    runtime_ms: float
    model_l1_gain: float
    common_gain: float
    frequency_diagnostics: FrequencyMaskDiagnostics
    model_diagnostics: tuple[str, ...]
    torch_threads: int
    device: str


class FrequencySpeechSeparation:
    """Apply shared model gain correction and Wiener masks to the original mix.

    One call consumes 1..10 seconds of normalized mono 16k PCM. Symmetric STFT
    context and a noncausal base model make this a chunk algorithm, not an 8ms
    latency guarantee. The original model's unmodeled tail metadata survives
    even though frequency reconstruction fills those samples with mixture audio.
    No output is sent to a speaker tracker by this class.
    """

    def __init__(self, separator: SpeechSeparation, *, time_smoothing_radius: int = 1,
                 frequency_smoothing_radius: int = 0, refinement_iterations: int = 0,
                 peak_limit: float | None = .95):
        for value in (time_smoothing_radius, frequency_smoothing_radius, refinement_iterations):
            if type(value) is not int or not 0 <= value <= 2:
                raise ValueError("Smoothing radii and extra passes must be integers in 0..2")
        if peak_limit is not None and (isinstance(peak_limit, bool)
                or not isinstance(peak_limit, (int, float)) or not 0 < peak_limit <= 1
                or not math.isfinite(peak_limit)):
            raise ValueError("peak_limit must be None or a finite number in (0, 1]")
        self._separator = separator
        self._options = dict(time_smoothing_radius=time_smoothing_radius,
                             frequency_smoothing_radius=frequency_smoothing_radius,
                             refinement_iterations=refinement_iterations, peak_limit=peak_limit)
        self._lock = threading.Lock()

    @property
    def identity(self) -> str:
        return self._separator.identity + ":wiener-power2-stft512-hop128-v1"

    def preload(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("frequency separation worker is already in use")
        try:
            preload = getattr(self._separator, "preload", None)
            if preload is not None:
                preload()
        finally:
            self._lock.release()

    def separate(self, samples: Iterable[float], sample_rate: int = SAMPLE_RATE,
                 window_start_ns: int = 0) -> FrequencySeparationResult:
        waveform = LocalConvTasNetSeparation._pcm(samples, sample_rate, window_start_ns)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("frequency separation worker is already in use")
        try:
            import numpy as np
            from .frequency_separation import refine_frequency_masks

            started = time.perf_counter_ns()
            result = self._separator.separate(waveform, sample_rate=sample_rate,
                                              window_start_ns=window_start_ns)
            expected_end = window_start_ns + len(waveform) * NS // SAMPLE_RATE
            if (result.window_start_ns != window_start_ns or result.window_end_ns != expected_end
                    or type(result.unmodeled_tail_samples) is not int
                    or not 0 <= result.unmodeled_tail_samples < 16):
                raise ValueError("Base separator changed the audio timeline or tail contract")
            mixture = np.asarray(waveform, dtype=np.float64)
            estimates = np.asarray(result.sources, dtype=np.float64)
            if estimates.shape != (2, len(waveform)) or not np.isfinite(estimates).all():
                raise ValueError("Base separator must return two finite aligned estimates")
            dsp_started = time.perf_counter_ns()
            # Asteroid's common L1 normalization, evaluated without summing
            # potentially huge raw values. Never normalize channels separately.
            peak = float(np.max(np.abs(estimates)))
            mixture_l1 = float(np.sum(np.abs(mixture)))
            if peak and mixture_l1:
                normalized = estimates / peak
                multiplier = mixture_l1 / float(np.sum(np.abs(normalized)))
                gain = multiplier / peak
                if not math.isfinite(gain):
                    raise ValueError("Model gain exceeds the supported numerical range")
                normalized = normalized * multiplier
            else:
                normalized = np.zeros_like(estimates)
                gain = 0.0
            refined = refine_frequency_masks(mixture, normalized, sample_rate=sample_rate,
                                              **self._options)
            frequency_ms = (time.perf_counter_ns() - dsp_started) / 1e6
            sources = (tuple(refined.sources[0].tolist()), tuple(refined.sources[1].tolist()))
            return FrequencySeparationResult(
                sources, window_start_ns, expected_end, result.model_identity,
                "wiener-power2-stft512-hop128-v1", result.unmodeled_tail_samples,
                result.runtime_ms, frequency_ms, (time.perf_counter_ns() - started) / 1e6,
                gain, refined.common_gain, refined.diagnostics, result.diagnostics,
                result.torch_threads, result.device)
        finally:
            self._lock.release()
