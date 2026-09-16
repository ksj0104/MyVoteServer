"""Mixture consistency for two already estimated mono 16 kHz waveforms.

This is a projection, not a separation model or a fixed frequency split. It
cannot establish that the estimates contain different people. The caller must
supply two time-aligned model estimates and decide whether their quality is
usable. All processing is local to a bounded, at most ten-second input.

Each complex STFT estimate receives a share of ``X - S0 - S1`` proportional to
its squared magnitude. Bins with total power <= epsilon receive equal shares.
A 512-sample periodic Hann window, 128-sample hop, zero center padding and
window-square-normalized overlap-add preserve the original sample count.
NumPy is imported only when the projection is called.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


SAMPLE_RATE = 16_000
MAX_SAMPLES = 10 * SAMPLE_RATE
FFT_SIZE = 512
HOP_SIZE = 128


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Descriptive values, never speaker confidence or a quality decision.

    Residual RMS compares summed estimates with the mixture. ``input`` and
    ``projected`` are before optional gain; ``output`` compares the returned
    sum with ``common_gain * mixture``. Energies are mean square per sample.
    Correlation is centered Pearson correlation of the returned waveforms,
    or None when either waveform has no measurable centered energy.
    """

    input_residual_rms: float
    projected_residual_rms: float
    output_residual_rms: float
    mixture_mean_square: float
    estimate_mean_square: tuple[float, float]
    output_mean_square: tuple[float, float]
    output_correlation: float | None


@dataclass(frozen=True)
class MixtureProjection:
    """Owned, read-only float64 ``sources[2, sample_count]`` and common gain.

    ``sources.sum(axis=0)`` equals ``common_gain * mixture`` within floating
    point reconstruction error. No independent normalization or clipping is
    applied to either output.
    """

    sources: np.ndarray
    common_gain: float
    diagnostics: ProjectionDiagnostics


def _stft(np, signals, window):
    # The extra right padding completes the last hop for every input length.
    padding = ((0, 0), (FFT_SIZE // 2, FFT_SIZE // 2 + (-signals.shape[1]) % HOP_SIZE))
    padded = np.pad(signals, padding, mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(padded, FFT_SIZE, axis=-1)
    return np.fft.rfft(frames[:, ::HOP_SIZE, :] * window, axis=-1)


def _istft(np, spectra, window, sample_count):
    frames = np.fft.irfft(spectra, n=FFT_SIZE, axis=-1) * window
    length = (frames.shape[1] - 1) * HOP_SIZE + FFT_SIZE
    signal = np.zeros((2, length), dtype=np.float64)
    weights = np.zeros(length, dtype=np.float64)
    window_squared = window * window
    for index in range(frames.shape[1]):
        start = index * HOP_SIZE
        signal[:, start:start + FFT_SIZE] += frames[:, index, :]
        weights[start:start + FFT_SIZE] += window_squared
    # The center crop excludes the unsupported outermost padded samples.
    region = slice(FFT_SIZE // 2, FFT_SIZE // 2 + sample_count)
    return signal[:, region] / weights[region]


def project_mixture_consistency(mixture: np.ndarray, estimates: np.ndarray, *,
                                sample_rate: int = SAMPLE_RATE,
                                peak_limit: float | None = None,
                                epsilon: float = 1e-12) -> MixtureProjection:
    """Project two model estimates onto their original mono mixture.

    Inputs must be real numeric ndarrays shaped ``(N,)`` and ``(2, N)`` with
    matching lengths, finite samples and ``1 <= N <= 160000``. Their alignment
    is a caller contract; no resampling, delay estimation or channel selection
    occurs here. Input arrays are never modified.

    ``peak_limit=None`` preserves amplitude. A limit in (0, 1] attenuates both
    outputs by the same gain if necessary. It does not alter their relative
    gain or turn this diagnostic transform into a quality gate.

    For numerical stability, all inputs are scaled by their common peak before
    STFT processing; epsilon is a power floor on those normalized STFT bins.
    This keeps the weighting invariant to common input gain. Numeric overflow
    (including unrepresentable diagnostic energies) is an explicit ValueError.
    """
    import numpy as np

    if type(sample_rate) is not int or sample_rate != SAMPLE_RATE:
        raise ValueError("Mixture projection requires mono 16000 Hz waveforms")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be a finite number in (0, 1)")
    if peak_limit is not None and (isinstance(peak_limit, bool)
            or not isinstance(peak_limit, (int, float)) or not math.isfinite(peak_limit)
            or not 0 < peak_limit <= 1):
        raise ValueError("peak_limit must be None or a finite number in (0, 1]")
    # Check shapes and size before allocating converted arrays or FFT frames.
    for value in (mixture, estimates):
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
            raise ValueError("Waveforms must be real numeric ndarrays")
    if mixture.ndim != 1 or not 1 <= mixture.size <= MAX_SAMPLES:
        raise ValueError("Mixture must contain 1-160000 mono samples")
    if estimates.ndim != 2 or estimates.shape != (2, mixture.size):
        raise ValueError("Exactly two estimates must match the mixture sample count")
    if not np.isfinite(mixture).all() or not np.isfinite(estimates).all():
        raise ValueError("Waveforms must contain only finite samples")

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mixture = np.asarray(mixture, dtype=np.float64)
            estimates = np.asarray(estimates, dtype=np.float64)
            scale = max(float(np.max(np.abs(mixture))), float(np.max(np.abs(estimates)))) or 1.0
            signals = np.concatenate((mixture[None, :], estimates), axis=0) / scale
            window = np.hanning(FFT_SIZE + 1)[:-1]
            spectra = _stft(np, signals, window)
            power = np.abs(spectra[1:]) ** 2
            total_power = power.sum(axis=0)
            weight0 = np.full_like(total_power, .5)
            np.divide(power[0], total_power, out=weight0, where=total_power > epsilon)
            residual = spectra[0] - spectra[1] - spectra[2]
            spectra[1] += weight0 * residual
            spectra[2] += (1.0 - weight0) * residual
            projected = _istft(np, spectra[1:], window, mixture.size) * scale
            peak = float(np.max(np.abs(projected)))
            gain = min(1.0, float(peak_limit) / peak) if peak_limit is not None and peak > 0 else 1.0
            output = projected * gain

            def mean_square(values):
                return float(np.mean(values * values))

            def rms(values):
                return math.sqrt(mean_square(values))

            centered = output - output.mean(axis=1, keepdims=True)
            centered_peak = np.max(np.abs(centered), axis=1)
            roundoff = 8 * np.finfo(np.float64).eps * np.max(np.abs(output), axis=1)
            correlation = None
            if bool(np.all(centered_peak > roundoff)):
                # Constant signals acquire tiny OLA roundoff. It is arithmetic
                # noise, not meaningful variance; no acoustic quality gate.
                normalized = centered / centered_peak[:, None]
                normalized /= np.sqrt(np.sum(normalized * normalized, axis=1))[:, None]
                correlation = float(np.clip(np.sum(normalized[0] * normalized[1]), -1, 1))
            diagnostics = ProjectionDiagnostics(
                input_residual_rms=rms(estimates[0] + estimates[1] - mixture),
                projected_residual_rms=rms(projected[0] + projected[1] - mixture),
                output_residual_rms=rms(output[0] + output[1] - gain * mixture),
                mixture_mean_square=mean_square(mixture),
                estimate_mean_square=(mean_square(estimates[0]), mean_square(estimates[1])),
                output_mean_square=(mean_square(output[0]), mean_square(output[1])),
                output_correlation=correlation,
            )
            if not np.isfinite(output).all():
                raise ValueError("Projection did not produce finite waveforms")
    except (FloatingPointError, OverflowError):
        raise ValueError("Waveform magnitude exceeds the supported numerical range") from None
    output.setflags(write=False)
    return MixtureProjection(output, gain, diagnostics)
