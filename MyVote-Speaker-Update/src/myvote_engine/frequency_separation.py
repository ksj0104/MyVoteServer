"""Neural-guided, mixture-phase frequency masks for two mono estimates.

This bounded refinement needs two time-aligned source estimates from a speech
separation model. It cannot discover speakers from a fixed frequency split:
different voices share harmonics and frequency bins. Squared estimated STFT
magnitudes define nonnegative Wiener-like masks whose sum is one. Applying the
masks to the ORIGINAL complex mixture keeps its phase and overall amplitude.

The default uses one mask pass, symmetric one-frame time smoothing, and no
frequency smoothing. Optional further passes reanalyse the masked waveforms;
this is mask refinement, not an expectation-maximization algorithm. Neither
mixture consistency nor the descriptive diagnostics establish voice quality.
NumPy loads only when the function is called.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .separation_dsp import FFT_SIZE, HOP_SIZE, MAX_SAMPLES, SAMPLE_RATE, _istft, _stft

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class FrequencyMaskDiagnostics:
    """Descriptive arithmetic measurements, never speaker confidence.

    Input and pre-gain residuals compare the sum with the unscaled mixture;
    output residual compares against ``common_gain * mixture``. Per-source
    correction RMS compares the pre-gain output with the corresponding input
    estimate. The ambiguity fraction measures mixture STFT power in bins with
    BOTH final masks in [0.4, 0.6]; it is None for a silent mixture. These values
    cannot distinguish two voices from noise, leakage or duplicated sources.

    Radii count STFT frames/bins. Additional refinement passes reuse the same
    mixture, after analysing the previous two estimated waveforms.
    """

    input_residual_rms: float
    pre_gain_residual_rms: float
    output_residual_rms: float
    estimate_correction_rms: tuple[float, float]
    mask_ambiguity_power_fraction: float | None
    common_gain: float
    sample_rate: int
    fft_size: int
    hop_size: int
    power_exponent: int
    time_smoothing_radius: int
    frequency_smoothing_radius: int
    refinement_iterations: int
    epsilon: float


@dataclass(frozen=True)
class FrequencyMaskSeparation:
    """Owned, read-only float64 sources shaped (2, input sample count).

    The sum equals ``common_gain * mixture`` within floating point error.
    There is no independent source normalization, clipping, identity decision,
    resampling, delay estimation or source-channel permutation tracking.
    """

    sources: np.ndarray
    common_gain: float
    diagnostics: FrequencyMaskDiagnostics


def _smooth_axis(np, values, radius, axis):
    if radius == 0:
        return values
    # Truncated symmetric boxes use their actual support at either edge.
    # Zero padding with a fixed denominator would attenuate boundary power.
    length = values.shape[axis]
    smoothed = np.zeros_like(values)
    count = np.zeros(length, dtype=np.float64)
    for offset in range(-radius, radius + 1):
        start, stop = max(0, -offset), min(length, length - offset)
        if start >= stop:
            continue
        source = [slice(None)] * values.ndim
        target = list(source)
        target[axis] = slice(start, stop)
        source[axis] = slice(start + offset, stop + offset)
        smoothed[tuple(target)] += values[tuple(source)]
        count[start:stop] += 1
    shape = [1] * values.ndim
    shape[axis] = length
    return smoothed / count.reshape(shape)


def _power_masks(np, guidance, time_radius, frequency_radius, epsilon):
    magnitude = np.abs(guidance)
    # One common peak preserves BETWEEN-source power ratios. Recompute it on
    # each pass: neural output gain and mixture amplitude may differ greatly.
    # Normalising each source separately would erase meaningful power ratios.
    peak = float(np.max(magnitude))
    power = (magnitude / peak) ** 2 if peak else np.zeros_like(magnitude)
    power = _smooth_axis(np, power, time_radius, axis=1)
    power = _smooth_axis(np, power, frequency_radius, axis=2)
    total = power[0] + power[1]
    first = np.full_like(total, 0.5)
    np.divide(power[0], total, out=first, where=total > epsilon)
    return np.stack((first, 1.0 - first))


def refine_frequency_masks(mixture: np.ndarray, estimates: np.ndarray, *,
                           sample_rate: int = SAMPLE_RATE,
                           time_smoothing_radius: int = 1,
                           frequency_smoothing_radius: int = 0,
                           refinement_iterations: int = 0,
                           peak_limit: float | None = None,
                           epsilon: float = 1e-12) -> FrequencyMaskSeparation:
    """Apply power-ratio masks guided by two aligned model estimates.

    Require real, finite numeric ndarrays (N,) and (2, N), mono 16000 Hz and
    1 <= N <= 160000. All lengths, including partial hops, keep their sample
    alignment. The 512-point periodic Hann STFT uses hop 128, zero centre
    padding and window-square-normalised overlap-add. Inputs remain unchanged.

    Symmetric smoothing radii and EXTRA refinement passes are integers 0..2.
    The default time radius 1 uses +/- 8 ms of frame context. Each power mask
    has exponent 2; no tunable-alpha or statistical EM claim is made. Bins
    whose total normalised power is <= epsilon split the mixture equally.

    Both source spectra share one peak before computing powers. Masks are
    invariant to a common gain or sign of the guidance, and the whole result
    scales with the mixture. Silent mixtures always produce silent outputs.
    ``peak_limit`` in (0, 1] optionally attenuates BOTH outputs together; None
    preserves amplitude. Unrepresentable arithmetic raises ValueError.
    """
    import numpy as np

    if type(sample_rate) is not int or sample_rate != SAMPLE_RATE:
        raise ValueError("Frequency masks require mono 16000 Hz waveforms")
    for name, value in (("time_smoothing_radius", time_smoothing_radius),
                        ("frequency_smoothing_radius", frequency_smoothing_radius),
                        ("refinement_iterations", refinement_iterations)):
        if type(value) is not int or not 0 <= value <= 2:
            raise ValueError(f"{name} must be an integer in 0..2")
    if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
            or not 0 < epsilon < 1 or not math.isfinite(epsilon)):
        raise ValueError("epsilon must be a finite number in (0, 1)")
    if peak_limit is not None and (isinstance(peak_limit, bool)
            or not isinstance(peak_limit, (int, float)) or not 0 < peak_limit <= 1
            or not math.isfinite(peak_limit)):
        raise ValueError("peak_limit must be None or a finite number in (0, 1]")
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
            mixture_scale = float(np.max(np.abs(mixture))) or 1.0
            estimate_scale = float(np.max(np.abs(estimates))) or 1.0
            signals = np.concatenate((mixture[None, :] / mixture_scale,
                                      estimates / estimate_scale), axis=0)
            window = np.hanning(FFT_SIZE + 1)[:-1]
            spectra = _stft(np, signals, window)
            mixture_spectrum, guidance = spectra[0], spectra[1:]
            for iteration in range(refinement_iterations + 1):
                masks = _power_masks(np, guidance, time_smoothing_radius,
                                     frequency_smoothing_radius, float(epsilon))
                reconstructed = _istft(np, masks * mixture_spectrum,
                                       window, mixture.size)
                if iteration < refinement_iterations:
                    guidance = _stft(np, reconstructed, window)
            projected = reconstructed * mixture_scale
            peak = float(np.max(np.abs(projected)))
            gain = min(1.0, float(peak_limit) / peak) if peak_limit is not None and peak else 1.0
            output = projected * gain

            def rms(values):
                # Avoid squaring large finite waveforms and avoid losing the
                # diagnostic merely because their squares are subnormal.
                peak_value = float(np.max(np.abs(values)))
                return peak_value * math.sqrt(float(np.mean((values / peak_value) ** 2))) if peak_value else 0.0

            mixture_power = np.abs(mixture_spectrum) ** 2
            total_mixture_power = float(np.sum(mixture_power))
            ambiguous = (masks[0] >= 0.4) & (masks[0] <= 0.6)
            ambiguity = (float(np.sum(mixture_power[ambiguous])) / total_mixture_power
                         if total_mixture_power else None)
            diagnostics = FrequencyMaskDiagnostics(
                input_residual_rms=rms(estimates[0] + estimates[1] - mixture),
                pre_gain_residual_rms=rms(projected[0] + projected[1] - mixture),
                output_residual_rms=rms(output[0] + output[1] - gain * mixture),
                estimate_correction_rms=(rms(projected[0] - estimates[0]),
                                         rms(projected[1] - estimates[1])),
                mask_ambiguity_power_fraction=ambiguity,
                common_gain=gain, sample_rate=sample_rate, fft_size=FFT_SIZE,
                hop_size=HOP_SIZE, power_exponent=2,
                time_smoothing_radius=time_smoothing_radius,
                frequency_smoothing_radius=frequency_smoothing_radius,
                refinement_iterations=refinement_iterations, epsilon=float(epsilon),
            )
            numeric_diagnostics = (diagnostics.input_residual_rms,
                                   diagnostics.pre_gain_residual_rms,
                                   diagnostics.output_residual_rms,
                                   *diagnostics.estimate_correction_rms, gain)
            if not np.isfinite(output).all() or not all(math.isfinite(value) for value in numeric_diagnostics):
                raise ValueError("Waveform magnitude exceeds the supported numerical range")
    except (FloatingPointError, OverflowError):
        raise ValueError("Waveform magnitude exceeds the supported numerical range") from None
    output.setflags(write=False)
    return FrequencyMaskSeparation(output, gain, diagnostics)
