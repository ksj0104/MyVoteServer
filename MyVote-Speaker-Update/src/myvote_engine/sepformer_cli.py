"""Review local SepFormer outputs, optionally beside frequency-mask refinement.

Run with --input mono16k.wav --model-dir local-checkpoints --output-dir new-dir.
Native source-A/source-B keep the model's relative amplitudes. Every review WAV
shares one attenuation gain. Output channels are anonymous, without speaker IDs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time

from .frequency_separation import refine_frequency_masks
from .separate_cli import read_excerpt, sample_position, write_wav
from .separation import NS, SAMPLE_RATE
from .sepformer import LocalSepformerSeparation, MODEL_REPOSITORY, MODEL_REVISION


def _residual_diagnostics(np, mixture, sources):
    """Describe the raw sum error; never repair it or use it as voice quality."""
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        residual = sources.sum(axis=0) - mixture
        return {
            "mixture_rms": float(np.sqrt(np.mean(mixture * mixture))),
            "source_rms": [float(np.sqrt(np.mean(row * row))) for row in sources],
            "source_peak": [float(np.max(np.abs(row))) for row in sources],
            "sum_residual_mean": float(np.mean(residual)),
            "sum_residual_rms": float(np.sqrt(np.mean(residual * residual))),
            "sum_residual_peak": float(np.max(np.abs(residual))),
        }


def _checked_sources(np, result, mixture, start_ns, identity):
    sources = np.asarray(result.sources)
    if (sources.dtype.kind not in "fiu" or sources.shape != (2, mixture.size)
            or not np.isfinite(sources).all()):
        raise ValueError("SepFormer must return two finite real sources matching the excerpt")
    if (type(result.window_start_ns) is not int or result.window_start_ns != start_ns
            or type(result.window_end_ns) is not int
            or result.window_end_ns != start_ns + mixture.size * NS // SAMPLE_RATE):
        raise ValueError("SepFormer output timestamps must match the selected excerpt")
    if (type(result.unmodeled_tail_samples) is not int
            or not 0 <= result.unmodeled_tail_samples < 8):
        raise ValueError("SepFormer unmodeled tail must be 0..7 samples")
    if (result.model_identity != identity or type(result.runtime_ms) not in (int, float)
            or not math.isfinite(result.runtime_ms) or result.runtime_ms < 0
            or type(result.torch_threads) is not int or not 1 <= result.torch_threads <= 8
            or result.device != "cpu"):
        raise ValueError("SepFormer output identity or runtime metadata is invalid")
    sources = sources.astype(np.float64, copy=False)
    if not np.any(mixture) and np.any(sources):
        raise ValueError("SepFormer returned nonzero audio for an exactly silent mixture")
    return sources


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True, help="Mono 16000 Hz PCM16 WAV")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Existing pinned encoder.ckpt, masknet.ckpt and decoder.ckpt directory")
    parser.add_argument("--start-s", type=sample_position, default=0)
    parser.add_argument("--duration-s", type=sample_position, default=8 * SAMPLE_RATE)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; never overwrites an earlier result")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=1,
                        help="CPU threads for this dedicated command process")
    parser.add_argument("--compare-frequency", action="store_true",
                        help="Also save frequency-A/B using the native estimates as mask guidance")
    args = parser.parse_args(argv)
    if not SAMPLE_RATE <= args.duration_s <= 10 * SAMPLE_RATE:
        parser.error("--duration-s must select 1..10 seconds")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("--output-dir must be a new directory")
    raw = read_excerpt(args.input, args.start_s, args.duration_s)
    verification_started = time.perf_counter_ns()
    separator = LocalSepformerSeparation(args.model_dir)
    model_verification_ms = (time.perf_counter_ns() - verification_started) / 1e6

    import numpy as np
    import torch
    # Only this dedicated CLI owns process-wide thread configuration.
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    mixture = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
    preload_started = time.perf_counter_ns()
    separator.preload()
    preload_ms = (time.perf_counter_ns() - preload_started) / 1e6
    start_ns = args.start_s * NS // SAMPLE_RATE
    result = separator.separate(mixture, sample_rate=SAMPLE_RATE, window_start_ns=start_ns)
    sources = _checked_sources(np, result, mixture, start_ns, separator.identity)
    try:
        native_diagnostics = _residual_diagnostics(np, mixture, sources)
    except FloatingPointError as exc:
        raise ValueError("SepFormer diagnostic arithmetic is not representable") from exc
    review = {"mixture.wav": mixture, "source-A.wav": sources[0], "source-B.wav": sources[1]}
    frequency_report = None
    if args.compare_frequency:
        started = time.perf_counter_ns()
        refined = refine_frequency_masks(mixture, sources, sample_rate=SAMPLE_RATE,
            time_smoothing_radius=1, frequency_smoothing_radius=0,
            refinement_iterations=0, peak_limit=None)
        frequency_ms = (time.perf_counter_ns() - started) / 1e6
        review.update({"frequency-A.wav": refined.sources[0], "frequency-B.wav": refined.sources[1]})
        frequency_report = {
            "runtime_ms": frequency_ms,
            "diagnostics": asdict(refined.diagnostics),
            "native_guidance_gain": 1.0,
            "unmodeled_native_tail_samples": result.unmodeled_tail_samples,
            "tail_scope": "Refinement may allocate mixture energy in the unmodeled native tail; it is not new model evidence",
        }
    peak = max(float(np.max(np.abs(values))) for values in review.values())
    gain = min(1.0, .95 / peak) if peak else 1.0
    report = {
        "schema": "myvote.sepformer_separation", "schema_version": 1,
        "input_path": str(args.input.resolve()),
        "selected_pcm16_sha256": hashlib.sha256(raw).hexdigest(),
        "start_sample": args.start_s, "samples": args.duration_s, "sample_rate": SAMPLE_RATE,
        "window_start_ns": result.window_start_ns, "window_end_ns": result.window_end_ns,
        "model_identity": result.model_identity,
        "model_repository": MODEL_REPOSITORY, "model_revision": MODEL_REVISION,
        "model_diagnostics": list(result.diagnostics),
        "unmodeled_tail_samples": result.unmodeled_tail_samples,
        "native_gain": 1.0, "native_mixture_projection_applied": False,
        "native_diagnostics": native_diagnostics,
        "frequency_comparison": frequency_report,
        "review_common_gain": gain, "review_peak_limit": .95,
        "runtime_ms": result.runtime_ms, "preload_ms": preload_ms,
        "model_verification_ms": model_verification_ms,
        "torch_threads": result.torch_threads, "torch_interop_threads": 1,
        "device": result.device, "platform": platform.platform(),
        "timing_scope": "Synchronous local CPU chunk; excludes audio accumulation, ASR, translation and display",
        "live_gateway_enabled": False, "speaker_ids_assigned": False,
        "quality_claim": "Descriptive signal measurements only; no speech purity or separation accuracy claim",
    }
    # Validate metadata before creating output, so bad model results leave no review files.
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name, values in review.items():
        path = args.output_dir / name
        pcm = write_wav(path, values * gain)
        outputs[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "samples": int(values.size), "pcm_peak_lsb": int(np.max(np.abs(pcm.astype(np.int32))))}
    report["review_audio"] = outputs
    target = args.output_dir / "result.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(target.resolve()), "runtime_ms": result.runtime_ms,
                     "frequency_runtime_ms": frequency_report["runtime_ms"] if frequency_report else None,
                     "outputs": list(outputs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
