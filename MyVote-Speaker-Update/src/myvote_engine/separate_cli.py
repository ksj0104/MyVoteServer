"""Separate a selected local WAV excerpt with the experimental frequency path.

Creates mixture.wav, source-A.wav, source-B.wav and a timing/algorithm report.
Output channels are anonymous; this command does not produce speaker labels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import platform
import time
import wave

from .separation import LocalConvTasNetSeparation, SAMPLE_RATE, NS
from .overlap_separation import FrequencySpeechSeparation


def sample_position(value: str) -> int:
    try:
        if len(value) > 128:
            raise ValueError
        number = Decimal(value)
        if not number.is_finite() or not 0 <= number <= 86_400:
            raise ValueError
        # Bound the exponent before exact-ratio conversion, avoiding both
        # Decimal context rounding and huge-denominator construction.
        if number and not -10 <= number.adjusted() <= 4:
            raise ValueError
        numerator, denominator = number.as_integer_ratio()
        samples, remainder = divmod(numerator * SAMPLE_RATE, denominator)
        if remainder:
            raise ValueError
        return samples
    except (InvalidOperation, ValueError):
        raise argparse.ArgumentTypeError("Use 0..86400 seconds on a 16kHz sample boundary") from None


def read_excerpt(path: Path, start_sample: int, count: int) -> bytes:
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
            raise ValueError("Input must be mono 16000 Hz PCM16 WAV")
        if start_sample + count > source.getnframes():
            raise ValueError("Selected excerpt extends beyond the real source")
        source.setpos(start_sample)
        raw = source.readframes(count)
    if len(raw) != count * 2:
        raise ValueError("Input WAV payload is truncated")
    return raw


def write_wav(path, values):
    import numpy as np
    if values.ndim != 1 or not np.isfinite(values).all() or np.max(np.abs(values)) > .9500000001:
        raise ValueError("Review audio must satisfy the shared peak bound")
    pcm = np.rint(values * 32768).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setparams((1, 2, SAMPLE_RATE, values.size, "NONE", "not compressed"))
        target.writeframes(pcm.tobytes())
    return pcm


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Existing allowlisted ConvTasNet pytorch_model.bin")
    parser.add_argument("--start-s", type=sample_position, default=0)
    parser.add_argument("--duration-s", type=sample_position, default=8 * SAMPLE_RATE)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwrites a previous result")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=1,
                        help="CPU threads for this dedicated command process")
    parser.add_argument("--time-smoothing-radius", type=int, choices=range(3), default=1)
    parser.add_argument("--frequency-smoothing-radius", type=int, choices=range(3), default=0)
    parser.add_argument("--refinement-iterations", type=int, choices=range(3), default=0)
    args = parser.parse_args(argv)
    if not SAMPLE_RATE <= args.duration_s <= 10 * SAMPLE_RATE:
        parser.error("--duration-s must select 1..10 seconds")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("--output-dir must be a new directory")
    raw = read_excerpt(args.input, args.start_s, args.duration_s)
    separator = LocalConvTasNetSeparation(args.model)
    import numpy as np
    import torch
    # This CLI owns its process. The reusable adapters never mutate global threads.
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    mixture = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
    engine = FrequencySpeechSeparation(separator,
        time_smoothing_radius=args.time_smoothing_radius,
        frequency_smoothing_radius=args.frequency_smoothing_radius,
        refinement_iterations=args.refinement_iterations, peak_limit=None)
    preload_start = time.perf_counter_ns()
    engine.preload()
    preload_ms = (time.perf_counter_ns() - preload_start) / 1e6
    result = engine.separate(mixture, window_start_ns=args.start_s * NS // SAMPLE_RATE)
    sources = np.asarray(result.sources, dtype=np.float64)
    peak = max(float(np.max(np.abs(mixture))), float(np.max(np.abs(sources))))
    gain = min(1.0, .95 / peak) if peak else 1.0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {field.name: getattr(result, field.name) for field in fields(result)
              if field.name != "sources"}
    report["frequency_diagnostics"] = asdict(result.frequency_diagnostics)
    pcm = []
    outputs = {}
    for name, values in (("mixture.wav", mixture), ("source-A.wav", sources[0]), ("source-B.wav", sources[1])):
        path = args.output_dir / name
        pcm.append(write_wav(path, values * gain))
        outputs[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "samples": values.size}
    rounding_error = int(np.max(np.abs(pcm[1].astype(np.int32) + pcm[2].astype(np.int32) - pcm[0].astype(np.int32))))
    if rounding_error > 1:
        raise ValueError("Review WAV mixture consistency exceeded PCM rounding")
    report.update(schema="myvote.frequency_separation", schema_version=1,
        input_path=str(args.input.resolve()), selected_pcm16_sha256=hashlib.sha256(raw).hexdigest(),
        start_sample=args.start_s, samples=args.duration_s, sample_rate=SAMPLE_RATE,
        review_common_gain=gain, review_pcm_sum_error_lsb=rounding_error, review_audio=outputs,
        preload_ms=preload_ms, platform=platform.platform(),
        timing_scope="Synchronous local CPU chunk; excludes audio accumulation, ASR, translation and display",
        live_gateway_enabled=False, speaker_ids_assigned=False,
        quality_claim="Arithmetic consistency only; no ground-truth speech separation accuracy")
    target = args.output_dir / "result.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(target.resolve()), "runtime_ms": result.runtime_ms,
                      "frequency_runtime_ms": result.frequency_runtime_ms, "outputs": list(outputs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
