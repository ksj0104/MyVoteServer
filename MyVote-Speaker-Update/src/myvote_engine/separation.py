"""Bounded, local CPU inference for one audited two-speaker ConvTasNet model.

This is an experimental separator, not a speaker-identity or purity detector.
The model card says 16 kHz training; its serialized sample_rate says 8000.
FreeFB convolution does not use that metadata. The card also disagrees with
itself about CC-BY-SA 3.0 versus 4.0. Both discrepancies remain explicit.

The forward equations follow Asteroid v0.4.0's BaseEncoderMaskerDecoder,
TDConvNet and GlobLN, using the installed TorchAudio ConvTasNet graph:
https://github.com/asteroid-team/asteroid/tree/v0.4.0/asteroid
https://github.com/pytorch/audio/blob/v2.9.1/src/torchaudio/models/conv_tasnet.py
No Asteroid package, executable checkpoint configuration, download, global
Torch thread-count change, resampling, gain normalization or clipping is used.
The caller must schedule this synchronous noncausal worker away from captions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import io
from itertools import islice
import math
from numbers import Real
from pathlib import Path
import threading
import time
from typing import Iterable, Protocol


SAMPLE_RATE = 16_000
NS = 1_000_000_000
MAX_MODEL_BYTES = 64 * 1024 * 1024
MODEL_BYTES = 20_394_640
MODEL_SHA256 = "8d97f012f7b2f22bb79cb0d0983a7ba27a52c1796ee3f63cbf25b4d28630adce"
MODEL_REVISION = "e1ef95ab7a037950f3a606b9a56760cf94701d3d"
MODEL_URL = (
    "https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k/resolve/"
    f"{MODEL_REVISION}/pytorch_model.bin")
FRONTEND_ID = "asteroid-freefb-convtasnet16k-padding0-gln-v1"
MODEL_ARGS = {
    "fb_name": "FreeFB", "n_filters": 512, "kernel_size": 32, "stride": 16,
    "sample_rate": 8000, "in_chan": 512, "out_chan": 512, "bn_chan": 128,
    "hid_chan": 512, "skip_chan": 128, "conv_kernel_size": 3,
    "n_blocks": 8, "n_repeats": 3, "n_src": 2, "norm_type": "gLN",
    "mask_act": "relu", "encoder_activation": None,
}
DIAGNOSTICS = (
    "experimental_pretrained_separator",
    "training_card_16000hz_serialized_metadata_8000hz",
    "license_card_metadata_cc_by_sa_4_0_body_cc_by_sa_3_0",
    "noncausal_whole_chunk_global_normalization",
    "raw_forward_unscaled_requires_explicit_gain_postprocessing",
    "output_channels_are_not_speaker_identities",
    "separation_does_not_establish_single_speaker_purity",
)


@dataclass(frozen=True)
class SeparationResult:
    sources: tuple[tuple[float, ...], tuple[float, ...]]
    window_start_ns: int
    window_end_ns: int
    model_identity: str
    runtime_ms: float
    # Asteroid restores up to 15 trailing samples with zeros after valid Conv1d.
    # These samples must not be treated as newly observed speech evidence.
    unmodeled_tail_samples: int
    torch_threads: int
    device: str = "cpu"
    diagnostics: tuple[str, ...] = DIAGNOSTICS


class SpeechSeparation(Protocol):
    @property
    def identity(self) -> str: ...

    def separate(self, samples: Iterable[float], sample_rate: int = SAMPLE_RATE,
                 window_start_ns: int = 0) -> SeparationResult: ...


def _state_mapping() -> dict[str, str]:
    """Explicit architecture map; arbitrary checkpoint key rewriting is forbidden."""
    result = {
        "encoder.filterbank._filters": "encoder.weight",
        "decoder.filterbank._filters": "decoder.weight",
        "masker.bottleneck.0.gamma": "mask_generator.input_norm.weight",
        "masker.bottleneck.0.beta": "mask_generator.input_norm.bias",
        "masker.mask_net.0.weight": "mask_generator.output_prelu.weight",
    }
    for parameter in ("weight", "bias"):
        result[f"masker.bottleneck.1.{parameter}"] = f"mask_generator.input_conv.{parameter}"
        result[f"masker.mask_net.1.{parameter}"] = f"mask_generator.output_conv.{parameter}"
    for block in range(24):
        for layer in range(6):
            parameters = (("gamma", "weight"), ("beta", "bias")) if layer in (2, 5) else (
                (("weight", "weight"),) if layer in (1, 4) else
                (("weight", "weight"), ("bias", "bias")))
            for before, after in parameters:
                result[f"masker.TCN.{block}.shared_block.{layer}.{before}"] = (
                    f"mask_generator.conv_layers.{block}.conv_layers.{layer}.{after}")
        for before, after in (("res_conv", "res_out"), ("skip_conv", "skip_out")):
            # The final residual is computed by Asteroid but never used again.
            if block == 23 and before == "res_conv":
                continue
            for parameter in ("weight", "bias"):
                result[f"masker.TCN.{block}.{before}.{parameter}"] = (
                    f"mask_generator.conv_layers.{block}.{after}.{parameter}")
    return result


def _plain_metadata(value, depth: int = 0) -> bool:
    if depth > 8:
        return False
    if value is None or type(value) in (bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is str:
        return len(value) <= 4096
    if type(value) is list:
        return len(value) <= 64 and all(_plain_metadata(item, depth + 1) for item in value)
    if type(value) is dict:
        return len(value) <= 64 and all(type(key) is str and len(key) <= 256
            and _plain_metadata(item, depth + 1) for key, item in value.items())
    return False


def _create_model(torch, torchaudio):
    model = torchaudio.models.ConvTasNet(
        num_sources=2, enc_kernel_size=32, enc_num_feats=512,
        msk_kernel_size=3, msk_num_feats=128, msk_num_hidden_feats=512,
        msk_num_layers=8, msk_num_stacks=3, msk_activate="relu")

    class ExactGlobalLayerNorm(torch.nn.Module):
        """Asteroid's explicit biased variance and per-channel affine transform."""

        def __init__(self, channels):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(channels))
            self.bias = torch.nn.Parameter(torch.zeros(channels))

        def forward(self, value):
            mean = value.mean(dim=(1, 2), keepdim=True)
            variance = torch.var(value, dim=(1, 2), keepdim=True, unbiased=False)
            normalized = (value - mean) / torch.sqrt(variance + 1e-8)
            return (self.weight * normalized.transpose(1, -1) + self.bias).transpose(1, -1)

    model.mask_generator.input_norm = ExactGlobalLayerNorm(512)
    for block in model.mask_generator.conv_layers:
        block.conv_layers[2] = ExactGlobalLayerNorm(512)
        block.conv_layers[5] = ExactGlobalLayerNorm(512)
    model.encoder.padding = (0,)
    model.decoder.padding = (0,)
    return model


def _validate_and_load(torch, model, checkpoint) -> None:
    required = {"model_name", "state_dict", "model_args", "infos", "dataset", "task", "licenses"}
    if type(checkpoint) is not dict or set(checkpoint) != required:
        raise ValueError("unsupported separation checkpoint envelope")
    if (checkpoint["model_name"] != "ConvTasNet" or checkpoint["dataset"] != "Libri2Mix"
            or checkpoint["task"] != "sep_clean"):
        raise ValueError("unsupported separation checkpoint model")
    arguments = checkpoint["model_args"]
    if (type(arguments) is not dict or set(arguments) != set(MODEL_ARGS)
            or any(type(arguments[key]) is not type(value) or arguments[key] != value
                   for key, value in MODEL_ARGS.items())):
        raise ValueError("unsupported separation architecture configuration")
    if not all(_plain_metadata(checkpoint[key]) for key in ("infos", "licenses")):
        raise ValueError("unsupported separation metadata")
    mapping = _state_mapping()
    state = checkpoint["state_dict"]
    ignored = {"masker.TCN.23.res_conv.weight": (128, 512, 1),
               "masker.TCN.23.res_conv.bias": (128,)}
    if not isinstance(state, dict) or set(state) != set(mapping) | set(ignored):
        raise ValueError("separation checkpoint has missing or extra tensors")
    expected = model.state_dict()
    if set(expected) != set(mapping.values()):
        raise RuntimeError("installed TorchAudio ConvTasNet architecture is incompatible")
    for source, tensor in state.items():
        shape = ignored[source] if source in ignored else tuple(expected[mapping[source]].shape)
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32 or tensor.device.type != "cpu"
                or tensor.layout != torch.strided or not bool(torch.isfinite(tensor).all())):
            raise ValueError("separation checkpoint tensor shape, dtype or finite check failed")
    model.load_state_dict({target: state[source] for source, target in mapping.items()}, strict=True)


def _forward_original(torch, model, waveform):
    """No TorchAudio stride alignment: Asteroid encodes first, then restores length."""
    features = model.encoder(waveform)
    masks = model.mask_generator(features)
    masked = (features.unsqueeze(1) * masks).reshape(2, 512, -1)
    decoded = model.decoder(masked).reshape(1, 2, -1)
    tail = waveform.shape[-1] - decoded.shape[-1]
    if not 0 <= tail < 16:
        raise RuntimeError("unexpected separation decoder length")
    if tail:
        decoded = torch.nn.functional.pad(decoded, (0, tail))
    return decoded, tail


class LocalConvTasNetSeparation:
    """One fixed checkpoint, lazy optional dependencies, CPU-only synchronous worker.

    ``runtime_ms`` includes tensor creation, inference and CPU output conversion;
    it excludes model preload and input validation. It is not caption latency.
    Outputs are raw, unclipped estimates and can have very large gain. Asteroid's
    separate.torch_separate wrapper applies a later shared L1 gain correction;
    that DSP step belongs to the caller and is not silently applied here.
    Calls on one instance may not overlap. Torch threads remain caller-owned.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError("provide the predownloaded local ConvTasNet pytorch_model.bin")
        with self.model_path.open("rb") as source:
            self._model_bytes = source.read(MAX_MODEL_BYTES + 1)
        if (len(self._model_bytes) != MODEL_BYTES
                or hashlib.sha256(self._model_bytes).hexdigest() != MODEL_SHA256):
            raise ValueError("separation model must match the allowlisted SHA256 and size")
        self._identity = (f"convtasnet-pytorch:sha256:{MODEL_SHA256}:{FRONTEND_ID}:"
                          "experimental-card16k-metadata8k-license3or4")
        self._lock = threading.Lock()
        self._torch = None
        self._model = None

    @property
    def identity(self) -> str:
        return self._identity

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            torchaudio = importlib.import_module("torchaudio")
        except (ImportError, OSError) as exc:
            raise RuntimeError("separation requires compatible optional torch and torchaudio packages") from exc
        # Explicitly safe: no unsafe retry, remote class lookup or YAML constructors.
        checkpoint = torch.load(io.BytesIO(self._model_bytes), map_location="cpu", weights_only=True)
        model = _create_model(torch, torchaudio)
        _validate_and_load(torch, model, checkpoint)
        model.requires_grad_(False)
        model.eval()
        self._torch, self._model = torch, model

    def preload(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("separation worker is already in use")
        try:
            self._load()
        finally:
            self._lock.release()

    @staticmethod
    def _pcm(samples, sample_rate, window_start_ns) -> tuple[float, ...]:
        if type(sample_rate) is not int or sample_rate != SAMPLE_RATE:
            raise ValueError("separation requires mono PCM at 16000 Hz")
        if type(window_start_ns) is not int or window_start_ns < 0:
            raise ValueError("window_start_ns must be nonnegative integer nanoseconds")
        values = tuple(islice(samples, SAMPLE_RATE * 10 + 1))
        if not SAMPLE_RATE <= len(values) <= SAMPLE_RATE * 10:
            raise ValueError("separation requires between 1 and 10 seconds of PCM")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values):
            raise ValueError("separation PCM must contain finite normalized real samples")
        return tuple(float(value) for value in values)

    def separate(self, samples: Iterable[float], sample_rate: int = SAMPLE_RATE,
                 window_start_ns: int = 0) -> SeparationResult:
        waveform = self._pcm(samples, sample_rate, window_start_ns)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("separation worker is already in use")
        try:
            self._load()
            torch = self._torch
            started = time.perf_counter_ns()
            with torch.inference_mode():
                tensor = torch.tensor([[waveform]], dtype=torch.float32, device="cpu")
                output, tail = _forward_original(torch, self._model, tensor)
                if tuple(output.shape) != (1, 2, len(waveform)) or not bool(torch.isfinite(output).all()):
                    raise RuntimeError("separation produced invalid or nonfinite output")
                first, second = output[0].tolist()
                sources = (tuple(first), tuple(second))
            runtime_ms = (time.perf_counter_ns() - started) / 1_000_000
            return SeparationResult(sources, window_start_ns,
                window_start_ns + len(waveform) * NS // SAMPLE_RATE,
                self.identity, runtime_ms, tail, torch.get_num_threads())
        finally:
            self._lock.release()
