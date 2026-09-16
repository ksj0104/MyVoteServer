"""Fixed, local CPU SepFormer WHAMR 16 kHz inference, without SpeechBrain imports.

The equations and module names are adapted from SpeechBrain's dual_path.py,
Transformer.py, attention.py and normalization.py at training commit
fc2eabb7416b0ae68b3baaadb39972ea1d153985 (Apache License, Version 2.0):
https://github.com/speechbrain/speechbrain/tree/fc2eabb7416b0ae68b3baaadb39972ea1d153985
Copyright 2020-2021 SpeechBrain authors. Modified for a single bounded CPU-only
inference graph, explicit checkpoint validation and dependency-free import.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy at
https://www.apache.org/licenses/LICENSE-2.0 . Unless required by applicable law
or agreed to in writing, software distributed under the License is distributed
on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied. See the License for permissions and limitations.

No YAML, remote code, resampling, gain normalization, clipping, global Torch
thread changes or identity assignment. Schedule this noncausal synchronous
worker separately from caption generation. Two outputs are estimates, not
proof of two clean speakers. Output order can change between windows.
"""

from __future__ import annotations

import hashlib
import importlib
import io
from itertools import islice
import math
from numbers import Real
from pathlib import Path
import threading
import time
from typing import Iterable

from .separation import NS, SAMPLE_RATE, SeparationResult


MODEL_REVISION = "21a5b500c6f52fddc387c5d9e5fb13ffd6f039c5"
MODEL_REPOSITORY = "speechbrain/sepformer-whamr16k"
SOURCE_REVISION = "fc2eabb7416b0ae68b3baaadb39972ea1d153985"
MODEL_FILES = {
    "encoder.ckpt": (17_272, "31d23d395a408b887b8f6ac01e477f00bcba27f95785426359fb52d52f1dc6ed"),
    "masknet.ckpt": (113_112_646, "e5fb8c690668e5d1bbbc9a8256974577093a09cd08e845e9f30024fdd33472ce"),
    "decoder.ckpt": (17_272, "2d959cb46de5b15f008bf5476cf7c19bc680b310c7e10883e3eddfdbde533cb8"),
}
FRONTEND_ID = "speechbrain-sepformer-whamr16k-fc2eabb7-cpu-v1"
DIAGNOSTICS = (
    "experimental_pretrained_separator",
    "training_whamr_16000hz_two_sources",
    "noncausal_whole_chunk_attention_and_global_normalization",
    "raw_forward_without_gain_normalization_or_clipping",
    "output_channels_are_not_speaker_identities",
    "separation_does_not_establish_single_speaker_purity",
)


def _segment(torch, value, chunk: int):
    """Interleave two 50%-overlapping chunk sequences, including boundary pads."""
    batch, channels, length = value.shape
    hop = chunk // 2
    # This is intentionally chunk (not zero) when already aligned.
    gap = chunk - (hop + length % chunk) % chunk
    value = torch.nn.functional.pad(value, (hop, gap + hop))
    first = value[:, :, :-hop].contiguous().view(batch, channels, -1, chunk)
    second = value[:, :, hop:].contiguous().view(batch, channels, -1, chunk)
    chunks = torch.cat((first, second), dim=3).view(batch, channels, -1, chunk)
    return chunks.transpose(2, 3).contiguous(), gap


def _overlap_add(value, gap: int):
    """Sum both chunk sequences; averaging would change the trained graph."""
    batch, channels, chunk, _ = value.shape
    hop = chunk // 2
    paired = value.transpose(2, 3).contiguous().view(batch, channels, -1, chunk * 2)
    first = paired[:, :, :, :chunk].contiguous().view(batch, channels, -1)[:, :, hop:]
    second = paired[:, :, :, chunk:].contiguous().view(batch, channels, -1)[:, :, :-hop]
    merged = first + second
    return merged[:, :, :-gap] if gap else merged


def _create_model(torch, *, _channels=256, _ffn=1024, _heads=8,
                  _transformer_layers=8, _dual_layers=2, _chunk=250,
                  _max_positions=2500):
    """Build the fixed graph. Underscored sizes exist only for small math tests.

    The public adapter always uses the pinned 256/1024/8/8/2/250/2500 graph;
    checkpoint data never determines architecture or executes constructors.
    """
    nn = torch.nn
    tensor_options = {"device": "cpu", "dtype": torch.float32}

    class WrappedNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(_channels, eps=1e-6, **tensor_options)

        def forward(self, value):
            return self.norm(value)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.att = nn.MultiheadAttention(_channels, _heads, dropout=0.0,
                bias=True, add_bias_kv=False, add_zero_attn=False, batch_first=False,
                **tensor_options)

        def forward(self, value):
            # Keep the official seq-first need_weights=True execution path.
            output, _ = self.att(value.permute(1, 0, 2), value.permute(1, 0, 2),
                                 value.permute(1, 0, 2), need_weights=True)
            return output.permute(1, 0, 2)

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = nn.Sequential(nn.Linear(_channels, _ffn, **tensor_options), nn.ReLU(),
                nn.Dropout(0.0), nn.Linear(_ffn, _channels, **tensor_options))

        def forward(self, value):
            return self.ffn(value.permute(1, 0, 2)).permute(1, 0, 2)

    class EncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_att = Attention()
            self.pos_ffn = FeedForward()
            self.norm1 = WrappedNorm()
            self.norm2 = WrappedNorm()

        def forward(self, value):
            value = value + self.self_att(self.norm1(value))
            return value + self.pos_ffn(self.norm2(value))

    class TransformerEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(EncoderLayer() for _ in range(_transformer_layers))
            self.norm = WrappedNorm()

        def forward(self, value):
            for layer in self.layers:
                value = layer(value)
            return self.norm(value)

    class PositionalEncoding(nn.Module):
        def __init__(self):
            super().__init__()
            pe = torch.zeros(_max_positions, _channels, requires_grad=False, **tensor_options)
            positions = torch.arange(0, _max_positions, **tensor_options).unsqueeze(1)
            denominator = torch.exp(torch.arange(0, _channels, 2, **tensor_options)
                                    * -(math.log(10000.0) / _channels))
            pe[:, 0::2] = torch.sin(positions * denominator)
            pe[:, 1::2] = torch.cos(positions * denominator)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, value):
            if value.size(1) > self.pe.size(1):
                raise RuntimeError("SepFormer positional encoding length exceeded")
            return self.pe[:, :value.size(1)].clone().detach()

    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.mdl = TransformerEncoder()
            self.pos_enc = PositionalEncoding()

        def forward(self, value):
            return self.mdl(value + self.pos_enc(value))

    class DualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.intra_mdl = TransformerBlock()
            self.inter_mdl = TransformerBlock()
            # SpeechBrain's norm='ln' means GroupNorm, not channel LayerNorm.
            self.intra_norm = nn.GroupNorm(1, _channels, eps=1e-8, **tensor_options)
            self.inter_norm = nn.GroupNorm(1, _channels, eps=1e-8, **tensor_options)

        def forward(self, value):
            batch, channels, chunk, count = value.shape
            intra = value.permute(0, 3, 2, 1).contiguous().view(batch * count, chunk, channels)
            intra = self.intra_mdl(intra).view(batch, count, chunk, channels)
            intra = intra.permute(0, 3, 2, 1).contiguous()
            intra = self.intra_norm(intra) + value
            inter = intra.permute(0, 2, 3, 1).contiguous().view(batch * chunk, count, channels)
            inter = self.inter_mdl(inter).view(batch, chunk, count, channels)
            inter = inter.permute(0, 3, 1, 2).contiguous()
            return self.inter_norm(inter) + intra

    class MaskNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.GroupNorm(1, _channels, eps=1e-8, **tensor_options)
            self.conv1d = nn.Conv1d(_channels, _channels, 1, bias=False, **tensor_options)
            self.dual_mdl = nn.ModuleList(DualBlock() for _ in range(_dual_layers))
            self.conv2d = nn.Conv2d(_channels, _channels * 2, 1, **tensor_options)
            self.end_conv1x1 = nn.Conv1d(_channels, _channels, 1, bias=False, **tensor_options)
            self.prelu = nn.PReLU(**tensor_options)
            self.output = nn.Sequential(nn.Conv1d(_channels, _channels, 1, **tensor_options), nn.Tanh())
            self.output_gate = nn.Sequential(nn.Conv1d(_channels, _channels, 1, **tensor_options), nn.Sigmoid())

        def forward(self, value):
            value = self.conv1d(self.norm(value))
            value, gap = _segment(torch, value, _chunk)
            for block in self.dual_mdl:
                value = block(value)
            value = self.conv2d(self.prelu(value))
            batch, _, chunk, count = value.shape
            value = value.view(batch * 2, _channels, chunk, count)
            value = _overlap_add(value, gap)
            value = self.output(value) * self.output_gate(value)
            value = self.end_conv1x1(value)
            value = value.view(batch, 2, _channels, -1).relu()
            return value.transpose(0, 1)

    class WaveformEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1d = nn.Conv1d(1, _channels, 16, stride=8, bias=False, **tensor_options)

        def forward(self, value):
            return self.conv1d(value.unsqueeze(1)).relu()

    return nn.ModuleDict({
        "encoder": WaveformEncoder(),
        "masknet": MaskNet(),
        "decoder": nn.ConvTranspose1d(_channels, 1, 16, stride=8, bias=False, **tensor_options),
    })


def _validate_and_load(torch, model, checkpoints) -> None:
    if type(checkpoints) is not dict or set(checkpoints) != {"encoder", "masknet", "decoder"}:
        raise ValueError("SepFormer requires exactly encoder, masknet and decoder states")
    for name, module in model.items():
        state = checkpoints[name]
        expected = module.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError(f"SepFormer {name} checkpoint has missing or extra tensors")
        for key, tensor in state.items():
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(expected[key].shape)
                    or tensor.dtype != torch.float32 or tensor.device.type != "cpu"
                    or tensor.layout != torch.strided or not bool(torch.isfinite(tensor).all())):
                raise ValueError(f"SepFormer {name} checkpoint tensor shape, dtype or finite check failed")
    # Validate all three before changing model state.
    for name, module in model.items():
        module.load_state_dict(checkpoints[name], strict=True)


def _forward_original(torch, model, waveform):
    """SpeechBrain separate_batch equations, before file-level peak scaling."""
    features = model["encoder"](waveform)
    masks = model["masknet"](features)
    masked = torch.stack((features, features)) * masks
    # Decode one source at a time, as in the pinned original implementation.
    output = torch.stack(tuple(model["decoder"](masked[index])[:, 0, :]
                               for index in range(2)), dim=1)
    tail = waveform.shape[-1] - output.shape[-1]
    if not 0 <= tail < 8:
        raise RuntimeError("unexpected SepFormer decoder length")
    if tail:
        output = torch.nn.functional.pad(output, (0, tail))
    return output, tail


class LocalSepformerSeparation:
    """One pinned three-file checkpoint; optional Torch is imported at preload.

    Input: 1..10 seconds of finite normalized mono 16 kHz samples. Output: two
    raw unscaled sample-aligned estimates with at most seven right-zero-padded
    samples, explicitly marked as unmodeled. There is no cross-window channel
    identity or quality guarantee. Calls cannot overlap on the same instance.
    Runtime includes tensor creation, inference and output conversion, excluding
    constructor verification/preload/input validation. It is not caption latency.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError("provide a local SepFormer directory containing three .ckpt files")
        self._checkpoint_bytes = {}
        for filename, (size, digest) in MODEL_FILES.items():
            path = self.model_path / filename
            if not path.is_file():
                raise FileNotFoundError(f"provide the predownloaded local SepFormer {filename}")
            with path.open("rb") as source:
                data = source.read(size + 1)
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"SepFormer {filename} must match the allowlisted SHA256 and size")
            self._checkpoint_bytes[Path(filename).stem] = data
        digest_identity = ":".join(MODEL_FILES[name][1] for name in sorted(MODEL_FILES))
        self._identity = f"sepformer-pytorch:sha256:{digest_identity}:{FRONTEND_ID}"
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
        except (ImportError, OSError) as exc:
            raise RuntimeError("SepFormer separation requires the optional torch package") from exc
        # Hash-verified constructor bytes are used directly, never a second file read.
        checkpoints = {name: torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
                       for name, data in self._checkpoint_bytes.items()}
        model = _create_model(torch)
        _validate_and_load(torch, model, checkpoints)
        model.requires_grad_(False)
        model.eval()
        self._torch, self._model = torch, model
        self._checkpoint_bytes.clear()

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
               or not math.isfinite(value) or not -1 <= value <= 1 for value in values):
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
            with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
                tensor = torch.tensor([waveform], dtype=torch.float32, device="cpu")
                output, tail = _forward_original(torch, self._model, tensor)
                if tuple(output.shape) != (1, 2, len(waveform)) or not bool(torch.isfinite(output).all()):
                    raise RuntimeError("SepFormer produced invalid or nonfinite output")
                first, second = output[0].tolist()
                sources = (tuple(first), tuple(second))
            runtime_ms = (time.perf_counter_ns() - started) / 1_000_000
            return SeparationResult(sources, window_start_ns,
                window_start_ns + len(waveform) * NS // SAMPLE_RATE,
                self.identity, runtime_ms, tail, torch.get_num_threads(), diagnostics=DIAGNOSTICS)
        finally:
            self._lock.release()
