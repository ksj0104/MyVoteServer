"""Report Mac inference prerequisites without loading models or generating text."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit

MAX_JSON_BYTES = 256 * 1024
WHISPER_DIMENSIONS = frozenset(("n_mels", "n_audio_ctx", "n_audio_state", "n_audio_head", "n_audio_layer",
    "n_vocab", "n_text_ctx", "n_text_state", "n_text_head", "n_text_layer"))
PACKAGES = (
    ("numpy", "numpy", "asarray"), ("onnxruntime", "onnxruntime", "InferenceSession"),
    ("mlx", "mlx.core", "array"), ("mlx-whisper", "mlx_whisper", "transcribe"),
    ("httpx", "httpx", "Client"), ("grpcio", "grpc.aio", "server"),
    ("protobuf", "google.protobuf", "__version__"),
    ("cryptography", "cryptography.x509", "load_pem_x509_certificate"),
)
SPEAKER_PACKAGES = (("torch", "torch", "tensor"),
                    ("torchaudio", "torchaudio.compliance.kaldi", "fbank"))


@dataclass(frozen=True)
class PreflightConfig:
    asr_model: Path | None = None
    vad_model: Path | None = None
    speaker_segmentation_model: Path | None = None
    speaker_embedding_model: Path | None = None
    speaker_profile: Path | None = None
    backend: str = "llamacpp"
    endpoint: str | None = None
    model: str | None = None
    check_backend: bool = False


def _check(name, status, category, action, *, required=True, evidence=None):
    return {"name": name, "status": status, "category": category, "required": required,
            "action": action, "evidence": evidence or {}}


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite_number")
        return result
    def constant(_):
        raise ValueError("nonfinite_number")
    value = json.loads(raw, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("object_required")
    return value


def _sysctl(name):
    """Fixed keys only; absent optional proc_translated remains unknown, not zero."""
    try:
        result = subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True,
                                timeout=3, check=False)
        if result.returncode or len(result.stdout) > 4096:
            return None
        value = result.stdout.decode("utf-8", errors="strict").strip()
        return value if value and value.isprintable() else None
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return None


def _host_checks():
    system, machine = platform.system(), platform.machine().lower()
    host = {"system": system, "machine": machine, "python_version": platform.python_version(),
            "python_executable": sys.executable, "macos_version": platform.mac_ver()[0] or None}
    checks = [_check("host.macos", "passed" if system == "Darwin" else "failed",
        "macos_detected" if system == "Darwin" else "unsupported_platform",
        "Run this report in the native Python environment on the target Mac.", evidence={"system": system})]
    native = False
    if system != "Darwin":
        checks += [_check("host.native_python", "not_checked", "unsupported_platform", "Use native arm64 macOS Python."),
                   _check("host.target", "not_checked", "unsupported_platform", "Collect hardware facts on the target Mac.")]
    else:
        facts = {key: _sysctl(key) for key in ("machdep.cpu.brand_string", "hw.memsize", "hw.physicalcpu", "hw.model", "sysctl.proc_translated")}
        host["sysctl"] = facts
        native = machine == "arm64" and facts["sysctl.proc_translated"] != "1"
        checks.append(_check("host.native_python", "passed" if native else "failed",
            "native_arm64" if native else "unsupported_python_architecture",
            "Use an arm64 Python interpreter outside Rosetta.", evidence={"machine": machine, "proc_translated": facts["sysctl.proc_translated"]}))
        values = (facts["machdep.cpu.brand_string"], facts["hw.memsize"], facts["hw.physicalcpu"])
        if any(value is None for value in values):
            checks.append(_check("host.target", "not_checked", "hardware_query_failed", "Verify chip, memory and CPU count with System Information.", evidence=facts))
        else:
            try:
                memory, cores = int(values[1]), int(values[2])
                if memory <= 0 or cores <= 0:
                    raise ValueError
                match = values[0] == "Apple M4 Max" and memory == 64 * 1024 ** 3 and cores == 16
                checks.append(_check("host.target", "passed" if match else "failed",
                    "target_chip_memory_cpu_match" if match else "unsupported_target",
                    "The experiment targets M4 Max, 64 GiB and 16 CPU cores; separately verify Mac Studio chassis and 40 GPU cores.",
                    evidence={"chip": values[0], "memory_bytes": memory, "physical_cpu_cores": cores, "model_identifier": facts["hw.model"],
                              "chassis_verified": False, "gpu_core_count_verified": False}))
            except ValueError:
                checks.append(_check("host.target", "not_checked", "hardware_query_invalid", "Inspect sysctl hardware values manually."))
    checks.append(_check("host.python_version", "passed" if sys.version_info >= (3, 11) else "failed",
                         "python_version_supported" if sys.version_info >= (3, 11) else "python_too_old", "Use Python 3.11 or newer."))
    return host, checks, native


# Imports happen in a short-lived interpreter, not in the gateway model process.
# Native stdout/stderr are suppressed as well as Python prints; only our fixed
# result schema is emitted. No adapter construction or model inference occurs.
_IMPORT_SCRIPT = r'''
import importlib,json,os,sys
output=os.dup(1)
with open(os.devnull,"w") as sink:
    os.dup2(sink.fileno(),1); os.dup2(sink.fileno(),2)
results={}
for module,attribute in json.loads(sys.argv[1]):
    try:
        value=importlib.import_module(module)
        ok=hasattr(value,attribute)
        results[module]="passed" if ok else "api_missing"
    except Exception:
        results[module]="import_failed"
os.write(output,json.dumps(results).encode("utf-8"))
os.close(output)
'''


def _probe_imports(packages):
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    try:
        result = subprocess.run([sys.executable, "-c", _IMPORT_SCRIPT,
                                 json.dumps([(module, attribute) for _, module, attribute in packages])],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment,
                                timeout=30, check=False)
        if result.returncode or len(result.stdout) > 16384:
            return {}, "import_probe_failed"
        statuses = _strict_json(result.stdout)
        if set(statuses) != {module for _, module, _ in packages} or any(value not in ("passed", "api_missing", "import_failed") for value in statuses.values()):
            return {}, "import_probe_invalid"
        return statuses, None
    except subprocess.TimeoutExpired:
        return {}, "import_probe_timeout"
    except (OSError, ValueError, UnicodeError):
        return {}, "import_probe_failed"


def _package_checks(native, speakers):
    packages = PACKAGES + (SPEAKER_PACKAGES if speakers else ())
    statuses, error = _probe_imports(packages) if native else ({}, "unsupported_platform_or_architecture")
    checks = []
    for distribution, module, _ in packages:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        status = statuses.get(module)
        checks.append(_check("package." + distribution,
            "not_checked" if not native else "passed" if status == "passed" else "failed",
            error or ("import_supported" if status == "passed" else status or "package_missing"),
            "Install compatible Mac extras in this Python environment and rerun; do not reuse the Windows lock as a Mac validation record.",
            evidence={"distribution": distribution, "version": version, "module": module, "import_verified": status == "passed"}))
    return checks


def _file_check(name, path, *, required=True):
    if path is None:
        return _check(name, "not_checked", "path_not_supplied", "Supply an explicit, already prepared local model path.", required=required)
    try:
        path = path.expanduser().resolve(strict=True)
        size = path.stat().st_size
        if not path.is_file() or path.suffix.lower() != ".onnx" or not 0 < size <= 512 * 1024 ** 2:
            raise ValueError
        with path.open("rb") as source:
            if len(source.read(1)) != 1:
                raise ValueError
        return _check(name, "passed", "local_file_accessible", "Run the adapter smoke to validate the ONNX graph before inference measurement.", required=required,
                      evidence={"path": str(path), "size_bytes": size, "graph_loaded": False, "sha256": None})
    except (OSError, ValueError):
        return _check(name, "failed", "local_model_unavailable", "Use a readable, nonempty single-file .onnx model no larger than 512 MiB.", required=required)


def _asr_check(path):
    if path is None:
        return _check("model.asr", "not_checked", "path_not_supplied", "Supply --asr-model with a prepared local MLX Whisper directory.")
    try:
        path = path.expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError
        with (path / "config.json").open("rb") as source:
            raw = source.read(65537)
        if len(raw) > 65536:
            raise ValueError
        config = _strict_json(raw)
        if set(config) - WHISPER_DIMENSIONS - {"model_type", "quantization"} or not WHISPER_DIMENSIONS <= set(config):
            raise ValueError
        if any(type(config[key]) is not int or config[key] <= 0 for key in WHISPER_DIMENSIONS):
            raise ValueError
        if config.get("model_type", "whisper") != "whisper" or ("quantization" in config and config["quantization"] is not None and not isinstance(config["quantization"], dict)):
            raise ValueError
        # Official loader preference order, verified against its primary source.
        weights = next((path / name for name in ("model.safetensors", "weights.safetensors", "weights.npz") if (path / name).exists()), None)
        if weights is None or not weights.is_file() or weights.stat().st_size <= 0:
            raise ValueError
        with weights.open("rb") as source:
            if len(source.read(1)) != 1:
                raise ValueError
        return _check("model.asr", "passed", "local_whisper_layout_present",
            "Run an actual short WAV transcription to validate weights and word timestamps.",
            evidence={"path": str(path), "config_sha256": hashlib.sha256(raw).hexdigest(),
                      "weights_file": weights.name, "weights_bytes": weights.stat().st_size,
                      "weights_sha256": None, "weights_loaded": False})
    except (OSError, ValueError, UnicodeError, RecursionError):
        return _check("model.asr", "failed", "local_whisper_layout_invalid",
                      "Prepare a local MLX Whisper directory with valid config.json and model.safetensors, weights.safetensors, or weights.npz; no model is downloaded.")


def _asset_checks(config):
    checks = [_asr_check(config.asr_model), _file_check("model.vad", config.vad_model)]
    paired = bool(config.speaker_segmentation_model) == bool(config.speaker_embedding_model)
    speakers = bool(config.speaker_segmentation_model or config.speaker_embedding_model or config.speaker_profile)
    checks.append(_check("model.speaker_configuration", "passed" if paired and (not config.speaker_profile or config.speaker_segmentation_model) else "failed",
        "speaker_paths_paired" if paired and (not config.speaker_profile or config.speaker_segmentation_model) else "speaker_options_incomplete",
        "Supply both speaker models; a profile also requires both model paths.", required=speakers))
    for name, path in (("model.speaker_segmentation", config.speaker_segmentation_model), ("model.speaker_embedding", config.speaker_embedding_model)):
        checks.append(_file_check(name, path, required=speakers))
    if config.speaker_profile:
        try:
            from .speaker_profile import load_speaker_profile
            load_speaker_profile(config.speaker_profile)
            checks.append(_check("model.speaker_profile", "passed", "profile_schema_valid",
                "Gateway initialization must still compare both exact model/frontend identities; this preflight does not apply or calibrate thresholds.",
                evidence={"identity_match_verified": False}))
        except (OSError, ValueError, UnicodeError):
            checks.append(_check("model.speaker_profile", "failed", "profile_invalid", "Use a valid local version-1 speaker profile for these exact models."))
    return checks, speakers


def _local_endpoint(endpoint):
    try:
        url = urlsplit(endpoint)
        if (url.scheme not in ("http", "https") or url.username is not None or url.password is not None or
                url.query or url.fragment or url.path not in ("", "/") or not url.hostname or
                any(ord(char) < 33 for char in endpoint)):
            raise ValueError
        host = "127.0.0.1" if url.hostname == "localhost" else url.hostname
        address = ipaddress.ip_address(host)
        if not address.is_loopback or (isinstance(address, ipaddress.IPv6Address) and address.scope_id):
            raise ValueError
        port = url.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
        host = f"[{address}]" if address.version == 6 else str(address)
        return f"{url.scheme}://{host}" + (f":{port}" if port is not None else "")
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Use a loopback HTTP(S) URL without credentials, query, fragment, or path.") from None


def _get_json(client, url):
    import httpx
    deadline = time.monotonic() + 6
    try:
        with client.stream("GET", url, headers={"Accept": "application/json"}) as response:
            if response.status_code != 200:
                return None, {503: "backend_loading", 401: "backend_auth_required", 403: "backend_auth_required"}.get(response.status_code, "backend_http_error"), response.status_code
            body = bytearray()
            for chunk in response.iter_bytes():
                if time.monotonic() > deadline:
                    return None, "backend_timeout", 200
                if len(body) + len(chunk) > MAX_JSON_BYTES:
                    return None, "backend_response_too_large", 200
                body.extend(chunk)
            return _strict_json(body), None, 200
    except httpx.TimeoutException:
        return None, "backend_timeout", None
    except httpx.HTTPError:
        return None, "backend_connection_failed", None
    except (ValueError, UnicodeError, RecursionError):
        return None, "backend_response_invalid", None


def _backend_checks(config, native, *, transport=None):
    requested = config.check_backend
    action = "Start the prepared local translation server, then use --check-backend with its exact model alias/tag."
    if not requested or not native:
        category = "not_requested" if not requested else "unsupported_platform_or_architecture"
        return [_check(name, "not_checked", category, action) for name in ("backend.health", "backend.model")]
    try:
        ports = {"llamacpp": 8080, "ollama": 11434, "lmstudio": 1234}
        if config.backend not in ports or not config.model or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:-]{0,255}", config.model):
            raise ValueError
        endpoint = _local_endpoint(config.endpoint or f"http://127.0.0.1:{ports[config.backend]}")
        if config.model.lower().endswith((":cloud", "-cloud")):
            return [_check("backend.model", "failed", "cloud_model_rejected", "Choose an explicitly prepared local model tag.")]
    except ValueError:
        return [_check("backend.configuration", "failed", "backend_configuration_invalid", "Use an exact model alias/tag and a loopback URL without credentials or query parameters.")]
    try:
        import httpx
    except ImportError:
        return [_check("backend.health", "failed", "httpx_unavailable", "Install the project base dependencies.")]
    with httpx.Client(timeout=3, follow_redirects=False, trust_env=False, transport=transport) as client:
        health_path = {"ollama": "/api/tags", "llamacpp": "/health", "lmstudio": "/v1/models"}[config.backend]
        data, error, status = _get_json(client, endpoint + health_path)
        if not error and config.backend == "llamacpp" and data.get("status") != "ok":
            error = "backend_not_ready"
        health = _check("backend.health", "failed" if error else "passed", error or "backend_responding", action,
                        evidence={"backend": config.backend, "endpoint": endpoint, "http_status": status})
        if error:
            return [health, _check("backend.model", "not_checked", "health_check_failed", action)]
        if config.backend == "llamacpp":
            data, error, status = _get_json(client, endpoint + "/v1/models")
        entries = None if error else data.get("models" if config.backend == "ollama" else "data")
        if error or not isinstance(entries, list) or len(entries) > 4096 or any(not isinstance(item, dict) for item in entries):
            return [health, _check("backend.model", "failed", error or "backend_response_invalid", action)]
        field = "name" if config.backend == "ollama" else "id"
        matches = [item for item in entries if item.get(field) == config.model]
        remote = any(item.get("remote_host") or item.get("remote_model") for item in matches)
        category = "remote_model_rejected" if remote else "model_listed" if matches else "model_not_listed"
        return [health, _check("backend.model", "passed" if matches and not remote else "failed", category,
            "Use the exact listed local tag/alias. With LM Studio JIT loading, downloaded models may be listed before loading. Verify loading and disable thinking in server model settings before measuring.",
            evidence={"model": config.model, "listed": bool(matches), "inference_verified": False,
                      "loaded_weights_verified": False, "thinking_disabled_verified": False,
                      "compute_residency_verified": False})]


def run_preflight(config: PreflightConfig):
    host, checks, native = _host_checks()
    assets, speakers = _asset_checks(config)
    checks += assets + _package_checks(native, speakers) + _backend_checks(config, native)
    required = [item for item in checks if item["required"]]
    status = "failed" if any(item["status"] == "failed" for item in required) else "not_checked" if any(item["status"] == "not_checked" for item in required) else "passed"
    return {"schema": "myvote.mac_preflight", "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": status, "host": host, "checks": checks,
            "scope": "Prerequisite inspection only; no model loading, transcription, translation, or speaker inference.",
            "performance_verified": False, "unverified": ["Mac Studio chassis and GPU count", "model weight contents and graph compatibility",
                "speaker profile model identity match", "Metal offload", "local-only backend compute", "mTLS/LAN connection",
                "cold/warm inference", "translation latency and quality", "speaker accuracy"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asr-model", "vad-model", "speaker-segmentation-model", "speaker-embedding-model", "speaker-profile"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--backend", choices=("llamacpp", "ollama", "lmstudio"), default="llamacpp")
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--check-backend", action="store_true", help="Read loopback health/model metadata; never submit inference text.")
    parser.add_argument("--output", type=Path, help="Write a new JSON report; existing files are not overwritten.")
    args = parser.parse_args(argv)
    output = args.output
    report = run_preflight(PreflightConfig(**{key: value for key, value in vars(args).items() if key != "output"}))
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if output:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
        except OSError:
            print(json.dumps({"schema": "myvote.mac_preflight", "schema_version": 1, "status": "failed",
                              "category": "report_write_failed", "action": "Choose a new writable output path."}), file=sys.stderr)
            return 2
    print(serialized)
    return 1 if report["status"] == "failed" else 2 if report["status"] == "not_checked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
