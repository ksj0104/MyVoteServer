"""Explicit Mac demo bootstrap; standard library only until the prepared venv runs.

Download sources (data only, never executable remote scripts):
https://huggingface.co/mlx-community/whisper-large-v3-turbo
https://huggingface.co/docs/huggingface_hub/guides/download
https://github.com/ml-explore/mlx-examples/blob/main/whisper/mlx_whisper/load_models.py
The package build places this file in KIT/scripts and start-demo.command in KIT.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request


ASR_REPO = "mlx-community/whisper-large-v3-turbo"
ASR_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
ASR_FILES = {
    "config.json": (268, "b34fc29e4e11e0a25e812775dd67f4dd16fc2c8eb43d28ae25ff7d660ecb6379"),
    "weights.safetensors": (1613977612, "951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6"),
}
LM_ENDPOINT = "http://127.0.0.1:1234"
# Reproducible first-demo selection, not a claim that later PyTorch is incompatible:
# TorchAudio 2.11 supports PyTorch >=2.11 through its stable ABI.
# https://docs.pytorch.org/audio/stable/installation.html
# Both 2.11.0 releases publish macOS arm64 CPython 3.12/3.13 wheels on PyPI.
MAC_TORCH_REQUIREMENTS = ("torch==2.11.0", "torchaudio==2.11.0")
MAX_MODEL_RESPONSE = 262144
SAFE_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+\-]{0,255}\Z")
QWEN_4B = re.compile(r"(?:^|[/_\-])qwen3[._\-]?5[\-_]4b(?:$|[/_.:@+\-])", re.I)
WHISPER_DIMENSIONS = (
    "n_mels", "n_audio_ctx", "n_audio_state", "n_audio_head", "n_audio_layer",
    "n_vocab", "n_text_ctx", "n_text_state", "n_text_head", "n_text_layer",
)


class SetupError(Exception):
    """A setup failure that can be displayed without exposing credential bytes."""


def validate_runtime() -> None:
    if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
        raise SetupError("Apple Silicon Mac의 native arm64 Python으로 실행하세요. Windows/Rosetta는 지원하지 않습니다.")
    if sys.version_info < (3, 11):
        raise SetupError("Python 3.11 이상이 필요합니다.")
    try:
        major = int(platform.mac_ver()[0].split(".")[0])
    except (ValueError, IndexError):
        raise SetupError("macOS 버전을 확인할 수 없습니다.") from None
    if major < 14:
        raise SetupError("MLX 실행에는 macOS 14 이상이 필요합니다.")


def parse_args(argv=None, *, kit_root: Path | None = None):
    root = (kit_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    parser = argparse.ArgumentParser(description="Mac 데모 서버 준비 및 실행. 설치·다운로드는 명시한 옵션에서만 수행합니다.", allow_abbrev=False)
    parser.add_argument("--models-dir", type=Path, default=root / "models")
    parser.add_argument("--cert-dir", type=Path, default=root / "connection")
    parser.add_argument("--host", default="192.168.219.103", help="Mac의 실제 LAN IP와 인증서 SAN이 같아야 합니다.")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--llm-model", help="LM Studio /v1/models 목록의 정확한 모델 ID")
    parser.add_argument("--translation-profile", choices=("generic", "translategemma"), default="generic")
    parser.add_argument("--orchestrator-model", help="의미 구간 판단·동일 언어 보정용 별도 지시 모델의 정확한 ID")
    parser.add_argument("--translation-workers", type=int, choices=range(1, 5),
                        help="논리 번역 동시 처리 수. 오케스트레이터 사용 시 기본 4이며 모델 복사 수가 아닙니다.")
    parser.add_argument("--translation-model-id", action="append",
                        help="번역 모델/인스턴스 ID. 여러 인스턴스는 반복 지정하고, 생략하면 --llm-model 하나를 공유합니다.")
    parser.add_argument("--install", action="store_true", help="이 폴더의 .venv 생성 및 Python 의존성 설치")
    parser.add_argument("--download-asr", action="store_true", help="공개 MLX Whisper turbo 약 1.61GB 다운로드")
    parser.add_argument("--experimental-speakers", action="store_true", help="동봉된 연구용 화자 프로필 적용; 정확도 보장 없음")
    parser.add_argument("--speaker-profile", type=Path, help="기존 모델과 일치하는 별도 화자 실험 프로필")
    parser.add_argument("--experimental-speaker-pooling", action="store_true",
                        help="짧은 단독 발화를 모아 화자 판단에 사용하는 실험 기능")
    parser.add_argument("--experimental-speaker-activity", choices=("trusted", "singleton"),
                        help="화자 특징 근거와 발화 표시 범위를 분리하는 연구 옵션")
    parser.add_argument("--experimental-overlap-model", type=Path,
                        help="겹침 분리·두 줄 자막 실험용 고정 ConvTasNet 체크포인트 경로")
    parser.add_argument("--overlap-threads", type=int, choices=range(1, 9), default=1,
                        help="겹침 실험 사용 시 시작할 때 고정할 Torch CPU 스레드 수")
    parser.add_argument("--skip-warmup", action="store_true", help="시작 전 실제 ASR 무음·번역 연결 진단 생략")
    args = parser.parse_args(argv)
    if args.experimental_speaker_pooling and args.experimental_speaker_activity:
        raise SetupError("화자 활동 연결과 짧은 발화 풀링은 각각 따로 비교해 주세요.")
    args.kit_root = root
    args.models_dir = args.models_dir.expanduser().absolute()
    args.cert_dir = args.cert_dir.expanduser().absolute()
    if args.speaker_profile is not None:
        args.speaker_profile = args.speaker_profile.expanduser().absolute()
    if args.experimental_overlap_model is not None:
        args.experimental_overlap_model = args.experimental_overlap_model.expanduser().absolute()
    try:
        address = ipaddress.ip_address(args.host)
        if address.is_unspecified or address.is_multicast or "%" in args.host:
            raise ValueError
    except ValueError:
        raise SetupError("--host에는 0.0.0.0, URL, 호스트명 대신 Mac의 단일 IP 주소를 입력하세요.") from None
    args.host = str(address)
    if not 1 <= args.port <= 65535:
        raise SetupError("--port는 1..65535여야 합니다.")
    if args.llm_model is not None and not SAFE_MODEL_ID.fullmatch(args.llm_model):
        raise SetupError("--llm-model에는 제어 문자나 공백 없는 정확한 모델 ID를 입력하세요.")
    validate_translation_options(args)
    return args


def validate_translation_options(args, model=None):
    profile = getattr(args, "translation_profile", "generic")
    orchestrator = getattr(args, "orchestrator_model", None)
    ids = getattr(args, "translation_model_id", None)
    workers = getattr(args, "translation_workers", None)
    for identifier in [*(ids or []), *([orchestrator] if orchestrator is not None else [])]:
        if not isinstance(identifier, str) or not SAFE_MODEL_ID.fullmatch(identifier):
            raise SetupError("모델 옵션에는 공백·제어 문자 없는 정확한 모델 ID를 입력하세요.")
    resolved_models = ids or ([model or args.llm_model] if model or args.llm_model else [])
    if profile == "translategemma":
        if not orchestrator:
            raise SetupError("TranslateGemma에는 --orchestrator-model로 별도 지시 모델을 지정하세요.")
        if not resolved_models:
            raise SetupError("TranslateGemma 번역기 ID를 --llm-model 또는 --translation-model-id로 지정하세요.")
        if (not 1 <= len(resolved_models) <= 4 or len(set(resolved_models)) != len(resolved_models)
                or len(resolved_models) > (workers if workers is not None else 4)):
            raise SetupError("번역기 ID는 중복 없이 1..4개이고 --translation-workers 이하여야 합니다.")
        if orchestrator in resolved_models or "translategemma" in re.sub(r"[^a-z0-9]", "", orchestrator.lower()):
            raise SetupError("오케스트레이터는 TranslateGemma 번역기와 다른 일반 지시 모델이어야 합니다.")
    else:
        if orchestrator is not None or ids is not None or workers is not None:
            raise SetupError("오케스트레이터·번역 worker 옵션에는 --translation-profile translategemma가 필요합니다.")
        if any("translategemma" in re.sub(r"[^a-z0-9]", "", identifier.lower()) for identifier in resolved_models):
            raise SetupError("TranslateGemma는 generic JSON·보정 프롬프트를 지원하지 않습니다. --translation-profile translategemma와 --orchestrator-model을 지정하세요.")


def child_environment(*, offline: bool) -> dict[str, str]:
    environment = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    else:
        environment.pop("HF_HUB_OFFLINE", None)
        environment.pop("TRANSFORMERS_OFFLINE", None)
    return environment


def regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise SetupError(f"파일이 없거나 비어 있거나 심볼릭 링크입니다: {path}")


def validate_assets(args) -> None:
    regular_file(args.kit_root / "pyproject.toml")
    if args.cert_dir.is_symlink() or not args.cert_dir.is_dir():
        raise SetupError("connection 인증서 폴더가 없거나 심볼릭 링크입니다.")
    for name in ("server.crt", "server.key", "ca.crt"):
        path = args.cert_dir / name
        regular_file(path)
        if path.stat().st_size > 65536:
            raise SetupError(f"인증서 파일이 크기 제한을 넘습니다: {name}")
    for name in ("silero_vad.onnx", "segmentation.onnx", "wespeaker.onnx"):
        regular_file(args.models_dir / name)
    if args.experimental_speakers or args.speaker_profile is not None:
        regular_file(args.speaker_profile or args.models_dir / "speaker-profile.json")
    asr = args.models_dir / "whisper-turbo"
    if asr.exists() or asr.is_symlink():
        validate_asr(asr)
    elif not args.download_asr:
        raise SetupError("models/whisper-turbo가 없습니다. 첫 실행에 --download-asr를 지정하세요.")


def protect_certificates(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise SetupError("인증서 폴더는 실제 디렉터리여야 합니다.")
    for name in ("server.key", "server.crt", "ca.crt"):
        regular_file(directory / name)
    # Repair permissions commonly broadened by ZIP extraction; no trust registration.
    directory.chmod(0o700)
    (directory / "server.key").chmod(0o600)


def ensure_environment(root: Path, *, install: bool) -> Path:
    environment = root / ".venv"
    python = environment / "bin" / "python"
    if environment.is_symlink():
        raise SetupError(".venv 심볼릭 링크는 사용하지 않습니다.")
    if environment.exists() and (not environment.is_dir() or not python.is_file()):
        raise SetupError("기존 .venv가 불완전합니다. 별도 이름으로 보관한 뒤 --install로 다시 만드세요. 자동 덮어쓰기는 하지 않습니다.")
    if not environment.exists():
        if not install:
            raise SetupError("준비된 .venv가 없습니다. 첫 실행에 --install을 지정하세요.")
        subprocess.run([sys.executable, "-I", "-m", "venv", str(environment)], check=True,
                       cwd=root, env=child_environment(offline=False))
    check = ("import platform,sys; "
             "assert platform.system()=='Darwin' and platform.machine()=='arm64', 'native Mac venv required'; "
             "assert sys.version_info >= (3,11), 'Python 3.11 required'; "
             "assert sys.prefix != sys.base_prefix, 'venv required'")
    subprocess.run([str(python), "-I", "-c", check], check=True, cwd=root, env=child_environment(offline=True))
    if install:
        print("Python 의존성을 설치합니다. 이 단계에는 인터넷 연결이 필요합니다.", flush=True)
        subprocess.run([str(python), "-I", "-m", "pip", "install", str(root) + "[audio,mac-asr,gateway,speaker]",
                        *MAC_TORCH_REQUIREMENTS],
                       check=True, cwd=root, env=child_environment(offline=False))
    return python


def validate_asr(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise SetupError("whisper-turbo는 기존 로컬 모델 디렉터리여야 합니다.")
    config = directory / "config.json"
    regular_file(config)
    if config.stat().st_size > 65536:
        raise SetupError("Whisper config.json 크기가 잘못되었습니다.")
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(type(data.get(k)) is int and data[k] > 0 for k in WHISPER_DIMENSIONS):
            raise ValueError
    except (ValueError, UnicodeError):
        raise SetupError("Whisper config.json의 모델 차원 정보가 잘못되었습니다.") from None
    weights = next((directory / name for name in ("model.safetensors", "weights.safetensors", "weights.npz")
                    if (directory / name).exists()), None)
    if weights is None:
        raise SetupError("Whisper 가중치 파일이 없습니다. 기존 폴더를 자동 덮어쓰지 않습니다.")
    regular_file(weights)


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download_asr(python: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise SetupError("ASR 다운로드 대상이 이미 있습니다. 기존 파일은 덮어쓰지 않습니다.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"명시적으로 요청한 ASR 모델 약 1.61GB를 다운로드합니다: {ASR_REPO}@{ASR_REVISION}", flush=True)
    download = ("import sys; from huggingface_hub import snapshot_download; "
                "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], "
                "allow_patterns=['config.json','weights.safetensors','README.md'], token=False, max_workers=2)")
    with tempfile.TemporaryDirectory(prefix=".whisper-download-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        subprocess.run([str(python), "-I", "-c", download, ASR_REPO, ASR_REVISION, str(staging)],
                       check=True, env=child_environment(offline=False))
        validate_asr(staging)
        for name, (size, digest) in ASR_FILES.items():
            path = staging / name
            if path.stat().st_size != size or file_sha256(path) != digest:
                raise SetupError(f"다운로드한 ASR 파일의 고정 SHA256/크기가 다릅니다: {name}")
        manifest = {"schema": "myvote.demo_asr_download", "schema_version": 1,
                    "repository": ASR_REPO, "revision": ASR_REVISION,
                    "files": {name: {"bytes": size, "sha256": digest} for name, (size, digest) in ASR_FILES.items()},
                    "source": f"https://huggingface.co/{ASR_REPO}/tree/{ASR_REVISION}",
                    "remote_code_executed": False}
        (staging / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Exclusive mkdir prevents replacement even if another launcher finishes first.
        destination.mkdir(exist_ok=False)
        for name in (*ASR_FILES, "README.md", "download-manifest.json"):
            if (staging / name).is_file():
                (staging / name).rename(destination / name)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def query_models() -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    request = urllib.request.Request(LM_ENDPOINT + "/v1/models", headers={"Accept": "application/json"})
    try:
        with opener.open(request, timeout=5) as response:
            raw = response.read(MAX_MODEL_RESPONSE + 1)
        if len(raw) > MAX_MODEL_RESPONSE:
            raise SetupError("LM Studio 모델 목록 응답이 크기 제한을 넘었습니다.")
        payload = json.loads(raw)
    except (urllib.error.URLError, TimeoutError, ValueError, UnicodeError):
        raise SetupError("LM Studio의 Local Server를 127.0.0.1:1234에서 시작하고 모델을 로드하세요. 모델 목록을 읽지 못했습니다.") from None
    return payload


def select_model(payload, requested: str | None) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list) or len(payload["data"]) > 256:
        raise SetupError("LM Studio 모델 목록 형식이 잘못되었습니다.")
    ids = []
    for item in payload["data"]:
        value = item.get("id") if isinstance(item, dict) else None
        if not isinstance(value, str) or not SAFE_MODEL_ID.fullmatch(value):
            raise SetupError("LM Studio 모델 목록에 잘못된 모델 ID가 있습니다.")
        if value not in ids:
            ids.append(value)
    if requested is not None and requested in ids:
        return requested
    candidates = [value for value in ids if QWEN_4B.search(value)] if requested is None else []
    if len(candidates) == 1:
        return candidates[0]
    displayed = json.dumps(ids[:20], ensure_ascii=True)
    raise SetupError("모델을 하나로 선택할 수 없습니다. LM Studio에 정확한 모델을 로드하고 --llm-model ID를 지정하세요. 목록: " + displayed)


def validate_dependencies_and_tls(python: Path, args) -> None:
    code = """import ipaddress, sys
import grpc, numpy, onnxruntime, mlx_whisper, torch, torchaudio
from cryptography import x509
from myvote_engine.tls import load_tls_material
material = load_tls_material(sys.argv[1], sys.argv[2], sys.argv[3], role='server')
cert = x509.load_pem_x509_certificate(material.certificate_chain)
addresses = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(x509.IPAddress)
if ipaddress.ip_address(sys.argv[4]) not in addresses:
    raise SystemExit('Server certificate IP SAN does not match --host; use the issued IP or create matching certificates.')
"""
    subprocess.run([str(python), "-I", "-c", code, str(args.cert_dir / "server.crt"),
                    str(args.cert_dir / "server.key"), str(args.cert_dir / "ca.crt"), args.host],
                   check=True, cwd=args.kit_root, env=child_environment(offline=True))


def gateway_command(python: Path, args, model: str) -> list[str]:
    validate_translation_options(args, model)
    command = [str(python), "-I", "-m", "myvote_engine.gateway_server", "--host", args.host,
               "--port", str(args.port), "--backend", "lmstudio", "--endpoint", LM_ENDPOINT,
               "--model", model, "--cert", str(args.cert_dir / "server.crt"),
               "--key", str(args.cert_dir / "server.key"), "--client-ca", str(args.cert_dir / "ca.crt"),
               "--asr-model", str(args.models_dir / "whisper-turbo"),
               "--vad-model", str(args.models_dir / "silero_vad.onnx"),
               "--speaker-segmentation-model", str(args.models_dir / "segmentation.onnx"),
               "--speaker-embedding-model", str(args.models_dir / "wespeaker.onnx")]
    if getattr(args, "translation_profile", "generic") == "translategemma":
        command += ["--translation-profile", "translategemma", "--orchestrator-model", args.orchestrator_model,
                    "--translation-workers", str(args.translation_workers if args.translation_workers is not None else 4)]
        for identifier in args.translation_model_id or []:
            command += ["--translation-model-id", identifier]
    if not args.skip_warmup:
        command.append("--warmup")
    if args.experimental_speakers or args.speaker_profile is not None:
        command += ["--speaker-profile", str(args.speaker_profile or args.models_dir / "speaker-profile.json")]
    if getattr(args, "experimental_speaker_pooling", False):
        command.append("--experimental-speaker-pooling")
    if getattr(args, "experimental_speaker_activity", None) is not None:
        command += ["--experimental-speaker-activity", args.experimental_speaker_activity]
    if getattr(args, "experimental_overlap_model", None) is not None:
        command += ["--experimental-overlap-model", str(args.experimental_overlap_model),
                    "--overlap-threads", str(args.overlap_threads)]
    return command


def main(argv=None, *, kit_root: Path | None = None, gateway_entrypoint: Path | None = None) -> int:
    try:
        args = parse_args(argv, kit_root=kit_root)
        validate_runtime()  # Refuse unsupported hosts before filesystem changes/network.
        validate_assets(args)
        protect_certificates(args.cert_dir)
        python = ensure_environment(args.kit_root, install=args.install)
        asr = args.models_dir / "whisper-turbo"
        if args.download_asr and not asr.exists():
            download_asr(python, asr)
        elif args.download_asr:
            print("기존 로컬 ASR 모델을 재사용합니다. 다운로드·덮어쓰기를 하지 않습니다.", flush=True)
        validate_asr(asr)
        validate_dependencies_and_tls(python, args)
        model_list = query_models()
        model = select_model(model_list, args.translation_model_id[0] if args.translation_model_id else args.llm_model)
        validate_translation_options(args, model)
        if args.orchestrator_model is not None:
            select_model(model_list, args.orchestrator_model)
            for identifier in args.translation_model_id or []:
                select_model(model_list, identifier)
        print(f"선택한 LM Studio 모델: {model}", flush=True)
        if args.orchestrator_model is not None:
            print(f"오케스트레이터: {args.orchestrator_model} · 번역 논리 worker {args.translation_workers or 4}개", flush=True)
        print("LM Studio에서 이 모델을 미리 로드하고 Thinking을 끄세요. 모델 목록 확인은 실제 추론 성공 검사가 아닙니다.", flush=True)
        if args.experimental_speakers or args.speaker_profile is not None:
            print("화자 실험 프로필을 적용합니다. 화자 확정은 번역 뒤에 갱신될 수 있으며 영상·강의·방송의 정확도는 추가 검증이 필요합니다.", flush=True)
        else:
            print("화자 모델을 기본 미보정 설정으로 실행합니다. 자동 ID 확인용 연구 프로필은 --experimental-speakers로 선택할 수 있습니다.", flush=True)
        print(f"데모 서버를 {args.host}:{args.port}에서 시작합니다. 이 터미널을 유지하세요. 종료: Ctrl+C", flush=True)
        if not args.skip_warmup:
            print("시작 전 ASR 무음·짧은 번역으로 연결을 점검합니다. 실제 음성 품질이나 화자 모델 준비를 입증하는 검사는 아닙니다.", flush=True)
        print("gateway.listening 로그 뒤에 Windows 앱에서 연결하세요. 실제 Mac 추론과 지연은 이번 데모에서 확인해야 합니다.", flush=True)
        command = gateway_command(python, args, model)
        if gateway_entrypoint is not None:
            # The separate update launcher reuses this kit's venv/models/TLS,
            # loading its own verified source without replacing installed code.
            regular_file(gateway_entrypoint)
            command = [str(python), "-I", str(gateway_entrypoint.resolve())] + command[4:]
        return subprocess.run(command, cwd=args.kit_root,
                              env=child_environment(offline=True), check=False).returncode
    except KeyboardInterrupt:
        return 130
    except (SetupError, OSError, subprocess.SubprocessError) as exc:
        if isinstance(exc, SetupError):
            message = str(exc)
        elif isinstance(exc, subprocess.CalledProcessError):
            message = f"하위 실행이 종료 코드 {exc.returncode}로 실패했습니다. 바로 위 로그를 확인하세요."
        else:
            message = f"준비 작업을 완료하지 못했습니다 ({type(exc).__name__}). 폴더 권한·파일과 위 로그를 확인하세요."
        print("데모 준비 실패: " + message, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
