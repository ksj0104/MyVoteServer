"""Mac 음성 서버와 번역·오케스트레이터 모델 준비 점검 및 업데이트 실행 진입점."""

import argparse
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import tomllib
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parent
OVERLAP_MODEL_BYTES = 20_394_640
OVERLAP_MODEL_SHA256 = "8d97f012f7b2f22bb79cb0d0983a7ba27a52c1796ee3f63cbf25b4d28630adce"
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+\-]{0,255}\Z")


def read_config(path):
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    bundle = Path(config["demo"]["bundle_dir"]).expanduser()
    config["demo"]["bundle_dir"] = (path.parent / bundle).resolve()
    ipaddress.IPv4Address(config["network"]["expected_ip"])
    # 첫 데모의 연결 설정을 사용합니다.
    if config["network"]["port"] != 50051:
        raise ValueError("현재 데모에서 확인된 서버 포트는 50051뿐입니다.")
    llm = config["lmstudio"]
    if llm["base_url"] != "http://127.0.0.1:1234":
        raise ValueError("첫 데모의 LM Studio 주소는 http://127.0.0.1:1234입니다.")
    if not isinstance(llm["model_id"], str) or not llm["model_id"].strip():
        raise ValueError("lmstudio.model_id에 로드한 모델의 정확한 ID를 넣으세요.")
    if type(llm["context_length"]) is not int or llm["context_length"] <= 0:
        raise ValueError("lmstudio.context_length는 양의 정수여야 합니다.")
    if type(llm.get("context_review_json_schema", False)) is not bool:
        raise ValueError("lmstudio.context_review_json_schema는 true/false여야 합니다.")
    if type(llm.get("semantic_boundary_refinement", False)) is not bool:
        raise ValueError("lmstudio.semantic_boundary_refinement는 true/false여야 합니다.")
    if type(llm.get("semantic_quality_first", False)) is not bool:
        raise ValueError("lmstudio.semantic_quality_first는 true/false여야 합니다.")
    if llm.get("semantic_quality_first", False) and not llm.get("semantic_boundary_refinement", False):
        raise ValueError("semantic_quality_first에는 semantic_boundary_refinement = true가 필요합니다.")
    if llm.get("semantic_boundary_refinement", False) and "semantic_boundary_prompt" not in llm:
        raise ValueError("semantic_boundary_refinement에는 semantic_boundary_prompt 설정이 필요합니다.")
    if type(llm.get("semantic_low_latency", False)) is not bool:
        raise ValueError("lmstudio.semantic_low_latency는 true/false여야 합니다.")
    if llm.get("semantic_low_latency", False) and not llm.get("semantic_quality_first", False):
        raise ValueError("semantic_low_latency에는 semantic_quality_first = true가 필요합니다.")
    quality_defaults = {"semantic_selection_timeout_s": 5, "semantic_translation_timeout_s": 15,
                        "semantic_max_hold_s": 12, "semantic_total_age_s": 45,
                        "semantic_min_request_interval_s": 1.5, "semantic_latency_target_s": 2}
    for key, default in quality_defaults.items():
        value = llm.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not .001 <= value <= 120:
            raise ValueError(f"lmstudio.{key}는 0.001~120초 유한 숫자여야 합니다.")
    source_lifetime = llm.get("semantic_total_age_s", 45)
    inactivity = llm.get("semantic_inactivity_flush_s", 0)
    if (type(inactivity) not in (int, float) or not math.isfinite(inactivity)
            or not 0 <= inactivity < source_lifetime):
        raise ValueError("semantic_inactivity_flush_s는 0(비활성) 이상, 원문 체류 시간 미만이어야 합니다.")
    if inactivity and not llm.get("semantic_quality_first", False):
        raise ValueError("semantic_inactivity_flush_s에는 semantic_quality_first = true가 필요합니다.")
    request_budget = llm.get("semantic_selection_timeout_s", 5) + llm.get("semantic_translation_timeout_s", 15)
    if (request_budget > min(120, source_lifetime)
            or llm.get("semantic_max_hold_s", 12) > source_lifetime
            or llm.get("semantic_min_request_interval_s", 1.5) >= source_lifetime):
        raise ValueError("구간 선택+번역 시간 및 대기 재확인 시간은 전체 원문 체류 시간 이하여야 합니다.")
    update = config.get("speaker_update", {})
    if llm.get("context_review_json_schema", False) and not update.get("enabled", False):
        raise ValueError("문맥 JSON 호환 설정은 소스 업데이트 실행에서만 사용할 수 있습니다.")
    for field in ("enabled", "overlap_enabled"):
        if field in update and type(update[field]) is not bool:
            raise ValueError(f"speaker_update.{field}는 true/false여야 합니다.")
    profile = llm.get("translation_profile", "generic")
    if profile not in ("generic", "translategemma"):
        raise ValueError("lmstudio.translation_profile은 generic 또는 translategemma여야 합니다.")
    routing_fields = ("orchestrator_model_id", "orchestrator_context_length",
                      "translation_workers", "translation_model_ids")
    if profile == "generic":
        if any(field in llm for field in routing_fields):
            raise ValueError("오케스트레이터·번역 worker 설정에는 translategemma 프로필이 필요합니다.")
    else:
        if not update.get("enabled", False):
            raise ValueError("translategemma 프로필에는 speaker_update.enabled = true가 필요합니다.")
        workers = llm.get("translation_workers", 4)
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError("lmstudio.translation_workers는 1~4 정수여야 합니다.")
        translation_ids = llm.get("translation_model_ids", [llm["model_id"]])
        if (not isinstance(translation_ids, list) or not 1 <= len(translation_ids) <= workers
                or any(not isinstance(value, str) or not MODEL_ID.fullmatch(value)
                       for value in translation_ids)
                or len(set(translation_ids)) != len(translation_ids)):
            raise ValueError("translation_model_ids는 worker 수 이하의 중복 없는 정확한 모델 ID 목록이어야 합니다.")
        orchestrator = llm.get("orchestrator_model_id")
        if (not isinstance(orchestrator, str) or not MODEL_ID.fullmatch(orchestrator)
                or orchestrator in translation_ids or orchestrator == llm["model_id"]
                or "translategemma" in re.sub(r"[^a-z0-9]", "", orchestrator.lower())):
            raise ValueError("orchestrator_model_id에는 번역기와 다른 일반 지시 모델의 정확한 ID를 넣으세요.")
        if not MODEL_ID.fullmatch(llm["model_id"]):
            raise ValueError("lmstudio.model_id에는 공백·제어 문자 없는 정확한 모델 ID를 넣으세요.")
        orchestrator_context = llm.get("orchestrator_context_length")
        if type(orchestrator_context) is not int or orchestrator_context <= 0:
            raise ValueError("lmstudio.orchestrator_context_length는 양의 정수여야 합니다.")
    if "semantic_boundary_prompt" in llm:
        prompt = llm["semantic_boundary_prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("lmstudio.semantic_boundary_prompt에는 비어 있지 않은 파일 경로를 넣으세요.")
        if (profile != "translategemma"
                or llm.get("orchestrator_model_id") != "google/gemma-4-26b-a4b"
                or llm.get("context_review_json_schema") is not True):
            raise ValueError("semantic_boundary_prompt는 translategemma·google/gemma-4-26b-a4b·문맥 JSON 호환 설정이 필요합니다.")
        llm["semantic_boundary_prompt"] = (path.parent / Path(prompt).expanduser()).absolute()
    if update.get("enabled", False):
        for field in ("bundle_dir", "overlap_model"):
            if field == "overlap_model" and not update.get("overlap_enabled", False):
                continue
            target = Path(update[field]).expanduser()
            update[field] = (path.parent / target).absolute()
        if update.get("overlap_enabled", False):
            threads = update["overlap_threads"]
            if type(threads) is not int or not 1 <= threads <= 8:
                raise ValueError("speaker_update.overlap_threads는 1~8 정수여야 합니다.")
    return config


def check(config):
    failures = []

    def report(ok, name, detail):
        print(f"[{'OK' if ok else 'BLOCK'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    mac_version = platform.mac_ver()[0]
    report(platform.system() == "Darwin" and bool(mac_version)
           and int(mac_version.split(".")[0]) >= 14,
           "macOS", mac_version or platform.system())
    report(platform.machine() == "arm64", "CPU", platform.machine())
    report(sys.version_info[:2] in ((3, 12), (3, 13)), "Python",
           f"{platform.python_version()} ({sys.executable}); 요구 버전: 3.12/3.13")

    network = config["network"]
    address = network["expected_ip"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((address, 0))
        report(True, "Mac IP", address)
    except OSError:
        report(False, "Mac IP", f"{address}가 이 Mac에 없습니다. Windows 주소와 인증서 SAN도 함께 맞추세요.")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((address, network["port"]))
        report(True, "서버 포트", f"{address}:{network['port']} 바인딩 가능")
    except OSError:
        report(False, "서버 포트", "50051을 사용할 수 없습니다. 기존 서버 실행 여부와 IP를 확인하세요.")

    bundle = config["demo"]["bundle_dir"]
    required = {"start-demo.command": "file", "scripts": "dir",
                "models": "dir", "connection": "dir"}
    missing = []
    for name, kind in required.items():
        item = bundle / name
        if not (item.is_file() if kind == "file" else item.is_dir()):
            missing.append(name)
        elif kind == "dir" and not any(item.iterdir()):
            missing.append(f"{name} (비어 있음)")
    report(not missing, "데모 패키지", f"{bundle}" + (f" — 없음: {', '.join(missing)}" if missing else ""))
    if not missing:
        unexpected = [p.name for p in (bundle / "connection").rglob("*")
                      if p.is_file() and p.name.lower() in ("client.key", "ca.key")]
        report(not unexpected, "키 구성", "client.key/ca.key가 서버 connection에 없어야 합니다."
               if unexpected else "connection에 client.key/ca.key 없음 (키 내용은 읽지 않음)")
        print("[INFO] 모델 파일·서버 인증서 SAN/유효기간/키 일치는 공식 실행기의 검사가 추가로 필요합니다.")

    update = config.get("speaker_update", {})
    if update.get("enabled", False):
        update_dir = update["bundle_dir"]
        required_update = ("start-update.command", "update-manifest.json",
                           "scripts/mac_update_start.py", "scripts/mac_update_gateway.py",
                           "src/myvote_engine/gateway_server.py")
        absent = [name for name in required_update if not (update_dir / name).is_file()]
        report(not absent, "화자 업데이트", f"{update_dir}" +
               (f" — 없음: {', '.join(absent)}" if absent else " (실행 시 공식 manifest 전체 검증)"))
        report((bundle / ".venv/bin/python").is_file(), "재사용 서버 환경", str(bundle / ".venv"))
        if config["lmstudio"].get("context_review_json_schema", False):
            compat_files = ("start_update_with_compat.py", "gemma_update_gateway.py", "gemma_json_compat.py")
            report(all((ROOT / "scripts" / name).is_file() for name in compat_files),
                   "문맥 JSON 호환", "프로젝트 호환 모듈 사용; 업데이트 원본·결과 검증 규칙 유지")
        if "semantic_boundary_prompt" in config["lmstudio"]:
            prompt = config["lmstudio"]["semantic_boundary_prompt"]
            try:
                prompt_ok = (not prompt.is_symlink() and prompt.is_file()
                             and prompt.stat().st_size <= 65536)
            except OSError:
                prompt_ok = False
            module_ok = (ROOT / "scripts/gemma_boundary_prompt.py").is_file()
            report(prompt_ok and module_ok, "구간 선택 프롬프트",
                   f"{prompt}; 64KiB 이하 일반 파일: {prompt_ok}; 전용 모듈: {module_ok}; 파싱은 실행 시 검증")
        if config["lmstudio"].get("semantic_boundary_refinement", False):
            refinement_files = ("semantic_boundary_routing.py", "semantic_boundary_pipeline.py")
            report(all((ROOT / "scripts" / name).is_file() for name in refinement_files),
                   "구간 선택 보완", "semantic_boundary_routing.py·semantic_boundary_pipeline.py 확인")
        if config["lmstudio"].get("semantic_quality_first", False):
            quality_files = ("semantic_quality_policy.py", "semantic_quality_gateway.py", "semantic_quality_output.py",
                             "semantic_quality_provider.py", "semantic_quality_capacity.py")
            if config["lmstudio"].get("semantic_inactivity_flush_s", 0):
                quality_files += ("semantic_transcript_idle.py",)
            elif config["lmstudio"].get("semantic_low_latency", False):
                quality_files += ("semantic_latency_metrics.py",)
            report(all((ROOT / "scripts" / name).is_file() for name in quality_files),
                   "의미 구간 품질 우선", "품질 정책·세션 안내·겹침 출력 보호 모듈 확인")
        if update.get("overlap_enabled", False):
            model = update["overlap_model"]
            model_ok = (not model.is_symlink() and model.is_file()
                        and model.stat().st_size == OVERLAP_MODEL_BYTES)
            if model_ok:
                with model.open("rb") as source:
                    model_ok = hashlib.file_digest(source, "sha256").hexdigest() == OVERLAP_MODEL_SHA256
            report(model_ok, "겹침 분리 모델", f"{model}; 고정 크기/SHA256 검사")
            print(f"[INFO] 겹침 분리 실험 활성화, Torch CPU 스레드 {update['overlap_threads']}개. Windows caption_groups_v1 지원 빌드가 필요합니다.")

    llm = config["lmstudio"]
    endpoint = llm["base_url"]
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(endpoint + "/v1/models", timeout=5) as response:
            payload = json.load(response)
        ids = [item.get("id") for item in payload["data"]]
        targets = [(llm["model_id"], llm["context_length"], None)]
        if llm.get("translation_profile", "generic") == "translategemma":
            translation_ids = llm.get("translation_model_ids", [llm["model_id"]])
            workers = llm.get("translation_workers", 4)
            targets = [(identifier, llm["context_length"],
                        sum(slot % len(translation_ids) == index for slot in range(workers)))
                       for index, identifier in enumerate(translation_ids)]
            targets.append((llm["orchestrator_model_id"], llm["orchestrator_context_length"], None))
        # 공식 실행기는 명시 목록을 사용해도 --llm-model을 모델 목록에서 먼저 선택합니다.
        required_ids = list(dict.fromkeys([llm["model_id"], *(item[0] for item in targets)]))
        for identifier in required_ids:
            report(identifier in ids, "LM Studio API", f"{endpoint}; 요청 모델: {identifier}")
        # /v1/models에는 아직 메모리에 로드하지 않은 모델도 나올 수 있습니다.
        with opener.open(endpoint + "/api/v1/models", timeout=5) as response:
            native = json.load(response)
        for identifier, expected_context, required_parallel in targets:
            selected = [model for model in native["models"]
                        if any(item.get("id") == identifier
                               for item in model.get("loaded_instances", []))]
            instances = [instance for model in selected
                         for instance in model.get("loaded_instances", [])
                         if instance.get("id") == identifier]
            report(bool(instances), "모델 로드", identifier)
            if instances:
                contexts = [item.get("config", {}).get("context_length") for item in instances]
                report(all(type(value) is int and value == expected_context for value in contexts),
                       "문맥 길이", f"{identifier}; {contexts}; 요구값: {expected_context}")
                if required_parallel is not None:
                    parallel = [next((item.get("config", {})[field]
                                      for field in ("parallel", "max_parallel_predictions")
                                      if field in item.get("config", {})), None)
                                for item in instances]
                    report(all(type(value) is int and value >= required_parallel for value in parallel),
                           "번역 런타임 병렬 처리", f"{identifier}; {parallel}; 필요한 동시 처리: {required_parallel}")
                defaults = [item.get("capabilities", {}).get("reasoning", {}).get("default")
                            for item in selected]
                print(f"[INFO] {identifier} Thinking 기본값 메타데이터: {defaults} (현재 설정과 다를 수 있음). OFF 적용은 서버 준비 추론에서 검증합니다.")
    except (URLError, OSError, ValueError, KeyError, TypeError, AttributeError):
        report(False, "LM Studio 응답", f"{endpoint}의 인증 없는 모델 목록/로드 설정을 확인할 수 없습니다.")
    print("[INFO] 이 점검은 파일 구조·환경 점검입니다. 실제 준비 추론과 Windows 연결은 별도 검증입니다.")
    print(f"\n차단 항목: {len(failures)}개. 서버 실행 후 gateway.listening / port: 50051을 확인하세요.")
    return not failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "start"), nargs="?", default="check")
    parser.add_argument("--config", type=Path, default=ROOT / "server.toml")
    parser.add_argument("--install", action="store_true", help="공식 실행기에 의존성 설치 요청")
    parser.add_argument("--download-asr", action="store_true", help="공식 실행기에 Whisper 다운로드 요청")
    parser.add_argument("--experimental-speakers", action="store_true", help="문서의 화자 실험 프로필 사용")
    args = parser.parse_args(argv)
    if args.command == "check" and any((args.install, args.download_asr, args.experimental_speakers)):
        parser.error("설치·다운로드·화자 옵션은 start 명령에서 사용하세요.")
    try:
        config = read_config(args.config.resolve())
        update = config.get("speaker_update", {})
        if update.get("enabled", False) and (args.install or args.download_asr):
            parser.error("화자 업데이트는 준비된 환경을 재사용합니다. --install/--download-asr를 빼고 실행하세요.")
        ready = check(config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[BLOCK] 설정/파일 확인 실패 ({type(exc).__name__}). server.toml의 경로·형식·필수 값을 확인하세요.", file=sys.stderr)
        return 2
    if not ready:
        return 1
    if args.command == "check":
        return 0
    bundle = config["demo"]["bundle_dir"]
    if update.get("enabled", False):
        if config["lmstudio"].get("context_review_json_schema", False):
            command = ["/usr/bin/caffeinate", "-i", str(bundle / ".venv/bin/python"), "-I",
                       str(ROOT / "scripts/start_update_with_compat.py"),
                       "--update-dir", str(update["bundle_dir"]), "--demo-dir", str(bundle)]
            if "semantic_boundary_prompt" in config["lmstudio"]:
                command.extend(("--semantic-boundary-prompt",
                                str(config["lmstudio"]["semantic_boundary_prompt"])))
            if config["lmstudio"].get("semantic_boundary_refinement", False):
                command.append("--semantic-boundary-refinement")
            if config["lmstudio"].get("semantic_quality_first", False):
                command.append("--semantic-quality-first")
                if config["lmstudio"].get("semantic_low_latency", False):
                    command.append("--semantic-low-latency")
                for key in ("semantic_selection_timeout_s", "semantic_translation_timeout_s",
                            "semantic_max_hold_s", "semantic_total_age_s",
                            "semantic_min_request_interval_s", "semantic_latency_target_s",
                            "semantic_inactivity_flush_s"):
                    if key in config["lmstudio"]:
                        command.extend(("--" + key.replace("_", "-"), str(config["lmstudio"][key])))
        else:
            entrypoint = update["bundle_dir"] / "start-update.command"
            command = ["/usr/bin/caffeinate", "-i", "/bin/bash", str(entrypoint), str(bundle)]
    else:
        command = ["/usr/bin/caffeinate", "-i", "/bin/bash", str(bundle / "start-demo.command")]
    for enabled, option in ((args.install, "--install"), (args.download_asr, "--download-asr"),
                            (args.experimental_speakers, "--experimental-speakers")):
        if enabled:
            command.append(option)
    command.extend(("--llm-model", config["lmstudio"]["model_id"],
                    "--host", config["network"]["expected_ip"],
                    "--port", str(config["network"]["port"])))
    llm = config["lmstudio"]
    if llm.get("translation_profile", "generic") == "translategemma":
        command.extend(("--translation-profile", "translategemma",
                        "--orchestrator-model", llm["orchestrator_model_id"],
                        "--translation-workers", str(llm.get("translation_workers", 4))))
        for identifier in llm.get("translation_model_ids", []):
            command.extend(("--translation-model-id", identifier))
    if update.get("enabled", False) and update.get("overlap_enabled", False):
        command.extend(("--experimental-overlap-model", str(update["overlap_model"]),
                        "--overlap-threads", str(update["overlap_threads"])))
    environment = os.environ.copy()
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    os.chdir(bundle)
    print("공식 데모 실행기를 시작합니다. 실행 중 자동 잠자기를 방지합니다. 종료: Ctrl+C", flush=True)
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    raise SystemExit(main())
