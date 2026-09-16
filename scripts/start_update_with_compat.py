"""Reuse verified update setup, selecting the project's explicit JSON gateway."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import ModuleType


def load_update(root):
    source = root / "scripts/mac_update_start.py"
    if (source.is_symlink() or not source.is_file()
            or not source.resolve().is_relative_to(root)
            or not 0 < source.stat().st_size <= 1048576):
        raise ValueError("Invalid update launcher")
    launcher = ModuleType("myvote_verified_update_start")
    launcher.__file__ = str(source)
    exec(compile(source.read_bytes(), str(source), "exec"), launcher.__dict__)
    manifest = launcher.verify_update(root)
    bootstrap = launcher.load_bootstrap(root, manifest=manifest)
    bootstrap.validate_runtime()
    return launcher, manifest, bootstrap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--update-dir", type=Path, required=True)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--semantic-boundary-prompt", type=Path)
    parser.add_argument("--semantic-boundary-refinement", action="store_true")
    parser.add_argument("--semantic-quality-first", action="store_true")
    parser.add_argument("--semantic-low-latency", action="store_true")
    quality_fields = ("semantic_selection_timeout_s", "semantic_translation_timeout_s",
                      "semantic_max_hold_s", "semantic_total_age_s",
                      "semantic_min_request_interval_s", "semantic_latency_target_s",
                      "semantic_inactivity_flush_s")
    for key in quality_fields:
        parser.add_argument("--" + key.replace("_", "-"), type=float)
    args, remaining = parser.parse_known_args(argv)
    try:
        update = args.update_dir.expanduser().resolve(strict=True)
        demo = args.demo_dir.expanduser().resolve(strict=True)
        if Path(sys.prefix).resolve() != (demo / ".venv").resolve():
            raise ValueError("Use the existing demo's Python environment")
        if any(option in remaining for option in ("--install", "--download-asr")):
            raise ValueError("Prepare the demo environment before running the update")
        _, manifest, bootstrap = load_update(update)
        # Clear inherited state when the explicit configuration is disabled.
        os.environ.pop("MYVOTE_SEMANTIC_BOUNDARY_PROMPT", None)
        os.environ.pop("MYVOTE_BOUNDARY_REFINEMENT", None)
        os.environ.pop("MYVOTE_SEMANTIC_QUALITY_FIRST", None)
        os.environ.pop("MYVOTE_SEMANTIC_QUALITY_TIMINGS", None)
        os.environ.pop("MYVOTE_SEMANTIC_LOW_LATENCY", None)
        if args.semantic_low_latency and not args.semantic_quality_first:
            raise ValueError("Low latency requires the quality-first safeguards")
        if args.semantic_inactivity_flush_s and not args.semantic_quality_first:
            raise ValueError("Inactivity flush requires the quality-first safeguards")
        if args.semantic_quality_first and not args.semantic_boundary_refinement:
            raise ValueError("Quality-first requires semantic boundary refinement")
        if args.semantic_boundary_refinement and args.semantic_boundary_prompt is None:
            raise ValueError("Boundary refinement requires an explicit boundary prompt document")
        if args.semantic_boundary_prompt is not None:
            import runpy
            document = args.semantic_boundary_prompt.expanduser().absolute()
            prompt_support = runpy.run_path(str(Path(__file__).with_name("gemma_boundary_prompt.py")))
            prompt_support["load_boundary_prompt"](document)
            os.environ["MYVOTE_SEMANTIC_BOUNDARY_PROMPT"] = str(document)
        if args.semantic_boundary_refinement:
            os.environ["MYVOTE_BOUNDARY_REFINEMENT"] = "1"
        if args.semantic_quality_first:
            os.environ["MYVOTE_SEMANTIC_QUALITY_FIRST"] = "1"
            os.environ["MYVOTE_SEMANTIC_QUALITY_TIMINGS"] = json.dumps({
                key: getattr(args, key) for key in quality_fields if getattr(args, key) is not None})
        if args.semantic_low_latency:
            os.environ["MYVOTE_SEMANTIC_LOW_LATENCY"] = "1"
        defaults = ["--experimental-speakers"]
        if "speaker_profile" in manifest:
            defaults += ["--speaker-profile", str(update / manifest["speaker_profile"])]
        print("검증한 업데이트: " + json.dumps(manifest["release"]), flush=True)
        print("프로젝트 호환 설정: 의미 구간 선택·문맥 재검토 JSON schema (원본 소스·파서 유지)", flush=True)
        if args.semantic_boundary_prompt is not None:
            print("구간 선택 시스템 프롬프트: " + str(document), flush=True)
        os.environ["MYVOTE_VERIFIED_UPDATE_DIR"] = str(update)
        return bootstrap.main([*defaults, *remaining], kit_root=demo,
                              gateway_entrypoint=Path(__file__).with_name("gemma_update_gateway.py"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"호환 실행 준비 실패: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
