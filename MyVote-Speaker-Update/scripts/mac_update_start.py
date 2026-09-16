"""Start a side-by-side source update using an existing Mac demo environment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys


def verify_update(root: Path) -> dict:
    manifest_path = root / "update-manifest.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 262144:
        raise ValueError("Invalid update manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "myvote.mac_source_update" or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported update manifest")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 512:
        raise ValueError("Invalid update inventory")
    seen = set()
    for row in rows:
        name = row["path"]
        if (not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
                or any(c in name for c in "\\:\x00")
                or any(part in ("", ".", "..") for part in name.split("/"))
                or name.casefold() in seen):
            raise ValueError("Invalid update path")
        seen.add(name.casefold())
        path = root / name
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root.resolve())
                or not 0 < path.stat().st_size <= 1048576):
            raise ValueError("Invalid update file")
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Update checksum mismatch: " + name)
    required = {"scripts/mac_demo_setup.py", "scripts/mac_update_start.py",
                "scripts/mac_update_gateway.py", "src/myvote_engine/__init__.py",
                "src/myvote_engine/gateway_server.py"}
    if not required <= seen:
        raise ValueError("Incomplete update")
    if "speaker_profile" in manifest:
        profile = manifest["speaker_profile"]
        if not isinstance(profile, str) or profile not in {row["path"] for row in rows}:
            raise ValueError("Speaker profile must be a registered update file")
    # Prevent unlisted Python modules from shadowing the verified engine.
    for folder in ("src", "scripts"):
        for path in (root / folder).rglob("*.py"):
            if path.relative_to(root).as_posix().casefold() not in seen:
                raise ValueError("Unlisted update module")
    return manifest


def _verified_source(root: Path, name: str, row: dict) -> bytes:
    """Read and recheck exactly the bytes that will be compiled, never a pyc."""
    path = root / name
    if (path.is_symlink() or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
            or not 0 < path.stat().st_size <= 1048576):
        raise ValueError("Invalid update source file")
    raw = path.read_bytes()
    if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError("Update checksum mismatch: " + name)
    return raw


class VerifiedSourceLoader(importlib.machinery.SourceFileLoader):
    """Compile manifest-bound source; never read, write or delete bytecode caches."""

    def __init__(self, fullname: str, root: Path, name: str, row: dict):
        super().__init__(fullname, str(root / name))
        self.root, self.relative_name, self.row = root, name, dict(row)

    def get_code(self, fullname):
        raw = _verified_source(self.root, self.relative_name, self.row)
        return self.source_to_code(raw, self.path)


class VerifiedEngineFinder(importlib.abc.MetaPathFinder):
    """Only listed Python sources may satisfy any myvote_engine import.

    Dependencies still use the existing venv's ordinary import machinery. The
    update src directory is not added to sys.path, so it cannot shadow those
    dependencies. Missing engine modules fail instead of trying an older
    installed package, a cached bytecode-only module or a native extension.
    """

    def __init__(self, root: Path, manifest: dict):
        self.root = root
        self.rows = {row["path"]: dict(row) for row in manifest["files"]}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "myvote_engine" and not fullname.startswith("myvote_engine."):
            return None
        stem = "src/" + fullname.replace(".", "/")
        for name in (stem + "/__init__.py", stem + ".py"):
            if name in self.rows:
                loader = VerifiedSourceLoader(fullname, self.root, name, self.rows[name])
                return importlib.util.spec_from_file_location(fullname, loader.path, loader=loader)
        raise ModuleNotFoundError("Engine module is not in the verified update: " + fullname, name=fullname)


def install_engine_importer(root: Path, *, manifest: dict | None = None):
    """Install once in the fresh gateway process, refusing preloaded old code."""
    if any(name == "myvote_engine" or name.startswith("myvote_engine.") for name in sys.modules):
        raise ValueError("Engine was already imported; use the isolated update gateway process")
    manifest = verify_update(root) if manifest is None else manifest
    finder = VerifiedEngineFinder(root, manifest)
    sys.meta_path.insert(0, finder)
    return finder


def load_bootstrap(root: Path, *, manifest: dict | None = None):
    manifest = verify_update(root) if manifest is None else manifest
    name = "scripts/mac_demo_setup.py"
    row = next(row for row in manifest["files"] if row["path"] == name)
    loader = VerifiedSourceLoader("myvote_update_bootstrap", root, name, row)
    specification = importlib.util.spec_from_file_location(
        "myvote_update_bootstrap", loader.path, loader=loader)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main(argv=None, *, update_root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--demo-dir", type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    root = (update_root or Path(__file__).resolve().parents[1]).resolve()
    try:
        manifest = verify_update(root)
        demo = load_bootstrap(root, manifest=manifest)
        demo.validate_runtime()
        existing = args.demo_dir.expanduser().resolve(strict=True)
        if Path(sys.prefix).resolve() != (existing / ".venv").resolve():
            raise ValueError("Use start-update.command with the existing Mac demo folder")
        if any(option in remaining for option in ("--install", "--download-asr")):
            raise ValueError("This update reuses the prepared environment; install/download flags are not used")
        print("화자 수정판을 실행합니다. 기존 모델·인증서·설치된 엔진은 그대로 재사용합니다.", flush=True)
        print("검증한 업데이트: " + json.dumps(manifest.get("release"), ensure_ascii=True), flush=True)
        defaults = ["--experimental-speakers"]
        if "speaker_profile" in manifest:
            defaults += ["--speaker-profile", str(root / manifest["speaker_profile"])]
        return demo.main([*defaults, *remaining], kit_root=existing,
                         gateway_entrypoint=root / "scripts/mac_update_gateway.py")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("Update startup failed: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
