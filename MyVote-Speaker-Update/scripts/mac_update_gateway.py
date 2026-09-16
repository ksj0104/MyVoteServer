"""Verified update entrypoint; dependencies come from the existing Mac venv."""
from pathlib import Path
from types import ModuleType


def main():
    root = Path(__file__).resolve().parents[1]
    # This bootstrap must itself bypass stale bytecode before it can verify the
    # rest of the update. Do not use a default spec loader or delete old caches.
    path = root / "scripts/mac_update_start.py"
    if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
            or not 0 < path.stat().st_size <= 1048576):
        raise ValueError("Invalid update launcher")
    launcher = ModuleType("myvote_update_start")
    launcher.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), launcher.__dict__)
    manifest = launcher.verify_update(root)
    launcher.load_bootstrap(root, manifest=manifest).validate_runtime()
    launcher.install_engine_importer(root, manifest=manifest)
    from myvote_engine.gateway_server import main as run_gateway
    return run_gateway()


if __name__ == "__main__":
    raise SystemExit(main())
