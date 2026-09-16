"""Run the SepFormer command from a verified side-by-side source bundle."""
from pathlib import Path
from types import ModuleType


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/mac_update_start.py"
    if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
            or not 0 < path.stat().st_size <= 1048576):
        raise ValueError("Invalid update launcher")
    launcher = ModuleType("myvote_update_start")
    launcher.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), launcher.__dict__)
    manifest = launcher.verify_update(root)
    launcher.install_engine_importer(root, manifest=manifest)
    from myvote_engine.sepformer_cli import main as run_separation
    return run_separation()


if __name__ == "__main__":
    raise SystemExit(main())
