"""One verified engine import shared by independent test modules in discovery."""

from pathlib import Path
import runpy
import sys


def load_verified_engine():
    root = Path(__file__).resolve().parents[1]
    update = root / "MyVote-Speaker-Update"
    support = runpy.run_path(str(root / "scripts/start_update_with_compat.py"))
    launcher, manifest, _ = support["load_update"](update)
    if "myvote_engine" not in sys.modules:
        launcher.install_engine_importer(update, manifest=manifest)
    else:
        engine_root = update / "src/myvote_engine"
        for name, module in tuple(sys.modules.items()):
            if name == "myvote_engine" or name.startswith("myvote_engine."):
                filename = getattr(module, "__file__", None)
                if filename is None or not Path(filename).resolve().is_relative_to(engine_root):
                    raise ValueError("Tests cannot reuse an engine imported from outside the verified tree")
    # Install in the same order as the gateway, before any test can import the
    # dedicated provider and freeze its base class. Both installers are idempotent.
    from myvote_engine import translation
    compat = runpy.run_path(str(root / "scripts/gemma_json_compat.py"))
    compat["install_context_review_schema"](translation)
    from myvote_engine import orchestrated_translation
    compat["install_selection_schema"](orchestrated_translation)
    return manifest
