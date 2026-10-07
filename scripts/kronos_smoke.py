"""Offline-safe container smoke check (does not load or download checkpoints)."""
import argparse
import importlib
import sys
from pathlib import Path

# Running the file directly puts /app/scripts first on sys.path; add the
# repository root so the local `model` package is imported from this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Kronos container smoke check")
    parser.add_argument("--self-test", action="store_true", help="import core modules and verify their public classes")
    args = parser.parse_args()
    if not args.self_test:
        parser.print_help()
        return 0
    for module_name in ("numpy", "pandas", "torch", "huggingface_hub", "model"):
        importlib.import_module(module_name)
    from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: F401
    print("smoke ok: dependencies and model API imported; no checkpoint was loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
