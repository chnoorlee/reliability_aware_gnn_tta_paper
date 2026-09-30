"""Run the unmodified official TSA entry point under recent PyG builds.

The upstream repository imports optional ``torch_sparse`` symbols eagerly,
including for adapters that do not use them.  PyTorch/PyG nightly wheels for
Python 3.14 do not ship ``torch_sparse``.  This launcher provides only the
``coalesce`` symbol needed to import the adapter registry and a fail-fast
placeholder for ``SparseTensor``.  It does not reimplement TSA and is intended
only for adapters whose execution never constructs ``SparseTensor``.
"""

from __future__ import annotations

import runpy
import sys
import types
import argparse
from pathlib import Path

from torch_geometric.utils import coalesce


class _UnavailableSparseTensor:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "This official TSA adapter requires torch_sparse, which is not "
            "available in the compatibility environment."
        )


def install_compat(tsa_root: Path | None = None) -> Path:
    """Install import-only compatibility shims and return the TSA checkout."""
    tsa_root = tsa_root or (
        Path(__file__).resolve().parents[1] / "external_official" / "TSA"
    )
    if not tsa_root.is_dir():
        raise FileNotFoundError(f"Official TSA checkout not found: {tsa_root}")

    sparse_module = types.ModuleType("torch_sparse")
    sparse_module.coalesce = coalesce
    sparse_module.SparseTensor = _UnavailableSparseTensor
    sys.modules.setdefault("torch_sparse", sparse_module)

    # The upstream dataset registry imports the optional Pileup reader even
    # when CSBM is selected.  A module placeholder is sufficient because CSBM
    # never calls uproot; selecting Pileup still fails at its first uproot use.
    sys.modules.setdefault("uproot", types.ModuleType("uproot"))
    sys.path.insert(0, str(tsa_root))

    # Python 3.14's argparse validates help strings with ``in``. Hydra 1.3.2
    # intentionally supplies a lazy help object, which is valid in Python
    # 3.10 but raises during parser construction in 3.14. Skip validation only
    # for non-string lazy help values; ordinary argparse checks remain intact.
    original_check_help = argparse.ArgumentParser._check_help

    def _check_help_compat(self, action):
        if not isinstance(action.help, str):
            return None
        return original_check_help(self, action)

    argparse.ArgumentParser._check_help = _check_help_compat

    return tsa_root


def main() -> None:
    tsa_root = install_compat()

    runpy.run_path(str(tsa_root / "src" / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
