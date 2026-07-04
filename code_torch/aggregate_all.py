"""Run the NumPy aggregation utilities against the PyG result tree.

``code/aggregate_supplementary.py``, ``code/aggregate_extended.py`` and
``code/generate_extended_figures.py`` are pure result-consumers (they read the
JSON files and print/plot summaries).  Rather than duplicating them, this
wrapper imports the originals and repoints their module-level result paths at
``results_torch/`` before invoking them.

Usage:
    python aggregate_all.py [supplementary|extended|figures|all]
"""

from __future__ import annotations

import sys
from pathlib import Path

import _np_bridge  # noqa: F401  (adds code/ to sys.path)

ROOT = Path(__file__).resolve().parents[1]


def run_supplementary():
    import aggregate_supplementary as agg_supp
    agg_supp.SUPP = ROOT / "results_torch" / "supplementary"
    agg_supp.main()


def run_extended():
    import aggregate_extended as agg_ext
    agg_ext.EXT = ROOT / "results_torch" / "extended"
    agg_ext.main()


def run_extended_figures():
    import generate_extended_figures as ext_figs
    ext_figs.EXT = ROOT / "results_torch" / "extended"
    ext_figs.adversarial_figure()
    ext_figs.streaming_figure()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("supplementary", "all"):
        run_supplementary()
    if what in ("extended", "all"):
        run_extended()
    if what in ("figures", "all"):
        run_extended_figures()
