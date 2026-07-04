"""Bridge to the dependency-light NumPy modules in ``../code``.

The data loaders, evaluation metrics, and adjacency helpers in ``code/`` already
reproduce the published source-only numbers and define the exact metric
conventions used in the paper (accuracy / macro-F1 / NLL / ECE / Brier).  Rather
than re-deriving splits or metrics, the PyG migration imports them here so the
two implementations stay apples-to-apples.

Only ``code/utils.py`` and ``code/data.py`` are imported (neither pulls in the
NumPy ``adaptation``/``detector`` modules), so adding ``code/`` to ``sys.path``
cannot shadow the PyG package's own ``adaptation.py`` / ``detector.py``.  The
reliability and detector *logic* is ported verbatim into this package (see
``reliability.py`` / ``detector.py``) to keep it byte-identical while avoiding an
import-name collision.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    # Append (not insert) so the PyG package's own modules keep priority.
    sys.path.append(str(CODE_DIR))

import data as np_data  # noqa: E402  (code/data.py)
import utils as np_utils  # noqa: E402  (code/utils.py)
import webkb_loader as np_webkb  # noqa: E402  (code/webkb_loader.py — real Geom-GCN WebKB)

# --- metrics (paper-exact definitions) ---
evaluate = np_utils.evaluate
expected_calibration_error = np_utils.expected_calibration_error

# --- adjacency helpers (NumPy-adj space, shared with reliability/detector) ---
degree_vector = np_utils.degree_vector
is_sparse_matrix = np_utils.is_sparse_matrix
upper_triangle_edges = np_utils.upper_triangle_edges
rebuild_adjacency = np_utils.rebuild_adjacency
normalize_adjacency = np_utils.normalize_adjacency

# --- data loaders / shifts / splits ---
make_contextual_sbm = np_data.make_contextual_sbm
make_heterophily_benchmark = np_data.make_heterophily_benchmark
load_public_graph_dataset = np_data.load_public_graph_dataset
make_arxiv_subset = np_data.make_arxiv_subset
apply_shift = np_data.apply_shift
split_indices = np_data.split_indices

# --- real WebKB (Geom-GCN) ---
load_real_webkb = np_webkb.load_real_webkb
graph_homophily = np_webkb.graph_homophily
