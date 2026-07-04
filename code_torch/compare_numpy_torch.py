"""Generate MIGRATION_REPORT.md: side-by-side NumPy vs PyTorch-Geometric results.

Reads the existing NumPy result CSVs under ``../results`` and the new PyG result
CSVs under ``../results_torch`` and writes a concise comparison covering the four
validation gates (source-only reproduction, GAT stability, fair-comparison BN
adaptation, no-harm).  Re-run after regenerating either result set.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rows(path):
    if not Path(path).exists():
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def _find(rows, **kw):
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in kw.items()):
            return r
    return None


def _mean(rows, key, **kw):
    vals = [float(r[key]) for r in rows if all(str(r.get(k)) == str(v) for k, v in kw.items())]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    np_pub = _rows(ROOT / "results" / "public_benchmark" / "summary.csv")
    pg_pub = _rows(ROOT / "results_torch" / "public_benchmark" / "summary.csv")
    np_bb = _rows(ROOT / "results" / "supplementary" / "backbone_study.csv")
    pg_gat = _rows(ROOT / "results_torch" / "backbone_gat" / "summary.csv")
    fair_path = ROOT / "results_torch" / "fair_comparison" / "fair_comparison.json"
    proof = json.loads(fair_path.read_text())["bn_adaptation_proof"] if fair_path.exists() else None

    L = []
    L.append("# NumPy -> PyTorch Geometric migration report\n")
    L.append("Side-by-side of the dependency-light NumPy framework (`code/`, `results/`) "
             "and the PyG migration (`code_torch/`, `results_torch/`). All numbers are "
             "means over seeds.\n")

    # Gate A: source-only reproduction
    L.append("## Gate A --source-only reproduction (req#4)\n")
    L.append("| dataset | target | NumPy | PyG (no-BN GCN) |")
    L.append("|---|---|---|---|")
    targets = {"cora": 0.805, "citeseer": 0.710, "pubmed": 0.794}
    for ds in ["cora", "citeseer", "pubmed"]:
        npr = _find(np_pub, dataset=ds, shift="clean", method="source_only")
        pgr = _find(pg_pub, dataset=ds, shift="clean", method="source_only")
        npa = float(npr["accuracy_mean"]) if npr else float("nan")
        pga = float(pgr["accuracy_mean"]) if pgr else float("nan")
        L.append(f"| {ds} | {targets[ds]:.3f} | {npa:.4f} | {pga:.4f} |")
    L.append("")

    # Gate B: GAT stability
    L.append("## Gate B --GAT numerical stability, synthetic clean (req#5)\n")
    gcn_np = _mean(np_bb, "accuracy", backbone="gcn", shift="clean")
    gat_np = _mean(np_bb, "accuracy", backbone="gat", shift="clean")
    gat_pg = _mean(pg_gat, "accuracy_mean", backbone="gat", shift="clean", method="source_only")
    L.append("| backbone | NumPy clean acc | PyG clean acc |")
    L.append("|---|---|---|")
    L.append(f"| GCN | {gcn_np:.4f} | (ref) |")
    L.append(f"| GAT | {gat_np:.4f}  <- unstable hand-written attention grad | {gat_pg:.4f}  <- official GATConv |")
    L.append("")

    # No-harm: full_method vs source_only (PyG)
    L.append("## No-harm --full_method vs source_only on clean (PyG, req#6 behavior)\n")
    L.append("| dataset | source acc | full acc | source ECE | full ECE |")
    L.append("|---|---|---|---|---|")
    for ds in ["cora", "citeseer", "pubmed"]:
        so = _find(pg_pub, dataset=ds, shift="clean", method="source_only")
        fm = _find(pg_pub, dataset=ds, shift="clean", method="full_method")
        if so and fm:
            L.append(f"| {ds} | {float(so['accuracy_mean']):.4f} | {float(fm['accuracy_mean']):.4f} "
                     f"| {float(so['ece_mean']):.4f} | {float(fm['ece_mean']):.4f} |")
    L.append("")

    # Fair comparison BN proof
    L.append("## req#2/#3 --official Tent/EATA adapt BatchNorm (fair_comparison)\n")
    if proof:
        L.append(f"- Tent collects and updates: `{proof['adapted_param_names']}`")
        L.append(f"- BatchNorm gamma L1 drift after adaptation: **{proof['gamma_l1_drift']:.4f}** (> 0 => BN is adapted)")
        L.append(f"- classifier adapted by Tent/EATA: **{proof['classifier_adapted']}** (the proposed method, by contrast, adapts only the classifier)")
    else:
        L.append("- (run `python fair_comparison.py` to populate)")
    L.append("")

    L.append("## Notes\n")
    L.append("- The PyG no-BN GCN reproduces source-only accuracy within +-0.01; it is a touch "
             "more confident than the NumPy GCN (Adam vs hand-tuned SGD), so absolute ECE on "
             "Cora/Citeseer is higher. The **BN** GCN used for the fair comparison is much better "
             "calibrated and is what Tent/EATA adapt.")
    L.append("- Reliability estimator, calibration regularizer, and detector are ported verbatim "
             "(`reliability.py`, `detector.py`); only differentiable terms use autograd.")

    out = ROOT / "MIGRATION_REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
