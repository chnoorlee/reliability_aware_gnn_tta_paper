# `code_torch/` — PyTorch Geometric migration

PyTorch Geometric reimplementation of the reliability-aware graph test-time
adaptation framework. It replaces the dependency-light NumPy code in `../code`
with official PyG layers and official (vendored) Tent/EATA baselines, while
keeping the paper's reliability estimator, calibration regularizer, and
negative-adaptation detector **unchanged**.

The original NumPy framework in `../code` is left intact as a reference/fallback.

## Why two backbones (important design point)

BatchNorm in a shallow GCN measurably *hurts* source accuracy on Citeseer/Pubmed
(a known effect), so it cannot reproduce the published source-only numbers. The
official Tent/EATA, however, *require* BatchNorm to adapt. We therefore mirror
the original paper's two-experiment design:

| experiment | backbone | who runs here |
|---|---|---|
| main benchmark (`main.py`) | **no-BN GCN** | source-only, proposed method, ablations — reproduces Cora/Citeseer/Pubmed ~ 0.81/0.71/0.79 |
| fair comparison (`fair_comparison.py`) | **BN GCN** | official **Tent/EATA** (adapt BN), Matcha, GTrans, proposed — all share one BN source model |

The proposed method is classifier-only on *both* backbones (it never touches BN),
preserving the closed-form drift bound.

## Files

| file | role |
|---|---|
| `models.py` | PyG `GCN` / `GAT` / `GraphSAGE` / `APPNPNet` (optional `BatchNorm1d`) + `train_model` |
| `data_adapter.py` | NumPy graph -> PyG `Data`; `GraphBundle` holds both NumPy-adj and torch tensors |
| `_np_bridge.py` | imports loaders + metrics from `../code` (reuse, not reimplement) |
| `reliability.py` | reliability estimator — **verbatim port** of `../code/adaptation.py` scoring |
| `detector.py` | negative-adaptation detector — **verbatim port** of `../code/detector.py` |
| `adaptation.py` | proposed classifier-only TTA + ablations (autograd for differentiable terms) |
| `baselines_official.py` | `GraphModelWrapper`, Fisher, `run_tent/eata/matcha/gtrans` |
| `third_party/` | vendored official `tent.py` / `eata.py` (MIT) + `ATTRIBUTION.md` |
| `main.py` | experiment runner -> `../results_torch/` |
| `fair_comparison.py` | official-baseline comparison -> `../results_torch/fair_comparison/` |
| `compare_numpy_torch.py` | generates `../MIGRATION_REPORT.md` |

## Run

```bash
# from code_torch/  (CPU full-batch; OGB uses the local GPU)
python main.py --dataset synthetic                       # main benchmark -> results_torch/
python main.py --datasets public_core --out ../results_torch/public_benchmark
python main.py --datasets heterophily_core --out ../results_torch/heterophily_benchmark
python fair_comparison.py                                # official Tent/EATA + auto-detector
python supplementary_experiments.py                      # 6 diagnostics -> results_torch/supplementary/
python extended_experiments.py                           # webkb/adversarial/streaming/large-scale
python run_scalability.py && python run_boundary.py
python detector_calibration.py && python webkb_assumption.py
python run_detector_main.py                              # tab:main detector rows (fixed vs auto)
python ogb_experiments.py --dataset ogbn-arxiv --seeds 0,1   # OGB minibatch TTA (GPU)
python ../code/significance_tests.py --results ../results_torch/results.json --out ../results_torch/significance.csv
python aggregate_all.py                                  # summaries + extended figures
python render_paper_numbers.py                           # LaTeX rows for every paper table
python compare_numpy_torch.py                            # NumPy vs PyG report
```

`render_paper_numbers.py` supersedes `code/update_paper_tables.py` for the
hand-maintained `sections/results.tex`: it renders every table body from
`results_torch/` so values can be swapped in place without regenerating the
(heavily hand-edited) section file.

## Validation gates (all passing)

- **req#4** source-only: Cora 0.814 / Citeseer 0.711 / Pubmed 0.798 (targets 0.805/0.710/0.794).
- **req#5** GAT synthetic clean: **0.98** (official `GATConv`) vs **0.62** (NumPy hand-written attention grad).
- **req#2/#3** Tent collects/updates `['model.bn.weight','model.bn.bias']` (BatchNorm), not the classifier.
- **req#6** reliability / calibration / detector logic ported verbatim.

## Fidelity note (req#6)

The proposed update mirrors `../code/adaptation.py::adapt_classifier` exactly:
classifier-only manual step (lr=0.05, grad-norm clip 2.0), the same reliability
weights / quantile masks / convergence test / detector rollback. Differentiable
terms (reliability-weighted entropy, anti-forgetting L2) use real autograd; the
non-differentiable calibration / graph-consistency terms use the same shrinkage
proxy as the NumPy code (they depend on argmax + quantile bins).

## Environment

Python 3.14 + torch 2.12 nightly + torch_geometric 2.8 (CPU). See `../requirements.txt`.
Any torch >= 2.4 with a matching PyG works; the code is version-agnostic.
