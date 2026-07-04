# Reproducible Experiment Code

## Quick pilot

```bash
python code/main.py --quick
```

## Full controlled-shift run

```bash
python code/main.py
```

Outputs:

- `results/results.json`
- `results/summary.csv`

## Public-data and heterophily-analogue benchmarks

```bash
python code/main.py --datasets public_core --graph-backend auto --out results/public_benchmark
python code/main.py --datasets heterophily_core --out results/heterophily_benchmark
```

## Supplementary diagnostics (Section 5.8 of the paper)

```bash
python code/supplementary_experiments.py --experiments all
python code/aggregate_supplementary.py
```

Covers: assumption verification (Cov(r,e)), detector tolerance sensitivity,
hyperparameter robustness, subgroup-drift trajectory, reliability-signal
correlation, and the multi-backbone study (GCN / GAT / GraphSAGE / APPNP).
Outputs under `results/supplementary/`.

## External-validity experiments (Section 5.9 of the paper)

```bash
python code/extended_experiments.py --experiments all
python code/aggregate_extended.py
```

Covers:

- `large_scale_study`   — Coauthor CS, Amazon Photo (induced subgraphs, sparse backend)
- `real_webkb_study`    — real Texas / Cornell / Wisconsin (Geom-GCN release)
- `streaming_tta_study` — continual TTA over a stream of accumulating shifts
- `adversarial_study`   — targeted adversarial edge and feature attacks

Outputs under `results/extended/`.  The real WebKB files are auto-downloaded to
`data/webkb/`; the Amazon/Coauthor NPZ files to `data/public/`.

## Module map

| File | Purpose |
|------|---------|
| `data.py` | graph generation, Planetoid / Amazon / Coauthor loaders, heterophily + boundary (Actor/Film) generators, shift protocols |
| `models.py` | two-layer GCN (NumPy full-batch) |
| `models_bn.py` | GCN with a batch-norm layer (full forward/backward); needed for faithful Tent/EATA |
| `backbones.py` | GAT, GraphSAGE, APPNP backbones for the backbone-agnostic study |
| `adaptation.py` | reliability estimator, reliability-weighted entropy, calibration / anti-forgetting losses, detector hook (dual-checkpoint) |
| `detector.py` | closed-loop negative-adaptation detector (Delta_t / Phi_t) |
| `full_baselines.py` | full original mechanisms: Tent (BN), EATA (BN+Fisher+filter), Matcha (graph-aware mask), GTrans (feature transform) |
| `fair_comparison.py` | Exp 1: fair head-to-head vs full baselines on shared BN backbone |
| `detector_calibration.py` | Exp 2: threshold auto-calibration sweep + dual-checkpoint streaming |
| `webkb_assumption.py` | Exp 3: Assumption-1 + per-signal correlation on real WebKB |
| `run_boundary.py` | boundary-region (Actor/Film) experiment |
| `run_scalability.py` | scalability on arxiv-like graphs up to 5,000 nodes |
| `webkb_loader.py` | real WebKB (Texas/Cornell/Wisconsin) loader |
| `supplementary_experiments.py` | six diagnostic experiments |
| `extended_experiments.py` | four external-validity experiments |
| `significance_tests.py` | seed-matched paired tests |
| `verify_references.py` | bibliography DOI/venue audit |

## Notes

The scientific core uses NumPy only (plus SciPy for sparse propagation on the
larger graphs). All graphs use real feature/adjacency generation or real public
data, real GCN training, and real test-time adaptation losses. No experimental
number should be copied into the manuscript unless it appears in a stored result
file under `results/`.
