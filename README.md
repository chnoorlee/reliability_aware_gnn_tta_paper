# Reliability Auditing for Graph Neural Network Test-Time Adaptation

Code and evidence for **Reliability Auditing for Graph Neural Network Test-Time Adaptation: Identifiability, Risk-Coverage, and Rollback**.

This repository studies the decision made after a graph test-time adaptation (TTA) candidate has been produced: deploy the candidate or retain an immutable source model. It implements source-relative label-free gates, transactional rollback, held-out risk-coverage analyses, a scope-matched replay, and an independent replay of configurations from the official TSA repository.

The evidence supports an audit protocol, not a distribution-free safety guarantee. A gate can prevent substantial aggregate downside while still admitting harmful candidates or collapsing toward zero coverage.

## Main findings

- In 216 held-out replays across six graphs and two entropy adapters, the frozen gate prevented **87.3%** of material downside for weighted entropy and **77.5%** for uniform entropy.
- The same gate still admitted **20 of 62** and **30 of 62** materially harmful candidates, respectively. Mean deployed gains were **-0.56** and **-1.04 percentage points**.
- Fixed-threshold decisions computed over all target nodes and benchmark evaluation nodes disagreed in **12 of 216** same-trajectory events, so proxy scope is part of the deployment claim.
- In 72 official-configuration attempts, T3A and TSA-T3A deployed no candidates. Matcha-T3A deployed 1 of 24 attempts and produced 5 non-finite candidates that failed closed.
- Prediction turnover had the strongest pooled harm-ranking point estimates, but its within-environment ordering weakened. It is a diagnostic lead, not a universal directional risk score.

See [EXPERIMENTAL_FINDINGS.md](EXPERIMENTAL_FINDINGS.md) for the denominators, operating points, uncertainty interpretation, and evidence boundaries.

## Repository layout

| Path | Purpose |
|---|---|
| `code_torch/` | Current PyTorch/PyG controllers, audit runners, analyses, and tests |
| `code/` | Shared data utilities and the earlier NumPy reference implementation |
| `results/`, `results_torch/` | Earlier benchmark and migration results retained for provenance |
| `ARTIFACT_README.md` | Commands and boundaries for the frozen audit artifact |
| `submission_artifact_manifest.json` | Repository-relative SHA-256 inventory for the artifact |
| `reliability_audit_artifact_v1.zip` | Deterministic package containing current code, frozen audit inputs, derived analyses, tests, and manuscript sources |

The authoritative evidence for the current manuscript is the audit artifact and its manifest. Earlier benchmark outputs in `results/` and `results_torch/` are not substitutes for the frozen audit inputs identified in `ARTIFACT_README.md`.

## Verify the artifact

From the repository root:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
python code_torch/build_submission_artifact.py --verify-only
```

Expected archive SHA-256:

```text
861046f83c7c6f3494750189338e6b7756ebaa503df994684047c0d9a1ba9eeb
```

The verifier checks the current files, external and embedded manifests, canonical ZIP metadata, CRCs, sizes, hashes, and deterministic archive bytes.

## Run the validated analysis suite

Create an environment compatible with `requirements.txt`, add the QA dependencies in `requirements-dev.txt`, then run:

```powershell
Push-Location code_torch
python -m pytest tests -q
Pop-Location
```

The dependency versions record the intended audit environment. `environment_snapshot.json` records the imports exercised during final QA. The requirements file has not been established as a portable clean-environment lock because the source host contained conflicting PyTorch distribution metadata.

To test exactly the code distributed in the artifact, extract `reliability_audit_artifact_v1.zip`, enter its root, and run:

```powershell
Push-Location code_torch
python -m pytest tests -q
Pop-Location
```

## Reproduce the reported analyses

These commands consume frozen raw results. They recompute reported analyses and do not rerun adaptation or retune the gate.

```powershell
python code_torch/stage4_reanalysis.py `
  --heldout revision_2026-08-19/stage4_fresh_full/heldout_endpoint_final_current/results.json `
  --official revision_2026-08-19/official_tsa_default_label_isolated_audit_v2/results.json `
  --out-dir revision_2026-08-19/stage4_fresh_full/reanalysis_current_final_10k_1 `
  --bootstrap-replicates 10000

python code_torch/stage5_selective_audit.py `
  --heldout revision_2026-08-19/stage4_fresh_full/heldout_endpoint_final_current/results.json `
  --official revision_2026-08-19/official_tsa_default_label_isolated_audit_v2/results.json `
  --out-dir revision_2026-09-02_stage5_redesign/selective_audit `
  --bootstrap-replicates 10000

python code_torch/scope_replay_analysis.py `
  --original revision_2026-08-19/stage4_fresh_full/heldout_endpoint_final_current/results.json `
  --replay revision_2026-09-02_stage5_redesign/scope_matched_heldout_v2/results.json `
  --out-dir revision_2026-09-02_stage5_redesign/scope_replay_analysis
```

These commands overwrite derived outputs. Run them in an extracted copy when the distributed artifact hashes must remain unchanged.

## Third-party audit dependency

The independent official audit uses `https://github.com/Graph-COM/TSA.git` at commit `cddde28bedcdaf0da68f8c7d9f6c9c45fe9e67b6`. That checkout and its data are not redistributed here. The local wrapper is `code_torch/official_tsa_risk_audit.py`; upstream licenses and terms continue to apply.

## Evidence boundary

The controlled audit contains six graph identities and three seeds. Its graph-block and hierarchical intervals are composition-sensitivity analyses, not population guarantees. The official replay is a same-backbone diagnostic rather than a ranking of published methods. Numerical failures, zero-coverage results, and negative results are retained in the artifact.
