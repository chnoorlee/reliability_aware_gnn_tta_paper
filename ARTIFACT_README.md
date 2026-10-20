# Reliability-Audit Reproducibility Artifact

This artifact supports the manuscript *Reliability Auditing for Graph Neural
Network Test-Time Adaptation: Identifiability, Risk--Coverage, and Rollback*.
It is an audit record, not evidence for a distribution-free safety guarantee.
The reported controlled-study intervals are six-graph sensitivity analyses,
and the official TSA replay is descriptive.

## Authoritative evidence

Only the following raw result files are final inputs to the reported analyses.
All paths are relative to the repository root.

| Role | Path | Records | SHA-256 |
|---|---|---:|---|
| Held-out endpoint replay | `revision_2026-08-19/stage4_fresh_full/heldout_endpoint_final_current/results.json` | 216 | `aa2a4cf290b03d279f8fb081ba268dff67bd490fb1943d8f8546dcbbe11b9ca4` |
| Stateful controller audit | `revision_2026-08-19/stage4_fresh_full/controller_audit_final_current_v3/results.json` | 216 | `ee68c88348446c6b544681847752fbf7d28839c92df08751009ba46b6b29cfd6` |
| Official TSA audit | `revision_2026-08-19/official_tsa_default_label_isolated_audit_v2/results.json` | 72 | `dbda336d23d379cb00cb6fee877cc8aea1bc94a5eda4dff2a53176c48f20409f` |
| Scope-matched diagnostic replay | `revision_2026-09-02_stage5_redesign/scope_matched_heldout_v2/results.json` | 216 | `6561cfd7427b7a7ce9a8822df1e5af5e1cd8399f5e8e318fd185845f03f28942` |

The package also contains the final derived tables, validation reports,
analysis code, manuscript sources, built PDFs, and exact historical source
snapshots used by the controlled replays. The top-level
`submission_artifact_manifest.json` is the authoritative inventory. It uses
repository-relative paths. The recovery provenance identifies source session
logs only by their original filenames; those local logs are not distributed.

## Verify the package

With the archive, manifest, and repository tree side by side at the repository
root, run the strict integrity verifier:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
python code_torch/build_submission_artifact.py --verify-only
```

The verifier rejects links/reparse points, non-canonical paths, unexpected or
duplicate manifest fields, an altered current file, and every ZIP inventory,
metadata, CRC, size, hash, or byte mismatch. It reconstructs the deterministic
ZIP and compares the complete archive bytes, and it requires the embedded and
external manifests to be byte-identical.

To test only the code actually distributed in the archive, extract the ZIP to
an empty directory and run this command from that extracted directory:

```powershell
Push-Location code_torch
python -m pytest tests -q
Pop-Location
```

This is the self-contained release suite: 10 test modules and 93 tests in the
validated source environment. The repository's broader development suite also
contains tests for the artifact builder and for excluded prospective or
exploratory modules. Those tests are deliberately absent from the archive. The
test count is evidence for the packaged scope, not a claim that a fresh
dependency installation has been validated.

The archive additionally carries
`revision_2026-08-19/heldout_risk_coverage_v3/results.json` only as a
hash-pinned negative-test fixture for the held-out schema validator. Its
manifest role is `test_fixture_not_evidence`; it is not an authoritative input
to a reported analysis.

To regenerate the manifest and deterministic archive after an intentional
source or manuscript update:

```powershell
python code_torch/build_submission_artifact.py
```

The builder fails if a cornerstone frozen result, its record count, or a
recovered result-generating source hash differs from its pinned value. The
recovered-source directory has a closed allowlist containing only
`RECOVERY_PROVENANCE.md` and four hash-pinned Python files. The builder never
rewrites a frozen result.

## Reproduce the reported analyses

The commands below consume the frozen raw results; they do not rerun adaptation
or retune a gate.

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

These commands intentionally overwrite derived outputs when rerun. Run them in
a copy of the artifact if the distributed byte hashes must be preserved.

## Raw replay provenance

The controlled result JSON files store their complete invocations, runtime
metadata, and per-module SHA-256 map. Exact historical modules are under
`revision_2026-09-02_stage5_redesign/recovered_source_snapshots/`; current
modules are used only where their hashes match the frozen record. Historical
modules were mechanically reconstructed and are supplied for auditability.
Reproducing the raw replays also requires downloading the public benchmark data
through PyTorch Geometric. Dataset licenses and terms remain those of the
upstream providers.

The independent official audit requires a separate checkout of
`https://github.com/Graph-COM/TSA.git` at commit
`cddde28bedcdaf0da68f8c7d9f6c9c45fe9e67b6` and the upstream data/configuration
files. The local wrapper is `code_torch/official_tsa_risk_audit.py`; the
third-party checkout is not redistributed in the compact archive, and its
license continues to apply.

## Environment status

`requirements.txt` records the intended direct dependency pins and
`environment_snapshot.json` records the imports actually exercised during
final QA. The source machine contains conflicting PyTorch distribution metadata
even though the imported runtime identifies itself as
`2.12.0.dev20260408+cu128`. Consequently this is an exact runtime snapshot, not
a clean-environment lock verification. A clean install must be tested before
claiming that the requirements file is a portable lock.

## Explicit exclusions

The archive excludes development scans, smoke runs, aborted/concurrently
corrupted runs, `scope_matched_heldout/`, `scope_smoke*`, the legacy incomplete
`source_snapshot/`, caches, downloaded data, checkpoints, LaTeX auxiliaries,
and rendered QA pages. It also excludes `code_torch/risk_estimator.py` and
`code_torch/topology_anchor.py`, together with their paired tests. These are
exploratory diagnostics, were not used to produce any manuscript result, and
are not validated safety certificates.
