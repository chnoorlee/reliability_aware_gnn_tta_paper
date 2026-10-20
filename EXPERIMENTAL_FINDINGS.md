# Experimental Findings

This note summarizes the frozen results used by the current manuscript. Values below come from the raw results and derived analyses listed in `submission_artifact_manifest.json`. The full interpretation, protocol, and limitations remain in the manuscript sources distributed in `reliability_audit_artifact_v1.zip`.

## Held-out endpoint replay

The held-out matrix contains 108 paired runs per adapter across six graphs. A materially harmful candidate has source-relative accuracy gain below -1 percentage point (pp).

| Adapter | Coverage | Harm recall | Accepted harmful / all harmful | Accepted-set harm risk | Retained material utility | Prevented material downside | Mean deployed gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Weighted entropy | 48.1% | 67.7% | 20 / 62 | 38.5% | 46.1% | 87.3% | -0.56 pp |
| Uniform entropy | 59.3% | 51.6% | 30 / 62 | 46.9% | 68.1% | 77.5% | -1.04 pp |

The graph-block composition-sensitivity intervals for mean deployed gain were `[-1.16, -0.09]` pp and `[-1.71, -0.39]` pp. Hierarchical graph-then-seed intervals were `[-1.42, 0.18]` pp and `[-2.15, -0.04]` pp. These are sensitivity analyses over the six observed graph identities, not confidence intervals for an unspecified population of future graphs.

The gate prevented much of the aggregate downside, but it neither removed all material harm nor preserved all available benefit. The weighted and uniform policies discarded 53.9% and 31.9% of material positive utility, respectively.

## Scope-matched replay

The same stored trajectories and the same thresholds were evaluated on different node scopes without changing the operational decision.

| Adapter | All-node accept | Evaluation-node accept | Decision agreement | All accept / evaluation reject | All reject / evaluation accept |
|---|---:|---:|---:|---:|---:|
| Weighted entropy | 52 / 108 | 49 / 108 | 101 / 108 | 5 | 2 |
| Uniform entropy | 64 / 108 | 61 / 108 | 103 / 108 | 4 | 1 |
| Combined | 116 / 216 | 110 / 216 | 204 / 216 | 9 | 3 |

The 12 disagreements show that an all-node proxy and an evaluation-subset proxy are not interchangeable, even when aggregate agreement is high. The evaluation-node rule was a diagnostic counterfactual; it was not deployed or retuned.

## Matched-coverage score diagnostic

At the frozen gate's observed acceptance counts (`K=52` for weighted entropy and `K=64` for uniform entropy), ranking by confidence drift alone had better point estimates than the combined maximum score for accepted-set harm, retained utility, and mean deployed gain. The paired graph-block intervals for the gain difference between confidence drift and the combined score crossed zero for both adapters. This falsifies empirical dominance of the handcrafted maximum on the audited matrix but does not validate a replacement score.

## Stateful controller audit

The direct two-strike controller audit covered 216 events and 53,063 proposed updates. It rejected 199 updates and returned the immutable source classifier in 99 events.

| Adapter | Events | Source halted | Final harmful models | Mean / worst deployed gain |
|---|---:|---:|---:|---:|
| Weighted entropy | 108 | 56 | 20 | -0.56 / -9.82 pp |
| Uniform entropy | 108 | 43 | 31 | -1.11 / -12.18 pp |

The controller enforces rollback when its software condition fires. It does not establish that every accepted model is safe.

## Official-repository endpoint audit

The official replay used the bundled code at TSA commit `cddde28bedcdaf0da68f8c7d9f6c9c45fe9e67b6` and contained 24 attempts per method.

| Method | Finite / attempts | Numerical failures | Coverage | Harm recall among finite candidates | Retained material utility among finite candidates | Mean deployed gain over all attempts |
|---|---:|---:|---:|---:|---:|---:|
| T3A | 24 / 24 | 0 | 0.0% | 100% | 0.0% | 0.00 pp |
| Matcha-T3A | 19 / 24 | 5 | 4.2% | 100% | 33.8% | 0.42 pp |
| TSA-T3A | 24 / 24 | 0 | 0.0% | 100% | 0.0% | 0.00 pp |

All materially harmful finite candidates were rejected, but two methods operated at the reject-all corner. Matcha-T3A accepted one improving candidate and failed closed on five non-finite outputs. These data do not rank the published methods because the replay is a same-backbone diagnostic rather than a separately optimized benchmark.

## Diagnostic ranking

Prediction turnover had the largest pooled harm AUROC point estimates across the internal and official audits. For the internal adapters, pooled values were 0.885 and 0.861, while within-stratum concordance fell to 0.625 and 0.458. Only 12 of 36 strata per adapter contained both outcome classes, producing 24 comparable pairs. Turnover may therefore reflect exposure, change magnitude, or environment composition as well as directional harm.

## Conclusion supported by these experiments

Unlabeled graph and prediction drift do not identify the direction of paired accuracy change without additional assumptions. A source-relative gate can be operationally useful, but it must be reported as a risk-coverage operating point with harm, benefit, scope, state transitions, and numerical failures all visible. The validated contribution is an auditable deployment protocol and rollback invariant, not a universal reliability guard or safety certificate.
