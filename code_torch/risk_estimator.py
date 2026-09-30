"""Source-prepared estimator of *paired* test-time adaptation gain.

The estimator is deliberately separate from the rollback state machine.  It
uses labels only before deployment, on the source validation split, to learn a
small regression artifact.  At test time it consumes source/candidate
predictions and stored source summaries; target labels are never inputs.

The estimand is the accuracy difference between a candidate adapted model and
the immutable source model on the same graph.  This paired target differs from
absolute unsupervised accuracy estimation and directly supports an
accept-or-rollback decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from _np_bridge import degree_vector
from reliability import (
    degree_group_masks,
    entropy,
    group_confidence,
    js_divergence,
    neighborhood_agreement,
    top_margin,
)


FEATURE_NAMES = (
    "confidence_change",
    "normalized_entropy_change",
    "margin_change",
    "prediction_flip_fraction",
    "mean_prediction_js",
    "class_histogram_tv",
    "degree_group_confidence_drift",
    "neighborhood_agreement_change",
    "pseudo_homophily_change",
    "log_mean_degree_ratio",
    "log_feature_rms_ratio",
    "source_confidence_shift",
    "source_normalized_entropy_shift",
    "atc_estimated_gain",
    "adapter_is_confidence_weighted",
)


def _mean_degree(adj) -> float:
    return float(np.mean(degree_vector(adj)))


def _feature_rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=float)))))


def _class_histogram(prediction: np.ndarray, num_classes: int) -> np.ndarray:
    return np.bincount(prediction, minlength=num_classes).astype(float) / max(len(prediction), 1)


def _atc_threshold(confidence: np.ndarray, correctness: np.ndarray) -> float:
    """Return the confidence threshold whose exceedance rate matches accuracy."""
    accuracy = float(np.mean(correctness))
    return float(np.quantile(confidence, np.clip(1.0 - accuracy, 0.0, 1.0)))


@dataclass(frozen=True)
class SourceRiskReference:
    mean_degree: float
    feature_rms: float
    mean_confidence: float
    mean_normalized_entropy: float
    atc_threshold: float

    @classmethod
    def from_predictions(cls, base, source_probs: np.ndarray) -> "SourceRiskReference":
        val = np.asarray(base.val_idx, dtype=int)
        source_pred = np.argmax(source_probs[val], axis=1)
        source_conf = np.max(source_probs[val], axis=1)
        correctness = source_pred == np.asarray(base.y_np)[val]
        return cls(
            mean_degree=_mean_degree(base.adj),
            feature_rms=_feature_rms(base.x_np),
            mean_confidence=float(np.mean(np.max(source_probs, axis=1))),
            mean_normalized_entropy=float(
                np.mean(entropy(source_probs) / np.log(source_probs.shape[1]))
            ),
            atc_threshold=_atc_threshold(source_conf, correctness),
        )


def paired_risk_features(
    reference: SourceRiskReference,
    target,
    source_probs: np.ndarray,
    candidate_probs: np.ndarray,
    *,
    confidence_weighted: bool,
) -> np.ndarray:
    """Compute label-free paired features for one candidate checkpoint."""
    source_probs = np.asarray(source_probs, dtype=float)
    candidate_probs = np.asarray(candidate_probs, dtype=float)
    num_classes = source_probs.shape[1]
    source_pred = np.argmax(source_probs, axis=1)
    candidate_pred = np.argmax(candidate_probs, axis=1)
    source_conf = np.max(source_probs, axis=1)
    candidate_conf = np.max(candidate_probs, axis=1)
    source_ent = entropy(source_probs) / np.log(num_classes)
    candidate_ent = entropy(candidate_probs) / np.log(num_classes)

    groups = degree_group_masks(target.adj)
    source_group_conf = group_confidence(target.adj, source_probs, groups=groups)
    candidate_group_conf = group_confidence(target.adj, candidate_probs, groups=groups)
    group_drift = np.mean(
        [abs(candidate_group_conf[name] - source_group_conf[name]) for name in groups]
    )

    source_agreement, source_h = neighborhood_agreement(target.adj, source_probs)
    candidate_agreement, candidate_h = neighborhood_agreement(target.adj, candidate_probs)
    source_hist = _class_histogram(source_pred, num_classes)
    candidate_hist = _class_histogram(candidate_pred, num_classes)
    atc_source = float(np.mean(source_conf >= reference.atc_threshold))
    atc_candidate = float(np.mean(candidate_conf >= reference.atc_threshold))

    eps = 1e-12
    values = (
        float(np.mean(candidate_conf - source_conf)),
        float(np.mean(candidate_ent - source_ent)),
        float(np.mean(top_margin(candidate_probs) - top_margin(source_probs))),
        float(np.mean(candidate_pred != source_pred)),
        float(np.mean(js_divergence(source_probs, candidate_probs))),
        float(0.5 * np.abs(candidate_hist - source_hist).sum()),
        float(group_drift),
        float(np.mean(candidate_agreement) - np.mean(source_agreement)),
        float(candidate_h - source_h),
        float(np.log((_mean_degree(target.adj) + eps) / (reference.mean_degree + eps))),
        float(np.log((_feature_rms(target.x_np) + eps) / (reference.feature_rms + eps))),
        float(np.mean(source_conf) - reference.mean_confidence),
        float(np.mean(source_ent) - reference.mean_normalized_entropy),
        float(atc_candidate - atc_source),
        float(bool(confidence_weighted)),
    )
    return np.asarray(values, dtype=float)


@dataclass
class PairedGainEstimator:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    lower_residual_quantile: float
    ridge_alpha: float
    empirical_residual_alpha: float

    def predict(self, features: Sequence[float]) -> float:
        x = (np.asarray(features, dtype=float) - self.feature_mean) / self.feature_scale
        return float(self.intercept + x @ self.coefficients)

    def lower_bound(self, features: Sequence[float]) -> float:
        return self.predict(features) + self.lower_residual_quantile

    def to_dict(self) -> dict:
        return {
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "lower_residual_quantile": self.lower_residual_quantile,
            "ridge_alpha": self.ridge_alpha,
            "empirical_residual_alpha": self.empirical_residual_alpha,
            "interval_status": (
                "empirical group-held-out residual bound; no finite-sample "
                "conformal coverage guarantee"
            ),
        }


def _fit_standardized_ridge(x: np.ndarray, y: np.ndarray, ridge_alpha: float):
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (x - mean) / scale
    y_mean = float(y.mean())
    centered = y - y_mean
    gram = z.T @ z + float(ridge_alpha) * np.eye(z.shape[1])
    coef = np.linalg.solve(gram, z.T @ centered)
    return mean, scale, coef, y_mean


def fit_paired_gain_estimator(
    features: Iterable[Sequence[float]],
    gains: Sequence[float],
    groups: Sequence[str],
    *,
    ridge_alpha: float = 3.0,
    empirical_residual_alpha: float = 0.10,
) -> PairedGainEstimator:
    """Fit ridge regression and an empirical group-held-out residual bound.

    Rows sharing a corruption condition (the two adapter families) stay in the
    same fold.  The residual quantile is therefore based on predictions for
    conditions excluded from fitting, avoiding an in-sample safety bound.
    This is a development diagnostic, not a split-conformal, CV+, or jackknife+
    construction, and it has no finite-sample coverage guarantee.
    """
    x = np.asarray(list(features), dtype=float)
    y = np.asarray(gains, dtype=float)
    group_array = np.asarray(groups)
    if x.ndim != 2 or x.shape[0] != len(y) or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Feature matrix, gains, or feature schema mismatch")
    unique_groups = np.unique(group_array)
    if len(unique_groups) < 3:
        raise ValueError("At least three calibration condition groups are required")

    heldout_predictions = np.empty_like(y)
    for group in unique_groups:
        test = group_array == group
        train = ~test
        mean, scale, coef, intercept = _fit_standardized_ridge(x[train], y[train], ridge_alpha)
        heldout_predictions[test] = intercept + ((x[test] - mean) / scale) @ coef
    residuals = y - heldout_predictions
    lower_q = float(
        np.quantile(residuals, empirical_residual_alpha, method="lower")
    )

    mean, scale, coef, intercept = _fit_standardized_ridge(x, y, ridge_alpha)
    return PairedGainEstimator(
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coef,
        intercept=intercept,
        lower_residual_quantile=lower_q,
        ridge_alpha=float(ridge_alpha),
        empirical_residual_alpha=float(empirical_residual_alpha),
    )
