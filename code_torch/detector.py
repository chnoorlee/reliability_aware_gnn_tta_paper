"""State for the candidate-update rollback detector.

Implements the unsupervised proxy from Section 3.6:

    Delta_t = (1/3) * sum_k | Conf_theta_t(G_k) - Conf_theta_0(G_k) |
    Phi_t   = (1/N)   * sum_i 1[ argmax p_i^(t) != argmax p_i^(0) ]

Each candidate is checked after its update.  A rejected candidate is never
deployed.  One rejection retains the latest accepted checkpoint; two
consecutive rejections restore the immutable source classifier and halt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from reliability import group_confidence


@dataclass
class DetectorState:
    delta_tolerance: float = 0.05
    phi_tolerance: float = 0.20
    delta_history: List[float] = field(default_factory=list)
    phi_history: List[float] = field(default_factory=list)
    triggered: bool = False
    trigger_step: Optional[int] = None
    trigger_reason: Optional[str] = None
    strikes: int = 0
    accepted_candidates: int = 0
    rejected_candidates: int = 0
    state: str = "ADAPTING"
    decision_history: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "delta_tolerance": self.delta_tolerance,
            "phi_tolerance": self.phi_tolerance,
            "delta_history": list(self.delta_history),
            "phi_history": list(self.phi_history),
            "triggered": self.triggered,
            "trigger_step": self.trigger_step,
            "trigger_reason": self.trigger_reason,
            "strikes": self.strikes,
            "accepted_candidates": self.accepted_candidates,
            "rejected_candidates": self.rejected_candidates,
            "state": self.state,
            "decision_history": list(self.decision_history),
        }


def compute_detector_signals(adj, current_probs, source_probs, source_group_conf) -> Tuple[float, float, Dict[str, float]]:
    """Compute (Delta_t, Phi_t, current_group_conf) for one adaptation step."""
    current_group_conf = group_confidence(adj, current_probs)
    delta = float(np.mean([abs(current_group_conf[k] - source_group_conf[k]) for k in source_group_conf]))
    current_argmax = np.argmax(current_probs, axis=1)
    source_argmax = np.argmax(source_probs, axis=1)
    phi = float(np.mean(current_argmax != source_argmax))
    return delta, phi, current_group_conf


def detector_should_halt(state: DetectorState, delta: float, phi: float) -> Tuple[bool, Optional[str]]:
    if not np.isfinite(delta) or not np.isfinite(phi):
        return True, "nonfinite_proxy"
    if delta > state.delta_tolerance:
        return True, f"delta>{state.delta_tolerance:.3f}"
    if phi > state.phi_tolerance:
        return True, f"phi>{state.phi_tolerance:.3f}"
    return False, None
