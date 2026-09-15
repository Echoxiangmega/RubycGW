"""Small result summaries shared by CLI/notebook/UI frontends."""
from __future__ import annotations

import numpy as np

from ..solvers.effective import EffectiveEDResult, leading_mode_label


def summarize_background(result) -> dict[str, object]:
    """Return serializable convergence/thermodynamic metadata."""
    return {
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "final_error": float(result.final_error),
        "mu": float(result.mu),
        "density": np.asarray(result.density, dtype=float).copy(),
        "bath_fit_error": float(result.bath_fit_error),
        "impurity_mismatch": float(result.impurity_mismatch),
    }


def summarize_effective_ed(result: EffectiveEDResult) -> dict[str, object]:
    """Return the dominant finite-size ordering correlation and ground-state data."""
    iq = int(np.argmax(np.asarray(result.lambda_max)))
    comp, parity, weight = leading_mode_label(result.sf_eigenvectors[iq, :, 0])
    return {
        "ground_energy": float(result.ground_energy),
        "gap": float(result.gap),
        "ground_degeneracy": int(result.ground_degeneracy),
        "leading_q": np.asarray(result.q_centered[iq], dtype=float).copy(),
        "leading_lambda": float(result.lambda_max[iq]),
        "leading_component": comp,
        "leading_parity": parity,
        "leading_component_weight": float(weight),
        "z_same_at_gamma": float(result.z_same[0]),
        "xy_max": float(np.max(result.xy_max)),
    }


__all__ = ["summarize_background", "summarize_effective_ed"]
