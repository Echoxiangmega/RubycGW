"""Finite-size quantum ED for the strong-coupling pseudospin model.

The implementation currently lives in the historical ``vprime_study`` package;
this facade is the supported public import path while the implementation is
migrated incrementally.
"""

from vprime_study.effective_pseudospin_ed import (
    EffectiveEDResult,
    allowed_qmesh,
    effective_couplings,
    leading_mode_label,
    solve_effective_pseudospin_ed,
)

__all__ = [
    "EffectiveEDResult",
    "effective_couplings",
    "allowed_qmesh",
    "solve_effective_pseudospin_ed",
    "leading_mode_label",
]
