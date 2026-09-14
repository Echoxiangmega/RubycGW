"""Scale-invariant Pulay/DIIS helpers for small-residual fixed points.

The historical Pulay implementation regularizes the residual Gram matrix with

    reg * max(max(diag(B)), 1)

which introduces an absolute floor.  Once the mean-square residual falls below
that floor, the regularizer can dominate the actual Gram matrix and DIIS loses
most of its acceleration.  This is especially visible in the cluster-ED+GW
outer loop, where residuals can drop rapidly to ~1e-3 and then decay almost
linearly.

The implementation below first normalizes the residual Gram block by its own
scale and then applies a dimensionless regularizer.  Therefore the Pulay
coefficients are invariant under an overall rescaling of all residuals.

The package initializer installs this implementation into :mod:`rubycgw.gw` by
default.  Consequently every solver that uses ``gw._mixed_self_energies`` gets
the corrected Pulay behavior, including standalone validation/source scripts;
no special launcher is required.  ``install_scale_invariant_pulay`` remains
public as an idempotent compatibility helper for older launchers.
"""
from __future__ import annotations

import numpy as np


def scale_invariant_pulay_coefficients(history, regularization: float) -> np.ndarray:
    """Return DIIS coefficients with residual-scale-invariant regularization.

    ``history`` uses the same tuple layout as :func:`rubycgw.gw._mixed_self_energies`:
    ``(hout, gout, residual_h, residual_gw)``.
    """
    # Import lazily to avoid a circular import when this helper is installed
    # while the package initializer is setting up the production GW module.
    from .gw import _residual_inner

    m = len(history)
    if m < 1:
        raise ValueError("Pulay history must contain at least one entry")

    gram = np.empty((m, m), dtype=float)
    for i in range(m):
        _, _, rhi, rgi = history[i]
        for j in range(i, m):
            _, _, rhj, rgj = history[j]
            val = _residual_inner(rhi, rgi, rhj, rgj)
            gram[i, j] = val
            gram[j, i] = val

    # Normalize before regularizing.  This removes the absolute residual scale
    # from the augmented DIIS system and prevents the regularizer from taking
    # over late in convergence.  Use the full Gram max as a fallback because
    # cancellation can make a diagonal entry anomalously small.
    diag_scale = float(np.max(np.abs(np.diag(gram))))
    full_scale = float(np.max(np.abs(gram)))
    scale = max(diag_scale, full_scale, np.finfo(float).tiny)
    gram = gram / scale
    gram += float(regularization) * np.eye(m)

    B = np.zeros((m + 1, m + 1), dtype=float)
    B[:m, :m] = gram
    B[:m, m] = 1.0
    B[m, :m] = 1.0
    rhs = np.zeros(m + 1, dtype=float)
    rhs[m] = 1.0

    try:
        sol = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(B, rhs, rcond=None)[0]
    return np.asarray(sol[:m], dtype=float)


def install_scale_invariant_pulay():
    """Install the scale-invariant coefficient solver into ``rubycgw.gw``.

    The package initializer already calls this by default.  Keeping the helper
    idempotent preserves compatibility with older launchers that install it
    explicitly.  ``_mixed_self_energies`` resolves ``_pulay_coefficients`` from
    the GW module at call time, so already-imported mixer aliases are covered.
    """
    from . import gw

    gw._pulay_coefficients = scale_invariant_pulay_coefficients
    return scale_invariant_pulay_coefficients


__all__ = [
    "scale_invariant_pulay_coefficients",
    "install_scale_invariant_pulay",
]
