"""Compatibility layer for the active 18-site GW solver.

Historically the driver imported this module for the Anderson implementation.
The active solver now lives in :mod:`rubycgw.supercell_gw_periodic_pulay`: it
keeps the same public solver API while using an analytic Hartree cold start,
separate Hartree/GW treatment, GW-only periodic Pulay acceleration, and exactly
one expensive GW map per outer iteration.
"""

import numpy as np

from .supercell_gw_periodic_pulay import (
    AndersonOptions,
    solve_matrix_gw_anderson,
    solve_supercell_gw_anderson,
)


def _metric_vector(h, gw, sh, sg):
    vh = np.asarray(h, dtype=complex).reshape(-1) / (
        float(sh) * np.sqrt(max(np.size(h), 1))
    )
    vg = np.asarray(gw, dtype=complex).reshape(-1) / (
        float(sg) * np.sqrt(max(np.size(gw), 1))
    )
    return np.concatenate([vh, vg])


def _anderson_type2_step(
    sigma_h,
    sigma_gw,
    res_h,
    res_gw,
    history,
    beta,
    regularization,
    sh,
    sg,
    step_cap,
):
    """Backward-compatible Type-II Anderson helper used by regression tests."""
    linear_h = beta * res_h
    linear_gw = beta * res_gw
    if len(history) < 2:
        return sigma_h + linear_h, sigma_gw + linear_gw, False

    dR_cols = []
    dX_h, dX_gw, dR_h, dR_gw = [], [], [], []
    for old, new in zip(history[:-1], history[1:]):
        xh0, xg0, rh0, rg0 = old
        xh1, xg1, rh1, rg1 = new
        dxh, dxg = xh1 - xh0, xg1 - xg0
        drh, drg = rh1 - rh0, rg1 - rg0
        dX_h.append(dxh)
        dX_gw.append(dxg)
        dR_h.append(drh)
        dR_gw.append(drg)
        dR_cols.append(_metric_vector(drh, drg, sh, sg))

    A = np.column_stack(dR_cols)
    rhs = _metric_vector(res_h, res_gw, sh, sg)
    m = A.shape[1]
    if regularization > 0.0:
        A = np.vstack([A, np.sqrt(regularization) * np.eye(m, dtype=complex)])
        rhs = np.concatenate([rhs, np.zeros(m, dtype=complex)])
    try:
        gamma = np.linalg.lstsq(A, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return sigma_h + linear_h, sigma_gw + linear_gw, False

    step_h = np.array(linear_h, copy=True)
    step_gw = np.array(linear_gw, copy=True)
    for c, dxh, dxg, drh, drg in zip(
        gamma, dX_h, dX_gw, dR_h, dR_gw
    ):
        step_h -= c * (dxh + beta * drh)
        step_gw -= c * (dxg + beta * drg)

    linear_norm = np.linalg.norm(_metric_vector(linear_h, linear_gw, sh, sg))
    step_norm = np.linalg.norm(_metric_vector(step_h, step_gw, sh, sg))
    if not np.isfinite(step_norm) or step_norm > float(step_cap) * max(linear_norm, 1e-16):
        return sigma_h + linear_h, sigma_gw + linear_gw, False
    return sigma_h + step_h, sigma_gw + step_gw, True


__all__ = [
    "AndersonOptions",
    "_anderson_type2_step",
    "solve_matrix_gw_anderson",
    "solve_supercell_gw_anderson",
]
