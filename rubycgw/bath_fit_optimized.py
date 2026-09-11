"""Optimized finite-bath fitting for the cluster-ED embedding.

The historical fitter minimizes an error in the hybridization function
``Delta(iw)``.  At strong coupling that error can look large even when the
actual impurity Weiss Green function is already accurate.  This module adds a
fit in the physically propagated quantity

    G0(iw) = [iw + mu - h_cluster - Delta(iw)]^{-1},

plus separate Delta/G0 diagnostics.  The G0 objective intentionally carries no
extra Matsubara weight: the 1/|omega| decay of G0 already emphasizes the low
frequencies that dominate the impurity physics.

The module is installed at runtime by the production driver, leaving the
historical implementation available for regression tests and reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from .cluster_ed_gw import BathParameters, _initial_bath_guess, bath_hybridization


BATH_FIT_METRICS = ("g0", "delta")


@dataclass(frozen=True)
class BathFitDiagnostics:
    delta_relerr: float
    g0_relerr: float
    g0_low_relerr: float
    nfit: int
    low_nfit: int
    metric: str


def _relerr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    return float(
        np.linalg.norm((aa - bb).ravel())
        / max(float(np.linalg.norm(bb.ravel())), 1.0e-300)
    )


def weiss_from_delta(
    delta: np.ndarray,
    omega: np.ndarray,
    mu: float,
    h_cluster: np.ndarray,
) -> np.ndarray:
    """Return the impurity Weiss Green function for a supplied hybridization."""
    d = np.asarray(delta, dtype=complex)
    w = np.asarray(omega, dtype=float).reshape(-1)
    h = np.asarray(h_cluster, dtype=complex)
    norb = int(h.shape[0])
    if h.shape != (norb, norb) or d.shape != (len(w), norb, norb):
        raise ValueError("delta/h_cluster shape mismatch")
    eye = np.eye(norb, dtype=complex)
    return np.linalg.inv(
        (1j * w[:, None, None] + float(mu)) * eye[None, :, :]
        - h[None, :, :]
        - d
    )


def bath_fit_diagnostics(
    target_delta: np.ndarray,
    model_delta: np.ndarray,
    omega: np.ndarray,
    mu: float,
    h_cluster: np.ndarray,
    *,
    low_nfit: int = 4,
    metric: str = "g0",
) -> BathFitDiagnostics:
    target = np.asarray(target_delta, dtype=complex)
    model = np.asarray(model_delta, dtype=complex)
    w = np.asarray(omega, dtype=float).reshape(-1)
    if target.shape != model.shape or target.shape[0] != len(w):
        raise ValueError("bath diagnostic shape mismatch")
    gt = weiss_from_delta(target, w, mu, h_cluster)
    gm = weiss_from_delta(model, w, mu, h_cluster)
    nlow = min(max(int(low_nfit), 1), len(w))
    return BathFitDiagnostics(
        delta_relerr=_relerr(model, target),
        g0_relerr=_relerr(gm, gt),
        g0_low_relerr=_relerr(gm[:nlow], gt[:nlow]),
        nfit=int(len(w)),
        low_nfit=int(nlow),
        metric=str(metric),
    )


def _delta_derivatives(
    omega: np.ndarray,
    mu: float,
    eps: np.ndarray,
    couplings: np.ndarray,
):
    """Yield analytic dDelta/dx in the fitter parameter ordering."""
    w = np.asarray(omega, dtype=float).reshape(-1)
    e = np.asarray(eps, dtype=float).reshape(-1)
    v = np.asarray(couplings, dtype=float)
    norb, nbath = v.shape
    den = 1.0 / (1j * w[:, None] + float(mu) - e[None, :])

    # Bath-energy derivatives.
    for p in range(nbath):
        outer = np.outer(v[:, p], v[:, p])
        yield (den[:, p] ** 2)[:, None, None] * outer[None, :, :]

    # Real cluster-bath coupling derivatives, flattened as v[a,p].
    for a in range(norb):
        for p in range(nbath):
            d = np.zeros((len(w), norb, norb), dtype=complex)
            d[:, a, :] += den[:, p, None] * v[:, p][None, :]
            d[:, :, a] += den[:, p, None] * v[:, p][None, :]
            yield d


def fit_finite_bath_optimized(
    target_delta: np.ndarray,
    omega: np.ndarray,
    mu: float,
    *,
    h_cluster: np.ndarray,
    metric: str = "g0",
    nbath: int = 6,
    nfit: int = 12,
    max_nfev: int = 300,
    energy_window: float = 4.0,
    coupling_bound: float = 4.0,
    xtol: float = 1.0e-9,
    initial: BathParameters | None = None,
    low_nfit: int = 4,
    analytic_jacobian: bool = True,
) -> BathParameters:
    """Fit a finite real bath using either a G0 or Delta objective.

    ``metric='g0'`` is the recommended strong-coupling default.  ``delta``
    reproduces the historical target (with its historical Matsubara weights)
    while still returning the new diagnostics.
    """
    metric = str(metric).lower()
    if metric not in BATH_FIT_METRICS:
        raise ValueError(f"metric must be one of {BATH_FIT_METRICS}")

    delta = np.asarray(target_delta, dtype=complex)
    w = np.asarray(omega, dtype=float).reshape(-1)
    h = np.asarray(h_cluster, dtype=complex)
    if delta.ndim != 3 or delta.shape[0] != len(w) or delta.shape[1] != delta.shape[2]:
        raise ValueError("target_delta must have shape (nf,norb,norb)")
    norb = int(delta.shape[-1])
    if h.shape != (norb, norb):
        raise ValueError("h_cluster shape mismatch")
    nbath = int(nbath)
    if nbath < 1:
        raise ValueError("nbath must be positive")

    pos = np.flatnonzero(w > 0.0)
    if pos.size == 0:
        raise ValueError("bath fit requires positive Matsubara frequencies")
    pos = pos[np.argsort(w[pos])[: min(int(nfit), len(pos))]]
    wfit = w[pos]
    dfit = delta[pos]
    g0_target = weiss_from_delta(dfit, wfit, mu, h)

    # Historical Delta metric uses an explicit low-frequency weight.  For G0
    # we deliberately do not add one: G0 itself decays as 1/omega and therefore
    # already gives low Matsubara points the appropriate leverage.
    w0 = max(float(wfit[0]), 1.0e-12)
    delta_weights = 1.0 / np.sqrt(wfit * wfit + w0 * w0)
    delta_weights /= float(np.max(delta_weights))

    if initial is not None:
        eps0 = np.asarray(initial.energies, dtype=float)
        c0raw = np.asarray(initial.couplings)
        bath_hyb0 = np.asarray(c0raw.real, dtype=float)
        if eps0.shape != (nbath,) or bath_hyb0.shape != (norb, nbath):
            eps0, bath_hyb0 = _initial_bath_guess(delta, w, mu, nbath, energy_window)
    else:
        eps0, bath_hyb0 = _initial_bath_guess(delta, w, mu, nbath, energy_window)

    lo_e = float(mu) - float(energy_window)
    hi_e = float(mu) + float(energy_window)
    eps0 = np.clip(eps0, lo_e + 1.0e-8, hi_e - 1.0e-8)
    bath_hyb0 = np.clip(
        bath_hyb0,
        -float(coupling_bound) + 1.0e-8,
        float(coupling_bound) - 1.0e-8,
    )
    x0 = np.concatenate([eps0, bath_hyb0.ravel()])
    lower = np.concatenate([
        np.full(nbath, lo_e),
        np.full(norb * nbath, -float(coupling_bound)),
    ])
    upper = np.concatenate([
        np.full(nbath, hi_e),
        np.full(norb * nbath, float(coupling_bound)),
    ])

    def unpack(x):
        return x[:nbath], x[nbath:].reshape(norb, nbath)

    def model_fields(x):
        eps, bath_hyb = unpack(x)
        dmodel = bath_hybridization(wfit, mu, eps, bath_hyb)
        if metric == "g0":
            return dmodel, weiss_from_delta(dmodel, wfit, mu, h)
        return dmodel, None

    def residual(x):
        dmodel, gmodel = model_fields(x)
        if metric == "g0":
            diff = gmodel - g0_target
        else:
            diff = (dmodel - dfit) * delta_weights[:, None, None]
        return np.concatenate([diff.real.ravel(), diff.imag.ravel()])

    def jacobian(x):
        eps, bath_hyb = unpack(x)
        dmodel, gmodel = model_fields(x)
        columns = []
        for ddelta in _delta_derivatives(wfit, mu, eps, bath_hyb):
            if metric == "g0":
                # d(A^-1) = G0 dDelta G0 because A=A0-Delta.
                deriv = np.einsum(
                    "nab,nbc,ncd->nad", gmodel, ddelta, gmodel, optimize=True
                )
            else:
                deriv = ddelta * delta_weights[:, None, None]
            columns.append(
                np.concatenate([deriv.real.ravel(), deriv.imag.ravel()])
            )
        return np.stack(columns, axis=1)

    opt = least_squares(
        residual,
        x0,
        jac=jacobian if analytic_jacobian else "2-point",
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        xtol=float(xtol),
        ftol=float(xtol),
        gtol=float(xtol),
    )
    eps, bath_hyb = unpack(opt.x)
    order = np.argsort(eps)
    eps = np.asarray(eps[order], dtype=float)
    bath_hyb = np.asarray(bath_hyb[:, order], dtype=float)
    fit = bath_hybridization(wfit, mu, eps, bath_hyb)
    diag = bath_fit_diagnostics(
        dfit,
        fit,
        wfit,
        mu,
        h,
        low_nfit=low_nfit,
        metric=metric,
    )
    primary = diag.g0_relerr if metric == "g0" else diag.delta_relerr
    bath = BathParameters(eps, bath_hyb, float(primary), int(opt.nfev))
    # BathParameters intentionally remains backward-compatible.  Attach richer
    # diagnostics dynamically so old callers need no schema change.
    bath.delta_fit_error = float(diag.delta_relerr)
    bath.g0_fit_error = float(diag.g0_relerr)
    bath.g0_low_fit_error = float(diag.g0_low_relerr)
    bath.fit_metric = str(metric)
    bath.fit_nfreq = int(diag.nfit)
    bath.low_fit_nfreq = int(diag.low_nfit)
    return bath


def make_bath_fitter(
    h_cluster: np.ndarray,
    *,
    metric: str = "g0",
    low_nfit: int = 4,
    analytic_jacobian: bool = True,
) -> Callable:
    """Return a drop-in replacement for ``fit_finite_bath``."""
    h = np.asarray(h_cluster, dtype=complex).copy()

    def fitter(target_delta, omega, mu, **kwargs):
        return fit_finite_bath_optimized(
            target_delta,
            omega,
            mu,
            h_cluster=h,
            metric=metric,
            low_nfit=low_nfit,
            analytic_jacobian=analytic_jacobian,
            **kwargs,
        )

    return fitter


def install_optimized_bath_fit(
    h_cluster: np.ndarray,
    *,
    metric: str = "g0",
    low_nfit: int = 4,
    analytic_jacobian: bool = True,
):
    """Install the optimized fitter into the production embedding modules."""
    fitter = make_bath_fitter(
        h_cluster,
        metric=metric,
        low_nfit=low_nfit,
        analytic_jacobian=analytic_jacobian,
    )
    from . import cluster_ed_gw as base
    from . import cluster_ed_gw_fast as gw_fast
    from . import cluster_ed_weak_fast as weak_fast

    base.fit_finite_bath = fitter
    gw_fast.fit_finite_bath = fitter
    weak_fast.fit_finite_bath = fitter
    return fitter


__all__ = [
    "BATH_FIT_METRICS",
    "BathFitDiagnostics",
    "weiss_from_delta",
    "bath_fit_diagnostics",
    "fit_finite_bath_optimized",
    "make_bath_fitter",
    "install_optimized_bath_fit",
]
