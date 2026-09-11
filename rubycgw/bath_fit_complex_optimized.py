"""Complex finite-bath fitting using the impurity Weiss Green function.

This is the source-response counterpart of :mod:`bath_fit_optimized`.  Bath
energies remain real while the cluster-bath couplings are complex, as required
for time-reversal-odd loop-current sources.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .bath_fit_optimized import BATH_FIT_METRICS, bath_fit_diagnostics, weiss_from_delta
from .cluster_ed_gw import BathParameters, bath_hybridization
from .cluster_ed_gw_covariant import _initial_complex_bath_guess


def fit_finite_bath_complex_optimized(
    target_delta: np.ndarray,
    omega: np.ndarray,
    mu: float,
    *,
    h_cluster: np.ndarray,
    metric: str = "g0",
    nbath: int = 6,
    nfit: int = 12,
    max_nfev: int = 500,
    energy_window: float = 4.0,
    coupling_bound: float = 4.0,
    xtol: float = 1.0e-10,
    initial: BathParameters | None = None,
    low_nfit: int = 4,
) -> BathParameters:
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

    pos = np.flatnonzero(w > 0.0)
    if pos.size == 0:
        raise ValueError("bath fit requires positive Matsubara frequencies")
    pos = pos[np.argsort(w[pos])[: min(int(nfit), len(pos))]]
    wfit = w[pos]
    dfit = delta[pos]
    g0_target = weiss_from_delta(dfit, wfit, mu, h)
    w0 = max(float(wfit[0]), 1.0e-12)
    weights = 1.0 / np.sqrt(wfit * wfit + w0 * w0)
    weights /= float(np.max(weights))

    if initial is None:
        eps0, hyb0 = _initial_complex_bath_guess(delta, w, mu, nbath, energy_window)
    else:
        eps0 = np.asarray(initial.energies, dtype=float)
        hyb0 = np.asarray(initial.couplings, dtype=complex)
        if eps0.shape != (nbath,) or hyb0.shape != (norb, nbath):
            eps0, hyb0 = _initial_complex_bath_guess(delta, w, mu, nbath, energy_window)

    lo_e = float(mu) - float(energy_window)
    hi_e = float(mu) + float(energy_window)
    eps0 = np.clip(eps0, lo_e + 1.0e-8, hi_e - 1.0e-8)
    bound = float(coupling_bound)
    re0 = np.clip(hyb0.real, -bound + 1.0e-8, bound - 1.0e-8)
    im0 = np.clip(hyb0.imag, -bound + 1.0e-8, bound - 1.0e-8)
    x0 = np.concatenate([eps0, re0.ravel(), im0.ravel()])
    lower = np.concatenate([
        np.full(nbath, lo_e),
        np.full(2 * norb * nbath, -bound),
    ])
    upper = np.concatenate([
        np.full(nbath, hi_e),
        np.full(2 * norb * nbath, bound),
    ])
    nvc = norb * nbath

    def unpack(x):
        eps = x[:nbath]
        re = x[nbath:nbath + nvc].reshape(norb, nbath)
        im = x[nbath + nvc:].reshape(norb, nbath)
        return eps, re + 1j * im

    def residual(x):
        eps, hyb = unpack(x)
        dmodel = bath_hybridization(wfit, mu, eps, hyb)
        if metric == "g0":
            diff = weiss_from_delta(dmodel, wfit, mu, h) - g0_target
        else:
            diff = (dmodel - dfit) * weights[:, None, None]
        return np.concatenate([diff.real.ravel(), diff.imag.ravel()])

    opt = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        xtol=float(xtol),
        ftol=float(xtol),
        gtol=float(xtol),
    )
    eps, hyb = unpack(opt.x)
    order = np.argsort(eps)
    eps = np.asarray(eps[order], dtype=float)
    hyb = np.asarray(hyb[:, order], dtype=complex)
    fit = bath_hybridization(wfit, mu, eps, hyb)
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
    bath = BathParameters(eps, hyb, float(primary), int(opt.nfev))
    bath.delta_fit_error = float(diag.delta_relerr)
    bath.g0_fit_error = float(diag.g0_relerr)
    bath.g0_low_fit_error = float(diag.g0_low_relerr)
    bath.fit_metric = str(metric)
    return bath


def install_complex_bath_fit(
    h_cluster: np.ndarray,
    *,
    metric: str = "g0",
    low_nfit: int = 4,
):
    h = np.asarray(h_cluster, dtype=complex).copy()

    def fitter(target_delta, omega, mu, **kwargs):
        return fit_finite_bath_complex_optimized(
            target_delta,
            omega,
            mu,
            h_cluster=h,
            metric=metric,
            low_nfit=low_nfit,
            **kwargs,
        )

    from . import cluster_ed_gw_covariant as gw_cov
    from . import cluster_ed_weak_covariant as weak_cov
    gw_cov.fit_finite_bath_complex = fitter
    weak_cov.fit_finite_bath_complex = fitter
    return fitter


__all__ = [
    "fit_finite_bath_complex_optimized",
    "install_complex_bath_fit",
]
