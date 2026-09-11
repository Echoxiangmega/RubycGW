import numpy as np

from rubycgw.bath_fit_optimized import (
    bath_fit_diagnostics,
    fit_finite_bath_optimized,
    weiss_from_delta,
)
from rubycgw.cluster_ed_gw import BathParameters, bath_hybridization


def test_g0_bath_fit_recovers_exact_seed_and_diagnostics():
    T = 0.11
    omega = (2 * np.arange(-6, 6) + 1) * np.pi * T
    mu = 0.2
    h = np.array([[0.1, 0.2], [0.2, -0.15]], dtype=float)
    eps = np.array([-0.7, 0.9])
    couplings = np.array([[0.45, 0.12], [-0.18, 0.38]])
    target = bath_hybridization(omega, mu, eps, couplings)
    seed = BathParameters(eps.copy(), couplings.copy(), 0.0, 0)
    fit = fit_finite_bath_optimized(
        target,
        omega,
        mu,
        h_cluster=h,
        metric="g0",
        nbath=2,
        nfit=5,
        max_nfev=20,
        initial=seed,
    )
    reproduced = bath_hybridization(omega, mu, fit.energies, fit.couplings)
    assert np.linalg.norm(reproduced - target) / np.linalg.norm(target) < 1e-9
    assert fit.g0_fit_error < 1e-10
    assert fit.delta_fit_error < 1e-9
    assert fit.fit_metric == "g0"


def test_weiss_diagnostics_detect_small_propagated_error():
    omega = np.array([0.3, 0.9, 1.5])
    mu = 0.1
    h = np.array([[0.0, 0.2], [0.2, 0.1]])
    target = np.zeros((3, 2, 2), dtype=complex)
    target[:, 0, 0] = np.array([1.0 - 0.5j, 0.3 - 0.2j, 0.1 - 0.1j])
    model = target.copy()
    model[:, 0, 0] *= 1.1
    gt = weiss_from_delta(target, omega, mu, h)
    gm = weiss_from_delta(model, omega, mu, h)
    diag = bath_fit_diagnostics(target, model, omega, mu, h, low_nfit=2)
    direct = np.linalg.norm((gm - gt).ravel()) / np.linalg.norm(gt.ravel())
    assert abs(diag.g0_relerr - direct) < 1e-14
    assert diag.delta_relerr > 0.0
    assert diag.g0_low_relerr > 0.0
