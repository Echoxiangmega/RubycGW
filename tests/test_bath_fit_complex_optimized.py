import numpy as np

from rubycgw.bath_fit_complex_optimized import fit_finite_bath_complex_optimized
from rubycgw.cluster_ed_gw import BathParameters, bath_hybridization


def test_complex_g0_fit_recovers_exact_seed():
    T = 0.13
    omega = (2 * np.arange(-5, 5) + 1) * np.pi * T
    mu = 0.07
    h = np.array([[0.1, 0.15j], [-0.15j, -0.05]], dtype=complex)
    eps = np.array([-0.6, 0.8])
    couplings = np.array(
        [[0.42 + 0.08j, 0.11 - 0.03j], [-0.17 + 0.06j, 0.35 + 0.04j]],
        dtype=complex,
    )
    target = bath_hybridization(omega, mu, eps, couplings)
    seed = BathParameters(eps.copy(), couplings.copy(), 0.0, 0)
    fit = fit_finite_bath_complex_optimized(
        target,
        omega,
        mu,
        h_cluster=h,
        metric="g0",
        nbath=2,
        nfit=4,
        max_nfev=20,
        initial=seed,
    )
    reproduced = bath_hybridization(omega, mu, fit.energies, fit.couplings)
    assert np.linalg.norm(reproduced - target) / np.linalg.norm(target) < 1e-8
    assert fit.g0_fit_error < 1e-9
