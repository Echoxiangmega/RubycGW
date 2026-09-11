import numpy as np

from rubycgw.cluster_ed_gw import BathParameters, bath_hybridization
from rubycgw.cluster_ed_gw_covariant import fit_finite_bath_complex
from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


def test_complex_bath_fit_reproduces_complex_hybridization_from_exact_seed():
    omega = (2 * np.arange(-5, 6) + 1) * np.pi * 0.1
    mu = 0.13
    eps = np.asarray([-0.7, 0.45])
    hyb = np.asarray([
        [0.31 + 0.08j, -0.17 + 0.03j],
        [0.12 - 0.05j, 0.28 + 0.09j],
    ])
    target = bath_hybridization(omega, mu, eps, hyb)
    initial = BathParameters(eps.copy(), hyb.copy(), 0.0, 0)
    fit = fit_finite_bath_complex(
        target,
        omega,
        mu,
        nbath=2,
        nfit=5,
        max_nfev=20,
        energy_window=2.0,
        coupling_bound=2.0,
        xtol=1e-12,
        initial=initial,
    )
    assert fit.fit_error < 1e-10
    rebuilt = bath_hybridization(omega, mu, fit.energies, fit.couplings)
    assert np.linalg.norm(rebuilt - target) / np.linalg.norm(target) < 1e-9


def test_exact_current_number_cross_response_vanishes_at_tr_symmetric_point():
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.4)
    exact = ExactSmallRubyThermal(1, 1, params)
    exact.diagonalize(params.V)
    T = 0.08
    mu = exact.solve_mu(2.0, T)
    same = exact.pseudospin_operator("z_same", (0.0, 0.0))
    opposite = exact.pseudospin_operator("z_opposite", (0.0, 0.0))
    number = np.eye(exact.n_sites, dtype=complex)
    chi, means = exact.static_susceptibility_matrix(
        np.stack([same, opposite, number]), mu, T
    )
    assert np.max(np.abs(means[:2])) < 1e-10
    assert np.max(np.abs(chi[:2, 2])) < 1e-9
    assert chi[0, 0] > 0.0
    assert chi[1, 1] > 0.0
