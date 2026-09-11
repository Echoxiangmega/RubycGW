import numpy as np

from rubycgw.cluster_ed_gw import (
    BathParameters,
    ClusterEDGWOptions,
    bath_hybridization,
    build_intracell_h0,
    cluster_gw_self_energy,
    fit_finite_bath,
    ruby_cluster_interactions,
    solve_cluster_ed_gw,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.impurity_ed import FiniteBathImpurityED
from rubycgw.model import RubyParameters, build_h0, build_interaction


def test_ruby_six_site_cluster_contains_all_interactions():
    p = RubyParameters(V=1.7)
    terms = ruby_cluster_interactions(p)
    assert len(terms) == 6
    assert {(min(i, j), max(i, j)) for i, j, _ in terms} == {
        (0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)
    }
    assert all(abs(u - 1.7) < 1e-14 for _, _, u in terms)


def test_finite_bath_ed_is_exact_for_noninteracting_resolvent():
    h = np.array(
        [
            [0.10, 0.20, 0.35, 0.00],
            [0.20, -0.15, 0.00, -0.25],
            [0.35, 0.00, 0.70, 0.10],
            [0.00, -0.25, 0.10, -0.55],
        ],
        dtype=float,
    )
    solver = FiniteBathImpurityED(h, (), correlated_orbitals=(0, 1))
    solver.diagonalize()
    T = 0.17
    mu = 0.08
    omega = (2 * np.arange(-4, 4) + 1) * np.pi * T
    G, _ = solver.green_iomega(1j * omega, mu, T, discard_weight_tol=0.0)
    eye = np.eye(4)
    exact = np.asarray([
        np.linalg.inv((1j*w + mu) * eye - h)[:2, :2]
        for w in omega
    ])
    # Projection and inversion do not commute: take the active block after the
    # full resolvent, exactly as the impurity Green function does.
    exact = np.asarray([
        np.linalg.inv((1j*w + mu) * eye - h)[:2, :2]
        for w in omega
    ])
    assert np.max(np.abs(G - exact)) < 2e-10


def test_bath_fit_recovers_exact_seed_hybridization():
    T = 0.11
    omega = (2 * np.arange(-6, 6) + 1) * np.pi * T
    mu = 0.2
    eps = np.array([-0.7, 0.9])
    V = np.array([[0.45, 0.12], [-0.18, 0.38]])
    target = bath_hybridization(omega, mu, eps, V)
    seed = BathParameters(eps.copy(), V.copy(), 0.0, 0)
    fit = fit_finite_bath(
        target,
        omega,
        mu,
        nbath=2,
        nfit=5,
        max_nfev=30,
        initial=seed,
    )
    assert fit.fit_error < 1e-10
    reproduced = bath_hybridization(omega, mu, fit.energies, fit.couplings)
    assert np.linalg.norm(reproduced - target) / np.linalg.norm(target) < 1e-9


def test_cluster_gw_double_counting_vanishes_at_zero_interaction():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=4, nOmega=2, T=0.12)
    h = build_intracell_h0(RubyParameters(V=0.0))
    eye = np.eye(6)
    mu = 0.03
    G = np.asarray([np.linalg.inv((1j*w + mu) * eye - h) for w in grid.omega])
    rho = np.eye(6) * 0.25
    sigma, P, W = cluster_gw_self_energy(G, rho, np.zeros((6, 6)), grid)
    assert np.max(np.abs(sigma)) < 1e-14
    assert np.max(np.abs(W)) < 1e-14
    assert np.all(np.isfinite(P))


def test_zero_interaction_embedding_reduces_to_lattice_gw():
    p = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=1, nw=3, nOmega=1, T=0.16)
    h0 = build_h0(grid.kmesh(), p)
    Vq = build_interaction(grid.qmesh(), p)
    gw_opts = GWOptions(
        target_filling=2.0,
        max_iter=5,
        tol=1e-10,
        mixing=0.5,
        mixing_method="linear",
        verbose=False,
        momentum_backend="direct",
    )
    eopts = ClusterEDGWOptions(
        max_iter=2,
        tol=1e-8,
        mixing=0.5,
        impurity_mixing=1.0,
        nbath=2,
        bath_fit_nfreq=2,
        bath_fit_max_nfev=20,
        discard_weight_tol=0.0,
        verbose=False,
    )
    out = solve_cluster_ed_gw(h0, Vq, p, grid, gw_opts=gw_opts, embed_opts=eopts)
    den = np.linalg.norm(out.background.G.ravel())
    assert np.linalg.norm((out.G - out.background.G).ravel()) / den < 5e-8
    assert np.max(np.abs(out.Sigma_ED_cluster)) < 5e-8
