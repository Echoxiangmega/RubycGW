import numpy as np

from rubycgw.cluster_ed_gw import (
    BathParameters,
    bath_hybridization,
    ruby_cluster_interactions,
)
from rubycgw.cluster_ed_gw_jf import (
    ClusterEDGWQJacobian,
    ClusterJFOptions,
    LinearizedBathFit,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import primitive_pseudospin_vertex


def test_linearized_bath_fit_reproduces_model_tangent():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=5, nOmega=2, T=0.13)
    eps = np.array([-0.8, 0.65])
    couplings = np.array(
        [
            [0.35 + 0.10j, -0.21 + 0.04j],
            [0.08 - 0.16j, 0.27 + 0.12j],
        ],
        dtype=complex,
    )
    bath = BathParameters(eps, couplings, 0.0, 0)
    lin = LinearizedBathFit(grid.omega, 0.17, bath, nfit=5, rcond=1e-12)

    theta0 = lin.theta()
    rng = np.random.default_rng(8)
    dtheta = rng.normal(size=theta0.size)
    h = 1e-7
    bp = lin.bath_from_theta(theta0 + h * dtheta)
    bm = lin.bath_from_theta(theta0 - h * dtheta)
    dp = bath_hybridization(grid.omega, 0.17, bp.energies, bp.couplings)
    dm = bath_hybridization(grid.omega, 0.17, bm.energies, bm.couplings)
    ddelta = (dp - dm) / (2.0 * h)

    recovered = lin.parameter_direction(ddelta)
    brp = lin.bath_from_theta(theta0 + h * recovered)
    brm = lin.bath_from_theta(theta0 - h * recovered)
    rp = bath_hybridization(grid.omega, 0.17, brp.energies, brp.couplings)
    rm = bath_hybridization(grid.omega, 0.17, brm.energies, brm.couplings)
    reconstructed = (rp - rm) / (2.0 * h)

    idx = lin.fit_indices
    den = max(np.linalg.norm(ddelta[idx].ravel()), 1e-300)
    err = np.linalg.norm((reconstructed[idx] - ddelta[idx]).ravel()) / den
    assert err < 2e-6
    assert lin.diagnostics.rank <= lin.diagnostics.n_parameters


def test_jf_finite_q_reduces_to_bare_vertex_at_zero_interaction():
    grid = MatsubaraGrid(nk1=2, nk2=1, nw=3, nOmega=1, T=0.20)
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0)
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_interaction(grid.qmesh(), params)
    mu = 0.11
    eye = np.eye(6, dtype=complex)
    G = np.empty((grid.nf, grid.nk1, grid.nk2, 6, 6), dtype=complex)
    for n, w in enumerate(grid.omega):
        for i in range(grid.nk1):
            for j in range(grid.nk2):
                G[n, i, j] = np.linalg.inv((1j * w + mu) * eye - h0[i, j])
    Gc = np.mean(G, axis=(1, 2))
    W = np.zeros((grid.nb, grid.nk1, grid.nk2, 6, 6), dtype=complex)
    sigma_imp = np.zeros((grid.nf, 6, 6), dtype=complex)

    couplings = np.array(
        [[0.18], [0.11], [0.07], [0.15], [0.09], [0.13]], dtype=complex
    )
    bath = BathParameters(np.array([0.37]), couplings, 0.0, 0)
    opts = ClusterJFOptions(
        rtol=2e-7,
        maxiter=10,
        gcrot_m=6,
        gcrot_k=2,
        preconditioner_order=0,
        bath_fit_nfreq=3,
        bath_svd_rcond=1e-10,
        difference_scheme="forward",
        fd_rel_step=1e-4,
        discard_weight_tol=1e-13,
        verbose=False,
    )
    jac = ClusterEDGWQJacobian.from_embedding(
        G=G,
        W=W,
        Vq=Vq,
        G_cluster=Gc,
        Sigma_imp=sigma_imp,
        bath=bath,
        h_cluster=np.mean(h0, axis=(0, 1)),
        interactions=ruby_cluster_interactions(params),
        mu=mu,
        grid=grid,
        q_index=(1, 0),
        opts=opts,
    )
    K = primitive_pseudospin_vertex("Ax")
    result, _, _ = jac.solve(K)
    expected = np.broadcast_to(K, G.shape)
    rel = np.linalg.norm((result.Gamma - expected).ravel()) / np.linalg.norm(expected.ravel())
    assert result.info == 0
    assert result.residual_max < 1e-5
    assert rel < 1e-5
