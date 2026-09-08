import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.gw_sox import solve_matrix_gw_sox
from rubycgw.production_cgw_sox import (
    solve_vertex_q0_tail_sox,
    static_gg_tail_completed,
)
from rubycgw.response_tail import build_tail_reference
from rubycgw.sox_covariant import SOXOptions
from rubycgw.supercell_cgw import SupercellVertexOptions


def _free_G(h0, mu, grid):
    eye = np.eye(h0.shape[-1], dtype=complex)
    return np.linalg.inv(
        (1j * grid.omega[:, None, None, None, None] + mu)
        * eye[None, None, None]
        - h0[None]
    )


def test_gw_sox_zero_interaction_reduces_to_free_background():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=10, nOmega=3, T=.2)
    h = np.array([[.15, .23 + .04j], [.23 - .04j, -.12]], dtype=complex)
    h0 = h[None, None]
    Vq = np.zeros_like(h0)
    opts = GWOptions(mu=.037, target_filling=None, max_iter=4, tol=1e-11, verbose=False)
    result = solve_matrix_gw_sox(
        h0,
        Vq,
        grid,
        opts=opts,
        sox_opts=SOXOptions(n_quad=16),
    )
    expected = _free_G(h0, opts.mu, grid)
    assert result.converged
    assert result.iterations == 1
    np.testing.assert_allclose(result.G, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(result.Sigma_H, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.Sigma_GW, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.Sigma_SOX, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.Sigma_corr, 0.0, atol=1e-14)


def test_cgw_sox_zero_interaction_reduces_to_bare_vertex():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=10, nOmega=3, T=.2)
    h = np.array([[.11, .17], [.17, -.19]], dtype=complex)
    h0 = h[None, None]
    Vq = np.zeros_like(h0)
    W = np.zeros((grid.nb, 1, 1, 2, 2), dtype=complex)
    mu = .023
    G = _free_G(h0, mu, grid)
    K = np.array([[.3, .08j], [-.08j, -.2]], dtype=complex)
    reference = build_tail_reference(h0, mu, np.zeros((2, 2)), grid)
    result = solve_vertex_q0_tail_sox(
        G,
        W,
        Vq,
        K,
        grid,
        reference,
        h0,
        mu,
        vertex_opts=SupercellVertexOptions(
            max_iter=8,
            tol=1e-11,
            solver="gmres",
            verbose=False,
        ),
        sox_opts=SOXOptions(n_quad=16),
    )
    expected = np.broadcast_to(K, G.shape)
    assert result.converged
    np.testing.assert_allclose(result.Gamma, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(result.Gamma_SOX, 0.0, atol=1e-14)


def test_tail_completed_free_gg_matches_finite_difference_thermodynamics():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=5, nOmega=2, T=.17)
    h = np.array([[.13, .22], [.22, -.16]], dtype=complex)
    K = np.array([[.4, .07], [.07, -.25]], dtype=complex)
    mu = .031
    h0 = h[None, None]
    G = _free_G(h0, mu, grid)
    response = static_gg_tail_completed(
        G, K[None], grid, h0, mu, edge_points=1
    )["completed"][0, 0].real

    def expectation(eps):
        hp = h + eps * K
        e, U = np.linalg.eigh(hp)
        f = 1.0 / (np.exp((e - mu) / grid.T) + 1.0)
        rho = (U * f[None, :]) @ U.conj().T
        return float(np.trace(K @ rho).real)

    eps = 2e-6
    numeric = -(expectation(eps) - expectation(-eps)) / (2 * eps)
    np.testing.assert_allclose(response, numeric, rtol=3e-7, atol=3e-8)
