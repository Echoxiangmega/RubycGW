import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.hedin_gamma import (
    GammaPFeedbackOptions,
    build_hedin_gamma_screening,
    covariant_chi_to_irreducible_p,
    screened_interaction_from_p_gamma,
    solve_matrix_gw_gamma_feedback,
)
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw import compute_screened_interaction_matrix
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast


def test_covariant_chi_to_p_gamma_recovers_hedin_screening():
    rng = np.random.default_rng(19)
    nk1, nk2, nb, norb = 2, 1, 3, 3
    Vq = rng.normal(scale=.18, size=(nk1, nk2, norb, norb))
    P = rng.normal(scale=.08, size=(nb, nk1, nk2, norb, norb))
    # Keep the test away from a screening pole without imposing commutativity.
    eye = np.eye(norb, dtype=complex)
    W = np.empty_like(P, dtype=complex)
    chi = np.empty_like(P, dtype=complex)
    for im in range(nb):
        for iq1 in range(nk1):
            v = Vq[iq1, 0]
            p = P[im, iq1, 0]
            W[im, iq1, 0] = np.linalg.solve(eye - v @ p, v)
            chi[im, iq1, 0] = -np.linalg.solve(eye - p @ v, p)

    recovered = covariant_chi_to_irreducible_p(Vq, chi)
    np.testing.assert_allclose(recovered, P, rtol=3e-12, atol=3e-13)
    W_from_p = screened_interaction_from_p_gamma(Vq, recovered)
    np.testing.assert_allclose(W_from_p, W, rtol=3e-12, atol=3e-13)

    built = build_hedin_gamma_screening(Vq, chi)
    np.testing.assert_allclose(built.W_direct, W, rtol=3e-12, atol=3e-13)
    np.testing.assert_allclose(built.W_hedin, W, rtol=3e-12, atol=3e-13)
    assert built.identity_error < 2e-12


def test_windowed_hedin_gamma_screening_uses_background_only_for_nan_transfers():
    Vq = np.array([[[[0.0, .3], [.3, 0.0]]]], dtype=complex)
    chi = np.zeros((3, 1, 1, 2, 2), dtype=complex)
    chi[0, 0, 0] = np.array([[.4, .03], [.02, .25]])
    chi[1, 0, 0] = np.nan
    chi[2, 0, 0] = np.array([[.2, -.01], [.04, .15]])
    background = np.broadcast_to(Vq[None], chi.shape).copy()
    background[1] *= 1.17

    built = build_hedin_gamma_screening(
        Vq,
        chi,
        background_W=background,
    )
    assert built.fallback_mask.tolist() == [[[False]], [[True]], [[False]]]
    np.testing.assert_allclose(built.W_direct[1], background[1])
    np.testing.assert_allclose(built.W_hedin[1], background[1])
    assert np.all(np.isnan(built.P_gamma[1]))
    np.testing.assert_allclose(built.W_direct[[0, 2]], built.W_hedin[[0, 2]], rtol=2e-12, atol=2e-13)


def test_zero_interaction_gamma_p_feedback_is_identity():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=4, nOmega=1, T=.22)
    h = np.array([[.11, .15], [.15, -.08]], dtype=complex)
    h0 = h[None, None]
    Vq = np.zeros_like(h0)
    gw_opts = GWOptions(
        mu=.013,
        target_filling=None,
        max_iter=4,
        tol=1e-11,
        verbose=False,
        momentum_backend="direct",
    )
    vertex_opts = SupercellVertexOptions(
        max_iter=8,
        tol=1e-11,
        solver="gmres",
        gmres_restart=6,
        verbose=False,
        momentum_backend="direct",
    )
    fb_opts = GammaPFeedbackOptions(
        max_iter=3,
        tol=1e-10,
        mixing=.3,
        m_max=0,
        verbose=False,
    )
    gw = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    out = solve_matrix_gw_gamma_feedback(
        h0,
        Vq,
        grid,
        gw_opts=gw_opts,
        vertex_opts=vertex_opts,
        feedback_opts=fb_opts,
        background=gw,
    )
    assert out.converged
    np.testing.assert_allclose(out.G, gw.G, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(out.W, 0.0, atol=1e-14)
    np.testing.assert_allclose(out.Sigma_GW, 0.0, atol=1e-14)
    assert out.screening_identity_error < 1e-13
