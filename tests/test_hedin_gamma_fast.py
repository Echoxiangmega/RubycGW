import numpy as np

from rubycgw.covariant_density import compute_covariant_density_susceptibility
from rubycgw.covariant_density_fast import compute_covariant_density_susceptibility_fast
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.hedin_gamma_fast import (
    GammaPFeedbackOptions,
    _pulay_coefficients,
    solve_matrix_gw_gamma_feedback,
)
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
)
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast
from rubycgw.supercell_gw_split import compute_sigma_gw_split_components


def _free_G(h0, mu, grid):
    eye = np.eye(h0.shape[-1], dtype=complex)
    return np.linalg.inv(
        (1j * grid.omega[:, None, None, None, None] + mu)
        * eye[None, None, None]
        - h0[None]
    )


def _small_background():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=4, nOmega=1, T=.21)
    h = np.array([[.14, .18 + .02j], [.18 - .02j, -.10]], dtype=complex)
    h0 = h[None, None]
    mu = .023
    G = _free_G(h0, mu, grid)
    Vq = np.array([[[[0.0, .16], [.16, 0.0]]]], dtype=complex)
    P = compute_polarization_matrix(G, grid, backend="direct")
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_h = np.zeros((2, 2), dtype=complex)
    _, sigma_f, _, _ = compute_sigma_gw_split_components(
        G, W, Vq, grid, h0, mu, sigma_h, backend="direct"
    )
    return grid, h0, mu, G, Vq, W, sigma_h, sigma_f


def _small_fft_background():
    grid = MatsubaraGrid(nk1=2, nk2=1, nw=4, nOmega=1, T=.20)
    h0 = np.empty((2, 1, 2, 2), dtype=complex)
    h0[0, 0] = np.array([[.13, .17 + .01j], [.17 - .01j, -.09]])
    h0[1, 0] = np.array([[.10, .14 - .025j], [.14 + .025j, -.06]])
    mu = .018
    G = _free_G(h0, mu, grid)
    Vq = np.empty_like(h0)
    Vq[0, 0] = np.array([[0.0, .14], [.14, 0.0]])
    Vq[1, 0] = np.array([[0.0, .09], [.09, 0.0]])
    P = compute_polarization_matrix(G, grid, backend="fft")
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_h = np.zeros((2, 2), dtype=complex)
    _, sigma_f, _, _ = compute_sigma_gw_split_components(
        G, W, Vq, grid, h0, mu, sigma_h, backend="fft"
    )
    return grid, h0, mu, G, Vq, W, sigma_h, sigma_f


def _vopts(backend="direct"):
    return SupercellVertexOptions(
        max_iter=80,
        tol=5e-10,
        solver="gmres",
        gmres_restart=8,
        verbose=False,
        momentum_backend=backend,
    )


def test_fast_covariant_density_matches_reference_and_reuses_cache():
    grid, h0, mu, G, Vq, W, sigma_h, sigma_f = _small_background()
    ref = compute_covariant_density_susceptibility(
        G, W, Vq, h0, mu, sigma_h, sigma_f, grid,
        vertex_opts=_vopts(), m_max=0,
    )
    fast, cache, stats1 = compute_covariant_density_susceptibility_fast(
        G, W, Vq, h0, mu, sigma_h, sigma_f, grid,
        vertex_opts=_vopts(), m_max=0,
    )
    np.testing.assert_allclose(fast.chi_completed, ref.chi_completed, rtol=2e-9, atol=3e-10)
    assert stats1.n_solves == 2
    assert len(cache) == 2

    fast2, cache2, stats2 = compute_covariant_density_susceptibility_fast(
        G, W, Vq, h0, mu, sigma_h, sigma_f, grid,
        vertex_opts=_vopts(), m_max=0, initial_gamma_cache=cache,
    )
    np.testing.assert_allclose(fast2.chi_completed, ref.chi_completed, rtol=2e-9, atol=3e-10)
    assert stats2.cache_hits == stats2.n_solves == 2
    assert len(cache2) == 2


def test_prepared_fft_density_kernel_matches_reference():
    grid, h0, mu, G, Vq, W, sigma_h, sigma_f = _small_fft_background()
    opts = _vopts("fft")
    ref = compute_covariant_density_susceptibility(
        G, W, Vq, h0, mu, sigma_h, sigma_f, grid,
        vertex_opts=opts, m_max=0,
    )
    fast, _, stats = compute_covariant_density_susceptibility_fast(
        G, W, Vq, h0, mu, sigma_h, sigma_f, grid,
        vertex_opts=opts, m_max=0,
    )
    np.testing.assert_allclose(fast.chi_completed, ref.chi_completed, rtol=3e-9, atol=5e-10)
    np.testing.assert_allclose(fast.chi_raw, ref.chi_raw, rtol=3e-9, atol=5e-10)
    assert stats.n_solves == 4


def test_three_block_pulay_coefficients_sum_to_one():
    z = np.zeros((2, 2), dtype=complex)
    o = np.ones((2, 2), dtype=complex)
    history = [
        (z, z, z, o, 2 * o, .5 * o),
        (o, o, o, .4 * o, .7 * o, .2 * o),
        (2 * o, 2 * o, 2 * o, .2 * o, .3 * o, .1 * o),
    ]
    c = _pulay_coefficients(history, 1e-10)
    np.testing.assert_allclose(np.sum(c), 1.0, atol=2e-12)
    assert np.all(np.isfinite(c))


def test_fast_gamma_feedback_zero_interaction_is_identity():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=4, nOmega=1, T=.22)
    h = np.array([[.11, .15], [.15, -.08]], dtype=complex)
    h0 = h[None, None]
    Vq = np.zeros_like(h0)
    gw_opts = GWOptions(
        mu=.017,
        target_filling=None,
        max_iter=4,
        tol=1e-11,
        verbose=False,
        momentum_backend="direct",
    )
    bg = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    out = solve_matrix_gw_gamma_feedback(
        h0,
        Vq,
        grid,
        gw_opts=gw_opts,
        vertex_opts=_vopts(),
        feedback_opts=GammaPFeedbackOptions(
            max_iter=4,
            tol=1e-9,
            mixing_method="pulay",
            verbose=False,
        ),
        background=bg,
    )
    assert out.converged
    assert out.iterations == 1
    np.testing.assert_allclose(out.G, bg.G, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(out.W, 0.0, atol=1e-14)
    assert getattr(out, "vertex_stats").n_solves == 2
