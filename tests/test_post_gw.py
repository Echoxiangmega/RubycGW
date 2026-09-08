import numpy as np

from rubycgw.benchmark_plot import plot_benchmark_npz
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.gw_sox import solve_matrix_gw_sox
from rubycgw.post_gw import build_post_screened_interaction, run_post_gw
from rubycgw.production_cgw import solve_vertex_q0_tail
from rubycgw.response_tail import build_tail_reference, reference_response_infinite
from rubycgw.sox_covariant import (
    SOXOptions,
    compute_sox_vertex_periodic,
    sox_self_energy_full,
)
from rubycgw.sox_transfer import (
    compute_sox_vertex_transfer_periodic,
    sox_vertex_transfer_full,
)
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
)
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast
from rubycgw.transfer_cgw import (
    solve_vertex_transfer_tail,
    transfer_response_tail_completed,
)


def _free_G(h0, mu, grid):
    eye = np.eye(h0.shape[-1], dtype=complex)
    return np.linalg.inv(
        (1j * grid.omega[:, None, None, None, None] + mu)
        * eye[None, None, None]
        - h0[None]
    )


def _small_background():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=7, nOmega=2, T=.19)
    h = np.array([[.16, .21 + .03j], [.21 - .03j, -.11]], dtype=complex)
    h0 = h[None, None]
    mu = .027
    G = _free_G(h0, mu, grid)
    Vq = np.array([[[[0.0, .18], [.18, 0.0]]]], dtype=complex)
    P = compute_polarization_matrix(G, grid, backend="direct")
    W = compute_screened_interaction_matrix(P, Vq)
    ref = build_tail_reference(h0, mu, np.zeros((2, 2)), grid)
    return grid, h0, mu, G, Vq, W, ref


def test_transfer_q0_m0_reduces_to_existing_production_vertex():
    grid, h0, mu, G, Vq, W, ref = _small_background()
    K = np.array([[.31, .04j], [-.04j, -.22]], dtype=complex)
    opts = SupercellVertexOptions(
        max_iter=80,
        tol=2e-10,
        solver="gmres",
        gmres_restart=10,
        verbose=False,
        momentum_backend="direct",
    )
    old = solve_vertex_q0_tail(G, W, Vq, K, grid, ref, opts=opts)
    new = solve_vertex_transfer_tail(
        G, W, Vq, K, (0, 0), 0, grid, ref, opts=opts
    )
    assert old.converged and new.converged
    np.testing.assert_allclose(new.Gamma, old.Gamma, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(new.Gamma_H, old.Gamma_H, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(new.Gamma_F, old.Gamma_F, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(new.Gamma_MT, old.Gamma_MT, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(new.Gamma_AL1, old.Gamma_AL1, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(new.Gamma_AL2, old.Gamma_AL2, rtol=2e-9, atol=2e-10)


def test_transfer_free_dynamic_tail_matches_analytic_reference_response():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=4, nOmega=2, T=.23)
    h = np.array([[.12, .17], [.17, -.15]], dtype=complex)
    h0 = h[None, None]
    mu = -.019
    G = _free_G(h0, mu, grid)
    Vq = np.zeros_like(h0)
    W = np.zeros((grid.nb, 1, 1, 2, 2), dtype=complex)
    ref = build_tail_reference(h0, mu, np.zeros((2, 2)), grid)
    Kleft = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    Kright = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    m_ext = 1
    opts = SupercellVertexOptions(
        max_iter=8,
        tol=1e-12,
        solver="gmres",
        verbose=False,
        momentum_backend="direct",
    )
    vr = solve_vertex_transfer_tail(
        G, W, Vq, Kright, (0, 0), m_ext, grid, ref, opts=opts
    )
    assert vr.converged
    resp = transfer_response_tail_completed(
        G,
        Kleft[None],
        [vr.Gamma],
        (0, 0),
        m_ext,
        grid,
        h0,
        mu,
        edge_points=1,
    )["completed"][0, 0]
    exact_R = reference_response_infinite(
        ref, Kright, grid, q_index=(0, 0), m_ext=m_ext
    )
    expected = -np.einsum("ij,xyji->", Kleft, exact_R) / grid.nk
    np.testing.assert_allclose(resp, expected, rtol=2e-11, atol=2e-12)


def test_sox_transfer_q0_m0_reduces_to_static_sox_vertex():
    grid, h0, mu, G, Vq, W, ref = _small_background()
    K = np.array([[.28, .03], [.03, -.17]], dtype=complex)
    Gamma = np.broadcast_to(K, G.shape).copy()
    opts = SOXOptions(n_quad=16, tail_complete=True, tail_edge_points=1)
    old = compute_sox_vertex_periodic(
        G, Gamma, Vq, h0, mu, grid, opts=opts, gamma_static=K
    )
    new = compute_sox_vertex_transfer_periodic(
        G, Gamma, Vq, h0, mu, (0, 0), 0, grid, opts=opts
    )
    np.testing.assert_allclose(new, old, rtol=4e-9, atol=5e-10)


def test_sox_transfer_endpoint_phases_match_finite_difference_of_skeleton():
    rng = np.random.default_rng(7)
    n = 4
    Gp = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Gm = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Xp = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Xm = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    v = rng.normal(size=(n, n))
    v = 0.5 * (v + v.T)
    np.fill_diagonal(v, 0.0)
    d = np.exp(1j * np.array([0.0, .4, 1.1, 1.7]))
    Omega = .83
    tau = .37
    analytic = sox_vertex_transfer_full(
        Gp, Gm, Xp, Xm, v, d, Omega, tau, interaction_tol=0.0
    )

    eps = 2e-7
    phase_t = np.exp(1j * Omega * tau)
    def sigma(sign):
        e = sign * eps
        gp = Gp + e * d[:, None] * Xp
        gm = Gm + e * phase_t * d[:, None] * Xm
        return sox_self_energy_full(gp, gm, v, interaction_tol=0.0)

    numeric_lab = (sigma(+1.0) - sigma(-1.0)) / (2.0 * eps)
    # Remove the output source phase D_i to compare with the transfer field.
    numeric = d.conj()[:, None] * numeric_lab
    np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=3e-7)


def test_post_screening_uses_v_minus_v_chi_v_and_window_fallback():
    Vq = np.array([[[[0.0, .4], [.4, 0.0]]]], dtype=complex)
    chi = np.zeros((3, 1, 1, 2, 2), dtype=complex)
    chi[0, 0, 0] = np.array([[.7, .1], [.1, .5]])
    chi[1, 0, 0] = np.nan
    chi[2, 0, 0] = np.array([[.3, -.04], [-.04, .2]])
    bg = np.broadcast_to(Vq[None], chi.shape).copy()
    bg[1, 0, 0] *= 1.3
    Wpost, fallback = build_post_screened_interaction(
        Vq, chi, background_W=bg
    )
    for im in (0, 2):
        v = Vq[0, 0]
        expected = v - v @ chi[im, 0, 0] @ v
        np.testing.assert_allclose(Wpost[im, 0, 0], expected)
    np.testing.assert_allclose(Wpost[1, 0, 0], bg[1, 0, 0])
    assert fallback.tolist() == [[[False]], [[True]], [[False]]]


def test_post_gw_zero_interaction_is_identity_for_gw_and_gw_sox():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=5, nOmega=1, T=.21)
    h = np.array([[.13, .16], [.16, -.09]], dtype=complex)
    h0 = h[None, None]
    Vq = np.zeros_like(h0)
    opts = GWOptions(
        mu=.017,
        target_filling=None,
        max_iter=4,
        tol=1e-11,
        verbose=False,
        momentum_backend="direct",
    )
    vopts = SupercellVertexOptions(
        max_iter=8,
        tol=1e-11,
        solver="gmres",
        verbose=False,
        momentum_backend="direct",
    )
    sopts = SOXOptions(n_quad=16, tail_edge_points=1)
    gw = solve_matrix_gw_fast(h0, Vq, grid, opts=opts)
    gwsox = solve_matrix_gw_sox(h0, Vq, grid, opts=opts, sox_opts=sopts)
    post = run_post_gw(
        gw, h0, Vq, grid, gw_opts=opts, vertex_opts=vopts,
        include_sox=False, sox_opts=sopts, m_max=0
    )
    post_sox = run_post_gw(
        gwsox, h0, Vq, grid, gw_opts=opts, vertex_opts=vopts,
        include_sox=True, sox_opts=sopts, m_max=0
    )
    np.testing.assert_allclose(post.G, gw.G, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(post_sox.G, gwsox.G, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(post.W_post, 0.0, atol=1e-14)
    np.testing.assert_allclose(post_sox.W_post, 0.0, atol=1e-14)


def test_benchmark_plotter_writes_response_figures(tmp_path):
    V = np.array([.1, .2])
    channels = np.array(["x_even", "z_same"])
    shape = (2, 2, 2)
    ed = np.zeros(shape)
    gg = np.zeros(shape, dtype=complex)
    cgw = np.zeros(shape, dtype=complex)
    sox = np.zeros(shape, dtype=complex)
    for iv in range(2):
        for ic in range(2):
            ed[iv, ic, ic] = 1.0 + iv + .1 * ic
            gg[iv, ic, ic] = .9 * ed[iv, ic, ic]
            cgw[iv, ic, ic] = 1.05 * ed[iv, ic, ic]
            sox[iv, ic, ic] = 1.01 * ed[iv, ic, ic]
    path = tmp_path / "benchmark.npz"
    np.savez_compressed(
        path,
        V=V,
        channels=channels,
        ed=ed,
        gg_completed=gg,
        cgw_completed=cgw,
        cgw_sox_completed=sox,
    )
    made = plot_benchmark_npz(path)
    names = {p.name for p in made}
    assert "benchmark_x_even.png" in names
    assert "benchmark_z_same.png" in names
    assert "benchmark_response_summary.png" in names
    assert "benchmark_response_relative_error.png" in names
    assert all(p.exists() for p in made)
