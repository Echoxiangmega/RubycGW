import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.sox_covariant import (
    SOXOptions,
    compute_sox_self_energy_periodic,
    compute_sox_vertex_periodic,
    full_periodic_to_kfield,
    kfield_to_full_periodic,
    sox_self_energy_full,
    sox_vertex_full,
)
from rubycgw.sox_diagnostic import (
    free_green_iomega,
    solve_free_mu,
    sox_vertex_iomega_free,
)


def test_periodic_k_full_roundtrip():
    rng = np.random.default_rng(31)
    field = rng.normal(size=(2, 3, 4, 4)) + 1j * rng.normal(size=(2, 3, 4, 4))
    full = kfield_to_full_periodic(field)
    back = full_periodic_to_kfield(full, 2, 3, 4)
    np.testing.assert_allclose(back, field, rtol=2e-14, atol=2e-14)


def test_sparse_full_sox_matches_direct_contraction():
    rng = np.random.default_rng(32)
    n = 7
    gp = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    gm = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    xp = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    xm = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    v = rng.normal(size=(n, n))
    v[np.abs(v) < 0.7] = 0.0
    v = 0.5 * (v + v.T)
    np.fill_diagonal(v, 0.0)

    sigma_ref = np.einsum("il,kj,ik,kl,lj->ij", v, v, gp, gm, gp, optimize=True)
    vertex_ref = (
        np.einsum("il,kj,ik,kl,lj->ij", v, v, xp, gm, gp, optimize=True)
        + np.einsum("il,kj,ik,kl,lj->ij", v, v, gp, xm, gp, optimize=True)
        + np.einsum("il,kj,ik,kl,lj->ij", v, v, gp, gm, xp, optimize=True)
    )
    np.testing.assert_allclose(
        sox_self_energy_full(gp, gm, v), sigma_ref, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        sox_vertex_full(gp, gm, xp, xm, v),
        vertex_ref,
        rtol=2e-13,
        atol=2e-13,
    )


def _free_G(h, mu, grid):
    eye = np.eye(h.shape[-1], dtype=complex)
    return np.stack([
        np.linalg.inv((1j * w + mu) * eye - h) for w in grid.omega
    ])[:, None, None, :, :]


def test_periodic_sox_vertex_is_finite_difference_of_self_energy():
    h = np.array(
        [[0.1, 0.21 + 0.03j, 0.04],
         [0.21 - 0.03j, -0.18, 0.13j],
         [0.04, -0.13j, 0.29]],
        dtype=complex,
    )
    K = np.array(
        [[0.2, 0.05j, 0.03],
         [-0.05j, -0.1, 0.02j],
         [0.03, -0.02j, -0.1]],
        dtype=complex,
    )
    v = np.array([[0, .7, .4], [.7, 0, .2], [.4, .2, 0]], dtype=complex)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=36, nOmega=4, T=.21)
    mu = .037
    G = _free_G(h, mu, grid)
    Gamma = np.broadcast_to(K, G.shape).copy()
    options = SOXOptions(n_quad=64, tail_complete=True)
    analytic = compute_sox_vertex_periodic(
        G, Gamma, v[None, None], h[None, None], mu, grid, opts=options
    )

    eps = 2.0e-6
    hp = h + eps * K
    hm = h - eps * K
    Gp = _free_G(hp, mu, grid)
    Gm = _free_G(hm, mu, grid)
    sp = compute_sox_self_energy_periodic(
        Gp, v[None, None], hp[None, None], mu, grid, opts=options
    )
    sm = compute_sox_self_energy_periodic(
        Gm, v[None, None], hm[None, None], mu, grid, opts=options
    )
    numeric = (sp - sm) / (2.0 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=4e-6, atol=4e-7)


def test_general_periodic_kernel_reproduces_strict_ed12_free_background():
    exact = ExactSmallRubyThermal(
        2, 1, RubyParameters(V=0.0, ti=.4, t1=.2, t2=.2)
    )
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=28, nOmega=4, T=.08)
    mu = solve_free_mu(exact.h0, 6.0, grid.T)
    G = free_green_iomega(exact.h0, mu, grid)
    options = SOXOptions(n_quad=64, tail_complete=True)
    for ch in ("z_same", "z_opposite"):
        K = exact.pseudospin_operator(ch, (0.0, 0.0))
        Gamma = np.broadcast_to(K, G.shape).copy()
        general = compute_sox_vertex_periodic(
            G,
            Gamma,
            exact.Vunit[None, None],
            exact.h0[None, None],
            mu,
            grid,
            opts=options,
        )
        strict = sox_vertex_iomega_free(
            exact.h0, mu, exact.Vunit, K, grid, n_quad=64
        )
        np.testing.assert_allclose(general, strict, rtol=8e-8, atol=8e-9)
