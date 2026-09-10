import numpy as np

from rubycgw.dynamic_sosex import (
    DynamicSOSEXOptions,
    _double_screened_contraction,
    _reference_pair_kernel,
    compute_dynamic_sosex_self_energy_periodic_fast,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.sox_covariant import SOXOptions
from rubycgw.sox_fast import compute_sox_self_energy_periodic_fast


def test_reference_pair_kernel_matches_large_matsubara_sum():
    T = 0.17
    xi = np.array([-0.41, 0.13, 0.72])
    occ = 1.0 / (np.exp(xi / T) + 1.0)
    for m in (0, 1, -2):
        exact = _reference_pair_kernel(xi, occ, m, T)
        n = np.arange(-20000, 20000)
        w = (2 * n + 1) * np.pi * T
        Om = 2 * np.pi * T * m
        brute = np.empty_like(exact)
        for p in range(len(xi)):
            for q in range(len(xi)):
                brute[p, q] = T * np.sum(
                    1.0 / ((1j*w + 1j*Om - xi[p]) * (1j*w - xi[q]))
                )
        np.testing.assert_allclose(exact, brute, rtol=2e-5, atol=2e-5)


def _fixture(norb=3):
    rng = np.random.default_rng(913)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=8, nOmega=2, T=0.19)
    raw = rng.normal(size=(norb, norb)) + 1j*rng.normal(size=(norb, norb))
    h = 0.15 * (raw + raw.conj().T)
    mu = -0.07
    eye = np.eye(norb, dtype=complex)
    G = np.linalg.inv(
        (1j*grid.omega[:, None, None] + mu)*eye[None] - h[None]
    )[:, None, None]
    V = np.ones((norb, norb), dtype=complex) - np.eye(norb)
    V *= 0.4
    return grid, h, mu, G, V[None, None]


def test_double_screened_fast_contraction_matches_direct_einsum():
    rng = np.random.default_rng(117)
    nf, nsite = 5, 4
    def c(shape):
        return rng.normal(size=shape) + 1j*rng.normal(size=shape)
    Wm = c((nsite, nsite))
    Wp = c((nsite, nsite))
    Gm = c((nf, nsite, nsite))
    Gmm = c((nf, nsite, nsite))
    Gp = c((nf, nsite, nsite))
    fast = _double_screened_contraction(Wm, Wp, Gm, Gmm, Gp)
    direct = np.einsum(
        "il,kj,nik,nkl,nlj->nij", Wm, Wp, Gm, Gmm, Gp, optimize=True
    )
    np.testing.assert_allclose(fast, direct, rtol=2e-13, atol=2e-13)


def test_all_dynamic_modes_reduce_to_bare_sox_when_W_equals_V():
    grid, h, mu, G, Vq = _fixture()
    href = h[None, None]
    W = np.broadcast_to(Vq, (grid.nb,) + Vq.shape).copy()
    bare = compute_sox_self_energy_periodic_fast(
        G, Vq, href, mu, grid,
        opts=SOXOptions(n_quad=32, tail_complete=True),
    )
    for mode in ("sosex", "2sosex", "g3w2"):
        dyn = compute_dynamic_sosex_self_energy_periodic_fast(
            G, Vq, W, href, mu, grid,
            opts=DynamicSOSEXOptions(n_quad=32, max_full_sites=8, mode=mode),
        )
        np.testing.assert_allclose(dyn, bare, rtol=2e-12, atol=2e-12)


def _screening_fixture():
    grid, h, mu, G, Vq = _fixture(norb=2)
    href = h[None, None]
    shape = (grid.nb,) + Vq.shape
    Wp = np.empty(shape, dtype=complex)
    M = np.array([[0.3, 0.2], [0.2, -0.1]], dtype=complex)
    for im, m in enumerate(grid.m_values):
        Wp[im, 0, 0] = (0.08/(1 + m*m))*M
    return grid, href, mu, G, Vq, Wp


def test_dynamic_one_screened_piece_is_linear_in_W_minus_V():
    grid, href, mu, G, Vq, Wp = _screening_fixture()
    shape = (grid.nb,) + Vq.shape

    def calc(scale):
        W = np.broadcast_to(Vq, shape).copy() + scale*Wp
        return compute_dynamic_sosex_self_energy_periodic_fast(
            G, Vq, W, href, mu, grid,
            opts=DynamicSOSEXOptions(n_quad=32, max_full_sites=8, mode="sosex"),
            return_parts=True,
        )

    p1 = calc(1.0)
    p2 = calc(0.37)
    np.testing.assert_allclose(
        p2.Sigma - p2.Sigma_SOX,
        0.37*(p1.Sigma - p1.Sigma_SOX),
        rtol=2e-11, atol=2e-11,
    )


def test_g3w2_double_screened_piece_is_quadratic_in_W_minus_V():
    grid, href, mu, G, Vq, Wp = _screening_fixture()
    shape = (grid.nb,) + Vq.shape

    def calc(scale):
        W = np.broadcast_to(Vq, shape).copy() + scale*Wp
        return compute_dynamic_sosex_self_energy_periodic_fast(
            G, Vq, W, href, mu, grid,
            opts=DynamicSOSEXOptions(n_quad=32, max_full_sites=8, mode="g3w2"),
            return_parts=True,
        )

    p1 = calc(1.0)
    p2 = calc(0.37)
    np.testing.assert_allclose(
        p2.Sigma_WpV, 0.37*p1.Sigma_WpV, rtol=3e-11, atol=3e-11
    )
    np.testing.assert_allclose(
        p2.Sigma_VWp, 0.37*p1.Sigma_VWp, rtol=3e-11, atol=3e-11
    )
    np.testing.assert_allclose(
        p2.Sigma_WpWp, (0.37**2)*p1.Sigma_WpWp, rtol=3e-11, atol=3e-11
    )
    np.testing.assert_allclose(
        p1.Sigma,
        p1.Sigma_SOX + p1.Sigma_WpV + p1.Sigma_VWp + p1.Sigma_WpWp,
        rtol=2e-13, atol=2e-13,
    )
