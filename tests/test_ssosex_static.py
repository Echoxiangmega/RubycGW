import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.sox_covariant import SOXOptions
from rubycgw.sox_fast import compute_sox_self_energy_periodic_fast
from rubycgw.ssosex_static import (
    ScreenedSOSEXOptions,
    compute_static_screened_sosex_self_energy_periodic_fast,
    static_W0,
)


def _fixture():
    rng = np.random.default_rng(701)
    nk1, nk2, norb = 2, 1, 3
    grid = MatsubaraGrid(nk1=nk1, nk2=nk2, nw=8, nOmega=2, T=0.21)
    raw = rng.normal(size=(nk1, nk2, norb, norb)) + 1j * rng.normal(
        size=(nk1, nk2, norb, norb)
    )
    h = 0.13 * (raw + np.swapaxes(raw.conj(), -1, -2))
    mu = -0.031
    eye = np.eye(norb, dtype=complex)
    G = np.linalg.inv(
        (1j * np.asarray(grid.omega)[:, None, None, None, None] + mu)
        * eye[None, None, None]
        - h[None]
    )
    v0 = np.array(
        [[0.0, 0.62, 0.31], [0.62, 0.0, 0.23], [0.31, 0.23, 0.0]],
        dtype=complex,
    )
    Vq = np.empty((nk1, nk2, norb, norb), dtype=complex)
    Vq[0, 0] = v0
    Vq[1, 0] = 0.71 * v0
    return grid, h, mu, G, Vq


def test_static_W0_extracts_zero_bosonic_sector():
    grid, _, _, _, Vq = _fixture()
    W = np.empty((grid.nb,) + Vq.shape, dtype=complex)
    for im, m in enumerate(grid.m_values):
        W[im] = (1.0 + 0.1 * float(m)) * Vq
    got = static_W0(W, grid)
    np.testing.assert_allclose(got, Vq, rtol=0.0, atol=0.0)


def test_screened_sosex_reduces_exactly_to_bare_sox_when_W0_equals_V():
    grid, h, mu, G, Vq = _fixture()
    W = np.broadcast_to(Vq, (grid.nb,) + Vq.shape).copy()
    bare = compute_sox_self_energy_periodic_fast(
        G,
        Vq,
        h,
        mu,
        grid,
        opts=SOXOptions(n_quad=16, tail_complete=True),
    )
    for mode in ("oneW-sym", "twoW"):
        screened = compute_static_screened_sosex_self_energy_periodic_fast(
            G,
            Vq,
            W,
            h,
            mu,
            grid,
            opts=ScreenedSOSEXOptions(n_quad=16, tail_complete=True, mode=mode),
        )
        np.testing.assert_allclose(screened, bare, rtol=3e-12, atol=3e-12)


def test_screened_sosex_has_expected_bilinear_scaling_for_W0_proportional_V():
    grid, h, mu, G, Vq = _fixture()
    c = 0.63
    W = np.broadcast_to(c * Vq, (grid.nb,) + Vq.shape).copy()
    bare = compute_sox_self_energy_periodic_fast(
        G,
        Vq,
        h,
        mu,
        grid,
        opts=SOXOptions(n_quad=16, tail_complete=True),
    )
    one = compute_static_screened_sosex_self_energy_periodic_fast(
        G,
        Vq,
        W,
        h,
        mu,
        grid,
        opts=ScreenedSOSEXOptions(n_quad=16, tail_complete=True, mode="oneW-sym"),
    )
    two = compute_static_screened_sosex_self_energy_periodic_fast(
        G,
        Vq,
        W,
        h,
        mu,
        grid,
        opts=ScreenedSOSEXOptions(n_quad=16, tail_complete=True, mode="twoW"),
    )
    np.testing.assert_allclose(one, c * bare, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(two, c * c * bare, rtol=3e-12, atol=3e-12)
