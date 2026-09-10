import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.ssosex_covariant_fd import build_static_post_w_same_torus


def test_build_static_post_w_same_torus_changes_only_zero_bosonic_sector():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=6, nOmega=2, T=0.2)
    v = np.array(
        [[0.0, 0.7, 0.2], [0.7, 0.0, 0.4], [0.2, 0.4, 0.0]],
        dtype=complex,
    )
    Vq = v[None, None]
    Wbg = np.empty((grid.nb, 1, 1, 3, 3), dtype=complex)
    for im, m in enumerate(grid.m_values):
        Wbg[im, 0, 0] = (1.0 + 0.03 * float(m)) * v
    chi = np.array(
        [[0.31, -0.05, 0.02], [-0.05, 0.27, 0.01], [0.02, 0.01, 0.22]],
        dtype=complex,
    )

    got = build_static_post_w_same_torus(Vq, Wbg, chi, grid)
    im0 = int(np.flatnonzero(np.asarray(grid.m_values) == 0)[0])
    np.testing.assert_allclose(got[im0, 0, 0], v - v @ chi @ v)
    for im in range(grid.nb):
        if im != im0:
            np.testing.assert_allclose(got[im], Wbg[im], rtol=0.0, atol=0.0)


def test_build_static_post_w_zero_interaction_is_zero_at_m0():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=4, nOmega=1, T=0.3)
    Vq = np.zeros((1, 1, 2, 2), dtype=complex)
    Wbg = np.zeros((grid.nb, 1, 1, 2, 2), dtype=complex)
    chi = np.array([[1.2, 0.1], [0.1, 0.9]], dtype=complex)
    got = build_static_post_w_same_torus(Vq, Wbg, chi, grid)
    np.testing.assert_allclose(got, 0.0, rtol=0.0, atol=0.0)
