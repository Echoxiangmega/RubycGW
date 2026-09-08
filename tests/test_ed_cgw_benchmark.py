import numpy as np

from rubycgw.ed_cgw_benchmark import (
    bubble_iomega,
    complex_q_projection,
    project_complex_q,
    static_response_from_gammas,
)
from rubycgw.grids import MatsubaraGrid


def test_bubble_zero_frequency_matches_static_bare_response():
    rng = np.random.default_rng(7)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=2, T=0.2)
    norb = 3
    G = rng.normal(size=(grid.nf, 1, 1, norb, norb)) + 1j * rng.normal(
        size=(grid.nf, 1, 1, norb, norb)
    )
    K = rng.normal(size=(2, norb, norb)) + 1j * rng.normal(size=(2, norb, norb))
    bare = [np.broadcast_to(x, G.shape).copy() for x in K]

    static = static_response_from_gammas(G, K, bare, grid)
    dynamic = bubble_iomega(G, K, grid)
    izero = int(np.where(grid.m_values == 0)[0][0])
    assert np.max(np.abs(static - dynamic[izero])) < 1e-12


def test_complex_q_projection_matches_operator_coefficients():
    rng = np.random.default_rng(11)
    n = 3
    A = rng.normal(size=(2 * n, 2 * n)) + 1j * rng.normal(size=(2 * n, 2 * n))
    M = A + A.conj().T
    C = complex_q_projection(n)
    got = project_complex_q(M, n)
    expected = C.conj().T @ M @ C
    assert np.max(np.abs(got - expected)) < 1e-12
    assert np.max(np.abs(got - got.conj().T)) < 1e-12
