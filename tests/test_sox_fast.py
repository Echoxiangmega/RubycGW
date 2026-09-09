import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.sox_covariant import (
    SOXOptions,
    compute_sox_self_energy_periodic,
    sox_self_energy_full,
)
from rubycgw.sox_fast import (
    _interaction_neighbor_tables,
    _sox_self_energy_sparse_batch,
    compute_sox_self_energy_periodic_fast,
)


def test_sparse_batch_sox_matches_reference_full_kernel():
    rng = np.random.default_rng(401)
    n = 9
    nbatch = 4
    gp = rng.normal(size=(nbatch, n, n)) + 1j * rng.normal(size=(nbatch, n, n))
    gm = rng.normal(size=(nbatch, n, n)) + 1j * rng.normal(size=(nbatch, n, n))
    v = rng.normal(size=(n, n))
    v[np.abs(v) < 0.9] = 0.0
    v = 0.5 * (v + v.T)
    np.fill_diagonal(v, 0.0)

    row_idx, row_w, col_idx, col_w = _interaction_neighbor_tables(v, 1.0e-13)
    fast = _sox_self_energy_sparse_batch(
        gp, gm, row_idx, row_w, col_idx, col_w
    )
    ref = np.stack([sox_self_energy_full(gp[t], gm[t], v) for t in range(nbatch)])
    np.testing.assert_allclose(fast, ref, rtol=3e-13, atol=3e-13)


def test_periodic_fast_sox_matches_reference_with_tail_completion():
    rng = np.random.default_rng(402)
    nk1, nk2, norb = 2, 1, 3
    grid = MatsubaraGrid(nk1=nk1, nk2=nk2, nw=12, nOmega=2, T=0.19)

    raw = rng.normal(size=(nk1, nk2, norb, norb)) + 1j * rng.normal(
        size=(nk1, nk2, norb, norb)
    )
    h = 0.18 * (raw + np.swapaxes(raw.conj(), -1, -2))
    mu = 0.041
    eye = np.eye(norb, dtype=complex)
    G = np.linalg.inv(
        (1j * np.asarray(grid.omega)[:, None, None, None, None] + mu)
        * eye[None, None, None]
        - h[None]
    )

    v0 = np.array(
        [[0.0, 0.7, 0.4], [0.7, 0.0, 0.25], [0.4, 0.25, 0.0]],
        dtype=complex,
    )
    # q-dependent Hermitian interaction to exercise the periodic transforms.
    Vq = np.empty((nk1, nk2, norb, norb), dtype=complex)
    Vq[0, 0] = v0
    Vq[1, 0] = 0.65 * v0

    opts = SOXOptions(n_quad=16, tail_complete=True, interaction_tol=1e-13)
    ref = compute_sox_self_energy_periodic(G, Vq, h, mu, grid, opts=opts)
    fast = compute_sox_self_energy_periodic_fast(G, Vq, h, mu, grid, opts=opts)
    np.testing.assert_allclose(fast, ref, rtol=2e-12, atol=2e-12)


def test_periodic_fast_sox_matches_reference_without_tail_completion():
    rng = np.random.default_rng(403)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=8, nOmega=2, T=0.23)
    norb = 2
    raw = rng.normal(size=(2, 2, norb, norb)) + 1j * rng.normal(
        size=(2, 2, norb, norb)
    )
    h = 0.12 * (raw + np.swapaxes(raw.conj(), -1, -2))
    mu = -0.017
    eye = np.eye(norb, dtype=complex)
    G = np.linalg.inv(
        (1j * np.asarray(grid.omega)[:, None, None, None, None] + mu)
        * eye[None, None, None]
        - h[None]
    )
    Vq = np.zeros((2, 2, norb, norb), dtype=complex)
    Vq[..., 0, 1] = 0.55
    Vq[..., 1, 0] = 0.55
    opts = SOXOptions(n_quad=16, tail_complete=False)
    ref = compute_sox_self_energy_periodic(G, Vq, h, mu, grid, opts=opts)
    fast = compute_sox_self_energy_periodic_fast(G, Vq, h, mu, grid, opts=opts)
    np.testing.assert_allclose(fast, ref, rtol=2e-12, atol=2e-12)
