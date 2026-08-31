import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.model import eta_vertices
from rubycgw.supercell_cgw import (
    SupercellVertexOptions,
    current_harmonic_transform,
    sector_harmonic_matrix,
    solve_vertex_q0,
    supercell_current_vertices,
    susceptibility_matrix_q0,
    vertex_corrections_q0,
)


def test_supercell_current_uniform_normalization_matches_primitive_vertex():
    vertices, labels = supercell_current_vertices()
    _, _, kp, km = eta_vertices()
    assert labels == [
        "opposite_s0", "opposite_s1", "opposite_s2",
        "same_s0", "same_s1", "same_s2",
    ]
    Kop = np.sum(vertices[:3], axis=0) / np.sqrt(3.0)
    Ksame = np.sum(vertices[3:], axis=0) / np.sqrt(3.0)
    for s in range(3):
        sl = slice(6 * s, 6 * (s + 1))
        assert np.max(np.abs(Kop[sl, sl] - kp / np.sqrt(3.0))) < 1e-14
        assert np.max(np.abs(Ksame[sl, sl] - km / np.sqrt(3.0))) < 1e-14


def test_sector_harmonics_are_orthonormal():
    u = sector_harmonic_matrix()
    assert np.max(np.abs(u @ u.T - np.eye(3))) < 1e-14
    T, labels = current_harmonic_transform()
    assert np.max(np.abs(T @ T.T - np.eye(6))) < 1e-14
    assert labels[0] == "opposite_q0"
    assert labels[3] == "same_q0"


def test_split_vertex_direct_and_fft_corrections_match():
    rng = np.random.default_rng(2026)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.1)
    norb = 3
    shape_g = (grid.nf, grid.nk1, grid.nk2, norb, norb)
    shape_w = (grid.nb, grid.nk1, grid.nk2, norb, norb)
    shape_v = (grid.nk1, grid.nk2, norb, norb)

    G = 0.1 * (rng.normal(size=shape_g) + 1j * rng.normal(size=shape_g))
    Gamma = 0.1 * (rng.normal(size=shape_g) + 1j * rng.normal(size=shape_g))
    Vq = 0.1 * (rng.normal(size=shape_v) + 1j * rng.normal(size=shape_v))
    W = 0.1 * (rng.normal(size=shape_w) + 1j * rng.normal(size=shape_w))

    od = SupercellVertexOptions(verbose=False, momentum_backend="direct")
    of = SupercellVertexOptions(verbose=False, momentum_backend="fft")
    direct = vertex_corrections_q0(G, W, Vq, Gamma, grid, od)
    fft = vertex_corrections_q0(G, W, Vq, Gamma, grid, of)
    for a, b in zip(direct, fft):
        assert np.max(np.abs(a - b)) < 1e-11


def test_v_zero_vertex_reduces_to_bare_source():
    rng = np.random.default_rng(11)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=2, nOmega=1, T=0.1)
    norb = 4
    G = 0.1 * (
        rng.normal(size=(grid.nf, 1, 1, norb, norb))
        + 1j * rng.normal(size=(grid.nf, 1, 1, norb, norb))
    )
    W = np.zeros((grid.nb, 1, 1, norb, norb), dtype=complex)
    Vq = np.zeros((1, 1, norb, norb), dtype=complex)
    K = np.zeros((norb, norb), dtype=complex)
    K[0, 1] = 1j
    K[1, 0] = -1j

    opts = SupercellVertexOptions(max_iter=3, tol=1e-13, mixing=0.4, verbose=False)
    result = solve_vertex_q0(G, W, Vq, K, grid, opts)
    expected = np.broadcast_to(K, G.shape)
    assert result.converged
    assert np.max(np.abs(result.Gamma - expected)) < 1e-13

    chi = susceptibility_matrix_q0(G, np.stack([K]), [result.Gamma], grid)
    assert chi.shape == (1, 1)
