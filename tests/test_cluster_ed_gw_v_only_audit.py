"""Regression tests for the triangle-V Fock and SEET-like DC bookkeeping."""
import numpy as np

from rubycgw.cluster_ed_gw import (
    audit_cluster_gw_decomposition,
    cluster_interaction_matrix,
    ruby_cluster_interactions,
    build_intracell_h0,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_interaction
from rubycgw.supercell_gw_split import compute_static_fock_matrix


def test_triangle_v_fock_matches_local_density_with_complex_bonds():
    # V(q) is independent of q because all six interaction bonds have R=0.
    p = RubyParameters(V=0.61)
    grid = MatsubaraGrid(nk1=3, nk2=2, nw=3, nOmega=2, T=0.22)
    Vq = build_interaction(grid.qmesh(), p)
    assert np.allclose(Vq, Vq[0, 0][None, None])

    rng = np.random.default_rng(20261111)
    rho_k = np.empty((grid.nk1, grid.nk2, 6, 6), dtype=complex)
    for ix in range(grid.nk1):
        for iy in range(grid.nk2):
            A = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
            rho_k[ix, iy] = 0.24 * np.eye(6) + 0.03 * (A + A.conj().T)

    f_direct = compute_static_fock_matrix(rho_k, Vq, grid, backend="direct")
    f_fft = compute_static_fock_matrix(rho_k, Vq, grid, backend="fft")
    rho_local = np.mean(rho_k, axis=(0, 1))
    expected = -rho_local * Vq[0, 0].T  # Sigma_F,ab=-V_ba <c_b^dagger c_a>
    assert np.max(np.abs(f_direct - expected[None, None])) < 1e-12
    assert np.max(np.abs(f_fft - f_direct)) < 1e-12
    # No V interaction between the A and B triangles.
    assert np.max(np.abs(expected[:3, 3:])) < 1e-13
    assert np.max(np.abs(expected[3:, :3])) < 1e-13


def test_cluster_v_only_dc_is_hartree_fock_plus_w_minus_v():
    p = RubyParameters(V=0.12, t1=0.2, t2=0.2)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=4, nOmega=2, T=0.25)
    Vc = cluster_interaction_matrix(ruby_cluster_interactions(p))
    h = build_intracell_h0(p)
    rho = 0.30 * np.eye(6, dtype=complex)
    rho[0, 1] = 0.04 + 0.03j
    rho[1, 0] = rho[0, 1].conjugate()
    rho[3, 4] = -0.02 + 0.01j
    rho[4, 3] = rho[3, 4].conjugate()
    Gc = np.asarray([
        np.linalg.inv((1j * w + 0.08) * np.eye(6) - h)
        for w in grid.omega
    ])
    report = audit_cluster_gw_decomposition(Gc, rho, Vc, grid)
    assert report["fock_error"] < 1e-12
    assert report["reconstruction_error"] < 1e-12
    assert np.max(np.abs(
        report["hartree"][~np.eye(6, dtype=bool)]
    )) < 1e-13
    expected = (
        report["hartree"][None]
        + report["fock"][None]
        + report["correlation"]
    )
    assert np.allclose(report["total"], expected)


def test_cluster_dc_uses_only_original_v_bonds():
    params = RubyParameters(V=1.0)
    Vc = cluster_interaction_matrix(ruby_cluster_interactions(params))
    expected = np.zeros((6, 6), dtype=complex)
    for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
        expected[i, j] = expected[j, i] = 1.0
    assert np.array_equal(Vc, expected)
