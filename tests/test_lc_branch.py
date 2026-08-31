import numpy as np

from rubycgw.checkpoint import GWCheckpointSeed
from rubycgw.grids import MatsubaraGrid
from rubycgw.lc_branch import (
    add_current_source,
    current_vertex_q0,
    primitive_translation_a1_matrix,
    project_primitive_translation_invariant,
    remove_charge_order_from_seed,
)
from rubycgw.model import RubyParameters, eta_vertices
from rubycgw.supercell import NSUP, build_supercell_h0, period3_real_pattern


def test_primitive_translation_projector_leaves_bare_h0_invariant():
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=1.0)
    for k in [np.array([0.13, 0.27]), np.array([0.31, 0.17])]:
        h = build_supercell_h0(k[None, :], params)[0]
        projected = project_primitive_translation_invariant(h, k)
        assert np.max(np.abs(projected - h)) < 1e-12


def test_translation_matrix_is_unitary_and_cubic_up_to_bloch_phase():
    k = np.array([0.17, 0.23])
    U = primitive_translation_a1_matrix(k)
    eye = np.eye(NSUP)
    assert np.max(np.abs(U @ U.conj().T - eye)) < 1e-12
    phase = np.exp(-2j * np.pi * np.dot(k, np.array([2.0, 1.0])))
    assert np.max(np.abs(U @ U @ U - phase * eye)) < 1e-12


def test_remove_charge_order_from_seed_equalizes_hartree_sectors():
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=2, nOmega=1, T=0.1)
    pattern = period3_real_pattern()
    sigma_h = np.diag(1.0 + 0.4 * pattern).astype(complex)
    sigma_gw = np.zeros((grid.nf, grid.nk1, grid.nk2, NSUP, NSUP), dtype=complex)
    seed = GWCheckpointSeed(Sigma_H=sigma_h, Sigma_GW=sigma_gw, mu=0.5)
    projected = remove_charge_order_from_seed(seed, grid)

    diag = np.diag(projected.Sigma_H).real.reshape(3, 6)
    assert np.max(np.abs(diag - diag.mean(axis=0, keepdims=True))) < 1e-12
    assert projected.mu == seed.mu


def test_uniform_current_vertex_matches_project_conventions():
    _, _, kp, km = eta_vertices()
    Kop = current_vertex_q0("opposite")
    Ksame = current_vertex_q0("same")
    norm = np.sqrt(3.0)
    for s in range(3):
        sl = slice(6 * s, 6 * (s + 1))
        assert np.max(np.abs(Kop[sl, sl] - kp / norm)) < 1e-12
        assert np.max(np.abs(Ksame[sl, sl] - km / norm)) < 1e-12
    assert np.max(np.abs(Kop - Kop.conj().T)) < 1e-12
    assert np.max(np.abs(Ksame - Ksame.conj().T)) < 1e-12


def test_current_source_is_hermitian_and_linear_in_strength():
    params = RubyParameters()
    kpts = np.array([[0.12, 0.21], [0.33, 0.07]])
    h0 = build_supercell_h0(kpts, params)
    h = 0.08
    sourced = add_current_source(h0, h, "same")
    K = current_vertex_q0("same")
    assert np.max(np.abs(sourced - np.swapaxes(sourced.conj(), -1, -2))) < 1e-12
    assert np.max(np.abs((sourced - h0) + h * K)) < 1e-12
