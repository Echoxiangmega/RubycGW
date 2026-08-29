import numpy as np

from rubycgw.model import RubyParameters, build_h0, build_interaction, eta_vertices
from rubycgw.grids import MatsubaraGrid, shift_fermion_field
from rubycgw.gw import build_g0_inverse
from rubycgw.susceptibility import chi_eta


def test_h0_hermitian():
    params = RubyParameters()
    k = np.array([[0.137, 0.291], [0.0, 0.0]])
    h = build_h0(k, params)
    assert np.max(np.abs(h - np.swapaxes(h.conj(), -1, -2))) < 1e-13


def test_interaction_hermitian():
    params = RubyParameters(V=0.37)
    q = np.array([[0.13, 0.29], [0.0, 0.0]])
    v = build_interaction(q, params)
    assert np.max(np.abs(v - np.swapaxes(v.conj(), -1, -2))) < 1e-13


def test_eta_vertices_hermitian_and_labels():
    ka, kb, kp, km = eta_vertices()
    for k in [ka, kb, kp, km]:
        assert np.max(np.abs(k - k.conj().T)) < 1e-13
    assert np.allclose(kp, (ka + kb) / np.sqrt(2.0))
    assert np.allclose(km, (ka - kb) / np.sqrt(2.0))


def test_shift_rule_k_to_k_plus_q():
    f = np.zeros((5, 3, 4, 1, 1), dtype=float)
    for n in range(5):
        for i in range(3):
            for j in range(4):
                f[n, i, j, 0, 0] = 100*n + 10*i + j
    s = shift_fermion_field(f, 1, 2, 1)
    assert s[0, 0, 0, 0, 0] == f[1, 1, 2, 0, 0]


def test_v_zero_bare_susceptibility_is_finite():
    params = RubyParameters(V=0.0)
    grid = MatsubaraGrid(nk1=2, nk2=2, nw=8, nOmega=2, T=0.1)
    h0 = build_h0(grid.kmesh(), params)
    G0 = np.linalg.inv(build_g0_inverse(h0, grid, mu=0.0))
    _, _, kp, km = eta_vertices()
    cp = chi_eta(G0, kp, grid)
    cm = chi_eta(G0, km, grid)
    assert np.isfinite(cp.real) and np.isfinite(cp.imag)
    assert np.isfinite(cm.real) and np.isfinite(cm.imag)
