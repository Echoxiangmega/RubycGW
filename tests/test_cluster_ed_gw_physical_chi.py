import numpy as np

from research.analyze_cluster_ed_gw_vprime_vcross_orientation_chi import (
    _eigen_block,
    _fixed_filling_schur,
    _hermitianize_pair,
)


def test_q_pair_hermitianization_obeys_static_reciprocity():
    chi_q = np.array(
        [[2.0 + 0.1j, 0.3 + 0.4j], [0.1 - 0.2j, 1.1 - 0.05j]],
        dtype=complex,
    )
    chi_mq = np.array(
        [[2.0 - 0.2j, 0.15 + 0.25j], [0.35 - 0.45j, 1.1 + 0.1j]],
        dtype=complex,
    )
    h = _hermitianize_pair(chi_q, chi_mq)
    # The q representative becomes Hermitian only after identifying the -q
    # response with its adjoint partner.
    href = 0.5 * (h + h.conj().T)
    assert np.max(np.abs(href - href.conj().T)) < 1e-14


def test_fixed_filling_schur_removes_uniform_density_relaxation():
    A = np.diag([4.0, 3.0, 2.0, 1.0, 0.8, 0.6]).astype(complex)
    b = np.array([0.5, 0.2, 0.0, 0.1, 0.0, 0.0], dtype=complex)
    c = 2.5
    chi = np.zeros((7, 7), dtype=complex)
    chi[:6, :6] = A
    chi[:6, 6] = b
    chi[6, :6] = b.conj()
    chi[6, 6] = c
    got, comp = _fixed_filling_schur(chi)
    ref = A - np.outer(b, b.conj()) / c
    assert np.max(np.abs(got - ref)) < 1e-14
    assert np.isclose(comp, c)


def test_soft_susceptibility_keeps_signed_inverse():
    block = np.diag([2.0, -5.0, 1.0, 0.5]).astype(complex)
    out = _eigen_block(block, ("a", "b", "c", "d"))
    assert np.isclose(out["chi_soft"], -5.0)
    assert np.isclose(out["inv_chi_soft"], -0.2)
    assert np.argmax(out["soft_weights"]) == 1
