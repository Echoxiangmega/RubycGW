import numpy as np

from rubycgw.folded_torus import (
    commensurate_source_matrix,
    fold_static_k_field,
)


def test_folded_torus_spectrum_is_union_of_discrete_bloch_spectra():
    rng = np.random.default_rng(123)
    Lx, Ly, norb = 2, 3, 2
    h = np.zeros((Lx, Ly, norb, norb), dtype=complex)
    for i in range(Lx):
        for j in range(Ly):
            x = rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb))
            h[i, j] = 0.5 * (x + x.conj().T)
    H = fold_static_k_field(h)
    got = np.sort(np.linalg.eigvalsh(H))
    ref = np.sort(np.concatenate([
        np.linalg.eigvalsh(h[i, j])
        for i in range(Lx)
        for j in range(Ly)
    ]))
    assert np.max(np.abs(got - ref)) < 1e-11


def test_commensurate_source_is_hermitian_and_q0_repeats_form_factor():
    M = np.array(
        [[0.3, 0.2j], [-0.2j, -0.3]],
        dtype=complex,
    )
    src = commensurate_source_matrix(M, (0, 0), 2, 2, normalize=False)
    assert np.max(np.abs(src - src.conj().T)) < 1e-13
    for c in range(4):
        block = src[2*c:2*c+2, 2*c:2*c+2]
        assert np.max(np.abs(block - M)) < 1e-13


def test_finite_q_source_changes_sign_on_period_two_torus():
    M = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    src = commensurate_source_matrix(M, (1, 0), 2, 1, normalize=False)
    b0 = src[:2, :2]
    b1 = src[2:4, 2:4]
    assert np.max(np.abs(b0 - M)) < 1e-13
    assert np.max(np.abs(b1 + M)) < 1e-13
