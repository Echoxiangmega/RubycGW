import numpy as np

from rubycgw.model import eta_vertices
from rubycgw.pseudospin import (
    primitive_triangle_pseudospin_vertices,
    primitive_cell_pseudospin_channels,
    pseudospin_harmonic_transform,
)


def _chiral_vectors():
    w = np.exp(2j * np.pi / 3.0)
    plus = np.asarray([1.0, w, w**2], dtype=complex) / np.sqrt(3.0)
    minus = np.asarray([1.0, w**2, w], dtype=complex) / np.sqrt(3.0)
    return plus, minus


def test_A_vertices_project_to_pauli_matrices():
    plus, minus = _chiral_vectors()
    U = np.column_stack([plus, minus])
    v = primitive_triangle_pseudospin_vertices()
    px = np.asarray([[0, 1], [1, 0]], dtype=complex)
    py = np.asarray([[0, -1j], [1j, 0]], dtype=complex)
    pz = np.asarray([[1, 0], [0, -1]], dtype=complex)
    for key, target in (("Ax", px), ("Ay", py), ("Az", pz)):
        block = v[key][:3, :3]
        assert np.allclose(U.conj().T @ block @ U, target, atol=1e-12)


def test_B_physical_frame_projects_to_same_paulis():
    plus, minus = _chiral_vectors()
    # Physical B chirality swaps the algebraic +/- labels because the B
    # algebraic arrow loop has the opposite geometric handedness.
    U = np.column_stack([minus, plus])
    v = primitive_triangle_pseudospin_vertices()
    px = np.asarray([[0, 1], [1, 0]], dtype=complex)
    py = np.asarray([[0, -1j], [1j, 0]], dtype=complex)
    pz = np.asarray([[1, 0], [0, -1]], dtype=complex)
    for key, target in (("Bx", px), ("By", py), ("Bz", pz)):
        block = v[key][3:, 3:]
        assert np.allclose(U.conj().T @ block @ U, target, atol=1e-12)


def test_z_same_opposite_match_legacy_current_channels_up_to_normalization():
    _, _, kp, km = eta_vertices()
    ch = primitive_cell_pseudospin_channels()
    # Overall minus sign follows the chosen physical tau_z convention and has
    # no effect on a diagonal susceptibility.
    assert np.allclose(ch["z_same"], -km / np.sqrt(3.0), atol=1e-12)
    assert np.allclose(ch["z_opposite"], -kp / np.sqrt(3.0), atol=1e-12)


def test_xy_are_tr_even_z_is_tr_odd_matrix_structure():
    v = primitive_triangle_pseudospin_vertices()
    for key in ("Ax", "Ay", "Bx", "By"):
        assert np.allclose(v[key].imag, 0.0, atol=1e-14)
        assert np.allclose(v[key], v[key].T, atol=1e-14)
    for key in ("Az", "Bz"):
        assert np.allclose(v[key].real, 0.0, atol=1e-14)
        assert np.allclose(v[key], -v[key].T, atol=1e-14)


def test_harmonic_transform_is_orthogonal_blockwise():
    T = pseudospin_harmonic_transform(4)
    assert T.shape == (12, 12)
    assert np.allclose(T @ T.T, np.eye(12), atol=1e-12)
