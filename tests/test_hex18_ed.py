import numpy as np

from rubycgw.hex18_ed import (
    Hex18Solver,
    build_hex18_one_body,
    canonical_ring_mode,
    hex18_interaction_pairs,
    hex18_local_current_matrices,
)
from rubycgw.model import RubyParameters


def test_hex18_geometry_and_hilbert_space():
    p = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=1.0)
    h = build_hex18_one_body(p)
    assert h.shape == (18, 18)
    assert np.max(np.abs(h - h.conj().T)) < 1e-13
    # Six triangles give 18 internal ti bonds; six adjacent triangle pairs give
    # two intertriangle bonds each, for 30 undirected hoppings total.
    upper = np.triu(np.abs(h) > 1e-14, k=1)
    assert int(np.count_nonzero(upper)) == 30
    assert len(hex18_interaction_pairs()) == 18

    solver = Hex18Solver(p, primitive_filling=2.0)
    assert solver.n_particles == 6
    assert solver.dimension == 18564


def test_hex18_current_vertices_are_hermitian_and_physical():
    J = hex18_local_current_matrices()
    assert J.shape == (6, 18, 18)
    for m in range(6):
        assert np.max(np.abs(J[m] - J[m].conj().T)) < 1e-13
        assert np.linalg.norm(J[m]) > 0.0
    # Ring Fourier basis is orthonormal.
    U = np.stack([canonical_ring_mode(l) for l in range(6)], axis=1)
    assert np.max(np.abs(U.conj().T @ U - np.eye(6))) < 1e-12
