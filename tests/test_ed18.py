import math

import numpy as np
from scipy import sparse

from rubycgw.ed18 import (
    ED18Solver,
    Q_GAMMA,
    Q_PERIOD3,
    fixed_particle_basis,
)
from rubycgw.ed18_plot import leading_subspace_weights
from rubycgw.model import RubyParameters


def test_n2_basis_dimension_is_18564():
    assert len(fixed_particle_basis(18, 6)) == math.comb(18, 6) == 18564


def test_ed18_interaction_is_triangle_only_and_translation_invariant():
    # Nf=2 keeps the unit test light while exercising the full 18-site geometry.
    solver = ED18Solver(RubyParameters(ti=0.4, t1=0.2, t2=0.2), n_particles=2)
    assert solver.dimension == math.comb(18, 2)
    assert len(solver.interaction_pairs) == 18  # 6 bonds/cell x 3 primitive cells
    assert np.max(solver.interaction_count) <= 1.0
    assert sparse.linalg.norm(solver.H_t - solver.H_t.T) < 1e-12
    assert sparse.linalg.norm(
        solver.translation_a1 @ solver.H_t - solver.H_t @ solver.translation_a1
    ) < 1e-12


def test_ed18_translation_has_order_three():
    solver = ED18Solver(n_particles=2)
    ident = sparse.eye(solver.dimension, format="csr")
    T3 = solver.translation_a1 @ solver.translation_a1 @ solver.translation_a1
    assert sparse.linalg.norm(T3 - ident) < 1e-12


def test_ed18_structure_factor_is_hermitian_positive_semidefinite():
    solver = ED18Solver(n_particles=2)
    spec = solver.solve(0.3, n_eigs=6, tol=1e-11)
    for q in (Q_GAMMA, Q_PERIOD3):
        sf = solver.structure_factor(spec, q)
        assert np.max(np.abs(sf.matrix - sf.matrix.conj().T)) < 1e-10
        assert np.min(np.linalg.eigvalsh(sf.matrix)) > -1e-9
        assert np.all(np.diff(sf.eigenvalues) <= 1e-12)


def test_leading_subspace_weights_are_invariant_under_degenerate_rotation():
    evals = np.array([[[3.0, 3.0, 1.0]]])
    evecs1 = np.eye(3, dtype=complex)[None, None, :, :]

    theta = 0.371
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=complex,
    )
    evecs2 = rot[None, None, :, :]

    w1, d1 = leading_subspace_weights(evals, evecs1)
    w2, d2 = leading_subspace_weights(evals, evecs2)
    assert d1[0, 0] == d2[0, 0] == 2
    assert np.allclose(w1, w2, atol=1e-14)
    assert np.allclose(w1[0, 0], [0.5, 0.5, 0.0], atol=1e-14)
