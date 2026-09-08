import numpy as np

from rubycgw.ed_cluster import (
    ED_CLUSTER_CHANNELS,
    RubyEDClusterSolver,
    fixed_popcount_basis,
    rectangular_momenta,
)
from rubycgw.model import RubyParameters, build_h0


def test_fixed_popcount_basis_is_ascending_and_complete():
    basis = fixed_popcount_basis(8, 3)
    assert len(basis) == 56
    assert np.all(np.diff(basis.astype(np.int64)) > 0)
    assert all(int(x).bit_count() == 3 for x in basis)


def test_2x2_single_particle_spectrum_matches_four_primitive_momenta():
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.0)
    solver = RubyEDClusterSolver(2, 2, params, n_particles=1)
    qpts, names = rectangular_momenta(2, 2)
    assert names == ["Gamma", "M2", "M1", "M3"]
    primitive = build_h0(qpts, params)
    expected = np.sort(np.linalg.eigvalsh(primitive).reshape(-1))
    actual = np.sort(np.linalg.eigvalsh(solver.h0))
    assert np.max(np.abs(expected - actual)) < 1e-12

    # At Nf=1 the many-body Hamiltonian is exactly the one-body problem.
    spec = solver.solve(0.0, n_eigs=6, tol=1e-12, maxiter=1000)
    assert np.max(np.abs(spec.energies - actual[:6])) < 1e-10


def test_2x2_has_24_local_interaction_bonds_and_m_points():
    solver = RubyEDClusterSolver(2, 2, n_particles=1)
    assert solver.n_sites == 24
    assert len(solver.interaction_pairs) == 24
    qpts, names = rectangular_momenta(2, 2)
    by_name = {name: q for name, q in zip(names, qpts)}
    assert np.allclose(by_name["Gamma"], [0.0, 0.0])
    assert np.allclose(by_name["M1"], [0.5, 0.0])
    assert np.allclose(by_name["M2"], [0.0, 0.5])
    assert np.allclose(by_name["M3"], [0.5, 0.5])
    assert not solver.is_commensurate_q(np.array([1.0 / 3.0, 1.0 / 3.0]))


def test_matrix_free_hamiltonian_is_hermitian_on_small_cluster():
    solver = RubyEDClusterSolver(2, 1, n_particles=2)
    rng = np.random.default_rng(7)
    x = rng.normal(size=solver.dimension) + 1j * rng.normal(size=solver.dimension)
    y = rng.normal(size=solver.dimension) + 1j * rng.normal(size=solver.dimension)
    Hx = solver.apply_hamiltonian(0.7, x)
    Hy = solver.apply_hamiltonian(0.7, y)
    assert abs(np.vdot(x, Hy) - np.vdot(Hx, y)) < 1e-10


def test_structure_factor_is_psd_on_small_cluster():
    solver = RubyEDClusterSolver(2, 1, n_particles=1)
    spec = solver.solve(0.3, n_eigs=4, tol=1e-12, maxiter=1000)
    sf = solver.structure_factor(spec, np.array([0.0, 0.0]))
    assert sf.channels == ED_CLUSTER_CHANNELS
    assert np.max(np.abs(sf.matrix - sf.matrix.conj().T)) < 1e-12
    assert np.min(sf.eigenvalues) > -1e-10
