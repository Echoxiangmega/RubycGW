import numpy as np

from rubycgw.ed18 import ED18_CHANNELS, ED18Solver, Q_GAMMA, Q_PERIOD3
from rubycgw.ed18_chi import zero_temperature_susceptibility
from rubycgw.model import RubyParameters


def _dense_reference(solver, V, q, deg_tol=1e-9):
    H = solver.hamiltonian(V).toarray().astype(complex)
    evals, evecs = np.linalg.eigh(H)
    mask = np.abs(evals - evals[0]) <= deg_tol
    gs = evecs[:, mask]
    exc = evecs[:, ~mask]
    de = evals[~mask] - evals[0]
    ng = gs.shape[1]
    ops = [solver.operator(ch, q) for ch in ED18_CHANNELS]
    nc = len(ops)

    reg = np.zeros((nc, nc), dtype=complex)
    for g in range(ng):
        phi = np.column_stack([op @ gs[:, g] for op in ops])
        amp = exc.conj().T @ phi
        reg += (2.0 / ng) * (amp.conj().T @ (amp / de[:, None]))
    reg = 0.5 * (reg + reg.conj().T)

    M = np.empty((nc, ng, ng), dtype=complex)
    for a, op in enumerate(ops):
        M[a] = gs.conj().T @ (op @ gs)
    means = np.trace(M, axis1=1, axis2=2) / ng
    centered = M - means[:, None, None] * np.eye(ng)[None, :, :]
    sing = np.empty((nc, nc), dtype=complex)
    for a in range(nc):
        for b in range(nc):
            sing[a, b] = np.vdot(centered[a], centered[b]) / ng
    sing = 0.5 * (sing + sing.conj().T)
    return reg, sing, ng


def test_zero_temperature_chi_matches_full_dense_spectral_sum():
    # Nf=2 gives a 153-dimensional Hilbert space, small enough for a complete
    # dense spectral benchmark while exercising the full 18-site geometry.
    solver = ED18Solver(RubyParameters(ti=0.4, t1=0.2, t2=0.2), n_particles=2)
    V = 0.37
    spec = solver.solve(V, n_eigs=12, tol=1e-12, degeneracy_tol=1e-9)

    for q in (Q_GAMMA, Q_PERIOD3):
        chi = zero_temperature_susceptibility(
            solver,
            spec,
            q,
            solve_tol=1e-12,
            maxiter=5000,
            singular_tol=1e-10,
        )
        reg_ref, sing_ref, ng_ref = _dense_reference(solver, V, q)
        assert spec.ground_multiplicity == ng_ref
        assert np.max(np.abs(chi.regular_matrix - reg_ref)) < 2e-8
        assert np.max(np.abs(chi.ground_singular_matrix - sing_ref)) < 2e-8
        assert np.max(chi.solver_residuals) < 2e-9
        assert np.all(chi.solver_info == 0)
        assert np.min(np.linalg.eigvalsh(chi.regular_matrix)) > -2e-8
        assert np.min(np.linalg.eigvalsh(chi.ground_singular_matrix)) > -2e-8


def test_zero_temperature_chi_singularity_flag_follows_ground_covariance():
    solver = ED18Solver(n_particles=2)
    spec = solver.solve(0.2, n_eigs=12, tol=1e-11, degeneracy_tol=1e-8)
    chi = zero_temperature_susceptibility(solver, spec, Q_PERIOD3, solve_tol=1e-11)
    expected = chi.ground_singular_eigenvalues[0] > 1e-10 * max(
        1.0, float(np.max(np.abs(chi.ground_singular_matrix)))
    )
    assert chi.is_singular == expected
