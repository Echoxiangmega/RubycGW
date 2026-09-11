import numpy as np

from rubycgw.ed_cluster import RubyEDClusterSolver
from rubycgw.ed_cluster_thermal import thermal_response
from rubycgw.model import RubyParameters


def _dense_from_apply(solver, V):
    dim = solver.dimension
    H = np.empty((dim, dim), dtype=complex)
    for j in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        H[:, j] = solver.apply_hamiltonian(V, e)
    return 0.5 * (H + H.conj().T)


def _dense_onebody(solver, matrix):
    dim = solver.dimension
    O = np.empty((dim, dim), dtype=complex)
    for j in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        O[:, j] = solver.apply_onebody_matrix(matrix, e)
    return O


def _canonical_lehmann(H, ops, T):
    beta = 1.0 / float(T)
    E, U = np.linalg.eigh(H)
    raw = np.exp(-beta * (E - E[0]))
    p = raw / np.sum(raw)
    n = len(ops)
    transformed = [U.conj().T @ O @ U for O in ops]
    means = np.asarray([np.dot(p, np.diag(O)) for O in transformed])

    delta = E[:, None] - E[None, :]
    numerator = p[None, :] - p[:, None]
    near = np.abs(delta) < 1e-12
    w = np.empty_like(delta)
    np.divide(numerator, delta, out=w, where=~near)
    w[near] = np.broadcast_to(
        beta * 0.5 * (p[:, None] + p[None, :]), w.shape
    )[near]

    chi = np.zeros((n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            chi[a, b] = np.sum(
                w * transformed[a].conj() * transformed[b]
            ) - beta * means[a].conj() * means[b]
    return 0.5 * (chi + chi.conj().T)


def test_rectangular_thermal_response_matches_exact_canonical_lehmann():
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.7)
    solver = RubyEDClusterSolver(1, 1, params, n_particles=2)
    V = 0.7
    T = 0.5
    H = _dense_from_apply(solver, V)
    assert abs(solver.hamiltonian_trace(V) - np.trace(H).real) < 1e-12

    channels = ("z_same", "z_opposite")
    q = np.array([0.0, 0.0])
    onebody = [solver.operator_matrix(ch, q) for ch in channels]
    ops = [_dense_onebody(solver, K) for K in onebody]
    chi_exact = _canonical_lehmann(H, ops, T)

    dim = solver.dimension
    trace_vectors = np.eye(dim, dtype=complex)
    response = thermal_response(
        solver,
        V,
        T,
        q,
        channels=channels,
        trace_vectors=trace_vectors,
        trace_weights=np.ones(dim),
        tau_points=65,
        energy_shift=float(np.linalg.eigvalsh(H)[0]),
    )

    np.testing.assert_allclose(
        response.susceptibility_matrix,
        chi_exact,
        rtol=3e-5,
        atol=3e-6,
    )
    assert np.max(np.abs(response.means)) < 1e-10
    assert np.max(np.abs(response.susceptibility_matrix.imag)) < 1e-10
