import numpy as np

from rubycgw.ed18 import ED18Solver, Q_GAMMA
from rubycgw.ed18_thermal import thermal_response
from rubycgw.model import RubyParameters


def _dense_exact_thermal(solver, V, T, q, channels):
    H = solver.hamiltonian(V).toarray().astype(complex)
    E, U = np.linalg.eigh(H)
    beta = 1.0 / T
    p = np.exp(-beta * (E - E[0]))
    Z = float(np.sum(p))

    ops = [solver.operator(ch, q).toarray() for ch in channels]
    oe = [U.conj().T @ op @ U for op in ops]
    n = len(channels)

    means = np.zeros(n, dtype=complex)
    for a in range(n):
        means[a] = np.sum(p * np.diag(oe[a])) / Z

    S = np.zeros((n, n), dtype=complex)
    chi = np.zeros((n, n), dtype=complex)

    # W[n,m] multiplies conj(Oa[n,m]) Ob[n,m].
    W = np.zeros((len(E), len(E)), dtype=float)
    for nst in range(len(E)):
        for mst in range(len(E)):
            d = E[nst] - E[mst]
            if abs(d) < 1e-12:
                W[nst, mst] = beta * p[mst]
            else:
                W[nst, mst] = (p[mst] - p[nst]) / d

    for a in range(n):
        for b in range(n):
            S[a, b] = np.einsum(
                "m,nm,nm->", p, np.conj(oe[a]), oe[b]
            ) / Z
            chi[a, b] = np.sum(W * np.conj(oe[a]) * oe[b]) / Z

    S -= np.outer(means.conj(), means)
    chi -= beta * np.outer(means.conj(), means)
    S = 0.5 * (S + S.conj().T)
    chi = 0.5 * (chi + chi.conj().T)
    return means, S, chi


def test_finite_temperature_krylov_trace_matches_full_spectrum():
    # Nf=1 has dimension 18, so the complete canonical basis can be used as
    # deterministic trace vectors and compared against an explicit full
    # eigendecomposition.  This validates both the thermal trace and Kubo
    # imaginary-time integral independently of stochastic typicality.
    solver = ED18Solver(
        RubyParameters(ti=0.4, t1=0.2, t2=0.2), n_particles=1
    )
    dim = solver.dimension
    basis_trace = np.eye(dim, dtype=complex)
    channels = ("x_even", "y_even", "z_same", "z_opposite")
    V = 0.35
    T = 0.4

    exact_mean, exact_S, exact_chi = _dense_exact_thermal(
        solver, V, T, Q_GAMMA, channels
    )
    got = thermal_response(
        solver,
        V,
        T,
        Q_GAMMA,
        channels=channels,
        trace_vectors=basis_trace,
        trace_weights=np.ones(dim),
        tau_points=65,
    )

    assert np.max(np.abs(got.means - exact_mean)) < 2e-9
    assert np.max(np.abs(got.structure_matrix - exact_S)) < 2e-9
    assert np.max(np.abs(got.susceptibility_matrix - exact_chi)) < 2e-6
    assert np.max(np.abs(got.susceptibility_matrix - got.susceptibility_matrix.conj().T)) < 1e-10
