"""Finite-temperature canonical response on rectangular Ruby ED clusters.

This is the ordinary-L1xL2 counterpart of :mod:`rubycgw.ed18_thermal`.  It uses
``RubyEDClusterSolver`` so the finite torus is exactly the same rectangular
geometry used by the cluster-ED+GW lattice calculation.

For fixed particle number N we estimate

    C_ab(q,tau) = <O_a^dag(q,tau) O_b(q,0)>_T,c,
    S_ab(q,T)   = C_ab(q,0),
    chi_ab(q,T) = integral_0^beta d tau C_ab(q,tau),

with random-phase thermal typicality and Krylov ``expm_multiply`` propagation.
The pseudospin operators inherit the solver normalization 1/sqrt(Ncell), so the
q=0 susceptibility is directly comparable to the per-cell uniform source
response used by ``benchmark_cluster_ed_gw_chi.py``.

The ensemble is canonical fixed-N.  This is an exact finite-torus definition;
only the thermal trace and imaginary-time integral are stochastic/numerically
approximated.  Increase ``n_trace_vectors`` and ``tau_points`` to check those
errors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply

from .ed_cluster import ED_CLUSTER_CHANNELS, RubyEDClusterSolver


@dataclass
class EDClusterThermalResponse:
    temperature: float
    beta: float
    V: float
    q: np.ndarray
    channels: tuple[str, ...]
    energy_shift: float
    shifted_partition_estimate: float
    means: np.ndarray
    tau_grid: np.ndarray
    correlation_tau_matrix: np.ndarray
    structure_matrix: np.ndarray
    structure_eigenvalues: np.ndarray
    structure_eigenvectors: np.ndarray
    susceptibility_matrix: np.ndarray
    susceptibility_eigenvalues: np.ndarray
    susceptibility_eigenvectors: np.ndarray
    n_trace_vectors: int
    tau_points: int
    n_sites: int
    n_particles: int
    hilbert_dimension: int


def random_phase_trace_vectors(
    dimension: int,
    n_vectors: int,
    *,
    seed: int = 12345,
) -> np.ndarray:
    """Normalized complex random-phase trace vectors, shape (dim,R)."""
    dimension = int(dimension)
    n_vectors = int(n_vectors)
    if dimension < 1 or n_vectors < 1:
        raise ValueError("dimension and n_vectors must be positive")
    rng = np.random.default_rng(int(seed))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(dimension, n_vectors))
    return np.exp(1j * theta) / np.sqrt(float(dimension))


def _simpson_weights(beta: float, n_points: int) -> np.ndarray:
    beta = float(beta)
    n_points = int(n_points)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if n_points < 3 or n_points % 2 != 1:
        raise ValueError("tau_points must be an odd integer >=3")
    h = beta / float(n_points - 1)
    w = np.ones(n_points, dtype=float)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * (h / 3.0)


def _eigh_desc(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(mat, dtype=complex)
    a = 0.5 * (a + a.conj().T)
    vals, vecs = np.linalg.eigh(a)
    order = np.argsort(vals.real)[::-1]
    return np.asarray(vals[order].real, dtype=float), np.asarray(vecs[:, order], dtype=complex)


def _shifted_generator(
    solver: RubyEDClusterSolver,
    V: float,
    energy_shift: float,
) -> tuple[LinearOperator, complex]:
    """Return A=-(H-Eshift) and its exact many-body trace."""
    dim = int(solver.dimension)
    V = float(V)
    shift = float(energy_shift)

    def matvec(x):
        arr = np.asarray(x)
        return -(solver.apply_hamiltonian(V, arr) - shift * arr)

    def matmat(X):
        arr = np.asarray(X)
        if arr.ndim != 2 or arr.shape[0] != dim:
            raise ValueError("matrix shape mismatch in thermal Krylov propagation")
        return np.column_stack([matvec(arr[:, j]) for j in range(arr.shape[1])])

    A = LinearOperator(
        (dim, dim),
        matvec=matvec,
        rmatvec=matvec,
        matmat=matmat,
        rmatmat=matmat,
        dtype=np.complex128,
    )
    trace_h = float(solver.hamiltonian_trace(V))
    trace_a = -(trace_h - shift * dim)
    return A, complex(trace_a)


def thermal_response(
    solver: RubyEDClusterSolver,
    V: float,
    temperature: float,
    q: np.ndarray,
    *,
    channels: tuple[str, ...] = ED_CLUSTER_CHANNELS,
    trace_vectors: np.ndarray | None = None,
    trace_weights: np.ndarray | None = None,
    n_trace_vectors: int = 8,
    seed: int = 12345,
    tau_points: int = 17,
    energy_shift: float | None = None,
    eig_tol: float = 1.0e-10,
    eig_maxiter: int = 5000,
) -> EDClusterThermalResponse:
    """Estimate canonical finite-T C(q,tau), S(q), and static chi(q).

    ``solver`` fixes the rectangular torus and particle number.  Passing a
    complete orthonormal basis through ``trace_vectors`` with unit
    ``trace_weights`` turns the thermal trace into a deterministic exact trace;
    this is used by small-system regression tests.
    """
    T = float(temperature)
    if T <= 0.0:
        raise ValueError("temperature must be positive")
    beta = 1.0 / T
    tau_points = int(tau_points)
    quad_w = _simpson_weights(beta, tau_points)
    tau_grid = np.linspace(0.0, beta, tau_points, dtype=float)

    q = np.asarray(q, dtype=float).reshape(2)
    if not solver.is_commensurate_q(q):
        raise ValueError(f"q={tuple(q)} is not commensurate with {solver.L1}x{solver.L2}")
    channels = tuple(str(x) for x in channels)
    nops = len(channels)
    if nops < 1:
        raise ValueError("at least one channel is required")

    if energy_shift is None:
        spec = solver.solve(
            float(V), n_eigs=2, tol=float(eig_tol), maxiter=int(eig_maxiter)
        )
        energy_shift = float(spec.energies[0])
    else:
        energy_shift = float(energy_shift)
    A, traceA = _shifted_generator(solver, float(V), energy_shift)

    op_mats = [solver.operator_matrix(ch, q) for ch in channels]

    if trace_vectors is None:
        vecs = random_phase_trace_vectors(
            solver.dimension, int(n_trace_vectors), seed=int(seed)
        )
        weights = np.full(vecs.shape[1], 1.0 / float(vecs.shape[1]), dtype=float)
    else:
        vecs = np.asarray(trace_vectors, dtype=complex)
        if vecs.ndim == 1:
            vecs = vecs[:, None]
        if vecs.ndim != 2 or vecs.shape[0] != solver.dimension:
            raise ValueError(
                f"trace_vectors must have shape ({solver.dimension},R), got {vecs.shape}"
            )
        if trace_weights is None:
            weights = np.full(vecs.shape[1], 1.0 / float(vecs.shape[1]), dtype=float)
        else:
            weights = np.asarray(trace_weights, dtype=float).reshape(-1)
            if len(weights) != vecs.shape[1]:
                raise ValueError("trace_weights length does not match trace_vectors")

    def apply_ops(v: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [solver.apply_onebody_matrix(op, v) for op in op_mats]
        )

    z_sum = 0.0
    mean_num = np.zeros(nops, dtype=complex)
    structure_num = np.zeros((nops, nops), dtype=complex)
    corr_tau_num = np.zeros((tau_points, nops, nops), dtype=complex)
    mid = tau_points // 2

    for ir in range(vecs.shape[1]):
        r = np.asarray(vecs[:, ir], dtype=complex)
        wr = float(weights[ir])
        if wr == 0.0:
            continue

        r_tau = expm_multiply(
            A,
            r,
            start=0.0,
            stop=beta,
            num=tau_points,
            endpoint=True,
            traceA=traceA,
        )

        w = np.asarray(r_tau[mid], dtype=complex)
        z_r = float(np.vdot(w, w).real)
        z_sum += wr * z_r
        ow = apply_ops(w)
        mean_num += wr * np.asarray([np.vdot(w, ow[:, a]) for a in range(nops)])
        structure_num += wr * (ow.conj().T @ ow)

        B0 = apply_ops(r)
        B_tau = expm_multiply(
            A,
            B0,
            start=0.0,
            stop=beta,
            num=tau_points,
            endpoint=True,
            traceA=traceA,
        )
        for j in range(tau_points):
            u = np.asarray(r_tau[tau_points - 1 - j], dtype=complex)
            ou = apply_ops(u)
            v = np.asarray(B_tau[j], dtype=complex)
            corr_tau_num[j] += wr * (ou.conj().T @ v)

    if not np.isfinite(z_sum) or z_sum <= 0.0:
        raise RuntimeError(f"invalid shifted partition-function estimate {z_sum}")

    means = mean_num / z_sum
    mean_outer = np.outer(means.conj(), means)
    structure = structure_num / z_sum - mean_outer
    corr_tau = corr_tau_num / z_sum - mean_outer[None, :, :]
    chi = np.tensordot(quad_w, corr_tau, axes=(0, 0))

    structure = 0.5 * (structure + structure.conj().T)
    chi = 0.5 * (chi + chi.conj().T)
    svals, svecs = _eigh_desc(structure)
    cvals, cvecs = _eigh_desc(chi)

    return EDClusterThermalResponse(
        temperature=T,
        beta=beta,
        V=float(V),
        q=q,
        channels=channels,
        energy_shift=float(energy_shift),
        shifted_partition_estimate=float(z_sum),
        means=np.asarray(means, dtype=complex),
        tau_grid=tau_grid,
        correlation_tau_matrix=np.asarray(corr_tau, dtype=complex),
        structure_matrix=np.asarray(structure, dtype=complex),
        structure_eigenvalues=svals,
        structure_eigenvectors=svecs,
        susceptibility_matrix=np.asarray(chi, dtype=complex),
        susceptibility_eigenvalues=cvals,
        susceptibility_eigenvectors=cvecs,
        n_trace_vectors=int(vecs.shape[1]),
        tau_points=tau_points,
        n_sites=int(solver.n_sites),
        n_particles=int(solver.n_particles),
        hilbert_dimension=int(solver.dimension),
    )


__all__ = [
    "EDClusterThermalResponse",
    "random_phase_trace_vectors",
    "thermal_response",
]
