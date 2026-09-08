"""Finite-temperature canonical response for the 18-site Ruby ED cluster.

The zero-temperature ED workflow uses exact Lanczos ground states.  For finite
T this module evaluates canonical thermal traces directly in the same fixed-N
Hilbert space using stochastic thermal typicality and Krylov exponential
propagation (``scipy.sparse.linalg.expm_multiply``).

For operators O_mu(q) normalized exactly as in :mod:`rubycgw.ed18`, we compute

    S_mu,nu(q,T)
      = <O_mu^dagger(q) O_nu(q)>_T
        - <O_mu^dagger(q)>_T <O_nu(q)>_T,

and the connected static Kubo susceptibility

    chi_mu,nu(q,T)
      = integral_0^beta d tau
          <O_mu^dagger(q,tau) O_nu(q,0)>_T,c .

The thermal trace estimator is unbiased in the number of random phase vectors;
the imaginary-time integral is evaluated with composite Simpson quadrature.
Using the same trace vectors at every V gives common-random-number noise and
therefore much smoother V scans.

This is a *canonical* finite-temperature ED/Krylov benchmark: particle number
is fixed exactly.  It should not be confused with the grand-canonical ensemble
used by a chemical-potential calculation on a finite cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from .ed18 import ED18_CHANNELS, ED18Solver


@dataclass
class ED18ThermalResponse:
    temperature: float
    beta: float
    V: float
    q: np.ndarray
    channels: tuple[str, ...]
    energy_shift: float
    shifted_partition_estimate: float
    means: np.ndarray
    structure_matrix: np.ndarray
    structure_eigenvalues: np.ndarray
    structure_eigenvectors: np.ndarray
    susceptibility_matrix: np.ndarray
    susceptibility_eigenvalues: np.ndarray
    susceptibility_eigenvectors: np.ndarray
    n_trace_vectors: int
    tau_points: int


def random_phase_trace_vectors(
    dimension: int,
    n_vectors: int,
    *,
    seed: int = 12345,
) -> np.ndarray:
    """Return normalized complex random-phase vectors, shape (dim, n_vectors).

    For one column |r>, E[|r><r|] = I/dim.  The missing overall factor ``dim``
    cancels from all normalized thermal expectation values, so no explicit
    Hilbert-space-dimension prefactor is required in production calculations.
    """
    dimension = int(dimension)
    n_vectors = int(n_vectors)
    if dimension < 1 or n_vectors < 1:
        raise ValueError("dimension and n_vectors must be positive")
    rng = np.random.default_rng(int(seed))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(dimension, n_vectors))
    return np.exp(1j * theta) / np.sqrt(float(dimension))


def _simpson_weights(beta: float, n_points: int) -> np.ndarray:
    """Composite Simpson weights on [0,beta] for an odd number of points."""
    beta = float(beta)
    n_points = int(n_points)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if n_points < 3 or n_points % 2 != 1:
        raise ValueError("tau_points must be an odd integer >= 3")
    h = beta / float(n_points - 1)
    w = np.ones(n_points, dtype=float)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * (h / 3.0)


def _eigh_desc(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(mat, dtype=complex)
    mat = 0.5 * (mat + mat.conj().T)
    vals, vecs = np.linalg.eigh(mat)
    order = np.argsort(vals.real)[::-1]
    return np.asarray(vals[order].real, dtype=float), np.asarray(vecs[:, order], dtype=complex)


def thermal_response(
    solver: ED18Solver,
    V: float,
    temperature: float,
    q: np.ndarray,
    *,
    channels: tuple[str, ...] = ED18_CHANNELS,
    trace_vectors: np.ndarray | None = None,
    trace_weights: np.ndarray | None = None,
    n_trace_vectors: int = 8,
    seed: int = 12345,
    tau_points: int = 17,
    energy_shift: float | None = None,
) -> ED18ThermalResponse:
    """Estimate finite-T S(q) and static chi(q) in the fixed-N Hilbert space.

    Parameters
    ----------
    solver:
        Existing :class:`ED18Solver` defining the fixed-particle Hilbert space.
    V, temperature:
        Interaction strength and temperature in the same energy units as the
        hopping amplitudes.
    q:
        Primitive reduced momentum, e.g. Gamma or (1/3,1/3).
    trace_vectors:
        Optional columns used to estimate the trace.  If omitted, normalized
        random-phase vectors are generated.  Passing a complete orthonormal
        basis with unit ``trace_weights`` gives a deterministic exact trace and
        is useful for small-system validation.
    trace_weights:
        Linear weights for the supplied trace vectors.  Production random
        vectors use equal weights 1/R.  A complete orthonormal basis should use
        unit weights.
    tau_points:
        Odd number of equally spaced imaginary-time points used by composite
        Simpson quadrature.
    energy_shift:
        Optional scalar E_ref.  The code propagates H-E_ref I for numerical
        stability.  The scalar Boltzmann factor cancels in normalized results.
    """
    T = float(temperature)
    if T <= 0.0:
        raise ValueError("temperature must be positive")
    beta = 1.0 / T
    tau_points = int(tau_points)
    quad_w = _simpson_weights(beta, tau_points)

    q = np.asarray(q, dtype=float).reshape(2)
    channels = tuple(str(x) for x in channels)
    nops = len(channels)
    if nops < 1:
        raise ValueError("at least one channel is required")

    H = sparse.csr_matrix(solver.hamiltonian(float(V)), dtype=complex)
    if energy_shift is None:
        # Only the lowest eigenvalue is needed as a stabilizing shift.  Reuse
        # the ordinary ED solver so all tolerances/conventions stay consistent.
        spec = solver.solve(float(V), n_eigs=2, tol=1e-10, maxiter=5000)
        energy_shift = float(spec.energies[0])
    else:
        energy_shift = float(energy_shift)
    K = H - energy_shift * sparse.eye(solver.dimension, dtype=complex, format="csr")
    A = -K
    traceA = complex(np.sum(A.diagonal())).real

    ops = [solver.operator(ch, q) for ch in channels]

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
                f"trace_vectors must have shape ({solver.dimension}, R), got {vecs.shape}"
            )
        if trace_weights is None:
            weights = np.full(vecs.shape[1], 1.0 / float(vecs.shape[1]), dtype=float)
        else:
            weights = np.asarray(trace_weights, dtype=float).reshape(-1)
            if len(weights) != vecs.shape[1]:
                raise ValueError("trace_weights length does not match trace_vectors")

    z_sum = 0.0
    mean_num = np.zeros(nops, dtype=complex)
    structure_num = np.zeros((nops, nops), dtype=complex)
    chi_num = np.zeros((nops, nops), dtype=complex)

    mid = tau_points // 2

    for ir in range(vecs.shape[1]):
        r = np.asarray(vecs[:, ir], dtype=complex)
        wr = float(weights[ir])
        if wr == 0.0:
            continue

        # r_tau[j] = exp[-tau_j (H-E_ref)] |r>.
        r_tau = expm_multiply(
            A,
            r,
            start=0.0,
            stop=beta,
            num=tau_points,
            endpoint=True,
            traceA=traceA,
        )

        # Symmetric estimator at beta/2 for Z, <O>, and equal-time S.
        w = np.asarray(r_tau[mid], dtype=complex)
        z_r = float(np.vdot(w, w).real)
        z_sum += wr * z_r
        ow = np.column_stack([op @ w for op in ops])
        mean_num += wr * np.asarray([np.vdot(w, ow[:, a]) for a in range(nops)])
        structure_num += wr * (ow.conj().T @ ow)

        # Kubo integrand.  B_tau[j,b] = exp(-tau_j K) O_b |r>.
        B0 = np.column_stack([op @ r for op in ops])
        B_tau = expm_multiply(
            A,
            B0,
            start=0.0,
            stop=beta,
            num=tau_points,
            endpoint=True,
            traceA=traceA,
        )
        chi_r = np.zeros((nops, nops), dtype=complex)
        for j in range(tau_points):
            # Because the grid is uniform, beta-tau_j is the reversed index.
            u = np.asarray(r_tau[tau_points - 1 - j], dtype=complex)
            ou = np.column_stack([op @ u for op in ops])
            v = np.asarray(B_tau[j], dtype=complex)
            chi_r += quad_w[j] * (ou.conj().T @ v)
        chi_num += wr * chi_r

    if not np.isfinite(z_sum) or z_sum <= 0.0:
        raise RuntimeError(f"invalid shifted partition-function estimate {z_sum}")

    means = mean_num / z_sum
    structure = structure_num / z_sum - np.outer(means.conj(), means)
    chi = chi_num / z_sum - beta * np.outer(means.conj(), means)

    # Stochastic noise and quadrature error can generate tiny anti-Hermitian
    # pieces; the physical static matrices are Hermitian.
    structure = 0.5 * (structure + structure.conj().T)
    chi = 0.5 * (chi + chi.conj().T)
    svals, svecs = _eigh_desc(structure)
    cvals, cvecs = _eigh_desc(chi)

    return ED18ThermalResponse(
        temperature=T,
        beta=beta,
        V=float(V),
        q=q,
        channels=channels,
        energy_shift=energy_shift,
        shifted_partition_estimate=float(z_sum),
        means=np.asarray(means, dtype=complex),
        structure_matrix=np.asarray(structure, dtype=complex),
        structure_eigenvalues=svals,
        structure_eigenvectors=svecs,
        susceptibility_matrix=np.asarray(chi, dtype=complex),
        susceptibility_eigenvalues=cvals,
        susceptibility_eigenvectors=cvecs,
        n_trace_vectors=int(vecs.shape[1]),
        tau_points=tau_points,
    )
