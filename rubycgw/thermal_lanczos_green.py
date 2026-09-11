"""Low-temperature grand-canonical Green functions from Lanczos resolvents.

This module extends finite-temperature ED benchmarking beyond the 12-site
full-spectrum solver.  It is designed primarily for the 3x1 (18-site) Ruby
torus and, with more aggressive truncation, for 2x2 (24-site).

The approximation is deliberately transparent:

1. In a window of particle-number sectors, compute the lowest ``thermal_eigs``
   energies and use them to solve the grand-canonical chemical potential.
2. Keep low-energy initial eigenstates until their cumulative grand-canonical
   probability exceeds ``1-thermal_discard_weight_tol``.
3. For every retained initial state, evaluate the addition/removal resolvents
   with scalar Lanczos.  The final N+/-1 sectors are therefore *not* truncated
   to a few eigenstates: all states represented by the Krylov resolvent enter.
4. Translation invariance reduces the source orbitals to the six sites of one
   primitive cell; the full real-space Green matrix is reconstructed from those
   columns.

This is not full-spectrum exact diagonalization.  Its convergence controls are
``sector_padding``, ``thermal_eigs``, ``thermal_discard_weight_tol`` and
``krylov_dim``.  The returned diagnostics expose the retained thermal weight and
particle-sector edge weight so production comparisons can be qualified.

The current implementation assumes the real-hopping, time-reversal-symmetric
Ruby Hamiltonian used by :class:`RubyEDClusterSolver`.  That makes the projected
removal resolvent symmetric and lets the same six reference-cell source runs
reconstruct the matrix Green function.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.sparse.linalg import eigsh

from .ed_cluster import RubyEDClusterSolver, _fermion_signs if False else None
from .ed_cluster import _parity_u64
from .model import NSUB, RubyParameters


@dataclass(frozen=True)
class ThermalLanczosOptions:
    sector_padding: int = 2
    thermal_eigs: int = 12
    thermal_discard_weight_tol: float = 1.0e-7
    krylov_dim: int = 160
    krylov_tol: float = 1.0e-12
    eig_tol: float = 1.0e-9
    eig_maxiter: int = 6000
    verbose: bool = True


@dataclass
class ThermalLanczosResult:
    G: np.ndarray
    mu: float
    average_particles: float
    kept_thermal_weight: float
    sector_numbers: np.ndarray
    sector_probabilities: np.ndarray
    sector_ground_energies: np.ndarray
    sector_last_gap: np.ndarray
    retained_states_per_sector: np.ndarray
    krylov_steps_mean: float
    krylov_steps_max: int
    translation_residual: float


def _dense_hamiltonian_from_matvec(solver: RubyEDClusterSolver, V: float) -> np.ndarray:
    dim = int(solver.dimension)
    H = np.empty((dim, dim), dtype=float)
    for j in range(dim):
        e = np.zeros(dim, dtype=float)
        e[j] = 1.0
        H[:, j] = np.asarray(solver.apply_hamiltonian(float(V), e), dtype=float)
    return 0.5 * (H + H.T)


def _lowest_energies(
    solver: RubyEDClusterSolver,
    V: float,
    n_eigs: int,
    tol: float,
    maxiter: int,
) -> np.ndarray:
    dim = int(solver.dimension)
    k = max(1, min(int(n_eigs), dim))
    if dim <= max(k + 1, 96):
        vals = np.linalg.eigvalsh(_dense_hamiltonian_from_matvec(solver, V))
        return np.asarray(vals[:k], dtype=float)
    vals = eigsh(
        solver.hamiltonian_operator(float(V)),
        k=min(k, dim - 2),
        which="SA",
        tol=float(tol),
        maxiter=int(maxiter),
        return_eigenvectors=False,
    )
    return np.sort(np.asarray(vals, dtype=float))


def _lowest_eigensystem(
    solver: RubyEDClusterSolver,
    V: float,
    n_eigs: int,
    tol: float,
    maxiter: int,
) -> tuple[np.ndarray, np.ndarray]:
    dim = int(solver.dimension)
    k = max(1, min(int(n_eigs), dim))
    if dim <= max(k + 1, 96):
        vals, vecs = np.linalg.eigh(_dense_hamiltonian_from_matvec(solver, V))
        return np.asarray(vals[:k], dtype=float), np.asarray(vecs[:, :k], dtype=float)
    vals, vecs = eigsh(
        solver.hamiltonian_operator(float(V)),
        k=min(k, dim - 2),
        which="SA",
        tol=float(tol),
        maxiter=int(maxiter),
    )
    order = np.argsort(vals)
    return (
        np.asarray(vals[order], dtype=float),
        np.asarray(np.real_if_close(vecs[:, order]), dtype=float),
    )


def _truncated_thermodynamics(
    spectra: dict[int, np.ndarray],
    mu: float,
    T: float,
) -> tuple[float, float, dict[int, np.ndarray]]:
    beta = 1.0 / float(T)
    logs = []
    for N, e in spectra.items():
        logs.append(-beta * (np.asarray(e, dtype=float) - float(mu) * int(N)))
    shift = max(float(np.max(x)) for x in logs)
    raw: dict[int, np.ndarray] = {}
    Z = 0.0
    for (N, e), x in zip(spectra.items(), logs):
        w = np.exp(x - shift)
        raw[int(N)] = w
        Z += float(np.sum(w))
    probs = {N: w / Z for N, w in raw.items()}
    Navg = float(sum(N * np.sum(p) for N, p in probs.items()))
    return Z, Navg, probs


def _solve_mu_truncated(
    spectra: dict[int, np.ndarray],
    target_particles: float,
    T: float,
    tol: float = 1.0e-11,
) -> tuple[float, dict[int, np.ndarray], float]:
    target = float(target_particles)
    lo, hi = -4.0, 4.0
    for _ in range(40):
        _, nlo, _ = _truncated_thermodynamics(spectra, lo, T)
        _, nhi, _ = _truncated_thermodynamics(spectra, hi, T)
        if nlo <= target <= nhi:
            break
        width = hi - lo
        lo -= width
        hi += width
    else:
        raise RuntimeError("could not bracket truncated grand-canonical chemical potential")
    mid = 0.5 * (lo + hi)
    probs = None
    Navg = np.nan
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        _, Navg, probs = _truncated_thermodynamics(spectra, mid, T)
        if abs(Navg - target) < float(tol):
            break
        if Navg > target:
            hi = mid
        else:
            lo = mid
    assert probs is not None
    return float(mid), probs, float(Navg)


def _transition_map(
    source: RubyEDClusterSolver,
    target: RubyEDClusterSolver,
    site: int,
    *,
    creation: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    site = int(site)
    states = np.asarray(source.basis, dtype=np.uint64)
    occupied = ((states >> np.uint64(site)) & np.uint64(1)).astype(bool)
    src = np.flatnonzero(~occupied if creation else occupied)
    selected = states[src]
    if creation:
        dst_state = selected | np.uint64(1 << site)
    else:
        dst_state = selected & ~np.uint64(1 << site)
    dst = target._state_indices(dst_state.astype(np.uint64))
    if site <= 0:
        sign = np.ones(len(src), dtype=float)
    else:
        mask = np.uint64((1 << site) - 1)
        parity = _parity_u64(selected & mask)
        sign = np.where(parity == 0, 1.0, -1.0)
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), sign


def _mapped_vector(
    psi: np.ndarray,
    target_dim: int,
    mapping: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    src, dst, sign = mapping
    out = np.zeros(int(target_dim), dtype=float)
    out[dst] = sign * np.asarray(psi, dtype=float)[src]
    return out


def _left_projections(
    psi: np.ndarray,
    q: np.ndarray,
    mappings: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    psi = np.asarray(psi, dtype=float)
    q = np.asarray(q, dtype=float)
    out = np.empty(len(mappings), dtype=float)
    for i, (src, dst, sign) in enumerate(mappings):
        out[i] = float(np.dot(sign * psi[src], q[dst]))
    return out


def _lanczos_cross_resolvent(
    solver: RubyEDClusterSolver,
    V: float,
    source_vec: np.ndarray,
    left_psi: np.ndarray,
    left_maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    zvals: np.ndarray,
    *,
    max_steps: int,
    tol: float,
) -> tuple[np.ndarray, int]:
    """Return u_i^T (z-H)^-1 v for all mapped left vectors u_i."""
    v = np.asarray(source_vec, dtype=float)
    norm = float(np.linalg.norm(v))
    zvals = np.asarray(zvals, dtype=complex)
    if norm < 1.0e-15:
        return np.zeros((len(zvals), len(left_maps)), dtype=complex), 0

    q = v / norm
    qprev = np.zeros_like(q)
    beta_prev = 0.0
    alpha: list[float] = []
    beta: list[float] = []
    proj: list[np.ndarray] = []

    for step in range(max(int(max_steps), 1)):
        proj.append(_left_projections(left_psi, q, left_maps))
        hq = np.asarray(solver.apply_hamiltonian(float(V), q), dtype=float)
        a = float(np.dot(q, hq))
        w = hq - a * q
        if step > 0:
            w -= beta_prev * qprev
        b = float(np.linalg.norm(w))
        alpha.append(a)
        if b < float(tol) or step + 1 >= int(max_steps):
            break
        beta.append(b)
        qprev, q = q, w / b
        beta_prev = b

    m = len(alpha)
    Tmat = np.diag(np.asarray(alpha, dtype=float))
    if m > 1:
        off = np.asarray(beta[: m - 1], dtype=float)
        Tmat += np.diag(off, 1) + np.diag(off, -1)
    theta, U = np.linalg.eigh(Tmat)
    C = np.asarray(proj, dtype=float)  # (m,nleft), C[j,i]=u_i^T q_j
    spectral = (C.T @ U) * U[0][None, :] * norm
    out = np.empty((len(zvals), C.shape[1]), dtype=complex)
    for iz, z in enumerate(zvals):
        out[iz] = np.sum(spectral / (z - theta)[None, :], axis=1)
    return out, m


def _reconstruct_translation_invariant(
    columns: np.ndarray,
    L1: int,
    L2: int,
) -> np.ndarray:
    """Reconstruct full G from columns whose source is cell (0,0), orbitals 0..5."""
    columns = np.asarray(columns, dtype=complex)
    nf, nsites, nsrc = columns.shape
    if nsrc != NSUB or nsites != NSUB * int(L1) * int(L2):
        raise ValueError("reference-cell column shape mismatch")
    cells = [(r1, r2) for r1 in range(int(L1)) for r2 in range(int(L2))]
    index = {cell: i for i, cell in enumerate(cells)}
    out = np.empty((nf, nsites, nsites), dtype=complex)
    for cR, (r1, r2) in enumerate(cells):
        for cS, (s1, s2) in enumerate(cells):
            d = ((r1 - s1) % int(L1), (r2 - s2) % int(L2))
            cd = index[d]
            rows = slice(NSUB * cR, NSUB * (cR + 1))
            cols = slice(NSUB * cS, NSUB * (cS + 1))
            refrows = slice(NSUB * cd, NSUB * (cd + 1))
            out[:, rows, cols] = columns[:, refrows, :]
    return out


def _translation_residual_from_columns(columns: np.ndarray) -> float:
    """A cheap Matsubara symmetry residual for the reconstructed reference columns."""
    arr = np.asarray(columns, dtype=complex)
    den = max(float(np.max(np.abs(arr))), 1.0e-300)
    # The diagonal reference-cell block should be symmetric for the real-hopping
    # time-reversal-symmetric problem at each Matsubara frequency up to the usual
    # complex-frequency transpose structure.
    block = arr[:, :NSUB, :]
    return float(np.max(np.abs(block - np.swapaxes(block, -1, -2))) / den)


def thermal_lanczos_green(
    L1: int,
    L2: int,
    params: RubyParameters,
    *,
    V: float,
    T: float,
    target_particles: float,
    omega: np.ndarray,
    opts: ThermalLanczosOptions = ThermalLanczosOptions(),
) -> ThermalLanczosResult:
    """Approximate the finite-T grand-canonical Green matrix on an 18/24-site torus."""
    L1, L2 = int(L1), int(L2)
    nsites = NSUB * L1 * L2
    if nsites > 24:
        raise ValueError("thermal Lanczos benchmark is currently limited to <=24 sites")
    if not (1.0 <= float(target_particles) <= nsites - 1.0):
        raise ValueError("target particle number must be interior to the finite Fock space")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if int(opts.sector_padding) < 1:
        raise ValueError("sector_padding must be at least 1")

    center = int(round(float(target_particles)))
    Nmin = max(1, center - int(opts.sector_padding))
    Nmax = min(nsites - 1, center + int(opts.sector_padding))
    sectors = list(range(Nmin, Nmax + 1))

    # First pass: eigenvalues only, so 24-site runs do not retain several huge
    # eigenvector arrays simultaneously just to determine mu and thermal weights.
    spectra: dict[int, np.ndarray] = {}
    dims: dict[int, int] = {}
    for N in sectors:
        solver = RubyEDClusterSolver(L1, L2, params, n_particles=N)
        dims[N] = int(solver.dimension)
        if opts.verbose:
            print(f"[TL-ED] sector N={N}: dim={solver.dimension}, low spectrum ...", flush=True)
        spectra[N] = _lowest_energies(
            solver, V, int(opts.thermal_eigs), float(opts.eig_tol), int(opts.eig_maxiter)
        )

    mu, probs, Navg = _solve_mu_truncated(spectra, target_particles, T)
    sector_p = np.asarray([float(np.sum(probs[N])) for N in sectors], dtype=float)
    if opts.verbose:
        print(
            f"[TL-ED] truncated grand canonical: mu={mu:+.10f}, "
            f"N={Navg:.10f}, sector_edge_weight={sector_p[0]+sector_p[-1]:.3e}",
            flush=True,
        )

    # Globally retain the most important initial eigenstates by thermal weight.
    entries: list[tuple[float, int, int]] = []
    for N in sectors:
        for a, p in enumerate(probs[N]):
            entries.append((float(p), N, int(a)))
    entries.sort(reverse=True, key=lambda x: x[0])
    target_weight = 1.0 - float(opts.thermal_discard_weight_tol)
    keep: dict[int, list[int]] = {N: [] for N in sectors}
    cumulative = 0.0
    for p, N, a in entries:
        keep[N].append(a)
        cumulative += p
        if cumulative >= target_weight:
            break
    for N in keep:
        keep[N].sort()

    iw = 1j * np.asarray(omega, dtype=float)
    columns = np.zeros((len(iw), nsites, NSUB), dtype=complex)
    krylov_steps: list[int] = []

    # Second pass: recompute eigenvectors only for sectors that actually carry
    # retained thermal states, then immediately consume them in the resolvents.
    for N in sectors:
        retained = keep[N]
        if not retained:
            continue
        src_solver = RubyEDClusterSolver(L1, L2, params, n_particles=N)
        nneed = max(retained) + 1
        energies, vecs = _lowest_eigensystem(
            src_solver,
            V,
            max(nneed, min(int(opts.thermal_eigs), nneed + 2)),
            float(opts.eig_tol),
            int(opts.eig_maxiter),
        )

        add_solver = None
        rem_solver = None
        add_maps = None
        rem_maps = None
        if N < nsites:
            add_solver = RubyEDClusterSolver(L1, L2, params, n_particles=N + 1)
            add_maps = [
                _transition_map(src_solver, add_solver, i, creation=True)
                for i in range(nsites)
            ]
        if N > 0:
            rem_solver = RubyEDClusterSolver(L1, L2, params, n_particles=N - 1)
            rem_maps = [
                _transition_map(src_solver, rem_solver, i, creation=False)
                for i in range(nsites)
            ]

        for a in retained:
            psi = np.asarray(vecs[:, a], dtype=float)
            Em = float(energies[a])
            weight = float(probs[N][a])
            if opts.verbose:
                print(
                    f"[TL-ED] N={N} state={a}: p={weight:.3e}, E={Em:+.9f}",
                    flush=True,
                )
            for b in range(NSUB):
                if add_solver is not None and add_maps is not None:
                    vadd = _mapped_vector(psi, add_solver.dimension, add_maps[b])
                    fadd, m = _lanczos_cross_resolvent(
                        add_solver,
                        V,
                        vadd,
                        psi,
                        add_maps,
                        iw + Em + mu,
                        max_steps=int(opts.krylov_dim),
                        tol=float(opts.krylov_tol),
                    )
                    columns[:, :, b] += weight * fadd
                    krylov_steps.append(m)

                if rem_solver is not None and rem_maps is not None:
                    vrem = _mapped_vector(psi, rem_solver.dimension, rem_maps[b])
                    # For real H and real many-body eigenvectors the projected
                    # resolvent is symmetric.  The removal Lehmann term is the
                    # transpose projected resolvent and is therefore obtained by
                    # the same six source runs, with an overall minus sign.
                    frem, m = _lanczos_cross_resolvent(
                        rem_solver,
                        V,
                        vrem,
                        psi,
                        rem_maps,
                        Em - mu - iw,
                        max_steps=int(opts.krylov_dim),
                        tol=float(opts.krylov_tol),
                    )
                    columns[:, :, b] -= weight * frem
                    krylov_steps.append(m)

    G = _reconstruct_translation_invariant(columns, L1, L2)
    retained_counts = np.asarray([len(keep[N]) for N in sectors], dtype=int)
    last_gap = np.asarray([
        float(spectra[N][-1] - spectra[N][0]) if len(spectra[N]) else np.nan
        for N in sectors
    ])
    steps = np.asarray(krylov_steps, dtype=int)
    return ThermalLanczosResult(
        G=np.asarray(G),
        mu=float(mu),
        average_particles=float(Navg),
        kept_thermal_weight=float(cumulative),
        sector_numbers=np.asarray(sectors, dtype=int),
        sector_probabilities=sector_p,
        sector_ground_energies=np.asarray([spectra[N][0] for N in sectors], dtype=float),
        sector_last_gap=last_gap,
        retained_states_per_sector=retained_counts,
        krylov_steps_mean=float(np.mean(steps)) if steps.size else 0.0,
        krylov_steps_max=int(np.max(steps)) if steps.size else 0,
        translation_residual=_translation_residual_from_columns(columns),
    )


__all__ = [
    "ThermalLanczosOptions",
    "ThermalLanczosResult",
    "thermal_lanczos_green",
    "_lanczos_cross_resolvent",
    "_transition_map",
]
