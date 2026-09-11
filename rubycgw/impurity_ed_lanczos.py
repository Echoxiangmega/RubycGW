"""Sparse low-temperature impurity ED with Krylov Green-function resolvents.

The historical finite-bath impurity solver diagonalizes every fixed-particle
sector densely.  That is excellent for 6 correlated + 6 bath orbitals, but the
cost rises sharply for 8 or more bath orbitals.  This module keeps the same
Hamiltonian and Matsubara Green-function definition while replacing the full
many-body spectrum by:

1. sparse lowest-eigenstate calculations in thermally relevant N sectors;
2. block-Krylov resolvents for the N->N+/-1 propagators, avoiding a full target
   sector eigendecomposition.

The approximation is controlled by the thermal truncation and Krylov settings.
For small Hilbert spaces the code automatically falls back to exact dense
sector diagonalization, which also provides a regression path against the
historical solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

from .impurity_ed import (
    FiniteBathImpurityED,
    ImpurityThermalSelection,
    _fermion_signs,
)
from .ed_cluster import _parity_u64


@dataclass
class SparseSector:
    n_particles: int
    basis: np.ndarray
    lookup: np.ndarray
    hamiltonian: csr_matrix


class FiniteBathImpurityLanczosED(FiniteBathImpurityED):
    """Low-temperature sparse/Lanczos replacement for ``FiniteBathImpurityED``."""

    def __init__(
        self,
        one_body: np.ndarray,
        interaction_terms,
        *,
        correlated_orbitals=None,
        thermal_state_tol: float = 1.0e-10,
        thermal_initial_states: int = 8,
        thermal_max_states: int = 64,
        dense_sector_threshold: int = 160,
        eig_tol: float = 1.0e-10,
        eig_maxiter: int = 5000,
        krylov_steps: int = 18,
        krylov_tol: float = 1.0e-11,
    ):
        super().__init__(
            one_body,
            interaction_terms,
            correlated_orbitals=correlated_orbitals,
        )
        self.thermal_state_tol = float(thermal_state_tol)
        self.thermal_initial_states = max(int(thermal_initial_states), 1)
        self.thermal_max_states = max(int(thermal_max_states), self.thermal_initial_states)
        self.dense_sector_threshold = max(int(dense_sector_threshold), 2)
        self.eig_tol = float(eig_tol)
        self.eig_maxiter = int(eig_maxiter)
        self.krylov_steps = max(int(krylov_steps), 2)
        self.krylov_tol = float(krylov_tol)
        self._sparse_sectors: tuple[SparseSector, ...] | None = None
        self._low_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.last_thermal_truncation_bound = np.nan
        self.last_krylov_max_dim = 0

    def sparse_hamiltonian(self, N: int) -> csr_matrix:
        basis, lookup = self._basis_lookup(int(N))
        dim = len(basis)
        rows = []
        cols = []
        data = []

        diag = np.zeros(dim, dtype=complex)
        onsite = np.diag(self.one_body)
        for i in np.flatnonzero(np.abs(onsite) > 1.0e-14):
            occ = ((basis >> np.uint64(int(i))) & np.uint64(1)).astype(float)
            diag += onsite[int(i)] * occ
        for i, j, u in self.interaction_terms:
            oi = (basis >> np.uint64(i)) & np.uint64(1)
            oj = (basis >> np.uint64(j)) & np.uint64(1)
            diag += float(u) * (oi & oj).astype(float)
        didx = np.arange(dim, dtype=np.int64)
        rows.extend(didx.tolist())
        cols.extend(didx.tolist())
        data.extend(diag.tolist())

        for i in range(self.n_orbitals):
            for j in range(i + 1, self.n_orbitals):
                aij = complex(self.one_body[i, j])
                aji = complex(self.one_body[j, i])
                if abs(aij) <= 1.0e-14 and abs(aji) <= 1.0e-14:
                    continue
                bi = (basis >> np.uint64(i)) & np.uint64(1)
                bj = (basis >> np.uint64(j)) & np.uint64(1)
                src = np.flatnonzero(np.asarray(bi ^ bj, dtype=bool))
                if src.size == 0:
                    continue
                states = basis[src]
                dst_states = states ^ np.uint64((1 << i) | (1 << j))
                dst = lookup[dst_states.astype(np.int64)]
                if np.any(dst < 0):
                    raise RuntimeError("fixed-N sparse state lookup failed")
                if j <= i + 1:
                    parity = np.zeros(src.size, dtype=np.uint8)
                else:
                    between = np.uint64(((1 << j) - 1) ^ ((1 << (i + 1)) - 1))
                    parity = _parity_u64(states & between)
                sign = np.where(parity == 0, 1.0, -1.0)
                j_occ = ((states >> np.uint64(j)) & np.uint64(1)).astype(bool)
                amp = np.where(j_occ, aij, aji)
                rows.extend(dst.astype(np.int64).tolist())
                cols.extend(src.astype(np.int64).tolist())
                data.extend((sign * amp).tolist())

        H = coo_matrix((np.asarray(data, dtype=complex), (rows, cols)), shape=(dim, dim)).tocsr()
        H = 0.5 * (H + H.getH())
        return H

    def diagonalize(self):
        """Build sparse fixed-N Hamiltonians; eigenstates are generated lazily."""
        if self._sparse_sectors is None:
            sectors = []
            for N in range(self.n_orbitals + 1):
                basis, lookup = self._basis_lookup(N)
                sectors.append(
                    SparseSector(
                        n_particles=N,
                        basis=basis,
                        lookup=lookup,
                        hamiltonian=self.sparse_hamiltonian(N),
                    )
                )
            self._sparse_sectors = tuple(sectors)
        return self._sparse_sectors

    @property
    def sparse_sectors(self) -> tuple[SparseSector, ...]:
        if self._sparse_sectors is None:
            self.diagonalize()
        return self._sparse_sectors

    def _lowest_states(self, N: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        sec = self.sparse_sectors[int(N)]
        dim = sec.hamiltonian.shape[0]
        k = min(max(int(k), 1), dim)
        cached = self._low_cache.get(int(N))
        if cached is not None and len(cached[0]) >= k:
            return cached[0][:k], cached[1][:, :k]

        if dim <= self.dense_sector_threshold or k >= dim:
            H = sec.hamiltonian.toarray()
            e, U = np.linalg.eigh(H)
        elif k >= dim - 1:
            H = sec.hamiltonian.toarray()
            e, U = np.linalg.eigh(H)
        else:
            e, U = eigsh(
                sec.hamiltonian,
                k=k,
                which="SA",
                tol=self.eig_tol,
                maxiter=self.eig_maxiter,
            )
            order = np.argsort(e)
            e = np.asarray(e[order].real, dtype=float)
            U = np.asarray(U[:, order], dtype=complex)
        self._low_cache[int(N)] = (np.asarray(e, dtype=float), np.asarray(U, dtype=complex))
        return self._low_cache[int(N)][0][:k], self._low_cache[int(N)][1][:, :k]

    def _thermal_low_states(self, mu: float, T: float, discard_weight_tol: float):
        if T <= 0.0:
            raise ValueError("T must be positive")
        tol = max(float(discard_weight_tol), self.thermal_state_tol)
        nsec = self.n_orbitals + 1

        e0 = np.empty(nsec, dtype=float)
        for N in range(nsec):
            e, _ = self._lowest_states(N, 1)
            e0[N] = float(e[0])
        grand0 = float(np.min(e0 - float(mu) * np.arange(nsec)))

        energies = []
        vectors = []
        omitted_bounds = []
        per_sector_tol = tol / float(nsec)
        for N in range(nsec):
            sec = self.sparse_sectors[N]
            dim = sec.hamiltonian.shape[0]
            sector_ground_scaled = np.exp(
                -max((e0[N] - float(mu) * N - grand0) / float(T), 0.0)
            )
            if dim * sector_ground_scaled < per_sector_tol:
                energies.append(np.empty(0, dtype=float))
                vectors.append(np.empty((dim, 0), dtype=complex))
                omitted_bounds.append(dim * sector_ground_scaled)
                continue

            k = min(self.thermal_initial_states, dim)
            while True:
                e, U = self._lowest_states(N, k)
                if k >= dim:
                    bound = 0.0
                    break
                edge = float(e[-1] - float(mu) * N - grand0)
                bound = max(dim - k, 0) * np.exp(-max(edge / float(T), 0.0))
                if bound <= per_sector_tol or k >= self.thermal_max_states:
                    break
                knew = min(max(2 * k, k + 1), self.thermal_max_states, dim)
                if knew == k:
                    break
                k = knew
            energies.append(np.asarray(e, dtype=float))
            vectors.append(np.asarray(U, dtype=complex))
            omitted_bounds.append(float(bound))

        scaled_weights = []
        for N, e in enumerate(energies):
            if len(e) == 0:
                scaled_weights.append(np.empty(0, dtype=float))
            else:
                scaled_weights.append(
                    np.exp(-(e - float(mu) * N - grand0) / float(T))
                )
        zlow = max(float(sum(np.sum(x) for x in scaled_weights)), 1.0e-300)
        omitted_rel = float(sum(omitted_bounds) / zlow)
        self.last_thermal_truncation_bound = omitted_rel

        probs = tuple(np.asarray(x / zlow, dtype=float) for x in scaled_weights)
        flat = []
        for N, p in enumerate(probs):
            flat.extend((float(val), N, i) for i, val in enumerate(p))
        flat.sort(key=lambda x: x[0], reverse=True)
        keep_lists = [[] for _ in range(nsec)]
        cumulative = 0.0
        target_keep = max(0.0, 1.0 - float(discard_weight_tol) - omitted_rel)
        for weight, N, i in flat:
            keep_lists[N].append(i)
            cumulative += weight
            if cumulative >= target_keep:
                break
        keep = tuple(np.asarray(sorted(x), dtype=int) for x in keep_lists)
        kept = float(sum(np.sum(probs[N][idx]) for N, idx in enumerate(keep)))
        navg = float(sum(N * np.sum(p) for N, p in enumerate(probs)))
        selection = ImpurityThermalSelection(
            probabilities=probs,
            kept_indices=keep,
            kept_weight=kept,
            discarded_weight=max(0.0, 1.0 - kept),
            average_particles=navg,
        )
        return energies, vectors, selection

    def _apply_creation_state(self, N: int, orbital: int, vector: np.ndarray) -> np.ndarray:
        sec = self.sparse_sectors[int(N)]
        dstsec = self.sparse_sectors[int(N) + 1]
        v = np.asarray(vector, dtype=complex).reshape(-1)
        occupied = ((sec.basis >> np.uint64(orbital)) & np.uint64(1)).astype(bool)
        src = np.flatnonzero(~occupied)
        out = np.zeros(len(dstsec.basis), dtype=complex)
        if src.size:
            states = sec.basis[src]
            dst_state = states | np.uint64(1 << int(orbital))
            dst = dstsec.lookup[dst_state.astype(np.int64)]
            sign = _fermion_signs(states, orbital)
            out[dst] = sign * v[src]
        return out

    def _apply_annihilation_state(self, N: int, orbital: int, vector: np.ndarray) -> np.ndarray:
        sec = self.sparse_sectors[int(N)]
        dstsec = self.sparse_sectors[int(N) - 1]
        v = np.asarray(vector, dtype=complex).reshape(-1)
        occupied = ((sec.basis >> np.uint64(orbital)) & np.uint64(1)).astype(bool)
        src = np.flatnonzero(occupied)
        out = np.zeros(len(dstsec.basis), dtype=complex)
        if src.size:
            states = sec.basis[src]
            dst_state = states & ~np.uint64(1 << int(orbital))
            dst = dstsec.lookup[dst_state.astype(np.int64)]
            sign = _fermion_signs(states, orbital)
            out[dst] = sign * v[src]
        return out

    def _block_krylov_resolvent(
        self,
        H: csr_matrix,
        B: np.ndarray,
        shifts: np.ndarray,
    ) -> np.ndarray:
        """Approximate B^dagger (s-H)^-1 B for many complex shifts."""
        B = np.asarray(B, dtype=complex)
        shifts = np.asarray(shifts, dtype=complex).reshape(-1)
        m = B.shape[1]
        if B.size == 0 or np.linalg.norm(B) < 1.0e-15:
            return np.zeros((len(shifts), m, m), dtype=complex)

        U0, s0, _ = np.linalg.svd(B, full_matrices=False)
        if len(s0) == 0 or s0[0] <= 1.0e-15:
            return np.zeros((len(shifts), m, m), dtype=complex)
        r0 = int(np.sum(s0 > max(self.krylov_tol * s0[0], 1.0e-14)))
        Qblocks = [U0[:, :r0]]
        current = Qblocks[0]

        for _ in range(1, self.krylov_steps):
            W = H @ current
            Qall = np.hstack(Qblocks)
            # Two-pass full reorthogonalization is cheap at these block sizes and
            # keeps the projected Hermitian problem numerically stable.
            for _pass in range(2):
                W = W - Qall @ (Qall.conj().T @ W)
            Uw, sw, _ = np.linalg.svd(W, full_matrices=False)
            if len(sw) == 0 or sw[0] <= 1.0e-14:
                break
            rw = int(np.sum(sw > max(self.krylov_tol * sw[0], 1.0e-13)))
            if rw == 0:
                break
            current = Uw[:, :rw]
            Qblocks.append(current)
            if sum(q.shape[1] for q in Qblocks) >= H.shape[0]:
                break

        Q = np.hstack(Qblocks)
        self.last_krylov_max_dim = max(self.last_krylov_max_dim, int(Q.shape[1]))
        HQ = H @ Q
        Tproj = Q.conj().T @ HQ
        Tproj = 0.5 * (Tproj + Tproj.conj().T)
        et, Ut = np.linalg.eigh(Tproj)
        C = Q.conj().T @ B
        D = Ut.conj().T @ C
        den = 1.0 / (shifts[:, None] - et[None, :])
        return np.einsum("ia,wi,ib->wab", D.conj(), den, D, optimize=True)

    def green_iomega(
        self,
        iomega: np.ndarray,
        mu: float,
        T: float,
        *,
        orbitals=None,
        discard_weight_tol: float = 1.0e-12,
    ):
        if self._sparse_sectors is None:
            self.diagonalize()
        z = np.asarray(iomega, dtype=complex).reshape(-1)
        sites = self.correlated_orbitals if orbitals is None else tuple(int(x) for x in orbitals)
        energies, vectors, selection = self._thermal_low_states(
            float(mu), float(T), float(discard_weight_tol)
        )
        G = np.zeros((len(z), len(sites), len(sites)), dtype=complex)

        for N in range(self.n_orbitals + 1):
            idx = selection.kept_indices[N]
            if idx.size == 0:
                continue
            eN = energies[N]
            UN = vectors[N]
            pN = selection.probabilities[N]
            for midx in idx:
                psi = UN[:, int(midx)]
                Em = float(eN[int(midx)])
                weight = float(pN[int(midx)])

                if N < self.n_orbitals:
                    B = np.column_stack([
                        self._apply_creation_state(N, site, psi) for site in sites
                    ])
                    shifts = z + Em + float(mu)
                    G += weight * self._block_krylov_resolvent(
                        self.sparse_sectors[N + 1].hamiltonian, B, shifts
                    )

                if N > 0:
                    A = np.column_stack([
                        self._apply_annihilation_state(N, site, psi) for site in sites
                    ])
                    shifts = Em - float(mu) - z
                    rem = self._block_krylov_resolvent(
                        self.sparse_sectors[N - 1].hamiltonian, A, shifts
                    )
                    G += weight * (-np.swapaxes(rem, 1, 2))

        return G, selection


def install_impurity_solver(
    *,
    mode: str = "auto",
    nbath: int = 6,
    thermal_state_tol: float = 1.0e-10,
    thermal_max_states: int = 64,
    krylov_steps: int = 18,
):
    """Install dense or sparse impurity ED into production embedding modules."""
    key = str(mode).lower()
    if key not in {"auto", "dense", "lanczos"}:
        raise ValueError("impurity solver mode must be auto, dense, or lanczos")
    use_lanczos = key == "lanczos" or (key == "auto" and int(nbath) >= 7)

    if use_lanczos:
        class ConfiguredLanczos(FiniteBathImpurityLanczosED):
            def __init__(self, one_body, interaction_terms, *, correlated_orbitals=None):
                super().__init__(
                    one_body,
                    interaction_terms,
                    correlated_orbitals=correlated_orbitals,
                    thermal_state_tol=thermal_state_tol,
                    thermal_max_states=thermal_max_states,
                    krylov_steps=krylov_steps,
                )
        solver_cls = ConfiguredLanczos
    else:
        solver_cls = FiniteBathImpurityED

    from . import cluster_ed_gw as base
    from . import cluster_ed_gw_fast as gw_fast
    from . import cluster_ed_weak_fast as weak_fast
    from . import cluster_ed_weak_warm as warm

    base.FiniteBathImpurityED = solver_cls
    gw_fast.FiniteBathImpurityED = solver_cls
    weak_fast.FiniteBathImpurityED = solver_cls
    warm.FiniteBathImpurityED = solver_cls
    return solver_cls, ("lanczos" if use_lanczos else "dense")


__all__ = [
    "SparseSector",
    "FiniteBathImpurityLanczosED",
    "install_impurity_solver",
]
