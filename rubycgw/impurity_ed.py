"""Finite-temperature exact diagonalization for a small Anderson impurity.

The solver is intentionally generic: an arbitrary finite one-body Hamiltonian is
combined with density-density interactions on a selected subset of orbitals.
It is used by the Ruby 6-site cluster + GW embedding path, where the first six
orbitals are correlated Ruby sites and the remaining orbitals discretize the
self-consistent bath.

For the default 6 correlated + 6 bath construction the full Fock space has only
2**12 = 4096 states, so dense sector-by-sector diagonalization is practical and
keeps the impurity Green function numerically exact for the chosen finite bath.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ed_cluster import fixed_popcount_basis, _parity_u64


@dataclass
class ImpuritySector:
    n_particles: int
    basis: np.ndarray
    lookup: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray


@dataclass
class ImpurityThermalSelection:
    probabilities: tuple[np.ndarray, ...]
    kept_indices: tuple[np.ndarray, ...]
    kept_weight: float
    discarded_weight: float
    average_particles: float


def _fermion_signs(states: np.ndarray, orbital: int) -> np.ndarray:
    orbital = int(orbital)
    if orbital <= 0:
        return np.ones(len(states), dtype=float)
    mask = np.uint64((1 << orbital) - 1)
    parity = _parity_u64(np.asarray(states, dtype=np.uint64) & mask)
    return np.where(parity == 0, 1.0, -1.0)


class FiniteBathImpurityED:
    """Dense finite-temperature ED for a small density-interacting impurity."""

    def __init__(
        self,
        one_body: np.ndarray,
        interaction_terms,
        *,
        correlated_orbitals=None,
    ):
        h = np.asarray(one_body, dtype=complex)
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise ValueError("one_body must be a square matrix")
        if np.max(np.abs(h - h.conj().T), initial=0.0) > 1e-10:
            raise ValueError("one_body must be Hermitian")
        self.one_body = 0.5 * (h + h.conj().T)
        self.n_orbitals = int(h.shape[0])
        if self.n_orbitals > 16:
            raise ValueError("dense finite-bath ED is intended for <=16 orbitals")

        terms = []
        for item in interaction_terms:
            if len(item) != 3:
                raise ValueError("interaction terms must be (i,j,U)")
            i, j, u = int(item[0]), int(item[1]), float(item[2])
            if i == j or not (0 <= i < self.n_orbitals and 0 <= j < self.n_orbitals):
                raise ValueError("invalid density-interaction pair")
            terms.append((i, j, u))
        self.interaction_terms = tuple(terms)

        if correlated_orbitals is None:
            correlated_orbitals = tuple(range(self.n_orbitals))
        corr = tuple(int(x) for x in correlated_orbitals)
        if len(set(corr)) != len(corr) or any(x < 0 or x >= self.n_orbitals for x in corr):
            raise ValueError("invalid correlated_orbitals")
        self.correlated_orbitals = corr

        self._basis_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._sectors: tuple[ImpuritySector, ...] | None = None

    def _basis_lookup(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        N = int(N)
        cached = self._basis_cache.get(N)
        if cached is not None:
            return cached
        basis = fixed_popcount_basis(self.n_orbitals, N)
        lookup = np.full(1 << self.n_orbitals, -1, dtype=np.int32)
        lookup[basis.astype(np.int64)] = np.arange(len(basis), dtype=np.int32)
        self._basis_cache[N] = (basis, lookup)
        return basis, lookup

    def dense_hamiltonian(self, N: int) -> np.ndarray:
        basis, lookup = self._basis_lookup(N)
        dim = len(basis)
        H = np.zeros((dim, dim), dtype=complex)

        diag = np.zeros(dim, dtype=complex)
        onsite = np.diag(self.one_body)
        for i in np.flatnonzero(np.abs(onsite) > 1e-14):
            occ = ((basis >> np.uint64(int(i))) & np.uint64(1)).astype(float)
            diag += onsite[int(i)] * occ
        for i, j, u in self.interaction_terms:
            oi = (basis >> np.uint64(i)) & np.uint64(1)
            oj = (basis >> np.uint64(j)) & np.uint64(1)
            diag += float(u) * (oi & oj).astype(float)
        H[np.diag_indices(dim)] = diag

        for i in range(self.n_orbitals):
            for j in range(i + 1, self.n_orbitals):
                aij = complex(self.one_body[i, j])
                aji = complex(self.one_body[j, i])
                if abs(aij) <= 1e-14 and abs(aji) <= 1e-14:
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
                    raise RuntimeError("fixed-N state lookup failed")
                if j <= i + 1:
                    parity = np.zeros(src.size, dtype=np.uint8)
                else:
                    between = np.uint64(((1 << j) - 1) ^ ((1 << (i + 1)) - 1))
                    parity = _parity_u64(states & between)
                sign = np.where(parity == 0, 1.0, -1.0)
                j_occ = ((states >> np.uint64(j)) & np.uint64(1)).astype(bool)
                amp = np.where(j_occ, aij, aji)
                H[dst, src] += sign * amp

        herr = float(np.max(np.abs(H - H.conj().T), initial=0.0))
        if herr > 1e-10:
            raise RuntimeError(f"impurity Hamiltonian is not Hermitian: {herr:.3e}")
        return 0.5 * (H + H.conj().T)

    def diagonalize(self) -> tuple[ImpuritySector, ...]:
        if self._sectors is not None:
            return self._sectors
        sectors = []
        for N in range(self.n_orbitals + 1):
            basis, lookup = self._basis_lookup(N)
            H = self.dense_hamiltonian(N)
            if H.shape == (1, 1):
                e = np.asarray([float(H[0, 0].real)])
                U = np.ones((1, 1), dtype=complex)
            else:
                e, U = np.linalg.eigh(H)
            sectors.append(
                ImpuritySector(
                    n_particles=N,
                    basis=basis,
                    lookup=lookup,
                    energies=np.asarray(e, dtype=float),
                    eigenvectors=np.asarray(U, dtype=complex),
                )
            )
        self._sectors = tuple(sectors)
        return self._sectors

    @property
    def sectors(self) -> tuple[ImpuritySector, ...]:
        if self._sectors is None:
            raise RuntimeError("call diagonalize() first")
        return self._sectors

    def _normalized_probabilities(self, mu: float, T: float):
        if T <= 0.0:
            raise ValueError("T must be positive")
        beta = 1.0 / float(T)
        logs = [
            -beta * (sec.energies - float(mu) * sec.n_particles)
            for sec in self.sectors
        ]
        shift = max(float(np.max(x)) for x in logs)
        raw = [np.exp(x - shift) for x in logs]
        Z = float(sum(np.sum(x) for x in raw))
        probs = tuple(np.asarray(x / Z, dtype=float) for x in raw)
        navg = float(
            sum(sec.n_particles * np.sum(p) for sec, p in zip(self.sectors, probs))
        )
        return probs, navg

    def thermal_selection(
        self,
        mu: float,
        T: float,
        *,
        discard_weight_tol: float = 1e-12,
    ) -> ImpurityThermalSelection:
        probs, navg = self._normalized_probabilities(mu, T)
        tol = float(discard_weight_tol)
        if not 0.0 <= tol < 1.0:
            raise ValueError("discard_weight_tol must lie in [0,1)")
        flat = []
        for N, p in enumerate(probs):
            flat.extend((float(x), N, i) for i, x in enumerate(p))
        flat.sort(key=lambda t: t[0], reverse=True)
        keep_lists = [[] for _ in probs]
        cumulative = 0.0
        for weight, N, i in flat:
            keep_lists[N].append(i)
            cumulative += weight
            if cumulative >= 1.0 - tol:
                break
        keep = tuple(np.asarray(sorted(x), dtype=int) for x in keep_lists)
        kept = float(sum(np.sum(probs[N][idx]) for N, idx in enumerate(keep)))
        return ImpurityThermalSelection(
            probabilities=probs,
            kept_indices=keep,
            kept_weight=kept,
            discarded_weight=max(0.0, 1.0 - kept),
            average_particles=navg,
        )

    def _apply_creation_batch(self, N: int, orbital: int, vectors: np.ndarray) -> np.ndarray:
        sec = self.sectors[int(N)]
        dstsec = self.sectors[int(N) + 1]
        X = np.asarray(vectors, dtype=complex)
        if X.ndim == 1:
            X = X[:, None]
        occupied = ((sec.basis >> np.uint64(orbital)) & np.uint64(1)).astype(bool)
        src = np.flatnonzero(~occupied)
        out = np.zeros((len(dstsec.basis), X.shape[1]), dtype=complex)
        if src.size:
            states = sec.basis[src]
            dst_state = states | np.uint64(1 << int(orbital))
            dst = dstsec.lookup[dst_state.astype(np.int64)]
            sign = _fermion_signs(states, orbital)
            out[dst] = sign[:, None] * X[src]
        return out

    def _apply_annihilation_batch(self, N: int, orbital: int, vectors: np.ndarray) -> np.ndarray:
        sec = self.sectors[int(N)]
        dstsec = self.sectors[int(N) - 1]
        X = np.asarray(vectors, dtype=complex)
        if X.ndim == 1:
            X = X[:, None]
        occupied = ((sec.basis >> np.uint64(orbital)) & np.uint64(1)).astype(bool)
        src = np.flatnonzero(occupied)
        out = np.zeros((len(dstsec.basis), X.shape[1]), dtype=complex)
        if src.size:
            states = sec.basis[src]
            dst_state = states & ~np.uint64(1 << int(orbital))
            dst = dstsec.lookup[dst_state.astype(np.int64)]
            sign = _fermion_signs(states, orbital)
            out[dst] = sign[:, None] * X[src]
        return out

    def green_iomega(
        self,
        iomega: np.ndarray,
        mu: float,
        T: float,
        *,
        orbitals=None,
        discard_weight_tol: float = 1e-12,
    ) -> tuple[np.ndarray, ImpurityThermalSelection]:
        """Grand-canonical Matsubara Green matrix on selected orbitals."""
        if self._sectors is None:
            self.diagonalize()
        z = np.asarray(iomega, dtype=complex).reshape(-1)
        sites = self.correlated_orbitals if orbitals is None else tuple(int(x) for x in orbitals)
        selection = self.thermal_selection(mu, T, discard_weight_tol=discard_weight_tol)
        probs = selection.probabilities
        keep = selection.kept_indices
        G = np.zeros((len(z), len(sites), len(sites)), dtype=complex)

        for N in range(self.n_orbitals):
            low = self.sectors[N]
            up = self.sectors[N + 1]
            xi_low = low.energies - float(mu) * N
            xi_up = up.energies - float(mu) * (N + 1)

            idx_low = keep[N]
            if idx_low.size:
                Vlow = low.eigenvectors[:, idx_low]
                r = len(idx_low)
                created = np.zeros((len(up.basis), len(sites) * r), dtype=complex)
                for a, site in enumerate(sites):
                    created[:, a * r:(a + 1) * r] = self._apply_creation_batch(N, site, Vlow)
                coeff = up.eigenvectors.conj().T @ created
                coeff = coeff.reshape(len(up.basis), len(sites), r)
                for a, midx in enumerate(idx_low):
                    B = coeff[:, :, a]
                    den = 1.0 / (z[:, None] + xi_low[midx] - xi_up[None, :])
                    G += probs[N][midx] * np.einsum(
                        "ni,wn,nj->wij", B.conj(), den, B, optimize=True
                    )

            idx_up = keep[N + 1]
            if idx_up.size:
                Vup = up.eigenvectors[:, idx_up]
                r = len(idx_up)
                annih = np.zeros((len(low.basis), len(sites) * r), dtype=complex)
                for a, site in enumerate(sites):
                    annih[:, a * r:(a + 1) * r] = self._apply_annihilation_batch(N + 1, site, Vup)
                coeff = low.eigenvectors.conj().T @ annih
                coeff = coeff.reshape(len(low.basis), len(sites), r)
                for a, nidx in enumerate(idx_up):
                    A = coeff[:, :, a]
                    den = 1.0 / (z[:, None] + xi_low[None, :] - xi_up[nidx])
                    G += probs[N + 1][nidx] * np.einsum(
                        "ni,wn,nj->wij", A, den, A.conj(), optimize=True
                    )

        return G, selection


__all__ = [
    "ImpuritySector",
    "ImpurityThermalSelection",
    "FiniteBathImpurityED",
]
