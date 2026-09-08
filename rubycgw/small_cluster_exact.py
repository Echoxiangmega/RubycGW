"""Exact finite-temperature Green functions and bilinear correlations on small Ruby tori.

This module is deliberately aimed at clusters small enough for *full* sector-by-sector
many-body diagonalization.  The primary use is the 2x1 Ruby torus (12 sites), where
all particle-number sectors together contain only 2**12=4096 states and the largest
fixed-N block has dimension C(12,6)=924.

The exact grand-canonical quantities implemented here are useful for separating
one-particle background errors from two-particle vertex errors:

    C_exact,
    bubble[G_exact],
    bubble[G_GW],
    cGW[G_GW].

The Hamiltonian is exactly the same microscopic spinless Ruby model used elsewhere:
real hoppings and only the six intra-triangle density-density bonds per primitive
cell.  No interaction bonds are added between triangles/cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ed_cluster import RubyEDClusterSolver, fixed_popcount_basis, _parity_u64
from .model import RubyParameters
from .pseudospin import primitive_pseudospin_vertex


@dataclass
class ExactSector:
    n_particles: int
    basis: np.ndarray
    lookup: np.ndarray
    energies: np.ndarray
    eigenvectors: np.ndarray


@dataclass
class ExactThermalStateSelection:
    probabilities: tuple[np.ndarray, ...]
    kept_indices: tuple[np.ndarray, ...]
    kept_weight: float
    discarded_weight: float
    sector_probabilities: np.ndarray
    average_particles: float
    variance_particles: float


def _fermion_signs(states: np.ndarray, site: int) -> np.ndarray:
    """(-1)^N_<site for an array of bit states."""
    site = int(site)
    if site <= 0:
        return np.ones(len(states), dtype=float)
    mask = np.uint64((1 << site) - 1)
    parity = _parity_u64(np.asarray(states, dtype=np.uint64) & mask)
    return np.where(parity == 0, 1.0, -1.0)


class ExactSmallRubyThermal:
    """Full many-body diagonalization for a small rectangular Ruby PBC torus."""

    def __init__(
        self,
        L1: int = 2,
        L2: int = 1,
        params: RubyParameters = RubyParameters(),
    ):
        self.L1 = int(L1)
        self.L2 = int(L2)
        if self.L1 < 1 or self.L2 < 1:
            raise ValueError("L1 and L2 must be positive")
        self.n_cells = self.L1 * self.L2
        self.n_sites = 6 * self.n_cells
        if self.n_sites > 16:
            raise ValueError(
                "ExactSmallRubyThermal is intended for small dense sectors; "
                "use <=16 sites (the production diagnostic is 12 sites)."
            )
        self.params = params

        # Reuse the already-tested rectangular-torus geometry builder.  N=1 is
        # sufficient because h0 and the interaction-pair list are N-independent.
        geometry = RubyEDClusterSolver(
            self.L1,
            self.L2,
            params,
            n_particles=1,
        )
        self.h0 = np.asarray(geometry.h0, dtype=complex)
        self.interaction_pairs = tuple(geometry.interaction_pairs)
        self.cell_coordinates = np.asarray(geometry.cell_coordinates, dtype=int)

        Vunit = np.zeros((self.n_sites, self.n_sites), dtype=complex)
        for i, j in self.interaction_pairs:
            Vunit[i, j] += 1.0
            Vunit[j, i] += 1.0
        self.Vunit = Vunit

        self._basis_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._sectors: tuple[ExactSector, ...] | None = None
        self._diagonalized_V: float | None = None

    # ------------------------------------------------------------------
    # Basis and exact Hamiltonian blocks
    # ------------------------------------------------------------------
    def _basis_lookup(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        N = int(N)
        cached = self._basis_cache.get(N)
        if cached is not None:
            return cached
        basis = fixed_popcount_basis(self.n_sites, N)
        lookup = np.full(1 << self.n_sites, -1, dtype=np.int32)
        lookup[basis.astype(np.int64)] = np.arange(len(basis), dtype=np.int32)
        self._basis_cache[N] = (basis, lookup)
        return basis, lookup

    def dense_hamiltonian(self, N: int, V: float) -> np.ndarray:
        """Dense exact many-body Hamiltonian in one fixed-N sector."""
        basis, lookup = self._basis_lookup(N)
        dim = len(basis)
        H = np.zeros((dim, dim), dtype=complex)

        # One-body diagonal.
        diag1 = np.diag(self.h0)
        diag_mb = np.zeros(dim, dtype=complex)
        for i in np.flatnonzero(np.abs(diag1) > 1e-14):
            occ = ((basis >> np.uint64(int(i))) & np.uint64(1)).astype(float)
            diag_mb += diag1[int(i)] * occ

        # Density interaction.  Each stored pair is an actual physical intra-
        # triangle bond; Vunit is not used here to avoid a factor-of-two trap.
        pair_count = np.zeros(dim, dtype=float)
        for i, j in self.interaction_pairs:
            oi = (basis >> np.uint64(i)) & np.uint64(1)
            oj = (basis >> np.uint64(j)) & np.uint64(1)
            pair_count += (oi & oj).astype(float)
        H[np.diag_indices(dim)] = diag_mb + float(V) * pair_count

        # Off-diagonal hopping.  The Hermitian one-body matrix may contain
        # different a_ij/a_ji phases in general, so choose the directed amplitude
        # according to which member of the pair is occupied in the source state.
        for i in range(self.n_sites):
            for j in range(i + 1, self.n_sites):
                aij = complex(self.h0[i, j])
                aji = complex(self.h0[j, i])
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
                lo, hi = i, j
                if hi <= lo + 1:
                    parity = np.zeros(src.size, dtype=np.uint8)
                else:
                    between = np.uint64(((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
                    parity = _parity_u64(states & between)
                sign = np.where(parity == 0, 1.0, -1.0)
                j_occ = ((states >> np.uint64(j)) & np.uint64(1)).astype(bool)
                amp = np.where(j_occ, aij, aji)
                H[dst, src] += sign * amp

        herr = float(np.max(np.abs(H - H.conj().T))) if dim else 0.0
        if herr > 1e-11:
            raise RuntimeError(f"exact sector Hamiltonian is not Hermitian: {herr:.3e}")
        return 0.5 * (H + H.conj().T)

    def diagonalize(self, V: float) -> tuple[ExactSector, ...]:
        """Fully diagonalize every particle-number sector."""
        if self._sectors is not None and self._diagonalized_V is not None:
            if abs(float(V) - self._diagonalized_V) <= 1e-14:
                return self._sectors
        sectors: list[ExactSector] = []
        for N in range(self.n_sites + 1):
            basis, lookup = self._basis_lookup(N)
            H = self.dense_hamiltonian(N, V)
            if H.shape == (1, 1):
                e = np.asarray([float(H[0, 0].real)])
                U = np.ones((1, 1), dtype=complex)
            else:
                e, U = np.linalg.eigh(H)
            sectors.append(
                ExactSector(
                    n_particles=N,
                    basis=basis,
                    lookup=lookup,
                    energies=np.asarray(e, dtype=float),
                    eigenvectors=np.asarray(U, dtype=complex),
                )
            )
        self._sectors = tuple(sectors)
        self._diagonalized_V = float(V)
        return self._sectors

    @property
    def sectors(self) -> tuple[ExactSector, ...]:
        if self._sectors is None:
            raise RuntimeError("call diagonalize(V) first")
        return self._sectors

    # ------------------------------------------------------------------
    # Grand-canonical thermodynamics
    # ------------------------------------------------------------------
    def _normalized_probabilities(
        self,
        mu: float,
        T: float,
    ) -> tuple[tuple[np.ndarray, ...], float, float]:
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
        Navg = float(
            sum(sec.n_particles * np.sum(p) for sec, p in zip(self.sectors, probs))
        )
        N2 = float(
            sum((sec.n_particles ** 2) * np.sum(p) for sec, p in zip(self.sectors, probs))
        )
        return probs, Navg, max(0.0, N2 - Navg * Navg)

    def solve_mu(
        self,
        target_particles: float,
        T: float,
        *,
        tol: float = 1e-12,
        max_iter: int = 200,
    ) -> float:
        target = float(target_particles)
        if not 0.0 <= target <= self.n_sites:
            raise ValueError("target_particles out of range")
        if target == 0.0:
            return float(min(sec.energies[0] for sec in self.sectors) - 100.0)
        if target == self.n_sites:
            return float(max(sec.energies[-1] for sec in self.sectors) + 100.0)
        lo, hi = -4.0, 4.0
        for _ in range(40):
            _, nlo, _ = self._normalized_probabilities(lo, T)
            _, nhi, _ = self._normalized_probabilities(hi, T)
            if nlo <= target <= nhi:
                break
            width = hi - lo
            lo -= width
            hi += width
        else:
            raise RuntimeError("could not bracket exact grand-canonical chemical potential")
        mid = 0.5 * (lo + hi)
        for _ in range(int(max_iter)):
            mid = 0.5 * (lo + hi)
            _, nmid, _ = self._normalized_probabilities(mid, T)
            if abs(nmid - target) < float(tol):
                return float(mid)
            if nmid > target:
                hi = mid
            else:
                lo = mid
        return float(mid)

    def thermal_selection(
        self,
        mu: float,
        T: float,
        *,
        discard_weight_tol: float = 1e-12,
    ) -> ExactThermalStateSelection:
        """Select initial states by exact cumulative grand-canonical weight."""
        probs, Navg, Nvar = self._normalized_probabilities(mu, T)
        tol = float(discard_weight_tol)
        if not 0.0 <= tol < 1.0:
            raise ValueError("discard_weight_tol must lie in [0,1)")
        flat = []
        for N, p in enumerate(probs):
            flat.extend((float(x), N, i) for i, x in enumerate(p))
        flat.sort(key=lambda t: t[0], reverse=True)
        target = 1.0 - tol
        cumulative = 0.0
        keep_lists = [[] for _ in probs]
        for weight, N, i in flat:
            keep_lists[N].append(i)
            cumulative += weight
            if cumulative >= target:
                break
        keep = tuple(np.asarray(sorted(x), dtype=int) for x in keep_lists)
        sector_p = np.asarray([float(np.sum(x)) for x in probs], dtype=float)
        kept = float(sum(np.sum(probs[N][idx]) for N, idx in enumerate(keep)))
        return ExactThermalStateSelection(
            probabilities=probs,
            kept_indices=keep,
            kept_weight=kept,
            discarded_weight=max(0.0, 1.0 - kept),
            sector_probabilities=sector_p,
            average_particles=Navg,
            variance_particles=Nvar,
        )

    # ------------------------------------------------------------------
    # Fermion and one-body operator actions on batches of vectors
    # ------------------------------------------------------------------
    def _apply_creation_batch(self, N: int, site: int, vectors: np.ndarray) -> np.ndarray:
        sec = self.sectors[int(N)]
        dstsec = self.sectors[int(N) + 1]
        X = np.asarray(vectors, dtype=complex)
        if X.ndim == 1:
            X = X[:, None]
        occupied = ((sec.basis >> np.uint64(site)) & np.uint64(1)).astype(bool)
        src = np.flatnonzero(~occupied)
        out = np.zeros((len(dstsec.basis), X.shape[1]), dtype=complex)
        if src.size:
            states = sec.basis[src]
            dst_state = states | np.uint64(1 << int(site))
            dst = dstsec.lookup[dst_state.astype(np.int64)]
            sign = _fermion_signs(states, site)
            out[dst] = sign[:, None] * X[src]
        return out

    def _apply_annihilation_batch(self, N: int, site: int, vectors: np.ndarray) -> np.ndarray:
        sec = self.sectors[int(N)]
        dstsec = self.sectors[int(N) - 1]
        X = np.asarray(vectors, dtype=complex)
        if X.ndim == 1:
            X = X[:, None]
        occupied = ((sec.basis >> np.uint64(site)) & np.uint64(1)).astype(bool)
        src = np.flatnonzero(occupied)
        out = np.zeros((len(dstsec.basis), X.shape[1]), dtype=complex)
        if src.size:
            states = sec.basis[src]
            dst_state = states & ~np.uint64(1 << int(site))
            dst = dstsec.lookup[dst_state.astype(np.int64)]
            sign = _fermion_signs(states, site)
            out[dst] = sign[:, None] * X[src]
        return out

    def _apply_onebody_batch(self, N: int, matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        sec = self.sectors[int(N)]
        K = np.asarray(matrix, dtype=complex)
        X = np.asarray(vectors, dtype=complex)
        if X.ndim == 1:
            X = X[:, None]
        if K.shape != (self.n_sites, self.n_sites):
            raise ValueError("one-body matrix shape mismatch")
        if X.shape[0] != len(sec.basis):
            raise ValueError("many-body vector shape mismatch")
        out = np.zeros_like(X, dtype=complex)

        diag = np.diag(K)
        if np.any(np.abs(diag) > 1e-14):
            dmb = np.zeros(len(sec.basis), dtype=complex)
            for i in np.flatnonzero(np.abs(diag) > 1e-14):
                occ = ((sec.basis >> np.uint64(int(i))) & np.uint64(1)).astype(float)
                dmb += diag[int(i)] * occ
            out += dmb[:, None] * X

        for i in range(self.n_sites):
            for j in range(i + 1, self.n_sites):
                aij, aji = K[i, j], K[j, i]
                if abs(aij) <= 1e-14 and abs(aji) <= 1e-14:
                    continue
                bi = (sec.basis >> np.uint64(i)) & np.uint64(1)
                bj = (sec.basis >> np.uint64(j)) & np.uint64(1)
                src = np.flatnonzero(np.asarray(bi ^ bj, dtype=bool))
                if src.size == 0:
                    continue
                states = sec.basis[src]
                dst_state = states ^ np.uint64((1 << i) | (1 << j))
                dst = sec.lookup[dst_state.astype(np.int64)]
                lo, hi = i, j
                if hi <= lo + 1:
                    parity = np.zeros(src.size, dtype=np.uint8)
                else:
                    between = np.uint64(((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
                    parity = _parity_u64(states & between)
                sign = np.where(parity == 0, 1.0, -1.0)
                j_occ = ((states >> np.uint64(j)) & np.uint64(1)).astype(bool)
                amp = np.where(j_occ, aij, aji)
                out[dst] += (sign * amp)[:, None] * X[src]
        return out

    # ------------------------------------------------------------------
    # Exact Green function
    # ------------------------------------------------------------------
    def green_iomega(
        self,
        iomega: np.ndarray,
        mu: float,
        T: float,
        *,
        discard_weight_tol: float = 1e-12,
    ) -> tuple[np.ndarray, ExactThermalStateSelection]:
        """Exact grand-canonical single-particle Green matrix G_ij(iomega)."""
        z = np.asarray(iomega, dtype=complex).reshape(-1)
        selection = self.thermal_selection(
            mu,
            T,
            discard_weight_tol=discard_weight_tol,
        )
        probs = selection.probabilities
        keep = selection.kept_indices
        G = np.zeros((len(z), self.n_sites, self.n_sites), dtype=complex)

        for N in range(self.n_sites):
            low = self.sectors[N]
            up = self.sectors[N + 1]
            xi_low = low.energies - float(mu) * N
            xi_up = up.energies - float(mu) * (N + 1)

            # w_m part of (w_m+w_n)/(iw+xi_m-xi_n): thermal lower states,
            # all upper final states.
            idx_low = keep[N]
            if idx_low.size:
                Vlow = low.eigenvectors[:, idx_low]
                r = len(idx_low)
                created = np.zeros((len(up.basis), self.n_sites * r), dtype=complex)
                for site in range(self.n_sites):
                    created[:, site * r:(site + 1) * r] = self._apply_creation_batch(
                        N, site, Vlow
                    )
                coeff = up.eigenvectors.conj().T @ created
                coeff = coeff.reshape(len(up.basis), self.n_sites, r)
                for a, midx in enumerate(idx_low):
                    B = coeff[:, :, a]  # <n|c_i^dag|m>
                    den = 1.0 / (z[:, None] + xi_low[midx] - xi_up[None, :])
                    G += probs[N][midx] * np.einsum(
                        "ni,wn,nj->wij", B.conj(), den, B, optimize=True
                    )

            # w_n part: thermal upper states, all lower final states.
            idx_up = keep[N + 1]
            if idx_up.size:
                Vup = up.eigenvectors[:, idx_up]
                r = len(idx_up)
                annih = np.zeros((len(low.basis), self.n_sites * r), dtype=complex)
                for site in range(self.n_sites):
                    annih[:, site * r:(site + 1) * r] = self._apply_annihilation_batch(
                        N + 1, site, Vup
                    )
                coeff = low.eigenvectors.conj().T @ annih
                coeff = coeff.reshape(len(low.basis), self.n_sites, r)
                for a, nidx in enumerate(idx_up):
                    A = coeff[:, :, a]  # <m|c_i|n>
                    den = 1.0 / (z[:, None] + xi_low[None, :] - xi_up[nidx])
                    G += probs[N + 1][nidx] * np.einsum(
                        "ni,wn,nj->wij", A, den, A.conj(), optimize=True
                    )

        return G, selection

    # ------------------------------------------------------------------
    # Exact bilinear correlation
    # ------------------------------------------------------------------
    def pseudospin_operator(self, channel: str, q=(0.0, 0.0)) -> np.ndarray:
        """Cell-normalized O_mu(q) in the 12-site one-body basis."""
        q = np.asarray(q, dtype=float).reshape(2)
        if abs(self.L1 * q[0] - round(self.L1 * q[0])) > 1e-10:
            raise ValueError("q1 is not commensurate with the rectangular torus")
        if abs(self.L2 * q[1] - round(self.L2 * q[1])) > 1e-10:
            raise ValueError("q2 is not commensurate with the rectangular torus")
        k6 = np.asarray(primitive_pseudospin_vertex(channel), dtype=complex)
        K = np.zeros((self.n_sites, self.n_sites), dtype=complex)
        norm = np.sqrt(float(self.n_cells))
        for ic, (r1, r2) in enumerate(self.cell_coordinates):
            phase = np.exp(-2j * np.pi * (q[0] * r1 + q[1] * r2)) / norm
            sl = slice(6 * ic, 6 * (ic + 1))
            K[sl, sl] = phase * k6
        return K

    def correlation_tau(
        self,
        operator: np.ndarray,
        tau: np.ndarray,
        mu: float,
        T: float,
        *,
        discard_weight_tol: float = 1e-12,
    ) -> tuple[np.ndarray, float, complex, ExactThermalStateSelection]:
        """Exact connected C(tau) and exact static integral for Hermitian O."""
        O = np.asarray(operator, dtype=complex)
        if O.shape != (self.n_sites, self.n_sites):
            raise ValueError("operator shape mismatch")
        if np.max(np.abs(O - O.conj().T)) > 1e-10:
            raise ValueError("correlation_tau currently expects a Hermitian q=-q operator")
        tau = np.asarray(tau, dtype=float).reshape(-1)
        beta = 1.0 / float(T)
        if np.min(tau) < -1e-12 or np.max(tau) > beta + 1e-12:
            raise ValueError("tau must lie in [0,beta]")
        selection = self.thermal_selection(
            mu,
            T,
            discard_weight_tol=discard_weight_tol,
        )
        C = np.zeros(len(tau), dtype=complex)
        mean = 0.0j
        chi_unc = 0.0

        for N, sec in enumerate(self.sectors):
            idx = selection.kept_indices[N]
            if idx.size == 0:
                continue
            V = sec.eigenvectors[:, idx]
            OV = self._apply_onebody_batch(N, O, V)
            coeff = sec.eigenvectors.conj().T @ OV  # <n|O|m_kept>
            for a, midx in enumerate(idx):
                amp = coeff[:, a]
                p = float(selection.probabilities[N][midx])
                delta = sec.energies - sec.energies[midx]
                spectral = np.abs(amp) ** 2
                C += p * np.einsum(
                    "t,n->t", np.ones(len(tau)), np.zeros(len(delta))
                )  # allocate correct dtype/shape without special casing below
                C -= p * np.einsum(
                    "tn,n->t", np.zeros((len(tau), len(delta))), spectral
                )
                C += p * (np.exp(-tau[:, None] * delta[None, :]) @ spectral)
                mean += p * amp[midx]

                small = np.abs(delta) < 1e-12
                integ = np.empty_like(delta, dtype=float)
                integ[small] = beta
                if np.any(~small):
                    integ[~small] = -np.expm1(-beta * delta[~small]) / delta[~small]
                chi_unc += p * float(np.dot(integ, spectral).real)

        connected = C - abs(mean) ** 2
        chi = float(chi_unc - beta * abs(mean) ** 2)
        return connected, chi, mean, selection


__all__ = [
    "ExactSector",
    "ExactThermalStateSelection",
    "ExactSmallRubyThermal",
]
