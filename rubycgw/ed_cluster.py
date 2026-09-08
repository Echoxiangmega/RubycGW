"""Exact diagonalization on rectangular periodic Ruby clusters.

This module complements the specialized three-cell :mod:`rubycgw.ed18` code.
It is intended for ordinary ``L1 x L2`` primitive-cell tori, in particular the
2x2 (24-site) cluster whose primitive momenta are Gamma and the three M points.

The main design goal is to make 24-site fixed-N Lanczos practical without
constructing a gigantic many-body sparse Hamiltonian.  The Hilbert basis is an
ascending fixed-popcount bit basis.  Hopping is applied matrix-free using
cached fermionic transition source lists, while the density interaction is a
small integer diagonal.  For Nsite<=24 a dense state->basis-index lookup table
uses only 64 MiB at 24 sites and avoids Python dictionaries with millions of
entries.

The solver is exact for the chosen finite torus.  It does not imply a
thermodynamic-limit result; use different cluster shapes/sizes when assessing
finite-size trends.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .model import NSUB, RubyParameters, ruby_hoppings, ruby_interaction_bonds
from .pseudospin import primitive_pseudospin_vertex


ED_CLUSTER_CHANNELS = (
    "x_even",
    "x_odd",
    "y_even",
    "y_odd",
    "z_same",
    "z_opposite",
)


@dataclass
class EDClusterSpectrum:
    V: float
    energies: np.ndarray
    eigenvectors: np.ndarray
    ground_multiplicity: int
    gap_above_manifold: float
    interaction_expectation: float


@dataclass
class EDClusterStructureFactor:
    q: np.ndarray
    channels: tuple[str, ...]
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    mean: np.ndarray


def fixed_popcount_basis(n_sites: int, n_particles: int) -> np.ndarray:
    """Ascending integer bit basis with exactly ``n_particles`` occupied sites.

    The implementation uses Gosper's next-combination rule, so integer order is
    exact and state lookup can use either a dense table or ``searchsorted``.
    """
    n_sites = int(n_sites)
    n_particles = int(n_particles)
    if not 0 <= n_particles <= n_sites:
        raise ValueError("n_particles must satisfy 0 <= Nf <= n_sites")
    dim = comb(n_sites, n_particles)
    out = np.empty(dim, dtype=np.uint64)
    if dim == 0:
        return out
    if n_particles == 0:
        out[0] = np.uint64(0)
        return out
    if n_particles == n_sites:
        out[0] = np.uint64((1 << n_sites) - 1)
        return out

    x = (1 << n_particles) - 1
    limit = 1 << n_sites
    i = 0
    while x < limit:
        out[i] = np.uint64(x)
        i += 1
        c = x & -x
        r = x + c
        x = (((r ^ x) >> 2) // c) | r
    if i != dim:
        raise RuntimeError(f"fixed-popcount basis mismatch: generated {i}, expected {dim}")
    return out


def _parity_u64(values: np.ndarray) -> np.ndarray:
    """Vectorized bit parity for uint64 arrays, returned as uint8 0/1."""
    x = np.asarray(values, dtype=np.uint64).copy()
    x ^= x >> np.uint64(32)
    x ^= x >> np.uint64(16)
    x ^= x >> np.uint64(8)
    x ^= x >> np.uint64(4)
    x ^= x >> np.uint64(2)
    x ^= x >> np.uint64(1)
    return np.asarray(x & np.uint64(1), dtype=np.uint8)


def _between_mask(i: int, j: int) -> np.uint64:
    lo, hi = sorted((int(i), int(j)))
    if hi <= lo + 1:
        return np.uint64(0)
    return np.uint64(((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))


def rectangular_momenta(L1: int, L2: int) -> tuple[np.ndarray, list[str]]:
    """All primitive reduced momenta compatible with an L1 x L2 PBC torus."""
    L1, L2 = int(L1), int(L2)
    if L1 < 1 or L2 < 1:
        raise ValueError("L1 and L2 must be positive")
    qpts: list[list[float]] = []
    labels: list[str] = []
    for m1 in range(L1):
        for m2 in range(L2):
            q = [m1 / float(L1), m2 / float(L2)]
            qpts.append(q)
            if m1 == 0 and m2 == 0:
                labels.append("Gamma")
            elif L1 == 2 and L2 == 2:
                if (m1, m2) == (1, 0):
                    labels.append("M1")
                elif (m1, m2) == (0, 1):
                    labels.append("M2")
                else:
                    labels.append("M3")
            else:
                labels.append(f"q({m1}/{L1},{m2}/{L2})")
    return np.asarray(qpts, dtype=float), labels


class RubyEDClusterSolver:
    """Matrix-free fixed-particle ED for an ordinary rectangular Ruby torus."""

    def __init__(
        self,
        L1: int,
        L2: int,
        params: RubyParameters = RubyParameters(),
        *,
        primitive_filling: float = 2.0,
        n_particles: int | None = None,
        dense_lookup_max_sites: int = 24,
    ):
        self.L1 = int(L1)
        self.L2 = int(L2)
        if self.L1 < 1 or self.L2 < 1:
            raise ValueError("L1 and L2 must be positive")
        self.n_cells = self.L1 * self.L2
        self.n_sites = NSUB * self.n_cells
        if self.n_sites > 63:
            raise ValueError("bit-basis implementation supports at most 63 sites")
        self.params = params

        if n_particles is None:
            nf_float = float(primitive_filling) * self.n_cells
            nf = int(round(nf_float))
            if abs(nf_float - nf) > 1e-12:
                raise ValueError("primitive_filling*L1*L2 must be an integer for fixed-N ED")
            n_particles = nf
        self.n_particles = int(n_particles)
        if not 0 <= self.n_particles <= self.n_sites:
            raise ValueError("invalid particle number")
        self.primitive_filling = self.n_particles / float(self.n_cells)

        self.basis = fixed_popcount_basis(self.n_sites, self.n_particles)
        self.dimension = int(len(self.basis))
        if self.dimension < 2:
            raise ValueError("nontrivial Lanczos ED requires Hilbert-space dimension >= 2")
        self._index_dtype = np.int32 if self.dimension < np.iinfo(np.int32).max else np.int64

        self._dense_lookup = None
        if self.n_sites <= int(dense_lookup_max_sites):
            lookup = np.full(1 << self.n_sites, -1, dtype=self._index_dtype)
            lookup[self.basis.astype(np.int64)] = np.arange(self.dimension, dtype=self._index_dtype)
            self._dense_lookup = lookup

        self.cell_coordinates = np.asarray(
            [(r1, r2) for r1 in range(self.L1) for r2 in range(self.L2)],
            dtype=int,
        )
        self.h0 = self._build_cluster_h0()
        if np.max(np.abs(self.h0 - self.h0.conj().T)) > 1e-12:
            raise RuntimeError("cluster one-body Hamiltonian is not Hermitian")
        if np.max(np.abs(self.h0.imag)) > 1e-12:
            raise RuntimeError("current RubyEDClusterSolver expects real hopping amplitudes")
        self.h0 = np.asarray(self.h0.real, dtype=float)

        self._hopping_pairs: list[tuple[int, int, float]] = []
        for i in range(self.n_sites):
            for j in range(i + 1, self.n_sites):
                amp = float(self.h0[i, j])
                if abs(amp) > 1e-14:
                    self._hopping_pairs.append((i, j, amp))
        self._onsite = np.diag(self.h0).copy()
        self._h_onsite_mb = self._manybody_diagonal(self._onsite)

        self.interaction_pairs = self._build_interaction_pairs()
        max_bonds = len(self.interaction_pairs)
        count_dtype = np.uint8 if max_bonds <= np.iinfo(np.uint8).max else np.uint16
        counts = np.zeros(self.dimension, dtype=count_dtype)
        for i, j in self.interaction_pairs:
            occ = ((self.basis >> np.uint64(i)) & np.uint64(1)) & (
                (self.basis >> np.uint64(j)) & np.uint64(1)
            )
            counts += occ.astype(count_dtype)
        self.interaction_count = counts

        # Cache per unordered site pair.  Each entry stores source basis indices
        # with positive/negative fermion parity; destinations are reconstructed
        # by XOR + the dense/searchsorted state lookup to save memory.
        self._pair_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.uint64]] = {}
        self._operator_matrix_cache: dict[tuple[str, float, float], np.ndarray] = {}

    def _cell_index(self, r1: int, r2: int) -> int:
        return (int(r1) % self.L1) * self.L2 + (int(r2) % self.L2)

    def _site(self, r1: int, r2: int, a: int) -> int:
        return NSUB * self._cell_index(r1, r2) + int(a)

    def _build_cluster_h0(self) -> np.ndarray:
        p0 = RubyParameters(ti=self.params.ti, t1=self.params.t1, t2=self.params.t2, V=0.0)
        h = np.zeros((self.n_sites, self.n_sites), dtype=complex)
        for r1 in range(self.L1):
            for r2 in range(self.L2):
                for i, j, R, amp in ruby_hoppings(p0):
                    I = self._site(r1, r2, int(i))
                    J = self._site(r1 + int(R[0]), r2 + int(R[1]), int(j))
                    h[I, J] += complex(amp)
        return 0.5 * (h + h.conj().T)

    def _build_interaction_pairs(self) -> tuple[tuple[int, int], ...]:
        p1 = RubyParameters(ti=self.params.ti, t1=self.params.t1, t2=self.params.t2, V=1.0)
        pairs: set[tuple[int, int]] = set()
        for r1 in range(self.L1):
            for r2 in range(self.L2):
                for i, j, R, coupling in ruby_interaction_bonds(p1):
                    if np.any(np.asarray(R, dtype=int) != 0):
                        raise RuntimeError("expected interaction to remain primitive-cell local")
                    if abs(complex(coupling)) <= 1e-14:
                        continue
                    I = self._site(r1, r2, int(i))
                    J = self._site(r1, r2, int(j))
                    pairs.add(tuple(sorted((I, J))))
        return tuple(sorted(pairs))

    def _state_indices(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.uint64)
        if self._dense_lookup is not None:
            idx = self._dense_lookup[states.astype(np.int64)]
        else:
            idx = np.searchsorted(self.basis, states).astype(self._index_dtype, copy=False)
            if np.any(idx >= self.dimension) or np.any(self.basis[idx] != states):
                raise RuntimeError("state lookup failed")
        if np.any(idx < 0):
            raise RuntimeError("state lookup failed")
        return idx

    def _pair_transition(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray, np.uint64]:
        i, j = sorted((int(i), int(j)))
        key = (i, j)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        bi = (self.basis >> np.uint64(i)) & np.uint64(1)
        bj = (self.basis >> np.uint64(j)) & np.uint64(1)
        qualifying = np.asarray(bi ^ bj, dtype=bool)
        src = np.flatnonzero(qualifying).astype(self._index_dtype, copy=False)
        parity = _parity_u64(self.basis[src] & _between_mask(i, j))
        plus = src[parity == 0]
        minus = src[parity == 1]
        flipmask = np.uint64((1 << i) | (1 << j))
        cached = (plus, minus, flipmask)
        self._pair_cache[key] = cached
        return cached

    def precompute_hopping_transitions(self) -> None:
        """Populate transition caches for every one-body hopping pair."""
        for i, j, _ in self._hopping_pairs:
            self._pair_transition(i, j)

    def _manybody_diagonal(self, site_diag: np.ndarray) -> np.ndarray:
        site_diag = np.asarray(site_diag)
        dtype = np.result_type(site_diag.dtype, np.float64)
        out = np.zeros(self.dimension, dtype=dtype)
        for i in np.flatnonzero(np.abs(site_diag) > 1e-14):
            occ = (self.basis >> np.uint64(int(i))) & np.uint64(1)
            out += site_diag[int(i)] * occ.astype(dtype)
        return out

    def _apply_unordered_pair(
        self,
        y: np.ndarray,
        x: np.ndarray,
        i: int,
        j: int,
        aij: complex,
        aji: complex,
    ) -> None:
        """Apply off-diagonal one-body terms for one unordered site pair."""
        plus, minus, flipmask = self._pair_transition(i, j)
        for src, sgn in ((plus, 1.0), (minus, -1.0)):
            if src.size == 0:
                continue
            states = self.basis[src]
            dst = self._state_indices(states ^ flipmask)
            # If j is occupied the allowed bilinear is c_i^dag c_j (aij),
            # otherwise c_j^dag c_i (aji).
            j_occ = ((states >> np.uint64(j)) & np.uint64(1)).astype(bool)
            amp = np.where(j_occ, complex(aij), complex(aji))
            y[dst] += sgn * amp * x[src]

    def apply_onebody_matrix(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Apply second-quantized ``sum_ij matrix[i,j] c_i^dag c_j``."""
        matrix = np.asarray(matrix, dtype=complex)
        if matrix.shape != (self.n_sites, self.n_sites):
            raise ValueError("one-body matrix shape mismatch")
        x = np.asarray(vector)
        if x.shape != (self.dimension,):
            raise ValueError("many-body vector shape mismatch")
        y = np.zeros(self.dimension, dtype=np.result_type(x.dtype, matrix.dtype))
        diag = np.diag(matrix)
        if np.any(np.abs(diag) > 1e-14):
            y += self._manybody_diagonal(diag) * x
        for i in range(self.n_sites):
            for j in range(i + 1, self.n_sites):
                aij = matrix[i, j]
                aji = matrix[j, i]
                if abs(aij) <= 1e-14 and abs(aji) <= 1e-14:
                    continue
                self._apply_unordered_pair(y, x, i, j, aij, aji)
        return y

    def apply_hamiltonian(self, V: float, vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector)
        if x.shape != (self.dimension,):
            raise ValueError("many-body vector shape mismatch")
        dtype = np.result_type(x.dtype, np.float64)
        y = np.asarray((self._h_onsite_mb + float(V) * self.interaction_count) * x, dtype=dtype)
        for i, j, amp in self._hopping_pairs:
            plus, minus, flipmask = self._pair_transition(i, j)
            for src, sgn in ((plus, 1.0), (minus, -1.0)):
                if src.size == 0:
                    continue
                dst = self._state_indices(self.basis[src] ^ flipmask)
                y[dst] += (sgn * amp) * x[src]
        return y

    def hamiltonian_operator(self, V: float, *, complex_dtype: bool = False) -> LinearOperator:
        dtype = np.complex128 if complex_dtype else np.float64
        return LinearOperator(
            (self.dimension, self.dimension),
            matvec=lambda x: self.apply_hamiltonian(float(V), x),
            dtype=dtype,
        )

    def hamiltonian(self, V: float) -> LinearOperator:
        """Compatibility alias returning the matrix-free Hamiltonian operator."""
        return self.hamiltonian_operator(V)

    def solve(
        self,
        V: float,
        *,
        n_eigs: int = 8,
        tol: float = 1e-10,
        maxiter: int = 5000,
        v0: np.ndarray | None = None,
        degeneracy_tol: float = 1e-8,
    ) -> EDClusterSpectrum:
        n_eigs = max(2, min(int(n_eigs), self.dimension - 1))
        energies, vecs = eigsh(
            self.hamiltonian_operator(V),
            k=n_eigs,
            which="SA",
            v0=v0,
            tol=float(tol),
            maxiter=int(maxiter),
        )
        order = np.argsort(energies)
        energies = np.asarray(energies[order], dtype=float)
        vecs = np.asarray(vecs[:, order], dtype=complex)
        mask = np.abs(energies - energies[0]) <= float(degeneracy_tol)
        ng = int(np.count_nonzero(mask))
        gap = float(energies[ng] - energies[0]) if ng < len(energies) else np.nan
        gs = vecs[:, :ng]
        dint = float(
            np.mean([
                np.vdot(gs[:, i], self.interaction_count.astype(float) * gs[:, i]).real
                for i in range(ng)
            ])
        )
        return EDClusterSpectrum(
            V=float(V),
            energies=energies,
            eigenvectors=vecs,
            ground_multiplicity=ng,
            gap_above_manifold=gap,
            interaction_expectation=dint,
        )

    def is_commensurate_q(self, q: np.ndarray, tol: float = 1e-10) -> bool:
        q = np.asarray(q, dtype=float).reshape(2)
        vals = np.asarray([self.L1 * q[0], self.L2 * q[1]])
        return bool(np.max(np.abs(vals - np.rint(vals))) <= float(tol))

    def operator_matrix(self, channel: str, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(2)
        if not self.is_commensurate_q(q):
            raise ValueError(f"q={tuple(q)} is not commensurate with {self.L1}x{self.L2} PBC")
        key = (str(channel), float(q[0]), float(q[1]))
        if key in self._operator_matrix_cache:
            return self._operator_matrix_cache[key]
        k6 = primitive_pseudospin_vertex(channel)
        one = np.zeros((self.n_sites, self.n_sites), dtype=complex)
        norm = np.sqrt(float(self.n_cells))
        for ic, (r1, r2) in enumerate(self.cell_coordinates):
            phase = np.exp(-2j * np.pi * (q[0] * r1 + q[1] * r2)) / norm
            sl = slice(NSUB * ic, NSUB * (ic + 1))
            one[sl, sl] = phase * k6
        self._operator_matrix_cache[key] = one
        return one

    def apply_operator(self, channel: str, q: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return self.apply_onebody_matrix(self.operator_matrix(channel, q), vector)

    def structure_factor(
        self,
        spectrum: EDClusterSpectrum,
        q: np.ndarray,
        channels: tuple[str, ...] = ED_CLUSTER_CHANNELS,
    ) -> EDClusterStructureFactor:
        q = np.asarray(q, dtype=float).reshape(2)
        channels = tuple(str(x) for x in channels)
        ng = int(spectrum.ground_multiplicity)
        gs = np.asarray(spectrum.eigenvectors[:, :ng], dtype=complex)
        n = len(channels)
        means = np.zeros(n, dtype=complex)
        S = np.zeros((n, n), dtype=complex)
        for g in range(ng):
            psi = gs[:, g]
            phi = [self.apply_operator(ch, q, psi) for ch in channels]
            for a in range(n):
                means[a] += np.vdot(psi, phi[a]) / float(ng)
                for b in range(n):
                    S[a, b] += np.vdot(phi[a], phi[b]) / float(ng)
        S -= np.outer(means.conj(), means)
        S = 0.5 * (S + S.conj().T)
        vals, vecs = np.linalg.eigh(S)
        order = np.argsort(vals.real)[::-1]
        return EDClusterStructureFactor(
            q=q,
            channels=channels,
            matrix=S,
            eigenvalues=np.asarray(vals[order].real, dtype=float),
            eigenvectors=np.asarray(vecs[:, order], dtype=complex),
            mean=means,
        )

    def memory_estimate(self) -> dict[str, float]:
        """Rough resident-memory estimate in GiB before Lanczos work vectors."""
        index_bytes = np.dtype(self._index_dtype).itemsize
        basis_bytes = self.basis.nbytes
        lookup_bytes = 0 if self._dense_lookup is None else self._dense_lookup.nbytes
        count_bytes = self.interaction_count.nbytes
        if 0 < self.n_particles < self.n_sites:
            qualifying_per_pair = 2 * comb(self.n_sites - 2, self.n_particles - 1)
        else:
            qualifying_per_pair = 0
        transition_bytes = len(self._hopping_pairs) * qualifying_per_pair * index_bytes
        gib = float(1024 ** 3)
        return {
            "basis_GiB": basis_bytes / gib,
            "lookup_GiB": lookup_bytes / gib,
            "interaction_GiB": count_bytes / gib,
            "hopping_transition_cache_GiB": transition_bytes / gib,
            "core_estimate_GiB": (basis_bytes + lookup_bytes + count_bytes + transition_bytes) / gib,
        }
