"""Exact diagonalization for the 18-site, three-cell Ruby torus.

The finite cluster is the same index-three supercell used by the cGW code,
with one supercell and periodic boundary conditions.  At primitive filling
n=2 this means N=18 spinless-fermion sites and Nf=6 particles, hence
C(18,6)=18564 basis states.

Important finite-size limitation
--------------------------------
The quotient of primitive translations has order three.  Consequently the
only primitive-cell momenta represented by this torus are Gamma and +/-Q with
Q=(1/3,1/3).  A period-two M point is NOT commensurate with this 18-site
cluster.  Results from this module therefore cannot exclude an M-ordered
thermodynamic ground state.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .model import RubyParameters, NSUB
from .pseudospin import primitive_pseudospin_vertex
from .supercell import (
    NSECTOR,
    NSUP,
    SUPERCELL_REPRESENTATIVES,
    build_supercell_h0,
    supercell_interaction_bonds,
)

ED18_CHANNELS = (
    "x_even",
    "x_odd",
    "y_even",
    "y_odd",
    "z_same",
    "z_opposite",
)

Q_GAMMA = np.array([0.0, 0.0], dtype=float)
Q_PERIOD3 = np.array([1.0 / 3.0, 1.0 / 3.0], dtype=float)


@dataclass
class ED18Spectrum:
    V: float
    energies: np.ndarray
    eigenvectors: np.ndarray
    ground_multiplicity: int
    gap_above_manifold: float
    interaction_expectation: float
    translation_eigenvalues: np.ndarray


@dataclass
class ED18StructureFactor:
    q: np.ndarray
    channels: tuple[str, ...]
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    mean: np.ndarray


def fixed_particle_basis(n_sites: int, n_particles: int) -> np.ndarray:
    """Sorted occupation-bit basis at fixed particle number."""
    n_sites = int(n_sites)
    n_particles = int(n_particles)
    if not 0 <= n_particles <= n_sites:
        raise ValueError("n_particles must satisfy 0 <= Nf <= n_sites")
    states = [sum(1 << i for i in occ) for occ in combinations(range(n_sites), n_particles)]
    out = np.asarray(states, dtype=np.int64)
    if len(out) != comb(n_sites, n_particles):
        raise RuntimeError("fixed-particle basis dimension mismatch")
    return out


def _apply_bilinear(state: int, i: int, j: int) -> tuple[int, int] | None:
    """Apply c_i^dagger c_j to one occupation bitstring."""
    if ((state >> j) & 1) == 0 or ((state >> i) & 1) == 1:
        return None
    sign = -1 if (state & ((1 << j) - 1)).bit_count() % 2 else 1
    tmp = state ^ (1 << j)
    sign *= -1 if (tmp & ((1 << i) - 1)).bit_count() % 2 else 1
    return tmp | (1 << i), sign


def _manybody_onebody(
    matrix: np.ndarray,
    basis: np.ndarray,
    index: dict[int, int],
) -> sparse.csr_matrix:
    """Second-quantize sum_ij matrix[i,j] c_i^dagger c_j."""
    matrix = np.asarray(matrix)
    if matrix.shape != (NSUP, NSUP):
        raise ValueError(f"expected {(NSUP, NSUP)} one-body matrix")

    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    diag_idx = np.flatnonzero(np.abs(np.diag(matrix)) > 1e-14)
    off = [(int(i), int(j)) for i, j in np.argwhere(np.abs(matrix) > 1e-14) if i != j]

    for col, raw in enumerate(basis):
        state = int(raw)
        dval = 0.0j
        for i in diag_idx:
            if (state >> int(i)) & 1:
                dval += matrix[i, i]
        if abs(dval) > 1e-14:
            rows.append(col)
            cols.append(col)
            data.append(dval)

        for i, j in off:
            applied = _apply_bilinear(state, i, j)
            if applied is None:
                continue
            new_state, sign = applied
            rows.append(index[new_state])
            cols.append(col)
            data.append(complex(matrix[i, j]) * sign)

    return sparse.coo_matrix(
        (np.asarray(data, dtype=complex), (rows, cols)),
        shape=(len(basis), len(basis)),
    ).tocsr()


def _permutation_sign(mapped: list[int]) -> int:
    inv = 0
    for i in range(len(mapped)):
        for j in range(i + 1, len(mapped)):
            if mapped[i] > mapped[j]:
                inv += 1
    return -1 if inv % 2 else 1


class ED18Solver:
    """Reusable fixed-N exact-diagonalization engine for the 18-site torus."""

    def __init__(
        self,
        params: RubyParameters = RubyParameters(),
        *,
        primitive_filling: float = 2.0,
        n_particles: int | None = None,
    ):
        self.params = params
        if n_particles is None:
            nf_float = float(primitive_filling) * NSECTOR
            nf = int(round(nf_float))
            if abs(nf_float - nf) > 1e-12:
                raise ValueError("primitive_filling*NSECTOR must be an integer for fixed-N ED")
            n_particles = nf
        self.n_particles = int(n_particles)
        self.primitive_filling = self.n_particles / float(NSECTOR)
        self.basis = fixed_particle_basis(NSUP, self.n_particles)
        self.index = {int(s): i for i, s in enumerate(self.basis)}
        self.dimension = len(self.basis)

        # One-supercell PBC is exactly the supercell Bloch Hamiltonian at k_sc=0.
        p0 = RubyParameters(ti=params.ti, t1=params.t1, t2=params.t2, V=0.0)
        h0 = np.asarray(build_supercell_h0(np.array([[0.0, 0.0]]), p0)[0])
        if np.max(np.abs(h0.imag)) > 1e-12:
            raise RuntimeError("k_sc=0 ED hopping Hamiltonian unexpectedly complex")
        self.h0 = h0.real
        ht_complex = _manybody_onebody(self.h0, self.basis, self.index)
        self.H_t = sparse.csr_matrix(ht_complex.real)
        if sparse.linalg.norm(self.H_t - self.H_t.T) > 1e-10:
            raise RuntimeError("ED hopping Hamiltonian is not symmetric")

        # H_V = V * D, where D counts occupied intra-triangle interaction bonds.
        p1 = RubyParameters(ti=params.ti, t1=params.t1, t2=params.t2, V=1.0)
        pairs: set[tuple[int, int]] = set()
        for I, J, S, coupling in supercell_interaction_bonds(p1):
            if np.any(np.asarray(S, dtype=int) != 0):
                raise RuntimeError("18-site ED expects all V bonds to be intra-supercell")
            if abs(complex(coupling)) > 1e-14:
                pairs.add(tuple(sorted((int(I), int(J)))))
        self.interaction_pairs = tuple(sorted(pairs))
        self.interaction_count = np.zeros(self.dimension, dtype=float)
        for ib, raw in enumerate(self.basis):
            state = int(raw)
            self.interaction_count[ib] = sum(
                ((state >> i) & 1) * ((state >> j) & 1)
                for i, j in self.interaction_pairs
            )

        self.translation_a1 = self._build_translation_a1()
        if sparse.linalg.norm(self.translation_a1 @ self.H_t - self.H_t @ self.translation_a1) > 1e-10:
            raise RuntimeError("primitive translation does not commute with H_t")
        permuted_diag = self.translation_a1 @ sparse.diags(self.interaction_count) - sparse.diags(self.interaction_count) @ self.translation_a1
        if sparse.linalg.norm(permuted_diag) > 1e-10:
            raise RuntimeError("primitive translation does not commute with H_V")

        self._operator_cache: dict[tuple[float, float, str], sparse.csr_matrix] = {}

    def _build_translation_a1(self) -> sparse.csr_matrix:
        """Many-body primitive translation a1 on the three-cell torus."""
        orbital_perm = np.asarray(
            [NSUB * ((s + 1) % NSECTOR) + a for s in range(NSECTOR) for a in range(NSUB)],
            dtype=int,
        )
        rows = np.empty(self.dimension, dtype=int)
        signs = np.empty(self.dimension, dtype=float)
        for col, raw in enumerate(self.basis):
            state = int(raw)
            occ = [i for i in range(NSUP) if (state >> i) & 1]
            mapped = [int(orbital_perm[i]) for i in occ]
            new_state = sum(1 << i for i in mapped)
            rows[col] = self.index[new_state]
            signs[col] = _permutation_sign(mapped)
        cols = np.arange(self.dimension, dtype=int)
        return sparse.csr_matrix((signs, (rows, cols)), shape=(self.dimension, self.dimension))

    def hamiltonian(self, V: float) -> sparse.csr_matrix:
        return self.H_t + sparse.diags(float(V) * self.interaction_count, format="csr")

    def solve(
        self,
        V: float,
        *,
        n_eigs: int = 10,
        tol: float = 1e-10,
        maxiter: int = 5000,
        v0: np.ndarray | None = None,
        degeneracy_tol: float = 1e-8,
    ) -> ED18Spectrum:
        n_eigs = max(2, min(int(n_eigs), self.dimension - 1))
        energies, vecs = eigsh(
            self.hamiltonian(V),
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
        gs = vecs[:, mask]

        dint = float(
            np.mean([
                np.vdot(gs[:, i], self.interaction_count * gs[:, i]).real
                for i in range(gs.shape[1])
            ])
        )
        tsub = gs.conj().T @ (self.translation_a1 @ gs)
        teigs = np.linalg.eigvals(tsub)
        teigs = teigs[np.argsort(np.angle(teigs))]

        return ED18Spectrum(
            V=float(V),
            energies=energies,
            eigenvectors=vecs,
            ground_multiplicity=ng,
            gap_above_manifold=gap,
            interaction_expectation=dint,
            translation_eigenvalues=np.asarray(teigs, dtype=complex),
        )

    def operator(self, channel: str, q: np.ndarray) -> sparse.csr_matrix:
        q = np.asarray(q, dtype=float).reshape(2)
        key = (float(q[0]), float(q[1]), str(channel))
        if key in self._operator_cache:
            return self._operator_cache[key]
        k6 = primitive_pseudospin_vertex(channel)
        one = np.zeros((NSUP, NSUP), dtype=complex)
        for s, R in enumerate(SUPERCELL_REPRESENTATIVES):
            phase = np.exp(-2j * np.pi * np.dot(q, R)) / np.sqrt(float(NSECTOR))
            sl = slice(NSUB * s, NSUB * (s + 1))
            one[sl, sl] = phase * k6
        op = _manybody_onebody(one, self.basis, self.index)
        self._operator_cache[key] = op
        return op

    def structure_factor(
        self,
        spectrum: ED18Spectrum,
        q: np.ndarray,
        channels: tuple[str, ...] = ED18_CHANNELS,
    ) -> ED18StructureFactor:
        """Ground-manifold averaged connected equal-time structure matrix.

        Operators are normalized as O_mu(q)=Ncell^{-1/2} sum_R exp(-iqR) O_mu(R),
        and S_{mu,nu}=<O_mu^dag O_nu>-<O_mu^dag><O_nu>.
        """
        q = np.asarray(q, dtype=float).reshape(2)
        ng = spectrum.ground_multiplicity
        gs = spectrum.eigenvectors[:, :ng]
        ops = [self.operator(ch, q) for ch in channels]
        n = len(ops)
        means = np.zeros(n, dtype=complex)
        S = np.zeros((n, n), dtype=complex)

        for g in range(ng):
            psi = gs[:, g]
            phi = [op @ psi for op in ops]
            for a in range(n):
                means[a] += np.vdot(psi, phi[a]) / ng
                for b in range(n):
                    S[a, b] += np.vdot(phi[a], phi[b]) / ng
        S -= np.outer(means.conj(), means)
        S = 0.5 * (S + S.conj().T)
        evals, evecs = np.linalg.eigh(S)
        order = np.argsort(evals.real)[::-1]
        evals = np.asarray(evals[order].real, dtype=float)
        evecs = np.asarray(evecs[:, order], dtype=complex)
        return ED18StructureFactor(
            q=q,
            channels=tuple(channels),
            matrix=S,
            eigenvalues=evals,
            eigenvectors=evecs,
            mean=means,
        )


def phase_fix_vector(vec: np.ndarray) -> np.ndarray:
    """Fix arbitrary global phase by making the largest component positive real."""
    out = np.asarray(vec, dtype=complex).copy()
    if out.size == 0:
        return out
    i = int(np.argmax(np.abs(out)))
    if abs(out[i]) > 1e-14:
        out *= np.exp(-1j * np.angle(out[i]))
    if out[i].real < 0:
        out *= -1
    return out


def translation_phase_reduced(z: complex) -> float:
    """Translation eigenvalue phase divided by 2pi, centered in [-1/2,1/2)."""
    x = float(np.angle(complex(z)) / (2.0 * np.pi))
    if x >= 0.5:
        x -= 1.0
    if x < -0.5:
        x += 1.0
    return x
