"""Exact diagonalization on a six-triangle Ruby hexagon (18-site OBC cluster).

The cluster is built directly from the primitive Ruby hopping graph by keeping
six elementary triangles whose centers form one honeycomb hexagon,

    A(0,0) - B(0,0) - A(1,0) - B(1,1)
      |                               |
    B(0,1) - A(1,1) -----------------

more precisely in cyclic order

    A00, B00, A10, B11, A11, B01.

Each triangle contributes three distinct Ruby sites, so the cluster has 18
spinless-fermion orbitals.  Open boundary conditions are used: a primitive
hopping is retained iff both of its endpoint sites belong to the selected
triangles.  The density interaction remains exactly the microscopic one, i.e.
V only on the three bonds within each selected triangle.

At primitive filling n=2 the corresponding fixed-particle problem has N=6 and
C(18,6)=18564 basis states, identical in Hilbert-space size to the existing
period-three ED18 torus.

The six local current operators use the *physical* pseudospin-z convention from
``pseudospin.py``.  Thus A/B geometric handedness is already accounted for and
a positive expectation value means the same physical circulation sense on all
six triangles.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .ed18 import fixed_particle_basis, _manybody_onebody
from .model import RubyParameters, ruby_hoppings
from .pseudospin import primitive_triangle_pseudospin_vertices

NHEX = 18
NTRI = 6

# Cyclic order around the central hexagon.  kind is 'A' or 'B', cell is in
# primitive reduced coordinates.
HEX_TRIANGLES = (
    ("A", (0, 0)),
    ("B", (0, 0)),
    ("A", (1, 0)),
    ("B", (1, 1)),
    ("A", (1, 1)),
    ("B", (0, 1)),
)


@dataclass
class Hex18Spectrum:
    V: float
    energies: np.ndarray
    eigenvectors: np.ndarray
    ground_multiplicity: int
    gap_above_manifold: float


@dataclass
class Hex18CurrentStructure:
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    means: np.ndarray
    circulant_error: float


def _selected_site_map():
    """Map primitive (cell_x,cell_y,sublattice) labels to 0..17."""
    mapping = {}
    inverse = []
    for m, (kind, cell) in enumerate(HEX_TRIANGLES):
        subs = (0, 1, 2) if kind == "A" else (3, 4, 5)
        for local, a in enumerate(subs):
            idx = 3 * m + local
            key = (int(cell[0]), int(cell[1]), int(a))
            if key in mapping:
                raise RuntimeError(f"duplicate selected Ruby site {key}")
            mapping[key] = idx
            inverse.append(key)
    if len(mapping) != NHEX:
        raise RuntimeError("hexagon site mapping is not 18-to-18")
    return mapping, tuple(inverse)


def build_hex18_one_body(params: RubyParameters) -> np.ndarray:
    """Return the exact 18x18 OBC hopping matrix inherited from the Ruby model."""
    mapping, inverse = _selected_site_map()
    h = np.zeros((NHEX, NHEX), dtype=complex)
    hops = ruby_hoppings(RubyParameters(ti=params.ti, t1=params.t1, t2=params.t2, V=0.0))
    by_source = {}
    for a, b, R, amp in hops:
        by_source.setdefault(int(a), []).append((int(b), np.asarray(R, dtype=int), complex(amp)))
    for i, (x, y, a) in enumerate(inverse):
        for b, R, amp in by_source.get(int(a), ()):
            key = (x + int(R[0]), y + int(R[1]), int(b))
            j = mapping.get(key)
            if j is not None:
                h[i, j] += amp
    herr = float(np.max(np.abs(h - h.conj().T), initial=0.0))
    if herr > 1e-12:
        raise RuntimeError(f"hex18 one-body Hamiltonian is not Hermitian: {herr:.3e}")
    return 0.5 * (h + h.conj().T)


def hex18_interaction_pairs() -> tuple[tuple[int, int], ...]:
    """The 18 intra-triangle V bonds of the six selected triangles."""
    pairs = []
    for m in range(NTRI):
        s = 3 * m
        pairs.extend(((s, s + 1), (s, s + 2), (s + 1, s + 2)))
    return tuple(pairs)


def hex18_local_current_matrices() -> np.ndarray:
    """Six physical-chirality one-body current matrices, shape (6,18,18)."""
    pv = primitive_triangle_pseudospin_vertices()
    out = np.zeros((NTRI, NHEX, NHEX), dtype=complex)
    for m, (kind, _) in enumerate(HEX_TRIANGLES):
        src = pv["Az"] if kind == "A" else pv["Bz"]
        subs = (0, 1, 2) if kind == "A" else (3, 4, 5)
        block = np.asarray(src[np.ix_(subs, subs)], dtype=complex)
        sl = slice(3 * m, 3 * m + 3)
        out[m, sl, sl] = block
    return out


def canonical_ring_mode(ell: int) -> np.ndarray:
    """Normalized Fourier mode exp(i*2*pi*ell*m/6) on the triangle ring."""
    ell = int(ell) % NTRI
    m = np.arange(NTRI, dtype=float)
    return np.exp(2j * np.pi * ell * m / float(NTRI)) / np.sqrt(float(NTRI))


class Hex18Solver:
    """Fixed-N sparse ED solver for the six-triangle OBC hexagon."""

    def __init__(
        self,
        params: RubyParameters = RubyParameters(),
        *,
        primitive_filling: float = 2.0,
        n_particles: int | None = None,
    ):
        self.params = params
        if n_particles is None:
            # Eighteen sites correspond to three primitive six-site cells.
            nf_float = 3.0 * float(primitive_filling)
            nf = int(round(nf_float))
            if abs(nf_float - nf) > 1e-12:
                raise ValueError("3*primitive_filling must be an integer")
            n_particles = nf
        self.n_particles = int(n_particles)
        self.primitive_filling = self.n_particles / 3.0
        self.basis = fixed_particle_basis(NHEX, self.n_particles)
        self.index = {int(s): i for i, s in enumerate(self.basis)}
        self.dimension = len(self.basis)
        if self.dimension != comb(NHEX, self.n_particles):
            raise RuntimeError("hex18 fixed-N basis dimension mismatch")

        self.h0 = build_hex18_one_body(params)
        ht = _manybody_onebody(self.h0, self.basis, self.index)
        self.H_t = sparse.csr_matrix(ht)
        if sparse.linalg.norm(self.H_t - self.H_t.getH()) > 1e-10:
            raise RuntimeError("hex18 hopping Hamiltonian is not Hermitian")

        self.interaction_pairs = hex18_interaction_pairs()
        self.interaction_count = np.zeros(self.dimension, dtype=float)
        for ib, raw in enumerate(self.basis):
            state = int(raw)
            self.interaction_count[ib] = sum(
                ((state >> i) & 1) * ((state >> j) & 1)
                for i, j in self.interaction_pairs
            )

        self.current_onebody = hex18_local_current_matrices()
        self.current_ops = tuple(
            _manybody_onebody(self.current_onebody[m], self.basis, self.index)
            for m in range(NTRI)
        )

    def hamiltonian(self, V: float, *, source_mode=None, source_h: float = 0.0):
        H = self.H_t + sparse.diags(float(V) * self.interaction_count, format="csr")
        if source_mode is not None and abs(float(source_h)) > 0.0:
            mode = np.asarray(source_mode, dtype=complex).reshape(NTRI)
            # Hermitian source requires a real combination of Hermitian local J_m.
            if np.max(np.abs(mode.imag), initial=0.0) > 1e-12:
                raise ValueError("source_mode must be real for a Hermitian static source")
            J = sum(float(mode[m].real) * self.current_ops[m] for m in range(NTRI))
            H = H - float(source_h) * J
        return sparse.csr_matrix(H, dtype=complex)

    def solve(
        self,
        V: float,
        *,
        n_eigs: int = 8,
        tol: float = 1e-10,
        maxiter: int = 5000,
        v0: np.ndarray | None = None,
        degeneracy_tol: float = 1e-8,
        source_mode=None,
        source_h: float = 0.0,
    ) -> Hex18Spectrum:
        n_eigs = max(2, min(int(n_eigs), self.dimension - 1))
        H = self.hamiltonian(V, source_mode=source_mode, source_h=source_h)
        e, vecs = eigsh(H, k=n_eigs, which="SA", v0=v0, tol=float(tol), maxiter=int(maxiter))
        order = np.argsort(e)
        e = np.asarray(e[order], dtype=float)
        vecs = np.asarray(vecs[:, order], dtype=complex)
        mask = np.abs(e - e[0]) <= float(degeneracy_tol)
        ng = int(np.count_nonzero(mask))
        gap = float(e[ng] - e[0]) if ng < len(e) else np.nan
        return Hex18Spectrum(
            V=float(V), energies=e, eigenvectors=vecs,
            ground_multiplicity=ng, gap_above_manifold=gap,
        )

    def current_structure(self, spectrum: Hex18Spectrum) -> Hex18CurrentStructure:
        """Ground-manifold-averaged connected <J_m J_n> matrix."""
        ng = int(spectrum.ground_multiplicity)
        gs = np.asarray(spectrum.eigenvectors[:, :ng], dtype=complex)
        means = np.zeros(NTRI, dtype=complex)
        C = np.zeros((NTRI, NTRI), dtype=complex)
        for g in range(ng):
            psi = gs[:, g]
            phi = [op @ psi for op in self.current_ops]
            for m in range(NTRI):
                means[m] += np.vdot(psi, phi[m]) / ng
            for m in range(NTRI):
                for n in range(NTRI):
                    C[m, n] += np.vdot(phi[m], phi[n]) / ng
        C -= np.outer(means.conj(), means)
        C = 0.5 * (C + C.conj().T)
        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals.real)[::-1]
        vals = np.asarray(vals[order].real, dtype=float)
        vecs = np.asarray(vecs[:, order], dtype=complex)

        # Distance-only average on a six-site ring, used as a symmetry diagnostic.
        circ = np.zeros_like(C)
        for d in range(NTRI):
            entries = [C[m, (m + d) % NTRI] for m in range(NTRI)]
            avg = sum(entries) / float(NTRI)
            for m in range(NTRI):
                circ[m, (m + d) % NTRI] = avg
        denom = max(float(np.linalg.norm(C.ravel())), 1e-300)
        cerr = float(np.linalg.norm((C - circ).ravel()) / denom)
        return Hex18CurrentStructure(C, vals, vecs, means, cerr)

    def current_expectations(self, state: np.ndarray) -> np.ndarray:
        psi = np.asarray(state, dtype=complex).reshape(self.dimension)
        return np.asarray([np.vdot(psi, op @ psi) for op in self.current_ops], dtype=complex)

    def source_response(
        self,
        V: float,
        mode: np.ndarray,
        *,
        h: float = 1e-3,
        tol: float = 1e-10,
        maxiter: int = 5000,
        v0: np.ndarray | None = None,
    ):
        """Central-difference T=0 susceptibility of one real ring-current mode."""
        mode = np.asarray(mode, dtype=float).reshape(NTRI)
        norm = float(np.linalg.norm(mode))
        if norm <= 0.0:
            raise ValueError("source mode is zero")
        mode = mode / norm
        sp = self.solve(V, n_eigs=3, tol=tol, maxiter=maxiter, v0=v0,
                        source_mode=mode, source_h=abs(float(h)))
        sm = self.solve(V, n_eigs=3, tol=tol, maxiter=maxiter, v0=v0,
                        source_mode=mode, source_h=-abs(float(h)))
        jp = self.current_expectations(sp.eigenvectors[:, 0]).real
        jm = self.current_expectations(sm.eigenvectors[:, 0]).real
        Mp = float(np.dot(mode, jp))
        Mm = float(np.dot(mode, jm))
        chi = (Mp - Mm) / (2.0 * abs(float(h)))
        return {
            "chi": float(chi),
            "M_plus": Mp,
            "M_minus": Mm,
            "J_plus": jp,
            "J_minus": jm,
            "E_plus": float(sp.energies[0]),
            "E_minus": float(sm.energies[0]),
        }


__all__ = [
    "NHEX", "NTRI", "HEX_TRIANGLES", "Hex18Solver", "Hex18Spectrum",
    "Hex18CurrentStructure", "build_hex18_one_body", "hex18_interaction_pairs",
    "hex18_local_current_matrices", "canonical_ring_mode",
]
