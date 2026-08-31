"""18-site commensurate Ruby supercell for the Q=(1/3,1/3) instability.

The primitive-cell translation vectors are denoted a1,a2.  We choose the
index-three supercell

    T1 =  a1 - a2
    T2 =  a1 + 2 a2

so that Q=(1/3,1/3) obeys Q.T1=0 and Q.T2=1.  The three primitive cells inside
one supercell are represented by R_s=(s,0), s=0,1,2.  Consequently the
primitive finite-Q charge pattern is folded to q=0 in the supercell Brillouin
zone while Bloch momentum is retained.

The 18-site basis is ordered as

    I = 6*s + a,   s=0,1,2,   a=0,...,5.

All phase conventions remain the same as the primitive model: a supercell
offset S=(S1,S2) contributes exp(2*pi*i*k.S), where k is reduced with respect
to T1,T2.

Important interaction convention: the supercell is a literal translation of
the primitive model.  Hopping uses the full Ruby graph, but the density
interaction V is repeated ONLY on the six intra-triangle bonds of each primitive
cell.  No V is added on t1/t2 bonds or between primitive cells/sectors.
"""

from __future__ import annotations

import numpy as np

from .model import (
    NSUB,
    RubyParameters,
    build_h0,
    ruby_hoppings,
    ruby_interaction_bonds,
)

NSECTOR = 3
NSUP = NSECTOR * NSUB

# Columns are the primitive-coordinate representations of T1 and T2.
SUPERCELL_MATRIX = np.array([[1, 1], [-1, 2]], dtype=int)
SUPERCELL_REPRESENTATIVES = np.array([[0, 0], [1, 0], [2, 0]], dtype=int)
Q_PERIOD3 = np.array([1.0 / 3.0, 1.0 / 3.0])


def primitive_cell_to_supercell(R: np.ndarray | tuple[int, int]) -> tuple[int, np.ndarray]:
    """Decompose a primitive-cell coordinate into (sector, supercell shift).

    Returns ``s,S`` such that

        R = R_s + S1*T1 + S2*T2,

    with ``R_s=(s,0)`` and ``s in {0,1,2}``.
    """
    x, y = (int(R[0]), int(R[1]))
    s = int((x + y) % 3)
    S2 = int((x + y - s) // 3)
    S1 = int(x - s - S2)
    return s, np.array([S1, S2], dtype=int)


def supercell_site_index(sector: int, sublattice: int) -> int:
    return NSUB * int(sector) + int(sublattice)


def supercell_hoppings(params: RubyParameters):
    """Directed 18-site hoppings ``(I,J,S,amp)`` in supercell coordinates."""
    out = []
    for s, Rs in enumerate(SUPERCELL_REPRESENTATIVES):
        for a, b, delta, amp in ruby_hoppings(params):
            sp, S = primitive_cell_to_supercell(Rs + delta)
            I = supercell_site_index(s, a)
            J = supercell_site_index(sp, b)
            out.append((I, J, S, amp))
    return out


def supercell_interaction_bonds(params: RubyParameters):
    """Directed 18-site density-interaction bonds ``(I,J,S,V)``.

    Start from exactly the six interacting bonds of one primitive cell and
    translate that motif into each of the three primitive cells contained in the
    supercell.  Since all interacting primitive bonds are intracell, every such
    bond remains inside one sector and has supercell offset S=(0,0).

    There are therefore 18 undirected = 36 directed V-bonds in one 18-site
    supercell, and every site has interaction coordination z_V=2.
    """
    out = []
    for s, Rs in enumerate(SUPERCELL_REPRESENTATIVES):
        for a, b, delta, coupling in ruby_interaction_bonds(params):
            delta = np.asarray(delta, dtype=int)
            sp, S = primitive_cell_to_supercell(Rs + delta)
            I = supercell_site_index(s, a)
            J = supercell_site_index(sp, b)
            out.append((I, J, S, complex(coupling)))
            out.append((J, I, -S, complex(coupling)))
    return out


def period3_complex_mode() -> np.ndarray:
    """Canonical complex 18-site Q=(1/3,1/3) charge mode.

    The primitive six-sublattice mode is chosen as

        (1, w, w^2, -1, -w, -w^2),  w=exp(2*pi*i/3),

    and is multiplied by exp(2*pi*i*Q.R_s)=w^s in sector s.  Its real part is
    the period-three pattern observed in the primitive-cell screening mode.
    """
    w = np.exp(2j * np.pi / 3.0)
    v6 = np.array([1.0, w, w**2, -1.0, -w, -(w**2)], dtype=complex)
    return np.concatenate([v6 * (w**s) for s in range(NSECTOR)])


def period3_real_pattern() -> np.ndarray:
    """Real seed pattern, normalized so its largest absolute entry is one."""
    return period3_complex_mode().real.copy()


def charge_order_parameter(density: np.ndarray) -> complex:
    """Return the complex period-three charge amplitude of an 18-site density.

    With ``z=period3_complex_mode()`` we define

        Phi = 2 <z|delta n> / <z|z>,

    where ``delta n`` has its spatial average removed.  Therefore if

        delta n = A Re[z exp(i theta)],

    the returned value is ``Phi=A exp(i theta)`` up to roundoff.  ``abs(Phi)``
    is translation/gauge independent within the threefold family.
    """
    density = np.asarray(density, dtype=float).reshape(NSUP)
    delta = density - float(np.mean(density))
    z = period3_complex_mode()
    return complex(2.0 * np.vdot(z, delta) / np.vdot(z, z))


def build_supercell_h0(
    kpts: np.ndarray,
    params: RubyParameters,
    source_strength: float = 0.0,
) -> np.ndarray:
    """Return the 18x18 supercell Bloch Hamiltonian.

    ``source_strength`` adds the temporary pinning field

        H_source = -h sum_I pattern_I n_I,

    with ``pattern=period3_real_pattern()``.  Positive h therefore lowers the
    onsite energy on the positive lobes of the chosen charge pattern.
    """
    kpts = np.asarray(kpts, dtype=float)
    flat = kpts.reshape(-1, 2)
    h0 = np.zeros((flat.shape[0], NSUP, NSUP), dtype=complex)
    hops = supercell_hoppings(params)
    for ik, k in enumerate(flat):
        for I, J, S, amp in hops:
            h0[ik, I, J] += amp * np.exp(2j * np.pi * np.dot(k, S))

    if source_strength != 0.0:
        diag = np.diag_indices(NSUP)
        h0[:, diag[0], diag[1]] -= float(source_strength) * period3_real_pattern()[None, :]

    h0 = 0.5 * (h0 + np.swapaxes(h0.conj(), -1, -2))
    return h0.reshape(kpts.shape[:-1] + (NSUP, NSUP))


def build_supercell_interaction(qpts: np.ndarray, params: RubyParameters) -> np.ndarray:
    """Return the 18x18 intra-triangle density interaction in the supercell BZ.

    The primitive interaction motif is copied into each of the three primitive
    cells of the supercell:

        sector s: (6s+0,6s+1), (6s+0,6s+2), (6s+1,6s+2),
                  (6s+3,6s+4), (6s+3,6s+5), (6s+4,6s+5).

    No other pair carries V.  In particular, all t1/t2 hopping bonds have zero
    density coupling.  Because every V-bond is intracell, the resulting matrix
    is independent of supercell momentum q.
    """
    qpts = np.asarray(qpts, dtype=float)
    flat = qpts.reshape(-1, 2)
    vq = np.zeros((flat.shape[0], NSUP, NSUP), dtype=complex)

    interactions = supercell_interaction_bonds(params)
    for iq, q in enumerate(flat):
        for I, J, S, coupling in interactions:
            vq[iq, I, J] += coupling * np.exp(2j * np.pi * np.dot(q, S))

    vq = 0.5 * (vq + np.swapaxes(vq.conj(), -1, -2))
    return vq.reshape(qpts.shape[:-1] + (NSUP, NSUP))


def supercell_to_primitive_momenta(k_sc: np.ndarray) -> np.ndarray:
    """Return the three primitive momenta folded onto one supercell momentum.

    If ``A`` has columns T1,T2, the base primitive reduced momentum obeys

        k_p = A^{-T} k_sc.

    The other two members of the folded star are ``k_p + Q`` and ``k_p+2Q``.
    Output shape is ``(...,3,2)`` and values are wrapped to [0,1).
    """
    k_sc = np.asarray(k_sc, dtype=float)
    inv_transpose = np.linalg.inv(SUPERCELL_MATRIX).T
    base = np.einsum("ij,...j->...i", inv_transpose, k_sc, optimize=True)
    folded = np.stack([(base + m * Q_PERIOD3) % 1.0 for m in range(3)], axis=-2)
    return folded


def folded_primitive_eigenvalues(k_sc: np.ndarray, params: RubyParameters) -> np.ndarray:
    """Reference primitive-cell bands corresponding to the 18-site folding."""
    kp = supercell_to_primitive_momenta(k_sc)
    h = build_h0(kp, params)
    evals = np.linalg.eigvalsh(h)
    return np.sort(evals.reshape(kp.shape[:-2] + (NSUP,)), axis=-1)
