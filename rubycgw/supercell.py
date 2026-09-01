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


def charge_source_pattern(channel: str) -> np.ndarray:
    """Return a normalized 18-site diagonal charge-source pattern.

    Supported channels are

    ``co``
        The selected Q=(1/3,1/3) period-three pattern already used by the
        supercell calculation.
    ``intra``
        A joint q=0 C3-breaking field on both triangles.  In each primitive
        cell the six-site pattern is ``(1,-1/2,-1/2,1,-1/2,-1/2)``.  This is
        the default intra-unit-cell charge seed because Delta_A and Delta_B are
        typically induced together in the self-consistent solutions.
    ``intra-a`` / ``intra-b``
        Legacy one-triangle probe fields retained for diagnostics/backward
        compatibility.  They are no longer separate default search branches.
    ``ab``
        A q=0 A-versus-B charge-transfer field,
        ``(1,1,1,-1,-1,-1)``.

    Every pattern has zero spatial mean and maximum absolute component one, so
    the source strength ``h`` has the same onsite-energy scale in all charge
    branches.  The q=0 patterns are copied identically into all three primitive
    sectors of the supercell.
    """
    channel = str(channel).lower()
    if channel == "co":
        pattern = period3_real_pattern()
    elif channel == "intra":
        p6 = np.array([1.0, -0.5, -0.5, 1.0, -0.5, -0.5], dtype=float)
        pattern = np.tile(p6, NSECTOR)
    elif channel == "intra-a":
        p6 = np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0], dtype=float)
        pattern = np.tile(p6, NSECTOR)
    elif channel == "intra-b":
        p6 = np.array([0.0, 0.0, 0.0, 1.0, -0.5, -0.5], dtype=float)
        pattern = np.tile(p6, NSECTOR)
    elif channel == "ab":
        p6 = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=float)
        pattern = np.tile(p6, NSECTOR)
    else:
        raise ValueError(
            f"unknown charge source channel {channel!r}; expected co, intra, intra-a, intra-b, or ab"
        )

    pattern = np.asarray(pattern, dtype=float).reshape(NSUP)
    if abs(float(np.mean(pattern))) > 1e-14:
        raise RuntimeError("charge source pattern must have zero spatial mean")
    maxabs = float(np.max(np.abs(pattern)))
    if maxabs == 0.0:
        raise RuntimeError("charge source pattern must be nonzero")
    return pattern / maxabs


def add_charge_source(h0: np.ndarray, strength: float, channel: str) -> np.ndarray:
    """Add ``H_source=-h sum_I p_I n_I`` to an 18-site Bloch Hamiltonian."""
    out = np.array(h0, dtype=complex, copy=True)
    if out.shape[-2:] != (NSUP, NSUP):
        raise ValueError(f"expected final h0 dimensions {(NSUP, NSUP)}, got {out.shape[-2:]}")
    if float(strength) == 0.0:
        return out
    pattern = charge_source_pattern(channel)
    diag = np.diag_indices(NSUP)
    out[..., diag[0], diag[1]] -= float(strength) * pattern
    return 0.5 * (out + np.swapaxes(out.conj(), -1, -2))


def charge_order_parameter(density: np.ndarray) -> complex:
    """Return the selected complex period-three charge amplitude ``Phi``.

    With ``z=period3_complex_mode()`` we define

        Phi = 2 <z|delta n> / <z|z>,

    where ``delta n`` has its spatial average removed.  Therefore if

        delta n = A Re[z exp(i theta)],

    the returned value is ``Phi=A exp(i theta)`` up to roundoff.  ``abs(Phi)``
    is translation/gauge independent within the threefold family.

    Important: this is a projection onto one particular Q form factor.  A zero
    value does NOT imply that all charge order is absent.  Use
    :func:`charge_order_diagnostics` for generic period-three and q=0
    intra-unit-cell charge diagnostics.
    """
    density = np.asarray(density, dtype=float).reshape(NSUP)
    delta = density - float(np.mean(density))
    z = period3_complex_mode()
    return complex(2.0 * np.vdot(z, delta) / np.vdot(z, z))


def charge_order_diagnostics(density: np.ndarray) -> dict[str, object]:
    """Decompose an 18-site density into generic q=0 and +/-Q charge sectors.

    Write the density as ``n[s,a]`` with sector ``s=0,1,2`` and primitive
    sublattice ``a=0,...,5``.  The sector Fourier components are

        n_q0[a] = (1/3) sum_s n[s,a],
        n_Q[a]  = (1/3) sum_s exp(-2*pi*i*s/3) n[s,a].

    For a real density the -Q component is ``n_Q.conj()``.  The returned
    diagnostics are:

    ``Phi``
        Projection onto the previously selected period-three form factor.
    ``Delta_Q``
        ``sqrt(sum_a |n_Q[a]|^2)``.  This detects any period-three translation
        breaking in the 18-site cell, including Q form factors orthogonal to
        the selected ``Phi`` mode.
    ``Delta_translation_rms``
        RMS density difference between the three primitive-cell sectors after
        removing ``n_q0``.  For this three-sector real-density decomposition it
        equals ``Delta_Q/sqrt(3)`` up to roundoff.
    ``Delta_intra``
        Combined q=0 intra-triangle charge-disproportionation amplitude,

            sqrt((Delta_A^2 + Delta_B^2)/2).

        It is invariant under exchanging A and B and equals the common triangle
        amplitude when Delta_A=Delta_B.  This is the quantity used for phase
        classification.
    ``Delta_A`` / ``Delta_B``
        Triangle-resolved components retained for detailed inspection.  For A,

            sqrt(((n0-n1)^2 + (n1-n2)^2 + (n2-n0)^2)/2).

        Thus a pattern ``(a,b,b)`` gives exactly ``|a-b|``.
    ``Delta_AB``
        Signed q=0 difference between the mean density of triangles A and B.

    Arrays ``n_q0`` and ``n_Q`` are also returned for detailed inspection.
    """
    n = np.asarray(density, dtype=float).reshape(NSECTOR, NSUB)
    n_q0 = np.mean(n, axis=0)

    w = np.exp(2j * np.pi / 3.0)
    phase = np.array([1.0, np.conj(w), np.conj(w**2)], dtype=complex)
    n_Q = np.einsum("s,sa->a", phase, n, optimize=True) / float(NSECTOR)

    delta_Q = float(np.linalg.norm(n_Q))
    translation_rms = float(np.sqrt(np.mean((n - n_q0[None, :]) ** 2)))

    A = n_q0[0:3]
    B = n_q0[3:6]
    delta_A = float(
        np.sqrt(
            0.5
            * (
                (A[0] - A[1]) ** 2
                + (A[1] - A[2]) ** 2
                + (A[2] - A[0]) ** 2
            )
        )
    )
    delta_B = float(
        np.sqrt(
            0.5
            * (
                (B[0] - B[1]) ** 2
                + (B[1] - B[2]) ** 2
                + (B[2] - B[0]) ** 2
            )
        )
    )
    delta_intra = float(np.sqrt(0.5 * (delta_A**2 + delta_B**2)))
    delta_AB = float(np.mean(A) - np.mean(B))

    return {
        "Phi": charge_order_parameter(n.reshape(NSUP)),
        "Delta_Q": delta_Q,
        "Delta_translation_rms": translation_rms,
        "Delta_intra": delta_intra,
        "Delta_A": delta_A,
        "Delta_B": delta_B,
        "Delta_AB": delta_AB,
        "n_q0": np.asarray(n_q0, dtype=float),
        "n_Q": np.asarray(n_Q, dtype=complex),
    }


def build_supercell_h0(
    kpts: np.ndarray,
    params: RubyParameters,
    source_strength: float = 0.0,
) -> np.ndarray:
    """Return the 18x18 supercell Bloch Hamiltonian.

    ``source_strength`` is retained for backward compatibility and adds the
    selected period-three ``co`` source

        H_source = -h sum_I pattern_I n_I.

    For other charge-source channels use :func:`add_charge_source`.
    """
    kpts = np.asarray(kpts, dtype=float)
    flat = kpts.reshape(-1, 2)
    h0 = np.zeros((flat.shape[0], NSUP, NSUP), dtype=complex)
    hops = supercell_hoppings(params)
    for ik, k in enumerate(flat):
        for I, J, S, amp in hops:
            h0[ik, I, J] += amp * np.exp(2j * np.pi * np.dot(k, S))

    h0 = 0.5 * (h0 + np.swapaxes(h0.conj(), -1, -2))
    h0 = h0.reshape(kpts.shape[:-1] + (NSUP, NSUP))
    if source_strength != 0.0:
        h0 = add_charge_source(h0, float(source_strength), "co")
    return h0


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
