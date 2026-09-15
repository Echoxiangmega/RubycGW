"""Ruby-lattice V' + Vx interaction extension.

``Vprime`` is the short-range interaction on the same two microscopic links as
the inter-triangle hopping for each neighboring A/B triangle pair.  ``Vcross``
adds the two crossed density links inside that same neighboring triangle pair.
For the reference pair (gamma=0),

    straight V' : A1-B1, A2-B2
    crossed  Vx : A1-B2, A2-B1

The other two orientations are generated with the same primitive-cell
bookkeeping as the Ruby hopping graph.  Importantly, this does *not* add the
very long same-cell A0-B0 interaction between the far-left/far-right sites in
the usual real-space drawing.

The lattice interaction retains all real-space offsets.  The six-site impurity
uses the q=0 primitive-cell projection, so distinct crossed bonds that project
to the same orbital pair are summed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rubycgw.model import NSUB
from .model import INTRATRIANGLE_BONDS, INTERTRIANGLE_BONDS


@dataclass(frozen=True)
class VPrimeCrossParameters:
    ti: float = 0.4
    t1: float = 0.2
    t2: float = 0.2
    V: float = 0.2
    Vprime: float = 0.0
    Vcross: float = 0.0


# Crossed links for the same three neighboring A/B triangle pairs represented
# by INTERTRIANGLE_BONDS.  We orient them A -> B for readability.
#
# gamma=0: A(cell R) with B(cell R)
#   straight: A1-B1, A2-B2
#   crossed : A1-B2, A2-B1
# gamma=1: A(cell R) with B(cell R+(0,1))
#   straight: A0-B2, A1-B0
#   crossed : A0-B0, A1-B2
# gamma=2: A(cell R) with B(cell R+(-1,0))
#   straight: A2-B0, A0-B1
#   crossed : A2-B1, A0-B0
#
# B local labels are B0=3, B1=4, B2=5.
CROSS_INTERTRIANGLE_BONDS = (
    (1, 5, (0, 0)),
    (2, 4, (0, 0)),
    (0, 3, (0, 1)),
    (1, 5, (0, 1)),
    (2, 4, (-1, 0)),
    (0, 3, (-1, 0)),
)

REFERENCE_STRAIGHT_BONDS = (
    (1, 4, (0, 0)),
    (2, 5, (0, 0)),
)

REFERENCE_CROSS_BONDS = (
    (1, 5, (0, 0)),
    (2, 4, (0, 0)),
)


def vprime_vcross_interaction_bonds(params: VPrimeCrossParameters):
    """Return all real-space density bonds ``(i,j,R,coupling)``."""
    out = [(i, j, R, float(params.V)) for i, j, R in INTRATRIANGLE_BONDS]
    out.extend(
        (i, j, R, float(params.Vprime)) for i, j, R in INTERTRIANGLE_BONDS
    )
    out.extend(
        (i, j, R, float(params.Vcross)) for i, j, R in CROSS_INTERTRIANGLE_BONDS
    )
    return tuple(out)


def build_vprime_vcross_interaction(
    qpts: np.ndarray,
    params: VPrimeCrossParameters,
) -> np.ndarray:
    """Build the full momentum-dependent 6x6 density interaction V(q)."""
    qpts = np.asarray(qpts, dtype=float)
    flat = qpts.reshape(-1, 2)
    vq = np.zeros((flat.shape[0], NSUB, NSUB), dtype=complex)
    for iq, q in enumerate(flat):
        for i, j, R, coupling in vprime_vcross_interaction_bonds(params):
            Rv = np.asarray(R, dtype=int)
            phase = np.exp(2j * np.pi * np.dot(q, Rv))
            vq[iq, i, j] += coupling * phase
            vq[iq, j, i] += coupling * np.conj(phase)
    vq = 0.5 * (vq + np.swapaxes(vq.conj(), -1, -2))
    return vq.reshape(qpts.shape[:-1] + (NSUB, NSUB))


def vprime_vcross_cluster_interactions(params: VPrimeCrossParameters):
    """Return the q=0 six-orbital projection for the impurity ED problem.

    Primitive-cell offsets are dropped and repeated projected orbital pairs are
    summed.  In particular, the six physical crossed bonds project onto only
    three correlated-orbital pairs, (0,3), (1,5), (2,4), each with coupling
    ``2*Vcross`` at q=0.
    """
    pair_coupling: dict[tuple[int, int], float] = {}
    for i, j, _R, coupling in vprime_vcross_interaction_bonds(params):
        a, b = sorted((int(i), int(j)))
        if a == b:
            continue
        pair_coupling[(a, b)] = pair_coupling.get((a, b), 0.0) + float(coupling)
    return tuple((a, b, u) for (a, b), u in sorted(pair_coupling.items()))


def vprime_vcross_cluster_matrix(params: VPrimeCrossParameters) -> np.ndarray:
    """Return the q=0 6x6 interaction matrix used by the cluster correction."""
    out = np.zeros((NSUB, NSUB), dtype=complex)
    for i, j, u in vprime_vcross_cluster_interactions(params):
        out[i, j] += u
        out[j, i] += u
    return out


def reference_pair_effective_couplings(
    *,
    t: float,
    V: float,
    Vprime: float,
    Vcross: float,
) -> tuple[float, float, float]:
    """Leading strong-coupling (Jn,Jm,Jz) for the reference triangle pair.

    This helper implements the working-note projection for t1=t2=t and keeps
    O(t^2/V), O(Vprime), O(Vcross):

      Jn = 5 t^2/(9V) + (Vprime + Vcross)/18
      Jm = -t^2/(3V) + (-Vprime + Vcross)/6
      Jz = -t^2/(3V)

    It is a diagnostic only; the microscopic ED+GW/JF calculation uses the full
    bond interaction above and does not rely on this approximation.
    """
    t = float(t)
    V = float(V)
    if V == 0.0:
        raise ValueError("V must be nonzero in the strong-coupling diagnostic")
    s = t * t / V
    jn = 5.0 * s / 9.0 + (float(Vprime) + float(Vcross)) / 18.0
    jm = -s / 3.0 + (-float(Vprime) + float(Vcross)) / 6.0
    jz = -s / 3.0
    return jn, jm, jz
