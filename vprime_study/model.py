"""Ruby-lattice extension with an inter-triangle density interaction ``Vprime``.

The baseline model has repulsive ``V`` only on the six bonds forming the two
triangles in each primitive cell.  The extension studied here adds

    H_V' = V' * sum_<ij>_inter n_i n_j

on the same six inter-triangle bonds that carry the ``t1``/``t2`` hoppings.
Negative ``Vprime`` therefore means attraction.

The lattice interaction keeps the true primitive-cell offsets of those six
bonds, so V(q) becomes momentum dependent.  The six-site impurity cannot retain
those spatial offsets explicitly.  For the cluster ED correction we therefore
use the primitive-cell/q=0 projection: every inter-triangle bond contributes to
the corresponding pair of the six correlated orbitals.  This makes the
interaction used by the impurity exactly equal to the V(q=0) matrix used by the
cluster-GW double counting, while the lattice GW/JF part still retains the full
q dependence of V'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rubycgw.model import NSUB


@dataclass(frozen=True)
class VPrimeParameters:
    ti: float = 0.4
    t1: float = 0.2
    t2: float = 0.2
    V: float = 0.2
    Vprime: float = 0.0


# Exactly the six inter-triangle hopping bonds of rubycgw.model._base_bonds.
# The offsets are from orbital i in cell R to orbital j in cell R+delta.
INTERTRIANGLE_BONDS = (
    (1, 4, (0, 0)),
    (5, 0, (0, -1)),
    (2, 3, (-1, 0)),
    (3, 1, (0, -1)),
    (2, 5, (0, 0)),
    (0, 4, (-1, 0)),
)

INTRATRIANGLE_BONDS = (
    (0, 1, (0, 0)),
    (0, 2, (0, 0)),
    (2, 1, (0, 0)),
    (3, 4, (0, 0)),
    (3, 5, (0, 0)),
    (4, 5, (0, 0)),
)


def vprime_interaction_bonds(params: VPrimeParameters):
    """Return all real-space density bonds ``(i,j,R,coupling)``."""
    out = [(i, j, R, float(params.V)) for i, j, R in INTRATRIANGLE_BONDS]
    out.extend(
        (i, j, R, float(params.Vprime)) for i, j, R in INTERTRIANGLE_BONDS
    )
    return tuple(out)


def build_vprime_interaction(qpts: np.ndarray, params: VPrimeParameters) -> np.ndarray:
    """Build the full 6x6 Fourier-space density interaction V(q)."""
    qpts = np.asarray(qpts, dtype=float)
    flat = qpts.reshape(-1, 2)
    vq = np.zeros((flat.shape[0], NSUB, NSUB), dtype=complex)
    for iq, q in enumerate(flat):
        for i, j, R, coupling in vprime_interaction_bonds(params):
            Rv = np.asarray(R, dtype=int)
            phase = np.exp(2j * np.pi * np.dot(q, Rv))
            vq[iq, i, j] += coupling * phase
            vq[iq, j, i] += coupling * np.conj(phase)
    vq = 0.5 * (vq + np.swapaxes(vq.conj(), -1, -2))
    return vq.reshape(qpts.shape[:-1] + (NSUB, NSUB))


def vprime_cluster_interactions(params: VPrimeParameters):
    """Return the q=0 six-orbital projection as impurity ``(i,j,Uij)`` terms.

    Primitive-cell offsets are intentionally dropped here.  If several physical
    bonds projected onto the same orbital pair their couplings would be summed.
    For the present Ruby convention the twelve projected pairs are distinct.
    """
    pair_coupling: dict[tuple[int, int], float] = {}
    for i, j, _R, coupling in vprime_interaction_bonds(params):
        a, b = sorted((int(i), int(j)))
        if a == b:
            continue
        pair_coupling[(a, b)] = pair_coupling.get((a, b), 0.0) + float(coupling)
    return tuple((a, b, u) for (a, b), u in sorted(pair_coupling.items()))


def vprime_cluster_matrix(params: VPrimeParameters) -> np.ndarray:
    """Return the 6x6 q=0 interaction matrix used by the cluster correction."""
    out = np.zeros((NSUB, NSUB), dtype=complex)
    for i, j, u in vprime_cluster_interactions(params):
        out[i, j] += u
        out[j, i] += u
    return out
