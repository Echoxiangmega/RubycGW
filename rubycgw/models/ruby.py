"""Public Ruby-lattice model definitions.

This module is the canonical user-facing model layer.  The original
``rubycgw.model`` module remains untouched for backwards compatibility, while
new code should construct :class:`RubyModel` here.

The extended interaction supports the two short-range terms used by the recent
Ruby-lattice studies:

``Vprime``
    Density interaction on the same two straight inter-triangle links as the
    hopping for each neighbouring A/B triangle pair.

``Vcross``
    Density interaction on the two crossed links inside the same neighbouring
    A/B triangle pair.  This does not add the long same-cell A0-B0 link.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model import NSUB, RubyParameters, build_h0 as _build_h0


@dataclass(frozen=True)
class ExtendedRubyParameters(RubyParameters):
    """Ruby parameters including optional straight/crossed inter-triangle terms."""

    Vprime: float = 0.0
    Vcross: float = 0.0


@dataclass(frozen=True)
class RubyModel:
    """High-level immutable description of the spinless Ruby model.

    Parameters use the conventions of the existing research code.  Setting
    ``Vprime=Vcross=0`` reproduces the baseline model.
    """

    ti: float = 0.4
    t1: float = 0.2
    t2: float = 0.2
    V: float = 0.2
    Vprime: float = 0.0
    Vcross: float = 0.0

    def parameters(self) -> ExtendedRubyParameters:
        return ExtendedRubyParameters(
            ti=float(self.ti),
            t1=float(self.t1),
            t2=float(self.t2),
            V=float(self.V),
            Vprime=float(self.Vprime),
            Vcross=float(self.Vcross),
        )

    def build_h0(self, kpts: np.ndarray) -> np.ndarray:
        return _build_h0(kpts, self.parameters())

    def build_interaction(self, qpts: np.ndarray) -> np.ndarray:
        return build_extended_interaction(qpts, self.parameters())

    def cluster_interactions(self):
        """Return the production ED impurity interaction subset (V only)."""
        return v_only_cluster_interactions(self.parameters())

    def effective_couplings(self) -> tuple[float, float, float]:
        if not np.isclose(float(self.t1), float(self.t2), rtol=0.0, atol=1e-14):
            raise ValueError("leading pseudospin couplings require t1=t2")
        return reference_pair_effective_couplings(
            t=float(self.t1),
            V=float(self.V),
            Vprime=float(self.Vprime),
            Vcross=float(self.Vcross),
        )


# Six intra-triangle density bonds.
INTRATRIANGLE_BONDS = (
    (0, 1, (0, 0)),
    (0, 2, (0, 0)),
    (2, 1, (0, 0)),
    (3, 4, (0, 0)),
    (3, 5, (0, 0)),
    (4, 5, (0, 0)),
)

# Straight inter-triangle links, exactly matching the six t1/t2 hopping bonds.
INTERTRIANGLE_BONDS = (
    (1, 4, (0, 0)),
    (5, 0, (0, -1)),
    (2, 3, (-1, 0)),
    (3, 1, (0, -1)),
    (2, 5, (0, 0)),
    (0, 4, (-1, 0)),
)

# Crossed links for the same three neighbouring A/B triangle pairs.
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


def extended_interaction_bonds(params: ExtendedRubyParameters):
    """Return all real-space density bonds ``(i,j,R,coupling)``."""
    out = [(i, j, R, float(params.V)) for i, j, R in INTRATRIANGLE_BONDS]
    out.extend((i, j, R, float(params.Vprime)) for i, j, R in INTERTRIANGLE_BONDS)
    out.extend((i, j, R, float(params.Vcross)) for i, j, R in CROSS_INTERTRIANGLE_BONDS)
    return tuple(out)


def build_extended_interaction(
    qpts: np.ndarray,
    params: ExtendedRubyParameters,
) -> np.ndarray:
    """Build the full momentum-dependent 6x6 density interaction ``V(q)``."""
    qpts = np.asarray(qpts, dtype=float)
    flat = qpts.reshape(-1, 2)
    vq = np.zeros((flat.shape[0], NSUB, NSUB), dtype=complex)
    for iq, q in enumerate(flat):
        for i, j, R, coupling in extended_interaction_bonds(params):
            rv = np.asarray(R, dtype=int)
            phase = np.exp(2j * np.pi * np.dot(q, rv))
            vq[iq, i, j] += coupling * phase
            vq[iq, j, i] += coupling * np.conj(phase)
    vq = 0.5 * (vq + np.swapaxes(vq.conj(), -1, -2))
    return vq.reshape(qpts.shape[:-1] + (NSUB, NSUB))


def v_only_cluster_interactions(params: ExtendedRubyParameters):
    """Return only the six intra-triangle V bonds used by production ED."""
    return tuple(
        (int(i), int(j), float(params.V))
        for i, j, _R in INTRATRIANGLE_BONDS
    )


def extended_cluster_interactions(params: ExtendedRubyParameters):
    """Return the q=0 six-orbital interaction projection used by impurity ED.

    Primitive-cell offsets are dropped and repeated projected orbital pairs are
    summed.  In particular, the six physical crossed bonds project onto three
    orbital pairs with coupling ``2*Vcross`` each.
    """
    pair_coupling: dict[tuple[int, int], float] = {}
    for i, j, _R, coupling in extended_interaction_bonds(params):
        a, b = sorted((int(i), int(j)))
        if a == b:
            continue
        pair_coupling[(a, b)] = pair_coupling.get((a, b), 0.0) + float(coupling)
    return tuple((a, b, u) for (a, b), u in sorted(pair_coupling.items()))


def extended_cluster_matrix(params: ExtendedRubyParameters) -> np.ndarray:
    out = np.zeros((NSUB, NSUB), dtype=complex)
    for i, j, u in extended_cluster_interactions(params):
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
    """Leading strong-coupling couplings ``(Jn,Jm,Jz)`` for t1=t2=t."""
    t = float(t)
    V = float(V)
    if V == 0.0:
        raise ValueError("V must be nonzero in the strong-coupling diagnostic")
    s = t * t / V
    jn = 5.0 * s / 9.0 + (float(Vprime) + float(Vcross)) / 18.0
    jm = -s / 3.0 + (-float(Vprime) + float(Vcross)) / 6.0
    jz = -s / 3.0
    return jn, jm, jz


__all__ = [
    "RubyParameters",
    "ExtendedRubyParameters",
    "RubyModel",
    "INTRATRIANGLE_BONDS",
    "INTERTRIANGLE_BONDS",
    "CROSS_INTERTRIANGLE_BONDS",
    "REFERENCE_STRAIGHT_BONDS",
    "REFERENCE_CROSS_BONDS",
    "extended_interaction_bonds",
    "build_extended_interaction",
    "v_only_cluster_interactions",
    "extended_cluster_interactions",
    "extended_cluster_matrix",
    "reference_pair_effective_couplings",
]
