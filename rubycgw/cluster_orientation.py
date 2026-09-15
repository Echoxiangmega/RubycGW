"""Gauge-related cluster orientations for the six-site Ruby embedding.

The physical Ruby Hamiltonian is C3 symmetric, but the six-site primitive-cell
cluster used by the ED self-energy correction is not: choosing which B triangle
belongs to the same cell as a given A triangle selects one of three equivalent
A-B neighbour directions.

Rather than forcing a single oriented impurity map onto a C3-symmetric fixed
point, we can run the ordinary convergent embedding in the three equivalent
unit-cell gauges and compare/average the resulting observables.

The three gauges differ only by the primitive-cell assignment of the B
triangle.  With A kept fixed, use B-cell shifts

    orientation 0 : s_B = ( 0, 0)
    orientation 1 : s_B = ( 0, 1)
    orientation 2 : s_B = (-1, 0)

For a fermionic Bloch matrix field X(k), changing the B-cell assignment by s
is the same-k unitary gauge transformation

    X_s(k) = D_s(k) X(k) D_s(k)^dagger,

where D_s=diag(1,1,1,e^{+2 pi i k.s},e^{+2 pi i k.s},e^{+2 pi i k.s}).
The same formula applies to the density interaction V(q), with k replaced by q.
At q=0 the projected six-orbital interaction is therefore unchanged.

The sign convention above is chosen so that the k-average one-body blocks
contain the three expected neighbouring A-B pairs:

    ori 0: A1-B1 and A2-B2
    ori 1: A0-B2 and A1-B0
    ori 2: A0-B1 and A2-B0.

These are exactly the three C3-related neighbour directions of the current Ruby
bond convention.
"""
from __future__ import annotations

import numpy as np

NSUB = 6

ORIENTATION_B_SHIFTS = np.asarray(
    [
        [0, 0],
        [0, 1],
        [-1, 0],
    ],
    dtype=int,
)


def orientation_b_shift(orientation: int) -> np.ndarray:
    """Return the integer B-cell shift for orientation 0, 1 or 2."""
    r = int(orientation)
    if r not in (0, 1, 2):
        raise ValueError(f"cluster orientation must be 0, 1 or 2; got {orientation!r}")
    return np.array(ORIENTATION_B_SHIFTS[r], copy=True)


def relative_b_shift(source_orientation: int, target_orientation: int) -> np.ndarray:
    """Return the gauge shift that maps source-orientation fields to target."""
    return orientation_b_shift(target_orientation) - orientation_b_shift(source_orientation)


def _phase_grid(n1: int, n2: int, shift) -> np.ndarray:
    s = np.asarray(shift, dtype=int).reshape(2)
    k1 = np.arange(int(n1), dtype=float) / float(n1)
    k2 = np.arange(int(n2), dtype=float) / float(n2)
    a, b = np.meshgrid(k1, k2, indexing="ij")
    return np.exp(2j * np.pi * (a * float(s[0]) + b * float(s[1])))


def gauge_transform_lattice(field: np.ndarray, shift) -> np.ndarray:
    """Apply a B-cell gauge shift to a lattice matrix field.

    ``field`` must end in ``(..., nk1, nk2, 6, 6)``.  Any leading axes, such as
    Matsubara frequency, are preserved.  The operation is unitary at every
    momentum point and therefore leaves spectra and gauge-invariant lattice
    observables unchanged.
    """
    x = np.asarray(field, dtype=complex)
    if x.ndim < 4 or x.shape[-2:] != (NSUB, NSUB):
        raise ValueError("lattice field must end in (...,nk1,nk2,6,6)")
    n1, n2 = int(x.shape[-4]), int(x.shape[-3])
    phase = _phase_grid(n1, n2, shift)
    d = np.ones((n1, n2, NSUB), dtype=complex)
    d[..., 3:] = phase[..., None]
    lead = (1,) * (x.ndim - 4)
    left = d.reshape(lead + (n1, n2, NSUB, 1))
    right = d.conj().reshape(lead + (n1, n2, 1, NSUB))
    return left * x * right


def transform_between_orientations(
    field: np.ndarray,
    source_orientation: int,
    target_orientation: int,
) -> np.ndarray:
    """Gauge-transform a lattice field between two cluster orientations."""
    return gauge_transform_lattice(
        field,
        relative_b_shift(source_orientation, target_orientation),
    )


def build_oriented_lattice_fields(
    h0: np.ndarray,
    Vq: np.ndarray,
    orientation: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(h0_r,Vq_r)`` for one of the three equivalent cluster gauges."""
    s = orientation_b_shift(orientation)
    return gauge_transform_lattice(h0, s), gauge_transform_lattice(Vq, s)


def intracell_block(field: np.ndarray) -> np.ndarray:
    """Return the k-average 6x6 block of a static lattice field."""
    x = np.asarray(field, dtype=complex)
    if x.ndim != 4 or x.shape[-2:] != (NSUB, NSUB):
        raise ValueError("static lattice field must have shape (nk1,nk2,6,6)")
    return np.mean(x, axis=(0, 1))


__all__ = [
    "ORIENTATION_B_SHIFTS",
    "orientation_b_shift",
    "relative_b_shift",
    "gauge_transform_lattice",
    "transform_between_orientations",
    "build_oriented_lattice_fields",
    "intracell_block",
]
