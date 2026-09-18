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

from .c3_constraint import rotate_lattice_c3

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
    """Return the pure gauge shift that maps source fields to target gauge."""
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
    """Pure-gauge transform of the *same physical state* between orientations."""
    return gauge_transform_lattice(
        field,
        relative_b_shift(source_orientation, target_orientation),
    )


def rotate_solution_between_orientations(
    field: np.ndarray,
    source_orientation: int,
    target_orientation: int,
) -> np.ndarray:
    """Map an oriented solution to its C3-related partner in another gauge.

    This differs from :func:`transform_between_orientations`, which keeps the
    physical state fixed.  Here we first return the source field to orientation
    0 gauge, actively rotate the physical state by the number of C3 steps that
    takes the source cluster cut into the target cut, and finally express the
    rotated state in the target gauge.  This is the appropriate warm start for
    the corresponding symmetry-related cluster solution.
    """
    src = int(source_orientation)
    dst = int(target_orientation)
    orientation_b_shift(src)
    orientation_b_shift(dst)
    x0 = gauge_transform_lattice(field, -orientation_b_shift(src))
    steps = (dst - src) % 3
    xr = np.asarray(x0, dtype=complex)
    for _ in range(steps):
        xr = rotate_lattice_c3(xr)
    return gauge_transform_lattice(xr, orientation_b_shift(dst))


def rotate_local_between_orientations(
    field: np.ndarray,
    source_orientation: int,
    target_orientation: int,
    *,
    nk1: int,
    nk2: int,
) -> np.ndarray:
    """Rotate a k-independent cluster matrix into the target cluster frame.

    A local matrix is broadcast over a reciprocal mesh, mapped with
    :func:`rotate_solution_between_orientations`, and then averaged.  For the
    three C3-related cluster frames the mapped field is k independent up to
    roundoff, so this provides a robust way to transform impurity self-energies
    or other local matrices without hard-coding a separate orbital permutation.
    """
    x = np.asarray(field, dtype=complex)
    if x.ndim < 2 or x.shape[-2:] != (NSUB, NSUB):
        raise ValueError("local field must end in (...,6,6)")
    lead = x.shape[:-2]
    lattice = np.broadcast_to(
        x.reshape(lead + (1, 1, NSUB, NSUB)),
        lead + (int(nk1), int(nk2), NSUB, NSUB),
    ).copy()
    mapped = rotate_solution_between_orientations(
        lattice, source_orientation, target_orientation
    )
    local = np.mean(mapped, axis=(-4, -3))
    ref = local.reshape(lead + (1, 1, NSUB, NSUB))
    spread = float(np.max(np.abs(mapped - ref), initial=0.0))
    if spread > 1.0e-10:
        raise RuntimeError(
            "C3-related local cluster matrix did not remain local under frame map: "
            f"max spread={spread:.3e}"
        )
    return local


def build_oriented_lattice_fields(
    h0: np.ndarray,
    Vq: np.ndarray,
    orientation: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(h0_r,Vq_r)`` for one of the three equivalent cluster gauges."""
    s = orientation_b_shift(orientation)
    return gauge_transform_lattice(h0, s), gauge_transform_lattice(Vq, s)



def gauge_transform_response_field(
    field: np.ndarray,
    shift,
    q_index: tuple[int, int],
) -> np.ndarray:
    """Gauge-transform a two-momentum response field Gamma(k;q).

    For c'_k=D_s(k)c_k, a vertex/self-energy response carrying external
    momentum q transforms as

        X'_s(k;q) = D_s(k+q) X(k;q) D_s(k)^dagger.

    This reduces to :func:`gauge_transform_lattice` at q=0.
    """
    x = np.asarray(field, dtype=complex)
    if x.ndim < 4 or x.shape[-2:] != (NSUB, NSUB):
        raise ValueError("response field must end in (...,nk1,nk2,6,6)")
    n1, n2 = int(x.shape[-4]), int(x.shape[-3])
    iq1 = int(q_index[0]) % n1
    iq2 = int(q_index[1]) % n2
    s = np.asarray(shift, dtype=int).reshape(2)

    phase_k = _phase_grid(n1, n2, s)
    qphase = np.exp(
        2j * np.pi * (
            float(iq1) * float(s[0]) / float(n1)
            + float(iq2) * float(s[1]) / float(n2)
        )
    )
    phase_kq = qphase * phase_k

    dl = np.ones((n1, n2, NSUB), dtype=complex)
    dr = np.ones((n1, n2, NSUB), dtype=complex)
    dl[..., 3:] = phase_kq[..., None]
    dr[..., 3:] = phase_k[..., None]

    lead = (1,) * (x.ndim - 4)
    left = dl.reshape(lead + (n1, n2, NSUB, 1))
    right = dr.conj().reshape(lead + (n1, n2, 1, NSUB))
    return left * x * right


def local_response_vertex_in_orientation(
    vertex: np.ndarray,
    orientation: int,
    q_index: tuple[int, int],
    *,
    nk1: int,
    nk2: int,
    atol: float = 1.0e-10,
) -> np.ndarray:
    """Transform a local source to an orientation frame and require locality.

    Primitive-cell pseudospin vertices are block diagonal in the A/B triangle
    sectors, so after a cell-gauge change they remain k independent even at
    finite q (the B block only acquires the overall external-momentum phase).
    """
    k = np.asarray(vertex, dtype=complex)
    if k.shape != (NSUB, NSUB):
        raise ValueError("local response vertex must be 6x6")
    field = np.broadcast_to(
        k.reshape(1, 1, NSUB, NSUB),
        (int(nk1), int(nk2), NSUB, NSUB),
    ).copy()
    mapped = gauge_transform_response_field(
        field, orientation_b_shift(int(orientation)), q_index
    )
    local = np.mean(mapped, axis=(0, 1))
    spread = float(np.max(np.abs(mapped - local[None, None]), initial=0.0))
    if spread > float(atol):
        raise ValueError(
            "source becomes k dependent in the requested orientation; "
            f"spread={spread:.3e}"
        )
    return local

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
    "rotate_solution_between_orientations",
    "rotate_local_between_orientations",
    "build_oriented_lattice_fields",
    "gauge_transform_response_field",
    "local_response_vertex_in_orientation",
    "intracell_block",
]
