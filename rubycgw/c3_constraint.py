"""C3 symmetry utilities for the primitive-cell Ruby representation.

The primitive cell is not mapped to itself site by site under a 120-degree
rotation: the B triangle acquires a primitive-cell shift.  In reduced lattice
coordinates the active C3 operation is

    R -> M R + s_a,
    M = [[0,-1],[1,-1]],

with orbital permutation

    A: 0->2, 1->0, 2->1,
    B: 3->4, 4->5, 5->3,

and shifts s_A=(0,0), s_B=(0,1).  For a fermionic matrix field X(k) this gives

    X(k') = U(k) X(k) U(k)^dagger,
    k' = M^{-T} k = (-k1-k2, k1),

where U[p(a),a] = exp[-2 pi i k' . s_a].

The helpers below implement this exact representation on an N x N reciprocal
mesh.  They are used to construct a C3-constrained normal-state background
without pretending that a simple on-site orbital permutation is the full
physical lattice rotation.
"""
from __future__ import annotations

import numpy as np

NSUB = 6
C3_ORBITAL_PERM = np.asarray([2, 0, 1, 4, 5, 3], dtype=int)
C3_REALSPACE_MATRIX = np.asarray([[0, -1], [1, -1]], dtype=int)
C3_B_SHIFT = np.asarray([0, 1], dtype=int)


def _check_square_field(field: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.asarray(field, dtype=complex)
    if x.ndim < 4 or x.shape[-2:] != (NSUB, NSUB):
        raise ValueError("C3 lattice field must end in (...,Nk,Nk,6,6)")
    n1, n2 = x.shape[-4], x.shape[-3]
    if n1 != n2:
        raise ValueError(
            "the exact C3 projector requires a square reciprocal mesh, "
            f"got {n1}x{n2}"
        )
    return x, int(n1)


def c3_unitary_index(i: int, j: int, n: int) -> tuple[int, int, np.ndarray]:
    """Return rotated momentum indices and the 6x6 orbital unitary.

    The input point is k=(i/n,j/n).  The rotated point is
    k'=(-i-j,i)/n modulo reciprocal lattice vectors.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("mesh size must be positive")
    i = int(i) % n
    j = int(j) % n
    ip = (-i - j) % n
    jp = i % n

    u = np.zeros((NSUB, NSUB), dtype=complex)
    phase_b = np.exp(-2j * np.pi * (jp / float(n)))
    for a in range(NSUB):
        phase = 1.0 if a < 3 else phase_b
        u[C3_ORBITAL_PERM[a], a] = phase
    return ip, jp, u


def rotate_lattice_c3(field: np.ndarray) -> np.ndarray:
    """Apply one active C3 rotation to a fermionic lattice matrix field."""
    x, n = _check_square_field(field)
    out = np.empty_like(x)
    for i in range(n):
        for j in range(n):
            ip, jp, u = c3_unitary_index(i, j, n)
            block = x[..., i, j, :, :]
            out[..., ip, jp, :, :] = np.einsum(
                "ab,...bc,cd->...ad", u, block, u.conj().T, optimize=True
            )
    return out


def project_lattice_c3(field: np.ndarray) -> np.ndarray:
    """Project a lattice matrix field onto the physical C3-invariant subspace."""
    x, _ = _check_square_field(field)
    r1 = rotate_lattice_c3(x)
    r2 = rotate_lattice_c3(r1)
    return (x + r1 + r2) / 3.0


def c3_lattice_residual(field: np.ndarray) -> float:
    """Maximum covariance residual |R X - X| for a lattice matrix field."""
    x, _ = _check_square_field(field)
    return float(np.max(np.abs(rotate_lattice_c3(x) - x), initial=0.0))


def project_density_c3(density: np.ndarray) -> np.ndarray:
    """Enforce the translation-invariant C3 density constraint on A and B.

    C3 relates the three sites of each elementary triangle even though the B
    sites are shifted between primitive cells under rotation.  Translation
    invariance therefore implies equal on-site density within each triangle.
    """
    d = np.asarray(density, dtype=float).reshape(NSUB).copy()
    d[:3] = np.mean(d[:3])
    d[3:] = np.mean(d[3:])
    return d


def density_c3_spread(density: np.ndarray) -> float:
    """Largest within-triangle density splitting."""
    d = np.asarray(density, dtype=float).reshape(NSUB)
    sa = float(np.max(d[:3]) - np.min(d[:3]))
    sb = float(np.max(d[3:]) - np.min(d[3:]))
    return max(sa, sb)


__all__ = [
    "C3_ORBITAL_PERM",
    "C3_REALSPACE_MATRIX",
    "c3_unitary_index",
    "rotate_lattice_c3",
    "project_lattice_c3",
    "c3_lattice_residual",
    "project_density_c3",
    "density_c3_spread",
]
