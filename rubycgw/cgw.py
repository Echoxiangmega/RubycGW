"""Reference q=(0,0) covariant-GW vertex solver for eta_+/-.

The solver keeps the transparent fixed-point formulation but avoids repeated
work inside each iteration: X=G Gamma G is formed once, and Hartree/MT/AL
corrections share the same internal-Q loop.  A previous converged vertex can
also be supplied as a continuation/warm-start guess.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices, roll_spatial
from .model import NSUB


@dataclass
class VertexOptions:
    max_iter: int = 100
    tol: float = 1e-8
    mixing: float = 0.25
    include_hartree: bool = True
    include_mt: bool = True
    include_al: bool = True
    verbose: bool = True


@dataclass
class VertexResult:
    Gamma: np.ndarray
    Gamma_H: np.ndarray
    Gamma_MT: np.ndarray
    Gamma_AL1: np.ndarray
    Gamma_AL2: np.ndarray
    converged: bool
    iterations: int


def _x_field(G: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    return np.einsum("...ab,...bc,...cd->...ad", G, Gamma, G, optimize=True)


def _hartree_from_x(X: np.ndarray, Vq0: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    xdiag = np.diagonal(X, axis1=-2, axis2=-1)
    response_density = (grid.T / grid.nk) * np.sum(xdiag, axis=(0, 1, 2))
    diag = Vq0 @ response_density
    mat = np.zeros((NSUB, NSUB), dtype=complex)
    mat[np.diag_indices(NSUB)] = diag
    return np.broadcast_to(mat, X.shape).copy()


def _vertex_corrections_q0(
    G: np.ndarray,
    W: np.ndarray,
    Gamma: np.ndarray,
    Vq0: np.ndarray,
    grid: MatsubaraGrid,
    include_hartree: bool,
    include_mt: bool,
    include_al: bool,
):
    """Compute all requested q=0 corrections in one shared pass.

    The previous implementation formed X=G Gamma G separately in Hartree, MT,
    and AL routines and shifted X independently for MT and AL.  Here X is
    formed once.  For every bosonic m only the valid Matsubara slices are used;
    MT, AL1, and AL2 then share the same spatially shifted X(k+Q), while AL also
    uses G(k+Q).  This removes the largest avoidable duplication without
    changing the equations.
    """
    X = _x_field(G, Gamma)
    zero = np.zeros_like(Gamma)
    gh = _hartree_from_x(X, Vq0, grid) if include_hartree else zero.copy()
    gmt = zero.copy()
    gal1 = zero.copy()
    gal2 = zero.copy()

    if not include_mt and not include_al:
        return gh, gmt, gal1, gal2

    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue

        Xsrc = X[src]
        Xbase = X[dst]
        Gsrc = G[src] if include_al else None
        Gbase = G[dst] if include_al else None

        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                Xq = roll_spatial(Xsrc, iq1, iq2)
                WQ = W[im, iq1, iq2]
                WT = WQ.T[None, None, None, :, :]

                if include_mt:
                    gmt[dst] -= pref * Xq * WT

                if include_al:
                    Gq = roll_spatial(Gsrc, iq1, iq2)
                    # L1_ef = int_k X_ef(k+Q) G_fe(k)
                    L1 = pref * np.einsum(
                        "nxyef,nxyfe->ef", Xq, Gbase, optimize=True
                    )
                    # L2_ef = int_k G_ef(k+Q) X_fe(k)
                    L2 = pref * np.einsum(
                        "nxyef,nxyfe->ef", Gq, Xbase, optimize=True
                    )
                    M1 = WQ @ L1 @ WQ
                    M2 = WQ @ L2 @ WQ
                    gal1[dst] -= pref * Gq * M1.T[None, None, None, :, :]
                    gal2[dst] -= pref * Gq * M2.T[None, None, None, :, :]

    return gh, gmt, gal1, gal2


def gamma_h_q0(G: np.ndarray, Gamma: np.ndarray, Vq0: np.ndarray,
               grid: MatsubaraGrid) -> np.ndarray:
    """Hartree vertex at external q=(0,0)."""
    return _hartree_from_x(_x_field(G, Gamma), Vq0, grid)


def gamma_mt_q0(G: np.ndarray, W: np.ndarray, Gamma: np.ndarray,
                grid: MatsubaraGrid) -> np.ndarray:
    """MT term: -int_Q [G Gamma G](p+Q) * W(Q)^T elementwise."""
    X = _x_field(G, Gamma)
    out = np.zeros_like(Gamma)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        Xsrc = X[src]
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                Xq = roll_spatial(Xsrc, iq1, iq2)
                out[dst] -= pref * Xq * W[im, iq1, iq2].T[None, None, None, :, :]
    return out


def gamma_al_q0(G: np.ndarray, W: np.ndarray, Gamma: np.ndarray,
                grid: MatsubaraGrid):
    """Return AL1 and AL2 at external q=(0,0)."""
    X = _x_field(G, Gamma)
    al1 = np.zeros_like(Gamma)
    al2 = np.zeros_like(Gamma)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        Xsrc, Xbase = X[src], X[dst]
        Gsrc, Gbase = G[src], G[dst]
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                Xq = roll_spatial(Xsrc, iq1, iq2)
                Gq = roll_spatial(Gsrc, iq1, iq2)
                L1 = pref * np.einsum("nxyef,nxyfe->ef", Xq, Gbase, optimize=True)
                L2 = pref * np.einsum("nxyef,nxyfe->ef", Gq, Xbase, optimize=True)
                WQ = W[im, iq1, iq2]
                M1 = WQ @ L1 @ WQ
                M2 = WQ @ L2 @ WQ
                al1[dst] -= pref * Gq * M1.T[None, None, None, :, :]
                al2[dst] -= pref * Gq * M2.T[None, None, None, :, :]
    return al1, al2


def _initial_gamma_field(initial_gamma: np.ndarray | None,
                         Kfield: np.ndarray) -> np.ndarray:
    if initial_gamma is None:
        return Kfield.copy()
    arr = np.asarray(initial_gamma)
    if arr.shape == Kfield.shape:
        return np.array(arr, copy=True)
    if arr.shape == Kfield.shape[-2:]:
        return np.broadcast_to(arr, Kfield.shape).copy()
    return Kfield.copy()


def solve_vertex_q0(G: np.ndarray, W: np.ndarray, Vq0: np.ndarray,
                    K: np.ndarray, grid: MatsubaraGrid,
                    opts: VertexOptions = VertexOptions(),
                    initial_gamma: np.ndarray | None = None) -> VertexResult:
    """Solve the q=(0,0) cGW eta vertex by fixed-point iteration.

    ``initial_gamma`` can be a previous converged vertex with the same
    fermionic grid shape.  It is particularly useful for continuation scans and
    for starting the full MT+AL solve from the already converged MT-only vertex.
    """
    Kfield = np.broadcast_to(K, G.shape).copy()
    Gamma = _initial_gamma_field(initial_gamma, Kfield)
    gh = np.zeros_like(Gamma)
    gmt = np.zeros_like(Gamma)
    gal1 = np.zeros_like(Gamma)
    gal2 = np.zeros_like(Gamma)
    converged = False

    for it in range(1, opts.max_iter + 1):
        gh, gmt, gal1, gal2 = _vertex_corrections_q0(
            G, W, Gamma, Vq0, grid,
            include_hartree=opts.include_hartree,
            include_mt=opts.include_mt,
            include_al=opts.include_al,
        )
        rhs = Kfield + gh + gmt + gal1 + gal2
        Gnew = (1.0 - opts.mixing) * Gamma + opts.mixing * rhs
        err = float(np.max(np.abs(Gnew - Gamma)))
        Gamma = Gnew
        if opts.verbose:
            print(f"cGW vertex iter {it:4d}: err={err:.3e}")
        if err < opts.tol:
            converged = True
            break

    return VertexResult(
        Gamma=Gamma,
        Gamma_H=gh,
        Gamma_MT=gmt,
        Gamma_AL1=gal1,
        Gamma_AL2=gal2,
        converged=converged,
        iterations=it,
    )
