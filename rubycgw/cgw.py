"""Reference q=(0,0) covariant-GW vertex solver for eta_+/-.

This module implements

    Gamma = K + Gamma_H + Gamma_MT + Gamma_AL1 + Gamma_AL2

on top of a converged GW solution. It is intentionally literal and intended
first for small-grid validation rather than production speed.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid, shift_fermion_field
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


def gamma_h_q0(G: np.ndarray, Gamma: np.ndarray, Vq0: np.ndarray,
               grid: MatsubaraGrid) -> np.ndarray:
    """Hartree vertex at external q=(0,0)."""
    X = _x_field(G, Gamma)
    xdiag = np.diagonal(X, axis1=-2, axis2=-1)
    response_density = (grid.T / grid.nk) * np.sum(xdiag, axis=(0, 1, 2))
    diag = Vq0 @ response_density
    mat = np.zeros((NSUB, NSUB), dtype=complex)
    mat[np.diag_indices(NSUB)] = diag
    return np.broadcast_to(mat, G.shape).copy()


def gamma_mt_q0(G: np.ndarray, W: np.ndarray, Gamma: np.ndarray,
                grid: MatsubaraGrid) -> np.ndarray:
    """MT term: -int_Q [G Gamma G](p+Q) * W(Q)^T elementwise."""
    X = _x_field(G, Gamma)
    out = np.zeros_like(Gamma)
    pref = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                Xq = shift_fermion_field(X, iq1, iq2, int(m))
                out -= pref * Xq * W[im, iq1, iq2].T[None, None, None, :, :]
    return out


def _al_loops_q0(G: np.ndarray, X: np.ndarray, iq1: int, iq2: int, m: int,
                 grid: MatsubaraGrid):
    Gq = shift_fermion_field(G, iq1, iq2, m)
    Xq = shift_fermion_field(X, iq1, iq2, m)
    pref = grid.T / grid.nk
    # L1_ef = int_k X_ef(k+Q) G_fe(k)
    L1 = pref * np.einsum("nxyef,nxyfe->ef", Xq, G, optimize=True)
    # L2_ef = int_k G_ef(k+Q) X_fe(k)
    L2 = pref * np.einsum("nxyef,nxyfe->ef", Gq, X, optimize=True)
    return Gq, L1, L2


def gamma_al_q0(G: np.ndarray, W: np.ndarray, Gamma: np.ndarray,
                grid: MatsubaraGrid):
    """Return AL1 and AL2 at external q=(0,0).

    The density projectors reduce the explicit four sublattice sums to

        M1(Q) = W(Q) L1(Q) W(Q)
        M2(Q) = W(Q) L2(Q) W(Q)

    followed by an elementwise contraction with G(p+Q).
    """
    X = _x_field(G, Gamma)
    al1 = np.zeros_like(Gamma)
    al2 = np.zeros_like(Gamma)
    prefQ = grid.T / grid.nk
    for im, m in enumerate(grid.m_values):
        for iq1 in range(grid.nk1):
            for iq2 in range(grid.nk2):
                Gq, L1, L2 = _al_loops_q0(G, X, iq1, iq2, int(m), grid)
                WQ = W[im, iq1, iq2]
                M1 = WQ @ L1 @ WQ
                M2 = WQ @ L2 @ WQ
                al1 -= prefQ * Gq * M1.T[None, None, None, :, :]
                al2 -= prefQ * Gq * M2.T[None, None, None, :, :]
    return al1, al2


def solve_vertex_q0(G: np.ndarray, W: np.ndarray, Vq0: np.ndarray,
                    K: np.ndarray, grid: MatsubaraGrid,
                    opts: VertexOptions = VertexOptions()) -> VertexResult:
    """Solve the full q=(0,0) cGW eta vertex by fixed-point iteration."""
    Kfield = np.broadcast_to(K, G.shape).copy()
    Gamma = Kfield.copy()
    gh = np.zeros_like(Gamma)
    gmt = np.zeros_like(Gamma)
    gal1 = np.zeros_like(Gamma)
    gal2 = np.zeros_like(Gamma)
    converged = False

    for it in range(1, opts.max_iter + 1):
        gh = gamma_h_q0(G, Gamma, Vq0, grid) if opts.include_hartree else 0.0
        gmt = gamma_mt_q0(G, W, Gamma, grid) if opts.include_mt else 0.0
        if opts.include_al:
            gal1, gal2 = gamma_al_q0(G, W, Gamma, grid)
        else:
            gal1 = 0.0
            gal2 = 0.0

        rhs = Kfield + gh + gmt + gal1 + gal2
        Gnew = (1.0 - opts.mixing) * Gamma + opts.mixing * rhs
        err = float(np.max(np.abs(Gnew - Gamma)))
        Gamma = Gnew
        if opts.verbose:
            print(f"cGW vertex iter {it:4d}: err={err:.3e}")
        if err < opts.tol:
            converged = True
            break

    def arr(x):
        return np.zeros_like(Gamma) if np.isscalar(x) else x

    return VertexResult(
        Gamma=Gamma,
        Gamma_H=arr(gh),
        Gamma_MT=arr(gmt),
        Gamma_AL1=arr(gal1),
        Gamma_AL2=arr(gal2),
        converged=converged,
        iterations=it,
    )
