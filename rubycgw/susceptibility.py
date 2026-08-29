"""Bare, dressed-bubble, and covariant eta susceptibilities."""

from __future__ import annotations
import numpy as np

from .grids import MatsubaraGrid, shift_fermion_field


def chi_eta(G: np.ndarray, K_left: np.ndarray, grid: MatsubaraGrid,
            q1: int = 0, q2: int = 0, m: int = 0,
            Gamma: np.ndarray | None = None) -> complex:
    """Compute -int_k Tr[K_left G(k+q) Gamma(k,q) G(k)].

    If Gamma is None, the bare vertex K_left is used also on the right, giving
    the dressed GG bubble for the supplied G. Supplying G0 gives the bare
    G0G0 bubble.
    """
    Gq = shift_fermion_field(G, q1, q2, m)
    if Gamma is None:
        Gamma = np.broadcast_to(K_left, G.shape)
    val = np.einsum(
        "ab,nxybc,nxycd,nxyda->", K_left, Gq, Gamma, G, optimize=True
    )
    return -(grid.T / grid.nk) * val


def channel_summary(G: np.ndarray, K_plus: np.ndarray, K_minus: np.ndarray,
                    grid: MatsubaraGrid):
    chi_plus = chi_eta(G, K_plus, grid)
    chi_minus = chi_eta(G, K_minus, grid)
    return {
        "chi_plus_opposite": chi_plus,
        "chi_minus_same": chi_minus,
        "delta_same_minus_opposite": chi_minus - chi_plus,
    }
