"""Strict bare second-order-exchange (SOX) response diagnostic.

This module is intentionally *not* a production GW+SOX solver. It evaluates the
O(V^2) exchange skeleton and its source derivative on a noninteracting finite
cluster (``nk=1``), which is the clean object needed by the 12-site weak-coupling
ledger.

For the spinless density-density Hamiltonian used in RubycGW, with an interaction
matrix ``v`` (unit bond strengths when extracting the coefficient of V^2), the
second-order exchange self-energy in the repository Matsubara convention is

    Sigma_SOX,ij(tau) = sum_{k,l} v[i,l] v[k,j]
                         G[i,k](tau) G[k,l](-tau) G[l,j](tau).

The plus sign is the exchange sign in the spin-orbital second-Born/GF2 formula;
the second-order direct term has the opposite contraction/sign structure and is
already contained in the O(V^2) expansion of GW. Differentiating the expression
with respect to G in the source direction X=G K G gives three SOX vertex terms.

The free G(tau) and its directional derivative are evaluated from the exact
one-body spectrum, so the SOX coefficient is not contaminated by a fermionic
Matsubara-box transform. Only the final observable contraction uses the same
represented fermionic box as the cGW ledger.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .grids import MatsubaraGrid
from .supercell_cgw import (
    _dynamic_corrections_direct,
    _x_field,
    susceptibility_matrix_q0,
)
from .supercell_gw import compute_polarization_matrix


def solve_free_mu(h0: np.ndarray, target_particles: float, T: float) -> float:
    """Chemical potential of a finite noninteracting one-body Hamiltonian."""
    h = np.asarray(h0, dtype=complex)
    e = np.linalg.eigvalsh(0.5 * (h + h.conj().T))
    target = float(target_particles)
    if not 0.0 <= target <= len(e):
        raise ValueError("target_particles out of range")
    if T <= 0.0:
        raise ValueError("T must be positive")

    def number(mu):
        x = np.clip((e - float(mu)) / float(T), -700.0, 700.0)
        return float(np.sum(1.0 / (np.exp(x) + 1.0)))

    lo = float(np.min(e) - 10.0 * max(1.0, T))
    hi = float(np.max(e) + 10.0 * max(1.0, T))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if number(mid) > target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-14:
            break
    return float(0.5 * (lo + hi))


def free_green_iomega(h0: np.ndarray, mu: float, grid: MatsubaraGrid) -> np.ndarray:
    """Exact free Green function in the one-point-cluster array convention."""
    h = np.asarray(h0, dtype=complex)
    norb = h.shape[-1]
    eye = np.eye(norb, dtype=complex)
    G = np.stack([
        np.linalg.inv((1j * float(w) + float(mu)) * eye - h)
        for w in np.asarray(grid.omega)
    ])
    return G[:, None, None, :, :]


def _divided_difference(values, derivatives, energies, tol=1e-11):
    """Hermitian matrix-function divided difference in an eigenbasis."""
    val = np.asarray(values, dtype=complex)
    der = np.asarray(derivatives, dtype=complex)
    e = np.asarray(energies, dtype=float)
    de = e[:, None] - e[None, :]
    numer = val[:, None] - val[None, :]
    out = np.empty_like(de, dtype=complex)
    mask = np.abs(de) > float(tol)
    out[mask] = numer[mask] / de[mask]
    dlim = 0.5 * (der[:, None] + der[None, :])
    out[~mask] = dlim[~mask]
    return out


def free_green_tau_direction(
    h0: np.ndarray,
    mu: float,
    K: np.ndarray,
    tau: float,
    T: float,
):
    """Return G(tau), G(-tau), and D G[K] for 0<tau<beta.

    The direction corresponds to ``h0 -> h0 + eps K``. In Matsubara frequency
    this is exactly ``D G[K] = G K G``, matching the cGW ``X`` convention.
    """
    if not 0.0 < float(tau) < 1.0 / float(T):
        raise ValueError("tau must lie strictly inside (0,beta)")
    h = np.asarray(h0, dtype=complex)
    K = np.asarray(K, dtype=complex)
    h = 0.5 * (h + h.conj().T)
    e, U = np.linalg.eigh(h)
    xi = e - float(mu)
    beta = 1.0 / float(T)
    x = np.clip(beta * xi, -700.0, 700.0)
    f = 1.0 / (np.exp(x) + 1.0)
    t = float(tau)

    gp = -np.exp(-xi * t) * (1.0 - f)
    gm = np.exp(xi * t) * f
    dgp = np.exp(-xi * t) * (1.0 - f) * (t - beta * f)
    dgm = np.exp(xi * t) * f * (t - beta * (1.0 - f))

    Ke = U.conj().T @ K @ U
    Xp_e = _divided_difference(gp, dgp, e) * Ke
    Xm_e = _divided_difference(gm, dgm, e) * Ke
    Gp = (U * gp[None, :]) @ U.conj().T
    Gm = (U * gm[None, :]) @ U.conj().T
    Xp = U @ Xp_e @ U.conj().T
    Xm = U @ Xm_e @ U.conj().T
    return Gp, Gm, Xp, Xm


def sox_self_energy_tau(Gp: np.ndarray, Gm: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bare spinless SOX self-energy at one imaginary-time separation."""
    return np.einsum(
        "il,kj,ik,kl,lj->ij", v, v, Gp, Gm, Gp, optimize=True
    )


def sox_vertex_tau(
    Gp: np.ndarray,
    Gm: np.ndarray,
    Xp: np.ndarray,
    Xm: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Directional derivative D Sigma_SOX[G][X] at one tau."""
    a = np.einsum("il,kj,ik,kl,lj->ij", v, v, Xp, Gm, Gp, optimize=True)
    b = np.einsum("il,kj,ik,kl,lj->ij", v, v, Gp, Xm, Gp, optimize=True)
    c = np.einsum("il,kj,ik,kl,lj->ij", v, v, Gp, Gm, Xp, optimize=True)
    return a + b + c


def sox_vertex_iomega_free(
    h0: np.ndarray,
    mu: float,
    v_unit: np.ndarray,
    K: np.ndarray,
    grid: MatsubaraGrid,
    *,
    n_quad: int = 128,
) -> np.ndarray:
    """Strict O(V^2) SOX vertex coefficient on the free background.

    ``v_unit`` is the interaction matrix with the scalar coupling V removed.
    The returned field is therefore the coefficient multiplying V^2.
    """
    if grid.nk != 1:
        raise ValueError("SOX weak-coupling diagnostic currently requires nk=1")
    if int(n_quad) < 16:
        raise ValueError("n_quad must be at least 16")
    h = np.asarray(h0, dtype=complex)
    v = np.asarray(v_unit, dtype=complex)
    K = np.asarray(K, dtype=complex)
    if h.shape != v.shape or h.shape != K.shape:
        raise ValueError("h0, v_unit and K must have the same square shape")

    x, w = leggauss(int(n_quad))
    beta = 1.0 / float(grid.T)
    taus = 0.5 * beta * (x + 1.0)
    weights = 0.5 * beta * w
    gtau = np.empty((len(taus), h.shape[0], h.shape[1]), dtype=complex)
    for it, tau in enumerate(taus):
        Gp, Gm, Xp, Xm = free_green_tau_direction(h, mu, K, float(tau), grid.T)
        gtau[it] = sox_vertex_tau(Gp, Gm, Xp, Xm, v)

    phase = np.exp(1j * np.outer(np.asarray(grid.omega), taus))
    gamma = np.einsum("nt,t,tij->nij", phase, weights, gtau, optimize=True)
    return gamma[:, None, None, :, :]


def sox_response_coefficient_free(
    h0: np.ndarray,
    mu: float,
    v_unit: np.ndarray,
    K: np.ndarray,
    grid: MatsubaraGrid,
    *,
    n_quad: int = 128,
):
    """Return the static susceptibility coefficient generated by bare SOX."""
    G = free_green_iomega(h0, mu, grid)
    gamma = sox_vertex_iomega_free(h0, mu, v_unit, K, grid, n_quad=n_quad)
    chi = susceptibility_matrix_q0(G, np.asarray([K]), [gamma], grid)[0, 0]
    return complex(chi), gamma


def gw_direct_order2_response_coefficient_free(
    h0: np.ndarray,
    mu: float,
    v_unit: np.ndarray,
    K: np.ndarray,
    grid: MatsubaraGrid,
):
    """Strict GW-direct O(V^2) MT/AL coefficient on the free background.

    This is a normalization/sign cross-check for the SOX diagnostic. Expanding
    W-V to second order gives ``Wc2 = v P0 v``. Differentiating ``-G Wc2``
    yields the MT term, while differentiating P0 yields AL1+AL2.
    """
    if grid.nk != 1:
        raise ValueError("second-order direct diagnostic currently requires nk=1")
    h = np.asarray(h0, dtype=complex)
    v = np.asarray(v_unit, dtype=complex)
    K = np.asarray(K, dtype=complex)
    G = free_green_iomega(h, mu, grid)
    P = compute_polarization_matrix(G, grid, backend="direct")
    vb = v[None, None, None, :, :]
    Wc2 = np.matmul(np.matmul(vb, P), vb)
    Wbare = np.broadcast_to(v, (grid.nb, 1, 1, v.shape[0], v.shape[1]))
    Kfield = np.broadcast_to(K, G.shape).copy()
    X = _x_field(G, Kfield)
    gmt, gal1, gal2 = _dynamic_corrections_direct(
        G, Wbare, Wc2, X, grid, include_mt=True, include_al=True
    )

    def measure(field):
        return susceptibility_matrix_q0(G, np.asarray([K]), [field], grid)[0, 0]

    mt = complex(measure(gmt))
    al1 = complex(measure(gal1))
    al2 = complex(measure(gal2))
    return dict(mt=mt, al1=al1, al2=al2, al=al1 + al2, total=mt + al1 + al2)


__all__ = [
    "solve_free_mu",
    "free_green_iomega",
    "free_green_tau_direction",
    "sox_self_energy_tau",
    "sox_vertex_tau",
    "sox_vertex_iomega_free",
    "sox_response_coefficient_free",
    "gw_direct_order2_response_coefficient_free",
]
