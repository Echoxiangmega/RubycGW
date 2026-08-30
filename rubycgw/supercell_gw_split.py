"""Static-Fock plus dynamic-screening GW self-energy for the 18-site supercell.

For an instantaneous density interaction, the screened interaction satisfies

    W(q,iOmega) -> V(q)

at large bosonic Matsubara frequency.  Summing the full ``-G W`` convolution
inside a finite bosonic box therefore gives the non-decaying bare-V piece an
artificial cutoff dependence.  This module rewrites the same infinite-frequency
GW self-energy as

    Sigma_GW = Sigma_F + Sigma_c,
    Sigma_F = - <G V>_{equal time},
    Sigma_c = - T sum_{q,m} G(k+q,iw+iOm) [W(q,iOm)-V(q)].

Only the retarded part ``W-V`` is Matsubara truncated.  The static Fock term is
computed from the equal-time one-body density matrix with the same static-tail
subtraction used elsewhere in the supercell solver.
"""

from __future__ import annotations

import numpy as np

from .grids import MatsubaraGrid
from .gw import _check_backend
from .supercell_gw import (
    _fermi,
    _reverse_fft_spectrum,
    compute_sigma_gw_matrix,
)


def one_body_density_matrix_tail(
    G: np.ndarray,
    grid: MatsubaraGrid,
    h0: np.ndarray,
    mu: float,
    sigma_h: np.ndarray,
) -> np.ndarray:
    """Return rho_ab(k)=<c^dagger_{k b} c_{k a}> with tail subtraction.

    The reference Green function uses the static Hermitian Hamiltonian
    ``h0 + sigma_h``.  Its equal-time density matrix is evaluated analytically,
    while only ``G-G_ref`` is summed over the finite fermionic Matsubara box.
    """
    G = np.asarray(G, dtype=complex)
    h0 = np.asarray(h0, dtype=complex)
    sigma_h = np.asarray(sigma_h, dtype=complex)
    norb = int(G.shape[-1])
    if G.shape != (grid.nf, grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected G shape")
    if h0.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected h0 shape")
    if sigma_h.shape != (norb, norb):
        raise ValueError("unexpected sigma_h shape")

    href = h0 + sigma_h[None, None, :, :]
    href = 0.5 * (href + np.swapaxes(href.conj(), -1, -2))
    evals, evecs = np.linalg.eigh(href)

    occ = _fermi(evals - float(mu), grid.T)
    rho_ref = np.einsum(
        "xyaj,xyj,xybj->xyab",
        evecs,
        occ,
        evecs.conj(),
        optimize=True,
    )

    denom = (
        1j * grid.omega[:, None, None, None]
        + float(mu)
        - evals[None, :, :, :]
    )
    gref = np.einsum(
        "xyaj,nxyj,xybj->nxyab",
        evecs,
        1.0 / denom,
        evecs.conj(),
        optimize=True,
    )
    rho = rho_ref + grid.T * np.sum(G - gref, axis=0)
    # The exact equal-time density matrix is Hermitian.  Symmetrizing removes
    # only finite-box / roundoff noise and makes the static Fock term Hermitian.
    return 0.5 * (rho + np.swapaxes(rho.conj(), -1, -2))


def compute_static_fock_matrix_direct(
    rho: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Static Fock term in the supercell Bloch basis by direct q summation."""
    rho = np.asarray(rho, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(rho.shape[-1])
    if rho.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected rho shape")
    if Vq.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected Vq shape")

    sigma = np.zeros_like(rho)
    pref = 1.0 / grid.nk
    for iq1 in range(grid.nk1):
        for iq2 in range(grid.nk2):
            rho_q = np.roll(
                rho,
                shift=(-int(iq1), -int(iq2)),
                axis=(0, 1),
            )
            sigma -= pref * rho_q * Vq[iq1, iq2].T[None, None, :, :]
    return sigma


def compute_static_fock_matrix_fft(
    rho: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Static Fock term using the same momentum-convolution convention as GW."""
    rho = np.asarray(rho, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(rho.shape[-1])
    if rho.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected rho shape")
    if Vq.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected Vq shape")

    Ahat = np.fft.fftn(rho, axes=(0, 1))
    B = np.swapaxes(Vq, -1, -2)
    Bhat_minus = _reverse_fft_spectrum(B, axes=(0, 1))
    conv = np.fft.ifftn(Ahat * Bhat_minus, axes=(0, 1))
    return -(1.0 / grid.nk) * conv


def compute_static_fock_matrix(
    rho: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    backend: str = "fft",
) -> np.ndarray:
    backend = _check_backend(backend)
    if backend == "fft":
        return compute_static_fock_matrix_fft(rho, Vq, grid)
    return compute_static_fock_matrix_direct(rho, Vq, grid)


def compute_sigma_gw_split_components(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    h0: np.ndarray,
    mu: float,
    sigma_h: np.ndarray,
    backend: str = "fft",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(Sigma_total, Sigma_F, Sigma_c, rho)``.

    ``Sigma_F`` has shape ``(nk1,nk2,norb,norb)`` and is frequency independent;
    ``Sigma_c`` and ``Sigma_total`` have the same shape as ``G``.
    """
    backend = _check_backend(backend)
    rho = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    sigma_f = compute_static_fock_matrix(rho, Vq, grid, backend=backend)
    Wc = np.asarray(W, dtype=complex) - np.asarray(Vq, dtype=complex)[None, :, :, :, :]
    sigma_c = compute_sigma_gw_matrix(G, Wc, grid, backend=backend)
    sigma_total = sigma_c + sigma_f[None, :, :, :, :]
    return sigma_total, sigma_f, sigma_c, rho


def compute_sigma_gw_split_matrix(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    h0: np.ndarray,
    mu: float,
    sigma_h: np.ndarray,
    backend: str = "fft",
) -> np.ndarray:
    """GW self-energy with static bare-V Fock and dynamic ``W-V`` convolution."""
    total, _, _, _ = compute_sigma_gw_split_components(
        G,
        W,
        Vq,
        grid,
        h0,
        mu,
        sigma_h,
        backend=backend,
    )
    return total
