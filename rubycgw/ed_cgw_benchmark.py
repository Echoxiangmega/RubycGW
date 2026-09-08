"""Helpers for same-torus ED versus GW/cGW response benchmarks.

The 18-site ED torus is exactly one index-three Ruby supercell with periodic
boundary conditions.  A supercell GW calculation with ``nk1=nk2=1`` therefore
uses the same one-particle torus (the primitive Gamma,+Q,-Q momenta are folded
into the 18 orbital basis).  This module contains small, testable helpers used
by :mod:`benchmark_ed18_cgw.py` to compare response functions on that common
finite geometry.

Two comparison levels are distinguished deliberately:

* ``GG``: the particle-hole bubble made from the converged SC-GW Green's
  function.  This can be evaluated at every bosonic Matsubara frequency and
  Fourier transformed to imaginary time.
* ``cGW``: the current production vertex solver is static (external bosonic
  frequency zero).  It is therefore compared to ED at ``iOmega=0`` only.

The finite-Q ED operator is complex,

    O_Q = (Qc - i Qs)/sqrt(2),

where Qc/Qs are the real orthonormal sector harmonics used by the supercell
code.  The projection routines below reconstruct the complex-Q six-channel
response from the real 12-channel Qc/Qs response without assuming that the two
real harmonics are exactly degenerate.
"""

from __future__ import annotations

import numpy as np

from .grids import MatsubaraGrid, frequency_shift_slices


def eigh_desc_hermitian(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hermitize one square matrix and return eigenpairs in descending order."""
    mat = np.asarray(matrix, dtype=complex)
    mat = 0.5 * (mat + mat.conj().T)
    vals, vecs = np.linalg.eigh(mat)
    order = np.argsort(vals.real)[::-1]
    return np.asarray(vals[order].real, dtype=float), np.asarray(vecs[:, order], dtype=complex)


def complex_q_projection(n_channels: int) -> np.ndarray:
    """Return C such that O_Q = [Qc,Qs] C in channel space.

    The real-harmonic ordering is ``[all Qc channels, all Qs channels]``.  For
    every primitive pseudospin channel,

        O_Q = (O_Qc - i O_Qs)/sqrt(2).

    Thus a real-harmonic response matrix M is converted to the complex-Q basis
    by ``C.conj().T @ M @ C``.
    """
    n = int(n_channels)
    if n < 1:
        raise ValueError("n_channels must be positive")
    eye = np.eye(n, dtype=complex)
    return np.vstack([eye, -1j * eye]) / np.sqrt(2.0)


def project_complex_q(matrix: np.ndarray, n_channels: int) -> np.ndarray:
    """Project one (...,2N,2N) Qc/Qs response to (...,N,N) complex-Q form."""
    mat = np.asarray(matrix, dtype=complex)
    n = int(n_channels)
    if mat.shape[-2:] != (2 * n, 2 * n):
        raise ValueError(
            f"expected final Qc/Qs matrix shape {(2*n, 2*n)}, got {mat.shape[-2:]}"
        )
    C = complex_q_projection(n)
    return np.einsum("ia,...ij,jb->...ab", C.conj(), mat, C, optimize=True)


def static_response_from_gammas(
    G: np.ndarray,
    left_vertices: np.ndarray,
    right_gammas: list[np.ndarray] | np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Return the zero-bosonic-frequency response ``-T/Nk Tr[K G Gamma G]``."""
    G = np.asarray(G, dtype=complex)
    K = np.asarray(left_vertices, dtype=complex)
    gammas = [np.asarray(x, dtype=complex) for x in right_gammas]
    if K.ndim != 3 or K.shape[-2:] != G.shape[-2:]:
        raise ValueError("left_vertices must have shape (N,norb,norb)")
    out = np.zeros((K.shape[0], len(gammas)), dtype=complex)
    pref = -(float(grid.T) / float(grid.nk))
    for b, gamma in enumerate(gammas):
        if gamma.shape != G.shape:
            raise ValueError("every driven Gamma must have the same shape as G")
        out[:, b] = pref * np.einsum(
            "aij,nxyjk,nxykl,nxyli->a",
            K,
            G,
            gamma,
            G,
            optimize=True,
        )
    return out


def bubble_iomega(
    G: np.ndarray,
    vertices: np.ndarray,
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Return GG susceptibility for every bosonic Matsubara index in ``grid``.

    For external supercell momentum zero,

        chi_ab(iOmega_m)
          = -(T/Nk) sum_{n,k} Tr[K_a G(k,iw_n+iOmega_m)
                                  K_b G(k,iw_n)].

    The finite fermion-frequency box is handled with the same truncation helper
    used by the production GW/cGW code.  At ``m=0`` this is exactly the bare
    static response computed by :func:`static_response_from_gammas`.
    """
    G = np.asarray(G, dtype=complex)
    K = np.asarray(vertices, dtype=complex)
    if K.ndim != 3 or K.shape[-2:] != G.shape[-2:]:
        raise ValueError("vertices must have shape (N,norb,norb)")
    out = np.zeros((grid.nb, K.shape[0], K.shape[0]), dtype=complex)
    pref = -(float(grid.T) / float(grid.nk))
    for im, m in enumerate(grid.m_values):
        src, dst = frequency_shift_slices(grid.nf, int(m))
        if src.stop == src.start:
            continue
        Gp = G[src]
        G0 = G[dst]
        out[im] = pref * np.einsum(
            "aij,nxyjk,bkl,nxyli->ab",
            K,
            Gp,
            K,
            G0,
            optimize=True,
        )
    return out


def bosonic_iomega_to_tau(
    chi_iomega: np.ndarray,
    grid: MatsubaraGrid,
    tau: np.ndarray,
) -> np.ndarray:
    """Truncated inverse bosonic Matsubara transform on arbitrary tau points."""
    chi = np.asarray(chi_iomega, dtype=complex)
    if chi.shape[0] != grid.nb:
        raise ValueError(f"first axis must have grid.nb={grid.nb} entries")
    tau = np.asarray(tau, dtype=float).reshape(-1)
    phase = np.exp(-1j * np.outer(tau, np.asarray(grid.Omega, dtype=float)))
    return float(grid.T) * np.einsum("tm,m...->t...", phase, chi, optimize=True)


def relative_frobenius_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    """Return ||approx-reference||_F / max(||reference||_F, tiny)."""
    ref = np.asarray(reference, dtype=complex)
    app = np.asarray(approximate, dtype=complex)
    if ref.shape != app.shape:
        raise ValueError("reference and approximate matrices must have the same shape")
    denom = max(float(np.linalg.norm(ref.ravel())), 1e-300)
    return float(np.linalg.norm((app - ref).ravel()) / denom)
