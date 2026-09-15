"""Reusable self-energy and finite-torus Green-function analysis helpers."""
from __future__ import annotations

import numpy as np


def realspace_to_k(GR: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    """Transform a finite-torus site-basis Green matrix to 6x6 k blocks."""
    arr = np.asarray(GR, dtype=complex)
    Lx, Ly = int(Lx), int(Ly)
    ncell = Lx * Ly
    nsite = 6 * ncell
    if arr.ndim != 3 or arr.shape[1:] != (nsite, nsite):
        raise ValueError("real-space Green function has incompatible shape")
    nf = int(arr.shape[0])
    cells = [(r1, r2) for r1 in range(Lx) for r2 in range(Ly)]
    out = np.zeros((nf, Lx, Ly, 6, 6), dtype=complex)
    for i in range(Lx):
        for j in range(Ly):
            block = np.zeros((nf, 6, 6), dtype=complex)
            for c, (r1, r2) in enumerate(cells):
                for d, (s1, s2) in enumerate(cells):
                    phase = np.exp(
                        -2j * np.pi * (
                            (i / float(Lx)) * (r1 - s1)
                            + (j / float(Ly)) * (r2 - s2)
                        )
                    )
                    block += phase * arr[:, 6*c:6*(c+1), 6*d:6*(d+1)]
            out[:, i, j] = block / float(ncell)
    return out


def k_to_realspace(Gk: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    """Transform 6x6 primitive k blocks to a finite-torus site-basis matrix."""
    arr = np.asarray(Gk, dtype=complex)
    Lx, Ly = int(Lx), int(Ly)
    if arr.ndim != 5 or arr.shape[1:3] != (Lx, Ly) or arr.shape[-2:] != (6, 6):
        raise ValueError("k-space Green function has incompatible shape")
    nf = int(arr.shape[0])
    ncell = Lx * Ly
    cells = [(r1, r2) for r1 in range(Lx) for r2 in range(Ly)]
    out = np.zeros((nf, 6*ncell, 6*ncell), dtype=complex)
    for c, (r1, r2) in enumerate(cells):
        for d, (s1, s2) in enumerate(cells):
            block = np.zeros((nf, 6, 6), dtype=complex)
            for i in range(Lx):
                for j in range(Ly):
                    phase = np.exp(
                        2j * np.pi * (
                            (i / float(Lx)) * (r1 - s1)
                            + (j / float(Ly)) * (r2 - s2)
                        )
                    )
                    block += phase * arr[:, i, j]
            out[:, 6*c:6*(c+1), 6*d:6*(d+1)] = block / float(ncell)
    return out


def dyson_kernel(Gk: np.ndarray, h0: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Return ``K = Sigma - mu I = iomega I - h0 - G^{-1}``."""
    G = np.asarray(Gk, dtype=complex)
    h = np.asarray(h0, dtype=complex)
    w = np.asarray(omega, dtype=float).reshape(-1)
    if G.shape != (len(w),) + h.shape:
        raise ValueError("G/h0/omega shape mismatch")
    eye = np.eye(6, dtype=complex)
    return (
        (1j * w[:, None, None, None, None]) * eye[None, None, None]
        - h[None, :, :, :, :]
        - np.linalg.inv(G)
    )


def decompose_kernel(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a k-dependent kernel into its k average and nonlocal remainder."""
    arr = np.asarray(K, dtype=complex)
    loc = np.mean(arr, axis=(1, 2))
    nonloc = arr - loc[:, None, None, :, :]
    return loc, nonloc


def green_from_kernel(K: np.ndarray, h0: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Reconstruct ``G`` from a Dyson kernel ``K = Sigma - mu I``."""
    arr = np.asarray(K, dtype=complex)
    h = np.asarray(h0, dtype=complex)
    w = np.asarray(omega, dtype=float).reshape(-1)
    eye = np.eye(6, dtype=complex)
    invg = (
        (1j * w[:, None, None, None, None]) * eye[None, None, None]
        - h[None, :, :, :, :]
        - arr
    )
    return np.linalg.inv(invg)


__all__ = [
    "realspace_to_k",
    "k_to_realspace",
    "dyson_kernel",
    "decompose_kernel",
    "green_from_kernel",
]
