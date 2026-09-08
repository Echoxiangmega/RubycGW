"""Finite-source checks for the static covariant-GW response.

The static cGW susceptibility is a functional derivative of the *same* SC-GW
fixed-point equations.  A direct numerical check is therefore obtained by
adding a small source

    H_source = - h K

re-solving the SC-GW equations at +h and -h, and comparing the central finite
difference of <K> with the cGW vertex result.

This module contains only normalization/sign-safe helpers.  The production
validation driver is ``validate_cgw_finite_source.py``.
"""

from __future__ import annotations

import numpy as np

from .grids import MatsubaraGrid


def add_bilinear_source(h0: np.ndarray, K: np.ndarray, h: float) -> np.ndarray:
    """Return ``h0 - h K`` without modifying ``h0``.

    ``K`` is an orbital-space Hermitian one-body operator.  The source
    convention matches the cGW response convention used throughout the project.
    """
    arr = np.asarray(h0, dtype=complex)
    vertex = np.asarray(K, dtype=complex)
    if vertex.shape != arr.shape[-2:]:
        raise ValueError(
            f"K shape {vertex.shape} must match h0 orbital shape {arr.shape[-2:]}"
        )
    out = np.array(arr, copy=True) - float(h) * vertex
    return 0.5 * (out + np.swapaxes(out.conj(), -1, -2))


def bilinear_expectation(
    G: np.ndarray,
    K: np.ndarray,
    grid: MatsubaraGrid,
) -> complex:
    """Return ``<K> = (T/Nk) sum_{n,k} Tr[K G(k,iw_n)]``.

    The public pseudospin vertices used by the validation driver are traceless,
    so the 1/iw high-frequency term cancels and this finite Matsubara sum is
    absolutely convergent.  The sign is consistent with ``H_source=-hK`` and
    ``chi=-int Tr[K G Gamma G]``.
    """
    arr = np.asarray(G, dtype=complex)
    vertex = np.asarray(K, dtype=complex)
    if arr.ndim != 5:
        raise ValueError("G must have shape (nf,nk1,nk2,norb,norb)")
    if vertex.shape != arr.shape[-2:]:
        raise ValueError("K shape does not match G orbital dimension")
    return complex(
        (float(grid.T) / float(grid.nk))
        * np.einsum("ab,nxyba->", vertex, arr, optimize=True)
    )


def central_finite_difference(
    m_plus: complex,
    m_minus: complex,
    h: float,
) -> complex:
    """Return ``[m(+h)-m(-h)]/(2h)``."""
    h = float(h)
    if h <= 0.0:
        raise ValueError("h must be positive")
    return (complex(m_plus) - complex(m_minus)) / (2.0 * h)


def quadratic_zero_source_extrapolation(
    h_values: np.ndarray,
    chi_values: np.ndarray,
) -> tuple[complex, complex]:
    """Fit ``chi(h)=chi0+a h^2`` and return ``(chi0,a)``.

    A symmetric central difference has O(h^2) truncation error.  With at least
    two source magnitudes, a linear least-squares fit versus h^2 gives a useful
    h->0 diagnostic.  Complex values are fitted component-wise automatically by
    NumPy's complex least-squares solver.
    """
    h = np.asarray(h_values, dtype=float).reshape(-1)
    chi = np.asarray(chi_values, dtype=complex).reshape(-1)
    if h.size != chi.size or h.size < 2:
        raise ValueError("need at least two matching h and chi values")
    if np.any(h <= 0.0):
        raise ValueError("all h values must be positive")
    A = np.column_stack([np.ones_like(h), h**2])
    coeff, *_ = np.linalg.lstsq(A, chi, rcond=None)
    return complex(coeff[0]), complex(coeff[1])


def relative_error(reference: complex, value: complex, floor: float = 1e-14) -> float:
    """Relative absolute error with a small denominator floor."""
    ref = complex(reference)
    val = complex(value)
    return float(abs(val - ref) / max(abs(ref), float(floor)))


__all__ = [
    "add_bilinear_source",
    "bilinear_expectation",
    "central_finite_difference",
    "quadratic_zero_source_extrapolation",
    "relative_error",
]
