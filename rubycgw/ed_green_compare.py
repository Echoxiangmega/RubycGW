"""Exact-ED local Green-function diagnostics for primitive-mesh calculations.

A dense primitive k mesh and a small exact torus do not share the same momentum
points in general.  The directly comparable one-particle quantity is therefore
the cell-local 6x6 Matsubara Green matrix

    G_loc(iw) = (1/Nk) sum_k G(k,iw),

versus the cell-averaged diagonal block of the exact torus Green matrix.  This
module provides that comparison and the finite-source ED construction used by
the GW-Gamma_P scan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import RubyParameters
from .small_cluster_exact import ExactSmallRubyThermal


@dataclass(frozen=True)
class ExactLocalGreenResult:
    mu: float
    J_ref: float
    G_local: np.ndarray
    kept_weight: float
    discarded_weight: float


def primitive_local_green(G: np.ndarray) -> np.ndarray:
    """Return the Brillouin-zone averaged 6x6 Matsubara Green matrix."""
    arr = np.asarray(G, dtype=complex)
    if arr.ndim != 5:
        raise ValueError("primitive G must have shape (nf,nk1,nk2,norb,norb)")
    return np.mean(arr, axis=(1, 2))


def cluster_local_green(G: np.ndarray, n_cells: int, norb: int = 6) -> np.ndarray:
    """Average same-cell diagonal blocks of a finite-torus Green matrix."""
    arr = np.asarray(G, dtype=complex)
    n_cells = int(n_cells)
    norb = int(norb)
    if arr.ndim != 3:
        raise ValueError("cluster G must have shape (nf,nsite,nsite)")
    nsite = n_cells * norb
    if arr.shape[1:] != (nsite, nsite):
        raise ValueError("cluster Green/site count mismatch")
    out = np.zeros((arr.shape[0], norb, norb), dtype=complex)
    for ic in range(n_cells):
        sl = slice(ic * norb, (ic + 1) * norb)
        out += arr[:, sl, sl]
    return out / float(n_cells)


def relative_green_error(G: np.ndarray, G_exact: np.ndarray, *, n_low: int | None = None) -> float:
    """Relative Frobenius error ||G-G_exact||_F / ||G_exact||_F.

    If ``n_low`` is supplied, retain the ``n_low`` Matsubara frequencies with
    smallest |omega|.  The caller supplies arrays in the same frequency order;
    for the standard symmetric RubycGW grid the central frequencies are the
    smallest-|omega| entries, so selection is performed by distance from the
    array center rather than by assuming a particular sign convention.
    """
    a = np.asarray(G, dtype=complex)
    b = np.asarray(G_exact, dtype=complex)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("Green arrays must have identical (nf,norb,norb) shape")
    if n_low is not None:
        nkeep = min(max(int(n_low), 1), a.shape[0])
        center = 0.5 * (a.shape[0] - 1)
        idx = np.argsort(np.abs(np.arange(a.shape[0]) - center))[:nkeep]
        a = a[idx]
        b = b[idx]
    den = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / den) if den > 0.0 else float("nan")


def _exact_onebody_expectation(exact: ExactSmallRubyThermal, K: np.ndarray, mu: float, T: float) -> complex:
    probs, _, _ = exact._normalized_probabilities(float(mu), float(T))
    value = 0.0j
    for N, (sec, p) in enumerate(zip(exact.sectors, probs)):
        if len(sec.energies) == 0 or not np.any(p > 0.0):
            continue
        U = np.asarray(sec.eigenvectors, dtype=complex)
        Opsi = exact._apply_onebody_batch(int(N), K, U)
        diag = np.sum(U.conj() * Opsi, axis=0)
        value += np.dot(p, diag)
    return complex(value)


def solve_exact_finite_source_local_green(
    *,
    L1: int,
    L2: int,
    params: RubyParameters,
    V: float,
    source_channel: str,
    h_ref: float,
    filling_per_cell: float,
    T: float,
    omega: np.ndarray,
    discard_weight_tol: float = 1e-12,
) -> ExactLocalGreenResult:
    """Solve the matching finite-source grand-canonical ED benchmark.

    ``h_ref`` multiplies the cell-normalized exact q=0 source operator.  For an
    ``Ncell`` torus this means a primitive per-cell field h_ref/sqrt(Ncell),
    matching the normalization used by the finite-source primitive scan when
    ``reference_ncell == L1*L2``.
    """
    base = RubyParameters(
        ti=float(params.ti),
        t1=float(params.t1),
        t2=float(params.t2),
        V=0.0,
    )
    exact = ExactSmallRubyThermal(int(L1), int(L2), base)
    K = np.asarray(exact.pseudospin_operator(str(source_channel), (0.0, 0.0)), dtype=complex)
    exact.h0 = np.asarray(exact.h0, dtype=complex) - float(h_ref) * K
    exact.h0 = 0.5 * (exact.h0 + exact.h0.conj().T)
    exact.diagonalize(float(V))
    target = float(filling_per_cell) * float(exact.n_cells)
    mu = exact.solve_mu(target, float(T))
    Gfull, selection = exact.green_iomega(
        1j * np.asarray(omega, dtype=float),
        float(mu),
        float(T),
        discard_weight_tol=float(discard_weight_tol),
    )
    J = _exact_onebody_expectation(exact, K, mu, T)
    return ExactLocalGreenResult(
        mu=float(mu),
        J_ref=float(np.real(J)),
        G_local=cluster_local_green(Gfull, exact.n_cells, norb=6),
        kept_weight=float(selection.kept_weight),
        discarded_weight=float(selection.discarded_weight),
    )


__all__ = [
    "ExactLocalGreenResult",
    "primitive_local_green",
    "cluster_local_green",
    "relative_green_error",
    "solve_exact_finite_source_local_green",
]
