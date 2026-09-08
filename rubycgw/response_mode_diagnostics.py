"""Utilities for fixed-mode susceptibility and vertex-pole diagnostics.

These helpers are deliberately conservative about what can be inferred from a
matrix-free cGW vertex solve.  For one driven source K and converged vertex
Gamma satisfying

    (I - L) Gamma = K,

we can evaluate the action L Gamma from the diagram-resolved vertex pieces and
form source-conditioned diagnostics without constructing the huge matrix
representation of ``I-L``.

In particular,

    sigma_source = ||(I-L) Gamma|| / ||Gamma||

is an *upper bound* on the global smallest singular value sigma_min(I-L),
because sigma_min is the minimum of that quotient over all vertex-space
vectors.  It is therefore useful as a physical-source pole proxy, but it is
not a replacement for a full singular-value calculation.

Likewise,

    lambda_eff = <Gamma, L Gamma> / <Gamma, Gamma>

is a Rayleigh quotient.  It can be interpreted as a kernel eigenvalue only
when the accompanying eigen-residual ||L Gamma-lambda_eff Gamma||/||Gamma|| is
small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VertexPoleDiagnostics:
    source_gain: float
    sigma_source_upper: float
    lambda_eff: complex
    lambda_eigen_residual: float
    equation_residual_relative: float
    correction_ratio: float
    source_alignment: float


def hermitian_leading_mode(
    matrix: np.ndarray,
    indices: np.ndarray | list[int] | tuple[int, ...] | None = None,
) -> tuple[float, np.ndarray]:
    """Return the leading eigenvalue/vector, optionally inside a subspace.

    The returned vector always has the full dimension of ``matrix``.  When
    ``indices`` is supplied the vector has support only on that subspace.
    """
    mat = np.asarray(matrix, dtype=complex)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")
    mat = 0.5 * (mat + mat.conj().T)
    n = mat.shape[0]

    if indices is None:
        vals, vecs = np.linalg.eigh(mat)
        j = int(np.argmax(vals.real))
        vec = np.asarray(vecs[:, j], dtype=complex)
        return float(vals[j].real), vec

    idx = np.asarray(indices, dtype=int).reshape(-1)
    if idx.size < 1:
        raise ValueError("indices must contain at least one channel")
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError("subspace index out of range")
    sub = mat[np.ix_(idx, idx)]
    vals, vecs = np.linalg.eigh(sub)
    j = int(np.argmax(vals.real))
    full = np.zeros(n, dtype=complex)
    full[idx] = vecs[:, j]
    return float(vals[j].real), full


def normalize_mode(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("mode vector must have nonzero finite norm")
    return vec / norm


def mode_response(matrix: np.ndarray, vector: np.ndarray) -> float:
    """Return Re[v^dagger matrix v] for a normalized or unnormalized mode."""
    mat = np.asarray(matrix, dtype=complex)
    vec = normalize_mode(vector)
    if mat.shape != (vec.size, vec.size):
        raise ValueError("matrix/vector size mismatch")
    return float(np.vdot(vec, mat @ vec).real)


def combine_vertex(vertices: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Linear combination sum_a coefficients[a] * vertices[a]."""
    verts = np.asarray(vertices, dtype=complex)
    coeff = normalize_mode(coefficients)
    if verts.ndim != 3 or verts.shape[0] != coeff.size:
        raise ValueError("vertices must have shape (N,norb,norb) matching coefficients")
    return np.einsum("a,aij->ij", coeff, verts, optimize=True)


def combine_complex_q_vertex(
    real_harmonic_vertices: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Combine Qc/Qs vertices into the complex primitive-Q source.

    If ``coefficients`` describes an internal-channel mode v, this returns

        K_Q(v) = sum_mu v_mu [K_Qc,mu - i K_Qs,mu] / sqrt(2),

    matching ``O_Q=(O_Qc-i O_Qs)/sqrt(2)`` used by the ED benchmark.
    """
    verts = np.asarray(real_harmonic_vertices, dtype=complex)
    coeff = normalize_mode(coefficients)
    n = coeff.size
    if verts.ndim != 3 or verts.shape[0] != 2 * n:
        raise ValueError("real_harmonic_vertices must have shape (2N,norb,norb)")
    real_coeff = np.concatenate([coeff, -1j * coeff]) / np.sqrt(2.0)
    return np.einsum("a,aij->ij", real_coeff, verts, optimize=True)


def scalar_static_response(
    G: np.ndarray,
    K: np.ndarray,
    Gamma: np.ndarray,
    temperature: float,
    nk: int,
) -> float:
    """Return -T/Nk Tr[K^dagger G Gamma G] for one complex source mode."""
    G = np.asarray(G, dtype=complex)
    K = np.asarray(K, dtype=complex)
    Gamma = np.asarray(Gamma, dtype=complex)
    if G.shape != Gamma.shape:
        raise ValueError("G and Gamma must have the same shape")
    if K.shape != G.shape[-2:]:
        raise ValueError("K shape does not match orbital dimensions")
    pref = -float(temperature) / float(nk)
    Kdag = K.conj().T
    value = pref * np.einsum(
        "ij,nxyjk,nxykl,nxyli->",
        Kdag,
        G,
        Gamma,
        G,
        optimize=True,
    )
    return float(complex(value).real)


def vertex_pole_diagnostics(
    Kfield: np.ndarray,
    Gamma: np.ndarray,
    kernel_gamma: np.ndarray,
) -> VertexPoleDiagnostics:
    """Source-conditioned diagnostics for ``(I-L)Gamma=K``.

    ``kernel_gamma`` must be the actual diagrammatic action ``L Gamma`` (the
    sum of H/F/MT/AL vertex corrections), not ``Gamma-K`` inferred by hand.
    This keeps the equation residual as an independent implementation check.
    """
    K = np.asarray(Kfield, dtype=complex)
    g = np.asarray(Gamma, dtype=complex)
    Lg = np.asarray(kernel_gamma, dtype=complex)
    if K.shape != g.shape or Lg.shape != g.shape:
        raise ValueError("Kfield, Gamma, and kernel_gamma must have identical shapes")

    nk = float(np.linalg.norm(K.ravel()))
    ng = float(np.linalg.norm(g.ravel()))
    if nk <= 0.0 or ng <= 0.0:
        raise ValueError("source and solved vertex must have nonzero norm")

    A_gamma = g - Lg
    nA = float(np.linalg.norm(A_gamma.ravel()))
    source_gain = ng / nk
    sigma_source = nA / ng

    denom = np.vdot(g.ravel(), g.ravel())
    lam = np.vdot(g.ravel(), Lg.ravel()) / denom
    eig_res = float(np.linalg.norm((Lg - lam * g).ravel()) / ng)
    eq_res = float(np.linalg.norm((A_gamma - K).ravel()) / nk)
    correction_ratio = float(np.linalg.norm(Lg.ravel()) / ng)
    alignment = float(abs(np.vdot(g.ravel(), K.ravel())) / (ng * nk))

    return VertexPoleDiagnostics(
        source_gain=float(source_gain),
        sigma_source_upper=float(sigma_source),
        lambda_eff=complex(lam),
        lambda_eigen_residual=float(eig_res),
        equation_residual_relative=float(eq_res),
        correction_ratio=float(correction_ratio),
        source_alignment=float(alignment),
    )
