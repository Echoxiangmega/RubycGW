"""Zero-temperature Kubo susceptibility for :mod:`rubycgw.ed_cluster`.

The implementation mirrors ``ed18_chi.py`` but works with the matrix-free
rectangular-cluster Hamiltonian.  Exact ground-manifold zero modes are returned
separately from the finite excited-state correction-vector contribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .ed_cluster import (
    ED_CLUSTER_CHANNELS,
    EDClusterSpectrum,
    RubyEDClusterSolver,
)


@dataclass
class EDClusterStaticSusceptibility:
    q: np.ndarray
    channels: tuple[str, ...]
    regular_matrix: np.ndarray
    regular_eigenvalues: np.ndarray
    regular_eigenvectors: np.ndarray
    ground_singular_matrix: np.ndarray
    ground_singular_eigenvalues: np.ndarray
    ground_singular_eigenvectors: np.ndarray
    is_singular: bool
    solver_residuals: np.ndarray
    solver_info: np.ndarray


def _sorted_eigh(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(mat, dtype=complex)
    a = 0.5 * (a + a.conj().T)
    vals, vecs = np.linalg.eigh(a)
    order = np.argsort(vals.real)[::-1]
    return np.asarray(vals[order].real, dtype=float), np.asarray(vecs[:, order], dtype=complex)


def _cg(A, b: np.ndarray, *, tol: float, maxiter: int) -> tuple[np.ndarray, int]:
    try:
        return cg(A, b, rtol=float(tol), atol=0.0, maxiter=int(maxiter))
    except TypeError:
        return cg(A, b, tol=float(tol), atol=0.0, maxiter=int(maxiter))


def zero_temperature_cluster_susceptibility(
    solver: RubyEDClusterSolver,
    spectrum: EDClusterSpectrum,
    q: np.ndarray,
    channels: tuple[str, ...] = ED_CLUSTER_CHANNELS,
    *,
    solve_tol: float = 1e-9,
    maxiter: int = 20000,
    singular_tol: float = 1e-10,
    projector_lift: float | None = None,
) -> EDClusterStaticSusceptibility:
    """Exact fixed-N T=0 static susceptibility at one commensurate momentum.

    ``regular_matrix`` is

        (2/dGS) sum_g <B_mu,g | (H-E0)^(-1)_Q | B_nu,g>,

    with ``B_mu,g = Q_ex O_mu(q)|g>``.  Matrix inversion is performed by CG
    against the matrix-free Hamiltonian.  Any covariance entirely inside an
    exactly degenerate ground manifold is reported separately as the
    coefficient of the T->0 Curie/singular contribution.
    """
    q = np.asarray(q, dtype=float).reshape(2)
    channels = tuple(str(x) for x in channels)
    ng = int(spectrum.ground_multiplicity)
    if ng < 1:
        raise ValueError("spectrum contains no ground state")
    gs = np.asarray(spectrum.eigenvectors[:, :ng], dtype=complex)
    gs, _ = np.linalg.qr(gs)
    E0 = float(spectrum.energies[0])
    nc = len(channels)

    # Ground-manifold projected operator matrices.
    Mgs = np.empty((nc, ng, ng), dtype=complex)
    for a, ch in enumerate(channels):
        Ogs = np.column_stack([solver.apply_operator(ch, q, gs[:, g]) for g in range(ng)])
        Mgs[a] = gs.conj().T @ Ogs
    means = np.trace(Mgs, axis1=1, axis2=2) / float(ng)
    centered = Mgs - means[:, None, None] * np.eye(ng, dtype=complex)[None, :, :]
    singular = np.empty((nc, nc), dtype=complex)
    for a in range(nc):
        for b in range(nc):
            singular[a, b] = np.vdot(centered[a], centered[b]) / float(ng)
    singular = 0.5 * (singular + singular.conj().T)
    sing_vals, sing_vecs = _sorted_eigh(singular)
    scale = max(1.0, float(np.max(np.abs(singular))))
    is_singular = bool(sing_vals[0] > float(singular_tol) * scale)

    gap = float(spectrum.gap_above_manifold)
    if projector_lift is None:
        lift = max(1.0, gap if np.isfinite(gap) and gap > 0.0 else 0.0)
    else:
        lift = float(projector_lift)
        if lift <= 0.0:
            raise ValueError("projector_lift must be positive")

    def project_ground(x: np.ndarray) -> np.ndarray:
        return gs @ (gs.conj().T @ x)

    def amatvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex)
        return solver.apply_hamiltonian(float(spectrum.V), x) - E0 * x + lift * project_ground(x)

    A = LinearOperator((solver.dimension, solver.dimension), matvec=amatvec, dtype=np.complex128)
    chi = np.zeros((nc, nc), dtype=complex)
    residuals = np.zeros((ng, nc), dtype=float)
    infos = np.zeros((ng, nc), dtype=int)

    for g in range(ng):
        psi = gs[:, g]
        B = np.column_stack([solver.apply_operator(ch, q, psi) for ch in channels])
        B = np.asarray(B, dtype=complex)
        B -= gs @ (gs.conj().T @ B)
        X = np.zeros_like(B)
        for b in range(nc):
            rhs = B[:, b]
            nrhs = float(np.linalg.norm(rhs))
            if nrhs <= 1e-14:
                continue
            x, info = _cg(A, rhs, tol=solve_tol, maxiter=maxiter)
            x = np.asarray(x, dtype=complex)
            x -= project_ground(x)
            residuals[g, b] = float(np.linalg.norm(amatvec(x) - rhs) / nrhs)
            infos[g, b] = int(info)
            X[:, b] = x
        chi += (2.0 / float(ng)) * (B.conj().T @ X)

    chi = 0.5 * (chi + chi.conj().T)
    reg_vals, reg_vecs = _sorted_eigh(chi)
    return EDClusterStaticSusceptibility(
        q=q,
        channels=channels,
        regular_matrix=chi,
        regular_eigenvalues=reg_vals,
        regular_eigenvectors=reg_vecs,
        ground_singular_matrix=singular,
        ground_singular_eigenvalues=sing_vals,
        ground_singular_eigenvectors=sing_vecs,
        is_singular=is_singular,
        solver_residuals=residuals,
        solver_info=infos,
    )
