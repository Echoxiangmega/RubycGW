"""Zero-temperature static susceptibility for the 18-site Ruby ED cluster.

This module complements the equal-time structure factor in :mod:`rubycgw.ed18`.
For normalized operators

    O_mu(q) = Ncell^(-1/2) sum_R exp(-i q.R) O_mu(R),

the regular zero-temperature thermodynamic/Kubo susceptibility is

    chi^reg_{mu,nu}(q)
      = (2/d_GS) sum_g sum_{m notin GS}
          <g|O_mu(q)^dag|m><m|O_nu(q)|g> / (E_m-E_0).

The factor two is the usual static thermodynamic response factor.  The full
excited spectrum is NOT diagonalized.  Instead correction vectors are solved
in the excited-state subspace,

    (H-E_0) |X_nu,g> = Q_ex O_nu(q)|g>.

Exact finite-size ground-state degeneracy needs special care.  If an operator
has nontrivial matrix elements inside the ground-state manifold, the T->0
static susceptibility contains a Curie/nonanalytic contribution rather than a
finite number.  We therefore return separately the positive-semidefinite
matrix

    C^GS_{mu,nu} = (1/d_GS) Tr[delta M_mu^dag delta M_nu],

where M_mu=P_GS O_mu(q) P_GS and delta M_mu removes the ground-manifold mean.
At low but finite temperature this contributes beta*C^GS.  Thus nonzero
C^GS means that the strict T=0 susceptibility is singular; ``regular_matrix``
is only its finite excited-state part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .ed18 import ED18_CHANNELS, ED18Solver, ED18Spectrum


@dataclass
class ED18StaticSusceptibility:
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
    mat = 0.5 * (np.asarray(mat, dtype=complex) + np.asarray(mat, dtype=complex).conj().T)
    vals, vecs = np.linalg.eigh(mat)
    order = np.argsort(vals.real)[::-1]
    return np.asarray(vals[order].real, dtype=float), np.asarray(vecs[:, order], dtype=complex)


def _cg(A, b: np.ndarray, *, tol: float, maxiter: int) -> tuple[np.ndarray, int]:
    """SciPy-version-compatible conjugate-gradient call."""
    try:
        return cg(A, b, rtol=float(tol), atol=0.0, maxiter=int(maxiter))
    except TypeError:  # scipy 1.10-1.13 use ``tol`` instead of ``rtol``
        return cg(A, b, tol=float(tol), atol=0.0, maxiter=int(maxiter))


def zero_temperature_susceptibility(
    solver: ED18Solver,
    spectrum: ED18Spectrum,
    q: np.ndarray,
    channels: tuple[str, ...] = ED18_CHANNELS,
    *,
    solve_tol: float = 1e-10,
    maxiter: int = 20000,
    singular_tol: float = 1e-10,
    projector_lift: float | None = None,
) -> ED18StaticSusceptibility:
    """Compute the exact fixed-N T=0 static susceptibility at one momentum.

    The ground manifold is averaged with equal weight, matching the convention
    already used by the ED equal-time structure factor.  The returned
    ``regular_matrix`` excludes zero-energy transitions within an exactly
    degenerate ground manifold.  Such transitions are *not* silently dropped:
    their covariance is returned as ``ground_singular_matrix`` and
    ``is_singular`` is set when it is nonzero.

    Parameters
    ----------
    solve_tol, maxiter
        Correction-vector CG controls.  ``solver_residuals[g,mu]`` gives the
        explicit relative residual for each right-hand side.
    projector_lift
        Positive energy used to lift the null ground subspace while solving.
        It cannot affect the projected exact solution.  By default a scale of
        max(1, gap above the ground manifold) is used.
    """
    q = np.asarray(q, dtype=float).reshape(2)
    channels = tuple(str(x) for x in channels)
    ng = int(spectrum.ground_multiplicity)
    if ng < 1:
        raise ValueError("spectrum contains no ground state")

    gs = np.asarray(spectrum.eigenvectors[:, :ng], dtype=complex)
    # Re-orthonormalize the numerically returned degenerate subspace.  This is
    # harmless for a nondegenerate state and stabilizes projector operations.
    gs, _ = np.linalg.qr(gs)
    E0 = float(spectrum.energies[0])
    H = solver.hamiltonian(float(spectrum.V)).astype(complex)
    ops = [solver.operator(ch, q) for ch in channels]
    nc = len(ops)

    # Ground-manifold matrix elements.  Their connected covariance is the
    # coefficient of the beta-divergent contribution as T -> 0.
    Mgs = np.empty((nc, ng, ng), dtype=complex)
    for a, op in enumerate(ops):
        Mgs[a] = gs.conj().T @ (op @ gs)
    means = np.trace(Mgs, axis1=1, axis2=2) / float(ng)
    ident_gs = np.eye(ng, dtype=complex)
    centered = Mgs - means[:, None, None] * ident_gs[None, :, :]
    singular = np.empty((nc, nc), dtype=complex)
    for a in range(nc):
        for b in range(nc):
            singular[a, b] = np.vdot(centered[a], centered[b]) / float(ng)
    singular = 0.5 * (singular + singular.conj().T)
    sing_evals, sing_evecs = _sorted_eigh(singular)
    sing_scale = max(1.0, float(np.max(np.abs(singular))))
    is_singular = bool(sing_evals[0] > float(singular_tol) * sing_scale)

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
        return H @ x - E0 * x + lift * project_ground(x)

    A = LinearOperator((solver.dimension, solver.dimension), matvec=amatvec, dtype=np.complex128)

    chi = np.zeros((nc, nc), dtype=complex)
    residuals = np.zeros((ng, nc), dtype=float)
    infos = np.zeros((ng, nc), dtype=int)

    for g in range(ng):
        psi = gs[:, g]
        # Columns are Q_ex O_mu(q)|g>.
        B = np.column_stack([op @ psi for op in ops]).astype(complex, copy=False)
        B -= gs @ (gs.conj().T @ B)
        X = np.zeros_like(B)

        for b in range(nc):
            rhs = B[:, b]
            nrhs = float(np.linalg.norm(rhs))
            if nrhs <= 1e-14:
                residuals[g, b] = 0.0
                infos[g, b] = 0
                continue
            x, info = _cg(A, rhs, tol=solve_tol, maxiter=maxiter)
            # Remove any roundoff ground component before contractions.
            x = np.asarray(x, dtype=complex) - project_ground(np.asarray(x, dtype=complex))
            relres = float(np.linalg.norm(amatvec(x) - rhs) / nrhs)
            X[:, b] = x
            residuals[g, b] = relres
            infos[g, b] = int(info)

        chi += (2.0 / float(ng)) * (B.conj().T @ X)

    chi = 0.5 * (chi + chi.conj().T)
    reg_evals, reg_evecs = _sorted_eigh(chi)

    return ED18StaticSusceptibility(
        q=q,
        channels=channels,
        regular_matrix=np.asarray(chi, dtype=complex),
        regular_eigenvalues=reg_evals,
        regular_eigenvectors=reg_evecs,
        ground_singular_matrix=np.asarray(singular, dtype=complex),
        ground_singular_eigenvalues=sing_evals,
        ground_singular_eigenvectors=sing_evecs,
        is_singular=is_singular,
        solver_residuals=residuals,
        solver_info=infos,
    )
