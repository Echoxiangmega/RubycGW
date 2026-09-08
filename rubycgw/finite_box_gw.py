"""Finite-Matsubara-box SC-GW map matching the static cGW discretization.

This module is diagnostic only.  Production SC-GW uses analytic tail subtraction
for the Hartree density and the equal-time density matrix entering the static
bare-V Fock term.  The static cGW kernel, however, differentiates the represented
finite Matsubara box directly through X=G Gamma G.

For an exact end-to-end implementation check we therefore also provide a
finite-box SC-GW fixed-point map with

    n_a = 1/2 + (T/Nk) sum_{k,n} G_aa(k,iw_n),
    rho(k) = 1/2 I + T sum_n G(k,iw_n),

while retaining the same split self-energy

    Sigma_GW = Sigma_F[rho] - G * (W-V).

At fixed chemical potential, the finite-source derivative of this diagnostic
map is exactly the discrete functional derivative implemented by
``supercell_cgw.py`` up to solver and finite-difference error.  This is not a
replacement for production tail-corrected SC-GW.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from .grids import MatsubaraGrid
from .gw import GWOptions, GWResult, _check_backend, _check_mixing_method, _mixed_self_energies, _residual_error
from .supercell_gw import (
    _compatible_initial,
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    compute_sigma_gw_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
    screening_soft_modes_matrix,
)
from .supercell_gw_split import compute_static_fock_matrix


def density_from_G_finite_box(G: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    """Orbital density from the represented fermionic Matsubara box only."""
    diag = np.diagonal(np.asarray(G), axis1=-2, axis2=-1)
    return 0.5 + (grid.T / grid.nk) * np.sum(diag, axis=(0, 1, 2)).real


def one_body_density_matrix_finite_box(G: np.ndarray, grid: MatsubaraGrid) -> np.ndarray:
    """rho_ab(k)=<c^dagger_{k b} c_{k a}> in the represented finite box."""
    arr = np.asarray(G, dtype=complex)
    norb = int(arr.shape[-1])
    eye = np.eye(norb, dtype=complex)
    rho = 0.5 * eye[None, None, :, :] + grid.T * np.sum(arr, axis=0)
    return 0.5 * (rho + np.swapaxes(rho.conj(), -1, -2))


def sigma_gw_finite_box(
    G: np.ndarray,
    W: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    backend: str = "fft",
) -> np.ndarray:
    """Static bare-V Fock plus finite-box retarded W-V convolution."""
    backend = _check_backend(backend)
    rho = one_body_density_matrix_finite_box(G, grid)
    sigma_f = compute_static_fock_matrix(rho, Vq, grid, backend=backend)
    Wc = np.asarray(W, dtype=complex) - np.asarray(Vq, dtype=complex)[None, ...]
    sigma_c = compute_sigma_gw_matrix(G, Wc, grid, backend=backend)
    return sigma_c + sigma_f[None, ...]


def solve_matrix_gw_finite_box(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    initial: GWResult | None = None,
) -> GWResult:
    """Solve the diagnostic finite-box SC-GW map at fixed chemical potential.

    The routine intentionally rejects ``target_filling``.  The present static
    cGW equation is the direct derivative at fixed mu; allowing a source-driven
    chemical-potential counterterm here would test a different response.
    """
    if opts.target_filling is not None:
        raise ValueError("finite-box cGW validation solver supports fixed mu only")
    backend = _check_backend(opts.momentum_backend)
    method = _check_mixing_method(opts.mixing_method)

    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(h0.shape[-1])
    if h0.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected h0 shape")
    if Vq.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected Vq shape")

    if _compatible_initial(initial, grid, norb):
        sigma_h = np.array(initial.Sigma_H, copy=True)
        sigma_gw = np.array(initial.Sigma_GW, copy=True)
        mu = float(initial.mu)
    else:
        sigma_h = np.zeros((norb, norb), dtype=complex)
        sigma_gw = np.zeros((grid.nf, grid.nk1, grid.nk2, norb, norb), dtype=complex)
        mu = float(opts.mu)

    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    converged = False
    err = float("inf")
    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
    W = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    P = np.zeros_like(W)

    for it in range(1, int(opts.max_iter) + 1):
        density = density_from_G_finite_box(G, grid)
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_out = sigma_gw_finite_box(G, W, Vq, grid, backend=backend)

        res_h = sigma_h_out - sigma_h
        res_gw = sigma_gw_out - sigma_gw
        err = _residual_error(res_h, res_gw)
        if opts.verbose:
            print(
                f"finite-box GW iter {it:4d}: residual={err:.3e}, mu={mu:.10f}, "
                f"n={np.sum(density):.10f}, method={method}, backend={backend}"
            )
        if err < float(opts.tol):
            converged = True
            break

        sigma_h, sigma_gw = _mixed_self_energies(
            sigma_h,
            sigma_gw,
            sigma_h_out,
            sigma_gw_out,
            opts,
            it,
            history,
        )
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
    else:
        it = int(opts.max_iter)

    # Re-evaluate the returned fixed-point residual and observables.
    G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
    density = density_from_G_finite_box(G, grid)
    sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = sigma_gw_finite_box(G, W, Vq, grid, backend=backend)
    err = _residual_error(sigma_h_out - sigma_h, sigma_gw_out - sigma_gw)
    converged = bool(err < float(opts.tol))

    (
        smin, mmin, omin, q1min, q2min,
        screening_mode, density_mode, density_mode_residual,
    ) = screening_soft_modes_matrix(P, Vq, grid)

    return GWResult(
        G=G,
        W=W,
        P=P,
        Sigma_H=sigma_h,
        Sigma_GW=sigma_gw,
        mu=mu,
        density=density,
        converged=converged,
        iterations=int(it),
        final_error=float(err),
        mixing_method=str(method),
        min_screening_singular_value=float(smin),
        min_screening_m=int(mmin),
        min_screening_Omega=float(omin),
        min_screening_q1=float(q1min),
        min_screening_q2=float(q2min),
        min_screening_mode=np.asarray(screening_mode),
        min_density_mode=np.asarray(density_mode),
        min_density_mode_residual=float(density_mode_residual),
    )


__all__ = [
    "density_from_G_finite_box",
    "one_body_density_matrix_finite_box",
    "sigma_gw_finite_box",
    "solve_matrix_gw_finite_box",
]
