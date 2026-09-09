"""Self-consistent GW + statically screened SOSEX solver.

This module deliberately excludes the covariant Gamma_P feedback.  Screening is
ordinary self-consistent GW screening,

    P = G G,
    W = (1 - V P)^(-1) V,

while the non-Hartree self-energy fixed point is

    Sigma_corr = Sigma_GW + Sigma_sSOSEX,
    Sigma_GW   = Sigma_F - G (W - V).

The screened-exchange correction is evaluated from the *same current ordinary
GW W* used in Sigma_GW.  With the default ``oneW-sym`` mode,

    Sigma_sSOSEX = 1/2 [ S[V,W0] + S[W0,V] ],
    W0(q) = W(q,Omega=0),

where S[U,X] is the crossed G^3 interaction skeleton.  ``twoW`` instead uses
S[W0,W0].  In either case W0 -> V reproduces bare SOX.

This is useful as a clean diagnostic of Sigma-side screened exchange without
mixing in any density-vertex/Gamma_P correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid
from .gw import (
    GWOptions,
    GWResult,
    _check_backend,
    _check_mixing_method,
    _mixed_self_energies,
    _residual_error,
)
from .model import RubyParameters, build_h0, build_interaction
from .ssosex_static import (
    ScreenedSOSEXOptions,
    compute_static_screened_sosex_self_energy_periodic_fast,
)
from .supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
    screening_soft_modes_matrix,
)
from .supercell_gw_fast import (
    _build_tail_cache,
    _effective_mu_tol,
    _solve_mu_matrix_fast,
    _strict_refine_fixed_filling,
    density_from_G_cached,
)
from .supercell_gw_split import compute_sigma_gw_split_components


@dataclass
class GWScreenedSOSEXResult:
    """Periodic GW+sSOSEX fixed point with resolved self-energy pieces."""

    G: np.ndarray
    W: np.ndarray
    P: np.ndarray
    Sigma_H: np.ndarray
    Sigma_corr: np.ndarray
    Sigma_GW: np.ndarray
    Sigma_F: np.ndarray
    Sigma_c: np.ndarray
    Sigma_sSOSEX: np.ndarray
    mu: float
    density: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    mixing_method: str
    ssosex_mode: str
    min_screening_singular_value: float
    min_screening_m: int
    min_screening_Omega: float
    min_screening_q1: float
    min_screening_q2: float
    min_screening_mode: np.ndarray
    min_density_mode: np.ndarray
    min_density_mode_residual: float

    @property
    def Sigma_total(self) -> np.ndarray:
        return self.Sigma_H[None, None, None, :, :] + self.Sigma_corr


def _compatible_initial(initial, grid: MatsubaraGrid, norb: int) -> bool:
    if initial is None:
        return False
    expected = (grid.nf, grid.nk1, grid.nk2, norb, norb)
    return (
        np.asarray(getattr(initial, "Sigma_H", np.empty(0))).shape == (norb, norb)
        and np.asarray(getattr(initial, "G", np.empty(0))).shape == expected
    )


def _initial_self_energies(initial, grid, norb, mu_default):
    if not _compatible_initial(initial, grid, norb):
        sigma_h = np.zeros((norb, norb), dtype=complex)
        sigma_corr = np.zeros(
            (grid.nf, grid.nk1, grid.nk2, norb, norb), dtype=complex
        )
        return sigma_h, sigma_corr, float(mu_default)

    sigma_h = np.asarray(initial.Sigma_H, dtype=complex).copy()
    if hasattr(initial, "Sigma_corr"):
        sigma_corr = np.asarray(initial.Sigma_corr, dtype=complex).copy()
    elif hasattr(initial, "Sigma_GW"):
        sigma_corr = np.asarray(initial.Sigma_GW, dtype=complex).copy()
    else:
        sigma_corr = np.zeros(
            (grid.nf, grid.nk1, grid.nk2, norb, norb), dtype=complex
        )
    return sigma_h, sigma_corr, float(getattr(initial, "mu", mu_default))


def _screened_exchange_reference_hamiltonian(h0, sigma_h, sigma_f):
    return (
        np.asarray(h0, dtype=complex)
        + np.asarray(sigma_h, dtype=complex)[None, None, :, :]
        + np.asarray(sigma_f, dtype=complex)
    )


def solve_matrix_gw_ssosex(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    ssosex_opts: ScreenedSOSEXOptions = ScreenedSOSEXOptions(),
    initial: GWScreenedSOSEXResult | GWResult | None = None,
) -> GWScreenedSOSEXResult:
    """Solve self-consistent ordinary-GW screening + static screened SOSEX."""
    backend = _check_backend(opts.momentum_backend)
    method = _check_mixing_method(opts.mixing_method)
    ssosex_opts.validate()
    if opts.pulay_history < 2:
        raise ValueError("pulay_history must be at least 2")
    if opts.pulay_start < 1:
        raise ValueError("pulay_start must be at least 1")

    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(h0.shape[-1])
    expected_h = (grid.nk1, grid.nk2, norb, norb)
    if h0.shape != expected_h or Vq.shape != expected_h:
        raise ValueError("h0/Vq shape mismatch")
    Vq0 = Vq[0, 0]

    sigma_h, sigma_corr, mu = _initial_self_energies(
        initial, grid, norb, opts.mu
    )

    initial_mu_tol = _effective_mu_tol(opts.mu_tol, None)
    if opts.target_filling is None:
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_corr)
        tail_cache = _build_tail_cache(h0, sigma_h)
        mu_neval = 0
    else:
        mu, G, tail_cache, mu_neval = _solve_mu_matrix_fast(
            h0, sigma_h, sigma_corr, grid, float(opts.target_filling), mu,
            initial_mu_tol, opts.mu_max_iter,
        )
    mu_tol_used = initial_mu_tol

    W = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    P = np.zeros_like(W)
    history = []
    err = float("inf")
    converged = False
    it = 0

    sigma_gw_out = np.zeros_like(sigma_corr)
    sigma_f_out = np.zeros((grid.nk1, grid.nk2, norb, norb), dtype=complex)
    sigma_c_out = np.zeros_like(sigma_corr)
    sigma_ssosex_out = np.zeros_like(sigma_corr)

    for it in range(1, int(opts.max_iter) + 1):
        density = density_from_G_cached(G, grid, mu, tail_cache)
        sigma_h_out = hartree_self_energy_matrix(density, Vq0)

        # Ordinary GW screening only: no covariant Gamma anywhere in this map.
        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)

        sigma_gw_out, sigma_f_out, sigma_c_out, _ = (
            compute_sigma_gw_split_components(
                G, W, Vq, grid, h0, mu, sigma_h, backend=backend
            )
        )
        h_x_ref = _screened_exchange_reference_hamiltonian(
            h0, sigma_h, sigma_f_out
        )
        sigma_ssosex_out = compute_static_screened_sosex_self_energy_periodic_fast(
            G, Vq, W, h_x_ref, mu, grid, opts=ssosex_opts
        )
        sigma_corr_out = sigma_gw_out + sigma_ssosex_out

        res_h = sigma_h_out - sigma_h
        res_corr = sigma_corr_out - sigma_corr
        err = _residual_error(res_h, res_corr)
        if opts.verbose:
            sx = float(np.max(np.abs(sigma_ssosex_out)))
            print(
                f"SC-GW+sSOSEX iter {it:4d}: residual={err:.3e}, "
                f"mu={mu:.10f}, n={np.sum(density):.10f}, "
                f"max|Sigma_sSOSEX|={sx:.3e}, mode={ssosex_opts.mode}, "
                f"mu_eval={mu_neval}, method={method}, backend={backend}"
            )
        if err < float(opts.tol):
            converged = True
            break

        sigma_h_next, sigma_corr_next = _mixed_self_energies(
            sigma_h, sigma_corr, sigma_h_out, sigma_corr_out,
            opts, it, history,
        )
        mu_tol_next = _effective_mu_tol(opts.mu_tol, err)
        if opts.target_filling is None:
            Gnext = dyson_from_sigma_matrix(
                h0, grid, mu, sigma_h_next, sigma_corr_next
            )
            tail_cache_next = _build_tail_cache(h0, sigma_h_next)
            mu_neval_next = 0
        else:
            mu, Gnext, tail_cache_next, mu_neval_next = _solve_mu_matrix_fast(
                h0, sigma_h_next, sigma_corr_next, grid,
                float(opts.target_filling), mu, mu_tol_next, opts.mu_max_iter,
            )
        sigma_h = sigma_h_next
        sigma_corr = sigma_corr_next
        G = Gnext
        tail_cache = tail_cache_next
        mu_neval = mu_neval_next
        mu_tol_used = mu_tol_next

    # Rebuild the complete map on the strictly refined returned state.
    mu, G, tail_cache, mu_neval_final = _strict_refine_fixed_filling(
        h0, sigma_h, sigma_corr, grid, opts.target_filling, mu,
        opts.mu_tol, opts.mu_max_iter,
    )
    density = density_from_G_cached(G, grid, mu, tail_cache)
    sigma_h_out = hartree_self_energy_matrix(density, Vq0)
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out, sigma_f_out, sigma_c_out, _ = compute_sigma_gw_split_components(
        G, W, Vq, grid, h0, mu, sigma_h, backend=backend
    )
    h_x_ref = _screened_exchange_reference_hamiltonian(h0, sigma_h, sigma_f_out)
    sigma_ssosex_out = compute_static_screened_sosex_self_energy_periodic_fast(
        G, Vq, W, h_x_ref, mu, grid, opts=ssosex_opts
    )
    sigma_corr_out = sigma_gw_out + sigma_ssosex_out
    err = _residual_error(sigma_h_out - sigma_h, sigma_corr_out - sigma_corr)
    converged = bool(np.isfinite(err) and err < float(opts.tol))

    if opts.verbose and opts.target_filling is not None:
        print(
            f"SC-GW+sSOSEX strict mu refine: mu={mu:.10f}, "
            f"n={np.sum(density):.10f}, mu_eval={mu_neval_final}, "
            f"residual={err:.3e}"
        )

    (
        smin, mmin, omin, q1min, q2min, screening_mode,
        density_mode, density_mode_residual,
    ) = screening_soft_modes_matrix(P, Vq, grid)

    return GWScreenedSOSEXResult(
        G=np.asarray(G), W=np.asarray(W), P=np.asarray(P),
        Sigma_H=np.asarray(sigma_h), Sigma_corr=np.asarray(sigma_corr),
        Sigma_GW=np.asarray(sigma_gw_out), Sigma_F=np.asarray(sigma_f_out),
        Sigma_c=np.asarray(sigma_c_out), Sigma_sSOSEX=np.asarray(sigma_ssosex_out),
        mu=float(mu), density=np.asarray(density), converged=converged,
        iterations=int(it), final_error=float(err), mixing_method=method,
        ssosex_mode=str(ssosex_opts.mode),
        min_screening_singular_value=smin, min_screening_m=mmin,
        min_screening_Omega=omin, min_screening_q1=q1min,
        min_screening_q2=q2min, min_screening_mode=screening_mode,
        min_density_mode=density_mode,
        min_density_mode_residual=density_mode_residual,
    )


def solve_primitive_gw_ssosex(
    params: RubyParameters,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    ssosex_opts: ScreenedSOSEXOptions = ScreenedSOSEXOptions(),
    initial: GWScreenedSOSEXResult | GWResult | None = None,
) -> GWScreenedSOSEXResult:
    h0 = np.asarray(build_h0(grid.kmesh(), params), dtype=complex)
    Vq = np.asarray(build_interaction(grid.qmesh(), params), dtype=complex)
    return solve_matrix_gw_ssosex(
        h0, Vq, grid, opts=opts, ssosex_opts=ssosex_opts, initial=initial
    )


__all__ = [
    "GWScreenedSOSEXResult",
    "solve_matrix_gw_ssosex",
    "solve_primitive_gw_ssosex",
]
