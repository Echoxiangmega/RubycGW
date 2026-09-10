"""Self-consistent GW plus fully frequency-dependent one-screened-line SOSEX.

Screening remains ordinary self-consistent GW,

    P = G G,   W = (1-VP)^(-1)V,

while

    Sigma_corr = Sigma_GW + Sigma_dSOSEX.

The default dynamic SOSEX correction is

    Sigma_dSOSEX = SOX + 1/2[(W-V,V)+(V,W-V)],

so no W(q,Omega=0) replacement is made.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .dynamic_sosex import (
    DynamicSOSEXOptions,
    compute_dynamic_sosex_self_energy_periodic_fast,
)
from .grids import MatsubaraGrid
from .gw import (
    GWOptions,
    GWResult,
    _check_backend,
    _check_mixing_method,
    _mixed_self_energies,
    _residual_error,
)
from .gw_ssosex import _initial_self_energies
from .model import RubyParameters, build_h0, build_interaction
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
class GWDynamicSOSEXResult:
    G: np.ndarray
    W: np.ndarray
    P: np.ndarray
    Sigma_H: np.ndarray
    Sigma_corr: np.ndarray
    Sigma_GW: np.ndarray
    Sigma_F: np.ndarray
    Sigma_c: np.ndarray
    Sigma_dSOSEX: np.ndarray
    Sigma_SOX: np.ndarray
    Sigma_WpV: np.ndarray
    Sigma_VWp: np.ndarray
    mu: float
    density: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    mixing_method: str
    dynamic_sosex_mode: str
    mixed_line_relative_difference: float
    min_screening_singular_value: float
    min_screening_m: int
    min_screening_Omega: float
    min_screening_q1: float
    min_screening_q2: float
    min_screening_mode: np.ndarray
    min_density_mode: np.ndarray
    min_density_mode_residual: float

    @property
    def Sigma_total(self):
        return self.Sigma_H[None, None, None, :, :] + self.Sigma_corr


def _href(h0, sigma_h, sigma_f):
    return (
        np.asarray(h0, dtype=complex)
        + np.asarray(sigma_h, dtype=complex)[None, None]
        + np.asarray(sigma_f, dtype=complex)
    )


def solve_matrix_gw_dynamic_sosex(
    h0,
    Vq,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    dsosex_opts: DynamicSOSEXOptions = DynamicSOSEXOptions(),
    initial: object | None = None,
):
    backend = _check_backend(opts.momentum_backend)
    method = _check_mixing_method(opts.mixing_method)
    dsosex_opts.validate()
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(h0.shape[-1])
    expected = (grid.nk1, grid.nk2, norb, norb)
    if h0.shape != expected or Vq.shape != expected:
        raise ValueError("h0/Vq shape mismatch")

    sigma_h, sigma_corr, mu = _initial_self_energies(
        initial, grid, norb, opts.mu
    )
    mu_tol_used = _effective_mu_tol(opts.mu_tol, None)
    if opts.target_filling is None:
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_corr)
        tail_cache = _build_tail_cache(h0, sigma_h)
        mu_neval = 0
    else:
        mu, G, tail_cache, mu_neval = _solve_mu_matrix_fast(
            h0, sigma_h, sigma_corr, grid, float(opts.target_filling), mu,
            mu_tol_used, opts.mu_max_iter,
        )

    history = []
    err = float("inf")
    converged = False
    it = 0
    P = np.zeros((grid.nb,) + expected, dtype=complex)
    W = np.zeros_like(P)
    sigma_gw_out = np.zeros_like(G)
    sigma_f_out = np.zeros(expected, dtype=complex)
    sigma_c_out = np.zeros_like(G)
    parts = None

    for it in range(1, int(opts.max_iter) + 1):
        density = density_from_G_cached(G, grid, mu, tail_cache)
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_out, sigma_f_out, sigma_c_out, _ = (
            compute_sigma_gw_split_components(
                G, W, Vq, grid, h0, mu, sigma_h, backend=backend
            )
        )
        parts = compute_dynamic_sosex_self_energy_periodic_fast(
            G, Vq, W, _href(h0, sigma_h, sigma_f_out), mu, grid,
            opts=dsosex_opts, return_parts=True,
        )
        sigma_corr_out = sigma_gw_out + parts.Sigma
        err = _residual_error(
            sigma_h_out - sigma_h, sigma_corr_out - sigma_corr
        )
        if opts.verbose:
            print(
                f"SC-GW+dSOSEX iter {it:4d}: residual={err:.3e}, "
                f"mu={mu:.10f}, n={np.sum(density):.10f}, "
                f"max|Sigma_dSOSEX|={np.max(np.abs(parts.Sigma)):.3e}, "
                f"line_asym={parts.mixed_line_relative_difference:.3e}, "
                f"mode={dsosex_opts.mode}, method={method}"
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
            cache_next = _build_tail_cache(h0, sigma_h_next)
            mu_neval_next = 0
        else:
            mu, Gnext, cache_next, mu_neval_next = _solve_mu_matrix_fast(
                h0, sigma_h_next, sigma_corr_next, grid,
                float(opts.target_filling), mu, mu_tol_next, opts.mu_max_iter,
            )
        sigma_h = sigma_h_next
        sigma_corr = sigma_corr_next
        G = Gnext
        tail_cache = cache_next
        mu_neval = mu_neval_next
        mu_tol_used = mu_tol_next

    mu, G, tail_cache, mu_neval_final = _strict_refine_fixed_filling(
        h0, sigma_h, sigma_corr, grid, opts.target_filling, mu,
        opts.mu_tol, opts.mu_max_iter,
    )
    density = density_from_G_cached(G, grid, mu, tail_cache)
    sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out, sigma_f_out, sigma_c_out, _ = compute_sigma_gw_split_components(
        G, W, Vq, grid, h0, mu, sigma_h, backend=backend
    )
    parts = compute_dynamic_sosex_self_energy_periodic_fast(
        G, Vq, W, _href(h0, sigma_h, sigma_f_out), mu, grid,
        opts=dsosex_opts, return_parts=True,
    )
    sigma_corr_out = sigma_gw_out + parts.Sigma
    err = _residual_error(
        sigma_h_out - sigma_h, sigma_corr_out - sigma_corr
    )
    converged = bool(np.isfinite(err) and err < float(opts.tol))

    if opts.verbose and opts.target_filling is not None:
        print(
            f"SC-GW+dSOSEX strict mu refine: mu={mu:.10f}, "
            f"n={np.sum(density):.10f}, mu_eval={mu_neval_final}, "
            f"residual={err:.3e}, mu_tol_last={mu_tol_used:.3e}"
        )

    (
        smin, mmin, omin, q1min, q2min, screening_mode,
        density_mode, density_mode_residual,
    ) = screening_soft_modes_matrix(P, Vq, grid)

    return GWDynamicSOSEXResult(
        G=np.asarray(G), W=np.asarray(W), P=np.asarray(P),
        Sigma_H=np.asarray(sigma_h), Sigma_corr=np.asarray(sigma_corr),
        Sigma_GW=np.asarray(sigma_gw_out), Sigma_F=np.asarray(sigma_f_out),
        Sigma_c=np.asarray(sigma_c_out), Sigma_dSOSEX=np.asarray(parts.Sigma),
        Sigma_SOX=np.asarray(parts.Sigma_SOX),
        Sigma_WpV=np.asarray(parts.Sigma_WpV),
        Sigma_VWp=np.asarray(parts.Sigma_VWp),
        mu=float(mu), density=np.asarray(density), converged=converged,
        iterations=int(it), final_error=float(err), mixing_method=method,
        dynamic_sosex_mode=str(dsosex_opts.mode),
        mixed_line_relative_difference=float(parts.mixed_line_relative_difference),
        min_screening_singular_value=smin, min_screening_m=mmin,
        min_screening_Omega=omin, min_screening_q1=q1min,
        min_screening_q2=q2min, min_screening_mode=screening_mode,
        min_density_mode=density_mode,
        min_density_mode_residual=density_mode_residual,
    )


def solve_primitive_gw_dynamic_sosex(
    params: RubyParameters,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    dsosex_opts: DynamicSOSEXOptions = DynamicSOSEXOptions(),
    initial: object | None = None,
):
    h0 = np.asarray(build_h0(grid.kmesh(), params), dtype=complex)
    Vq = np.asarray(build_interaction(grid.qmesh(), params), dtype=complex)
    return solve_matrix_gw_dynamic_sosex(
        h0, Vq, grid, opts=opts, dsosex_opts=dsosex_opts, initial=initial
    )


__all__ = [
    "GWDynamicSOSEXResult",
    "solve_matrix_gw_dynamic_sosex",
    "solve_primitive_gw_dynamic_sosex",
]
