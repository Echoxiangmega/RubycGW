"""Covariant density-vertex screening and an iterative GW-Gamma_P diagnostic.

This module connects the covariant density response already implemented in
:mod:`rubycgw.covariant_density` to the Hedin representation of screening.
With the sign convention used in RubycGW,

    W = V + V P W,
    chi_nn = -(1 - P V)^(-1) P,

so the same screened interaction can be written either as

    W = (1 - V P)^(-1) V

or directly from the *reducible* physical density response as

    W = V - V chi_nn V.

For a finite covariant response one may therefore recover the corresponding
Hedin irreducible polarization without inverting V,

    P_Gamma = - chi_nn (1 - V chi_nn)^(-1).

The helper functions below expose this conversion and explicitly verify the two
forms of W.

``solve_matrix_gw_gamma_feedback`` iterates this covariant screening back into
Sigma_GW = Sigma_F - G (W-V).  It is deliberately labelled a ``Gamma_P``
*feedback diagnostic*: Gamma enters the density/polarization sector while the
self-energy still has the GW form and no explicit three-point Gamma factor.
Moreover, the covariant kernel is the functional derivative of the current
GW-like self-energy; a fully conserving Hedin GWGamma closure would also require
updating the self-energy vertex consistently.  The present solver is therefore
useful for testing whether vertex-corrected screening can remove or strongly
renormalize a GW branch, but it should not be advertised as the final exact
Hedin closure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .covariant_density import (
    CovariantDensityResult,
    compute_covariant_density_susceptibility,
)
from .grids import MatsubaraGrid
from .gw import GWOptions, GWResult, _check_backend
from .post_gw import build_post_screened_interaction
from .supercell_cgw import SupercellVertexOptions
from .supercell_gw import hartree_self_energy_matrix
from .supercell_gw_fast import (
    _build_tail_cache,
    _solve_mu_matrix_fast,
    density_from_G_cached,
    solve_matrix_gw_fast,
)
from .supercell_gw_split import (
    compute_sigma_gw_split_components,
    compute_sigma_gw_split_matrix,
)


@dataclass(frozen=True)
class GammaPFeedbackOptions:
    max_iter: int = 20
    tol: float = 2.0e-6
    mixing: float = 0.15
    w_mixing: float | None = None
    m_max: int | None = 0
    allow_unconverged_vertex: bool = False
    verbose: bool = True


@dataclass
class HedinGammaScreening:
    chi_cov: np.ndarray
    P_gamma: np.ndarray
    W_direct: np.ndarray
    W_hedin: np.ndarray
    fallback_mask: np.ndarray
    identity_error: float


@dataclass
class GWGammaPResult:
    G: np.ndarray
    W: np.ndarray
    P_gamma: np.ndarray
    chi_cov: np.ndarray
    Sigma_H: np.ndarray
    Sigma_GW: np.ndarray
    Sigma_F: np.ndarray
    Sigma_c: np.ndarray
    mu: float
    density: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    density_response: CovariantDensityResult
    fallback_mask: np.ndarray
    screening_identity_error: float
    background: GWResult


def _right_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return b @ inv(a) using a linear solve rather than an explicit inverse."""
    return np.linalg.solve(np.swapaxes(a, -1, -2), np.swapaxes(b, -1, -2)).swapaxes(-1, -2)


def covariant_chi_to_irreducible_p(
    Vq: np.ndarray,
    chi_cov: np.ndarray,
) -> np.ndarray:
    """Convert reducible ``chi_cov`` to the Hedin irreducible ``P_Gamma``.

    The repository convention is

        chi = -(1 - P V)^(-1) P,

    hence

        P = -chi (1 - V chi)^(-1).

    Non-finite transfers are left as NaN; this is convenient for a static-only
    diagnostic where high bosonic frequencies are intentionally not solved.
    """
    Vq = np.asarray(Vq, dtype=complex)
    chi = np.asarray(chi_cov, dtype=complex)
    if chi.ndim != 5:
        raise ValueError("chi_cov must have shape (nb,nk1,nk2,norb,norb)")
    if Vq.shape != chi.shape[1:]:
        raise ValueError("Vq/chi_cov shape mismatch")

    norb = int(chi.shape[-1])
    eye = np.eye(norb, dtype=complex)
    out = np.full_like(chi, np.nan + 0j)
    for im in range(chi.shape[0]):
        for iq1 in range(chi.shape[1]):
            for iq2 in range(chi.shape[2]):
                c = chi[im, iq1, iq2]
                if not np.all(np.isfinite(c)):
                    continue
                v = Vq[iq1, iq2]
                a = eye - v @ c
                out[im, iq1, iq2] = _right_solve(a, -c)
    return out


def screened_interaction_from_p_gamma(
    Vq: np.ndarray,
    P_gamma: np.ndarray,
) -> np.ndarray:
    """Build ``W=(1-V P_Gamma)^(-1)V`` on finite transfers."""
    Vq = np.asarray(Vq, dtype=complex)
    P = np.asarray(P_gamma, dtype=complex)
    if P.ndim != 5 or Vq.shape != P.shape[1:]:
        raise ValueError("Vq/P_gamma shape mismatch")
    norb = int(P.shape[-1])
    eye = np.eye(norb, dtype=complex)
    out = np.full_like(P, np.nan + 0j)
    for im in range(P.shape[0]):
        for iq1 in range(P.shape[1]):
            for iq2 in range(P.shape[2]):
                p = P[im, iq1, iq2]
                if not np.all(np.isfinite(p)):
                    continue
                v = Vq[iq1, iq2]
                out[im, iq1, iq2] = np.linalg.solve(eye - v @ p, v)
    return out


def build_hedin_gamma_screening(
    Vq: np.ndarray,
    chi_cov: np.ndarray,
    *,
    background_W: np.ndarray | None = None,
) -> HedinGammaScreening:
    """Construct the same vertex-corrected W in reducible and Hedin forms."""
    W_direct, fallback = build_post_screened_interaction(
        Vq,
        chi_cov,
        background_W=background_W,
    )
    P_gamma = covariant_chi_to_irreducible_p(Vq, chi_cov)
    W_hedin = screened_interaction_from_p_gamma(Vq, P_gamma)

    finite = np.all(np.isfinite(P_gamma), axis=(-2, -1))
    if np.any(finite):
        err = float(np.max(np.abs(W_direct[finite] - W_hedin[finite])))
    else:
        err = float("nan")
    # Make the Hedin representation usable by callers with a windowed response:
    # unsolved transfers simply inherit the same background W as W_direct.
    if background_W is not None:
        W_hedin = np.asarray(W_hedin).copy()
        W_hedin[~finite] = np.asarray(background_W, dtype=complex)[~finite]

    return HedinGammaScreening(
        chi_cov=np.asarray(chi_cov),
        P_gamma=np.asarray(P_gamma),
        W_direct=np.asarray(W_direct),
        W_hedin=np.asarray(W_hedin),
        fallback_mask=np.asarray(fallback),
        identity_error=err,
    )


def _maxabs(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a))))


def _feedback_error(
    sigma_h: np.ndarray,
    sigma_h_out: np.ndarray,
    sigma_gw: np.ndarray,
    sigma_gw_out: np.ndarray,
    W: np.ndarray,
    W_out: np.ndarray,
) -> float:
    return max(
        _maxabs(sigma_h_out - sigma_h),
        _maxabs(sigma_gw_out - sigma_gw),
        _maxabs(W_out - W),
    )


def solve_matrix_gw_gamma_feedback(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    *,
    gw_opts: GWOptions = GWOptions(),
    vertex_opts: SupercellVertexOptions = SupercellVertexOptions(),
    feedback_opts: GammaPFeedbackOptions = GammaPFeedbackOptions(),
    background: GWResult | None = None,
    initial_state: GWGammaPResult | None = None,
) -> GWGammaPResult:
    """Iterate covariant density screening back into a GW self-energy.

    ``background`` is the ordinary SC-GW solution used as the default starting
    point and retained in the returned result for comparison.  ``initial_state``
    may be a previous converged Gamma_P solution (for example the preceding
    source field in a continuation scan).
    """
    backend = _check_backend(gw_opts.momentum_backend)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(h0.shape[-1])
    if h0.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected h0 shape")
    if Vq.shape != h0.shape:
        raise ValueError("Vq/h0 shape mismatch")
    if not (0.0 < float(feedback_opts.mixing) <= 1.0):
        raise ValueError("feedback mixing must lie in (0,1]")
    wmix = float(feedback_opts.mixing if feedback_opts.w_mixing is None else feedback_opts.w_mixing)
    if not (0.0 < wmix <= 1.0):
        raise ValueError("feedback w_mixing must lie in (0,1]")
    if int(feedback_opts.max_iter) < 1:
        raise ValueError("feedback max_iter must be positive")
    if float(feedback_opts.tol) <= 0.0:
        raise ValueError("feedback tol must be positive")

    if background is None:
        background = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    if not background.converged and not feedback_opts.allow_unconverged_vertex:
        raise RuntimeError(
            f"ordinary GW background is not converged: {background.final_error:.3e}"
        )

    if initial_state is not None and initial_state.G.shape == background.G.shape:
        sigma_h = np.asarray(initial_state.Sigma_H, dtype=complex).copy()
        sigma_gw = np.asarray(initial_state.Sigma_GW, dtype=complex).copy()
        W = np.asarray(initial_state.W, dtype=complex).copy()
        mu = float(initial_state.mu)
    else:
        sigma_h = np.asarray(background.Sigma_H, dtype=complex).copy()
        sigma_gw = np.asarray(background.Sigma_GW, dtype=complex).copy()
        W = np.asarray(background.W, dtype=complex).copy()
        mu = float(background.mu)

    if gw_opts.target_filling is None:
        from .supercell_gw import dyson_from_sigma_matrix
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
        cache = _build_tail_cache(h0, sigma_h)
    else:
        mu, G, cache, _ = _solve_mu_matrix_fast(
            h0,
            sigma_h,
            sigma_gw,
            grid,
            float(gw_opts.target_filling),
            mu,
            float(gw_opts.mu_tol),
            int(gw_opts.mu_max_iter),
        )

    last_density_response = None
    last_screening = None
    err = float("inf")
    converged = False
    it = 0

    for it in range(1, int(feedback_opts.max_iter) + 1):
        density = density_from_G_cached(G, grid, mu, cache)
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
        _, sigma_f, _, _ = compute_sigma_gw_split_components(
            G,
            W,
            Vq,
            grid,
            h0,
            mu,
            sigma_h,
            backend=backend,
        )
        dens = compute_covariant_density_susceptibility(
            G,
            W,
            Vq,
            h0,
            mu,
            sigma_h,
            sigma_f,
            grid,
            vertex_opts=vertex_opts,
            m_max=feedback_opts.m_max,
            allow_unconverged=feedback_opts.allow_unconverged_vertex,
        )
        screening = build_hedin_gamma_screening(
            Vq,
            dens.chi_completed,
            background_W=W if feedback_opts.m_max is not None else None,
        )
        W_out = screening.W_direct
        sigma_gw_out = compute_sigma_gw_split_matrix(
            G,
            W_out,
            Vq,
            grid,
            h0,
            mu,
            sigma_h,
            backend=backend,
        )
        err = _feedback_error(
            sigma_h,
            sigma_h_out,
            sigma_gw,
            sigma_gw_out,
            W,
            W_out,
        )
        if feedback_opts.verbose:
            explicit = np.asarray(dens.solved_explicitly, dtype=bool)
            vmax = float(np.nanmax(dens.transfer_max_error[explicit])) if np.any(explicit) else float("nan")
            print(
                f"GWGamma_P iter {it:3d}: residual={err:.3e}, "
                f"mu={mu:.10f}, n={np.sum(density):.10f}, "
                f"vertex_err={vmax:.3e}, Hedin_id={screening.identity_error:.3e}"
            )
        last_density_response = dens
        last_screening = screening
        if err < float(feedback_opts.tol):
            converged = True
            break

        a = float(feedback_opts.mixing)
        sigma_h = (1.0 - a) * sigma_h + a * sigma_h_out
        sigma_gw = (1.0 - a) * sigma_gw + a * sigma_gw_out
        W = (1.0 - wmix) * W + wmix * W_out

        if gw_opts.target_filling is None:
            from .supercell_gw import dyson_from_sigma_matrix
            G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
            cache = _build_tail_cache(h0, sigma_h)
        else:
            mu, G, cache, _ = _solve_mu_matrix_fast(
                h0,
                sigma_h,
                sigma_gw,
                grid,
                float(gw_opts.target_filling),
                mu,
                float(gw_opts.mu_tol),
                int(gw_opts.mu_max_iter),
            )

    # Re-evaluate the target map on the final state.  This also guarantees that
    # the returned chi/P_Gamma correspond to the returned Green function.
    density = density_from_G_cached(G, grid, mu, cache)
    sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
    sigma_total, sigma_f, sigma_c, _ = compute_sigma_gw_split_components(
        G,
        W,
        Vq,
        grid,
        h0,
        mu,
        sigma_h,
        backend=backend,
    )
    dens = compute_covariant_density_susceptibility(
        G,
        W,
        Vq,
        h0,
        mu,
        sigma_h,
        sigma_f,
        grid,
        vertex_opts=vertex_opts,
        m_max=feedback_opts.m_max,
        allow_unconverged=feedback_opts.allow_unconverged_vertex,
    )
    screening = build_hedin_gamma_screening(
        Vq,
        dens.chi_completed,
        background_W=W if feedback_opts.m_max is not None else None,
    )
    sigma_gw_out = compute_sigma_gw_split_matrix(
        G,
        screening.W_direct,
        Vq,
        grid,
        h0,
        mu,
        sigma_h,
        backend=backend,
    )
    err = _feedback_error(
        sigma_h,
        sigma_h_out,
        sigma_gw,
        sigma_gw_out,
        W,
        screening.W_direct,
    )
    converged = bool(err < float(feedback_opts.tol))

    return GWGammaPResult(
        G=np.asarray(G),
        W=np.asarray(W),
        P_gamma=np.asarray(screening.P_gamma),
        chi_cov=np.asarray(dens.chi_completed),
        Sigma_H=np.asarray(sigma_h),
        Sigma_GW=np.asarray(sigma_gw),
        Sigma_F=np.asarray(sigma_f),
        Sigma_c=np.asarray(sigma_c),
        mu=float(mu),
        density=np.asarray(density),
        converged=converged,
        iterations=int(it),
        final_error=float(err),
        density_response=dens,
        fallback_mask=np.asarray(screening.fallback_mask),
        screening_identity_error=float(screening.identity_error),
        background=background,
    )


__all__ = [
    "GammaPFeedbackOptions",
    "HedinGammaScreening",
    "GWGammaPResult",
    "covariant_chi_to_irreducible_p",
    "screened_interaction_from_p_gamma",
    "build_hedin_gamma_screening",
    "solve_matrix_gw_gamma_feedback",
]
