"""Self-consistent GW-Gamma_P plus statically screened SOSEX.

The fixed-point map is

    W_out = V - V chi_cov V,
    Sigma_GW,out = Sigma_F[G] - G (W_out - V),
    Sigma_corr,out = Sigma_GW,out + Sigma_sSOSEX[G,V,W_out(0)],

with the default screened-exchange term

    Sigma_sSOSEX = 1/2 { S[V,W0] + S[W0,V] },
    W0(q) = W_out(q,Omega=0).

Thus bare SOX is recovered when W0=V, while bubble screening of either crossed
interaction line is resummed through the current Gamma_P-screened W0.  The
density-vertex kernel itself remains the production GW H/F/MT/AL kernel; the
functional derivative of screened SOSEX is deliberately *not* inserted here.
That would require derivatives both of the three G lines and of W0[G] and is a
separate higher-level closure.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .covariant_density_fast import (
    DensityVertexStats,
    compute_covariant_density_susceptibility_fast,
)
from .grids import MatsubaraGrid
from .gw import GWOptions, GWResult, _check_backend
from .hedin_gamma import GWGammaPResult, build_hedin_gamma_screening
from .hedin_gamma_fast import (
    GammaPFeedbackOptions,
    _dyson_fixed_filling,
    _feedback_error,
    _mixed_feedback,
)
from .ssosex_static import (
    ScreenedSOSEXOptions,
    compute_static_screened_sosex_self_energy_periodic_fast,
)
from .supercell_cgw import SupercellVertexOptions
from .supercell_gw import compute_sigma_gw_matrix, hartree_self_energy_matrix
from .supercell_gw_fast import density_from_G_cached, solve_matrix_gw_fast
from .supercell_gw_split import compute_sigma_gw_split_components


@dataclass
class GWGammaPSSOSEXResult:
    """Returned fixed point for GW-Gamma_P plus static screened SOSEX."""

    G: np.ndarray
    W: np.ndarray
    P_gamma: np.ndarray
    chi_cov: np.ndarray
    Sigma_H: np.ndarray
    Sigma_corr: np.ndarray
    Sigma_GW: np.ndarray
    Sigma_F: np.ndarray
    Sigma_c: np.ndarray
    Sigma_SSOSEX: np.ndarray
    mu: float
    density: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    density_response: object
    fallback_mask: np.ndarray
    screening_identity_error: float
    background: GWResult
    ssosex_mode: str

    @property
    def Sigma_total(self) -> np.ndarray:
        return self.Sigma_H[None, None, None, :, :] + self.Sigma_corr


def _reference_hamiltonian(h0, sigma_h, sigma_f):
    return (
        np.asarray(h0, dtype=complex)
        + np.asarray(sigma_h, dtype=complex)[None, None, :, :]
        + np.asarray(sigma_f, dtype=complex)
    )


def _initial_state_arrays(initial_state, background, grid, norb):
    expected = (grid.nf, grid.nk1, grid.nk2, norb, norb)
    compatible = (
        initial_state is not None
        and np.asarray(getattr(initial_state, "G", np.empty(0))).shape == expected
        and np.asarray(getattr(initial_state, "Sigma_H", np.empty(0))).shape == (norb, norb)
    )
    if not compatible:
        return (
            np.asarray(background.Sigma_H, dtype=complex).copy(),
            np.asarray(background.Sigma_GW, dtype=complex).copy(),
            np.asarray(background.W, dtype=complex).copy(),
            float(background.mu),
            {},
        )

    sigma_h = np.asarray(initial_state.Sigma_H, dtype=complex).copy()
    if hasattr(initial_state, "Sigma_corr"):
        sigma_corr = np.asarray(initial_state.Sigma_corr, dtype=complex).copy()
    elif hasattr(initial_state, "Sigma_GW"):
        sigma_corr = np.asarray(initial_state.Sigma_GW, dtype=complex).copy()
        if hasattr(initial_state, "Sigma_SOX"):
            sigma_corr = sigma_corr + np.asarray(initial_state.Sigma_SOX, dtype=complex)
        if hasattr(initial_state, "Sigma_SSOSEX"):
            sigma_corr = sigma_corr + np.asarray(initial_state.Sigma_SSOSEX, dtype=complex)
    else:
        sigma_corr = np.asarray(background.Sigma_GW, dtype=complex).copy()

    W = np.asarray(getattr(initial_state, "W", background.W), dtype=complex).copy()
    mu = float(getattr(initial_state, "mu", background.mu))
    gamma_cache = dict(getattr(initial_state, "vertex_gamma_cache", {}))
    return sigma_h, sigma_corr, W, mu, gamma_cache


def solve_matrix_gw_gamma_ssosex_feedback(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    *,
    gw_opts: GWOptions = GWOptions(),
    vertex_opts: SupercellVertexOptions = SupercellVertexOptions(),
    feedback_opts: GammaPFeedbackOptions = GammaPFeedbackOptions(),
    ssosex_opts: ScreenedSOSEXOptions = ScreenedSOSEXOptions(),
    background: GWResult | None = None,
    initial_state: object | None = None,
) -> GWGammaPSSOSEXResult:
    """Solve the Gamma_P-screened GW + static screened-SOSEX fixed point."""
    backend = _check_backend(gw_opts.momentum_backend)
    ssosex_opts.validate()
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(h0.shape[-1])
    expected_h = (grid.nk1, grid.nk2, norb, norb)
    if h0.shape != expected_h or Vq.shape != expected_h:
        raise ValueError("h0/Vq shape mismatch")
    if not (0.0 < float(feedback_opts.mixing) <= 1.0):
        raise ValueError("feedback mixing must lie in (0,1]")
    if int(feedback_opts.max_iter) < 1 or float(feedback_opts.tol) <= 0.0:
        raise ValueError("invalid feedback max_iter/tol")

    if background is None:
        background = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    if not background.converged and not feedback_opts.allow_unconverged_vertex:
        raise RuntimeError(
            f"ordinary GW background is not converged: {background.final_error:.3e}"
        )

    sigma_h, sigma_corr, W, mu, gamma_cache = _initial_state_arrays(
        initial_state, background, grid, norb
    )
    mu, G, cache = _dyson_fixed_filling(
        h0, grid, sigma_h, sigma_corr, gw_opts, mu
    )

    history = []
    timing_totals = {
        "vertex": 0.0,
        "screening": 0.0,
        "gw_sigma": 0.0,
        "ssosex": 0.0,
        "dyson": 0.0,
        "total": 0.0,
    }
    converged = False
    err = float("inf")
    dens = screening = density = None
    sigma_f = sigma_c_out = sigma_gw_out = sigma_ssosex_out = None
    vertex_stats = DensityVertexStats(0, 0, 0, 0.0, 0)
    it = 0

    for it in range(1, int(feedback_opts.max_iter) + 1):
        t_iter = perf_counter()
        density = density_from_G_cached(G, grid, mu, cache)
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])

        _, sigma_f, _, _ = compute_sigma_gw_split_components(
            G, W, Vq, grid, h0, mu, sigma_h, backend=backend
        )

        tv = perf_counter()
        dens, gamma_cache, vertex_stats = compute_covariant_density_susceptibility_fast(
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
            initial_gamma_cache=gamma_cache,
        )
        dt_vertex = perf_counter() - tv
        timing_totals["vertex"] += dt_vertex

        ts = perf_counter()
        screening = build_hedin_gamma_screening(
            Vq,
            dens.chi_completed,
            background_W=W if feedback_opts.m_max is not None else None,
        )
        W_out = np.asarray(screening.W_direct)
        dt_screen = perf_counter() - ts
        timing_totals["screening"] += dt_screen

        tg = perf_counter()
        Wc_out = W_out - Vq[None, ...]
        sigma_c_out = compute_sigma_gw_matrix(G, Wc_out, grid, backend=backend)
        sigma_gw_out = sigma_f[None, ...] + sigma_c_out
        dt_gw = perf_counter() - tg
        timing_totals["gw_sigma"] += dt_gw

        tx = perf_counter()
        href = _reference_hamiltonian(h0, sigma_h, sigma_f)
        sigma_ssosex_out = compute_static_screened_sosex_self_energy_periodic_fast(
            G, Vq, W_out, href, mu, grid, opts=ssosex_opts
        )
        dt_ssosex = perf_counter() - tx
        timing_totals["ssosex"] += dt_ssosex

        sigma_corr_out = sigma_gw_out + sigma_ssosex_out
        err = _feedback_error(
            sigma_h,
            sigma_h_out,
            sigma_corr,
            sigma_corr_out,
            W,
            W_out,
        )

        explicit = np.asarray(dens.solved_explicitly, dtype=bool)
        vmax = (
            float(np.nanmax(dens.transfer_max_error[explicit]))
            if np.any(explicit)
            else float("nan")
        )
        sx_scale = float(np.max(np.abs(sigma_ssosex_out)))

        if err < float(feedback_opts.tol):
            converged = True
            step_tag = "done"
            coeffmax = 0.0
            dt_dyson = 0.0
        elif it >= int(feedback_opts.max_iter):
            step_tag = "maxiter"
            coeffmax = 0.0
            dt_dyson = 0.0
        else:
            sigma_h_next, sigma_corr_next, W_next, step_tag, coeffmax = _mixed_feedback(
                sigma_h,
                sigma_corr,
                W,
                sigma_h_out,
                sigma_corr_out,
                W_out,
                feedback_opts,
                it,
                history,
            )
            td = perf_counter()
            mu, Gnext, cache_next = _dyson_fixed_filling(
                h0, grid, sigma_h_next, sigma_corr_next, gw_opts, mu
            )
            dt_dyson = perf_counter() - td
            timing_totals["dyson"] += dt_dyson
            sigma_h = sigma_h_next
            sigma_corr = sigma_corr_next
            W = W_next
            G = Gnext
            cache = cache_next

        dt_total = perf_counter() - t_iter
        timing_totals["total"] += dt_total
        if feedback_opts.verbose:
            print(
                f"GWGamma_P+sSOSEX iter {it:3d}: residual={err:.3e}, "
                f"mu={mu:.10f}, n={np.sum(density):.10f}, "
                f"vertex_err={vmax:.3e}, Hedin_id={screening.identity_error:.3e}, "
                f"max|sSOSEX|={sx_scale:.3e}, mode={ssosex_opts.mode}, "
                f"step={step_tag}, coeffmax={coeffmax:.2e}, "
                f"GMRES={vertex_stats.total_iterations}/{vertex_stats.n_solves} "
                f"cache={vertex_stats.cache_hits}/{vertex_stats.n_solves}, "
                f"time={dt_total:.2f}s [vertex={dt_vertex:.2f}, screen={dt_screen:.2f}, "
                f"gwSigma={dt_gw:.2f}, ssosex={dt_ssosex:.2f}, dyson={dt_dyson:.2f}]"
            )

        if converged or it >= int(feedback_opts.max_iter):
            break

    assert dens is not None and screening is not None and density is not None
    assert sigma_f is not None and sigma_c_out is not None
    assert sigma_gw_out is not None and sigma_ssosex_out is not None

    result = GWGammaPSSOSEXResult(
        G=np.asarray(G),
        W=np.asarray(W),
        P_gamma=np.asarray(screening.P_gamma),
        chi_cov=np.asarray(dens.chi_completed),
        Sigma_H=np.asarray(sigma_h),
        Sigma_corr=np.asarray(sigma_corr),
        Sigma_GW=np.asarray(sigma_gw_out),
        Sigma_F=np.asarray(sigma_f),
        Sigma_c=np.asarray(sigma_c_out),
        Sigma_SSOSEX=np.asarray(sigma_ssosex_out),
        mu=float(mu),
        density=np.asarray(density),
        converged=bool(converged),
        iterations=int(it),
        final_error=float(err),
        density_response=dens,
        fallback_mask=np.asarray(screening.fallback_mask),
        screening_identity_error=float(screening.identity_error),
        background=background,
        ssosex_mode=str(ssosex_opts.mode),
    )
    result.vertex_gamma_cache = gamma_cache
    result.vertex_stats = vertex_stats
    result.timing_totals = timing_totals
    result.feedback_mixing_method = str(feedback_opts.mixing_method)
    return result


__all__ = [
    "GWGammaPSSOSEXResult",
    "solve_matrix_gw_gamma_ssosex_feedback",
]
