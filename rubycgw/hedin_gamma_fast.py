"""Pulay-accelerated and warm-started GW-Gamma_P feedback.

This module is a performance-oriented companion to :mod:`rubycgw.hedin_gamma`.
It preserves the same Gamma_P fixed point but adds:

1. balanced three-block Pulay/DIIS mixing of Sigma_H, Sigma_GW and W;
2. exact-transfer density-vertex warm starts across outer iterations;
3. no redundant final covariant-density solve after a converged iteration;
4. reuse of the already-computed static Fock term when forming Sigma_out;
5. per-iteration timing and GMRES-work diagnostics.

The approximation is still Gamma_P only: the covariant vertex corrects the
polarization/screening sector, while the self-energy retains the GW form.
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
from .supercell_cgw import SupercellVertexOptions
from .supercell_gw import (
    compute_sigma_gw_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
)
from .supercell_gw_fast import (
    _build_tail_cache,
    _solve_mu_matrix_fast,
    density_from_G_cached,
    solve_matrix_gw_fast,
)
from .supercell_gw_split import compute_sigma_gw_split_components


@dataclass(frozen=True)
class GammaPFeedbackOptions:
    max_iter: int = 20
    tol: float = 2.0e-6
    mixing: float = 0.15
    w_mixing: float | None = None
    m_max: int | None = 0
    allow_unconverged_vertex: bool = False
    verbose: bool = True

    # Fast solver additions.  Existing callers need not provide them.
    mixing_method: str = "pulay"
    pulay_history: int = 6
    pulay_start: int = 3
    pulay_regularization: float = 1.0e-9
    pulay_damping: float = 0.80
    pulay_w_damping: float = 0.60
    pulay_step_factor: float = 3.0


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


def _balanced_inner(
    ah: np.ndarray,
    ag: np.ndarray,
    aw: np.ndarray,
    bh: np.ndarray,
    bg: np.ndarray,
    bw: np.ndarray,
) -> float:
    """Balanced real inner product for the three feedback residual blocks."""
    h = np.vdot(ah, bh).real / max(ah.size, 1)
    g = np.vdot(ag, bg).real / max(ag.size, 1)
    w = np.vdot(aw, bw).real / max(aw.size, 1)
    return float(h + g + w)


def _pulay_coefficients(history, regularization: float) -> np.ndarray:
    m = len(history)
    B = np.zeros((m + 1, m + 1), dtype=float)
    for i in range(m):
        *_, rhi, rgi, rwi = history[i]
        for j in range(i, m):
            *_, rhj, rgj, rwj = history[j]
            val = _balanced_inner(rhi, rgi, rwi, rhj, rgj, rwj)
            B[i, j] = val
            B[j, i] = val
    diag_scale = max(float(np.max(np.abs(np.diag(B[:m, :m])))), 1.0)
    B[:m, :m] += float(regularization) * diag_scale * np.eye(m)
    B[:m, m] = 1.0
    B[m, :m] = 1.0
    rhs = np.zeros(m + 1, dtype=float)
    rhs[m] = 1.0
    try:
        sol = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(B, rhs, rcond=None)[0]
    return sol[:m]


def _cap_candidate(
    current: np.ndarray,
    candidate: np.ndarray,
    residual: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Limit a DIIS extrapolation relative to the current raw fixed-point step."""
    step = np.asarray(candidate) - np.asarray(current)
    step_norm = _maxabs(step)
    ref = _maxabs(residual)
    limit = float(factor) * max(ref, 1.0e-15)
    if np.isfinite(step_norm) and step_norm > limit:
        step = step * (limit / step_norm)
    return np.asarray(current) + step


def _mixed_feedback(
    sigma_h,
    sigma_gw,
    W,
    sigma_h_out,
    sigma_gw_out,
    W_out,
    opts: GammaPFeedbackOptions,
    it: int,
    history,
):
    rh = sigma_h_out - sigma_h
    rg = sigma_gw_out - sigma_gw
    rw = W_out - W
    a = float(opts.mixing)
    wa = float(a if opts.w_mixing is None else opts.w_mixing)

    method = str(opts.mixing_method).strip().lower()
    if method not in {"linear", "pulay"}:
        raise ValueError("Gamma_P mixing_method must be 'linear' or 'pulay'")
    if method == "linear":
        return (
            sigma_h + a * rh,
            sigma_gw + a * rg,
            W + wa * rw,
            "linear",
            0.0,
        )

    history.append((
        np.array(sigma_h_out, copy=True),
        np.array(sigma_gw_out, copy=True),
        np.array(W_out, copy=True),
        np.array(rh, copy=True),
        np.array(rg, copy=True),
        np.array(rw, copy=True),
    ))
    keep = max(int(opts.pulay_history), 2)
    if len(history) > keep:
        del history[:-keep]

    if it < int(opts.pulay_start) or len(history) < 2:
        return (
            sigma_h + a * rh,
            sigma_gw + a * rg,
            W + wa * rw,
            "linear-warmup",
            0.0,
        )

    coeff = _pulay_coefficients(history, opts.pulay_regularization)
    h_diis = np.zeros_like(sigma_h)
    g_diis = np.zeros_like(sigma_gw)
    w_diis = np.zeros_like(W)
    for c, (hout, gout, wout, _, _, _) in zip(coeff, history):
        h_diis += c * hout
        g_diis += c * gout
        w_diis += c * wout

    fac = float(opts.pulay_step_factor)
    h_diis = _cap_candidate(sigma_h, h_diis, rh, fac)
    g_diis = _cap_candidate(sigma_gw, g_diis, rg, fac)
    w_diis = _cap_candidate(W, w_diis, rw, fac)

    dh = float(opts.pulay_damping)
    dw = float(opts.pulay_w_damping)
    hnext = (1.0 - dh) * sigma_h + dh * h_diis
    gnext = (1.0 - dh) * sigma_gw + dh * g_diis
    wnext = (1.0 - dw) * W + dw * w_diis
    return hnext, gnext, wnext, "pulay", float(np.max(np.abs(coeff)))


def _dyson_fixed_filling(
    h0,
    grid,
    sigma_h,
    sigma_gw,
    gw_opts,
    mu,
):
    if gw_opts.target_filling is None:
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
        return float(mu), G, _build_tail_cache(h0, sigma_h)
    mu, G, cache, _ = _solve_mu_matrix_fast(
        h0,
        sigma_h,
        sigma_gw,
        grid,
        float(gw_opts.target_filling),
        float(mu),
        float(gw_opts.mu_tol),
        int(gw_opts.mu_max_iter),
    )
    return float(mu), G, cache


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
    """Solve the Gamma_P feedback fixed point with DIIS and vertex warm starts."""
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
    wmix = float(
        feedback_opts.mixing
        if feedback_opts.w_mixing is None
        else feedback_opts.w_mixing
    )
    if not (0.0 < wmix <= 1.0):
        raise ValueError("feedback w_mixing must lie in (0,1]")
    if int(feedback_opts.max_iter) < 1:
        raise ValueError("feedback max_iter must be positive")
    if float(feedback_opts.tol) <= 0.0:
        raise ValueError("feedback tol must be positive")
    if int(feedback_opts.pulay_history) < 2:
        raise ValueError("pulay_history must be at least 2")

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
        gamma_cache = dict(getattr(initial_state, "vertex_gamma_cache", {}))
    else:
        sigma_h = np.asarray(background.Sigma_H, dtype=complex).copy()
        sigma_gw = np.asarray(background.Sigma_GW, dtype=complex).copy()
        W = np.asarray(background.W, dtype=complex).copy()
        mu = float(background.mu)
        gamma_cache = {}

    mu, G, cache = _dyson_fixed_filling(
        h0, grid, sigma_h, sigma_gw, gw_opts, mu
    )

    history = []
    timing_totals = {"vertex": 0.0, "screening": 0.0, "sigma": 0.0, "dyson": 0.0, "total": 0.0}
    converged = False
    err = float("inf")
    it = 0
    dens = None
    screening = None
    density = None
    sigma_f = None
    sigma_c_current = None
    vertex_stats = DensityVertexStats(0, 0, 0, 0.0, 0)

    for it in range(1, int(feedback_opts.max_iter) + 1):
        t_iter = perf_counter()
        density = density_from_G_cached(G, grid, mu, cache)
        sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])

        # Compute the current Fock term once.  It is independent of W and is
        # reused below when building the new Sigma target.
        _, sigma_f, sigma_c_current, _ = compute_sigma_gw_split_components(
            G,
            W,
            Vq,
            grid,
            h0,
            mu,
            sigma_h,
            backend=backend,
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
        W_out = screening.W_direct
        dt_screen = perf_counter() - ts
        timing_totals["screening"] += dt_screen

        tsig = perf_counter()
        Wc_out = np.asarray(W_out) - Vq[None, ...]
        sigma_c_out = compute_sigma_gw_matrix(
            G,
            Wc_out,
            grid,
            backend=backend,
        )
        sigma_gw_out = sigma_f[None, ...] + sigma_c_out
        dt_sigma = perf_counter() - tsig
        timing_totals["sigma"] += dt_sigma

        err = _feedback_error(
            sigma_h,
            sigma_h_out,
            sigma_gw,
            sigma_gw_out,
            W,
            W_out,
        )
        explicit = np.asarray(dens.solved_explicitly, dtype=bool)
        vmax = (
            float(np.nanmax(dens.transfer_max_error[explicit]))
            if np.any(explicit)
            else float("nan")
        )

        # If converged, the current G/W and the just-computed density response
        # are already the desired returned state; avoid another full BSE solve.
        if err < float(feedback_opts.tol):
            converged = True
            step_tag = "done"
            coeffmax = 0.0
            dt_dyson = 0.0
        elif it >= int(feedback_opts.max_iter):
            # Do not make an unevaluated final mixing step.  Return the last
            # explicitly evaluated state and its raw fixed-point residual.
            step_tag = "maxiter"
            coeffmax = 0.0
            dt_dyson = 0.0
        else:
            sigma_h_next, sigma_gw_next, W_next, step_tag, coeffmax = _mixed_feedback(
                sigma_h,
                sigma_gw,
                W,
                sigma_h_out,
                sigma_gw_out,
                W_out,
                feedback_opts,
                it,
                history,
            )
            td = perf_counter()
            mu, Gnext, cache_next = _dyson_fixed_filling(
                h0,
                grid,
                sigma_h_next,
                sigma_gw_next,
                gw_opts,
                mu,
            )
            dt_dyson = perf_counter() - td
            timing_totals["dyson"] += dt_dyson
            sigma_h = sigma_h_next
            sigma_gw = sigma_gw_next
            W = W_next
            G = Gnext
            cache = cache_next

        dt_total = perf_counter() - t_iter
        timing_totals["total"] += dt_total
        if feedback_opts.verbose:
            print(
                f"GWGamma_P iter {it:3d}: residual={err:.3e}, "
                f"mu={mu:.10f}, n={np.sum(density):.10f}, "
                f"vertex_err={vmax:.3e}, Hedin_id={screening.identity_error:.3e}, "
                f"step={step_tag}, coeffmax={coeffmax:.2e}, "
                f"GMRES={vertex_stats.total_iterations}/{vertex_stats.n_solves} "
                f"cache={vertex_stats.cache_hits}/{vertex_stats.n_solves}, "
                f"time={dt_total:.2f}s "
                f"[vertex={dt_vertex:.2f}, screen={dt_screen:.2f}, "
                f"sigma={dt_sigma:.2f}, dyson={dt_dyson:.2f}]"
            )

        if converged or it >= int(feedback_opts.max_iter):
            break

    assert dens is not None and screening is not None
    assert density is not None and sigma_f is not None and sigma_c_current is not None

    result = GWGammaPResult(
        G=np.asarray(G),
        W=np.asarray(W),
        P_gamma=np.asarray(screening.P_gamma),
        chi_cov=np.asarray(dens.chi_completed),
        Sigma_H=np.asarray(sigma_h),
        Sigma_GW=np.asarray(sigma_gw),
        Sigma_F=np.asarray(sigma_f),
        Sigma_c=np.asarray(sigma_c_current),
        mu=float(mu),
        density=np.asarray(density),
        converged=bool(converged),
        iterations=int(it),
        final_error=float(err),
        density_response=dens,
        fallback_mask=np.asarray(screening.fallback_mask),
        screening_identity_error=float(screening.identity_error),
        background=background,
    )
    # GWGammaPResult intentionally remains backward compatible; attach optional
    # performance state dynamically rather than changing the public dataclass.
    result.vertex_gamma_cache = gamma_cache
    result.vertex_stats = vertex_stats
    result.timing_totals = timing_totals
    result.feedback_mixing_method = str(feedback_opts.mixing_method)
    return result


__all__ = [
    "GammaPFeedbackOptions",
    "solve_matrix_gw_gamma_feedback",
]
