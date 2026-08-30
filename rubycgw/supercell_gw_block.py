"""Cheap adaptive block-mixing solver for the 18-site Ruby GW problem.

The GW equations are unchanged.  The numerical strategy is deliberately
minimal: use the analytic uniform Hartree field for a cold start, mix Hartree
and dynamic GW with different damping factors, evaluate the expensive GW map
exactly once per outer iteration, and use a small-history Anderson step only
late in the convergence.  There is no multi-trial line search.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .grids import MatsubaraGrid
from .gw import GWOptions, GWResult, _check_backend, _residual_error
from .model import RubyParameters
from .supercell import NSUP, build_supercell_h0, build_supercell_interaction
from .supercell_gw import (
    _compatible_initial,
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    compute_sigma_gw_matrix,
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


@dataclass(frozen=True)
class AndersonOptions:
    """API-compatible controls for the lightweight solver."""

    history: int = 6
    start: int = 8
    warmup_beta: float = 0.20
    beta: float = 0.30
    beta_min: float = 0.04
    beta_max: float = 0.30
    regularization: float = 1e-7
    enter_residual: float = 1e-2
    enter_ratio: float = 0.50
    stable_steps: int = 5
    reject_factor: float = 1.25
    recovery_steps: int = 3
    step_cap: float = 2.5
    scale_floor: float = 1e-4
    growth_factor: float = 1.20
    growth_patience: int = 3

    hartree_beta: float = 0.30
    gw_beta: float = 0.15
    gw_beta_floor: float = 0.04
    gw_beta_ceiling: float = 0.25
    diagnostic_interval: int = 10


def _validate(a: AndersonOptions) -> None:
    if a.history < 2:
        raise ValueError("history must be at least 2")
    if a.start < 1:
        raise ValueError("start must be positive")
    if not (0.0 < a.hartree_beta <= 1.0):
        raise ValueError("hartree_beta must lie in (0,1]")
    if not (0.0 < a.gw_beta_floor <= a.gw_beta <= a.gw_beta_ceiling <= 1.0):
        raise ValueError("Require 0 < gw_beta_floor <= gw_beta <= gw_beta_ceiling <= 1")
    if a.enter_residual <= 0.0:
        raise ValueError("enter_residual must be positive")
    if a.stable_steps < 1:
        raise ValueError("stable_steps must be positive")
    if a.regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if a.step_cap <= 1.0:
        raise ValueError("step_cap must exceed 1")
    if a.diagnostic_interval < 1:
        raise ValueError("diagnostic_interval must be positive")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def _metric_vector(
    h: np.ndarray,
    gw: np.ndarray,
    hscale: float,
    gwscale: float,
) -> np.ndarray:
    vh = np.asarray(h, dtype=complex).reshape(-1) / (
        hscale * np.sqrt(max(h.size, 1))
    )
    vg = np.asarray(gw, dtype=complex).reshape(-1) / (
        gwscale * np.sqrt(max(gw.size, 1))
    )
    return np.concatenate([vh, vg])


def _late_anderson_step(
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
    res_h: np.ndarray,
    res_gw: np.ndarray,
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    beta: float,
    regularization: float,
    step_cap: float,
    scale_floor: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """One Type-II Anderson proposal with a cheap local step-size cap."""
    if len(history) < 2:
        return sigma_h, sigma_gw, False

    hscale = max(_rms(sigma_h), float(scale_floor))
    gwscale = max(_rms(sigma_gw), float(scale_floor))
    dR_cols: list[np.ndarray] = []
    dX_h: list[np.ndarray] = []
    dX_gw: list[np.ndarray] = []
    dR_h: list[np.ndarray] = []
    dR_gw: list[np.ndarray] = []

    for old, new in zip(history[:-1], history[1:]):
        xh0, xg0, rh0, rg0 = old
        xh1, xg1, rh1, rg1 = new
        dxh, dxg = xh1 - xh0, xg1 - xg0
        drh, drg = rh1 - rh0, rg1 - rg0
        dX_h.append(dxh)
        dX_gw.append(dxg)
        dR_h.append(drh)
        dR_gw.append(drg)
        dR_cols.append(_metric_vector(drh, drg, hscale, gwscale))

    A = np.column_stack(dR_cols)
    rhs = _metric_vector(res_h, res_gw, hscale, gwscale)
    m = A.shape[1]
    if regularization > 0.0:
        A = np.vstack([A, np.sqrt(regularization) * np.eye(m, dtype=complex)])
        rhs = np.concatenate([rhs, np.zeros(m, dtype=complex)])

    try:
        gamma = np.linalg.lstsq(A, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return sigma_h, sigma_gw, False

    step_h = beta * res_h
    step_gw = beta * res_gw
    for c, dxh, dxg, drh, drg in zip(
        gamma, dX_h, dX_gw, dR_h, dR_gw
    ):
        step_h -= c * (dxh + beta * drh)
        step_gw -= c * (dxg + beta * drg)

    if not (np.all(np.isfinite(step_h)) and np.all(np.isfinite(step_gw))):
        return sigma_h, sigma_gw, False

    linear_norm = np.linalg.norm(
        _metric_vector(beta * res_h, beta * res_gw, hscale, gwscale)
    )
    step_norm = np.linalg.norm(_metric_vector(step_h, step_gw, hscale, gwscale))
    if step_norm > float(step_cap) * max(linear_norm, 1e-16):
        return sigma_h, sigma_gw, False

    return sigma_h + step_h, sigma_gw + step_gw, True


def _screening_min(P: np.ndarray, Vq: np.ndarray) -> float:
    norb = int(P.shape[-1])
    eye = np.eye(norb, dtype=complex)
    lhs = eye[None, None, None, :, :] - np.matmul(Vq[None, :, :, :, :], P)
    vals = np.linalg.svd(lhs, compute_uv=False)
    return float(np.min(vals[..., -1]))


def _uniform_hartree_seed(
    Vq0: np.ndarray,
    norb: int,
    target_filling: float | None,
) -> np.ndarray:
    if target_filling is None:
        return np.zeros((norb, norb), dtype=complex)
    nbar = float(target_filling) / float(norb)
    return hartree_self_energy_matrix(
        np.full(norb, nbar, dtype=float), Vq0
    )


def solve_matrix_gw_anderson(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    initial: GWResult | None = None,
    anderson: AndersonOptions = AndersonOptions(),
) -> GWResult:
    """Solve matrix GW with analytic Hartree start and one map per iteration."""
    _validate(anderson)
    backend = _check_backend(opts.momentum_backend)
    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(h0.shape[-1])
    if h0.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected h0 shape")
    if Vq.shape != (grid.nk1, grid.nk2, norb, norb):
        raise ValueError("unexpected Vq shape")
    Vq0 = Vq[0, 0]

    cold_start = not _compatible_initial(initial, grid, norb)
    if cold_start:
        sigma_h = _uniform_hartree_seed(Vq0, norb, opts.target_filling)
        sigma_gw = np.zeros(
            (grid.nf, grid.nk1, grid.nk2, norb, norb), dtype=complex
        )
        hartree_shift = float(np.mean(np.diag(sigma_h).real))
        mu = float(opts.mu) + hartree_shift
    else:
        sigma_h = np.array(initial.Sigma_H, copy=True)
        sigma_gw = np.array(initial.Sigma_GW, copy=True)
        mu = float(initial.mu)

    initial_mu_tol = _effective_mu_tol(opts.mu_tol, None)
    if opts.target_filling is None:
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
        cache = _build_tail_cache(h0, sigma_h)
        mu_neval = 0
    else:
        mu, G, cache, mu_neval = _solve_mu_matrix_fast(
            h0,
            sigma_h,
            sigma_gw,
            grid,
            float(opts.target_filling),
            mu,
            initial_mu_tol,
            opts.mu_max_iter,
        )

    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    gw_beta = float(anderson.gw_beta)
    prev_err = float("inf")
    prev_err_gw = float("inf")
    stable_count = 0
    used_anderson_last = False
    last_smin = float("nan")
    mu_tol_used = initial_mu_tol

    W = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    P = np.zeros_like(W)
    density = np.zeros(norb, dtype=float)
    converged = False
    err = err_h = err_gw = float("inf")
    it = 0

    for it in range(1, opts.max_iter + 1):
        density = density_from_G_cached(G, grid, mu, cache)
        sigma_h_out = hartree_self_energy_matrix(density, Vq0)
        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_out = compute_sigma_gw_matrix(G, W, grid, backend=backend)

        res_h = sigma_h_out - sigma_h
        res_gw = sigma_gw_out - sigma_gw
        err_h = float(np.max(np.abs(res_h)))
        err_gw = float(np.max(np.abs(res_gw)))
        err = _residual_error(res_h, res_gw)

        # If the previous late-Anderson proposal was genuinely poor, forget its
        # quasi-Newton history.  We do not roll back or evaluate extra trial maps.
        if used_anderson_last and np.isfinite(prev_err) and err > 1.25 * prev_err:
            history.clear()
            stable_count = 0
            used_anderson_last = False
            gw_beta = max(float(anderson.gw_beta_floor), 0.7 * gw_beta)

        if np.isfinite(prev_err):
            stable_count = stable_count + 1 if err <= 1.02 * prev_err else 0

        if np.isfinite(prev_err_gw):
            ratio = err_gw / max(prev_err_gw, 1e-30)
            if ratio > 1.25:
                gw_beta = max(float(anderson.gw_beta_floor), 0.65 * gw_beta)
            elif ratio < 0.75 and stable_count >= 2:
                gw_beta = min(float(anderson.gw_beta_ceiling), 1.08 * gw_beta)

        # SVD is diagnostic only and therefore never changes the numerical path.
        if opts.verbose and (it == 1 or it % int(anderson.diagnostic_interval) == 0):
            last_smin = _screening_min(P, Vq)

        if opts.verbose:
            smin_text = f"{last_smin:.3e}" if np.isfinite(last_smin) else "--"
            phase = "anderson-late" if used_anderson_last else "block-linear"
            print(
                f"SC-GW iter {it:4d}: residual={err:.3e}, "
                f"rH={err_h:.3e}, rGW={err_gw:.3e}, smin={smin_text}, "
                f"mu={mu:.10f}, n={np.sum(density):.10f}, "
                f"mu_eval={mu_neval}, mu_tol={mu_tol_used:.1e}, "
                f"mixer={phase}, alphaH={anderson.hartree_beta:.3f}, "
                f"alphaGW={gw_beta:.3f}, backend={backend}"
            )

        if err < opts.tol:
            converged = True
            break

        history.append(
            (
                np.array(sigma_h, copy=True),
                np.array(sigma_gw, copy=True),
                np.array(res_h, copy=True),
                np.array(res_gw, copy=True),
            )
        )
        keep = max(int(anderson.history) + 1, 2)
        if len(history) > keep:
            del history[:-keep]

        sigma_h_next = sigma_h + float(anderson.hartree_beta) * res_h
        sigma_gw_next = sigma_gw + gw_beta * res_gw
        used_anderson_next = False

        can_anderson = bool(
            it >= int(anderson.start)
            and err < float(anderson.enter_residual)
            and stable_count >= int(anderson.stable_steps)
            and len(history) >= 2
        )
        if can_anderson:
            beta_a = min(0.50, max(0.20, float(anderson.beta)))
            ah, agw, valid = _late_anderson_step(
                sigma_h,
                sigma_gw,
                res_h,
                res_gw,
                history,
                beta_a,
                float(anderson.regularization),
                min(float(anderson.step_cap), 2.5),
                float(anderson.scale_floor),
            )
            if valid:
                sigma_h_next = ah
                sigma_gw_next = agw
                used_anderson_next = True

        mu_tol_used = _effective_mu_tol(opts.mu_tol, err)
        if opts.target_filling is None:
            mu_next = mu
            Gnext = dyson_from_sigma_matrix(
                h0, grid, mu_next, sigma_h_next, sigma_gw_next
            )
            cache_next = _build_tail_cache(h0, sigma_h_next)
            mu_neval_next = 0
        else:
            mu_next, Gnext, cache_next, mu_neval_next = _solve_mu_matrix_fast(
                h0,
                sigma_h_next,
                sigma_gw_next,
                grid,
                float(opts.target_filling),
                mu,
                mu_tol_used,
                opts.mu_max_iter,
            )

        prev_err = float(err)
        prev_err_gw = float(err_gw)
        used_anderson_last = used_anderson_next
        sigma_h = sigma_h_next
        sigma_gw = sigma_gw_next
        mu = float(mu_next)
        G = Gnext
        cache = cache_next
        mu_neval = int(mu_neval_next)

    mu, G, cache, mu_neval_final = _strict_refine_fixed_filling(
        h0,
        sigma_h,
        sigma_gw,
        grid,
        opts.target_filling,
        mu,
        opts.mu_tol,
        opts.mu_max_iter,
    )
    density = density_from_G_cached(G, grid, mu, cache)
    sigma_h_out = hartree_self_energy_matrix(density, Vq0)
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_matrix(G, W, grid, backend=backend)
    res_h = sigma_h_out - sigma_h
    res_gw = sigma_gw_out - sigma_gw
    err_h = float(np.max(np.abs(res_h)))
    err_gw = float(np.max(np.abs(res_gw)))
    err = _residual_error(res_h, res_gw)
    converged = bool(err < opts.tol)

    if opts.verbose and opts.target_filling is not None:
        print(
            f"SC-GW strict mu refine: mu={mu:.10f}, n={np.sum(density):.10f}, "
            f"mu_eval={mu_neval_final}, mu_tol={opts.mu_tol:.1e}, "
            f"residual={err:.3e}, rH={err_h:.3e}, rGW={err_gw:.3e}"
        )

    (
        smin,
        mmin,
        omin,
        q1min,
        q2min,
        screening_mode,
        density_mode,
        density_mode_residual,
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
        iterations=it,
        final_error=float(err),
        mixing_method="adaptive-block-late-anderson",
        min_screening_singular_value=smin,
        min_screening_m=mmin,
        min_screening_Omega=omin,
        min_screening_q1=q1min,
        min_screening_q2=q2min,
        min_screening_mode=screening_mode,
        min_density_mode=density_mode,
        min_density_mode_residual=density_mode_residual,
    )


def solve_supercell_gw_anderson(
    params: RubyParameters,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    source_strength: float = 0.0,
    initial: GWResult | None = None,
    anderson: AndersonOptions = AndersonOptions(),
) -> GWResult:
    h0 = build_supercell_h0(
        grid.kmesh(), params, source_strength=source_strength
    )
    Vq = build_supercell_interaction(grid.qmesh(), params)
    if h0.shape[-1] != NSUP:
        raise RuntimeError("unexpected supercell matrix dimension")
    return solve_matrix_gw_anderson(
        h0, Vq, grid, opts=opts, initial=initial, anderson=anderson
    )
