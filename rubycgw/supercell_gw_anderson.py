"""Adaptive Anderson acceleration for the 18-site period-three Ruby GW solver.

This module keeps the same GW fixed-point map as :mod:`supercell_gw_fast`.
Anderson acceleration is used only after a contractive basin has been reached;
trial steps that make the raw GW residual substantially worse are rolled back
and replaced by a conservative linear recovery step.

No physical equation is changed.  Only the numerical path used to solve
``Sigma = F[Sigma]`` is modified.
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
    """Controls for conservative safeguarded Type-II Anderson acceleration."""

    history: int = 6
    start: int = 8
    warmup_beta: float = 0.20
    beta: float = 0.30
    beta_min: float = 0.08
    beta_max: float = 0.70
    regularization: float = 1e-7
    enter_residual: float = 0.10
    enter_ratio: float = 0.50
    stable_steps: int = 5
    reject_factor: float = 1.10
    recovery_steps: int = 4
    step_cap: float = 3.0
    scale_floor: float = 1e-4


def _validate_anderson_options(aopts: AndersonOptions) -> None:
    if aopts.history < 2:
        raise ValueError("Anderson history must be at least 2")
    if aopts.start < 1:
        raise ValueError("Anderson start must be at least 1")
    if not (
        0.0 < aopts.beta_min
        <= aopts.warmup_beta
        <= aopts.beta_max
        <= 1.0
    ):
        raise ValueError("Require 0 < beta_min <= warmup_beta <= beta_max <= 1")
    if not (aopts.beta_min <= aopts.beta <= aopts.beta_max):
        raise ValueError("Require beta_min <= beta <= beta_max")
    if aopts.regularization < 0.0:
        raise ValueError("Anderson regularization must be non-negative")
    if aopts.enter_residual <= 0.0:
        raise ValueError("Anderson enter_residual must be positive")
    if not (0.0 < aopts.enter_ratio <= 1.0):
        raise ValueError("Anderson enter_ratio must lie in (0,1]")
    if aopts.stable_steps < 1:
        raise ValueError("Anderson stable_steps must be positive")
    if aopts.reject_factor <= 1.0:
        raise ValueError("Anderson reject_factor must exceed 1")
    if aopts.recovery_steps < 0:
        raise ValueError("Anderson recovery_steps must be non-negative")
    if aopts.step_cap <= 1.0:
        raise ValueError("Anderson step_cap must exceed 1")
    if aopts.scale_floor <= 0.0:
        raise ValueError("Anderson scale_floor must be positive")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def _block_scales(
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
    sigma_h_out: np.ndarray,
    sigma_gw_out: np.ndarray,
    floor: float,
) -> tuple[float, float]:
    sh = max(_rms(sigma_h), _rms(sigma_h_out), float(floor))
    sg = max(_rms(sigma_gw), _rms(sigma_gw_out), float(floor))
    return sh, sg


def _metric_vector(
    h: np.ndarray, gw: np.ndarray, sh: float, sg: float
) -> np.ndarray:
    """Equal-weight Hartree/GW block metric, with amplitude normalization."""
    vh = np.asarray(h, dtype=complex).reshape(-1) / (
        sh * np.sqrt(max(h.size, 1))
    )
    vg = np.asarray(gw, dtype=complex).reshape(-1) / (
        sg * np.sqrt(max(gw.size, 1))
    )
    return np.concatenate([vh, vg])


def _metric_norm(
    h: np.ndarray, gw: np.ndarray, sh: float, sg: float
) -> float:
    return float(np.linalg.norm(_metric_vector(h, gw, sh, sg)))


def _anderson_type2_step(
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
    res_h: np.ndarray,
    res_gw: np.ndarray,
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    beta: float,
    regularization: float,
    sh: float,
    sg: float,
    step_cap: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return one locally safeguarded Type-II Anderson proposal."""
    linear_h = beta * res_h
    linear_gw = beta * res_gw
    if len(history) < 2:
        return sigma_h + linear_h, sigma_gw + linear_gw, False

    dR_cols = []
    dX_h, dX_gw, dR_h, dR_gw = [], [], [], []
    for old, new in zip(history[:-1], history[1:]):
        xh0, xg0, rh0, rg0 = old
        xh1, xg1, rh1, rg1 = new
        dxh, dxg = xh1 - xh0, xg1 - xg0
        drh, drg = rh1 - rh0, rg1 - rg0
        dX_h.append(dxh)
        dX_gw.append(dxg)
        dR_h.append(drh)
        dR_gw.append(drg)
        dR_cols.append(_metric_vector(drh, drg, sh, sg))

    A = np.column_stack(dR_cols)
    rhs = _metric_vector(res_h, res_gw, sh, sg)
    m = A.shape[1]
    lam = float(regularization)
    if lam > 0.0:
        A_aug = np.vstack([A, np.sqrt(lam) * np.eye(m, dtype=complex)])
        rhs_aug = np.concatenate([rhs, np.zeros(m, dtype=complex)])
    else:
        A_aug, rhs_aug = A, rhs

    try:
        gamma = np.linalg.lstsq(A_aug, rhs_aug, rcond=None)[0]
    except np.linalg.LinAlgError:
        return sigma_h + linear_h, sigma_gw + linear_gw, False

    step_h = np.array(linear_h, copy=True)
    step_gw = np.array(linear_gw, copy=True)
    for c, dxh, dxg, drh, drg in zip(
        gamma, dX_h, dX_gw, dR_h, dR_gw
    ):
        step_h -= c * (dxh + beta * drh)
        step_gw -= c * (dxg + beta * drg)

    if not (
        np.all(np.isfinite(step_h)) and np.all(np.isfinite(step_gw))
    ):
        return sigma_h + linear_h, sigma_gw + linear_gw, False

    linear_norm = _metric_norm(linear_h, linear_gw, sh, sg)
    step_norm = _metric_norm(step_h, step_gw, sh, sg)
    if step_norm > float(step_cap) * max(linear_norm, 1e-16):
        return sigma_h + linear_h, sigma_gw + linear_gw, False

    return sigma_h + step_h, sigma_gw + step_gw, True


def _solve_next_G(
    h0: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions,
    sigma_h_next: np.ndarray,
    sigma_gw_next: np.ndarray,
    mu_guess: float,
    outer_residual: float,
):
    mu_tol = _effective_mu_tol(opts.mu_tol, outer_residual)
    if opts.target_filling is None:
        Gnext = dyson_from_sigma_matrix(
            h0, grid, mu_guess, sigma_h_next, sigma_gw_next
        )
        return (
            float(mu_guess),
            Gnext,
            _build_tail_cache(h0, sigma_h_next),
            0,
            mu_tol,
        )
    mu, Gnext, cache, neval = _solve_mu_matrix_fast(
        h0,
        sigma_h_next,
        sigma_gw_next,
        grid,
        float(opts.target_filling),
        float(mu_guess),
        mu_tol,
        opts.mu_max_iter,
    )
    return mu, Gnext, cache, neval, mu_tol


def solve_matrix_gw_anderson(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    initial: GWResult | None = None,
    anderson: AndersonOptions = AndersonOptions(),
) -> GWResult:
    """Solve matrix GW with basin finding, Anderson acceleration and rollback."""
    _validate_anderson_options(anderson)
    backend = _check_backend(opts.momentum_backend)

    h0 = np.asarray(h0, dtype=complex)
    Vq = np.asarray(Vq, dtype=complex)
    norb = int(h0.shape[-1])
    expected_h = (grid.nk1, grid.nk2, norb, norb)
    expected_v = (grid.nk1, grid.nk2, norb, norb)
    if h0.shape != expected_h:
        raise ValueError(f"h0 shape {h0.shape} != expected {expected_h}")
    if Vq.shape != expected_v:
        raise ValueError(f"Vq shape {Vq.shape} != expected {expected_v}")
    Vq0 = Vq[0, 0]

    if _compatible_initial(initial, grid, norb):
        sigma_h = np.array(initial.Sigma_H, copy=True)
        sigma_gw = np.array(initial.Sigma_GW, copy=True)
        mu = float(initial.mu)
    else:
        sigma_h = np.zeros((norb, norb), dtype=complex)
        sigma_gw = np.zeros(
            (grid.nf, grid.nk1, grid.nk2, norb, norb), dtype=complex
        )
        mu = float(opts.mu)

    initial_mu_tol = _effective_mu_tol(opts.mu_tol, None)
    if opts.target_filling is None:
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
        tail_cache = _build_tail_cache(h0, sigma_h)
        mu_neval = 0
    else:
        mu, G, tail_cache, mu_neval = _solve_mu_matrix_fast(
            h0,
            sigma_h,
            sigma_gw,
            grid,
            float(opts.target_filling),
            mu,
            initial_mu_tol,
            opts.mu_max_iter,
        )
    mu_tol_used = initial_mu_tol

    W = np.zeros(
        (grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex
    )
    P = np.zeros_like(W)

    history: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    beta = float(anderson.warmup_beta)
    entered_anderson = False
    initial_err = None
    prev_err = float("inf")
    stable_count = 0
    recovery_remaining = 0
    pending_parent = None
    err = float("inf")
    converged = False
    it = 0

    for it in range(1, opts.max_iter + 1):
        density = density_from_G_cached(G, grid, mu, tail_cache)
        sigma_h_out = hartree_self_energy_matrix(density, Vq0)
        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_out = compute_sigma_gw_matrix(
            G, W, grid, backend=backend
        )

        res_h = sigma_h_out - sigma_h
        res_gw = sigma_gw_out - sigma_gw
        err = _residual_error(res_h, res_gw)
        if initial_err is None and np.isfinite(err):
            initial_err = float(err)

        # A trial Anderson step is accepted only after its *actual* GW map has
        # been evaluated.  If it made the residual substantially worse, roll
        # back to the parent and take a small linear recovery step instead.
        if pending_parent is not None:
            parent_err = float(pending_parent[5])
            if err > float(anderson.reject_factor) * parent_err:
                if opts.verbose:
                    print(
                        f"SC-GW iter {it:4d}: reject Anderson trial "
                        f"residual={err:.3e} > "
                        f"{anderson.reject_factor:.2f}*{parent_err:.3e}; rollback"
                    )
                (
                    parent_h,
                    parent_gw,
                    parent_rh,
                    parent_rgw,
                    parent_mu,
                    parent_err,
                ) = pending_parent
                recovery_beta = max(
                    float(anderson.beta_min),
                    min(float(anderson.warmup_beta), 0.5 * beta),
                )
                sigma_h_next = parent_h + recovery_beta * parent_rh
                sigma_gw_next = parent_gw + recovery_beta * parent_rgw
                (
                    mu,
                    G,
                    tail_cache,
                    mu_neval,
                    mu_tol_used,
                ) = _solve_next_G(
                    h0,
                    grid,
                    opts,
                    sigma_h_next,
                    sigma_gw_next,
                    parent_mu,
                    parent_err,
                )
                sigma_h = sigma_h_next
                sigma_gw = sigma_gw_next
                history.clear()
                entered_anderson = False
                stable_count = 0
                recovery_remaining = int(anderson.recovery_steps)
                beta = recovery_beta
                prev_err = float(parent_err)
                pending_parent = None
                continue
            pending_parent = None

        if np.isfinite(prev_err):
            if err <= 1.02 * prev_err:
                stable_count += 1
            else:
                stable_count = 0

            if entered_anderson:
                if err < 0.70 * prev_err:
                    beta = min(float(anderson.beta_max), 1.08 * beta)
                elif err > prev_err:
                    beta = max(float(anderson.beta_min), 0.75 * beta)

        can_enter = bool(
            not entered_anderson
            and recovery_remaining <= 0
            and it >= int(anderson.start)
            and (
                err <= float(anderson.enter_residual)
                or (
                    initial_err is not None
                    and stable_count >= int(anderson.stable_steps)
                    and err <= float(anderson.enter_ratio) * initial_err
                )
            )
        )
        if can_enter:
            entered_anderson = True
            beta = float(anderson.beta)
            history.clear()

        phase = (
            "recovery"
            if recovery_remaining > 0
            else ("anderson" if entered_anderson else "basin-linear")
        )
        if opts.verbose:
            print(
                f"SC-GW iter {it:4d}: residual={err:.3e}, "
                f"mu={mu:.10f}, n={np.sum(density):.10f}, "
                f"mu_eval={mu_neval}, mu_tol={mu_tol_used:.1e}, "
                f"mixer={phase}, beta={beta:.3f}, backend={backend}"
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
        keep_states = max(int(anderson.history) + 1, 2)
        if len(history) > keep_states:
            del history[:-keep_states]

        sh, sg = _block_scales(
            sigma_h,
            sigma_gw,
            sigma_h_out,
            sigma_gw_out,
            anderson.scale_floor,
        )

        use_anderson = bool(
            entered_anderson
            and recovery_remaining <= 0
            and len(history) >= 2
        )
        accepted_proposal = False
        if use_anderson:
            (
                sigma_h_next,
                sigma_gw_next,
                accepted_proposal,
            ) = _anderson_type2_step(
                sigma_h,
                sigma_gw,
                res_h,
                res_gw,
                history,
                beta,
                anderson.regularization,
                sh,
                sg,
                anderson.step_cap,
            )
            if accepted_proposal:
                pending_parent = (
                    np.array(sigma_h, copy=True),
                    np.array(sigma_gw, copy=True),
                    np.array(res_h, copy=True),
                    np.array(res_gw, copy=True),
                    float(mu),
                    float(err),
                )
            else:
                history.clear()
                entered_anderson = False
                recovery_remaining = max(int(anderson.recovery_steps), 2)
                beta = max(float(anderson.beta_min), 0.5 * beta)
        else:
            sigma_h_next = sigma_h + beta * res_h
            sigma_gw_next = sigma_gw + beta * res_gw

        if recovery_remaining > 0:
            recovery_remaining -= 1

        (
            mu,
            Gnext,
            tail_cache_next,
            mu_neval_next,
            mu_tol_next,
        ) = _solve_next_G(
            h0,
            grid,
            opts,
            sigma_h_next,
            sigma_gw_next,
            mu,
            err,
        )

        sigma_h = sigma_h_next
        sigma_gw = sigma_gw_next
        G = Gnext
        tail_cache = tail_cache_next
        mu_neval = mu_neval_next
        mu_tol_used = mu_tol_next
        prev_err = float(err)

    # The returned solution always satisfies the strict requested filling
    # tolerance, even though intermediate iterations used inexact inner solves.
    mu, G, tail_cache, mu_neval_final = _strict_refine_fixed_filling(
        h0,
        sigma_h,
        sigma_gw,
        grid,
        opts.target_filling,
        mu,
        opts.mu_tol,
        opts.mu_max_iter,
    )
    density = density_from_G_cached(G, grid, mu, tail_cache)
    sigma_h_out = hartree_self_energy_matrix(density, Vq0)
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_matrix(G, W, grid, backend=backend)
    err = _residual_error(
        sigma_h_out - sigma_h, sigma_gw_out - sigma_gw
    )
    converged = bool(err < opts.tol)

    if opts.verbose and opts.target_filling is not None:
        print(
            f"SC-GW strict mu refine: mu={mu:.10f}, "
            f"n={np.sum(density):.10f}, mu_eval={mu_neval_final}, "
            f"mu_tol={opts.mu_tol:.1e}, residual={err:.3e}"
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
        mixing_method="anderson",
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
