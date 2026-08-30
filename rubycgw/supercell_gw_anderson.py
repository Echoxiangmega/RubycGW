"""Adaptive Anderson acceleration for the 18-site period-three Ruby GW solver.

The physical GW fixed-point map is unchanged.  This module only changes the
numerical path used to solve ``Sigma = F[Sigma]``.  The current strategy is:

* find a reasonably contractive basin with damped linear mixing;
* use Type-II Anderson acceleration once the basin is reached;
* backtrack along an Anderson direction instead of discarding the whole
  quasi-Newton history when the full proposal is too aggressive;
* apply the same trust-region idea to linear/recovery steps so a single update
  cannot jump deep into a screening-pole region;
* clear Anderson history only after repeated genuine direction failures.

Verbose output separates Hartree and dynamic-GW residuals and reports the
instantaneous minimum singular value of ``I - V P``.  This makes it possible to
distinguish a slow numerical mode from an iterate that is approaching a
screening pole.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    """Controls for conservative Type-II Anderson with backtracking.

    ``growth_factor`` and ``growth_patience`` are retained for compatibility
    with older driver command-line options.  The active safeguards are the
    line-search and repeated-direction-failure controls below.
    """

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
    recovery_steps: int = 3
    step_cap: float = 3.0
    scale_floor: float = 1e-4
    growth_factor: float = 1.20
    growth_patience: int = 3

    # Trust-region / line-search controls.
    line_search_steps: int = 5
    line_search_shrink: float = 0.50
    anderson_accept_factor: float = 1.000
    linear_accept_factor: float = 1.10
    direction_fail_patience: int = 2


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
    if not (aopts.beta_min <= aopts.beta <= 1.0):
        raise ValueError("Require beta_min <= beta <= 1")
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
    if aopts.line_search_steps < 1:
        raise ValueError("line_search_steps must be positive")
    if not (0.0 < aopts.line_search_shrink < 1.0):
        raise ValueError("line_search_shrink must lie in (0,1)")
    if aopts.anderson_accept_factor <= 0.0:
        raise ValueError("anderson_accept_factor must be positive")
    if aopts.linear_accept_factor < 1.0:
        raise ValueError("linear_accept_factor must be at least 1")
    if aopts.direction_fail_patience < 1:
        raise ValueError("direction_fail_patience must be positive")


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
    """Flatten Hartree/GW blocks with equal block weight and amplitude scaling."""
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
    """Return an Anderson proposal; global acceptance is handled by line search."""
    linear_h = beta * res_h
    linear_gw = beta * res_gw
    if len(history) < 2:
        return sigma_h + linear_h, sigma_gw + linear_gw, False

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


@dataclass(frozen=True)
class _StateEval:
    sigma_h: np.ndarray
    sigma_gw: np.ndarray
    mu: float
    G: np.ndarray
    tail_cache: object
    mu_neval: int
    mu_tol: float
    density: np.ndarray
    P: np.ndarray
    W: np.ndarray
    sigma_h_out: np.ndarray
    sigma_gw_out: np.ndarray
    res_h: np.ndarray
    res_gw: np.ndarray
    err_h: float
    err_gw: float
    err: float


def _map_from_G(
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
    mu: float,
    G: np.ndarray,
    tail_cache,
    mu_neval: int,
    mu_tol: float,
    Vq0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    backend: str,
) -> _StateEval:
    density = density_from_G_cached(G, grid, mu, tail_cache)
    sigma_h_out = hartree_self_energy_matrix(density, Vq0)
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_matrix(G, W, grid, backend=backend)
    res_h = sigma_h_out - sigma_h
    res_gw = sigma_gw_out - sigma_gw
    err_h = float(np.max(np.abs(res_h)))
    err_gw = float(np.max(np.abs(res_gw)))
    err = _residual_error(res_h, res_gw)
    return _StateEval(
        sigma_h=np.asarray(sigma_h),
        sigma_gw=np.asarray(sigma_gw),
        mu=float(mu),
        G=G,
        tail_cache=tail_cache,
        mu_neval=int(mu_neval),
        mu_tol=float(mu_tol),
        density=density,
        P=P,
        W=W,
        sigma_h_out=sigma_h_out,
        sigma_gw_out=sigma_gw_out,
        res_h=res_h,
        res_gw=res_gw,
        err_h=err_h,
        err_gw=err_gw,
        err=float(err),
    )


def _evaluate_trial(
    h0: np.ndarray,
    Vq0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions,
    backend: str,
    sigma_h: np.ndarray,
    sigma_gw: np.ndarray,
    mu_guess: float,
    outer_residual: float,
) -> _StateEval:
    mu, G, cache, neval, mu_tol = _solve_next_G(
        h0,
        grid,
        opts,
        sigma_h,
        sigma_gw,
        mu_guess,
        outer_residual,
    )
    return _map_from_G(
        sigma_h,
        sigma_gw,
        mu,
        G,
        cache,
        neval,
        mu_tol,
        Vq0,
        Vq,
        grid,
        backend,
    )


def _screening_min_value(P: np.ndarray, Vq: np.ndarray) -> float:
    """Minimum singular value of I-VP, used only as a verbose diagnostic."""
    norb = int(P.shape[-1])
    eye = np.eye(norb, dtype=complex)
    lhs = eye[None, None, None, :, :] - np.matmul(
        Vq[None, :, :, :, :], P
    )
    svals = np.linalg.svd(lhs, compute_uv=False)
    return float(np.min(svals[..., -1]))


def _line_search(
    parent: _StateEval,
    direction_h: np.ndarray,
    direction_gw: np.ndarray,
    h0: np.ndarray,
    Vq0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions,
    backend: str,
    aopts: AndersonOptions,
    accept_factor: float,
) -> tuple[_StateEval, float, bool, int]:
    """Backtrack a proposed direction and reuse the accepted full GW map.

    The first trial whose raw fixed-point residual satisfies
    ``err_trial <= accept_factor * err_parent`` is accepted.  If none satisfies
    that condition, the least-bad tested state is returned with ``accepted=False``.
    """
    scale = 1.0
    best: _StateEval | None = None
    best_scale = scale
    total_mu_eval = 0
    ntrial = 0

    for _ in range(int(aopts.line_search_steps)):
        trial_h = parent.sigma_h + scale * direction_h
        trial_gw = parent.sigma_gw + scale * direction_gw
        trial = _evaluate_trial(
            h0,
            Vq0,
            Vq,
            grid,
            opts,
            backend,
            trial_h,
            trial_gw,
            parent.mu,
            parent.err,
        )
        ntrial += 1
        total_mu_eval += int(trial.mu_neval)
        trial = replace(trial, mu_neval=total_mu_eval)
        if best is None or trial.err < best.err:
            best = trial
            best_scale = scale
        if np.isfinite(trial.err) and trial.err <= float(accept_factor) * parent.err:
            return trial, scale, True, ntrial
        scale *= float(aopts.line_search_shrink)

    assert best is not None
    best = replace(best, mu_neval=total_mu_eval)
    return best, best_scale, False, ntrial


def solve_matrix_gw_anderson(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    initial: GWResult | None = None,
    anderson: AndersonOptions = AndersonOptions(),
) -> GWResult:
    """Solve matrix GW with basin finding and trust-region Anderson mixing."""
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

    state = _map_from_G(
        sigma_h,
        sigma_gw,
        mu,
        G,
        cache,
        mu_neval,
        initial_mu_tol,
        Vq0,
        Vq,
        grid,
        backend,
    )

    history: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    beta = float(anderson.warmup_beta)
    entered_anderson = False
    initial_err = float(state.err) if np.isfinite(state.err) else None
    prev_err = float("inf")
    stable_count = 0
    recovery_remaining = 0
    direction_failures = 0
    converged = False
    it = 0

    for it in range(1, opts.max_iter + 1):
        err = float(state.err)
        if np.isfinite(prev_err):
            if err <= 1.02 * prev_err:
                stable_count += 1
            else:
                stable_count = 0

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
            beta = min(float(anderson.beta), 0.30)
            history.clear()
            direction_failures = 0

        phase = (
            "recovery"
            if recovery_remaining > 0
            else ("anderson" if entered_anderson else "basin-linear")
        )
        if opts.verbose:
            smin_now = _screening_min_value(state.P, Vq)
            print(
                f"SC-GW iter {it:4d}: residual={state.err:.3e}, "
                f"rH={state.err_h:.3e}, rGW={state.err_gw:.3e}, "
                f"smin={smin_now:.3e}, mu={state.mu:.10f}, "
                f"n={np.sum(state.density):.10f}, mu_eval={state.mu_neval}, "
                f"mu_tol={state.mu_tol:.1e}, mixer={phase}, "
                f"beta={beta:.3f}, backend={backend}"
            )

        if state.err < opts.tol:
            converged = True
            break

        history.append(
            (
                np.array(state.sigma_h, copy=True),
                np.array(state.sigma_gw, copy=True),
                np.array(state.res_h, copy=True),
                np.array(state.res_gw, copy=True),
            )
        )
        keep_states = max(int(anderson.history) + 1, 2)
        if len(history) > keep_states:
            del history[:-keep_states]

        sh, sg = _block_scales(
            state.sigma_h,
            state.sigma_gw,
            state.sigma_h_out,
            state.sigma_gw_out,
            anderson.scale_floor,
        )

        use_anderson = bool(
            entered_anderson
            and recovery_remaining <= 0
            and len(history) >= 2
        )

        next_state: _StateEval | None = None
        accepted_anderson = False
        if use_anderson:
            prop_h, prop_gw, valid_direction = _anderson_type2_step(
                state.sigma_h,
                state.sigma_gw,
                state.res_h,
                state.res_gw,
                history,
                beta,
                anderson.regularization,
                sh,
                sg,
                min(float(anderson.step_cap), 3.0),
            )
            if valid_direction:
                dir_h = prop_h - state.sigma_h
                dir_gw = prop_gw - state.sigma_gw
                trial, scale, accepted, ntrial = _line_search(
                    state,
                    dir_h,
                    dir_gw,
                    h0,
                    Vq0,
                    Vq,
                    grid,
                    opts,
                    backend,
                    anderson,
                    anderson.anderson_accept_factor,
                )
                if accepted:
                    next_state = trial
                    accepted_anderson = True
                    direction_failures = 0
                    improvement = trial.err / max(state.err, 1e-30)
                    if scale < 0.999:
                        beta = max(float(anderson.beta_min), 0.85 * beta)
                    elif improvement < 0.90:
                        beta = min(float(anderson.beta_max), 1.05 * beta)
                    if opts.verbose and (scale < 0.999 or ntrial > 1):
                        print(
                            f"  Anderson line search: accepted scale={scale:.3f} "
                            f"after {ntrial} trial(s), residual={trial.err:.3e}"
                        )
                else:
                    direction_failures += 1
                    if opts.verbose:
                        print(
                            f"  Anderson direction failed to decrease residual "
                            f"after {ntrial} backtracking trial(s); "
                            f"best={trial.err:.3e}"
                        )
            else:
                direction_failures += 1
                if opts.verbose:
                    print("  Anderson proposal invalid/too large; use linear trust step")

        if next_state is None:
            # Linear/recovery direction also gets a trust-region check.  This is
            # what suppresses the occasional O(10)-O(100) residual spikes seen
            # in the old basin-linear iteration.
            linear_beta = beta
            if recovery_remaining > 0:
                linear_beta = min(
                    linear_beta,
                    max(float(anderson.beta_min), 0.15),
                )
            dir_h = linear_beta * state.res_h
            dir_gw = linear_beta * state.res_gw
            trial, scale, accepted, ntrial = _line_search(
                state,
                dir_h,
                dir_gw,
                h0,
                Vq0,
                Vq,
                grid,
                opts,
                backend,
                anderson,
                anderson.linear_accept_factor,
            )
            next_state = trial
            if scale < 0.999:
                beta = max(
                    float(anderson.beta_min),
                    beta * max(scale, float(anderson.line_search_shrink)),
                )
            if opts.verbose and (scale < 0.999 or not accepted):
                status = "accepted" if accepted else "best available"
                print(
                    f"  linear trust step: {status}, scale={scale:.3f}, "
                    f"trials={ntrial}, residual={trial.err:.3e}"
                )

        if use_anderson and not accepted_anderson:
            if direction_failures >= int(anderson.direction_fail_patience):
                # Only now is the quasi-Newton memory considered genuinely bad.
                history.clear()
                entered_anderson = False
                recovery_remaining = int(anderson.recovery_steps)
                stable_count = 0
                direction_failures = 0
                beta = max(
                    float(anderson.beta_min),
                    min(float(anderson.warmup_beta), beta),
                )
                if opts.verbose:
                    print("  clear Anderson history after repeated direction failures")
        elif accepted_anderson:
            recovery_remaining = 0

        if recovery_remaining > 0 and not accepted_anderson:
            recovery_remaining -= 1

        prev_err = float(state.err)
        state = next_state

    # The returned state is always refined to the strict fixed-filling tolerance,
    # then the physical GW map is re-evaluated on exactly that state.
    mu, G, cache, mu_neval_final = _strict_refine_fixed_filling(
        h0,
        state.sigma_h,
        state.sigma_gw,
        grid,
        opts.target_filling,
        state.mu,
        opts.mu_tol,
        opts.mu_max_iter,
    )
    final_state = _map_from_G(
        state.sigma_h,
        state.sigma_gw,
        mu,
        G,
        cache,
        mu_neval_final,
        opts.mu_tol,
        Vq0,
        Vq,
        grid,
        backend,
    )
    converged = bool(final_state.err < opts.tol)

    if opts.verbose and opts.target_filling is not None:
        print(
            f"SC-GW strict mu refine: mu={final_state.mu:.10f}, "
            f"n={np.sum(final_state.density):.10f}, "
            f"mu_eval={mu_neval_final}, mu_tol={opts.mu_tol:.1e}, "
            f"residual={final_state.err:.3e}, "
            f"rH={final_state.err_h:.3e}, rGW={final_state.err_gw:.3e}"
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
    ) = screening_soft_modes_matrix(final_state.P, Vq, grid)

    return GWResult(
        G=final_state.G,
        W=final_state.W,
        P=final_state.P,
        Sigma_H=final_state.sigma_h,
        Sigma_GW=final_state.sigma_gw,
        mu=final_state.mu,
        density=final_state.density,
        converged=converged,
        iterations=it,
        final_error=float(final_state.err),
        mixing_method="anderson-linesearch",
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
    """Solve the 18-site Q=(1/3,1/3) supercell with trust-region Anderson."""
    h0 = build_supercell_h0(
        grid.kmesh(), params, source_strength=source_strength
    )
    Vq = build_supercell_interaction(grid.qmesh(), params)
    if h0.shape[-1] != NSUP:
        raise RuntimeError("unexpected supercell matrix dimension")
    return solve_matrix_gw_anderson(
        h0, Vq, grid, opts=opts, initial=initial, anderson=anderson
    )
