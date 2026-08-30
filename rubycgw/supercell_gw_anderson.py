"""Adaptive Anderson acceleration for the 18-site period-three Ruby GW solver.

This module keeps the same GW fixed-point map as :mod:`supercell_gw_fast`, but
uses a safeguarded Type-II Anderson update after a short linear warm-up.  The
algorithm is deliberately conservative:

* the first few steps use ordinary damped fixed-point iteration;
* Anderson coefficients are obtained from a block-scaled residual metric so the
  static Hartree block and the much larger dynamic GW block are both visible;
* excessive residual growth clears the history and triggers a few recovery
  linear steps with reduced damping;
* an over-large Anderson proposal is rejected in favour of a linear step.

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
    _solve_mu_matrix_fast,
    density_from_G_cached,
)


@dataclass(frozen=True)
class AndersonOptions:
    """Numerical controls for safeguarded Type-II Anderson acceleration."""

    history: int = 6
    start: int = 8
    beta: float = 0.70
    beta_min: float = 0.15
    beta_max: float = 0.90
    regularization: float = 1e-8
    growth_factor: float = 1.20
    growth_patience: int = 3
    recovery_steps: int = 4
    step_cap: float = 5.0
    scale_floor: float = 1e-4


def _validate_anderson_options(aopts: AndersonOptions) -> None:
    if aopts.history < 2:
        raise ValueError("Anderson history must be at least 2")
    if aopts.start < 1:
        raise ValueError("Anderson start must be at least 1")
    if not (0.0 < aopts.beta_min <= aopts.beta <= aopts.beta_max <= 1.0):
        raise ValueError("Require 0 < beta_min <= beta <= beta_max <= 1")
    if aopts.regularization < 0.0:
        raise ValueError("Anderson regularization must be non-negative")
    if aopts.growth_factor <= 1.0:
        raise ValueError("Anderson growth_factor must exceed 1")
    if aopts.growth_patience < 1 or aopts.recovery_steps < 0:
        raise ValueError("Invalid Anderson growth/recovery settings")
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
    """Current scales used only in the Anderson least-squares metric."""
    sh = max(_rms(sigma_h), _rms(sigma_h_out), float(floor))
    sg = max(_rms(sigma_gw), _rms(sigma_gw_out), float(floor))
    return sh, sg


def _metric_vector(h: np.ndarray, gw: np.ndarray, sh: float, sg: float) -> np.ndarray:
    """Flatten two blocks with equal block weight and dynamic amplitude scaling."""
    vh = np.asarray(h, dtype=complex).reshape(-1) / (sh * np.sqrt(max(h.size, 1)))
    vg = np.asarray(gw, dtype=complex).reshape(-1) / (sg * np.sqrt(max(gw.size, 1)))
    return np.concatenate([vh, vg])


def _metric_norm(h: np.ndarray, gw: np.ndarray, sh: float, sg: float) -> float:
    v = _metric_vector(h, gw, sh, sg)
    return float(np.linalg.norm(v))


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
    """Return a safeguarded Type-II Anderson update.

    ``history`` contains consecutive ``(X_H, X_GW, R_H, R_GW)`` states and must
    already include the current state as its last entry.
    """
    linear_h = beta * res_h
    linear_gw = beta * res_gw
    if len(history) < 2:
        return sigma_h + linear_h, sigma_gw + linear_gw, False

    dR_cols = []
    dX_h = []
    dX_gw = []
    dR_h = []
    dR_gw = []
    for old, new in zip(history[:-1], history[1:]):
        xh0, xg0, rh0, rg0 = old
        xh1, xg1, rh1, rg1 = new
        dxh = xh1 - xh0
        dxg = xg1 - xg0
        drh = rh1 - rh0
        drg = rg1 - rg0
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
        A_aug = A
        rhs_aug = rhs

    try:
        gamma = np.linalg.lstsq(A_aug, rhs_aug, rcond=None)[0]
    except np.linalg.LinAlgError:
        return sigma_h + linear_h, sigma_gw + linear_gw, False

    step_h = np.array(linear_h, copy=True)
    step_gw = np.array(linear_gw, copy=True)
    for c, dxh, dxg, drh, drg in zip(gamma, dX_h, dX_gw, dR_h, dR_gw):
        step_h -= c * (dxh + beta * drh)
        step_gw -= c * (dxg + beta * drg)

    if not (np.all(np.isfinite(step_h)) and np.all(np.isfinite(step_gw))):
        return sigma_h + linear_h, sigma_gw + linear_gw, False

    linear_norm = _metric_norm(linear_h, linear_gw, sh, sg)
    step_norm = _metric_norm(step_h, step_gw, sh, sg)
    if step_norm > float(step_cap) * max(linear_norm, 1e-16):
        return sigma_h + linear_h, sigma_gw + linear_gw, False

    return sigma_h + step_h, sigma_gw + step_gw, True


def solve_matrix_gw_anderson(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    initial: GWResult | None = None,
    anderson: AndersonOptions = AndersonOptions(),
) -> GWResult:
    """Solve matrix-valued periodic GW with safeguarded adaptive Anderson mixing."""
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
        sigma_gw = np.zeros((grid.nf, grid.nk1, grid.nk2, norb, norb), dtype=complex)
        mu = float(opts.mu)

    if opts.target_filling is None:
        G = dyson_from_sigma_matrix(h0, grid, mu, sigma_h, sigma_gw)
        tail_cache = _build_tail_cache(h0, sigma_h)
        mu_neval = 0
    else:
        mu, G, tail_cache, mu_neval = _solve_mu_matrix_fast(
            h0, sigma_h, sigma_gw, grid, float(opts.target_filling),
            mu, opts.mu_tol, opts.mu_max_iter,
        )

    W = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    P = np.zeros_like(W)

    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    beta = float(anderson.beta)
    prev_err = float("inf")
    growth_count = 0
    recovery_remaining = 0
    err = float("inf")
    converged = False
    it = 0

    for it in range(1, opts.max_iter + 1):
        density = density_from_G_cached(G, grid, mu, tail_cache)
        sigma_h_out = hartree_self_energy_matrix(density, Vq0)
        P = compute_polarization_matrix(G, grid, backend=backend)
        W = compute_screened_interaction_matrix(P, Vq)
        sigma_gw_out = compute_sigma_gw_matrix(G, W, grid, backend=backend)

        res_h = sigma_h_out - sigma_h
        res_gw = sigma_gw_out - sigma_gw
        err = _residual_error(res_h, res_gw)

        if np.isfinite(prev_err):
            if err > float(anderson.growth_factor) * prev_err:
                growth_count += 1
            else:
                growth_count = 0

            if err < 0.70 * prev_err:
                beta = min(float(anderson.beta_max), 1.08 * beta)
            elif err > prev_err:
                beta = max(float(anderson.beta_min), 0.70 * beta)

        if growth_count >= int(anderson.growth_patience):
            history.clear()
            recovery_remaining = int(anderson.recovery_steps)
            beta = max(float(anderson.beta_min), 0.5 * beta)
            growth_count = 0

        if opts.verbose:
            phase = "recovery" if recovery_remaining > 0 else (
                "warmup" if it < int(anderson.start) else "anderson"
            )
            print(
                f"SC-GW iter {it:4d}: residual={err:.3e}, mu={mu:.10f}, "
                f"n={np.sum(density):.10f}, mu_eval={mu_neval}, "
                f"mixer={phase}, beta={beta:.3f}, backend={backend}"
            )

        if err < opts.tol:
            converged = True
            break

        history.append((
            np.array(sigma_h, copy=True),
            np.array(sigma_gw, copy=True),
            np.array(res_h, copy=True),
            np.array(res_gw, copy=True),
        ))
        keep_states = max(int(anderson.history) + 1, 2)
        if len(history) > keep_states:
            del history[:-keep_states]

        sh, sg = _block_scales(
            sigma_h, sigma_gw, sigma_h_out, sigma_gw_out, anderson.scale_floor
        )

        use_anderson = (
            recovery_remaining <= 0
            and it >= int(anderson.start)
            and len(history) >= 2
        )
        accepted_anderson = False
        if use_anderson:
            sigma_h_next, sigma_gw_next, accepted_anderson = _anderson_type2_step(
                sigma_h, sigma_gw, res_h, res_gw,
                history, beta, anderson.regularization,
                sh, sg, anderson.step_cap,
            )
            if not accepted_anderson:
                history.clear()
                recovery_remaining = max(int(anderson.recovery_steps), 2)
        else:
            sigma_h_next = sigma_h + beta * res_h
            sigma_gw_next = sigma_gw + beta * res_gw

        if recovery_remaining > 0:
            recovery_remaining -= 1

        if opts.target_filling is None:
            Gnext = dyson_from_sigma_matrix(h0, grid, mu, sigma_h_next, sigma_gw_next)
            tail_cache_next = _build_tail_cache(h0, sigma_h_next)
            mu_neval_next = 0
        else:
            mu, Gnext, tail_cache_next, mu_neval_next = _solve_mu_matrix_fast(
                h0, sigma_h_next, sigma_gw_next, grid, float(opts.target_filling),
                mu, opts.mu_tol, opts.mu_max_iter,
            )

        sigma_h = sigma_h_next
        sigma_gw = sigma_gw_next
        G = Gnext
        tail_cache = tail_cache_next
        mu_neval = mu_neval_next
        prev_err = float(err)

    density = density_from_G_cached(G, grid, mu, tail_cache)
    sigma_h_out = hartree_self_energy_matrix(density, Vq0)
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_matrix(G, W, grid, backend=backend)
    err = _residual_error(sigma_h_out - sigma_h, sigma_gw_out - sigma_gw)
    converged = bool(err < opts.tol)

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
    """Solve the 18-site Q=(1/3,1/3) supercell using adaptive Anderson mixing."""
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=source_strength)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    if h0.shape[-1] != NSUP:
        raise RuntimeError("unexpected supercell matrix dimension")
    return solve_matrix_gw_anderson(
        h0, Vq, grid, opts=opts, initial=initial, anderson=anderson
    )
