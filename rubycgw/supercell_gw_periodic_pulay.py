"""GW-only periodic Pulay solver for the 18-site Ruby supercell.

The physical GW equations are unchanged.  This module only changes the
numerical fixed-point path.  It is designed for the long oscillatory cycles
seen when ordinary positive linear mixing is already very small but the
self-energy residual still does not contract.

Strategy
--------
* cold starts use the analytic uniform Hartree field implied by the filling;
* Hartree is always mixed separately with a fixed, comparatively large step;
* dynamic Sigma_GW uses a modest linear step on ordinary iterations;
* every few eligible iterations, a small-history Pulay/DIIS extrapolation is
  applied to Sigma_GW only;
* the DIIS Gram matrix is normalized before regularization, so Pulay remains
  effective when the raw GW residual has already fallen to 1e-5 and below;
* optional residual diagnostics decompose a plateau into scalar-shift,
  low-frequency, frequency-edge and maximum-element contributions;
* a bad Pulay pulse is detected on the next (single) GW-map evaluation, its
  history is discarded, and a few cheap recovery-linear iterations are used;
* there is no rollback and no multi-trial line search: every outer iteration
  evaluates the expensive G -> P -> W -> Sigma_GW map exactly once;
* the fixed-filling chemical potential keeps using the cached-tail safeguarded
  Newton solver, with a strict refinement before returning.
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
    """Compatibility controls plus defaults for GW-only periodic Pulay.

    The historical Anderson fields are retained because ``run_supercell_gw.py``
    already constructs this object from those command-line options.  The active
    periodic-Pulay controls are the fields at the end of the dataclass.
    """

    history: int = 6
    start: int = 8
    warmup_beta: float = 0.20
    beta: float = 0.50
    beta_min: float = 0.04
    beta_max: float = 0.30
    regularization: float = 1e-8
    enter_residual: float = 1e-2
    enter_ratio: float = 0.50
    stable_steps: int = 5
    reject_factor: float = 2.0
    recovery_steps: int = 3
    step_cap: float = 5.0
    scale_floor: float = 1e-4
    growth_factor: float = 1.20
    growth_patience: int = 3

    # Active block/periodic-Pulay controls.
    hartree_beta: float = 0.30
    gw_beta: float = 0.10
    recovery_gw_beta: float = 0.05
    pulay_period: int = 3
    pulay_enter_h: float = 0.10
    pulay_enter_gw: float = 0.50
    pulay_min_start: int = 12
    spike_factor: float = 3.0
    diagnostic_interval: int = 10

    # Zero-extra-map residual diagnostics.  These are deliberately disabled by
    # default because they allocate a few temporary arrays and produce verbose
    # output, but they do not evaluate G -> P -> W -> Sigma_GW again.
    residual_diagnostics: bool = False
    residual_diagnostic_every: int = 1


def _validate(a: AndersonOptions) -> None:
    if a.history < 2:
        raise ValueError("history must be at least 2")
    if a.start < 1 or a.pulay_min_start < 1:
        raise ValueError("Pulay start must be positive")
    if not (0.0 < a.hartree_beta <= 1.0):
        raise ValueError("hartree_beta must lie in (0,1]")
    if not (0.0 < a.gw_beta <= 1.0):
        raise ValueError("gw_beta must lie in (0,1]")
    if not (0.0 < a.recovery_gw_beta <= a.gw_beta):
        raise ValueError("Require 0 < recovery_gw_beta <= gw_beta")
    if a.pulay_period < 1:
        raise ValueError("pulay_period must be positive")
    if a.pulay_enter_h <= 0.0 or a.pulay_enter_gw <= 0.0:
        raise ValueError("Pulay entry residuals must be positive")
    if not (0.0 < a.beta <= 1.0):
        raise ValueError("Pulay damping beta must lie in (0,1]")
    if a.regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if a.reject_factor <= 1.0 or a.spike_factor <= 1.0:
        raise ValueError("reject/spike factors must exceed 1")
    if a.recovery_steps < 0:
        raise ValueError("recovery_steps must be non-negative")
    if a.step_cap <= 1.0:
        raise ValueError("step_cap must exceed 1")
    if a.scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive")
    if a.diagnostic_interval < 1:
        raise ValueError("diagnostic_interval must be positive")
    if int(a.residual_diagnostic_every) < 1:
        raise ValueError("residual_diagnostic_every must be positive")


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def _uniform_hartree_seed(
    Vq0: np.ndarray,
    norb: int,
    target_filling: float | None,
) -> np.ndarray:
    """Uniform-density Hartree seed; at half filling it is exactly 2 V I_18."""
    if target_filling is None:
        return np.zeros((norb, norb), dtype=complex)
    nbar = float(target_filling) / float(norb)
    density = np.full(norb, nbar, dtype=float)
    return hartree_self_energy_matrix(density, Vq0)


def _gw_residual_inner(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.vdot(a, b).real / max(a.size, 1))


def _gw_pulay_coefficients(
    history: list[tuple[np.ndarray, np.ndarray]],
    regularization: float,
) -> np.ndarray:
    """DIIS coefficients for dynamic-GW residuals with sum(c)=1.

    Each history entry is ``(Sigma_GW_out, R_GW)``.  The coefficients minimize
    the norm of the residual combination; the extrapolated object is the fixed-
    point output ``Sigma_GW_out`` rather than the input iterate.

    The residual Gram block is normalized by its own diagonal scale before the
    Tikhonov regularization is added.  This is important near convergence:
    without the normalization a residual of order 1e-5 gives a Gram matrix of
    order 1e-10, while an absolute 1e-8 regularizer overwhelms the physical
    secant information and turns DIIS into little more than averaging.
    """
    m = len(history)
    if m < 2:
        raise ValueError("Pulay history needs at least two states")

    gram = np.zeros((m, m), dtype=float)
    for i in range(m):
        _, ri = history[i]
        for j in range(i, m):
            _, rj = history[j]
            val = _gw_residual_inner(ri, rj)
            gram[i, j] = val
            gram[j, i] = val

    diag_scale = float(np.max(np.abs(np.diag(gram))))
    if not np.isfinite(diag_scale) or diag_scale <= np.finfo(float).tiny:
        return np.full(m, 1.0 / float(m), dtype=float)

    gram /= diag_scale
    gram += float(regularization) * np.eye(m)

    B = np.zeros((m + 1, m + 1), dtype=float)
    B[:m, :m] = gram
    B[:m, m] = 1.0
    B[m, :m] = 1.0
    rhs = np.zeros(m + 1, dtype=float)
    rhs[m] = 1.0
    try:
        sol = np.linalg.solve(B, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(B, rhs, rcond=None)[0]
    return sol[:m]


def _gw_only_pulay_step(
    sigma_gw: np.ndarray,
    res_gw: np.ndarray,
    history: list[tuple[np.ndarray, np.ndarray]],
    damping: float,
    regularization: float,
    step_cap: float,
    linear_beta: float,
    scale_floor: float,
) -> tuple[np.ndarray, bool]:
    """Return one damped GW-only DIIS proposal without evaluating another map."""
    if len(history) < 2:
        return sigma_gw + float(linear_beta) * res_gw, False

    coeff = _gw_pulay_coefficients(history, regularization)
    gw_diis = np.zeros_like(sigma_gw)
    for c, (gw_out, _) in zip(coeff, history):
        gw_diis += c * gw_out

    step = float(damping) * (gw_diis - sigma_gw)
    if not np.all(np.isfinite(step)):
        return sigma_gw + float(linear_beta) * res_gw, False

    linear_rms = float(linear_beta) * _rms(res_gw)
    state_rms = max(_rms(sigma_gw), float(scale_floor))
    reference = max(linear_rms, 0.02 * state_rms, float(scale_floor))
    if _rms(step) > float(step_cap) * reference:
        return sigma_gw + float(linear_beta) * res_gw, False

    return sigma_gw + step, True


def _screening_min(P: np.ndarray, Vq: np.ndarray) -> float:
    norb = int(P.shape[-1])
    eye = np.eye(norb, dtype=complex)
    lhs = eye[None, None, None, :, :] - np.matmul(Vq[None, :, :, :, :], P)
    vals = np.linalg.svd(lhs, compute_uv=False)
    return float(np.min(vals[..., -1]))


def _gw_residual_mode_diagnostics(
    res_gw: np.ndarray,
    grid: MatsubaraGrid,
) -> dict:
    """Decompose one raw GW residual without changing the fixed-point map.

    ``scalar_shift`` is the orthogonal projection onto a real, frequency- and
    momentum-independent orbital identity.  ``projected_max`` removes exactly
    that one direction.  ``lowfreq_max`` uses the four Matsubara slices closest
    to zero, while ``edge_max`` uses the four slices with largest |omega|.
    """
    res = np.asarray(res_gw, dtype=complex)
    if res.ndim != 5:
        raise ValueError("Expected res_gw with shape (nf,nk1,nk2,norb,norb)")
    nf, nk1, nk2, norb, norb2 = res.shape
    if norb != norb2:
        raise ValueError("GW residual matrices must be square")

    diag = np.diagonal(res, axis1=-2, axis2=-1)
    scalar_shift = float(np.mean(diag.real))
    eye = np.eye(norb, dtype=complex)
    projected = res - scalar_shift * eye[None, None, None, :, :]

    raw_norm2 = float(np.vdot(res, res).real)
    scalar_norm2 = float((scalar_shift ** 2) * nf * nk1 * nk2 * norb)
    scalar_fraction = 0.0 if raw_norm2 <= 0.0 else scalar_norm2 / raw_norm2
    scalar_fraction = float(np.clip(scalar_fraction, 0.0, 1.0))

    freq_max = np.max(np.abs(res), axis=(1, 2, 3, 4))
    nsel = min(4, nf)
    low_idx = np.argsort(np.abs(grid.omega))[:nsel]
    edge_idx = np.argsort(np.abs(grid.omega))[-nsel:]

    flat_index = int(np.argmax(np.abs(res)))
    imax = np.unravel_index(flat_index, res.shape)
    iw, ik1, ik2, ia, ib = [int(x) for x in imax]
    kmesh = grid.kmesh()
    kval = kmesh[ik1, ik2]
    value = complex(res[imax])

    return {
        "raw_max": float(np.max(np.abs(res))),
        "projected_max": float(np.max(np.abs(projected))),
        "scalar_shift": scalar_shift,
        "scalar_fraction": scalar_fraction,
        "lowfreq_max": float(np.max(freq_max[low_idx])),
        "edge_max": float(np.max(freq_max[edge_idx])),
        "max_iw": iw,
        "max_omega": float(grid.omega[iw]),
        "max_k1_index": ik1,
        "max_k2_index": ik2,
        "max_k1": float(kval[0]),
        "max_k2": float(kval[1]),
        "max_a": ia,
        "max_b": ib,
        "max_value_re": float(value.real),
        "max_value_im": float(value.imag),
    }


def solve_matrix_gw_anderson(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    initial: GWResult | None = None,
    anderson: AndersonOptions = AndersonOptions(),
) -> GWResult:
    """Solve matrix GW with GW-only periodic Pulay and one map per iteration."""
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

    history: list[tuple[np.ndarray, np.ndarray]] = []
    pulay_start = max(int(anderson.start), int(anderson.pulay_min_start))
    eligible_count = 0
    recovery_remaining = 0
    last_step_kind = "initial"
    prev_err = float("inf")
    prev_err_gw = float("inf")
    last_smin = float("nan")
    mu_tol_used = initial_mu_tol
    previous_dyson_effective = None

    W = np.zeros((grid.nb, grid.nk1, grid.nk2, norb, norb), dtype=complex)
    P = np.zeros_like(W)
    density = np.zeros(norb, dtype=float)
    err = err_h = err_gw = float("inf")
    converged = False
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

        current_dyson_effective = None
        dyson_drift_max = float("nan")
        if anderson.residual_diagnostics:
            eye = np.eye(norb, dtype=complex)
            current_dyson_effective = (
                float(mu) * eye[None, None, None, :, :]
                - sigma_h[None, None, None, :, :]
                - sigma_gw
            )
            if previous_dyson_effective is not None:
                dyson_drift_max = float(
                    np.max(np.abs(current_dyson_effective - previous_dyson_effective))
                )

        bad_pulay = bool(
            last_step_kind == "gw-pulay"
            and np.isfinite(prev_err_gw)
            and err_gw > float(anderson.reject_factor) * prev_err_gw
        )
        huge_spike = bool(
            np.isfinite(prev_err_gw)
            and err_gw > float(anderson.spike_factor) * prev_err_gw
        )
        if bad_pulay or huge_spike:
            history.clear()
            eligible_count = 0
            recovery_remaining = max(
                recovery_remaining, int(anderson.recovery_steps)
            )

        if opts.verbose and (
            it == 1 or it % int(anderson.diagnostic_interval) == 0
        ):
            last_smin = _screening_min(P, Vq)

        if opts.verbose:
            smin_text = f"{last_smin:.3e}" if np.isfinite(last_smin) else "--"
            print(
                f"SC-GW iter {it:4d}: residual={err:.3e}, "
                f"rH={err_h:.3e}, rGW={err_gw:.3e}, smin={smin_text}, "
                f"mu={mu:.10f}, n={np.sum(density):.10f}, "
                f"mu_eval={mu_neval}, mu_tol={mu_tol_used:.1e}, "
                f"mixer={last_step_kind}, alphaH={anderson.hartree_beta:.3f}, "
                f"alphaGW={(anderson.recovery_gw_beta if recovery_remaining > 0 else anderson.gw_beta):.3f}, "
                f"backend={backend}"
            )
            if bad_pulay:
                print(
                    f"  reset GW-Pulay history: rGW grew from "
                    f"{prev_err_gw:.3e} to {err_gw:.3e}"
                )
            elif huge_spike:
                print(
                    f"  GW residual spike: {prev_err_gw:.3e} -> {err_gw:.3e}; "
                    f"enter recovery-linear"
                )

        if (
            anderson.residual_diagnostics
            and (it == 1 or it % int(anderson.residual_diagnostic_every) == 0)
        ):
            diag = _gw_residual_mode_diagnostics(res_gw, grid)
            ddyson = (
                f"{dyson_drift_max:.3e}"
                if np.isfinite(dyson_drift_max)
                else "--"
            )
            print(
                "  GW-resdiag: "
                f"raw={diag['raw_max']:.3e}, "
                f"nonscalar={diag['projected_max']:.3e}, "
                f"scalar_shift={diag['scalar_shift']:+.3e}, "
                f"scalar_fraction={diag['scalar_fraction']:.3f}, "
                f"lowfreq={diag['lowfreq_max']:.3e}, "
                f"edge={diag['edge_max']:.3e}, dDyson={ddyson}"
            )
            print(
                "    max@ "
                f"iw={diag['max_iw']} omega={diag['max_omega']:+.6e}, "
                f"kidx=({diag['max_k1_index']},{diag['max_k2_index']}), "
                f"k=({diag['max_k1']:.6f},{diag['max_k2']:.6f}), "
                f"orb=({diag['max_a']},{diag['max_b']}), "
                f"R={diag['max_value_re']:+.3e}{diag['max_value_im']:+.3e}i"
            )

        if err < opts.tol:
            converged = True
            break

        history.append(
            (np.array(sigma_gw_out, copy=True), np.array(res_gw, copy=True))
        )
        keep = max(int(anderson.history), 2)
        if len(history) > keep:
            del history[:-keep]

        sigma_h_next = sigma_h + float(anderson.hartree_beta) * res_h

        eligible = bool(
            recovery_remaining <= 0
            and it >= pulay_start
            and err_h < float(anderson.pulay_enter_h)
            and err_gw < float(anderson.pulay_enter_gw)
            and len(history) >= 2
        )
        if eligible:
            eligible_count += 1
        else:
            eligible_count = 0

        do_pulay = bool(
            eligible
            and eligible_count >= int(anderson.pulay_period)
            and eligible_count % int(anderson.pulay_period) == 0
        )

        if recovery_remaining > 0:
            linear_beta = float(anderson.recovery_gw_beta)
            sigma_gw_next = sigma_gw + linear_beta * res_gw
            next_step_kind = "recovery-linear"
            recovery_remaining -= 1
        elif do_pulay:
            pulay_beta = min(0.50, max(0.20, float(anderson.beta)))
            sigma_gw_next, valid = _gw_only_pulay_step(
                sigma_gw,
                res_gw,
                history,
                damping=pulay_beta,
                regularization=float(anderson.regularization),
                step_cap=float(anderson.step_cap),
                linear_beta=float(anderson.gw_beta),
                scale_floor=float(anderson.scale_floor),
            )
            if valid:
                next_step_kind = "gw-pulay"
            else:
                sigma_gw_next = sigma_gw + float(anderson.gw_beta) * res_gw
                next_step_kind = "block-linear"
                history[:] = history[-1:]
                eligible_count = 0
        else:
            sigma_gw_next = sigma_gw + float(anderson.gw_beta) * res_gw
            next_step_kind = "block-linear"

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

        if anderson.residual_diagnostics:
            previous_dyson_effective = current_dyson_effective
        prev_err = float(err)
        prev_err_gw = float(err_gw)
        sigma_h = sigma_h_next
        sigma_gw = sigma_gw_next
        mu = float(mu_next)
        G = Gnext
        cache = cache_next
        mu_neval = int(mu_neval_next)
        last_step_kind = next_step_kind

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

    if anderson.residual_diagnostics:
        diag = _gw_residual_mode_diagnostics(res_gw, grid)
        print(
            "  GW-resdiag-final: "
            f"raw={diag['raw_max']:.3e}, nonscalar={diag['projected_max']:.3e}, "
            f"scalar_shift={diag['scalar_shift']:+.3e}, "
            f"scalar_fraction={diag['scalar_fraction']:.3f}, "
            f"lowfreq={diag['lowfreq_max']:.3e}, edge={diag['edge_max']:.3e}"
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
        mixing_method="gw-periodic-pulay",
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
