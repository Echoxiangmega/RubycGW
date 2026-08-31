"""Bootstrap wrapper for the 18-site Ruby supercell GW solver.

The expensive fixed-point map is still evaluated exactly once per outer
iteration by :mod:`rubycgw.supercell_gw_periodic_pulay`.  This wrapper addresses
a different problem: starting the first strong-coupling point (historically
V=0.7) from ``Sigma_GW=0`` drives the trajectory through very poorly screened
intermediate states even though the Hartree field is initialized analytically.

For a genuine cold start at zero source, we therefore establish the normal
branch first at a few weak/intermediate couplings and use each converged state
as the seed for the next one.  Once the driver already supplies a previous-V
state or a source-removal state, no internal bootstrap is performed.

A second, narrow numerical safeguard handles states that are already within a
few requested tolerances of convergence.  The periodic solver always performs a
strict fixed-filling chemical-potential refinement before returning.  That
refinement can move a residual that was just below ``tol`` to just above it,
for example 9.8e-9 -> 1.01e-8.  Such a state is not accepted as converged, but
it is also too close to the fixed point to justify abandoning a long run.  The
wrapper therefore continues from that strict-mu state for a short polishing
solve with a stricter internal target.  The final state is accepted only if its
actual residual is below the user's original tolerance.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from .gw import GWOptions, GWResult
from .grids import MatsubaraGrid
from .model import RubyParameters
from .supercell_gw_periodic_pulay import (
    AndersonOptions,
    solve_matrix_gw_anderson as _solve_matrix_periodic,
    solve_supercell_gw_anderson as _solve_supercell_periodic,
)


# These values are deliberately below the charge-order source onset.  They are
# not extra physics approximations: they are only a numerical continuation path
# to the same final GW fixed point.
_BOOTSTRAP_V = (0.10, 0.25, 0.50, 0.60, 0.65)

# A returned state in this narrow window is close enough that a short continuation
# is much cheaper and safer than handing it to the driver's generic fallback path.
_FINISH_WINDOW_FACTOR = 10.0
_FINISH_INTERNAL_TOL_FACTOR = 0.50
_FINISH_MAX_ITER = 60


def _safe_periodic_options(a: AndersonOptions) -> AndersonOptions:
    """Use conservative GW-only Pulay defaults without adding extra GW maps.

    The previous periodic-Pulay run showed two numerical pathologies:
    (i) a long history could contain non-eligible recovery states, and
    (ii) the DIIS pulse was too large in max norm even when its RMS norm looked
    harmless.  Keeping only three states means that, after three consecutive
    eligible iterations, the complete history is local to that smooth segment.
    A smaller damping/step cap makes the pulse an accelerator rather than a
    branch-jumping extrapolation.
    """

    return replace(
        a,
        history=min(int(a.history), 3),
        beta=min(float(a.beta), 0.25),
        reject_factor=min(float(a.reject_factor), 1.5),
        recovery_steps=max(int(a.recovery_steps), 5),
        step_cap=min(float(a.step_cap), 2.0),
        gw_beta=min(float(a.gw_beta), 0.08),
        recovery_gw_beta=min(float(a.recovery_gw_beta), 0.04),
        pulay_enter_gw=min(float(a.pulay_enter_gw), 0.25),
        pulay_period=max(int(a.pulay_period), 3),
    )


def _bootstrap_values(target_V: float) -> list[float]:
    """Weak-coupling continuation values strictly below ``target_V``."""

    target = float(target_V)
    return [v for v in _BOOTSTRAP_V if v < target - 1e-12]


def _finite_result(gw: GWResult | None) -> bool:
    if gw is None:
        return False
    return bool(
        np.isfinite(float(gw.final_error))
        and np.isfinite(float(gw.mu))
        and np.all(np.isfinite(gw.Sigma_H))
        and np.all(np.isfinite(gw.Sigma_GW))
    )


def _needs_finish_polish(gw: GWResult | None, requested_tol: float) -> bool:
    """Return True only for a finite non-converged state very near tolerance."""

    tol = float(requested_tol)
    if tol <= 0.0 or not _finite_result(gw) or bool(gw.converged):
        return False
    err = float(gw.final_error)
    return bool(err <= _FINISH_WINDOW_FACTOR * tol)


def _finish_options(opts: GWOptions) -> GWOptions:
    """Short stricter target used only to get safely below the user's tolerance."""

    internal_tol = max(
        np.finfo(float).tiny,
        _FINISH_INTERNAL_TOL_FACTOR * float(opts.tol),
    )
    return replace(
        opts,
        tol=internal_tol,
        max_iter=min(_FINISH_MAX_ITER, max(1, int(opts.max_iter))),
    )


def _choose_polished_result(
    base: GWResult,
    trial: GWResult | None,
    requested_tol: float,
) -> GWResult:
    """Keep the better finite state and judge convergence by the original tol."""

    total_iterations = int(base.iterations)
    if _finite_result(trial):
        total_iterations += int(trial.iterations)
        best = trial if float(trial.final_error) < float(base.final_error) else base
    else:
        best = base

    return replace(
        best,
        converged=bool(float(best.final_error) < float(requested_tol)),
        iterations=total_iterations,
    )


def _polish_matrix_if_needed(
    result: GWResult,
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions,
    safe: AndersonOptions,
) -> GWResult:
    if not _needs_finish_polish(result, opts.tol):
        return result

    finish_opts = _finish_options(opts)
    if opts.verbose:
        print(
            f"--- strict-tolerance polish: residual={result.final_error:.3e} is "
            f"within {_FINISH_WINDOW_FACTOR:g}x requested tol={opts.tol:.3e}; "
            f"continue from the strict-mu state with internal tol="
            f"{finish_opts.tol:.3e} for <= {finish_opts.max_iter} iterations ---"
        )

    trial = _solve_matrix_periodic(
        h0,
        Vq,
        grid,
        opts=finish_opts,
        initial=result,
        anderson=safe,
    )
    best = _choose_polished_result(result, trial, opts.tol)
    if opts.verbose:
        status = "accepted" if best.converged else "stopped"
        print(
            f"--- strict-tolerance polish {status}: residual={best.final_error:.3e}, "
            f"requested tol={opts.tol:.3e} ---"
        )
    return best


def _polish_supercell_if_needed(
    result: GWResult,
    params: RubyParameters,
    grid: MatsubaraGrid,
    opts: GWOptions,
    source_strength: float,
    safe: AndersonOptions,
) -> GWResult:
    if not _needs_finish_polish(result, opts.tol):
        return result

    finish_opts = _finish_options(opts)
    if opts.verbose:
        print(
            f"--- strict-tolerance polish: residual={result.final_error:.3e} is "
            f"within {_FINISH_WINDOW_FACTOR:g}x requested tol={opts.tol:.3e}; "
            f"continue from the strict-mu state with internal tol="
            f"{finish_opts.tol:.3e} for <= {finish_opts.max_iter} iterations ---"
        )

    trial = _solve_supercell_periodic(
        params,
        grid,
        opts=finish_opts,
        source_strength=source_strength,
        initial=result,
        anderson=safe,
    )
    best = _choose_polished_result(result, trial, opts.tol)
    if opts.verbose:
        status = "accepted" if best.converged else "stopped"
        print(
            f"--- strict-tolerance polish {status}: residual={best.final_error:.3e}, "
            f"requested tol={opts.tol:.3e} ---"
        )
    return best


def solve_matrix_gw_anderson(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    initial: GWResult | None = None,
    anderson: AndersonOptions = AndersonOptions(),
) -> GWResult:
    """Matrix-level entry point with conservative settings and final polishing."""

    safe = _safe_periodic_options(anderson)
    result = _solve_matrix_periodic(
        h0,
        Vq,
        grid,
        opts=opts,
        initial=initial,
        anderson=safe,
    )
    return _polish_matrix_if_needed(result, h0, Vq, grid, opts, safe)


def solve_supercell_gw_anderson(
    params: RubyParameters,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    source_strength: float = 0.0,
    initial: GWResult | None = None,
    anderson: AndersonOptions = AndersonOptions(),
) -> GWResult:
    """Solve supercell GW, internally bootstrapping only a true cold start.

    For the default first strong-coupling point, the path is approximately

        0.10 -> 0.25 -> 0.50 -> 0.60 -> 0.65 -> target V.

    If ``initial`` is already supplied (normal V continuation, source removal,
    checkpoint restart, or retry), the target point is solved directly.
    """

    safe = _safe_periodic_options(anderson)
    target_V = float(params.V)

    needs_bootstrap = bool(
        initial is None
        and abs(float(source_strength)) < 1e-15
        and target_V > _BOOTSTRAP_V[0] + 1e-12
    )
    if not needs_bootstrap:
        result = _solve_supercell_periodic(
            params,
            grid,
            opts=opts,
            source_strength=source_strength,
            initial=initial,
            anderson=safe,
        )
        return _polish_supercell_if_needed(
            result,
            params,
            grid,
            opts,
            source_strength,
            safe,
        )

    seed: GWResult | None = None
    # Intermediate bootstrap points need only the normal continuation tolerance;
    # the requested target is always recomputed afterwards with the original tol.
    bootstrap_tol = max(float(opts.tol), 1e-6)
    boot_opts = replace(opts, tol=bootstrap_tol)

    for Vboot in _bootstrap_values(target_V):
        if opts.verbose:
            print(
                f"--- internal weak-V bootstrap: V={Vboot:g} "
                f"(tol={bootstrap_tol:.1e}) ---"
            )
        pboot = replace(params, V=float(Vboot))
        trial = _solve_supercell_periodic(
            pboot,
            grid,
            opts=boot_opts,
            source_strength=0.0,
            initial=seed,
            anderson=safe,
        )
        if not (_finite_result(trial) and trial.converged):
            if opts.verbose:
                err = float(trial.final_error) if _finite_result(trial) else float("nan")
                print(
                    f"--- bootstrap V={Vboot:g} did not converge "
                    f"(residual={err:.3e}); stop bootstrap and use the last "
                    "converged seed ---"
                )
            break
        seed = trial

    if opts.verbose:
        label = "cold seed" if seed is None else f"bootstrap seed mu={seed.mu:.8f}"
        print(f"--- solve requested V={target_V:g} from {label} ---")

    result = _solve_supercell_periodic(
        params,
        grid,
        opts=opts,
        source_strength=source_strength,
        initial=seed,
        anderson=safe,
    )
    return _polish_supercell_if_needed(
        result,
        params,
        grid,
        opts,
        source_strength,
        safe,
    )


__all__ = [
    "AndersonOptions",
    "_bootstrap_values",
    "_safe_periodic_options",
    "_needs_finish_polish",
    "_choose_polished_result",
    "solve_matrix_gw_anderson",
    "solve_supercell_gw_anderson",
]
