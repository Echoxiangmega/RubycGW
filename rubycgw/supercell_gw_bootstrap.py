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


def solve_matrix_gw_anderson(
    h0: np.ndarray,
    Vq: np.ndarray,
    grid: MatsubaraGrid,
    opts: GWOptions = GWOptions(),
    initial: GWResult | None = None,
    anderson: AndersonOptions = AndersonOptions(),
) -> GWResult:
    """Matrix-level entry point with the conservative periodic-Pulay settings."""

    return _solve_matrix_periodic(
        h0,
        Vq,
        grid,
        opts=opts,
        initial=initial,
        anderson=_safe_periodic_options(anderson),
    )


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
        return _solve_supercell_periodic(
            params,
            grid,
            opts=opts,
            source_strength=source_strength,
            initial=initial,
            anderson=safe,
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

    return _solve_supercell_periodic(
        params,
        grid,
        opts=opts,
        source_strength=source_strength,
        initial=seed,
        anderson=safe,
    )


__all__ = [
    "AndersonOptions",
    "_bootstrap_values",
    "_safe_periodic_options",
    "solve_matrix_gw_anderson",
    "solve_supercell_gw_anderson",
]
