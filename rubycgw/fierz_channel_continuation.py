"""Adaptive interaction ramp for spinless Ruby same-torus Fierz GW.

A warm start at nearby V does *not* change the fixed-point problem:
only the converged static/dynamic self-energy and chemical potential are
reused. Polarization and W are always rebuilt at the new interaction.

A converged solution cannot be inferred from small step size or a declining
but nonzero residual. When the target cannot be reached, return the last
certified converged V and all failed-attempt diagnostics; never substitute
it for the requested final V.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from time import perf_counter

import numpy as np

from .fierz_channel_gw import (
    ChannelGWResult, build_channel_definition, solve_channel_gw_same_torus,
)
from .gw import GWOptions
from .grids import MatsubaraGrid


@dataclass
class GWRampStage:
    V: float
    previous_V: float
    step: float
    method: str
    mixing: float
    iterations: int
    residual: float
    screening_smin: float
    converged: bool
    wall_seconds: float
    error: str | None = None


@dataclass
class GWRampResult:
    state: ChannelGWResult | None
    reached_target: bool
    last_converged_V: float | None
    attempted: list[GWRampStage]
    target_V: float
    smallest_failed_step: float | None = None

    def diagnostics(self) -> dict:
        return {
            "target_V": self.target_V,
            "reached_target": self.reached_target,
            "last_converged_V": self.last_converged_V,
            "attempts": [asdict(x) for x in self.attempted],
            "smallest_failed_step": self.smallest_failed_step,
        }


def solve_channel_gw_v_ramp(
    h0: np.ndarray,
    interaction_pairs,
    grid: MatsubaraGrid,
    target_V: float,
    mode: str,
    opts: GWOptions,
    *,
    initial_steps: int = 10,
    min_step: float = 0.02,
    max_stages: int = 80,
    mixing_factors: tuple[float, ...] = (1.0, 0.5),
) -> GWRampResult:
    """Warm-start ramp with step bisection and strict physical residual tests.

    Starts from the exactly noninteracting channel GW solution at V=0.
    A failed attempt retries with smaller mixing before halving the V step.
    The next step is reset to the original scale after a successful step,
    so a long difficult interval is not permanently resolved too finely.
    """
    Vtarget = float(target_V)
    if not np.isfinite(Vtarget):
        raise ValueError("target V must be finite")
    if int(initial_steps) < 1 or int(max_stages) < 1:
        raise ValueError("initial_steps and max_stages must be positive")
    if not np.isfinite(min_step) or float(min_step) <= 0:
        raise ValueError("min_step must be positive")
    if not mixing_factors or any(not (0 < f <= 1) for f in mixing_factors):
        raise ValueError("mixing factors must lie in (0, 1]")

    pairs = tuple((int(i), int(j)) for i, j in interaction_pairs)
    norb = int(np.asarray(h0).shape[-1])
    records: list[GWRampStage] = []
    Vnow = 0.0
    seed = None
    worst_step = None
    tolV = 1e-12 * max(1.0, abs(Vtarget))

    if abs(Vtarget) <= tolV:
        definition = build_channel_definition(pairs, norb, 0.0, mode)
        seed = solve_channel_gw_same_torus(h0, definition, grid, opts=opts)
        return GWRampResult(seed, bool(seed.converged), 0.0 if seed.converged else None,
                            records, Vtarget)

    # The V=0 system provides a cheap exact starting point.
    base_def = build_channel_definition(pairs, norb, 0.0, mode)
    seed = solve_channel_gw_same_torus(h0, base_def, grid, opts=opts)
    if not seed.converged:
        return GWRampResult(None, False, None, records, Vtarget)

    direction = 1.0 if Vtarget > 0 else -1.0
    base_step = abs(Vtarget) / float(initial_steps)
    step = base_step
    while direction * (Vtarget - Vnow) > tolV and len(records) < int(max_stages):
        candidate = Vnow + direction * min(step, abs(Vtarget - Vnow))
        definition = build_channel_definition(pairs, norb, candidate, mode)
        success = None
        for factor in mixing_factors:
            if len(records) >= max_stages:
                break
            new_opts = replace(opts, mixing=float(opts.mixing) * float(factor))
            t0 = perf_counter()
            err_txt = None
            trial = None
            try:
                trial = solve_channel_gw_same_torus(
                    h0, definition, grid, opts=new_opts, initial=seed
                )
            except (np.linalg.LinAlgError, FloatingPointError, RuntimeError) as exc:
                err_txt = f"{type(exc).__name__}: {exc}"
            dt = perf_counter() - t0
            good = bool(trial is not None and trial.converged and
                        np.isfinite(trial.final_error) and
                        trial.final_error < new_opts.tol)
            records.append(GWRampStage(
                V=float(candidate), previous_V=float(Vnow),
                step=float(abs(candidate - Vnow)),
                method=str(new_opts.mixing_method),
                mixing=float(new_opts.mixing),
                iterations=int(trial.iterations) if trial is not None else 0,
                residual=float(trial.final_error) if trial is not None else float("nan"),
                screening_smin=float(trial.min_screening_singular_value)
                    if trial is not None else float("nan"),
                converged=good,
                wall_seconds=float(dt),
                error=err_txt,
            ))
            if opts.verbose:
                msg=(f"[GW-ramp {mode}] V={candidate:+.6g}, "
                     f"step={abs(candidate-Vnow):.4g}, mixing={new_opts.mixing:.3g}, "
                     f"residual={records[-1].residual:.3e}, "
                     f"smin={records[-1].screening_smin:.3e}, "
                     f"converged={good}")
                print(msg, flush=True)
            if good:
                success = trial
                break
        if success is None:
            failed_step = abs(candidate - Vnow)
            worst_step = min(worst_step, failed_step) if worst_step is not None else failed_step
            if failed_step * 0.5 < float(min_step) - tolV:
                break
            step = failed_step * 0.5
        else:
            Vnow = float(candidate)
            seed = success
            # Return to base increments after a successful bisection.
            step = min(base_step, max(step * 1.5, float(min_step)))
    reached = direction * (Vtarget - Vnow) <= tolV
    return GWRampResult(
        seed, bool(reached),
        float(Vnow) if seed is not None else None,
        records, float(Vtarget), worst_step
    )
