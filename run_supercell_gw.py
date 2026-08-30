#!/usr/bin/env python3
"""Run optimized 18-site periodic Ruby GW across the period-three charge instability.

Optimizations used by default:
1. warm-started fixed-filling mu solves with cached high-frequency tails;
2. loose ramp tolerance for intermediate V/source points and strict final tolerance;
3. safeguarded Type-II Anderson mixing with linear warmup and recovery;
4. failed-but-finite retries continue from the best state reached;
5. secant prediction in V once two consecutive zero-source states are on the same branch.

Every converged zero-source state is checkpointed by default.  Use
``--restart-from auto`` to continue from the nearest compatible checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stdout
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.checkpoint import (
    GWCheckpointSeed,
    checkpoint_filename,
    find_nearest_compatible_checkpoint,
    load_supercell_checkpoint,
    save_supercell_checkpoint,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.supercell import NSUP, charge_order_parameter, period3_real_pattern
from rubycgw.supercell_gw_anderson import (
    AndersonOptions,
    solve_supercell_gw_anderson,
)
from rubycgw.supercell_gw_fast import solve_supercell_gw_fast


class _IterationPrintFilter:
    """Forward solver stdout while thinning only per-iteration status lines.

    Messages such as Pulay resets, residual spikes, bootstrap headers, strict
    mu-refinement summaries and final diagnostics are never suppressed.  Only
    lines beginning with ``SC-GW iter`` or ``GW iter`` are sampled.
    """

    _ITER_RE = re.compile(r"^(?:SC-)?GW iter\s+(\d+):")

    def __init__(self, stream, every: int):
        self.stream = stream
        self.every = max(int(every), 1)
        self._buffer = ""

    def _emit_line(self, line: str) -> None:
        match = self._ITER_RE.match(line)
        if match is not None:
            it = int(match.group(1))
            if it != 1 and it % self.every != 0:
                return
        self.stream.write(line)

    def write(self, text: str) -> int:
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit_line(self._buffer)
            self._buffer = ""
        self.stream.flush()


def _v_schedule(target: float, explicit: list[float] | None) -> list[float]:
    if explicit:
        values = [float(x) for x in explicit]
        if not np.isclose(values[-1], target):
            values.append(float(target))
        return values
    base = [
        0.70, 0.75, 0.78, 0.80, 0.85, 0.90, 1.00,
        1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75,
    ]
    if target <= 0.70:
        return [float(target)]
    values = [v for v in base if v < target - 1e-12]
    values.append(float(target))
    return values


def _v_schedule_after_restart(
    target: float,
    explicit: list[float] | None,
    restart_V: float,
) -> list[float]:
    target = float(target)
    restart_V = float(restart_V)
    if np.isclose(target, restart_V):
        return []
    if explicit:
        values = [float(x) for x in explicit]
        if not np.isclose(values[-1], target):
            values.append(target)
        if target > restart_V:
            return [v for v in values if v > restart_V + 1e-12]
        return [v for v in values if v < restart_V - 1e-12]
    if target < restart_V:
        return [target]
    return [v for v in _v_schedule(target, None) if v > restart_V + 1e-12]


def _source_schedule(values: list[float]) -> list[float]:
    out = [float(x) for x in values]
    if not out:
        out = [0.0]
    if not np.isclose(out[-1], 0.0):
        out.append(0.0)
    return out


def _attempt_schedule(args) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if not args.no_anderson:
        out.append(("anderson", float(args.anderson_beta)))

    for mixing in [args.gw_mixing] + list(args.gw_retry_mixings):
        mixing = float(mixing)
        if not (0.0 < mixing <= 1.0):
            raise ValueError("GW mixing must lie in (0,1].")
        if not any(m == "linear" and np.isclose(x, mixing) for m, x in out):
            out.append(("linear", mixing))

    if not args.no_gw_pulay:
        if not (0.0 < args.gw_pulay_mixing <= 1.0):
            raise ValueError("Pulay damping must lie in (0,1].")
        out.append(("pulay", float(args.gw_pulay_mixing)))
    return out


def _point_tolerance(args, V: float, source: float) -> float:
    is_final = np.isclose(float(V), float(args.V)) and np.isclose(float(source), 0.0)
    return float(args.gw_tol if is_final else args.ramp_tol)


def _gw_options(
    args,
    target_supercell: float,
    mu: float,
    method: str,
    mixing: float,
    tol: float,
) -> GWOptions:
    return GWOptions(
        mu=float(mu),
        target_filling=float(target_supercell),
        max_iter=args.gw_max_iter,
        tol=float(tol),
        mixing=float(mixing),
        mixing_method=str(method),
        pulay_history=args.gw_pulay_history,
        pulay_start=args.gw_pulay_start,
        pulay_regularization=args.gw_pulay_regularization,
        mu_tol=args.mu_tol,
        mu_max_iter=args.mu_max_iter,
        verbose=args.verbose_iterations,
        momentum_backend=args.momentum_backend,
    )


def _anderson_options(args) -> AndersonOptions:
    return AndersonOptions(
        history=args.anderson_history,
        start=args.anderson_start,
        warmup_beta=args.anderson_warmup_mixing,
        beta=args.anderson_beta,
        beta_min=args.anderson_beta_min,
        beta_max=args.anderson_beta_max,
        regularization=args.anderson_regularization,
        growth_factor=args.anderson_growth_factor,
        growth_patience=args.anderson_growth_patience,
        recovery_steps=args.anderson_recovery_steps,
        step_cap=args.anderson_step_cap,
    )


def _diagnostics(gw) -> dict:
    phi = charge_order_parameter(gw.density)
    return {
        "mu": float(gw.mu),
        "actual_supercell_filling": float(np.sum(gw.density)),
        "actual_primitive_filling": float(np.sum(gw.density) / 3.0),
        "charge_order_re": float(phi.real),
        "charge_order_im": float(phi.imag),
        "charge_order_abs": float(abs(phi)),
        "charge_order_phase_rad": float(np.angle(phi)) if abs(phi) > 1e-15 else 0.0,
        "density_rms_modulation": float(np.std(gw.density)),
        "min_screening_singular_value": float(gw.min_screening_singular_value),
        "screening_m": int(gw.min_screening_m),
        "screening_Omega": float(gw.min_screening_Omega),
        "screening_q_sc1": float(gw.min_screening_q1),
        "screening_q_sc2": float(gw.min_screening_q2),
        "density_mode_residual": float(gw.min_density_mode_residual),
    }


def _finite_gw_state(gw) -> bool:
    if gw is None or not np.isfinite(float(gw.final_error)):
        return False
    return bool(
        np.all(np.isfinite(gw.Sigma_H))
        and np.all(np.isfinite(gw.Sigma_GW))
        and np.isfinite(float(gw.mu))
    )


def _solve_adaptive(
    args,
    params,
    grid,
    source: float,
    target_supercell: float,
    mu_guess: float,
    initial,
    tol: float,
):
    """Try Anderson first, then conservative linear/Pulay fallbacks."""
    attempts = []
    best = None
    best_error = float("inf")
    retry_seed = initial
    local_mu = float(mu_guess)

    for i, (method, mixing) in enumerate(_attempt_schedule(args), start=1):
        t0 = time.perf_counter()
        used_carried_seed = bool(i > 1 and retry_seed is not initial)
        try:
            opts = _gw_options(
                args, target_supercell, local_mu, method, mixing, tol
            )

            def _run_solver():
                if method == "anderson":
                    return solve_supercell_gw_anderson(
                        params,
                        grid,
                        opts,
                        source_strength=source,
                        initial=retry_seed,
                        anderson=_anderson_options(args),
                    )
                return solve_supercell_gw_fast(
                    params,
                    grid,
                    opts,
                    source_strength=source,
                    initial=retry_seed,
                )

            if args.verbose_iterations and int(args.verbose_every) > 1:
                stream = _IterationPrintFilter(sys.stdout, args.verbose_every)
                with redirect_stdout(stream):
                    gw = _run_solver()
                stream.flush()
            else:
                gw = _run_solver()

            runtime = time.perf_counter() - t0
            row = {
                "attempt": i,
                "method": method,
                "mixing": float(mixing),
                "requested_tol": float(tol),
                "carried_retry_seed": used_carried_seed,
                "converged": bool(gw.converged),
                "iterations": int(gw.iterations),
                "final_error": float(gw.final_error),
                "runtime_s": float(runtime),
                "exception": "",
            }
            row.update(_diagnostics(gw))

            if _finite_gw_state(gw) and float(gw.final_error) < best_error:
                best = gw
                best_error = float(gw.final_error)

            if row["converged"]:
                attempts.append(row)
                return gw, attempts

            if best is not None:
                retry_seed = best
                local_mu = float(best.mu)

        except Exception as exc:
            runtime = time.perf_counter() - t0
            row = {
                "attempt": i,
                "method": method,
                "mixing": float(mixing),
                "requested_tol": float(tol),
                "carried_retry_seed": used_carried_seed,
                "converged": False,
                "iterations": np.nan,
                "final_error": np.nan,
                "runtime_s": float(runtime),
                "exception": repr(exc),
                "mu": np.nan,
                "actual_supercell_filling": np.nan,
                "actual_primitive_filling": np.nan,
                "charge_order_re": np.nan,
                "charge_order_im": np.nan,
                "charge_order_abs": np.nan,
                "charge_order_phase_rad": np.nan,
                "density_rms_modulation": np.nan,
                "min_screening_singular_value": np.nan,
                "screening_m": np.nan,
                "screening_Omega": np.nan,
                "screening_q_sc1": np.nan,
                "screening_q_sc2": np.nan,
                "density_mode_residual": np.nan,
            }
        attempts.append(row)

    return best, attempts


def _phi_of_state(gw) -> complex:
    if hasattr(gw, "density"):
        return complex(charge_order_parameter(gw.density))
    return 0.0j


def _same_zero_source_branch(phi1: complex, phi2: complex, threshold: float) -> bool:
    a1, a2 = abs(phi1), abs(phi2)
    b1, b2 = a1 > threshold, a2 > threshold
    if b1 != b2:
        return False
    if not b1:
        return True
    overlap = (phi2 * np.conj(phi1)).real / max(a1 * a2, 1e-30)
    return bool(overlap > 0.5)


def _predictor_seed(
    target_V: float,
    zero_history: list[tuple[float, object, complex]],
    args,
) -> tuple[GWCheckpointSeed | None, str]:
    """Secant predictor from the last two same-branch zero-source solutions."""
    if args.no_v_predictor or len(zero_history) < 2:
        return None, ""

    V1, g1, phi1 = zero_history[-2]
    V2, g2, phi2 = zero_history[-1]
    denom = float(V2 - V1)
    if abs(denom) < 1e-14:
        return None, ""
    ratio = float((target_V - V2) / denom)
    if ratio <= 0.0 or ratio > float(args.predictor_max_ratio):
        return None, ""
    if not _same_zero_source_branch(
        phi1, phi2, float(args.predictor_order_threshold)
    ):
        return None, ""

    factor = float(args.predictor_damping) * ratio
    sigma_h = g2.Sigma_H + factor * (g2.Sigma_H - g1.Sigma_H)
    sigma_gw = g2.Sigma_GW + factor * (g2.Sigma_GW - g1.Sigma_GW)
    mu = float(g2.mu + factor * (g2.mu - g1.mu))
    if not (
        np.all(np.isfinite(sigma_h))
        and np.all(np.isfinite(sigma_gw))
        and np.isfinite(mu)
    ):
        return None, ""

    seed = GWCheckpointSeed(
        Sigma_H=np.asarray(sigma_h, dtype=complex),
        Sigma_GW=np.asarray(sigma_gw, dtype=complex),
        mu=mu,
    )
    label = f"secant V={V1:g},{V2:g} -> {target_V:g} (factor={factor:.3f})"
    return seed, label


def _update_zero_history(
    history: list[tuple[float, object, complex]],
    V: float,
    gw,
    phi: complex | None = None,
) -> None:
    value = _phi_of_state(gw) if phi is None else complex(phi)
    if history and np.isclose(history[-1][0], V):
        history[-1] = (float(V), gw, value)
    else:
        history.append((float(V), gw, value))
    if len(history) > 3:
        del history[:-3]


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _append_density_rows(
    rows: list[dict],
    gw,
    V: float,
    source: float,
    v_step: int,
    source_step: int,
) -> None:
    pattern = period3_real_pattern()
    mean_density = float(np.mean(gw.density))
    for I, density in enumerate(gw.density):
        rows.append(
            {
                "V": float(V),
                "source": float(source),
                "v_step": int(v_step),
                "source_step": int(source_step),
                "site": int(I),
                "sector": int(I // 6),
                "sublattice": int(I % 6),
                "density": float(density),
                "delta_from_mean": float(density - mean_density),
                "seed_pattern": float(pattern[I]),
            }
        )


def _plot_zero_source(rows: list[dict], outdir: Path) -> None:
    good = [
        r for r in rows
        if r["converged"] and np.isclose(r["source"], 0.0)
    ]
    by_key = {}
    for row in good:
        by_key[(row["v_step"], row["source_step"])] = row
    good = list(by_key.values())
    if not good:
        return
    good.sort(key=lambda r: float(r["V"]))
    V = np.array([r["V"] for r in good], dtype=float)
    amp = np.array([r["charge_order_abs"] for r in good], dtype=float)
    smin = np.array(
        [r["min_screening_singular_value"] for r in good], dtype=float
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(V, amp, marker="o")
    ax.set_xlabel("V")
    ax.set_ylabel(r"$|\Phi_{Q=(1/3,1/3)}|$")
    ax.set_title("18-site zero-source charge order")
    fig.tight_layout()
    fig.savefig(outdir / "charge_order_vs_V.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(V, smin, marker="o")
    ax.set_xlabel("V")
    ax.set_ylabel(r"$s_{\min}[I-VP]$")
    ax.set_yscale("log")
    ax.set_title("18-site zero-source screening diagnostic")
    fig.tight_layout()
    fig.savefig(outdir / "screening_smin_vs_V.png", dpi=200)
    plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--V-values", nargs="+", type=float, default=None)
    p.add_argument("--T", type=float, default=0.05)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--primitive-filling", type=float, default=3.0)

    p.add_argument("--nk1", type=int, default=3)
    p.add_argument("--nk2", type=int, default=3)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument(
        "--momentum-backend", choices=["fft", "direct"], default="fft"
    )

    p.add_argument("--source-onset-V", type=float, default=0.78)
    p.add_argument(
        "--source-sequence",
        nargs="+",
        type=float,
        default=[1e-2, 5e-3, 1e-3, 0.0],
    )

    p.add_argument("--gw-max-iter", type=int, default=300)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--ramp-tol", type=float, default=1e-6)
    p.add_argument("--gw-mixing", type=float, default=0.20)
    p.add_argument(
        "--gw-retry-mixings", nargs="+", type=float, default=[0.10]
    )
    p.add_argument("--no-gw-pulay", action="store_true")
    p.add_argument("--gw-pulay-mixing", type=float, default=0.70)
    p.add_argument("--gw-pulay-history", type=int, default=6)
    p.add_argument("--gw-pulay-start", type=int, default=3)
    p.add_argument("--gw-pulay-regularization", type=float, default=1e-10)

    p.add_argument("--no-anderson", action="store_true")
    p.add_argument("--anderson-history", type=int, default=6)
    p.add_argument("--anderson-start", type=int, default=8)
    p.add_argument("--anderson-warmup-mixing", type=float, default=0.20)
    p.add_argument("--anderson-beta", type=float, default=0.70)
    p.add_argument("--anderson-beta-min", type=float, default=0.15)
    p.add_argument("--anderson-beta-max", type=float, default=0.90)
    p.add_argument("--anderson-regularization", type=float, default=1e-8)
    p.add_argument("--anderson-growth-factor", type=float, default=1.20)
    p.add_argument("--anderson-growth-patience", type=int, default=3)
    p.add_argument("--anderson-recovery-steps", type=int, default=4)
    p.add_argument("--anderson-step-cap", type=float, default=5.0)

    p.add_argument("--no-v-predictor", action="store_true")
    p.add_argument("--predictor-damping", type=float, default=0.80)
    p.add_argument("--predictor-max-ratio", type=float, default=2.0)
    p.add_argument("--predictor-order-threshold", type=float, default=1e-4)

    p.add_argument("--mu-tol", type=float, default=1e-8)
    p.add_argument("--mu-max-iter", type=int, default=40)
    p.add_argument("--mu0", type=float, default=0.0)
    p.add_argument("--verbose-iterations", action="store_true")
    p.add_argument(
        "--verbose-every",
        type=int,
        default=1,
        help=(
            "With --verbose-iterations, print one SC-GW/GW iteration line "
            "every N outer iterations. Event and summary lines are always shown."
        ),
    )

    p.add_argument(
        "--restart-from",
        type=str,
        default=None,
        help="Checkpoint .npz path, or 'auto' for nearest compatible V<=target.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="results/supercell18/checkpoints",
    )
    p.add_argument("--no-checkpoints", action="store_true")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    if not (0.0 < args.primitive_filling < 6.0):
        raise ValueError("--primitive-filling must lie between 0 and 6")
    if not (0.0 < args.gw_tol <= args.ramp_tol):
        raise ValueError("Require 0 < --gw-tol <= --ramp-tol.")
    if args.mu_tol <= 0.0:
        raise ValueError("--mu-tol must be positive.")
    if int(args.verbose_every) < 1:
        raise ValueError("--verbose-every must be a positive integer.")
    if not (0.0 < args.predictor_damping <= 1.5):
        raise ValueError("--predictor-damping must lie in (0,1.5].")

    target_supercell = 3.0 * float(args.primitive_filling)
    if args.outdir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path("results") / "supercell18" / stamp
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid = MatsubaraGrid(
        nk1=args.nk1,
        nk2=args.nk2,
        nw=args.nw,
        nOmega=args.nomega,
        T=args.T,
    )
    checkpoint_dir = Path(args.checkpoint_dir)

    previous = None
    previous_point_V = None
    previous_point_source = None
    zero_history: list[tuple[float, object, complex]] = []
    mu_guess = float(args.mu0)
    restart_meta = None
    restart_path = None

    compatibility_params = RubyParameters(
        ti=args.ti, t1=args.t1, t2=args.t2, V=float(args.V)
    )

    if args.restart_from is not None:
        if args.restart_from.lower() == "auto":
            restart_path = find_nearest_compatible_checkpoint(
                checkpoint_dir,
                args.V,
                compatibility_params,
                grid,
                args.primitive_filling,
            )
            if restart_path is None:
                print("No compatible checkpoint found; using ordinary V ramp.")
        else:
            restart_path = Path(args.restart_from)

        if restart_path is not None:
            previous, restart_meta, _ = load_supercell_checkpoint(
                restart_path,
                compatibility_params,
                grid,
                args.primitive_filling,
            )
            mu_guess = float(previous.mu)
            restart_V = float(restart_meta["V"])
            restart_phi = complex(
                float(restart_meta.get("charge_order_re", 0.0)),
                float(restart_meta.get("charge_order_im", 0.0)),
            )
            _update_zero_history(
                zero_history, restart_V, previous, phi=restart_phi
            )
            previous_point_V = restart_V
            previous_point_source = 0.0

    if restart_meta is None:
        schedule = _v_schedule(args.V, args.V_values)
        source_has_been_used = False
    else:
        restart_V = float(restart_meta["V"])
        schedule = _v_schedule_after_restart(
            args.V, args.V_values, restart_V
        )
        source_has_been_used = bool(
            restart_V >= args.source_onset_V - 1e-12
            and float(restart_meta.get("charge_order_abs", 0.0)) > 1e-6
        )
        if (
            not source_has_been_used
            and restart_V >= args.source_onset_V - 1e-12
            and (not schedule or not np.isclose(schedule[0], restart_V))
        ):
            schedule = [restart_V] + schedule

        if (
            np.isclose(restart_V, args.V)
            and float(restart_meta.get("final_error", np.inf)) > args.gw_tol
            and not schedule
        ):
            schedule = [float(args.V)]

    source_sequence = _source_schedule(args.source_sequence)

    settings = dict(vars(args))
    settings["resolved_V_schedule"] = schedule
    settings["resolved_source_sequence"] = source_sequence
    settings["target_supercell_filling"] = target_supercell
    settings["matrix_dimension"] = NSUP
    settings["resolved_restart_path"] = (
        None if restart_path is None else str(restart_path)
    )
    settings["restart_metadata"] = restart_meta
    settings["solver"] = "anderson + fast fallback"
    with (outdir / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    print("=" * 92)
    print("18-site periodic Ruby supercell GW (predictor + adaptive Anderson)")
    print("T1=a1-a2, T2=a1+2a2; primitive Q=(1/3,1/3) -> q_sc=0")
    print(
        f"primitive filling={args.primitive_filling:g}; "
        f"supercell target={target_supercell:g}; "
        f"grid={args.nk1}x{args.nk2}, nw={args.nw}, "
        f"nOmega={args.nomega}, T={args.T:g}"
    )
    print(
        f"tolerances: ramp={args.ramp_tol:.1e}, "
        f"final={args.gw_tol:.1e}, mu={args.mu_tol:.1e}"
    )
    if args.verbose_iterations:
        print(f"verbose iterations: every {args.verbose_every} outer step(s)")
    print(
        f"Anderson: warmup={args.anderson_start} steps at "
        f"{args.anderson_warmup_mixing:g}, history={args.anderson_history}, "
        f"beta={args.anderson_beta:g}"
    )
    print(
        f"V predictor: enabled={not args.no_v_predictor}, "
        f"damping={args.predictor_damping:g}, "
        f"max_ratio={args.predictor_max_ratio:g}"
    )
    if restart_meta is not None:
        print(
            f"restart: {restart_path} "
            f"(V={float(restart_meta['V']):g}, "
            f"err={float(restart_meta.get('final_error', np.nan)):.2e}, "
            f"|Phi|={float(restart_meta.get('charge_order_abs', 0.0)):.3e})"
        )
    print(
        "V ramp:",
        " -> ".join(f"{v:g}" for v in schedule)
        if schedule
        else "already strict-converged",
    )
    print(
        f"source when needed at V>={args.source_onset_V:g}: "
        + " -> ".join(f"{h:g}" for h in source_sequence)
    )
    print(
        "GW attempts:",
        " -> ".join(f"{m}:{x:g}" for m, x in _attempt_schedule(args)),
    )
    if not args.no_checkpoints:
        print("checkpoint directory:", checkpoint_dir)
    print("=" * 92)

    if not schedule:
        print("Requested V is already available at the strict target tolerance.")
        print("checkpoint:", restart_path)
        return

    attempt_rows: list[dict] = []
    density_rows: list[dict] = []
    stopped = False

    for iv, V in enumerate(schedule, start=1):
        params = RubyParameters(
            ti=args.ti, t1=args.t1, t2=args.t2, V=float(V)
        )
        if (
            not source_has_been_used
            and V >= args.source_onset_V - 1e-12
        ):
            sources = source_sequence
            source_has_been_used = True
        else:
            sources = [0.0]

        for ih, source in enumerate(sources, start=1):
            tol = _point_tolerance(args, V, source)
            point_initial = previous
            seed_kind = "previous"
            predictor_label = ""

            can_predict = (
                np.isclose(source, 0.0)
                and previous is not None
                and previous_point_source is not None
                and np.isclose(previous_point_source, 0.0)
                and previous_point_V is not None
                and not np.isclose(previous_point_V, V)
            )
            if can_predict:
                predicted, predictor_label = _predictor_seed(
                    float(V), zero_history, args
                )
                if predicted is not None:
                    point_initial = predicted
                    seed_kind = "V-secant"

            print(
                f"\n[V {iv}/{len(schedule)}, source {ih}/{len(sources)}] "
                f"V={V:g}, h={source:g}, tol={tol:.1e}, seed={seed_kind}"
            )
            if predictor_label:
                print("  predictor:", predictor_label)

            gw, attempts = _solve_adaptive(
                args,
                params,
                grid,
                source,
                target_supercell,
                mu_guess,
                point_initial,
                tol,
            )

            for att in attempts:
                row = {
                    "v_step": iv,
                    "source_step": ih,
                    "V": float(V),
                    "source": float(source),
                    "seed_kind": seed_kind,
                    "predictor_label": predictor_label,
                    "primitive_filling_target": float(
                        args.primitive_filling
                    ),
                    "supercell_filling_target": float(target_supercell),
                    "nk1": int(args.nk1),
                    "nk2": int(args.nk2),
                    "nw": int(args.nw),
                    "nOmega": int(args.nomega),
                }
                row.update(att)
                attempt_rows.append(row)
                exc = (
                    ""
                    if not att["exception"]
                    else f" exception={att['exception']}"
                )
                print(
                    f"  try {att['attempt']}/{len(_attempt_schedule(args))}: "
                    f"{att['method']} mix={att['mixing']:.3f} "
                    f"carry={att['carried_retry_seed']} "
                    f"conv={att['converged']} "
                    f"it={att['iterations']} "
                    f"res={att['final_error']:.3e} "
                    f"|Phi|={att['charge_order_abs']:.3e} "
                    f"smin={att['min_screening_singular_value']:.3e} "
                    f"q_sc=({att['screening_q_sc1']:.3f},"
                    f"{att['screening_q_sc2']:.3f}) "
                    f"time={att['runtime_s']:.1f}s{exc}"
                )
                _write_csv(
                    attempt_rows, outdir / "supercell_scan.csv"
                )

            if gw is None or not gw.converged:
                if gw is not None:
                    print(
                        f"\nSTOP: best retry state reached "
                        f"residual={gw.final_error:.3e}, "
                        f"but requested tol={tol:.3e}."
                    )
                else:
                    print(
                        "\nSTOP: all GW attempts failed at "
                        "this V/source point."
                    )
                stopped = True
                break

            previous = gw
            previous_point_V = float(V)
            previous_point_source = float(source)
            mu_guess = float(gw.mu)
            _append_density_rows(
                density_rows, gw, V, source, iv, ih
            )
            _write_csv(
                density_rows, outdir / "density_profile.csv"
            )
            d = _diagnostics(gw)
            print(
                f"  converged: mu={d['mu']:.8f}, "
                f"n_primitive={d['actual_primitive_filling']:.10f}, "
                f"Phi={d['charge_order_re']:+.4e}"
                f"{d['charge_order_im']:+.4e}i, "
                f"|Phi|={d['charge_order_abs']:.4e}, "
                f"density_rms={d['density_rms_modulation']:.4e}"
            )

            if np.isclose(source, 0.0):
                _update_zero_history(zero_history, V, gw)
                if not args.no_checkpoints:
                    ckpt = checkpoint_dir / checkpoint_filename(
                        V, args.primitive_filling, grid
                    )
                    save_supercell_checkpoint(
                        ckpt,
                        gw,
                        params,
                        grid,
                        args.primitive_filling,
                        source=0.0,
                    )
                    print("  checkpoint:", ckpt)

        if stopped:
            break

    _plot_zero_source(attempt_rows, outdir)
    print("\n=== finished ===")
    print("output:", outdir)
    print("attempt diagnostics:", outdir / "supercell_scan.csv")
    print("site densities:", outdir / "density_profile.csv")
    print("charge plot:", outdir / "charge_order_vs_V.png")
    print("screening plot:", outdir / "screening_smin_vs_V.png")
    if not args.no_checkpoints:
        print("checkpoints:", checkpoint_dir)


if __name__ == "__main__":
    main()
