#!/usr/bin/env python3
"""Run 18-site periodic Ruby GW across the period-three charge instability.

The supercell has translations T1=a1-a2 and T2=a1+2a2, so the primitive
Q=(1/3,1/3) mode is folded to supercell q=0.  A temporary period-three charge
source can be turned on once the V ramp reaches ``--source-onset-V`` and is then
adiabatically removed.  Subsequent V points continue from the zero-source
broken-symmetry solution.

``--primitive-filling`` is always quoted per original six-site Ruby unit cell;
the internal 18-site target is three times larger.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.supercell import NSUP, charge_order_parameter, period3_real_pattern
from rubycgw.supercell_gw import solve_supercell_gw


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


def _source_schedule(values: list[float]) -> list[float]:
    out = [float(x) for x in values]
    if not out:
        out = [0.0]
    if not np.isclose(out[-1], 0.0):
        out.append(0.0)
    return out


def _attempt_schedule(args) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
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


def _gw_options(args, target_supercell: float, mu: float, method: str, mixing: float) -> GWOptions:
    return GWOptions(
        mu=float(mu),
        target_filling=float(target_supercell),
        max_iter=args.gw_max_iter,
        tol=args.gw_tol,
        mixing=float(mixing),
        mixing_method=str(method),
        pulay_history=args.gw_pulay_history,
        pulay_start=args.gw_pulay_start,
        pulay_regularization=args.gw_pulay_regularization,
        verbose=args.verbose_iterations,
        momentum_backend=args.momentum_backend,
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


def _solve_adaptive(args, params, grid, source: float, target_supercell: float,
                    mu_guess: float, initial):
    attempts = []
    last = None
    for i, (method, mixing) in enumerate(_attempt_schedule(args), start=1):
        t0 = time.perf_counter()
        try:
            gw = solve_supercell_gw(
                params,
                grid,
                _gw_options(args, target_supercell, mu_guess, method, mixing),
                source_strength=source,
                initial=initial,
            )
            runtime = time.perf_counter() - t0
            last = gw
            row = {
                "attempt": i,
                "method": method,
                "mixing": float(mixing),
                "converged": bool(gw.converged),
                "iterations": int(gw.iterations),
                "final_error": float(gw.final_error),
                "runtime_s": float(runtime),
                "exception": "",
            }
            row.update(_diagnostics(gw))
        except Exception as exc:
            runtime = time.perf_counter() - t0
            row = {
                "attempt": i,
                "method": method,
                "mixing": float(mixing),
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
        if row["converged"]:
            break
    return last, attempts


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _append_density_rows(rows: list[dict], gw, V: float, source: float,
                         v_step: int, source_step: int) -> None:
    pattern = period3_real_pattern()
    mean_density = float(np.mean(gw.density))
    for I, density in enumerate(gw.density):
        rows.append({
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
        })


def _plot_zero_source(rows: list[dict], outdir: Path) -> None:
    good = [r for r in rows if r["converged"] and np.isclose(r["source"], 0.0)]
    # Keep only the final successful attempt for each V/source step.
    by_key = {}
    for row in good:
        by_key[(row["v_step"], row["source_step"])] = row
    good = list(by_key.values())
    if not good:
        return
    good.sort(key=lambda r: float(r["V"]))
    V = np.array([r["V"] for r in good], dtype=float)
    amp = np.array([r["charge_order_abs"] for r in good], dtype=float)
    smin = np.array([r["min_screening_singular_value"] for r in good], dtype=float)

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
    p.add_argument("--V", type=float, default=1.0,
                   help="Target interaction. Use --V 3 after the first validation run.")
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
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")

    p.add_argument("--source-onset-V", type=float, default=0.78)
    p.add_argument("--source-sequence", nargs="+", type=float,
                   default=[1e-2, 5e-3, 1e-3, 0.0])

    p.add_argument("--gw-max-iter", type=int, default=300)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--gw-mixing", type=float, default=0.20)
    p.add_argument("--gw-retry-mixings", nargs="+", type=float, default=[0.10])
    p.add_argument("--no-gw-pulay", action="store_true")
    p.add_argument("--gw-pulay-mixing", type=float, default=0.70)
    p.add_argument("--gw-pulay-history", type=int, default=6)
    p.add_argument("--gw-pulay-start", type=int, default=3)
    p.add_argument("--gw-pulay-regularization", type=float, default=1e-10)
    p.add_argument("--mu0", type=float, default=0.0)
    p.add_argument("--verbose-iterations", action="store_true")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    if not (0.0 < args.primitive_filling < 6.0):
        raise ValueError("--primitive-filling must lie between 0 and 6")

    schedule = _v_schedule(args.V, args.V_values)
    source_sequence = _source_schedule(args.source_sequence)
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

    settings = dict(vars(args))
    settings["resolved_V_schedule"] = schedule
    settings["resolved_source_sequence"] = source_sequence
    settings["target_supercell_filling"] = target_supercell
    settings["matrix_dimension"] = NSUP
    with (outdir / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    print("=" * 84)
    print("18-site periodic Ruby supercell GW")
    print("T1=a1-a2, T2=a1+2a2; primitive Q=(1/3,1/3) is folded to q_sc=0")
    print(
        f"primitive filling={args.primitive_filling:g}; supercell target={target_supercell:g}; "
        f"grid={args.nk1}x{args.nk2}, nw={args.nw}, nOmega={args.nomega}, T={args.T:g}"
    )
    print("V ramp:", " -> ".join(f"{v:g}" for v in schedule))
    print(
        f"source first applied at V>={args.source_onset_V:g}: "
        + " -> ".join(f"{h:g}" for h in source_sequence)
    )
    print(
        "GW attempts:",
        " -> ".join(f"{m}:{x:g}" for m, x in _attempt_schedule(args)),
    )
    print("=" * 84)

    attempt_rows: list[dict] = []
    density_rows: list[dict] = []
    previous = None
    mu_guess = float(args.mu0)
    source_has_been_used = False
    stopped = False

    for iv, V in enumerate(schedule, start=1):
        params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=float(V))
        if (not source_has_been_used) and V >= args.source_onset_V - 1e-12:
            sources = source_sequence
            source_has_been_used = True
        else:
            sources = [0.0]

        for ih, source in enumerate(sources, start=1):
            print(f"\n[V {iv}/{len(schedule)}, source {ih}/{len(sources)}] V={V:g}, h={source:g}")
            gw, attempts = _solve_adaptive(
                args, params, grid, source, target_supercell, mu_guess, previous
            )

            for att in attempts:
                row = {
                    "v_step": iv,
                    "source_step": ih,
                    "V": float(V),
                    "source": float(source),
                    "primitive_filling_target": float(args.primitive_filling),
                    "supercell_filling_target": float(target_supercell),
                    "nk1": int(args.nk1),
                    "nk2": int(args.nk2),
                    "nw": int(args.nw),
                    "nOmega": int(args.nomega),
                }
                row.update(att)
                attempt_rows.append(row)
                exc = "" if not att["exception"] else f" exception={att['exception']}"
                print(
                    f"  try {att['attempt']}/{len(_attempt_schedule(args))}: "
                    f"{att['method']} mix={att['mixing']:.3f} conv={att['converged']} "
                    f"it={att['iterations']} res={att['final_error']:.3e} "
                    f"|Phi|={att['charge_order_abs']:.3e} "
                    f"smin={att['min_screening_singular_value']:.3e} "
                    f"q_sc=({att['screening_q_sc1']:.3f},{att['screening_q_sc2']:.3f}) "
                    f"time={att['runtime_s']:.1f}s{exc}"
                )
                _write_csv(attempt_rows, outdir / "supercell_scan.csv")

            if gw is None or not gw.converged:
                print("\nSTOP: all GW attempts failed at this V/source point.")
                stopped = True
                break

            previous = gw
            mu_guess = float(gw.mu)
            _append_density_rows(density_rows, gw, V, source, iv, ih)
            _write_csv(density_rows, outdir / "density_profile.csv")
            d = _diagnostics(gw)
            print(
                f"  converged: mu={d['mu']:.8f}, n_primitive={d['actual_primitive_filling']:.10f}, "
                f"Phi={d['charge_order_re']:+.4e}{d['charge_order_im']:+.4e}i, "
                f"|Phi|={d['charge_order_abs']:.4e}, density_rms={d['density_rms_modulation']:.4e}"
            )

        if stopped:
            break

    _plot_zero_source(attempt_rows, outdir)
    print("\n=== finished ===")
    print("output:", outdir)
    print("attempt diagnostics:", outdir / "supercell_scan.csv")
    print("site densities:", outdir / "density_profile.csv")
    print("charge plot:", outdir / "charge_order_vs_V.png")
    print("screening plot:", outdir / "screening_smin_vs_V.png")


if __name__ == "__main__":
    main()
