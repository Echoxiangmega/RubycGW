#!/usr/bin/env python3
"""Scan filling and plot the physical effective quadratic masses r=1/chi.

The plotted quantities are

    r_opposite = 1 / chi_opposite,   eta_+ channel
    r_same     = 1 / chi_same,       eta_- channel

where chi is the selected normal-state susceptibility (GW+MT by default, or
full cGW when requested).  These r values are the curvature of the effective
action written in terms of the physical loop-current order parameter eta.  They
are not the auxiliary-field HS coefficient 3V-(V^2/2)chi0.

For strong coupling the default scan is deliberately continuation based:

1. choose the requested filling closest to ``--anchor-filling`` (default 3),
2. obtain a GW seed there by ramping V from weak coupling to the target V,
3. solve the target-V anchor point,
4. continue from the anchor independently toward lower and higher fillings.

If GW does not converge at a filling, the eta vertex is not evaluated there;
response quantities are stored as NaN instead of reporting a divergent fixed-
point iterate as a physical susceptibility.
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

from rubycgw import (
    RubyParameters,
    MatsubaraGrid,
    GWOptions,
    VertexOptions,
    build_interaction,
    eta_vertices,
    solve_gw,
    solve_noninteracting,
    solve_vertex_q0,
    chi_eta,
)


RESPONSE_PREFIXES = ("G0G0", "GG", "GW_MT", "full_cGW", "selected")


def _safe_inverse(z: complex | None, eps: float = 1e-14) -> complex:
    if z is None:
        return complex(np.nan, np.nan)
    z = complex(z)
    if not np.isfinite(z.real) or not np.isfinite(z.imag) or abs(z) < eps:
        return complex(np.nan, np.nan)
    return 1.0 / z


def _parts(z: complex | None) -> tuple[float, float]:
    if z is None:
        return float("nan"), float("nan")
    z = complex(z)
    return float(z.real), float(z.imag)


def _format_number(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.8g}"


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class _ProgressBar:
    """Dependency-free single-line terminal progress bar with elapsed time/ETA."""

    def __init__(self, total: int, enabled: bool = True, width: int = 28):
        self.total = max(int(total), 1)
        self.enabled = bool(enabled)
        self.width = int(width)
        self.start = time.perf_counter()
        self.last_len = 0

    def clear(self) -> None:
        if not self.enabled or self.last_len == 0:
            return
        print("\r" + " " * self.last_len + "\r", end="", flush=True)
        self.last_len = 0

    def update(self, completed: int, filling: float | None = None) -> None:
        if not self.enabled:
            return
        completed = min(max(int(completed), 0), self.total)
        elapsed = time.perf_counter() - self.start
        fraction = completed / self.total
        nfill = int(round(self.width * fraction))
        nfill = min(max(nfill, 0), self.width)
        bar = "#" * nfill + "-" * (self.width - nfill)
        eta = elapsed * (self.total - completed) / completed if completed > 0 else float("nan")
        filling_text = "" if filling is None else f" | filling={filling:.4f}"
        text = (
            f"[{bar}] {completed:3d}/{self.total:<3d} "
            f"{100.0 * fraction:6.2f}% | elapsed {_format_duration(elapsed)} "
            f"| ETA {_format_duration(eta)}{filling_text}"
        )
        padding = max(self.last_len - len(text), 0)
        print("\r" + text + " " * padding, end="", flush=True)
        self.last_len = len(text)

    def finish(self) -> None:
        if not self.enabled:
            return
        print()
        self.last_len = 0


def _vertex_options(args, include_al: bool) -> VertexOptions:
    return VertexOptions(
        max_iter=args.vertex_max_iter,
        tol=args.vertex_tol,
        mixing=args.vertex_mixing,
        include_hartree=not args.skip_hartree,
        include_mt=True,
        include_al=include_al,
        verbose=args.verbose_iterations,
        momentum_backend=args.momentum_backend,
    )


def _gw_options(args, filling: float, mu: float) -> GWOptions:
    return GWOptions(
        mu=float(mu),
        target_filling=float(filling),
        max_iter=args.gw_max_iter,
        tol=args.gw_tol,
        mixing=args.gw_mixing,
        verbose=args.verbose_iterations,
        momentum_backend=args.momentum_backend,
    )


def _selected_stage_name(args) -> str:
    return "GW+MT" if args.vertex_stage == "mt" else "full cGW"


def _base_row(args, filling: float, branch: str) -> dict:
    row = {
        "filling": float(filling),
        "scan_branch": branch,
        "V": float(args.V),
        "T": float(args.T),
        "nk": int(args.nk),
        "nw": int(args.nw),
        "nOmega": int(args.nomega),
        "vertex_stage": args.vertex_stage,
        "selected_stage": _selected_stage_name(args),
        "momentum_backend": args.momentum_backend,
        "mu0": np.nan,
        "mu_GW": np.nan,
        "actual_filling": np.nan,
        "GW_converged": False,
        "GW_iterations": np.nan,
        "selected_plus_converged": False,
        "selected_minus_converged": False,
        "selected_plus_iterations": np.nan,
        "selected_minus_iterations": np.nan,
        "time_bare_s": np.nan,
        "time_GW_s": np.nan,
        "time_MT_s": 0.0,
        "time_full_s": 0.0,
        "runtime_s": np.nan,
        "vertex_skipped_because_GW_failed": False,
    }
    for prefix in RESPONSE_PREFIXES:
        row[f"{prefix}_opposite_re"] = np.nan
        row[f"{prefix}_opposite_im"] = np.nan
        row[f"{prefix}_same_re"] = np.nan
        row[f"{prefix}_same_im"] = np.nan
    row.update({
        "r_eff_opposite_re": np.nan,
        "r_eff_opposite_im": np.nan,
        "r_eff_same_re": np.nan,
        "r_eff_same_im": np.nan,
        "delta_r_same_minus_opposite_re": np.nan,
        "delta_r_same_minus_opposite_im": np.nan,
    })
    return row


def _store_pair(row: dict, prefix: str, zp: complex | None, zm: complex | None) -> None:
    p_re, p_im = _parts(zp)
    m_re, m_im = _parts(zm)
    row[f"{prefix}_opposite_re"] = p_re
    row[f"{prefix}_opposite_im"] = p_im
    row[f"{prefix}_same_re"] = m_re
    row[f"{prefix}_same_im"] = m_im


def _v_ramp_schedule(target_v: float, explicit_values: list[float] | None) -> list[float]:
    target_v = float(target_v)
    if explicit_values:
        values = [float(v) for v in explicit_values]
        if not np.isclose(values[-1], target_v):
            values.append(target_v)
        return values

    if np.isclose(target_v, 0.0):
        return [0.0]

    sign = 1.0 if target_v > 0 else -1.0
    target_abs = abs(target_v)
    base = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
    values = [sign * v for v in base if v < target_abs - 1e-12]
    values.append(target_v)
    return values


def _write_v_ramp_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _prepare_anchor_gw_seed(args, grid, anchor_filling: float, outdir: Path):
    """Ramp V at the anchor filling and return a converged target-V GW seed."""
    if args.no_v_ramp or args.no_continuation:
        return None, True

    schedule = _v_ramp_schedule(args.V, args.v_ramp_values)
    params_target = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    bare = solve_noninteracting(
        params_target,
        grid,
        mu=args.mu0,
        target_filling=float(anchor_filling),
    )

    print("\nPreparing anchor GW seed by interaction continuation")
    print(f"anchor filling = {anchor_filling:.8g}")
    print("V ramp = " + " -> ".join(f"{v:g}" for v in schedule))

    previous = None
    ramp_rows: list[dict] = []
    for i, value in enumerate(schedule, start=1):
        params_v = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=float(value))
        mu_guess = bare.mu if previous is None else previous.mu
        t0 = time.perf_counter()
        gw = solve_gw(
            params_v,
            grid,
            _gw_options(args, anchor_filling, mu_guess),
            initial=previous,
        )
        runtime = time.perf_counter() - t0
        ramp_rows.append({
            "step": i,
            "V": float(value),
            "filling": float(anchor_filling),
            "converged": bool(gw.converged),
            "iterations": int(gw.iterations),
            "mu": float(gw.mu),
            "actual_filling": float(np.sum(gw.density)),
            "runtime_s": float(runtime),
        })
        print(
            f"  [V-ramp {i:2d}/{len(schedule):2d}] V={value:7.4f}  "
            f"converged={str(gw.converged):5s}  it={gw.iterations:3d}  "
            f"mu={gw.mu: .8f}  time={runtime:.1f}s"
        )
        _write_v_ramp_csv(ramp_rows, outdir / "v_ramp.csv")

        if not gw.converged:
            print(
                f"\nV-ramp stopped: GW failed to converge at V={value:g}. "
                "The filling scan will not cold-start at the target interaction."
            )
            return None, False
        previous = gw

    return {"mu0": bare.mu, "gw": previous}, True


def _run_point(args, grid, params, filling: float, state: dict | None, branch: str):
    previous_state = {} if state is None else dict(state)
    point_start = time.perf_counter()
    row = _base_row(args, filling, branch)
    _, _, k_plus, k_minus = eta_vertices()

    # Bare reference changes with filling.  The previous mu0 is only a scalar
    # seed for the bisection and does not alter the target-filling equation.
    t0 = time.perf_counter()
    mu0_guess = previous_state.get("mu0", args.mu0)
    bare = solve_noninteracting(
        params,
        grid,
        mu=float(mu0_guess),
        target_filling=float(filling),
    )
    chi_plus_g0 = chi_eta(bare.G0, k_plus, grid)
    chi_minus_g0 = chi_eta(bare.G0, k_minus, grid)
    time_bare = time.perf_counter() - t0
    row["mu0"] = float(bare.mu)
    row["time_bare_s"] = float(time_bare)
    _store_pair(row, "G0G0", chi_plus_g0, chi_minus_g0)

    # Interacting GW background.
    t0 = time.perf_counter()
    gw = solve_gw(
        params,
        grid,
        _gw_options(args, filling, bare.mu),
        initial=None if args.no_continuation else previous_state.get("gw"),
    )
    time_gw = time.perf_counter() - t0
    row.update({
        "mu_GW": float(gw.mu),
        "actual_filling": float(np.sum(gw.density)),
        "GW_converged": bool(gw.converged),
        "GW_iterations": int(gw.iterations),
        "time_GW_s": float(time_gw),
    })

    new_state = dict(previous_state)
    new_state["mu0"] = bare.mu

    # A nonconverged GW fixed point is not a valid background for a covariant
    # vertex calculation.  Do not propagate gigantic fixed-point iterates into
    # chi; keep response quantities NaN and retain the last converged state as
    # the seed for the next filling.
    if not gw.converged:
        row["vertex_skipped_because_GW_failed"] = True
        row["runtime_s"] = float(time.perf_counter() - point_start)
        print(
            f"filling={filling:6.3f}  GW FAILED after {gw.iterations:3d} iterations; "
            f"vertex skipped  time={row['runtime_s']:.1f}s"
        )
        return row, new_state

    new_state["gw"] = gw
    chi_plus_gg = chi_eta(gw.G, k_plus, grid)
    chi_minus_gg = chi_eta(gw.G, k_minus, grid)
    _store_pair(row, "GG", chi_plus_gg, chi_minus_gg)
    vq0 = build_interaction(grid.qmesh(), params)[0, 0]

    vp_mt = vm_mt = vp_full = vm_full = None
    chi_plus_mt = chi_minus_mt = None
    chi_plus_full = chi_minus_full = None
    time_mt = time_full = 0.0

    if args.vertex_stage in ("mt", "both"):
        t0 = time.perf_counter()
        mt_opts = _vertex_options(args, include_al=False)
        prev_p = None if args.no_continuation else previous_state.get("mt_plus")
        prev_m = None if args.no_continuation else previous_state.get("mt_minus")
        vp_mt = solve_vertex_q0(
            gw.G, gw.W, vq0, k_plus, grid, mt_opts,
            initial_gamma=None if prev_p is None else prev_p.Gamma,
        )
        vm_mt = solve_vertex_q0(
            gw.G, gw.W, vq0, k_minus, grid, mt_opts,
            initial_gamma=None if prev_m is None else prev_m.Gamma,
        )
        time_mt = time.perf_counter() - t0
        if vp_mt.converged:
            chi_plus_mt = chi_eta(gw.G, k_plus, grid, Gamma=vp_mt.Gamma)
            new_state["mt_plus"] = vp_mt
        if vm_mt.converged:
            chi_minus_mt = chi_eta(gw.G, k_minus, grid, Gamma=vm_mt.Gamma)
            new_state["mt_minus"] = vm_mt
        _store_pair(row, "GW_MT", chi_plus_mt, chi_minus_mt)

    if args.vertex_stage in ("full", "both"):
        t0 = time.perf_counter()
        full_opts = _vertex_options(args, include_al=True)
        prev_fp = None if args.no_continuation else previous_state.get("full_plus")
        prev_fm = None if args.no_continuation else previous_state.get("full_minus")
        if args.vertex_stage == "both":
            init_p = vp_mt.Gamma if vp_mt is not None and vp_mt.converged else (None if prev_fp is None else prev_fp.Gamma)
            init_m = vm_mt.Gamma if vm_mt is not None and vm_mt.converged else (None if prev_fm is None else prev_fm.Gamma)
        else:
            init_p = None if prev_fp is None else prev_fp.Gamma
            init_m = None if prev_fm is None else prev_fm.Gamma
        vp_full = solve_vertex_q0(
            gw.G, gw.W, vq0, k_plus, grid, full_opts,
            initial_gamma=init_p,
        )
        vm_full = solve_vertex_q0(
            gw.G, gw.W, vq0, k_minus, grid, full_opts,
            initial_gamma=init_m,
        )
        time_full = time.perf_counter() - t0
        if vp_full.converged:
            chi_plus_full = chi_eta(gw.G, k_plus, grid, Gamma=vp_full.Gamma)
            new_state["full_plus"] = vp_full
        if vm_full.converged:
            chi_minus_full = chi_eta(gw.G, k_minus, grid, Gamma=vm_full.Gamma)
            new_state["full_minus"] = vm_full
        _store_pair(row, "full_cGW", chi_plus_full, chi_minus_full)

    if args.vertex_stage == "mt":
        chi_plus, chi_minus = chi_plus_mt, chi_minus_mt
        vp_selected, vm_selected = vp_mt, vm_mt
    else:
        chi_plus, chi_minus = chi_plus_full, chi_minus_full
        vp_selected, vm_selected = vp_full, vm_full

    plus_conv = bool(vp_selected is not None and vp_selected.converged)
    minus_conv = bool(vm_selected is not None and vm_selected.converged)
    row.update({
        "selected_plus_converged": plus_conv,
        "selected_minus_converged": minus_conv,
        "selected_plus_iterations": np.nan if vp_selected is None else int(vp_selected.iterations),
        "selected_minus_iterations": np.nan if vm_selected is None else int(vm_selected.iterations),
        "time_MT_s": float(time_mt),
        "time_full_s": float(time_full),
    })
    _store_pair(row, "selected", chi_plus, chi_minus)

    r_plus = _safe_inverse(chi_plus)
    r_minus = _safe_inverse(chi_minus)
    delta_r = r_minus - r_plus
    rp_re, rp_im = _parts(r_plus)
    rm_re, rm_im = _parts(r_minus)
    dr_re, dr_im = _parts(delta_r)
    row.update({
        "r_eff_opposite_re": rp_re,
        "r_eff_opposite_im": rp_im,
        "r_eff_same_re": rm_re,
        "r_eff_same_im": rm_im,
        "delta_r_same_minus_opposite_re": dr_re,
        "delta_r_same_minus_opposite_im": dr_im,
        "runtime_s": float(time.perf_counter() - point_start),
    })

    print(
        f"filling={filling:6.3f}  "
        f"chi_opp={_format_number(row['selected_opposite_re'])}  "
        f"chi_same={_format_number(row['selected_same_re'])}  "
        f"r_opp={_format_number(rp_re)}  r_same={_format_number(rm_re)}  "
        f"GW it={gw.iterations:3d}  "
        f"vertex it=({row['selected_plus_iterations']},{row['selected_minus_iterations']})  "
        f"time={row['runtime_s']:.1f}s"
    )
    if not (plus_conv and minus_conv):
        print("  WARNING: at least one selected vertex did not converge; that channel is stored as NaN.")

    return row, new_state


def _sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: float(r["filling"]))


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    ordered = _sorted_rows(rows)
    keys = list(ordered[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(ordered)


def _integer_guides(ax):
    for n in range(1, 6):
        ax.axvline(n, linewidth=0.7, alpha=0.25)


def _plot(rows: list[dict], outdir: Path, V: float, stage_label: str) -> None:
    ordered = _sorted_rows(rows)
    good = [r for r in ordered if np.isfinite(r["r_eff_opposite_re"]) and np.isfinite(r["r_eff_same_re"])]
    if not good:
        print("No filling has two converged selected vertices; no r_eff figure was produced.")
        return

    x = np.array([r["filling"] for r in good], dtype=float)
    chi_o = np.array([r["selected_opposite_re"] for r in good], dtype=float)
    chi_s = np.array([r["selected_same_re"] for r in good], dtype=float)
    r_o = np.array([r["r_eff_opposite_re"] for r in good], dtype=float)
    r_s = np.array([r["r_eff_same_re"] for r in good], dtype=float)
    dr = r_s - r_o

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, r_o, label=r"$r_{+}^{\rm eff}=1/\chi_{\rm opposite}$")
    ax.plot(x, r_s, label=r"$r_{-}^{\rm eff}=1/\chi_{\rm same}$")
    ax.axhline(0.0, linewidth=0.8)
    _integer_guides(ax)
    ax.set_xlabel("filling per six-site unit cell")
    ax.set_ylabel(r"effective quadratic mass $r^{\rm eff}=1/\chi$")
    ax.set_title(f"Ruby lattice: V={V:g}, {stage_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "r_eff_vs_filling.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, chi_o, label=r"$\chi_{\rm opposite}$ (+)")
    ax.plot(x, chi_s, label=r"$\chi_{\rm same}$ (-)")
    _integer_guides(ax)
    ax.set_xlabel("filling per six-site unit cell")
    ax.set_ylabel("static susceptibility")
    ax.set_title(f"Ruby lattice: V={V:g}, {stage_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "chi_vs_filling.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, dr, label=r"$\Delta r=r_{\rm same}-r_{\rm opposite}$")
    ax.axhline(0.0, linewidth=0.8)
    _integer_guides(ax)
    ax.set_xlabel("filling per six-site unit cell")
    ax.set_ylabel(r"$\Delta r$")
    ax.set_title("negative: same is the softer continuous-instability channel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "delta_r_vs_filling.png", dpi=200)
    plt.close(fig)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", type=float, default=3.0)
    p.add_argument("--T", type=float, default=0.05)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)

    p.add_argument("--nk", type=int, default=6)
    p.add_argument("--nw", type=int, default=60)
    p.add_argument("--nomega", type=int, default=12)

    p.add_argument("--filling-min", type=float, default=0.05)
    p.add_argument("--filling-max", type=float, default=5.95)
    p.add_argument("--num-fillings", type=int, default=241)
    p.add_argument(
        "--fillings", nargs="+", type=float, default=None,
        help="Explicit filling list; overrides min/max/num-fillings.",
    )
    p.add_argument(
        "--anchor-filling", type=float, default=3.0,
        help="Start at the requested filling closest to this value, then scan downward/upward independently.",
    )

    p.add_argument("--vertex-stage", choices=["mt", "full", "both"], default="mt")
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--no-continuation", action="store_true")
    p.add_argument("--no-v-ramp", action="store_true", help="Cold-start the anchor directly at target V.")
    p.add_argument(
        "--v-ramp-values", nargs="+", type=float, default=None,
        help="Explicit interaction continuation values. Target --V is appended if needed.",
    )
    p.add_argument("--skip-hartree", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--no-progress", action="store_true")

    # V=3 is much stronger than the weak-coupling convergence benchmarks, so
    # the default mixing is deliberately conservative.
    p.add_argument("--gw-max-iter", type=int, default=300)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--gw-mixing", type=float, default=0.08)
    p.add_argument("--vertex-max-iter", type=int, default=300)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-mixing", type=float, default=0.10)
    p.add_argument("--mu0", type=float, default=0.0)
    p.add_argument("--verbose-iterations", action="store_true")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    if args.fillings is None:
        fillings = np.linspace(args.filling_min, args.filling_max, args.num_fillings)
    else:
        fillings = np.array(args.fillings, dtype=float)
    fillings = np.unique(np.asarray(fillings, dtype=float))

    if fillings.size == 0:
        raise ValueError("No filling values were supplied.")
    if np.any(fillings <= 0.0) or np.any(fillings >= 6.0):
        raise ValueError("For the six-site spinless cell use fillings strictly between 0 and 6.")

    if args.outdir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path("results") / "filling" / stamp
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(
        nk1=args.nk,
        nk2=args.nk,
        nw=args.nw,
        nOmega=args.nomega,
        T=args.T,
    )

    anchor_index = int(np.argmin(np.abs(fillings - args.anchor_filling)))
    anchor = float(fillings[anchor_index])

    settings = dict(vars(args))
    settings["resolved_anchor_filling"] = anchor
    with (outdir / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    total_start = time.perf_counter()
    print("=" * 78)
    print("RubycGW filling scan")
    print(
        f"V={args.V}, T={args.T}, nk={args.nk}x{args.nk}, nw={args.nw}, "
        f"nOmega={args.nomega}, stage={args.vertex_stage}, backend={args.momentum_backend}"
    )
    print(f"number of fillings: {len(fillings)}")
    print(f"anchor filling: {anchor:.8g} (requested {args.anchor_filling:g})")
    print("scan order: anchor -> lower fillings; anchor -> higher fillings")
    print("r_eff is defined as 1/chi for the physical eta order parameter.")
    print("+ = physical opposite, - = physical same")
    print("=" * 78)

    anchor_seed, ramp_ok = _prepare_anchor_gw_seed(args, grid, anchor, outdir)
    if not ramp_ok:
        print("\nAborting before the filling scan because the anchor V-ramp did not reach the target interaction.")
        print("See:", outdir / "v_ramp.csv")
        return

    rows: list[dict] = []
    progress = _ProgressBar(len(fillings), enabled=not args.no_progress)
    completed = 0
    progress.update(0)

    # Anchor point.  If V-ramp was disabled, this is a deliberate cold start.
    progress.clear()
    try:
        anchor_row, anchor_state = _run_point(args, grid, params, anchor, anchor_seed, "anchor")
    except Exception as exc:
        print(f"ERROR at anchor filling={anchor:.6g}: {exc}")
        if args.fail_fast:
            progress.finish()
            raise
        anchor_row = _base_row(args, anchor, "anchor")
        anchor_row["runtime_s"] = np.nan
        anchor_state = {} if anchor_seed is None else dict(anchor_seed)
    rows.append(anchor_row)
    completed += 1
    _write_csv(rows, outdir / "filling_scan.csv")
    progress.update(completed, anchor)

    # The two branches must start from the same anchor solution rather than
    # contaminating the upper branch with the final lower-filling state.
    lower_state = dict(anchor_state)
    upper_state = dict(anchor_state)

    lower_fillings = fillings[:anchor_index][::-1]
    upper_fillings = fillings[anchor_index + 1:]

    for branch_name, branch_fillings, state0 in [
        ("lower", lower_fillings, lower_state),
        ("upper", upper_fillings, upper_state),
    ]:
        state = state0
        for filling in branch_fillings:
            progress.clear()
            try:
                row, state = _run_point(args, grid, params, float(filling), state, branch_name)
            except Exception as exc:
                print(f"ERROR at filling={filling:.6g}: {exc}")
                if args.fail_fast:
                    progress.finish()
                    raise
                row = _base_row(args, float(filling), branch_name)
                row["runtime_s"] = np.nan
            rows.append(row)
            completed += 1
            _write_csv(rows, outdir / "filling_scan.csv")
            progress.update(completed, float(filling))

    progress.finish()
    stage_label = "GW+MT" if args.vertex_stage == "mt" else "full cGW"
    _plot(rows, outdir, args.V, stage_label)

    elapsed = time.perf_counter() - total_start
    print("\n=== filling scan finished ===")
    print(f"total time: {elapsed:.1f} s")
    print("output directory:", outdir)
    print("CSV:", outdir / "filling_scan.csv")
    if not args.no_v_ramp and not args.no_continuation:
        print("V-ramp diagnostics:", outdir / "v_ramp.csv")
    print("figure:", outdir / "r_eff_vs_filling.png")
    print("figure:", outdir / "chi_vs_filling.png")
    print("figure:", outdir / "delta_r_vs_filling.png")


if __name__ == "__main__":
    main()
