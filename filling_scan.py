#!/usr/bin/env python3
"""Scan filling and plot the physical effective quadratic masses r=1/chi.

The plotted quantities are

    r_opposite = 1 / chi_opposite,   eta_+ channel
    r_same     = 1 / chi_same,       eta_- channel

where chi is the selected normal-state susceptibility (GW+MT by default, or
full cGW when requested).  These r values are the curvature of the effective
action written in terms of the physical loop-current order parameter eta.  They
are not the auxiliary-field HS coefficient 3V-(V^2/2)chi0.

By default the scan reproduces the filling grid of the earlier Ruby HS plot:
0.05 <= filling <= 5.95 with 241 points, at fixed V=3 and T=0.05.
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


def _safe_inverse(z: complex, eps: float = 1e-14) -> complex:
    z = complex(z)
    if abs(z) < eps:
        return complex(np.nan, np.nan)
    return 1.0 / z


def _parts(z: complex) -> tuple[float, float]:
    z = complex(z)
    return float(z.real), float(z.imag)


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


def _run_point(args, grid, params, filling: float, state: dict | None):
    state = state or {}
    point_start = time.perf_counter()
    _, _, k_plus, k_minus = eta_vertices()

    # Bare reference must be recomputed because filling changes, but the
    # previous noninteracting chemical potential is an excellent scalar seed.
    t0 = time.perf_counter()
    mu0_guess = state.get("mu0", args.mu0)
    bare = solve_noninteracting(
        params,
        grid,
        mu=float(mu0_guess),
        target_filling=float(filling),
    )
    chi_plus_g0 = chi_eta(bare.G0, k_plus, grid)
    chi_minus_g0 = chi_eta(bare.G0, k_minus, grid)
    time_bare = time.perf_counter() - t0

    t0 = time.perf_counter()
    gw_opts = GWOptions(
        mu=bare.mu,
        target_filling=float(filling),
        max_iter=args.gw_max_iter,
        tol=args.gw_tol,
        mixing=args.gw_mixing,
        verbose=args.verbose_iterations,
        momentum_backend=args.momentum_backend,
    )
    gw = solve_gw(
        params,
        grid,
        gw_opts,
        initial=None if args.no_continuation else state.get("gw"),
    )
    time_gw = time.perf_counter() - t0

    chi_plus_gg = chi_eta(gw.G, k_plus, grid)
    chi_minus_gg = chi_eta(gw.G, k_minus, grid)
    vq0 = build_interaction(grid.qmesh(), params)[0, 0]

    vp_mt = vm_mt = vp_full = vm_full = None
    chi_plus_mt = chi_minus_mt = None
    chi_plus_full = chi_minus_full = None
    time_mt = time_full = 0.0

    if args.vertex_stage in ("mt", "both"):
        t0 = time.perf_counter()
        mt_opts = _vertex_options(args, include_al=False)
        prev_p = None if args.no_continuation else state.get("mt_plus")
        prev_m = None if args.no_continuation else state.get("mt_minus")
        vp_mt = solve_vertex_q0(
            gw.G, gw.W, vq0, k_plus, grid, mt_opts,
            initial_gamma=None if prev_p is None else prev_p.Gamma,
        )
        vm_mt = solve_vertex_q0(
            gw.G, gw.W, vq0, k_minus, grid, mt_opts,
            initial_gamma=None if prev_m is None else prev_m.Gamma,
        )
        time_mt = time.perf_counter() - t0
        chi_plus_mt = chi_eta(gw.G, k_plus, grid, Gamma=vp_mt.Gamma)
        chi_minus_mt = chi_eta(gw.G, k_minus, grid, Gamma=vm_mt.Gamma)

    if args.vertex_stage in ("full", "both"):
        t0 = time.perf_counter()
        full_opts = _vertex_options(args, include_al=True)
        if args.vertex_stage == "both":
            init_p = vp_mt.Gamma
            init_m = vm_mt.Gamma
        else:
            prev_p = None if args.no_continuation else state.get("full_plus")
            prev_m = None if args.no_continuation else state.get("full_minus")
            init_p = None if prev_p is None else prev_p.Gamma
            init_m = None if prev_m is None else prev_m.Gamma
        vp_full = solve_vertex_q0(
            gw.G, gw.W, vq0, k_plus, grid, full_opts,
            initial_gamma=init_p,
        )
        vm_full = solve_vertex_q0(
            gw.G, gw.W, vq0, k_minus, grid, full_opts,
            initial_gamma=init_m,
        )
        time_full = time.perf_counter() - t0
        chi_plus_full = chi_eta(gw.G, k_plus, grid, Gamma=vp_full.Gamma)
        chi_minus_full = chi_eta(gw.G, k_minus, grid, Gamma=vm_full.Gamma)

    if args.vertex_stage == "mt":
        selected_name = "GW+MT"
        chi_plus = chi_plus_mt
        chi_minus = chi_minus_mt
        vp_selected, vm_selected = vp_mt, vm_mt
    else:
        selected_name = "full cGW"
        chi_plus = chi_plus_full
        chi_minus = chi_minus_full
        vp_selected, vm_selected = vp_full, vm_full

    r_plus = _safe_inverse(chi_plus)
    r_minus = _safe_inverse(chi_minus)
    delta_r = r_minus - r_plus

    row = {
        "filling": float(filling),
        "V": float(args.V),
        "T": float(args.T),
        "nk": int(args.nk),
        "nw": int(args.nw),
        "nOmega": int(args.nomega),
        "vertex_stage": args.vertex_stage,
        "selected_stage": selected_name,
        "momentum_backend": args.momentum_backend,
        "mu0": float(bare.mu),
        "mu_GW": float(gw.mu),
        "actual_filling": float(np.sum(gw.density)),
        "GW_converged": bool(gw.converged),
        "GW_iterations": int(gw.iterations),
        "selected_plus_converged": bool(vp_selected.converged),
        "selected_minus_converged": bool(vm_selected.converged),
        "selected_plus_iterations": int(vp_selected.iterations),
        "selected_minus_iterations": int(vm_selected.iterations),
        "time_bare_s": float(time_bare),
        "time_GW_s": float(time_gw),
        "time_MT_s": float(time_mt),
        "time_full_s": float(time_full),
        "runtime_s": float(time.perf_counter() - point_start),
    }

    for prefix, zp, zm in [
        ("G0G0", chi_plus_g0, chi_minus_g0),
        ("GG", chi_plus_gg, chi_minus_gg),
        ("GW_MT", chi_plus_mt, chi_minus_mt),
        ("full_cGW", chi_plus_full, chi_minus_full),
        ("selected", chi_plus, chi_minus),
    ]:
        if zp is None or zm is None:
            row[f"{prefix}_opposite_re"] = np.nan
            row[f"{prefix}_opposite_im"] = np.nan
            row[f"{prefix}_same_re"] = np.nan
            row[f"{prefix}_same_im"] = np.nan
            continue
        p_re, p_im = _parts(zp)
        m_re, m_im = _parts(zm)
        row[f"{prefix}_opposite_re"] = p_re
        row[f"{prefix}_opposite_im"] = p_im
        row[f"{prefix}_same_re"] = m_re
        row[f"{prefix}_same_im"] = m_im

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
    })

    new_state = {
        "mu0": bare.mu,
        "gw": gw if gw.converged else None,
        "mt_plus": vp_mt if vp_mt is not None and vp_mt.converged else None,
        "mt_minus": vm_mt if vm_mt is not None and vm_mt.converged else None,
        "full_plus": vp_full if vp_full is not None and vp_full.converged else None,
        "full_minus": vm_full if vm_full is not None and vm_full.converged else None,
    }

    print(
        f"filling={filling:6.3f}  "
        f"chi_opp={complex(chi_plus).real: .8g}  "
        f"chi_same={complex(chi_minus).real: .8g}  "
        f"r_opp={rp_re: .8g}  r_same={rm_re: .8g}  "
        f"GW it={gw.iterations:3d}  "
        f"vertex it=({vp_selected.iterations:3d},{vm_selected.iterations:3d})  "
        f"time={row['runtime_s']:.1f}s"
    )
    if not gw.converged:
        print("  WARNING: GW did not converge at this filling.")
    if not (vp_selected.converged and vm_selected.converged):
        print("  WARNING: at least one selected vertex did not converge.")

    return row, new_state


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _integer_guides(ax):
    for n in range(1, 6):
        ax.axvline(n, linewidth=0.7, alpha=0.25)


def _plot(rows: list[dict], outdir: Path, V: float, stage_label: str) -> None:
    good = [r for r in rows if np.isfinite(r["r_eff_opposite_re"]) and np.isfinite(r["r_eff_same_re"])]
    if not good:
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

    p.add_argument("--vertex-stage", choices=["mt", "full", "both"], default="mt")
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--no-continuation", action="store_true")
    p.add_argument("--skip-hartree", action="store_true")
    p.add_argument("--fail-fast", action="store_true")

    # V=3 is much stronger than the convergence examples around V=0.1, so the
    # default mixing is deliberately conservative.
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

    if np.any(fillings <= 0.0) or np.any(fillings >= 6.0):
        raise ValueError("For the six-site spinless cell use fillings strictly between 0 and 6.")

    if args.outdir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path("results") / "filling" / stamp
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(
        nk1=args.nk,
        nk2=args.nk,
        nw=args.nw,
        nOmega=args.nomega,
        T=args.T,
    )

    rows: list[dict] = []
    state: dict | None = None
    total_start = time.perf_counter()

    print("=" * 78)
    print("RubycGW filling scan")
    print(
        f"V={args.V}, T={args.T}, nk={args.nk}x{args.nk}, nw={args.nw}, "
        f"nOmega={args.nomega}, stage={args.vertex_stage}, backend={args.momentum_backend}"
    )
    print(f"number of fillings: {len(fillings)}")
    print("r_eff is defined as 1/chi for the physical eta order parameter.")
    print("+ = physical opposite, - = physical same")
    print("=" * 78)

    for filling in fillings:
        try:
            row, new_state = _run_point(args, grid, params, float(filling), state)
            rows.append(row)
            if not args.no_continuation:
                state = new_state
        except Exception as exc:
            print(f"ERROR at filling={filling:.6g}: {exc}")
            if args.fail_fast:
                raise
            rows.append({
                "filling": float(filling),
                "V": float(args.V),
                "T": float(args.T),
                "nk": int(args.nk),
                "nw": int(args.nw),
                "nOmega": int(args.nomega),
                "vertex_stage": args.vertex_stage,
                "selected_stage": "GW+MT" if args.vertex_stage == "mt" else "full cGW",
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
                "time_MT_s": np.nan,
                "time_full_s": np.nan,
                "runtime_s": np.nan,
                "G0G0_opposite_re": np.nan,
                "G0G0_opposite_im": np.nan,
                "G0G0_same_re": np.nan,
                "G0G0_same_im": np.nan,
                "GG_opposite_re": np.nan,
                "GG_opposite_im": np.nan,
                "GG_same_re": np.nan,
                "GG_same_im": np.nan,
                "GW_MT_opposite_re": np.nan,
                "GW_MT_opposite_im": np.nan,
                "GW_MT_same_re": np.nan,
                "GW_MT_same_im": np.nan,
                "full_cGW_opposite_re": np.nan,
                "full_cGW_opposite_im": np.nan,
                "full_cGW_same_re": np.nan,
                "full_cGW_same_im": np.nan,
                "selected_opposite_re": np.nan,
                "selected_opposite_im": np.nan,
                "selected_same_re": np.nan,
                "selected_same_im": np.nan,
                "r_eff_opposite_re": np.nan,
                "r_eff_opposite_im": np.nan,
                "r_eff_same_re": np.nan,
                "r_eff_same_im": np.nan,
                "delta_r_same_minus_opposite_re": np.nan,
                "delta_r_same_minus_opposite_im": np.nan,
            })

        _write_csv(rows, outdir / "filling_scan.csv")

    stage_label = "GW+MT" if args.vertex_stage == "mt" else "full cGW"
    _plot(rows, outdir, args.V, stage_label)

    elapsed = time.perf_counter() - total_start
    print("\n=== filling scan finished ===")
    print(f"total time: {elapsed:.1f} s")
    print("output directory:", outdir)
    print("CSV:", outdir / "filling_scan.csv")
    print("figure:", outdir / "r_eff_vs_filling.png")
    print("figure:", outdir / "chi_vs_filling.png")
    print("figure:", outdir / "delta_r_vs_filling.png")


if __name__ == "__main__":
    main()
