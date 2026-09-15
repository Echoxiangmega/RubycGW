#!/usr/bin/env python3
"""Automated convergence scans for RubycGW.

The script varies one cutoff at a time.  It supports three vertex stages:

    --vertex-stage mt    : G0G0 -> GG -> GW+MT only (fast exploratory mode)
    --vertex-stage full  : G0G0 -> GG -> full cGW
    --vertex-stage both  : G0G0 -> GG -> GW+MT -> full cGW

When array shapes are unchanged, the previous converged GW/vertex solutions are
used as continuation guesses.  This is especially effective for nOmega scans
and later for parameter scans in V/filling/T.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
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


def _complex_parts(z: complex) -> tuple[float, float]:
    z = complex(z)
    return float(z.real), float(z.imag)


def _add_response_fields(out: dict, prefix: str, chi_plus, chi_minus) -> None:
    if chi_plus is None or chi_minus is None:
        for suffix in ["opposite_re", "opposite_im", "same_re", "same_im", "delta_re", "delta_im"]:
            out[f"{prefix}_{suffix}"] = float("nan")
        return
    cp_re, cp_im = _complex_parts(chi_plus)
    cm_re, cm_im = _complex_parts(chi_minus)
    d_re, d_im = _complex_parts(chi_minus - chi_plus)
    out[f"{prefix}_opposite_re"] = cp_re
    out[f"{prefix}_opposite_im"] = cp_im
    out[f"{prefix}_same_re"] = cm_re
    out[f"{prefix}_same_im"] = cm_im
    out[f"{prefix}_delta_re"] = d_re
    out[f"{prefix}_delta_im"] = d_im


def _same_fermion_shape(obj, grid: MatsubaraGrid) -> bool:
    if obj is None:
        return False
    expected = (grid.nf, grid.nk1, grid.nk2, 6, 6)
    arr = getattr(obj, "G", None)
    if arr is None:
        arr = getattr(obj, "G0", None)
    return arr is not None and arr.shape == expected


def _run_point(args, nk: int, nw: int, nomega: int, initial_state: dict | None = None):
    total_start = time.perf_counter()
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(nk1=nk, nk2=nk, nw=nw, nOmega=nomega, T=args.T)
    _, _, k_plus, k_minus = eta_vertices()
    initial_state = initial_state or {}

    print("\n" + "=" * 78)
    print(f"point: nk={nk}x{nk}, nw={nw}, nOmega={nomega}, T={args.T}, V={args.V}")
    print(f"vertex stage: {args.vertex_stage}; continuation: {not args.no_continuation}")
    print("=" * 78)

    # Bare reference is independent of nOmega and V.  Reuse it whenever the
    # fermionic grid/hopping/filling are unchanged.
    t0 = time.perf_counter()
    previous_bare = initial_state.get("bare") if not args.no_continuation else None
    if _same_fermion_shape(previous_bare, grid):
        bare = previous_bare
    else:
        bare = solve_noninteracting(
            params, grid, mu=args.mu0, target_filling=args.filling,
        )
    chi_plus_g0 = chi_eta(bare.G0, k_plus, grid)
    chi_minus_g0 = chi_eta(bare.G0, k_minus, grid)
    time_bare = time.perf_counter() - t0

    t0 = time.perf_counter()
    gw_opts = GWOptions(
        mu=bare.mu,
        target_filling=args.filling,
        max_iter=args.gw_max_iter,
        tol=args.gw_tol,
        mixing=args.gw_mixing,
        verbose=args.verbose_iterations,
    )
    initial_gw = initial_state.get("gw") if not args.no_continuation else None
    gw = solve_gw(params, grid, gw_opts, initial=initial_gw)
    time_gw = time.perf_counter() - t0
    chi_plus_gg = chi_eta(gw.G, k_plus, grid)
    chi_minus_gg = chi_eta(gw.G, k_minus, grid)
    vq0 = build_interaction(grid.qmesh(), params)[0, 0]

    mt_opts = VertexOptions(
        max_iter=args.vertex_max_iter,
        tol=args.vertex_tol,
        mixing=args.vertex_mixing,
        include_hartree=not args.skip_hartree,
        include_mt=True,
        include_al=False,
        verbose=args.verbose_iterations,
    )
    full_opts = VertexOptions(
        max_iter=args.vertex_max_iter,
        tol=args.vertex_tol,
        mixing=args.vertex_mixing,
        include_hartree=not args.skip_hartree,
        include_mt=True,
        include_al=True,
        verbose=args.verbose_iterations,
    )

    vp_mt = vm_mt = vp_full = vm_full = None
    chi_plus_mt = chi_minus_mt = None
    chi_plus_full = chi_minus_full = None
    time_mt = 0.0
    time_full = 0.0

    if args.vertex_stage in ("mt", "both"):
        t0 = time.perf_counter()
        prev_p = initial_state.get("mt_plus") if not args.no_continuation else None
        prev_m = initial_state.get("mt_minus") if not args.no_continuation else None
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
        if args.vertex_stage == "both":
            init_p = vp_mt.Gamma
            init_m = vm_mt.Gamma
        else:
            prev_p = initial_state.get("full_plus") if not args.no_continuation else None
            prev_m = initial_state.get("full_minus") if not args.no_continuation else None
            init_p = None if prev_p is None else prev_p.Gamma
            init_m = None if prev_m is None else prev_m.Gamma
        vp_full = solve_vertex_q0(
            gw.G, gw.W, vq0, k_plus, grid, full_opts, initial_gamma=init_p,
        )
        vm_full = solve_vertex_q0(
            gw.G, gw.W, vq0, k_minus, grid, full_opts, initial_gamma=init_m,
        )
        time_full = time.perf_counter() - t0
        chi_plus_full = chi_eta(gw.G, k_plus, grid, Gamma=vp_full.Gamma)
        chi_minus_full = chi_eta(gw.G, k_minus, grid, Gamma=vm_full.Gamma)

    selected_plus = chi_plus_mt if args.vertex_stage == "mt" else chi_plus_full
    selected_minus = chi_minus_mt if args.vertex_stage == "mt" else chi_minus_full
    selected_name = "GW+MT" if args.vertex_stage == "mt" else "full cGW"

    def _vattr(obj, name, default=np.nan):
        return default if obj is None else getattr(obj, name)

    diagnostic_vertex_p = vp_full if vp_full is not None else vp_mt
    diagnostic_vertex_m = vm_full if vm_full is not None else vm_mt

    out = {
        "nk": nk,
        "nw": nw,
        "nOmega": nomega,
        "T": args.T,
        "V": args.V,
        "ti": args.ti,
        "t1": args.t1,
        "t2": args.t2,
        "target_filling": args.filling,
        "vertex_stage": args.vertex_stage,
        "mu0": float(bare.mu),
        "mu_GW": float(gw.mu),
        "actual_filling": float(np.sum(gw.density)),
        "GW_converged": bool(gw.converged),
        "GW_iterations": int(gw.iterations),
        "MT_plus_converged": _vattr(vp_mt, "converged", False),
        "MT_minus_converged": _vattr(vm_mt, "converged", False),
        "MT_plus_iterations": _vattr(vp_mt, "iterations"),
        "MT_minus_iterations": _vattr(vm_mt, "iterations"),
        "full_plus_converged": _vattr(vp_full, "converged", False),
        "full_minus_converged": _vattr(vm_full, "converged", False),
        "full_plus_iterations": _vattr(vp_full, "iterations"),
        "full_minus_iterations": _vattr(vm_full, "iterations"),
        "GammaH_plus_max": float(np.max(np.abs(diagnostic_vertex_p.Gamma_H))),
        "GammaH_minus_max": float(np.max(np.abs(diagnostic_vertex_m.Gamma_H))),
        "GammaMT_plus_max": float(np.max(np.abs(diagnostic_vertex_p.Gamma_MT))),
        "GammaMT_minus_max": float(np.max(np.abs(diagnostic_vertex_m.Gamma_MT))),
        "GammaAL1_plus_max": float(np.max(np.abs(diagnostic_vertex_p.Gamma_AL1))),
        "GammaAL1_minus_max": float(np.max(np.abs(diagnostic_vertex_m.Gamma_AL1))),
        "GammaAL2_plus_max": float(np.max(np.abs(diagnostic_vertex_p.Gamma_AL2))),
        "GammaAL2_minus_max": float(np.max(np.abs(diagnostic_vertex_m.Gamma_AL2))),
        "time_bare_s": float(time_bare),
        "time_GW_s": float(time_gw),
        "time_MT_s": float(time_mt),
        "time_full_s": float(time_full),
    }
    _add_response_fields(out, "G0G0", chi_plus_g0, chi_minus_g0)
    _add_response_fields(out, "GG", chi_plus_gg, chi_minus_gg)
    _add_response_fields(out, "GW_MT", chi_plus_mt, chi_minus_mt)
    _add_response_fields(out, "full_cGW", chi_plus_full, chi_minus_full)
    _add_response_fields(out, "selected", selected_plus, selected_minus)
    out["runtime_s"] = float(time.perf_counter() - total_start)

    print(
        f"{selected_name}: opposite={out['selected_opposite_re']:.10g}, "
        f"same={out['selected_same_re']:.10g}, "
        f"delta={out['selected_delta_re']:.10g}, runtime={out['runtime_s']:.1f} s"
    )
    print(
        f"timing: bare={time_bare:.1f}s, GW={time_gw:.1f}s, "
        f"MT={time_mt:.1f}s, full={time_full:.1f}s"
    )
    if not gw.converged:
        print("WARNING: GW did not reach the requested tolerance at this point.")
    if diagnostic_vertex_p is not None and not (
        diagnostic_vertex_p.converged and diagnostic_vertex_m.converged
    ):
        print("WARNING: at least one requested vertex did not reach tolerance.")

    state = {
        "bare": bare,
        "gw": gw,
        "mt_plus": vp_mt,
        "mt_minus": vm_mt,
        "full_plus": vp_full,
        "full_minus": vm_full,
    }
    return out, state


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_scan(rows: list[dict], scan_name: str, outdir: Path) -> None:
    selected = [r for r in rows if r["scan"] == scan_name]
    if not selected:
        return
    selected.sort(key=lambda r: float(r["scan_value"]))
    x = np.array([float(r["scan_value"]) for r in selected])
    opposite = np.array([r["selected_opposite_re"] for r in selected])
    same = np.array([r["selected_same_re"] for r in selected])
    delta = np.array([r["selected_delta_re"] for r in selected])
    stage = selected[0]["vertex_stage"]
    stage_label = "GW+MT" if stage == "mt" else "full cGW"

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(x, opposite, marker="o", label=f"opposite (+), {stage_label}")
    ax.plot(x, same, marker="o", label=f"same (-), {stage_label}")
    ax.plot(x, delta, marker="o", label="same - opposite")
    ax.set_xlabel(scan_name)
    ax.set_ylabel("static susceptibility")
    ax.set_title(f"RubycGW convergence versus {scan_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"convergence_{scan_name}.png", dpi=180)
    plt.close(fig)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", choices=["all", "nw", "nomega", "nk"], default="all")
    parser.add_argument(
        "--vertex-stage", choices=["mt", "full", "both"], default="both",
        help="mt = fast GW+MT only; full = full cGW only; both = staged MT then full cGW.",
    )
    parser.add_argument(
        "--no-continuation", action="store_true",
        help="Disable reuse of previous converged solutions when shapes match.",
    )
    parser.add_argument(
        "--skip-hartree", action="store_true",
        help="Skip q=0 eta Hartree vertex after its symmetry-zero has been verified.",
    )

    parser.add_argument("--nw-values", nargs="+", type=int, default=[8, 12, 16, 24])
    parser.add_argument("--nomega-values", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--nk-values", nargs="+", type=int, default=[2, 3, 4, 6])
    parser.add_argument("--base-nw", type=int, default=16)
    parser.add_argument("--base-nomega", type=int, default=6)
    parser.add_argument("--base-nk", type=int, default=4)

    parser.add_argument("--ti", type=float, default=0.4)
    parser.add_argument("--t1", type=float, default=0.2)
    parser.add_argument("--t2", type=float, default=0.2)
    parser.add_argument("--V", type=float, default=0.10)
    parser.add_argument("--T", type=float, default=0.05)
    parser.add_argument("--filling", type=float, default=2.0)
    parser.add_argument("--mu0", type=float, default=0.0)

    parser.add_argument("--gw-max-iter", type=int, default=150)
    parser.add_argument("--gw-tol", type=float, default=1e-8)
    parser.add_argument("--gw-mixing", type=float, default=0.20)
    parser.add_argument("--vertex-max-iter", type=int, default=180)
    parser.add_argument("--vertex-tol", type=float, default=1e-8)
    parser.add_argument("--vertex-mixing", type=float, default=0.20)
    parser.add_argument("--verbose-iterations", action="store_true")
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.outdir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path("results") / "convergence" / stamp
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    requested = ["nw", "nomega", "nk"] if args.scan == "all" else [args.scan]
    cache: dict[tuple[int, int, int, str], tuple[dict, dict]] = {}
    rows: list[dict] = []

    for scan in requested:
        if scan == "nw":
            specs = [(args.base_nk, v, args.base_nomega, v) for v in args.nw_values]
            label = "nw"
        elif scan == "nomega":
            specs = [(args.base_nk, args.base_nw, v, v) for v in args.nomega_values]
            label = "nOmega"
        else:
            specs = [(v, args.base_nw, args.base_nomega, v) for v in args.nk_values]
            label = "nk"

        continuation_state = None
        for nk, nw, nomega, scan_value in specs:
            key = (nk, nw, nomega, args.vertex_stage)
            if key not in cache:
                cache[key] = _run_point(
                    args, nk=nk, nw=nw, nomega=nomega,
                    initial_state=continuation_state,
                )
            result, state = cache[key]
            continuation_state = state if not args.no_continuation else None
            row = {"scan": label, "scan_value": scan_value}
            row.update(result)
            rows.append(row)

    csv_path = outdir / "convergence.csv"
    _write_csv(rows, csv_path)
    for scan_name in ["nw", "nOmega", "nk"]:
        _plot_scan(rows, scan_name, outdir)

    print("\n=== convergence scan finished ===")
    print("output directory:", outdir)
    print("CSV:", csv_path)
    for name in ["nw", "nOmega", "nk"]:
        png = outdir / f"convergence_{name}.png"
        if png.exists():
            print("figure:", png)


if __name__ == "__main__":
    main()
