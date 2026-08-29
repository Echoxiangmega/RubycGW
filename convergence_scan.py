#!/usr/bin/env python3
"""Automated convergence scans for RubycGW.

The script varies one numerical cutoff at a time while keeping the other two
at user-selected baseline values.  For every point it runs the same staged
calculation as ``run_ruby_cgw.py``:

    G0G0 -> GG -> GW+MT -> full cGW.

Results are written to a CSV file and one convergence figure is produced for
each scanned cutoff.  Each figure shows the full-cGW opposite susceptibility,
the full-cGW same susceptibility, and their difference.

Examples
--------
Run all three scans with the reference defaults::

    python convergence_scan.py --scan all

Only converge the fermionic Matsubara cutoff::

    python convergence_scan.py --scan nw --nw-values 8 12 16 24 32 48

Use a larger fixed momentum grid while scanning nOmega::

    python convergence_scan.py --scan nomega --base-nk 8 --nomega-values 4 6 8 12 16
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


def _add_response_fields(out: dict, prefix: str, chi_plus: complex, chi_minus: complex) -> None:
    cp_re, cp_im = _complex_parts(chi_plus)
    cm_re, cm_im = _complex_parts(chi_minus)
    delta = chi_minus - chi_plus
    d_re, d_im = _complex_parts(delta)
    out[f"{prefix}_opposite_re"] = cp_re
    out[f"{prefix}_opposite_im"] = cp_im
    out[f"{prefix}_same_re"] = cm_re
    out[f"{prefix}_same_im"] = cm_im
    out[f"{prefix}_delta_re"] = d_re
    out[f"{prefix}_delta_im"] = d_im


def _run_point(args, nk: int, nw: int, nomega: int) -> dict:
    start = time.perf_counter()
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(nk1=nk, nk2=nk, nw=nw, nOmega=nomega, T=args.T)
    _, _, k_plus, k_minus = eta_vertices()

    print("\n" + "=" * 78)
    print(f"point: nk={nk}x{nk}, nw={nw}, nOmega={nomega}, T={args.T}, V={args.V}")
    print("=" * 78)

    bare = solve_noninteracting(
        params,
        grid,
        mu=args.mu0,
        target_filling=args.filling,
    )
    chi_plus_g0 = chi_eta(bare.G0, k_plus, grid)
    chi_minus_g0 = chi_eta(bare.G0, k_minus, grid)

    gw_opts = GWOptions(
        mu=bare.mu,
        target_filling=args.filling,
        max_iter=args.gw_max_iter,
        tol=args.gw_tol,
        mixing=args.gw_mixing,
        verbose=args.verbose_iterations,
    )
    gw = solve_gw(params, grid, gw_opts)
    chi_plus_gg = chi_eta(gw.G, k_plus, grid)
    chi_minus_gg = chi_eta(gw.G, k_minus, grid)

    vq0 = build_interaction(grid.qmesh(), params)[0, 0]

    mt_opts = VertexOptions(
        max_iter=args.vertex_max_iter,
        tol=args.vertex_tol,
        mixing=args.vertex_mixing,
        include_hartree=True,
        include_mt=True,
        include_al=False,
        verbose=args.verbose_iterations,
    )
    vp_mt = solve_vertex_q0(gw.G, gw.W, vq0, k_plus, grid, mt_opts)
    vm_mt = solve_vertex_q0(gw.G, gw.W, vq0, k_minus, grid, mt_opts)
    chi_plus_mt = chi_eta(gw.G, k_plus, grid, Gamma=vp_mt.Gamma)
    chi_minus_mt = chi_eta(gw.G, k_minus, grid, Gamma=vm_mt.Gamma)

    full_opts = VertexOptions(
        max_iter=args.vertex_max_iter,
        tol=args.vertex_tol,
        mixing=args.vertex_mixing,
        include_hartree=True,
        include_mt=True,
        include_al=True,
        verbose=args.verbose_iterations,
    )
    vp_full = solve_vertex_q0(gw.G, gw.W, vq0, k_plus, grid, full_opts)
    vm_full = solve_vertex_q0(gw.G, gw.W, vq0, k_minus, grid, full_opts)
    chi_plus_full = chi_eta(gw.G, k_plus, grid, Gamma=vp_full.Gamma)
    chi_minus_full = chi_eta(gw.G, k_minus, grid, Gamma=vm_full.Gamma)

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
        "mu0": float(bare.mu),
        "mu_GW": float(gw.mu),
        "actual_filling": float(np.sum(gw.density)),
        "GW_converged": bool(gw.converged),
        "GW_iterations": int(gw.iterations),
        "MT_plus_converged": bool(vp_mt.converged),
        "MT_minus_converged": bool(vm_mt.converged),
        "MT_plus_iterations": int(vp_mt.iterations),
        "MT_minus_iterations": int(vm_mt.iterations),
        "full_plus_converged": bool(vp_full.converged),
        "full_minus_converged": bool(vm_full.converged),
        "full_plus_iterations": int(vp_full.iterations),
        "full_minus_iterations": int(vm_full.iterations),
        "GammaH_plus_max": float(np.max(np.abs(vp_full.Gamma_H))),
        "GammaH_minus_max": float(np.max(np.abs(vm_full.Gamma_H))),
        "GammaMT_plus_max": float(np.max(np.abs(vp_full.Gamma_MT))),
        "GammaMT_minus_max": float(np.max(np.abs(vm_full.Gamma_MT))),
        "GammaAL1_plus_max": float(np.max(np.abs(vp_full.Gamma_AL1))),
        "GammaAL1_minus_max": float(np.max(np.abs(vm_full.Gamma_AL1))),
        "GammaAL2_plus_max": float(np.max(np.abs(vp_full.Gamma_AL2))),
        "GammaAL2_minus_max": float(np.max(np.abs(vm_full.Gamma_AL2))),
    }
    _add_response_fields(out, "G0G0", chi_plus_g0, chi_minus_g0)
    _add_response_fields(out, "GG", chi_plus_gg, chi_minus_gg)
    _add_response_fields(out, "GW_MT", chi_plus_mt, chi_minus_mt)
    _add_response_fields(out, "full_cGW", chi_plus_full, chi_minus_full)
    out["runtime_s"] = float(time.perf_counter() - start)

    print(
        "full cGW: "
        f"opposite={out['full_cGW_opposite_re']:.10g}, "
        f"same={out['full_cGW_same_re']:.10g}, "
        f"delta={out['full_cGW_delta_re']:.10g}, "
        f"runtime={out['runtime_s']:.1f} s"
    )
    if not gw.converged:
        print("WARNING: GW did not reach the requested tolerance at this point.")
    if not (vp_full.converged and vm_full.converged):
        print("WARNING: at least one full cGW vertex did not reach the requested tolerance.")

    return out


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
    opposite = np.array([r["full_cGW_opposite_re"] for r in selected])
    same = np.array([r["full_cGW_same_re"] for r in selected])
    delta = np.array([r["full_cGW_delta_re"] for r in selected])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(x, opposite, marker="o", label="opposite (+), full cGW")
    ax.plot(x, same, marker="o", label="same (-), full cGW")
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
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory. Default: results/convergence/<timestamp>.",
    )
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
    cache: dict[tuple[int, int, int], dict] = {}
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

        for nk, nw, nomega, scan_value in specs:
            key = (nk, nw, nomega)
            if key not in cache:
                cache[key] = _run_point(args, nk=nk, nw=nw, nomega=nomega)
            row = {"scan": label, "scan_value": scan_value}
            row.update(cache[key])
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
