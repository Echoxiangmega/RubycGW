#!/usr/bin/env python3
"""Scan finite-bath size for the 6-site cluster-ED + GW embedding benchmark.

The scan keeps all physical/model/self-consistency parameters fixed and varies
only ``nbath``.  The ordinary SC-GW background and the exact finite-torus ED
benchmark are cached by ``run_cluster_ed_gw.py`` and therefore reused across the
scan.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nbath-list", type=int, nargs="+", default=[4, 5, 6, 7])
    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=1)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--gw-max", type=int, default=160)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--gw-mixing", type=float, default=0.25)
    p.add_argument("--gw-mixing-method", choices=("linear", "pulay"), default="pulay")
    p.add_argument("--embed-max", type=int, default=100)
    p.add_argument("--embed-tol", type=float, default=2e-5)
    p.add_argument("--embed-mixing", type=float, default=0.80)
    p.add_argument("--embed-mixing-method", choices=("linear", "pulay"), default="pulay")
    p.add_argument("--embed-pulay-history", type=int, default=6)
    p.add_argument("--embed-pulay-start", type=int, default=3)
    p.add_argument("--embed-pulay-regularization", type=float, default=1e-7)
    p.add_argument("--embed-pulay-step-cap", type=float, default=3.0)
    p.add_argument("--impurity-mixing", type=float, default=1.0)
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-fit-max-nfev", type=int, default=300)
    p.add_argument("--bath-energy-window", type=float, default=4.0)
    p.add_argument("--bath-coupling-bound", type=float, default=4.0)
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)
    p.add_argument("--out", type=Path, default=Path("results/cluster_ed_gw_bath_scan"))
    p.add_argument("--cache-dir", type=Path, default=Path("results/cluster_ed_gw/cache"))
    p.add_argument("--refresh-cache", action="store_true")
    return p.parse_args()


def _scalar(z, key, default=np.nan):
    if key not in z:
        return default
    a = np.asarray(z[key])
    if a.size != 1:
        return default
    return a.reshape(()).item()


def _run_one(args, nbath: int, work_out: Path) -> Path:
    cmd = [
        sys.executable,
        "run_cluster_ed_gw.py",
        "--Lx", str(args.Lx),
        "--Ly", str(args.Ly),
        "--V", str(args.V),
        "--ti", str(args.ti),
        "--t1", str(args.t1),
        "--t2", str(args.t2),
        "--filling", str(args.filling),
        "--T", str(args.T),
        "--nw", str(args.nw),
        "--nomega", str(args.nomega),
        "--gw-max", str(args.gw_max),
        "--gw-tol", str(args.gw_tol),
        "--gw-mixing", str(args.gw_mixing),
        "--gw-mixing-method", str(args.gw_mixing_method),
        "--embed-max", str(args.embed_max),
        "--embed-tol", str(args.embed_tol),
        "--embed-mixing", str(args.embed_mixing),
        "--embed-mixing-method", str(args.embed_mixing_method),
        "--embed-pulay-history", str(args.embed_pulay_history),
        "--embed-pulay-start", str(args.embed_pulay_start),
        "--embed-pulay-regularization", str(args.embed_pulay_regularization),
        "--embed-pulay-step-cap", str(args.embed_pulay_step_cap),
        "--impurity-mixing", str(args.impurity_mixing),
        "--nbath", str(nbath),
        "--bath-fit-nfreq", str(args.bath_fit_nfreq),
        "--bath-fit-max-nfev", str(args.bath_fit_max_nfev),
        "--bath-energy-window", str(args.bath_energy_window),
        "--bath-coupling-bound", str(args.bath_coupling_bound),
        "--discard-weight-tol", str(args.discard_weight_tol),
        "--benchmark-ed",
        "--cache-dir", str(args.cache_dir),
        "--out", str(work_out),
        "--quiet-gw",
    ]
    if args.refresh_cache:
        cmd.append("--refresh-cache")

    print(f"\n=== nbath={nbath} ===", flush=True)
    subprocess.run(cmd, check=True)

    produced = work_out / (
        f"cluster_ed_gw_L{args.Lx}x{args.Ly}_V{args.V:.6g}_fill{args.filling:.6g}.npz"
    )
    if not produced.exists():
        raise FileNotFoundError(f"expected run output was not created: {produced}")
    return produced


def main():
    args = _args()
    nbaths = [int(x) for x in args.nbath_list]
    if not nbaths or any(x < 1 for x in nbaths):
        raise ValueError("--nbath-list must contain positive integers")
    if len(set(nbaths)) != len(nbaths):
        raise ValueError("--nbath-list contains duplicates")

    args.out.mkdir(parents=True, exist_ok=True)
    work_out = args.out / "_work"
    work_out.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    saved_files = []
    for nbath in nbaths:
        produced = _run_one(args, nbath, work_out)
        saved = args.out / (
            f"cluster_ed_gw_L{args.Lx}x{args.Ly}_V{args.V:.6g}_fill{args.filling:.6g}_nbath{nbath}.npz"
        )
        shutil.copy2(produced, saved)
        saved_files.append(str(saved))

        with np.load(saved, allow_pickle=False) as z:
            gerr_bg = float(_scalar(z, "Gerr_background"))
            gerr_emb = float(_scalar(z, "Gerr_embedded"))
            row = {
                "nbath": nbath,
                "converged": bool(_scalar(z, "converged", False)),
                "iterations": int(_scalar(z, "iterations", -1)),
                "residual": float(_scalar(z, "final_error")),
                "impurity_mismatch": float(_scalar(z, "impurity_mismatch")),
                "bath_fit": float(_scalar(z, "bath_fit_error")),
                "mu": float(_scalar(z, "mu")),
                "mu_ed": float(_scalar(z, "mu_ed")),
                "Gerr_GW": gerr_bg,
                "Gerr_clusterED_GW": gerr_emb,
                "Gerr_ratio": gerr_emb / gerr_bg if np.isfinite(gerr_bg) and gerr_bg != 0 else np.nan,
                "pulay_fallbacks": int(_scalar(z, "pulay_fallbacks", 0)),
                "file": str(saved),
            }
        rows.append(row)
        print(
            f"nbath={nbath}: conv={row['converged']}, it={row['iterations']}, "
            f"R={row['residual']:.3e}, bath={row['bath_fit']:.3e}, "
            f"Gimp/Gc={row['impurity_mismatch']:.3e}, "
            f"Gerr={row['Gerr_clusterED_GW']:.6e}",
            flush=True,
        )

    csv_path = args.out / (
        f"bath_scan_L{args.Lx}x{args.Ly}_V{args.V:.6g}_fill{args.filling:.6g}.csv"
    )
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    npz_path = args.out / (
        f"bath_scan_L{args.Lx}x{args.Ly}_V{args.V:.6g}_fill{args.filling:.6g}.npz"
    )
    np.savez_compressed(
        npz_path,
        nbath=np.asarray([r["nbath"] for r in rows], dtype=int),
        converged=np.asarray([r["converged"] for r in rows], dtype=bool),
        iterations=np.asarray([r["iterations"] for r in rows], dtype=int),
        residual=np.asarray([r["residual"] for r in rows], dtype=float),
        impurity_mismatch=np.asarray([r["impurity_mismatch"] for r in rows], dtype=float),
        bath_fit=np.asarray([r["bath_fit"] for r in rows], dtype=float),
        mu=np.asarray([r["mu"] for r in rows], dtype=float),
        mu_ed=np.asarray([r["mu_ed"] for r in rows], dtype=float),
        Gerr_GW=np.asarray([r["Gerr_GW"] for r in rows], dtype=float),
        Gerr_clusterED_GW=np.asarray([r["Gerr_clusterED_GW"] for r in rows], dtype=float),
        Gerr_ratio=np.asarray([r["Gerr_ratio"] for r in rows], dtype=float),
        pulay_fallbacks=np.asarray([r["pulay_fallbacks"] for r in rows], dtype=int),
        files=np.asarray(saved_files),
    )

    print("\n=== bath convergence summary ===", flush=True)
    print("nbath  conv   iter    residual    bath_fit    Gimp/Gc      Gerr", flush=True)
    for r in rows:
        print(
            f"{r['nbath']:5d}  {str(r['converged']):5s}  {r['iterations']:5d}  "
            f"{r['residual']:10.3e}  {r['bath_fit']:10.3e}  "
            f"{r['impurity_mismatch']:10.3e}  {r['Gerr_clusterED_GW']:10.3e}",
            flush=True,
        )
    print(f"saved {csv_path}", flush=True)
    print(f"saved {npz_path}", flush=True)


if __name__ == "__main__":
    main()
