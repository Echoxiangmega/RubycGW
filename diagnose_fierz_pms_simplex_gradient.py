#!/usr/bin/env python3
"""Diagnose whether a 1D Fierz-PMS root is stationary in the full n/B/J simplex.

This script reads an NPZ produced by ``benchmark_fierz_pms_target.py`` and, for
each stored PMS root, evaluates the explicit fixed-G derivatives

    g_B = dF/dlambda_B,
    g_J = dF/dlambda_J,

with lambda_n=1-lambda_B-lambda_J.  The existing one-parameter PMS line has
lambda_B=lambda_J=(1-lambda)/2 and therefore only enforces

    dF/dlambda = -(g_B+g_J)/2 = 0.

The perpendicular component

    g_perp = (g_B-g_J)/sqrt(2)

is the decisive test of whether the root is also stationary away from that
symmetric line.  No cGW vertex solve or ED diagonalization is repeated.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rubycgw.fierz_pms import WeightedFierzPMSResidual
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-npz", type=Path, required=True)
    p.add_argument(
        "--lambda-fd-h",
        type=float,
        default=None,
        help="optional 2D simplex finite-difference step; default uses the PMS checkpoint value",
    )
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    args = _args()
    with np.load(args.target_npz, allow_pickle=False) as data:
        mode = str(np.asarray(data["mode"]).item())
        if mode != "weighted_nbj_pms":
            raise RuntimeError("input NPZ is not a weighted_nbj_pms target result")
        V = float(np.asarray(data["V"]).item())
        Y = np.asarray(data["X_target"], dtype=float)
        source_checkpoint = Path(str(np.asarray(data["source_checkpoint"]).item()))
        stored_line_dF = np.asarray(data["dF_dlambda_per_cell"], dtype=float)
        stored_lambda = np.asarray(data["lambda_star"], dtype=float)

    if Y.ndim == 1:
        Y = Y[None, :]
    if not source_checkpoint.exists():
        raise RuntimeError(
            f"source PMS checkpoint not found: {source_checkpoint}. "
            "Run from the same repository/results tree or restore the checkpoint."
        )

    with np.load(source_checkpoint, allow_pickle=False) as data:
        signature = str(np.asarray(data["signature"]).item())
    meta = json.loads(signature)
    if str(meta.get("mode", "")).lower() != "weighted_nbj_pms":
        raise RuntimeError("source checkpoint is not a PMS continuation checkpoint")

    params = RubyParameters(
        ti=float(meta["ti"]),
        t1=float(meta["t1"]),
        t2=float(meta["t2"]),
        V=0.0,
    )
    geometry = ExactSmallRubyThermal(int(meta["L1"]), int(meta["L2"]), params)
    grid = MatsubaraGrid(
        nk1=1,
        nk2=1,
        nw=int(meta["nw"]),
        nOmega=int(meta["nomega"]),
        T=float(meta["T"]),
    )
    h0 = np.asarray(geometry.h0, dtype=complex)[None, None]
    npc = int(meta["L1"]) * int(meta["L2"])
    problem = WeightedFierzPMSResidual(
        h0,
        geometry.interaction_pairs,
        grid,
        float(meta["target"]),
        primitive_cells=npc,
        lambda_fd_h=float(meta["lambda_fd_h"]),
        lambda_margin=float(meta["lambda_margin"]),
        free_energy_scale_floor=float(meta["pms_scale_floor"]),
    )
    if Y.shape[1] != problem.codec.size:
        raise RuntimeError(
            f"stored PMS state size {Y.shape[1]} does not match reconstructed codec {problem.codec.size}"
        )

    print("=== Full two-dimensional Fierz-simplex gradient diagnostic ===")
    print(f"target={args.target_npz}")
    print(f"V={V:g}, roots={Y.shape[0]}")
    print("coordinates: b=lambda_B, j=lambda_J, lambda_n=1-b-j")
    print("1D PMS checks only -(gB+gJ)/2=0; full stationarity also requires gB-gJ=0.\n")

    rows = []
    for ir, y in enumerate(Y):
        ev = problem.evaluate(y, V)
        grad = problem.simplex_gradient(
            y,
            V,
            h=args.lambda_fd_h,
            evaluation=ev,
        )
        lam = float(ev.lambda_value)
        b = 0.5 * (1.0 - lam)
        j = b
        print(f"root {ir}: lambda*={lam:.10f}, weights=(n,B,J)=({lam:.6f},{b:.6f},{j:.6f})")
        print(
            f"  gB=dF/dlambda_B/cell={grad.dF_dlambda_B_per_cell:+.9e}, "
            f"gJ=dF/dlambda_J/cell={grad.dF_dlambda_J_per_cell:+.9e}"
        )
        print(
            f"  grad_norm={grad.gradient_norm_per_cell:.9e}, "
            f"parallel={grad.parallel_per_cell:+.9e}, "
            f"perp=(gB-gJ)/sqrt(2)={grad.perpendicular_per_cell:+.9e}"
        )
        print(
            f"  reconstructed dF/dlambda={grad.dF_dlambda_reconstructed_per_cell:+.9e}, "
            f"stored/evaluated={ev.dF_dlambda_per_cell:+.9e}, "
            f"consistency={grad.line_consistency_error_per_cell:+.3e}, h={grad.fd_h_used:.3e}"
        )
        if abs(grad.perpendicular_per_cell) <= max(1e-6, 10.0 * abs(grad.parallel_per_cell)):
            verdict = "near-full-stationary" if grad.gradient_norm_per_cell < 1e-4 else "perp-small-but-gradient-not-tiny"
        else:
            verdict = "1D-only-stationary"
        print(f"  verdict: {verdict}\n")

        rows.append({
            "root": ir,
            "V": f"{V:.16g}",
            "lambda_star": f"{lam:.16g}",
            "lambda_n": f"{lam:.16g}",
            "lambda_B": f"{b:.16g}",
            "lambda_J": f"{j:.16g}",
            "gB_per_cell": f"{grad.dF_dlambda_B_per_cell:.16g}",
            "gJ_per_cell": f"{grad.dF_dlambda_J_per_cell:.16g}",
            "gradient_norm_per_cell": f"{grad.gradient_norm_per_cell:.16g}",
            "parallel_per_cell": f"{grad.parallel_per_cell:.16g}",
            "perpendicular_per_cell": f"{grad.perpendicular_per_cell:.16g}",
            "dF_dlambda_reconstructed_per_cell": f"{grad.dF_dlambda_reconstructed_per_cell:.16g}",
            "dF_dlambda_target_npz_per_cell": f"{stored_line_dF[ir]:.16g}",
            "dF_dlambda_evaluated_per_cell": f"{ev.dF_dlambda_per_cell:.16g}",
            "line_consistency_error_per_cell": f"{grad.line_consistency_error_per_cell:.16g}",
            "fd_h_used": f"{grad.fd_h_used:.16g}",
            "lambda_target_npz": f"{stored_lambda[ir]:.16g}",
            "verdict": verdict,
        })

    outfile = args.out
    if outfile is None:
        outfile = args.target_npz.with_name(args.target_npz.stem + "_simplex_gradient.csv")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {outfile}")


if __name__ == "__main__":
    main()
