#!/usr/bin/env python3
"""Refine sign changes from a transverse PMS scan to exact full-simplex roots.

Input is one ``*_transverse.npz`` produced by ``scan_fierz_pms_transverse.py``.
For every sign change of dF/da, interpolate the stored fermionic state and
(s,a), then solve at fixed physical V the augmented square system

    R_GW = 0,
    dF/ds = 0,
    dF/da = 0.

The result is an interior stationary point in the full n/B/J Fierz simplex.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rubycgw.fierz_pms_full import FullSimplexFierzPMSResidual
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.pseudo_arclength import PACOptions, refine_fixed_parameter
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transverse-npz", type=Path, required=True)
    p.add_argument("--crossing", type=int, default=None, help="refine only this sign-change index")
    p.add_argument("--newton-tol", type=float, default=1e-10)
    p.add_argument("--newton-max", type=int, default=24)
    p.add_argument("--fd-eps", type=float, default=2e-7)
    p.add_argument("--gmres-rtol", type=float, default=3e-4)
    p.add_argument("--gmres-maxiter", type=int, default=120)
    p.add_argument("--gmres-restart", type=int, default=24)
    p.add_argument("--line-search-min", type=float, default=1.0 / 8192.0)
    p.add_argument("--screening-floor", type=float, default=1e-8)
    p.add_argument("--fd-h", type=float, default=None)
    p.add_argument("--simplex-margin", type=float, default=1e-8)
    p.add_argument("--out", type=Path, default=Path("results/fierz_pms_full"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _load_meta(transverse_npz: Path):
    with np.load(transverse_npz, allow_pickle=False) as d:
        V = float(np.asarray(d["V"]).item())
        root_index = int(np.asarray(d["root"]).item())
        a = np.asarray(d["a"], dtype=float)
        s = np.asarray(d["s"], dtype=float)
        Fa = np.asarray(d["dF_da_per_cell"], dtype=float)
        X = np.asarray(d["X_base"], dtype=float)
        source_target = Path(str(np.asarray(d["source_target_npz"]).item()))
    if not source_target.exists():
        raise RuntimeError(f"source target NPZ not found: {source_target}")
    with np.load(source_target, allow_pickle=False) as d:
        source_checkpoint = Path(str(np.asarray(d["source_checkpoint"]).item()))
        F_ed_pc = float(np.asarray(d["F_ed_per_cell"]).item()) if "F_ed_per_cell" in d else np.nan
    if not source_checkpoint.exists():
        raise RuntimeError(f"source checkpoint not found: {source_checkpoint}")
    with np.load(source_checkpoint, allow_pickle=False) as d:
        signature = str(np.asarray(d["signature"]).item())
    meta = json.loads(signature)
    return V, root_index, a, s, Fa, X, source_target, meta, F_ed_pc


def _crossings(a, Fa):
    out = []
    for i in range(len(a) - 1):
        f0, f1 = float(Fa[i]), float(Fa[i + 1])
        if not np.isfinite(f0) or not np.isfinite(f1):
            continue
        if f0 == 0.0 or f1 == 0.0 or np.signbit(f0) != np.signbit(f1):
            out.append((i, i + 1))
    return out


def _reconstruct(meta, V, fd_h, simplex_margin):
    params = RubyParameters(
        ti=float(meta["ti"]), t1=float(meta["t1"]), t2=float(meta["t2"]), V=0.0
    )
    geometry = ExactSmallRubyThermal(int(meta["L1"]), int(meta["L2"]), params)
    grid = MatsubaraGrid(
        nk1=1, nk2=1, nw=int(meta["nw"]), nOmega=int(meta["nomega"]), T=float(meta["T"])
    )
    h0 = np.asarray(geometry.h0, dtype=complex)[None, None]
    npc = int(meta["L1"]) * int(meta["L2"])
    h = float(meta["lambda_fd_h"]) if fd_h is None else float(fd_h)
    problem = FullSimplexFierzPMSResidual(
        h0,
        geometry.interaction_pairs,
        grid,
        float(meta["target"]),
        V=float(V),
        primitive_cells=npc,
        fd_h=h,
        simplex_margin=float(simplex_margin),
        free_energy_scale_floor=float(meta["pms_scale_floor"]),
    )
    return problem


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    args = _args()
    V, root_index, a, s, Fa, X, source_target, meta, F_ed_pc = _load_meta(args.transverse_npz)
    if X.shape[0] != len(a) or len(s) != len(a) or len(Fa) != len(a):
        raise RuntimeError("transverse arrays have inconsistent lengths")
    brackets = _crossings(a, Fa)
    if not brackets:
        raise RuntimeError("no dF/da sign change found in transverse NPZ")
    if args.crossing is not None:
        if args.crossing < 0 or args.crossing >= len(brackets):
            raise IndexError(f"crossing {args.crossing} outside [0,{len(brackets)-1}]")
        brackets = [brackets[args.crossing]]

    problem = _reconstruct(meta, V, args.fd_h, args.simplex_margin)
    opts = PACOptions(
        tol=args.newton_tol,
        max_newton=args.newton_max,
        fd_eps=args.fd_eps,
        gmres_rtol=args.gmres_rtol,
        gmres_maxiter=args.gmres_maxiter,
        gmres_restart=args.gmres_restart,
        line_search_min=args.line_search_min,
        verbose=args.verbose,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    saved = []
    print("=== Exact full-simplex Fierz-PMS refinement ===")
    print(f"source={args.transverse_npz}")
    print(f"V={V:g}, source root={root_index}, sign-change candidates={len(brackets)}")

    for ic, (i0, i1) in enumerate(brackets):
        a0, a1 = float(a[i0]), float(a[i1])
        f0, f1 = float(Fa[i0]), float(Fa[i1])
        alpha = 0.5 if abs(f1 - f0) < 1e-30 else float(np.clip(-f0 / (f1 - f0), 0.0, 1.0))
        ag = (1.0 - alpha) * a0 + alpha * a1
        sg = (1.0 - alpha) * float(s[i0]) + alpha * float(s[i1])
        xg = (1.0 - alpha) * X[i0] + alpha * X[i1]
        zg = problem.encode(xg, sg, ag)

        print(
            f"candidate {ic}: bracket a={a0:+.9f}->{a1:+.9f}, "
            f"Fa={f0:+.3e}->{f1:+.3e}, interpolated a={ag:+.10f}, s={sg:.10f}"
        )

        def guarded(z, _dummy):
            ev = problem.evaluate(z)
            if ev.smin < float(args.screening_floor):
                raise FloatingPointError("screening matrix below numerical floor")
            return ev.residual

        root = refine_fixed_parameter(zg, 0.0, guarded, opts=opts)
        if not root.converged:
            print(f"  FAILED: |R|={root.residual_norm:.3e}")
            continue
        ev = problem.evaluate(root.x)
        sigma_static, sigma_c, mu, ss, aa, _, y = problem.decode(root.x)
        d = ev.transverse
        w = ev.weights
        Fpc = float(ev.free_energy.free_energy_per_primitive_cell)
        Ferr = Fpc - F_ed_pc if np.isfinite(F_ed_pc) else np.nan
        print(
            f"  OK: a*={aa:+.12f}, s*={ss:.12f}, "
            f"w=(n,B,J)=({w.density:.9f},{w.bond:.9f},{w.current:.9f})"
        )
        print(
            f"      Fs={d.dF_ds_per_cell:+.3e}, Fa={d.dF_da_per_cell:+.3e}, "
            f"gB={d.dF_dlambda_B_per_cell:+.3e}, gJ={d.dF_dlambda_J_per_cell:+.3e}, "
            f"|grad|={d.gradient_norm_per_cell:.3e}"
        )
        print(
            f"      F/cell={Fpc:+.12e}, F-F_ED/cell={Ferr:+.6e}, "
            f"phys={ev.physical_residual:.3e}, smin={ev.smin:.3e}, "
            f"Newton={root.newton_iterations}, GMRES={root.gmres_iterations}"
        )

        rows.append({
            "candidate": ic,
            "source_root": root_index,
            "V": f"{V:.16g}",
            "a_star": f"{aa:.16g}",
            "s_star": f"{ss:.16g}",
            "lambda_n": f"{w.density:.16g}",
            "lambda_B": f"{w.bond:.16g}",
            "lambda_J": f"{w.current:.16g}",
            "mu": f"{mu:.16g}",
            "F_per_cell": f"{Fpc:.16g}",
            "F_ED_per_cell": f"{F_ed_pc:.16g}",
            "F_minus_ED_per_cell": f"{Ferr:.16g}",
            "dF_ds_per_cell": f"{d.dF_ds_per_cell:.16g}",
            "dF_da_per_cell": f"{d.dF_da_per_cell:.16g}",
            "gB_per_cell": f"{d.dF_dlambda_B_per_cell:.16g}",
            "gJ_per_cell": f"{d.dF_dlambda_J_per_cell:.16g}",
            "gradient_norm_per_cell": f"{d.gradient_norm_per_cell:.16g}",
            "d2F_ds2_per_cell": f"{ev.longitudinal.d2F_ds2_per_cell:.16g}",
            "d2F_da2_per_cell": f"{d.d2F_da2_per_cell:.16g}",
            "physical_residual": f"{ev.physical_residual:.16g}",
            "smin": f"{ev.smin:.16g}",
            "newton_iterations": root.newton_iterations,
            "gmres_iterations": root.gmres_iterations,
            "residual_norm": f"{root.residual_norm:.16g}",
        })
        saved.append((root.x, np.asarray(y), np.asarray(sigma_static), np.asarray(sigma_c)))

    if not rows:
        raise RuntimeError("all full-simplex refinement candidates failed")

    stem = args.transverse_npz.stem.replace("_transverse", "")
    csvfile = args.out / f"{stem}_full_pms.csv"
    npzfile = args.out / f"{stem}_full_pms.npz"
    _write_csv(csvfile, rows)
    np.savez_compressed(
        npzfile,
        V=float(V),
        source_root=int(root_index),
        a_star=np.asarray([float(r["a_star"]) for r in rows]),
        s_star=np.asarray([float(r["s_star"]) for r in rows]),
        lambda_n=np.asarray([float(r["lambda_n"]) for r in rows]),
        lambda_B=np.asarray([float(r["lambda_B"]) for r in rows]),
        lambda_J=np.asarray([float(r["lambda_J"]) for r in rows]),
        F_per_cell=np.asarray([float(r["F_per_cell"]) for r in rows]),
        F_ed_per_cell=float(F_ed_pc),
        dF_ds_per_cell=np.asarray([float(r["dF_ds_per_cell"]) for r in rows]),
        dF_da_per_cell=np.asarray([float(r["dF_da_per_cell"]) for r in rows]),
        gradient_norm_per_cell=np.asarray([float(r["gradient_norm_per_cell"]) for r in rows]),
        X_full=np.stack([z[0] for z in saved]),
        X_longitudinal=np.stack([z[1] for z in saved]),
        Sigma_static=np.stack([z[2] for z in saved]),
        Sigma_c=np.stack([z[3] for z in saved]),
        source_transverse_npz=np.asarray(str(args.transverse_npz)),
        source_target_npz=np.asarray(str(source_target)),
    )
    print(f"saved {csvfile}")
    print(f"saved {npzfile}")


if __name__ == "__main__":
    main()
