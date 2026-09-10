#!/usr/bin/env python3
"""Scan away from the symmetric B=J PMS line at fixed physical V.

Input is the NPZ produced by ``benchmark_fierz_pms_target.py``. For each chosen
PMS root, define

    s = lambda_B + lambda_J,
    a = lambda_B - lambda_J,

and use ``a`` as the scan coordinate. At every accepted a the fermionic GW state
and s are re-solved self-consistently from

    R_GW = 0,
    dF/ds = 0.

The reported transverse derivative dF/da therefore tests the missing Fierz
direction without freezing the longitudinal PMS coordinate. By default the scan
stops as soon as two consecutive accepted points on the same continued branch
bracket a zero of dF/da. Partial CSV/NPZ output is written after every accepted
point, so an expensive scan always leaves a usable checkpoint. Use ``--scan-full``
only when the full longitudinal-PMS branch is explicitly needed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rubycgw.fierz_pms import WeightedFierzPMSResidual
from rubycgw.fierz_pms_transverse import LongitudinalFierzPMSResidual
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.pseudo_arclength import PACOptions, refine_fixed_parameter
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-npz", type=Path, required=True)
    p.add_argument("--roots", type=int, nargs="*", default=None, help="root indices; default all")
    p.add_argument("--direction", choices=("positive", "negative", "both"), default="positive")
    p.add_argument("--a-step", type=float, default=0.03)
    p.add_argument(
        "--a-max-frac",
        type=float,
        default=0.95,
        help="scan endpoint |a|max = fraction * initial s of each root",
    )
    p.add_argument("--a-min-step", type=float, default=5e-4)
    p.add_argument("--fd-h", type=float, default=None, help="Fierz derivative stencil; default checkpoint value")
    p.add_argument("--simplex-margin", type=float, default=1e-8)
    p.add_argument("--newton-tol", type=float, default=1e-9)
    p.add_argument("--newton-max", type=int, default=18)
    p.add_argument("--fd-eps", type=float, default=3e-7)
    p.add_argument("--gmres-rtol", type=float, default=7e-4)
    p.add_argument("--gmres-maxiter", type=int, default=80)
    p.add_argument("--gmres-restart", type=int, default=20)
    p.add_argument("--line-search-min", type=float, default=1.0 / 4096.0)
    p.add_argument("--screening-floor", type=float, default=1e-8)
    p.add_argument(
        "--scan-full",
        action="store_true",
        help="continue to the requested endpoint after a dF/da crossing; default stops at first crossing",
    )
    p.add_argument("--out", type=Path, default=Path("results/fierz_pms_transverse"))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _write_csv(path: Path, rows):
    if not rows:
        return
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def _source_meta(target_npz: Path):
    with np.load(target_npz, allow_pickle=False) as data:
        mode = str(np.asarray(data["mode"]).item())
        if mode != "weighted_nbj_pms":
            raise RuntimeError("input NPZ is not a weighted_nbj_pms target result")
        V = float(np.asarray(data["V"]).item())
        Y = np.asarray(data["X_target"], dtype=float)
        source_checkpoint = Path(str(np.asarray(data["source_checkpoint"]).item()))
        lambdas = np.asarray(data["lambda_star"], dtype=float)
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
        raise RuntimeError("source checkpoint is not from the PMS continuation")
    return V, Y, lambdas, meta, source_checkpoint


def _reconstruct(meta):
    params = RubyParameters(
        ti=float(meta["ti"]), t1=float(meta["t1"]), t2=float(meta["t2"]), V=0.0
    )
    geometry = ExactSmallRubyThermal(int(meta["L1"]), int(meta["L2"]), params)
    grid = MatsubaraGrid(
        nk1=1, nk2=1, nw=int(meta["nw"]), nOmega=int(meta["nomega"]), T=float(meta["T"])
    )
    h0 = np.asarray(geometry.h0, dtype=complex)[None, None]
    return geometry, grid, h0


def _point_row(root_index, seq, direction, a, y, ev, diag, solver):
    sigma_static, sigma_c, mu, s, _, _ = solver.decode(y, a)
    w = ev.weights
    return {
        "root": int(root_index),
        "seq": int(seq),
        "direction": str(direction),
        "a": f"{float(a):.16g}",
        "s": f"{float(s):.16g}",
        "lambda_n": f"{w.density:.16g}",
        "lambda_B": f"{w.bond:.16g}",
        "lambda_J": f"{w.current:.16g}",
        "mu": f"{mu:.16g}",
        "F_per_cell": f"{ev.free_energy.free_energy_per_primitive_cell:.16g}",
        "dF_ds_per_cell": f"{diag.dF_ds_per_cell:.16g}",
        "dF_da_per_cell": f"{diag.dF_da_per_cell:.16g}",
        "gB_per_cell": f"{diag.dF_dlambda_B_per_cell:.16g}",
        "gJ_per_cell": f"{diag.dF_dlambda_J_per_cell:.16g}",
        "gradient_norm_per_cell": f"{diag.gradient_norm_per_cell:.16g}",
        "d2F_ds2_per_cell": f"{ev.d2F_ds2_per_cell:.16g}",
        "d2F_da2_per_cell": f"{diag.d2F_da2_per_cell:.16g}",
        "fd_h_s": f"{ev.fd_h_s_used:.16g}",
        "fd_h_a": f"{diag.fd_h_a_used:.16g}",
        "physical_residual": f"{ev.physical_residual:.8e}",
        "filling_error": f"{ev.filling_error:.8e}",
        "smin": f"{ev.smin:.8e}",
    }


def _solve_at_a(solver, x_guess, s_guess, a, opts, screening_floor):
    try:
        y_guess = solver.encode(x_guess, s_guess, a)
    except ValueError:
        # If the previous longitudinal point lies below the new |a| boundary,
        # move the initial s minimally into the allowed interval.
        lo, hi = solver.codec.bounds(a)
        s_guess = min(hi - 1e-6, max(lo + 1e-4, 0.5 * (lo + hi)))
        y_guess = solver.encode(x_guess, s_guess, a)

    def guarded(y, aa):
        ev = solver.evaluate(y, aa)
        if ev.smin < float(screening_floor):
            raise FloatingPointError("screening matrix below numerical floor")
        return ev.residual

    root = refine_fixed_parameter(y_guess, float(a), guarded, opts=opts)
    if not root.converged:
        return None
    ev = solver.evaluate(root.x, float(a))
    if ev.smin < float(screening_floor):
        return None
    diag = solver.transverse_diagnostics(root.x, float(a), evaluation=ev)
    x, s, _ = solver.codec.decode(root.x, float(a))
    return np.asarray(root.x), np.asarray(x), float(s), ev, diag, root


def _crosses_zero(f0, f1):
    f0 = float(f0)
    f1 = float(f1)
    if not (np.isfinite(f0) and np.isfinite(f1)):
        return False
    return f0 == 0.0 or f1 == 0.0 or np.signbit(f0) != np.signbit(f1)


def _root_paths(args, ir):
    stem = args.target_npz.stem
    csvfile = args.out / f"{stem}_root{ir}_transverse.csv"
    npzfile = args.out / f"{stem}_root{ir}_transverse.npz"
    return csvfile, npzfile


def _save_root_outputs(*, args, ir, V, points, solver, crossing=None, scan_complete=False):
    """Atomically update per-root CSV/NPZ from all accepted points so far."""
    if not points:
        return [], None, None

    ordered = sorted(points, key=lambda z: z[0])
    rows = []
    for seq, (a, y, x, s, ev, diag, root, direction) in enumerate(ordered):
        rows.append(_point_row(ir, seq, direction, a, y, ev, diag, solver))

    csvfile, npzfile = _root_paths(args, ir)
    _write_csv(csvfile, rows)

    if crossing is None:
        crossing_a = np.empty(0, dtype=float)
        crossing_Fa = np.empty(0, dtype=float)
        crossing_s = np.empty(0, dtype=float)
        crossing_X_base = np.empty((0, np.asarray(ordered[0][2], dtype=float).size), dtype=float)
        crossing_Y = np.empty((0, np.asarray(ordered[0][1], dtype=float).size), dtype=float)
    else:
        left, right = crossing
        crossing_a = np.asarray([float(left[0]), float(right[0])], dtype=float)
        crossing_Fa = np.asarray(
            [float(left[5].dF_da_per_cell), float(right[5].dF_da_per_cell)], dtype=float
        )
        crossing_s = np.asarray([float(left[3]), float(right[3])], dtype=float)
        crossing_X_base = np.stack(
            [np.asarray(left[2], dtype=float), np.asarray(right[2], dtype=float)]
        )
        crossing_Y = np.stack(
            [np.asarray(left[1], dtype=float), np.asarray(right[1], dtype=float)]
        )

    tmp_npz = npzfile.with_name(npzfile.stem + ".tmp.npz")
    save_npz = np.savez_compressed if scan_complete else np.savez
    save_npz(
        tmp_npz,
        V=float(V),
        root=int(ir),
        a=np.asarray([float(r["a"]) for r in rows]),
        s=np.asarray([float(r["s"]) for r in rows]),
        lambda_n=np.asarray([float(r["lambda_n"]) for r in rows]),
        lambda_B=np.asarray([float(r["lambda_B"]) for r in rows]),
        lambda_J=np.asarray([float(r["lambda_J"]) for r in rows]),
        F_per_cell=np.asarray([float(r["F_per_cell"]) for r in rows]),
        dF_ds_per_cell=np.asarray([float(r["dF_ds_per_cell"]) for r in rows]),
        dF_da_per_cell=np.asarray([float(r["dF_da_per_cell"]) for r in rows]),
        gB_per_cell=np.asarray([float(r["gB_per_cell"]) for r in rows]),
        gJ_per_cell=np.asarray([float(r["gJ_per_cell"]) for r in rows]),
        gradient_norm_per_cell=np.asarray([float(r["gradient_norm_per_cell"]) for r in rows]),
        d2F_da2_per_cell=np.asarray([float(r["d2F_da2_per_cell"]) for r in rows]),
        X_base=np.stack([np.asarray(z[2], dtype=float) for z in ordered]),
        Y_longitudinal=np.stack([np.asarray(z[1], dtype=float) for z in ordered]),
        direction=np.asarray([str(z[7]) for z in ordered]),
        crossing_found=np.asarray(crossing is not None),
        crossing_a=crossing_a,
        crossing_Fa_per_cell=crossing_Fa,
        crossing_s=crossing_s,
        crossing_X_base=crossing_X_base,
        crossing_Y_longitudinal=crossing_Y,
        scan_complete=np.asarray(bool(scan_complete)),
        stopped_on_crossing=np.asarray(crossing is not None and not args.scan_full),
        source_target_npz=np.asarray(str(args.target_npz)),
    )
    tmp_npz.replace(npzfile)
    return rows, csvfile, npzfile


def _scan_direction(
    *, root_index, sign, endpoint, solver, x0, s0, origin, opts, args, on_accept,
):
    """Continue one direction until endpoint, failure floor, or first Fa crossing."""
    direction = "positive" if sign > 0 else "negative"
    accepted = []
    current_a = 0.0
    current_x = np.asarray(x0, dtype=float)
    current_s = float(s0)
    previous = origin
    step_nominal = float(args.a_step)
    step = step_nominal
    seq = 1

    while sign * (endpoint - current_a) > 1e-12:
        trial_a = current_a + sign * step
        if sign * (trial_a - endpoint) > 0.0:
            trial_a = endpoint
        solved = _solve_at_a(
            solver, current_x, current_s, trial_a, opts, args.screening_floor
        )
        if solved is None:
            step *= 0.5
            print(f"  {direction}: refine failed at a={trial_a:+.6f}; reduce da -> {step:.6g}")
            if step < float(args.a_min_step):
                print(f"  {direction}: STOP below --a-min-step")
                break
            continue

        y, x, s, ev, diag, root = solved
        item = (float(trial_a), y, x, s, ev, diag, root, direction)
        print(
            f"  {direction} {seq:02d}: a={trial_a:+.6f}, s={s:.6f}, "
            f"w=({ev.weights.density:.5f},{ev.weights.bond:.5f},{ev.weights.current:.5f}), "
            f"Fs={diag.dF_ds_per_cell:+.2e}, Fa={diag.dF_da_per_cell:+.6e}, "
            f"|grad|={diag.gradient_norm_per_cell:.3e}, F/cell={ev.free_energy.free_energy_per_primitive_cell:+.9f}, "
            f"smin={ev.smin:.3e}"
        )

        crossing = None
        if _crosses_zero(previous[5].dF_da_per_cell, diag.dF_da_per_cell):
            crossing = (previous, item)

        accepted.append(item)
        on_accept(item, crossing)

        if crossing is not None:
            print("  >>> transverse dF/da sign change bracketed on the continued PMS branch")
            print(
                f"      a={previous[0]:+.6f}->{item[0]:+.6f}: "
                f"Fa={previous[5].dF_da_per_cell:+.4e}->{item[5].dF_da_per_cell:+.4e}"
            )
            if not args.scan_full:
                print("  >>> saved checkpoint/bracket; stop this root immediately")
                return accepted, crossing, True

        previous = item
        current_a = float(trial_a)
        current_x = np.asarray(x)
        current_s = float(s)
        step = min(step_nominal, 1.25 * step)
        seq += 1

    return accepted, None, False


def main():
    args = _args()
    if args.a_step <= 0.0 or args.a_min_step <= 0.0:
        raise ValueError("a-step and a-min-step must be positive")
    if not 0.0 < args.a_max_frac < 1.0:
        raise ValueError("--a-max-frac must lie in (0,1)")

    V, Y, stored_lambda, meta, source_checkpoint = _source_meta(args.target_npz)
    geometry, grid, h0 = _reconstruct(meta)
    npc = int(meta["L1"]) * int(meta["L2"])
    fd_h = float(meta["lambda_fd_h"]) if args.fd_h is None else float(args.fd_h)

    pms = WeightedFierzPMSResidual(
        h0,
        geometry.interaction_pairs,
        grid,
        float(meta["target"]),
        primitive_cells=npc,
        lambda_fd_h=float(meta["lambda_fd_h"]),
        lambda_margin=float(meta["lambda_margin"]),
        free_energy_scale_floor=float(meta["pms_scale_floor"]),
    )
    if Y.shape[1] != pms.codec.size:
        raise RuntimeError("stored PMS state size does not match source checkpoint parameters")

    solver = LongitudinalFierzPMSResidual(
        h0,
        geometry.interaction_pairs,
        grid,
        float(meta["target"]),
        V=V,
        primitive_cells=npc,
        fd_h=fd_h,
        simplex_margin=float(args.simplex_margin),
        free_energy_scale_floor=float(meta["pms_scale_floor"]),
    )
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

    roots = list(range(Y.shape[0])) if args.roots is None or len(args.roots) == 0 else list(args.roots)
    for ir in roots:
        if ir < 0 or ir >= Y.shape[0]:
            raise IndexError(f"root index {ir} outside [0,{Y.shape[0]-1}]")

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    print("=== Self-consistent transverse Fierz-PMS scan ===")
    print(f"target={args.target_npz}")
    print(f"source checkpoint={source_checkpoint}")
    print(f"V={V:g}, roots={roots}, direction={args.direction}, da={args.a_step:g}, fd_h={fd_h:g}")
    print("At each a: solve R_GW=0 and dF/ds=0; monitor dF/da.")
    if args.scan_full:
        print("scan mode: FULL branch (--scan-full); crossings are saved but do not stop the scan.\n")
    else:
        print("scan mode: STOP at first dF/da crossing; checkpoint after every accepted point.\n")

    for ir in roots:
        base_x, _, lam = pms.codec.decode(Y[ir])
        s0 = 1.0 - float(lam)
        print(f"root {ir}: start lambda*={lam:.10f}, s0={s0:.10f}")
        solved0 = _solve_at_a(solver, base_x, s0, 0.0, opts, args.screening_floor)
        if solved0 is None:
            print("  failed to reproduce a=0 longitudinal PMS point; skip root\n")
            continue
        y0, x0, s0_ref, ev0, diag0, root0 = solved0
        print(
            f"  a=0: s={s0_ref:.10f}, dF/ds={diag0.dF_ds_per_cell:+.3e}, "
            f"dF/da={diag0.dF_da_per_cell:+.9e}, |grad|={diag0.gradient_norm_per_cell:.3e}, "
            f"F/cell={ev0.free_energy.free_energy_per_primitive_cell:+.12e}"
        )

        origin = (0.0, y0, x0, s0_ref, ev0, diag0, root0, "origin")
        points = [origin]
        root_crossing = None

        # Write the first usable NPZ immediately, before any expensive transverse step.
        root_rows, csvfile, npzfile = _save_root_outputs(
            args=args,
            ir=ir,
            V=V,
            points=points,
            solver=solver,
            crossing=None,
            scan_complete=False,
        )
        print(f"  checkpoint {npzfile}")

        if float(diag0.dF_da_per_cell) == 0.0 and not args.scan_full:
            print("  >>> a=0 already has dF/da=0; no transverse scan is needed")
            _save_root_outputs(
                args=args,
                ir=ir,
                V=V,
                points=points,
                solver=solver,
                crossing=(origin, origin),
                scan_complete=True,
            )
            all_rows.extend(root_rows)
            if not args.scan_full:
                print("  >>> first full-PMS zero found; stop the entire scan")
                break
            continue

        endpoint_abs = float(args.a_max_frac) * float(s0_ref)

        def on_accept(item, crossing):
            nonlocal root_crossing
            points.append(item)
            if crossing is not None and root_crossing is None:
                root_crossing = crossing
            _save_root_outputs(
                args=args,
                ir=ir,
                V=V,
                points=points,
                solver=solver,
                crossing=root_crossing,
                scan_complete=False,
            )

        stopped_on_crossing = False
        if args.direction in ("positive", "both"):
            _, crossing, stopped = _scan_direction(
                root_index=ir,
                sign=+1.0,
                endpoint=endpoint_abs,
                solver=solver,
                x0=x0,
                s0=s0_ref,
                origin=origin,
                opts=opts,
                args=args,
                on_accept=on_accept,
            )
            if crossing is not None and root_crossing is None:
                root_crossing = crossing
            stopped_on_crossing = stopped

        if not stopped_on_crossing and args.direction in ("negative", "both"):
            _, crossing, stopped = _scan_direction(
                root_index=ir,
                sign=-1.0,
                endpoint=-endpoint_abs,
                solver=solver,
                x0=x0,
                s0=s0_ref,
                origin=origin,
                opts=opts,
                args=args,
                on_accept=on_accept,
            )
            if crossing is not None and root_crossing is None:
                root_crossing = crossing
            stopped_on_crossing = stopped

        root_rows, csvfile, npzfile = _save_root_outputs(
            args=args,
            ir=ir,
            V=V,
            points=points,
            solver=solver,
            crossing=root_crossing,
            scan_complete=True,
        )
        all_rows.extend(root_rows)

        if root_crossing is not None:
            left, right = root_crossing
            print("  transverse dF/da sign-change candidate saved:")
            print(
                f"    a={left[0]:+.6f}->{right[0]:+.6f}: "
                f"Fa={left[5].dF_da_per_cell:+.4e}->{right[5].dF_da_per_cell:+.4e}"
            )
        else:
            print("  no transverse dF/da sign change on scanned longitudinal-PMS manifold")

        print(f"  saved {csvfile}")
        print(f"  saved {npzfile}\n")

        if stopped_on_crossing and not args.scan_full:
            print("first transverse crossing found; stop the entire scan.")
            break

    if all_rows:
        combined = args.out / f"{args.target_npz.stem}_transverse_allroots.csv"
        _write_csv(combined, all_rows)
        print(f"combined: {combined}")


if __name__ == "__main__":
    main()
