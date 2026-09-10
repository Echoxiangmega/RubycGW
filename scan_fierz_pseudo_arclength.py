#!/usr/bin/env python3
"""Track a density/B/J/BJ GW fixed-point branch by pseudo-arclength continuation.

This is a solver/branch diagnostic, not an ED calculation.  Two weak-coupling GW
solutions seed a secant tangent.  Subsequent points solve the augmented system

    R_GW(x,V) = 0,
    t . [(x,V)-(x_pred,V_pred)] = 0,

with a matrix-free Newton-Krylov corrector and backtracking line search.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rubycgw.fierz_channel_gw import build_channel_definition, solve_channel_gw_same_torus
from rubycgw.fierz_pac import FierzGWResidual, soft_mode_overlap
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.pseudo_arclength import PACOptions, pseudo_arclength_step
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["density", "J", "B", "BJ"], default="J")
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)

    p.add_argument("--V-start", type=float, default=0.05)
    p.add_argument("--V-second", type=float, default=0.10)
    p.add_argument("--V-stop", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--ds", type=float, default=0.03)
    p.add_argument("--ds-min", type=float, default=0.001)
    p.add_argument("--ds-max", type=float, default=0.08)
    p.add_argument("--parameter-scale", type=float, default=1.0)

    p.add_argument("--seed-max-iter", type=int, default=2500)
    p.add_argument("--seed-tol", type=float, default=2e-9)
    p.add_argument("--seed-mixing", type=float, default=0.04)
    p.add_argument("--seed-mixing-method", choices=["linear", "pulay"], default="pulay")

    p.add_argument("--newton-tol", type=float, default=1e-8)
    p.add_argument("--newton-max", type=int, default=10)
    p.add_argument("--fd-eps", type=float, default=1e-6)
    p.add_argument("--gmres-rtol", type=float, default=2e-3)
    p.add_argument("--gmres-maxiter", type=int, default=24)
    p.add_argument("--gmres-restart", type=int, default=12)
    p.add_argument("--line-search-min", type=float, default=1.0 / 1024.0)
    p.add_argument("--screening-floor", type=float, default=1e-8)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/fierz_pseudo_arclength"))
    return p.parse_args()


def _signature(args, norb, target):
    return json.dumps({
        "mode": args.mode.lower(), "L1": args.L1, "L2": args.L2,
        "filling": args.filling, "target": target, "T": args.T,
        "ti": args.ti, "t1": args.t1, "t2": args.t2,
        "nw": args.nw, "nomega": args.nomega, "norb": norb,
    }, sort_keys=True)


def _write_history(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _read_history(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _diag_row(step, kind, V, x, ev, definition, operators, ds, tangent_v,
              newton=0, gmres=0, ls=0, pac_norm=0.0):
    return {
        "step": int(step), "kind": kind, "V": f"{V:.16g}",
        "mu": f"{x[-1]:.16g}", "ds": f"{ds:.8g}",
        "pac_residual": f"{pac_norm:.8e}",
        "physical_residual": f"{ev.physical_residual:.8e}",
        "filling_error": f"{ev.filling_error:.8e}",
        "smin": f"{ev.smin:.8e}", "soft_m": int(ev.soft_m),
        "soft_Omega": f"{ev.soft_Omega:.8e}",
        "soft_overlap_same": f"{soft_mode_overlap(operators[0], definition, ev.soft_mode):.8e}",
        "soft_overlap_opposite": f"{soft_mode_overlap(operators[1], definition, ev.soft_mode):.8e}",
        "tangent_V": f"{tangent_v:.8e}", "newton_iter": int(newton),
        "gmres_iter": int(gmres), "line_search_reductions": int(ls),
    }


def _save_checkpoint(path, signature, x_prev, V_prev, x_curr, V_curr, ds, ev):
    sigma_static, sigma_c, mu = codec_global.decode(x_curr)
    np.savez_compressed(
        path, signature=np.asarray(signature), x_prev=x_prev, V_prev=V_prev,
        x_curr=x_curr, V_curr=V_curr, ds=ds, mu=mu,
        Sigma_static=sigma_static, Sigma_c=sigma_c, G=ev.G,
        P=ev.P, smin=ev.smin, soft_m=ev.soft_m, soft_Omega=ev.soft_Omega,
        soft_mode=ev.soft_mode,
    )


# Set in main; kept module-level only so checkpoint writing stays compact.
codec_global = None


def main():
    global codec_global
    args = _args()
    if args.V_second == args.V_start:
        raise ValueError("V-start and V-second must differ")
    if min(args.ds, args.ds_min, args.ds_max) <= 0 or args.ds_min > args.ds_max:
        raise ValueError("invalid arclength step bounds")

    ncell = int(args.L1) * int(args.L2)
    target = float(args.filling) * ncell
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    geometry = ExactSmallRubyThermal(args.L1, args.L2, params)
    norb = int(geometry.n_sites)
    h0 = np.asarray(geometry.h0, dtype=complex)[None, None]
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=args.nw, nOmega=args.nomega, T=args.T)
    problem = FierzGWResidual(h0, geometry.interaction_pairs, args.mode, grid, target)
    codec_global = problem.codec
    operators = np.stack([
        np.asarray(geometry.pseudospin_operator(ch, (0.0, 0.0)), dtype=complex)
        for ch in CHANNELS
    ])

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode.lower()}_{args.L1}x{args.L2}_fill{args.filling:g}_T{args.T:g}"
    checkpoint = args.out / f"{tag}_checkpoint.npz"
    history_file = args.out / f"{tag}_history.csv"
    signature = _signature(args, norb, target)

    pac_opts = PACOptions(
        tol=args.newton_tol, max_newton=args.newton_max, fd_eps=args.fd_eps,
        gmres_rtol=args.gmres_rtol, gmres_maxiter=args.gmres_maxiter,
        gmres_restart=args.gmres_restart, line_search_min=args.line_search_min,
        verbose=args.verbose,
    )

    print("=== Fierz GW pseudo-arclength continuation ===")
    print(
        f"mode={args.mode}, torus={args.L1}x{args.L2} ({norb} sites), "
        f"filling={args.filling:g}, T={args.T:g}, nw={args.nw}, nOmega={args.nomega}"
    )
    print(
        f"V seeds=({args.V_start:g},{args.V_second:g}), V_stop={args.V_stop:g}, "
        f"ds={args.ds:g} [{args.ds_min:g},{args.ds_max:g}]"
    )

    rows = []
    if args.resume and checkpoint.exists():
        data = np.load(checkpoint, allow_pickle=False)
        got = str(np.asarray(data["signature"]).item())
        if got != signature:
            raise RuntimeError("checkpoint parameters do not match this run")
        x_prev = np.asarray(data["x_prev"], dtype=float)
        V_prev = float(data["V_prev"])
        x_curr = np.asarray(data["x_curr"], dtype=float)
        V_curr = float(data["V_curr"])
        ds = float(data["ds"])
        rows = _read_history(history_file)
        print(f"resume: V_prev={V_prev:.9f}, V_curr={V_curr:.9f}, ds={ds:.5g}")
    else:
        seed_opts = GWOptions(
            target_filling=target, max_iter=args.seed_max_iter, tol=args.seed_tol,
            mixing=args.seed_mixing, mixing_method=args.seed_mixing_method,
            momentum_backend="direct", verbose=False,
        )
        seeds = []
        for iseed, V in enumerate((args.V_start, args.V_second)):
            definition = build_channel_definition(
                geometry.interaction_pairs, norb, V, args.mode
            )
            print(f"seed {iseed}: solving {args.mode}-GW at V={V:g} ...")
            bg = solve_channel_gw_same_torus(h0, definition, grid, opts=seed_opts)
            if not bg.converged:
                raise RuntimeError(
                    f"seed V={V:g} did not converge: residual={bg.final_error:.3e}; "
                    "move the seeds to weaker coupling or reduce --seed-mixing"
                )
            x = problem.codec.encode_result(bg)
            ev = problem.evaluate(x, V)
            seeds.append((x, float(V), ev))
            print(
                f"  OK iter={bg.iterations}, fixed-point residual={ev.physical_residual:.3e}, "
                f"mu={bg.mu:+.9f}, smin={ev.smin:.3e}"
            )
            rows.append(_diag_row(
                iseed, "seed", V, x, ev, definition, operators,
                args.ds, np.nan, pac_norm=float(np.linalg.norm(ev.residual)),
            ))
        x_prev, V_prev, _ = seeds[0]
        x_curr, V_curr, ev_curr = seeds[1]
        ds = float(args.ds)
        _write_history(history_file, rows)
        _save_checkpoint(
            checkpoint, signature, x_prev, V_prev, x_curr, V_curr, ds, ev_curr
        )

    direction = np.sign(float(args.V_second) - float(args.V_start))
    step_index = len(rows)

    def guarded_residual(x, V):
        if V < 0.0:
            raise ValueError("negative V is outside this physical continuation")
        ev = problem.evaluate(x, V)
        if ev.smin < float(args.screening_floor):
            raise FloatingPointError("screening matrix below numerical floor")
        return ev.residual

    for _ in range(int(args.max_steps)):
        success = False
        trial_ds = ds
        out = None
        for retry in range(int(args.max_retries) + 1):
            try:
                out = pseudo_arclength_step(
                    x_prev, V_prev, x_curr, V_curr, trial_ds,
                    guarded_residual, parameter_scale=args.parameter_scale,
                    opts=pac_opts,
                )
            except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                if args.verbose:
                    print(f"  corrector exception: {exc}")
                out = None
            if out is not None and out.converged:
                success = True
                break
            trial_ds *= 0.5
            print(f"  retry {retry + 1}: corrector failed; ds -> {trial_ds:.5g}")
            if trial_ds < float(args.ds_min):
                break
        if not success:
            print("STOP: pseudo-arclength corrector failed at minimum/retry-limited ds")
            break

        x_new = np.asarray(out.x, dtype=float)
        V_new = float(out.parameter)
        ev = problem.evaluate(x_new, V_new)
        definition = build_channel_definition(
            geometry.interaction_pairs, norb, V_new, args.mode
        )
        tangent_v = float(out.tangent[-1] / float(args.parameter_scale))
        row = _diag_row(
            step_index, "pac", V_new, x_new, ev, definition, operators,
            trial_ds, tangent_v, newton=out.newton_iterations,
            gmres=out.gmres_iterations, ls=out.line_search_reductions,
            pac_norm=out.residual_norm,
        )
        rows.append(row)
        print(
            f"step={step_index:3d} V={V_new:+.9f} ds={trial_ds:.4g} "
            f"PAC={out.residual_norm:.2e} phys={ev.physical_residual:.2e} "
            f"Nerr={ev.filling_error:+.2e} smin={ev.smin:.3e} "
            f"msoft={ev.soft_m:+d} tV={tangent_v:+.3e} "
            f"N/G={out.newton_iterations}/{out.gmres_iterations} "
            f"ov(same,opp)=({float(row['soft_overlap_same']):.3f},"
            f"{float(row['soft_overlap_opposite']):.3f})"
        )

        x_prev, V_prev = x_curr, V_curr
        x_curr, V_curr = x_new, V_new

        if out.newton_iterations <= 3 and out.line_search_reductions <= 1:
            ds = min(float(args.ds_max), 1.3 * trial_ds)
        elif out.newton_iterations >= 8 or out.line_search_reductions >= 4:
            ds = max(float(args.ds_min), 0.6 * trial_ds)
        else:
            ds = trial_ds

        _write_history(history_file, rows)
        _save_checkpoint(
            checkpoint, signature, x_prev, V_prev, x_curr, V_curr, ds, ev
        )
        step_index += 1

        if direction * (V_curr - float(args.V_stop)) >= 0.0:
            print(f"reached requested V_stop={args.V_stop:g}")
            break

    print(f"history: {history_file}")
    print(f"checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
