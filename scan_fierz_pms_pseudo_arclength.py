#!/usr/bin/env python3
"""Track a self-consistent Fierz-PMS branch with lambda allowed to vary with V.

The one-parameter exact rewriting is

    lambda_n = lambda,
    lambda_B = lambda_J = (1-lambda)/2.

Unlike ``scan_fierz_weighted_pseudo_arclength.py``, lambda is not fixed here.
The continuation state contains both the fermionic GW state X and a logit
coordinate for lambda.  At every accepted point we solve

    R_GW(X; V, lambda) = 0,
    d F_GW(X; V, lambda) / d lambda = 0,

where the second equation is the principle-of-minimal-sensitivity (PMS)
condition for the fixed-filling Luttinger-Ward Helmholtz free energy.  Pseudo-
arclength continuation then follows the resulting lambda_*(V) branch through
turning points in V when possible.

This is branch-resolved PMS, not minimization of F over different Fierz
representations.  A stationary point may be a minimum or maximum in the
unphysical Fierz direction; the reported curvature distinguishes them.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rubycgw.branch_state_archive import (
    BranchStateArchive,
    append_state_in_memory,
    load_branch_state_archive,
    save_branch_state_archive,
    states_match,
)
from rubycgw.fierz_channel_gw import solve_channel_gw_same_torus
from rubycgw.fierz_mixed import (
    build_weighted_nbj_definition,
    soft_mode_sector_fractions,
    weights_from_lambda,
)
from rubycgw.fierz_pac import soft_mode_overlap
from rubycgw.fierz_pms import WeightedFierzPMSResidual
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.pseudo_arclength import PACOptions, pseudo_arclength_step, refine_fixed_parameter
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lambda-init", type=float, default=0.5)
    p.add_argument("--lambda-fd-h", type=float, default=2e-3)
    p.add_argument("--lambda-margin", type=float, default=1e-8)
    p.add_argument("--pms-scale-floor", type=float, default=1e-4)

    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)

    p.add_argument("--V-start", type=float, default=0.10)
    p.add_argument("--V-second", type=float, default=0.15)
    p.add_argument("--V-stop", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--ds", type=float, default=0.03)
    p.add_argument("--ds-min", type=float, default=5e-5)
    p.add_argument("--ds-max", type=float, default=0.08)
    p.add_argument("--parameter-scale", type=float, default=1.0)

    p.add_argument("--seed-max-iter", type=int, default=3000)
    p.add_argument("--seed-tol", type=float, default=2e-9)
    p.add_argument("--seed-mixing", type=float, default=0.035)
    p.add_argument("--seed-mixing-method", choices=["linear", "pulay"], default="pulay")

    p.add_argument("--newton-tol", type=float, default=1e-8)
    p.add_argument("--newton-max", type=int, default=16)
    p.add_argument("--fd-eps", type=float, default=3e-7)
    p.add_argument("--gmres-rtol", type=float, default=8e-4)
    p.add_argument("--gmres-maxiter", type=int, default=70)
    p.add_argument("--gmres-restart", type=int, default=20)
    p.add_argument("--line-search-min", type=float, default=1.0 / 4096.0)
    p.add_argument("--screening-floor", type=float, default=1e-8)
    p.add_argument("--max-retries", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/fierz_pms_pseudo_arclength"))
    return p.parse_args()


def _signature(args, norb, target):
    return json.dumps({
        "mode": "weighted_nbj_pms",
        "family": "lambda_n=lambda;lambda_B=lambda_J=(1-lambda)/2",
        "L1": args.L1,
        "L2": args.L2,
        "filling": args.filling,
        "target": target,
        "T": args.T,
        "ti": args.ti,
        "t1": args.t1,
        "t2": args.t2,
        "nw": args.nw,
        "nomega": args.nomega,
        "norb": norb,
        "lambda_fd_h": args.lambda_fd_h,
        "lambda_margin": args.lambda_margin,
        "pms_scale_floor": args.pms_scale_floor,
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


def _diag_row(step, kind, V, y, ev, problem, operators, ds, tangent_v,
              newton=0, gmres=0, ls=0, pac_norm=0.0, dlam_dV=np.nan):
    _, _, mu, lam, _ = problem.decode(y)
    weights = weights_from_lambda(lam)
    definition = build_weighted_nbj_definition(
        problem.pairs, problem.norb, V, weights
    )
    frac = soft_mode_sector_fractions(definition, ev.soft_mode)
    thermo = ev.free_energy
    curvature = float(ev.d2F_dlambda2_per_cell)
    if curvature > 0:
        stationary_type = "min"
    elif curvature < 0:
        stationary_type = "max"
    else:
        stationary_type = "flat"
    return {
        "step": int(step),
        "kind": str(kind),
        "V": f"{V:.16g}",
        "lambda_star": f"{lam:.16g}",
        "lambda_n": f"{weights.density:.16g}",
        "lambda_B": f"{weights.bond:.16g}",
        "lambda_J": f"{weights.current:.16g}",
        "mu": f"{mu:.16g}",
        "F_per_cell": f"{thermo.free_energy_per_primitive_cell:.16g}",
        "Omega_per_cell": f"{thermo.grand_potential_per_primitive_cell:.16g}",
        "dF_dlambda_per_cell": f"{ev.dF_dlambda_per_cell:.8e}",
        "d2F_dlambda2_per_cell": f"{curvature:.8e}",
        "lambda_stationary_type": stationary_type,
        "lambda_fd_h_used": f"{ev.fd_h_used:.8e}",
        "pms_scaled_residual": f"{ev.pms_scaled_residual:.8e}",
        "dlam_dV_secant": f"{float(dlam_dV):.8e}",
        "nch": int(len(definition.labels)),
        "ds": f"{ds:.8g}",
        "pac_residual": f"{pac_norm:.8e}",
        "physical_residual": f"{ev.physical_residual:.8e}",
        "filling_error": f"{ev.filling_error:.8e}",
        "smin": f"{ev.smin:.8e}",
        "soft_m": int(ev.soft_m),
        "soft_Omega": f"{ev.soft_Omega:.8e}",
        "soft_sector_n": f"{frac['n']:.8e}",
        "soft_sector_B": f"{frac['B']:.8e}",
        "soft_sector_J": f"{frac['J']:.8e}",
        "soft_overlap_same": f"{soft_mode_overlap(operators[0], definition, ev.soft_mode):.8e}",
        "soft_overlap_opposite": f"{soft_mode_overlap(operators[1], definition, ev.soft_mode):.8e}",
        "tangent_V": f"{tangent_v:.8e}",
        "newton_iter": int(newton),
        "gmres_iter": int(gmres),
        "line_search_reductions": int(ls),
    }


def _save_checkpoint(path, signature, problem, y_prev, V_prev, y_curr, V_curr, ds, ev):
    sigma_static, sigma_c, mu, lam, u = problem.decode(y_curr)
    np.savez_compressed(
        path,
        signature=np.asarray(signature),
        x_prev=np.asarray(y_prev, dtype=float),
        V_prev=float(V_prev),
        x_curr=np.asarray(y_curr, dtype=float),
        V_curr=float(V_curr),
        ds=float(ds),
        lambda_star=float(lam),
        lambda_logit=float(u),
        lambda_n=float(lam),
        lambda_B=float(0.5 * (1.0 - lam)),
        lambda_J=float(0.5 * (1.0 - lam)),
        dF_dlambda_per_cell=float(ev.dF_dlambda_per_cell),
        d2F_dlambda2_per_cell=float(ev.d2F_dlambda2_per_cell),
        F_per_cell=float(ev.free_energy.free_energy_per_primitive_cell),
        Omega_per_cell=float(ev.free_energy.grand_potential_per_primitive_cell),
        mu=float(mu),
        Sigma_static=np.asarray(sigma_static),
        Sigma_c=np.asarray(sigma_c),
        G=np.asarray(ev.G),
        P=np.asarray(ev.P),
        W=np.asarray(ev.W),
        smin=float(ev.smin),
        soft_m=int(ev.soft_m),
        soft_Omega=float(ev.soft_Omega),
        soft_mode=np.asarray(ev.soft_mode),
    )


def _metadata_for_last_two(rows):
    if len(rows) >= 2:
        return (
            [int(rows[-2]["step"]), int(rows[-1]["step"])],
            [str(rows[-2]["kind"]), str(rows[-1]["kind"])],
        )
    return ([-1, 0], ["resume", "resume"])


def _legacy_archive(signature, rows, x_prev, V_prev, x_curr, V_curr):
    steps, kinds = _metadata_for_last_two(rows)
    return BranchStateArchive(
        str(signature),
        np.asarray(steps, dtype=np.int64),
        np.asarray(kinds, dtype=str),
        np.asarray([V_prev, V_curr], dtype=float),
        np.stack([np.asarray(x_prev, dtype=float), np.asarray(x_curr, dtype=float)]),
    )


def _sync_archive_with_checkpoint(archive, rows, x_prev, V_prev, x_curr, V_curr):
    if states_match(archive.X[-1], archive.V[-1], x_curr, V_curr):
        return archive
    if states_match(archive.X[-1], archive.V[-1], x_prev, V_prev):
        if rows:
            step = int(rows[-1]["step"])
            kind = str(rows[-1]["kind"])
        else:
            step = int(archive.step[-1]) + 1
            kind = "resume"
        return append_state_in_memory(archive, step, kind, V_curr, x_curr)
    raise RuntimeError(
        "full-state archive and resume checkpoint disagree; refusing to corrupt PMS branch history"
    )


def main():
    args = _args()
    if not 0.0 < args.lambda_init < 1.0:
        raise ValueError("--lambda-init must lie strictly inside (0,1)")
    if args.V_second == args.V_start:
        raise ValueError("--V-start and --V-second must differ")
    if args.ds <= 0 or args.ds_min <= 0 or args.ds_max < args.ds_min:
        raise ValueError("invalid pseudo-arclength step sizes")

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    geometry = ExactSmallRubyThermal(args.L1, args.L2, params)
    norb = int(geometry.n_sites)
    target = float(args.filling) * int(args.L1) * int(args.L2)
    grid = MatsubaraGrid(
        nk1=1, nk2=1, nw=args.nw, nOmega=args.nomega, T=args.T
    )
    h0 = np.asarray(geometry.h0, dtype=complex)[None, None]
    operators = np.stack([
        np.asarray(geometry.pseudospin_operator(ch, (0.0, 0.0)), dtype=complex)
        for ch in CHANNELS
    ])

    problem = WeightedFierzPMSResidual(
        h0,
        geometry.interaction_pairs,
        grid,
        target,
        primitive_cells=geometry.n_cells,
        lambda_fd_h=args.lambda_fd_h,
        lambda_margin=args.lambda_margin,
        free_energy_scale_floor=args.pms_scale_floor,
    )
    signature = _signature(args, norb, target)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"nbj_pms_{args.L1}x{args.L2}_fill{args.filling:g}_T{args.T:g}"
    checkpoint = args.out / f"{tag}_checkpoint.npz"
    history_file = args.out / f"{tag}_history.csv"
    states_file = args.out / f"{tag}_states.npz"

    pac_opts = PACOptions(
        tol=args.newton_tol,
        max_newton=args.newton_max,
        fd_eps=args.fd_eps,
        gmres_rtol=args.gmres_rtol,
        gmres_maxiter=args.gmres_maxiter,
        gmres_restart=args.gmres_restart,
        line_search_min=args.line_search_min,
        verbose=args.verbose,
    )

    def guarded_residual(y, V):
        ev = problem.evaluate(y, V)
        if ev.smin < float(args.screening_floor):
            raise FloatingPointError("screening matrix below numerical floor")
        return ev.residual

    print("=== Self-consistent weighted n/B/J Fierz PMS continuation ===")
    print(
        f"family: lambda_n=lambda, lambda_B=lambda_J=(1-lambda)/2; "
        f"lambda_init={args.lambda_init:g}"
    )
    print(
        f"torus={args.L1}x{args.L2}, filling={args.filling:g}, T={args.T:g}, "
        f"nw={args.nw}, nOmega={args.nomega}, target N={target:g}"
    )
    print(
        f"PMS: five-point partial dF/dlambda, nominal h={args.lambda_fd_h:g}; "
        f"V seeds=({args.V_start:g},{args.V_second:g}), V_stop={args.V_stop:g}"
    )

    rows = []
    if args.resume and checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as data:
            got = str(np.asarray(data["signature"]).item())
            if got != signature:
                raise RuntimeError("PMS checkpoint parameters do not match this run")
            y_prev = np.asarray(data["x_prev"], dtype=float)
            V_prev = float(data["V_prev"])
            y_curr = np.asarray(data["x_curr"], dtype=float)
            V_curr = float(data["V_curr"])
            ds = float(data["ds"])
        rows = _read_history(history_file)
        if states_file.exists():
            archive = load_branch_state_archive(
                states_file,
                expected_signature=signature,
                expected_state_size=problem.codec.size,
            )
            before = archive.nstate
            archive = _sync_archive_with_checkpoint(
                archive, rows, y_prev, V_prev, y_curr, V_curr
            )
            if archive.nstate != before:
                save_branch_state_archive(
                    states_file, archive.signature, archive.step, archive.kind,
                    archive.V, archive.X,
                )
                print("resume: repaired PMS state archive from newer checkpoint")
        else:
            archive = _legacy_archive(
                signature, rows, y_prev, V_prev, y_curr, V_curr
            )
            save_branch_state_archive(
                states_file, archive.signature, archive.step, archive.kind,
                archive.V, archive.X,
            )
            print("resume: legacy PMS checkpoint had no full-X archive; starting archive now")
        ev_curr = problem.evaluate(y_curr, V_curr)
        print(
            f"resume: V_prev={V_prev:.9f}, V_curr={V_curr:.9f}, "
            f"lambda*={ev_curr.lambda_value:.9f}, ds={ds:.5g}, states={archive.nstate}"
        )
    else:
        seed_opts = GWOptions(
            target_filling=target,
            max_iter=args.seed_max_iter,
            tol=args.seed_tol,
            mixing=args.seed_mixing,
            mixing_method=args.seed_mixing_method,
            momentum_backend="direct",
            verbose=False,
        )
        weights0 = weights_from_lambda(args.lambda_init)
        definition0 = build_weighted_nbj_definition(
            geometry.interaction_pairs, norb, args.V_start, weights0
        )
        print(f"initial ordinary GW seed at V={args.V_start:g}, lambda={args.lambda_init:g} ...")
        bg0 = solve_channel_gw_same_torus(h0, definition0, grid, opts=seed_opts)
        if not bg0.converged:
            raise RuntimeError(
                f"ordinary weighted GW seed failed: residual={bg0.final_error:.3e}; "
                "reduce V-start or seed mixing"
            )
        y_guess = problem.encode_result(bg0, args.lambda_init)

        seeds = []
        for iseed, V in enumerate((args.V_start, args.V_second)):
            if iseed == 0:
                guess = y_guess
            else:
                guess = seeds[-1][0]
            print(f"PMS seed {iseed}: refine [R_GW,dF/dlambda]=0 at V={V:g} ...")
            root = refine_fixed_parameter(guess, V, guarded_residual, opts=pac_opts)
            if not root.converged:
                raise RuntimeError(
                    f"PMS seed V={V:g} failed: |R|={root.residual_norm:.3e}. "
                    "Try a different --lambda-init, slightly larger V seeds, or tighter solver limits."
                )
            ev = problem.evaluate(root.x, V)
            seeds.append((np.asarray(root.x), float(V), ev))
            print(
                f"  OK lambda*={ev.lambda_value:.9f}, dF/dlambda/cell={ev.dF_dlambda_per_cell:+.3e}, "
                f"curvature/cell={ev.d2F_dlambda2_per_cell:+.3e}, "
                f"phys={ev.physical_residual:.3e}, smin={ev.smin:.3e}"
            )
            dldv = np.nan
            if iseed == 1:
                dv = V - seeds[0][1]
                if abs(dv) > 1e-14:
                    dldv = (ev.lambda_value - seeds[0][2].lambda_value) / dv
            rows.append(_diag_row(
                iseed, "seed", V, root.x, ev, problem, operators,
                args.ds, np.nan,
                newton=root.newton_iterations,
                gmres=root.gmres_iterations,
                ls=root.line_search_reductions,
                pac_norm=root.residual_norm,
                dlam_dV=dldv,
            ))

        y_prev, V_prev, _ = seeds[0]
        y_curr, V_curr, ev_curr = seeds[1]
        ds = float(args.ds)
        archive = BranchStateArchive(
            signature,
            np.asarray([0, 1], dtype=np.int64),
            np.asarray(["seed", "seed"], dtype=str),
            np.asarray([V_prev, V_curr], dtype=float),
            np.stack([y_prev, y_curr]),
        )
        _write_history(history_file, rows)
        _save_checkpoint(
            checkpoint, signature, problem, y_prev, V_prev, y_curr, V_curr, ds, ev_curr
        )
        save_branch_state_archive(
            states_file, archive.signature, archive.step, archive.kind,
            archive.V, archive.X,
        )

    direction = 1.0 if args.V_second > args.V_start else -1.0
    if direction * (args.V_stop - args.V_second) < 0.0:
        print("WARNING: V-stop lies opposite the original seed direction; PAC can reach it only after a fold.")

    start_step = int(rows[-1]["step"]) + 1 if rows else 2
    for step in range(start_step, start_step + int(args.max_steps)):
        if direction * (V_curr - args.V_stop) >= 0.0:
            print(f"target V-stop reached at V={V_curr:.12g}")
            break

        accepted = None
        ds_try = float(ds)
        for retry in range(int(args.max_retries) + 1):
            if ds_try < float(args.ds_min) * (1.0 - 1e-12):
                break
            try:
                result = pseudo_arclength_step(
                    y_prev, V_prev, y_curr, V_curr, ds_try,
                    guarded_residual,
                    parameter_scale=args.parameter_scale,
                    opts=pac_opts,
                )
                if result.converged:
                    ev_new = problem.evaluate(result.x, result.parameter)
                    if ev_new.smin < float(args.screening_floor):
                        raise FloatingPointError("screening matrix below numerical floor")
                    accepted = (result, ev_new, ds_try)
                    break
            except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                if args.verbose:
                    print(f"  retry {retry}: {type(exc).__name__}: {exc}")
            ds_try *= 0.5
            print(f"  PAC retry {retry + 1}: reduce ds -> {ds_try:.5g}")

        if accepted is None:
            print("STOP: PMS PAC corrector failed after retries / ds_min.")
            break

        result, ev_new, ds_used = accepted
        y_new = np.asarray(result.x, dtype=float)
        V_new = float(result.parameter)
        lam_old = problem.codec.decode(y_curr)[2]
        lam_new = float(ev_new.lambda_value)
        dv = V_new - V_curr
        dldv = (lam_new - lam_old) / dv if abs(dv) > 1e-14 else np.nan
        tangent_v = float(result.tangent[-1] * args.parameter_scale)

        print(
            f"step {step:4d}: V={V_new:.10f}, lambda*={lam_new:.9f}, "
            f"dF/dlambda/cell={ev_new.dF_dlambda_per_cell:+.2e}, "
            f"curv={ev_new.d2F_dlambda2_per_cell:+.3e}, "
            f"phys={ev_new.physical_residual:.2e}, smin={ev_new.smin:.3e}, "
            f"ds={ds_used:.4g}, tV={tangent_v:+.3f}"
        )

        rows.append(_diag_row(
            step, "pac", V_new, y_new, ev_new, problem, operators,
            ds_used, tangent_v,
            newton=result.newton_iterations,
            gmres=result.gmres_iterations,
            ls=result.line_search_reductions,
            pac_norm=result.residual_norm,
            dlam_dV=dldv,
        ))
        archive = append_state_in_memory(archive, step, "pac", V_new, y_new)
        _write_history(history_file, rows)
        save_branch_state_archive(
            states_file, archive.signature, archive.step, archive.kind,
            archive.V, archive.X,
        )

        # Conservative adaptive arclength step.
        if result.newton_iterations <= 4 and result.line_search_reductions <= 2:
            ds_next = min(float(args.ds_max), 1.25 * ds_used)
        elif result.newton_iterations >= 9 or result.line_search_reductions >= 8:
            ds_next = max(float(args.ds_min), 0.7 * ds_used)
        else:
            ds_next = ds_used

        y_prev, V_prev = y_curr, V_curr
        y_curr, V_curr = y_new, V_new
        ev_curr = ev_new
        ds = ds_next
        _save_checkpoint(
            checkpoint, signature, problem, y_prev, V_prev, y_curr, V_curr, ds, ev_curr
        )
    else:
        print("STOP: reached --max-steps before V-stop.")

    print(f"history: {history_file}")
    print(f"checkpoint: {checkpoint}")
    print(f"states: {states_file}")


if __name__ == "__main__":
    main()
