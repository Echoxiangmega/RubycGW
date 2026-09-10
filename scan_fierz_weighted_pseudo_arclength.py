#!/usr/bin/env python3
"""Track weighted density+B+J Fierz-GW branches by pseudo-arclength.

For each requested lambda in [0,1], use the exact-rewriting family

    lambda_n = lambda,
    lambda_B = lambda_J = (1-lambda)/2,

so lambda_n+lambda_B+lambda_J=1 and the underlying fermionic interaction is
unchanged before truncation.  Density, B and J vertices are solved in one joint
bosonic channel space with full off-diagonal P and W matrices.  Each lambda is
an independent GW approximation/branch continuation, seeded at weak coupling.

This is a branch solver, not a free-energy selector: PAC follows every connected
fixed point it encounters, including stable, metastable and unstable segments.
Every accepted codec state X is also persisted in a separate *_states.npz
archive so arbitrary V crossings can be reconstructed after a long run.
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
    WeightedFierzGWResidual,
    build_weighted_nbj_definition,
    soft_mode_sector_fractions,
    weights_from_lambda,
)
from rubycgw.fierz_pac import soft_mode_overlap
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.pseudo_arclength import PACOptions, pseudo_arclength_step
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lambda", dest="lambdas", nargs="+", type=float, default=[0.5],
        help=(
            "one or more density weights lambda in [0,1]; "
            "lambda_B=lambda_J=(1-lambda)/2"
        ),
    )
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
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--ds", type=float, default=0.03)
    p.add_argument("--ds-min", type=float, default=0.001)
    p.add_argument("--ds-max", type=float, default=0.08)
    p.add_argument("--parameter-scale", type=float, default=1.0)

    p.add_argument("--seed-max-iter", type=int, default=3000)
    p.add_argument("--seed-tol", type=float, default=2e-9)
    p.add_argument("--seed-mixing", type=float, default=0.035)
    p.add_argument("--seed-mixing-method", choices=["linear", "pulay"], default="pulay")

    p.add_argument("--newton-tol", type=float, default=1e-8)
    p.add_argument("--newton-max", type=int, default=14)
    p.add_argument("--fd-eps", type=float, default=3e-7)
    p.add_argument("--gmres-rtol", type=float, default=8e-4)
    p.add_argument("--gmres-maxiter", type=int, default=60)
    p.add_argument("--gmres-restart", type=int, default=18)
    p.add_argument("--line-search-min", type=float, default=1.0 / 4096.0)
    p.add_argument("--screening-floor", type=float, default=1e-8)
    p.add_argument("--max-retries", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/fierz_weighted_pseudo_arclength"))
    return p.parse_args()


def _lambda_tag(lam: float) -> str:
    return f"{float(lam):.6f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _signature(args, norb, target, lam, weights):
    return json.dumps({
        "mode": "weighted_nbj",
        "lambda": float(lam),
        "lambda_n": float(weights.density),
        "lambda_B": float(weights.bond),
        "lambda_J": float(weights.current),
        "L1": args.L1, "L2": args.L2,
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
              weights, newton=0, gmres=0, ls=0, pac_norm=0.0):
    frac = soft_mode_sector_fractions(definition, ev.soft_mode)
    return {
        "step": int(step), "kind": kind, "V": f"{V:.16g}",
        "lambda_n": f"{weights.density:.16g}",
        "lambda_B": f"{weights.bond:.16g}",
        "lambda_J": f"{weights.current:.16g}",
        "nch": int(len(definition.labels)),
        "mu": f"{x[-1]:.16g}", "ds": f"{ds:.8g}",
        "pac_residual": f"{pac_norm:.8e}",
        "physical_residual": f"{ev.physical_residual:.8e}",
        "filling_error": f"{ev.filling_error:.8e}",
        "smin": f"{ev.smin:.8e}", "soft_m": int(ev.soft_m),
        "soft_Omega": f"{ev.soft_Omega:.8e}",
        "soft_sector_n": f"{frac['n']:.8e}",
        "soft_sector_B": f"{frac['B']:.8e}",
        "soft_sector_J": f"{frac['J']:.8e}",
        "soft_overlap_same": f"{soft_mode_overlap(operators[0], definition, ev.soft_mode):.8e}",
        "soft_overlap_opposite": f"{soft_mode_overlap(operators[1], definition, ev.soft_mode):.8e}",
        "tangent_V": f"{tangent_v:.8e}", "newton_iter": int(newton),
        "gmres_iter": int(gmres), "line_search_reductions": int(ls),
    }


def _save_checkpoint(path, signature, problem, x_prev, V_prev, x_curr, V_curr, ds, ev):
    sigma_static, sigma_c, mu = problem.codec.decode(x_curr)
    np.savez_compressed(
        path, signature=np.asarray(signature), x_prev=x_prev, V_prev=V_prev,
        x_curr=x_curr, V_curr=V_curr, ds=ds, mu=mu,
        Sigma_static=sigma_static, Sigma_c=sigma_c, G=ev.G,
        P=ev.P, W=ev.W, smin=ev.smin, soft_m=ev.soft_m,
        soft_Omega=ev.soft_Omega, soft_mode=ev.soft_mode,
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
    """Repair the only expected interrupted-write case: archive one point behind."""
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
        "full-state archive and resume checkpoint disagree; refusing to corrupt branch history"
    )


def _run_lambda(args, lam, geometry, h0, grid, target, norb, operators, pac_opts):
    weights = weights_from_lambda(lam)
    problem = WeightedFierzGWResidual(
        h0, geometry.interaction_pairs, weights, grid, target
    )
    def_probe = problem.definition(args.V_start)

    tag = (
        f"nbj_lam{_lambda_tag(lam)}_{args.L1}x{args.L2}_"
        f"fill{args.filling:g}_T{args.T:g}"
    )
    checkpoint = args.out / f"{tag}_checkpoint.npz"
    history_file = args.out / f"{tag}_history.csv"
    states_file = args.out / f"{tag}_states.npz"
    signature = _signature(args, norb, target, lam, weights)

    print("\n" + "=" * 78)
    print(
        f"lambda={lam:g}: (lambda_n,lambda_B,lambda_J)="
        f"({weights.density:g},{weights.bond:g},{weights.current:g}), "
        f"active nch={len(def_probe.labels)}"
    )
    print(
        f"V seeds=({args.V_start:g},{args.V_second:g}), V_stop={args.V_stop:g}, "
        f"ds={args.ds:g} [{args.ds_min:g},{args.ds_max:g}]"
    )

    rows = []
    if args.resume and checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as data:
            got = str(np.asarray(data["signature"]).item())
            if got != signature:
                raise RuntimeError(f"checkpoint parameters do not match lambda={lam:g} run")
            x_prev = np.asarray(data["x_prev"], dtype=float)
            V_prev = float(data["V_prev"])
            x_curr = np.asarray(data["x_curr"], dtype=float)
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
                archive, rows, x_prev, V_prev, x_curr, V_curr
            )
            if archive.nstate != before:
                save_branch_state_archive(
                    states_file, archive.signature, archive.step, archive.kind,
                    archive.V, archive.X,
                )
                print("resume: repaired full-state archive from newer checkpoint")
        else:
            archive = _legacy_archive(
                signature, rows, x_prev, V_prev, x_curr, V_curr
            )
            save_branch_state_archive(
                states_file, archive.signature, archive.step, archive.kind,
                archive.V, archive.X,
            )
            print(
                "resume: legacy checkpoint had no full-X archive; "
                "recording starts from its last two states"
            )
        print(
            f"resume: V_prev={V_prev:.9f}, V_curr={V_curr:.9f}, ds={ds:.5g}, "
            f"full-X states={archive.nstate}"
        )
    else:
        seed_opts = GWOptions(
            target_filling=target, max_iter=args.seed_max_iter, tol=args.seed_tol,
            mixing=args.seed_mixing, mixing_method=args.seed_mixing_method,
            momentum_backend="direct", verbose=False,
        )
        seeds = []
        for iseed, V in enumerate((args.V_start, args.V_second)):
            definition = build_weighted_nbj_definition(
                geometry.interaction_pairs, norb, V, weights
            )
            print(f"seed {iseed}: solving weighted n/B/J GW at V={V:g} ...")
            bg = solve_channel_gw_same_torus(h0, definition, grid, opts=seed_opts)
            if not bg.converged:
                raise RuntimeError(
                    f"lambda={lam:g} seed V={V:g} did not converge: "
                    f"residual={bg.final_error:.3e}; move seeds weaker or reduce --seed-mixing"
                )
            x = problem.codec.encode_result(bg)
            ev = problem.evaluate(x, V)
            seeds.append((x, float(V), ev))
            frac = soft_mode_sector_fractions(definition, ev.soft_mode)
            print(
                f"  OK iter={bg.iterations}, phys={ev.physical_residual:.3e}, "
                f"mu={bg.mu:+.9f}, smin={ev.smin:.3e}, "
                f"soft(n,B,J)=({frac['n']:.3f},{frac['B']:.3f},{frac['J']:.3f})"
            )
            rows.append(_diag_row(
                iseed, "seed", V, x, ev, definition, operators,
                args.ds, np.nan, weights, pac_norm=float(np.linalg.norm(ev.residual)),
            ))
        x_prev, V_prev, _ = seeds[0]
        x_curr, V_curr, ev_curr = seeds[1]
        ds = float(args.ds)
        archive = BranchStateArchive(
            str(signature),
            np.asarray([0, 1], dtype=np.int64),
            np.asarray(["seed", "seed"], dtype=str),
            np.asarray([V_prev, V_curr], dtype=float),
            np.stack([np.asarray(x_prev, dtype=float), np.asarray(x_curr, dtype=float)]),
        )
        _write_history(history_file, rows)
        _save_checkpoint(
            checkpoint, signature, problem, x_prev, V_prev, x_curr, V_curr, ds, ev_curr
        )
        save_branch_state_archive(
            states_file, archive.signature, archive.step, archive.kind,
            archive.V, archive.X,
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

    reached = False
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
        definition = problem.definition(V_new)
        tangent_v = float(out.tangent[-1] / float(args.parameter_scale))
        row = _diag_row(
            step_index, "pac", V_new, x_new, ev, definition, operators,
            trial_ds, tangent_v, weights,
            newton=out.newton_iterations, gmres=out.gmres_iterations,
            ls=out.line_search_reductions, pac_norm=out.residual_norm,
        )
        rows.append(row)
        print(
            f"step={step_index:3d} V={V_new:+.9f} ds={trial_ds:.4g} "
            f"PAC={out.residual_norm:.2e} phys={ev.physical_residual:.2e} "
            f"smin={ev.smin:.3e} tV={tangent_v:+.3e} "
            f"N/G={out.newton_iterations}/{out.gmres_iterations} "
            f"soft(n,B,J)=({float(row['soft_sector_n']):.3f},"
            f"{float(row['soft_sector_B']):.3f},{float(row['soft_sector_J']):.3f}) "
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
            checkpoint, signature, problem, x_prev, V_prev, x_curr, V_curr, ds, ev
        )
        archive = append_state_in_memory(
            archive, step_index, "pac", V_curr, x_curr
        )
        save_branch_state_archive(
            states_file, archive.signature, archive.step, archive.kind,
            archive.V, archive.X,
        )
        step_index += 1

        if direction * (V_curr - float(args.V_stop)) >= 0.0:
            print(f"reached requested V_stop={args.V_stop:g}")
            reached = True
            break

    print(f"history: {history_file}")
    print(f"checkpoint: {checkpoint}")
    print(f"full-X states: {states_file} ({archive.nstate} states)")
    return reached


def main():
    args = _args()
    if args.V_second == args.V_start:
        raise ValueError("V-start and V-second must differ")
    if min(args.ds, args.ds_min, args.ds_max) <= 0 or args.ds_min > args.ds_max:
        raise ValueError("invalid arclength step bounds")
    lambdas = []
    for value in args.lambdas:
        lam = float(value)
        weights_from_lambda(lam)  # validation
        if lam not in lambdas:
            lambdas.append(lam)

    ncell = int(args.L1) * int(args.L2)
    target = float(args.filling) * ncell
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    geometry = ExactSmallRubyThermal(args.L1, args.L2, params)
    norb = int(geometry.n_sites)
    h0 = np.asarray(geometry.h0, dtype=complex)[None, None]
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=args.nw, nOmega=args.nomega, T=args.T)
    operators = np.stack([
        np.asarray(geometry.pseudospin_operator(ch, (0.0, 0.0)), dtype=complex)
        for ch in CHANNELS
    ])
    args.out.mkdir(parents=True, exist_ok=True)

    pac_opts = PACOptions(
        tol=args.newton_tol, max_newton=args.newton_max, fd_eps=args.fd_eps,
        gmres_rtol=args.gmres_rtol, gmres_maxiter=args.gmres_maxiter,
        gmres_restart=args.gmres_restart, line_search_min=args.line_search_min,
        verbose=args.verbose,
    )

    print("=== weighted n/B/J Fierz-GW pseudo-arclength scan ===")
    print(
        f"torus={args.L1}x{args.L2} ({norb} sites), filling={args.filling:g}, "
        f"T={args.T:g}, nw={args.nw}, nOmega={args.nomega}"
    )
    print(f"lambdas={lambdas}")

    status = []
    for lam in lambdas:
        ok = _run_lambda(
            args, lam, geometry, h0, grid, target, norb, operators, pac_opts
        )
        status.append((lam, ok))

    print("\n=== lambda scan summary ===")
    for lam, ok in status:
        print(f"lambda={lam:g}: {'reached V_stop' if ok else 'stopped before V_stop'}")


if __name__ == "__main__":
    main()
