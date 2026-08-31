#!/usr/bin/env python3
"""Search a small, physically focused set of 18-site GW branches.

The default branch set is deliberately small:

    normal    primitive-translation-symmetric seed, no source
    co        period-3 charge source, then h_CO -> 0
    same      physical-same uniform loop-current source, then h_eta -> 0
    opposite  physical-opposite uniform loop-current source, then h_eta -> 0

All branches are solved at the same (V,T,N,k-grid,Matsubara-grid).  By default,
the LC branches start from the input checkpoint after projecting out primitive
+/-Q self-energy components, while the CO branch starts from the original
checkpoint.  With ``--hf-seed`` this history dependence is removed at the first
source point of every branch: the code first solves a fully self-consistent
18-site static Hartree-Fock Green function for that branch Hamiltonian and uses
that HF solution as the initial full-GW self-energy.

Charge order is never constrained away during the full-GW LC solves, so an LC
seed is allowed to end in a mixed CO+LC state.

For every converged zero-source endpoint the script evaluates the split-GW
Luttinger-Ward functional and ranks states by the fixed-filling Helmholtz free
energy F=Omega+mu*N.  Thus the output distinguishes "which seed converged" from
"which zero-source basin has the lowest free energy".
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import numpy as np

from rubycgw.checkpoint import (
    checkpoint_filename,
    load_supercell_checkpoint,
    read_checkpoint_metadata,
    save_supercell_checkpoint,
)
from rubycgw.free_energy import evaluate_gw_free_energy
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.lc_branch import (
    add_current_source,
    current_diagnostics,
    remove_charge_order_from_seed,
)
from rubycgw.model import RubyParameters
from rubycgw.supercell import (
    build_supercell_h0,
    build_supercell_interaction,
    charge_order_parameter,
)
from rubycgw.supercell_gw_bootstrap import AndersonOptions, solve_matrix_gw_anderson
from rubycgw.supercell_hf import solve_supercell_hf_seed


DEFAULT_BRANCHES = ("normal", "co", "same", "opposite")


def _schedule(values: list[float]) -> list[float]:
    out = [float(x) for x in values]
    if not out:
        out = [0.0]
    if not np.isclose(out[-1], 0.0):
        out.append(0.0)
    return out


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help=(
            "Compatible converged zero-source 18-site checkpoint.  Normally it is "
            "also the common GW seed; with --hf-seed its self-energies are ignored "
            "and it supplies only model/numerical parameters and an initial mu guess."
        ),
    )
    p.add_argument(
        "--branches",
        nargs="+",
        choices=list(DEFAULT_BRANCHES),
        default=list(DEFAULT_BRANCHES),
        help="Branches to search. Default: normal co same opposite.",
    )
    p.add_argument(
        "--co-source-sequence",
        nargs="+",
        type=float,
        default=[0.10, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0],
        help="Temporary period-3 charge source ladder.",
    )
    p.add_argument(
        "--current-source-sequence",
        nargs="+",
        type=float,
        default=[0.10, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0],
        help="Temporary uniform current source ladder for same/opposite branches.",
    )
    p.add_argument(
        "--hf-seed",
        action="store_true",
        help=(
            "At the first source point of each branch, ignore checkpoint self-energies, "
            "solve self-consistent static Hartree-Fock for that branch Hamiltonian, "
            "and use the converged HF Green function/self-energy as the full-GW seed."
        ),
    )
    p.add_argument("--hf-max-iter", type=int, default=500)
    p.add_argument("--hf-tol", type=float, default=1e-9)
    p.add_argument("--hf-mixing", type=float, default=0.25)
    p.add_argument("--hf-mu-tol", type=float, default=1e-12)
    p.add_argument("--hf-mu-max-iter", type=int, default=80)
    p.add_argument("--hf-verbose", action="store_true")
    p.add_argument("--gw-max-iter", type=int, default=500)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--ramp-tol", type=float, default=1e-6)
    p.add_argument("--mu-tol", type=float, default=5e-12)
    p.add_argument("--mu-max-iter", type=int, default=60)
    p.add_argument("--verbose-iterations", action="store_true")
    p.add_argument("--order-threshold", type=float, default=1e-6)
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def _options(args, mu: float, target_N: float, tol: float) -> GWOptions:
    return GWOptions(
        mu=float(mu),
        target_filling=float(target_N),
        max_iter=int(args.gw_max_iter),
        tol=float(tol),
        mixing=0.20,
        mixing_method="linear",
        pulay_history=6,
        pulay_start=3,
        pulay_regularization=1e-10,
        mu_tol=float(args.mu_tol),
        mu_max_iter=int(args.mu_max_iter),
        verbose=bool(args.verbose_iterations),
        momentum_backend="fft",
    )


def _branch_h0(
    branch: str,
    h: float,
    base_h0: np.ndarray,
    params: RubyParameters,
    grid: MatsubaraGrid,
) -> np.ndarray:
    if branch == "co":
        return build_supercell_h0(grid.kmesh(), params, source_strength=float(h))
    if branch in {"same", "opposite"}:
        return add_current_source(base_h0, float(h), branch)
    if branch == "normal":
        if not np.isclose(h, 0.0):
            raise ValueError("normal branch must have zero source")
        return base_h0
    raise ValueError(f"unknown branch {branch!r}")


def _branch_schedule(branch: str, args) -> list[float]:
    if branch == "normal":
        return [0.0]
    if branch == "co":
        return _schedule(args.co_source_sequence)
    return _schedule(args.current_source_sequence)


def _classify(phi: complex, currents: dict[str, complex], threshold: float) -> str:
    co = abs(phi) > threshold
    same = abs(currents["same_q0"]) / np.sqrt(3.0) > threshold
    opposite = abs(currents["opposite_q0"]) / np.sqrt(3.0) > threshold
    pieces = []
    if co:
        pieces.append("CO")
    if same:
        pieces.append("same-LC")
    if opposite:
        pieces.append("opposite-LC")
    return "+".join(pieces) if pieces else "normal"


def _checkpoint_name(
    V: float,
    primitive_filling: float,
    grid: MatsubaraGrid,
    branch: str,
    h: float,
) -> str:
    base = checkpoint_filename(V, primitive_filling, grid)
    stem = base[:-4] if base.endswith(".npz") else base
    return f"{stem}_branch-{branch}_h{float(h):.6g}.npz"


def main():
    args = _parse_args()
    if args.gw_tol <= 0.0 or args.ramp_tol <= 0.0:
        raise ValueError("GW tolerances must be positive")
    if args.gw_tol > args.ramp_tol:
        raise ValueError("Require --gw-tol <= --ramp-tol")
    if args.order_threshold <= 0.0:
        raise ValueError("--order-threshold must be positive")
    if args.hf_max_iter < 1:
        raise ValueError("--hf-max-iter must be positive")
    if args.hf_tol <= 0.0 or args.hf_mu_tol <= 0.0:
        raise ValueError("HF tolerances must be positive")
    if not (0.0 < args.hf_mixing <= 1.0):
        raise ValueError("--hf-mixing must lie in (0,1]")

    checkpoint = Path(args.checkpoint)
    meta = read_checkpoint_metadata(checkpoint)
    if not bool(meta.get("converged", False)):
        raise ValueError("Input checkpoint is not marked converged")
    if not np.isclose(float(meta.get("source", np.nan)), 0.0, atol=1e-14):
        raise ValueError("Use a zero-source checkpoint as the branch-search reference")

    primitive_filling = float(meta["primitive_filling"])
    params = RubyParameters(
        ti=float(meta["ti"]),
        t1=float(meta["t1"]),
        t2=float(meta["t2"]),
        V=float(meta["V"]),
    )
    grid = MatsubaraGrid(
        nk1=int(meta["nk1"]),
        nk2=int(meta["nk2"]),
        nw=int(meta["nw"]),
        nOmega=int(meta["nOmega"]),
        T=float(meta["T"]),
    )
    target_N = 3.0 * primitive_filling

    seed_original, _, density_original = load_supercell_checkpoint(
        checkpoint, params, grid, primitive_filling
    )
    seed_deco = remove_charge_order_from_seed(seed_original, grid)

    if args.outdir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = Path("results") / "supercell18" / "branch_search" / stamp
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    base_h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    Vq = build_supercell_interaction(grid.qmesh(), params)

    print("=" * 104)
    print("18-site focused GW branch search")
    print(
        f"V={params.V:g}, T={grid.T:g}, primitive filling={primitive_filling:g}, "
        f"grid={grid.nk1}x{grid.nk2}, nw={grid.nw}, nOmega={grid.nOmega}"
    )
    print("branches:", ", ".join(args.branches))
    if args.hf_seed:
        print(
            "seed mode: SELF-CONSISTENT HF at the first source point of each branch; "
            "checkpoint self-energies are ignored for branch initialization"
        )
        print(
            f"reference checkpoint |Phi|={abs(charge_order_parameter(density_original)):.6e}; "
            "it is used only for parameters and the initial chemical-potential guess"
        )
    else:
        print(
            f"input |Phi|={abs(charge_order_parameter(density_original)):.6e}; "
            "LC/normal seeds use primitive-translation projection, CO seed keeps the original state"
        )
    print("Only zero-source endpoints are ranked thermodynamically by F=Omega+mu*N.")
    print("=" * 104)

    endpoint_rows: list[dict] = []
    scan_rows: list[dict] = []

    for branch in args.branches:
        schedule = _branch_schedule(branch, args)
        # In checkpoint-seed mode, CO keeps the original state while normal/LC
        # begin from its primitive-translation projection.  In HF-seed mode this
        # value is replaced before the first full-GW solve.
        previous = seed_original if branch == "co" else seed_deco
        endpoint = None
        endpoint_thermo = None

        print(f"\n### branch: {branch} ###")
        print("source ladder:", " -> ".join(f"{x:g}" for x in schedule))

        for istep, h in enumerate(schedule, start=1):
            is_zero = np.isclose(h, 0.0)
            tol = float(args.gw_tol if is_zero else args.ramp_tol)
            h0 = _branch_h0(branch, h, base_h0, params, grid)

            if istep == 1 and args.hf_seed:
                print(
                    f"[{branch} HF seed] solving static self-consistent HF at h={h:g} "
                    f"(tol={args.hf_tol:.1e}, mixing={args.hf_mixing:g})"
                )
                t_hf = time.perf_counter()
                hf = solve_supercell_hf_seed(
                    h0,
                    Vq,
                    grid,
                    target_N,
                    mu0=float(meta["mu"]),
                    max_iter=int(args.hf_max_iter),
                    tol=float(args.hf_tol),
                    mixing=float(args.hf_mixing),
                    mu_tol=float(args.hf_mu_tol),
                    mu_max_iter=int(args.hf_mu_max_iter),
                    momentum_backend="fft",
                    verbose=bool(args.hf_verbose),
                )
                hf_runtime = time.perf_counter() - t_hf
                hf_phi = charge_order_parameter(hf.density)
                hf_currents = current_diagnostics(hf.G, grid)
                hf_same_pc = hf_currents["same_q0"] / np.sqrt(3.0)
                hf_opp_pc = hf_currents["opposite_q0"] / np.sqrt(3.0)
                print(
                    f"    HF conv={hf.converged}, it={hf.iterations}, res={hf.final_error:.3e}, "
                    f"mu={hf.mu:.9f}, |Phi|={abs(hf_phi):.6e}, "
                    f"|m_same|/pc={abs(hf_same_pc):.6e}, "
                    f"|m_opp|/pc={abs(hf_opp_pc):.6e}, time={hf_runtime:.1f}s"
                )
                if not hf.converged:
                    print(
                        f"    STOP branch {branch}: self-consistent HF seed did not converge"
                    )
                    break
                previous = hf.seed
                seed_label = "self-consistent-HF"
            elif istep == 1:
                seed_label = "original-checkpoint" if branch == "co" else "de-CO-checkpoint"
            else:
                seed_label = "previous-GW"

            opts = _options(args, previous.mu, target_N, tol)
            print(
                f"[{branch} {istep}/{len(schedule)}] h={h:g}, tol={tol:.1e}, "
                f"seed={seed_label}"
            )
            t0 = time.perf_counter()
            gw = solve_matrix_gw_anderson(
                h0,
                Vq,
                grid,
                opts=opts,
                initial=previous,
                anderson=AndersonOptions(),
            )
            runtime = time.perf_counter() - t0

            phi = charge_order_parameter(gw.density)
            currents = current_diagnostics(gw.G, grid)
            m_same_pc = currents["same_q0"] / np.sqrt(3.0)
            m_opp_pc = currents["opposite_q0"] / np.sqrt(3.0)
            state_label = _classify(phi, currents, args.order_threshold)

            scan_rows.append(
                {
                    "seed_branch": branch,
                    "step": istep,
                    "source": float(h),
                    "gw_seed": seed_label,
                    "converged": bool(gw.converged),
                    "iterations": int(gw.iterations),
                    "final_error": float(gw.final_error),
                    "runtime_s": float(runtime),
                    "mu": float(gw.mu),
                    "Phi_abs": float(abs(phi)),
                    "m_same_pc_abs": float(abs(m_same_pc)),
                    "m_opposite_pc_abs": float(abs(m_opp_pc)),
                    "classification": state_label,
                }
            )
            print(
                f"    conv={gw.converged}, it={gw.iterations}, res={gw.final_error:.3e}, "
                f"mu={gw.mu:.9f}, |Phi|={abs(phi):.6e}, "
                f"|m_same|/pc={abs(m_same_pc):.6e}, "
                f"|m_opp|/pc={abs(m_opp_pc):.6e}, state={state_label}, "
                f"time={runtime:.1f}s"
            )

            if not gw.converged:
                print(f"    STOP branch {branch}: source point did not converge")
                break

            ckpt_path = ckpt_dir / _checkpoint_name(
                params.V, primitive_filling, grid, branch, h
            )
            save_supercell_checkpoint(
                ckpt_path,
                gw,
                params,
                grid,
                primitive_filling,
                source=float(h),
            )
            previous = gw

            if is_zero:
                endpoint = gw
                endpoint_thermo = evaluate_gw_free_energy(
                    gw,
                    h0,
                    Vq,
                    grid,
                    target_particles=target_N,
                    primitive_cells_per_supercell=3,
                    momentum_backend="fft",
                )

        if endpoint is None or endpoint_thermo is None:
            endpoint_rows.append(
                {
                    "seed_branch": branch,
                    "endpoint_found": False,
                    "classification": "no-converged-zero-source-endpoint",
                    "mu": np.nan,
                    "Phi_abs": np.nan,
                    "m_same_pc_abs": np.nan,
                    "m_opposite_pc_abs": np.nan,
                    "Omega_supercell": np.nan,
                    "F_supercell": np.nan,
                    "F_per_primitive_cell": np.nan,
                    "DeltaF_per_primitive_cell": np.nan,
                    "final_error": np.nan,
                }
            )
            continue

        phi = charge_order_parameter(endpoint.density)
        currents = current_diagnostics(endpoint.G, grid)
        m_same_pc = currents["same_q0"] / np.sqrt(3.0)
        m_opp_pc = currents["opposite_q0"] / np.sqrt(3.0)
        label = _classify(phi, currents, args.order_threshold)
        endpoint_rows.append(
            {
                "seed_branch": branch,
                "endpoint_found": True,
                "classification": label,
                "mu": float(endpoint.mu),
                "Phi_abs": float(abs(phi)),
                "m_same_pc_abs": float(abs(m_same_pc)),
                "m_opposite_pc_abs": float(abs(m_opp_pc)),
                "Omega_supercell": float(endpoint_thermo.grand_potential),
                "F_supercell": float(endpoint_thermo.helmholtz_free_energy),
                "F_per_primitive_cell": float(endpoint_thermo.free_energy_per_primitive_cell),
                "DeltaF_per_primitive_cell": np.nan,
                "final_error": float(endpoint.final_error),
            }
        )
        print(
            f"    ZERO-SOURCE: state={label}, F/pc={endpoint_thermo.free_energy_per_primitive_cell:+.12e}, "
            f"Omega/sc={endpoint_thermo.grand_potential:+.12e}"
        )

    # Write source scans irrespective of whether every branch reached h=0.
    if scan_rows:
        with (outdir / "branch_scan.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
            writer.writeheader()
            writer.writerows(scan_rows)

    valid = [
        r
        for r in endpoint_rows
        if bool(r["endpoint_found"]) and np.isfinite(r["F_per_primitive_cell"])
    ]
    if valid:
        fmin = min(float(r["F_per_primitive_cell"]) for r in valid)
        for row in endpoint_rows:
            if bool(row["endpoint_found"]) and np.isfinite(row["F_per_primitive_cell"]):
                row["DeltaF_per_primitive_cell"] = float(
                    row["F_per_primitive_cell"] - fmin
                )
        valid_sorted = sorted(valid, key=lambda r: float(r["F_per_primitive_cell"]))
    else:
        valid_sorted = []

    with (outdir / "branch_atlas.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(endpoint_rows[0].keys()))
        writer.writeheader()
        writer.writerows(endpoint_rows)

    print("\n" + "=" * 104)
    print("ZERO-SOURCE BRANCH ATLAS")
    print("=" * 104)
    if not valid_sorted:
        print("No converged zero-source endpoint was obtained.")
    else:
        for rank, row in enumerate(valid_sorted, start=1):
            print(
                f"#{rank} seed={row['seed_branch']:<8s} -> {row['classification']:<18s} "
                f"F/pc={row['F_per_primitive_cell']:+.12e}  "
                f"DeltaF/pc={row['DeltaF_per_primitive_cell']:+.6e}  "
                f"|Phi|={row['Phi_abs']:.3e}  "
                f"|m_same|/pc={row['m_same_pc_abs']:.3e}  "
                f"|m_opp|/pc={row['m_opposite_pc_abs']:.3e}"
            )
        winner = valid_sorted[0]
        print(
            f"\nlowest found basin: seed={winner['seed_branch']} -> {winner['classification']}"
        )
        print(
            "Important: this is the lowest among the searched CO/LC-focused branches, "
            "not a proof that no other symmetry-breaking basin exists."
        )

    print("\noutput:", outdir)
    print("source scan:", outdir / "branch_scan.csv")
    print("zero-source atlas:", outdir / "branch_atlas.csv")
    print("branch checkpoints:", ckpt_dir)


if __name__ == "__main__":
    main()
