#!/usr/bin/env python3
"""Search a physically focused set of 18-site charge/loop-current branches.

The default branch set is

    normal    no explicit source
    co        selected period-3 Q=(1/3,1/3) charge source, then h -> 0
    intra-a   q=0 C3-breaking charge source on triangle A, then h -> 0
    intra-b   q=0 C3-breaking charge source on triangle B, then h -> 0
    ab        q=0 A-versus-B triangle charge-transfer source, then h -> 0
    same      physical-same uniform loop-current source, then h_eta -> 0
    opposite  physical-opposite uniform loop-current source, then h_eta -> 0

The q=0 charge seeds are repeated identically in all three primitive sectors of
the 18-site supercell.  ``intra-a`` uses the primitive pattern
``(1,-1/2,-1/2,0,0,0)``, ``intra-b`` uses
``(0,0,0,1,-1/2,-1/2)``, and ``ab`` uses
``(1,1,1,-1,-1,-1)``.  Other C3-related orientations are symmetry-equivalent
seed choices; after the source is removed the self-consistent solution is not
constrained to keep only the seeded component.

By default the script solves the full split-GW equations.  The selected period-3
``co`` branch starts from the input checkpoint, while normal/q=0-charge/LC
branches start from its primitive-translation-symmetric projection.

With ``--hf-only`` the approximation itself is changed: no polarization P, no
screened interaction W, no dynamic Sigma_c, and no GW iteration is performed.
Instead each branch is followed entirely within self-consistent static
Hartree-Fock,

    G_HF^{-1} = iw + mu - h0 - Sigma_H^HF - Sigma_F^HF.

The first source point of each HF branch starts from zero HF self-energy; later
source points use the previous HF solution, so a source-selected broken-symmetry
branch can be followed adiabatically to h=0.  The input checkpoint then supplies
only (V,T,filling, hopping/grid parameters) and an initial chemical-potential
guess.  Because its self-energies are not loaded in HF-only mode, even a legacy
checkpoint can be used purely as a parameter container.

Charge classification is broader than the selected period-three order parameter
Phi.  Every converged point reports:

    Phi                    selected Q form-factor projection
    Delta_Q                generic period-three translation breaking
    Delta_translation_rms  sector-to-sector density RMS
    Delta_A, Delta_B       q=0 intra-triangle charge disproportionation
    Delta_AB               q=0 mean A-minus-B triangle charge imbalance

Thus Phi=0 is not interpreted as absence of charge order.

Only zero-source endpoints are ranked thermodynamically.  GW mode uses the
split-GW Luttinger-Ward free energy; HF-only mode uses the finite-temperature
Hartree-Fock free energy evaluated directly from the static density matrix.
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
    current_vertex_q0,
    remove_charge_order_from_seed,
)
from rubycgw.model import RubyParameters
from rubycgw.supercell import (
    add_charge_source,
    build_supercell_h0,
    build_supercell_interaction,
    charge_order_diagnostics,
)
from rubycgw.supercell_gw_bootstrap import AndersonOptions, solve_matrix_gw_anderson
from rubycgw.supercell_hf import (
    evaluate_supercell_hf_free_energy,
    solve_supercell_hf,
)


CHARGE_BRANCHES = ("co", "intra-a", "intra-b", "ab")
CURRENT_BRANCHES = ("same", "opposite")
DEFAULT_BRANCHES = ("normal",) + CHARGE_BRANCHES + CURRENT_BRANCHES


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
            "Compatible converged zero-source 18-site GW checkpoint. In normal GW "
            "mode it also seeds the branches. With --hf-only its self-energies are "
            "ignored; only model/numerical parameters and the mu guess are used."
        ),
    )
    p.add_argument(
        "--branches",
        nargs="+",
        choices=list(DEFAULT_BRANCHES),
        default=list(DEFAULT_BRANCHES),
        help=(
            "Branches to search. Default: normal co intra-a intra-b ab same opposite."
        ),
    )
    p.add_argument(
        "--charge-source-sequence",
        "--co-source-sequence",
        dest="charge_source_sequence",
        nargs="+",
        type=float,
        default=[0.10, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0],
        help=(
            "Temporary source ladder for co/intra-a/intra-b/ab charge branches. "
            "--co-source-sequence is retained as a backward-compatible alias."
        ),
    )
    p.add_argument(
        "--current-source-sequence",
        nargs="+",
        type=float,
        default=[0.10, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0],
        help="Temporary uniform current source ladder for same/opposite branches.",
    )
    p.add_argument(
        "--hf-only",
        action="store_true",
        help=(
            "Use self-consistent static Hartree-Fock for the entire branch search. "
            "Do not enter GW at any source point."
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
    p.add_argument(
        "--order-threshold",
        type=float,
        default=1e-6,
        help=(
            "Common threshold for generic charge amplitudes Delta_Q/Delta_A/Delta_B/|Delta_AB| "
            "and for per-primitive-cell current amplitudes. Phi is reported but is not used alone "
            "to decide whether charge order exists."
        ),
    )
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
    # params/grid stay in the signature for backward compatibility with tests and
    # older helper code.  All source Hamiltonians are generated from base_h0 so
    # every branch uses exactly the same one-body Ruby model.
    del params, grid
    if branch in CHARGE_BRANCHES:
        return add_charge_source(base_h0, float(h), branch)
    if branch in CURRENT_BRANCHES:
        return add_current_source(base_h0, float(h), branch)
    if branch == "normal":
        if not np.isclose(h, 0.0):
            raise ValueError("normal branch must have zero source")
        return np.array(base_h0, copy=True)
    raise ValueError(f"unknown branch {branch!r}")


def _branch_schedule(branch: str, args) -> list[float]:
    if branch == "normal":
        return [0.0]
    if branch in CHARGE_BRANCHES:
        return _schedule(args.charge_source_sequence)
    if branch in CURRENT_BRANCHES:
        return _schedule(args.current_source_sequence)
    raise ValueError(f"unknown branch {branch!r}")


def _classify(charge: dict[str, object], currents: dict[str, complex], threshold: float) -> str:
    pieces: list[str] = []
    if float(charge["Delta_Q"]) > threshold:
        pieces.append("Q-CO")
    if float(charge["Delta_A"]) > threshold:
        pieces.append("intra-A-CO")
    if float(charge["Delta_B"]) > threshold:
        pieces.append("intra-B-CO")
    if abs(float(charge["Delta_AB"])) > threshold:
        pieces.append("AB-CO")

    if abs(currents["same_q0"]) / np.sqrt(3.0) > threshold:
        pieces.append("same-LC")
    if abs(currents["opposite_q0"]) / np.sqrt(3.0) > threshold:
        pieces.append("opposite-LC")
    return "+".join(pieces) if pieces else "normal"


def _charge_fields(charge: dict[str, object]) -> dict[str, float]:
    phi = complex(charge["Phi"])
    return {
        "Phi_abs": float(abs(phi)),
        "Delta_Q": float(charge["Delta_Q"]),
        "Delta_translation_rms": float(charge["Delta_translation_rms"]),
        "Delta_A": float(charge["Delta_A"]),
        "Delta_B": float(charge["Delta_B"]),
        "Delta_AB": float(charge["Delta_AB"]),
        "Delta_AB_abs": float(abs(float(charge["Delta_AB"]))),
    }


def _hf_current_diagnostics(rho: np.ndarray, grid: MatsubaraGrid) -> dict[str, complex]:
    """Exact static-HF q=0 currents from rho, with no Matsubara truncation."""
    out: dict[str, complex] = {}
    for channel in CURRENT_BRANCHES:
        K = current_vertex_q0(channel)
        value = (1.0 / grid.nk) * np.einsum(
            "ab,xyba->", K, np.asarray(rho, dtype=complex), optimize=True
        )
        out[f"{channel}_q0"] = complex(value)
    return out


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


def _failed_endpoint_row(branch: str, method: str) -> dict:
    return {
        "method": method,
        "seed_branch": branch,
        "endpoint_found": False,
        "classification": "no-converged-zero-source-endpoint",
        "mu": np.nan,
        "Phi_abs": np.nan,
        "Delta_Q": np.nan,
        "Delta_translation_rms": np.nan,
        "Delta_A": np.nan,
        "Delta_B": np.nan,
        "Delta_AB": np.nan,
        "Delta_AB_abs": np.nan,
        "m_same_pc_abs": np.nan,
        "m_opposite_pc_abs": np.nan,
        "Omega_supercell": np.nan,
        "F_supercell": np.nan,
        "F_per_primitive_cell": np.nan,
        "DeltaF_per_primitive_cell": np.nan,
        "final_error": np.nan,
    }


def _print_point(prefix: str, charge: dict[str, object], m_same_pc: complex, m_opp_pc: complex, state: str) -> None:
    print(
        f"{prefix}|Phi|={abs(complex(charge['Phi'])):.3e}, "
        f"Delta_Q={float(charge['Delta_Q']):.3e}, "
        f"Delta_A={float(charge['Delta_A']):.3e}, "
        f"Delta_B={float(charge['Delta_B']):.3e}, "
        f"Delta_AB={float(charge['Delta_AB']):+.3e}, "
        f"|m_same|/pc={abs(m_same_pc):.3e}, "
        f"|m_opp|/pc={abs(m_opp_pc):.3e}, state={state}"
    )


def main():
    args = _parse_args()
    if args.order_threshold <= 0.0:
        raise ValueError("--order-threshold must be positive")
    if args.hf_max_iter < 1:
        raise ValueError("--hf-max-iter must be positive")
    if args.hf_tol <= 0.0 or args.hf_mu_tol <= 0.0:
        raise ValueError("HF tolerances must be positive")
    if not (0.0 < args.hf_mixing <= 1.0):
        raise ValueError("--hf-mixing must lie in (0,1]")
    if not args.hf_only:
        if args.gw_tol <= 0.0 or args.ramp_tol <= 0.0:
            raise ValueError("GW tolerances must be positive")
        if args.gw_tol > args.ramp_tol:
            raise ValueError("Require --gw-tol <= --ramp-tol")

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

    # HF-only mode deliberately does not load checkpoint self-energies.  Full GW
    # mode requires strict checkpoint compatibility and rejects legacy interaction data.
    if args.hf_only:
        seed_original = None
        seed_deco = None
        density_original = None
    else:
        seed_original, _, density_original = load_supercell_checkpoint(
            checkpoint, params, grid, primitive_filling
        )
        seed_deco = remove_charge_order_from_seed(seed_original, grid)

    if args.outdir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        mode = "hf" if args.hf_only else "gw"
        outdir = Path("results") / "supercell18" / "branch_search" / f"{mode}_{stamp}"
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir / "checkpoints"
    if not args.hf_only:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    base_h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    Vq = build_supercell_interaction(grid.qmesh(), params)

    print("=" * 112)
    print("18-site charge/LC branch search")
    print(
        f"V={params.V:g}, T={grid.T:g}, primitive filling={primitive_filling:g}, "
        f"grid={grid.nk1}x{grid.nk2}, nw={grid.nw}, nOmega={grid.nOmega}"
    )
    print("branches:", ", ".join(args.branches))
    if args.hf_only:
        print("approximation: STATIC SELF-CONSISTENT HARTREE-FOCK ONLY")
        print("GW is disabled: P, W, Sigma_c and GW iterations are not evaluated.")
        print(
            "reference checkpoint is used only for V,T,filling,hopping/grid parameters "
            "and the initial mu guess; its self-energies are not loaded"
        )
    else:
        input_charge = charge_order_diagnostics(density_original)
        print("approximation: FULL SPLIT-GW")
        print(
            f"input |Phi|={abs(complex(input_charge['Phi'])):.3e}, "
            f"Delta_Q={float(input_charge['Delta_Q']):.3e}, "
            f"Delta_A={float(input_charge['Delta_A']):.3e}, "
            f"Delta_B={float(input_charge['Delta_B']):.3e}, "
            f"Delta_AB={float(input_charge['Delta_AB']):+.3e}; "
            "normal/q0-charge/LC seeds use primitive-translation projection, "
            "selected Q-CO seed keeps the original state"
        )
    print("Only zero-source endpoints are ranked by the free energy of the selected approximation.")
    print("=" * 112)

    endpoint_rows: list[dict] = []
    scan_rows: list[dict] = []

    for branch in args.branches:
        schedule = _branch_schedule(branch, args)
        print(f"\n### branch: {branch} ###")
        print("source ladder:", " -> ".join(f"{x:g}" for x in schedule))

        if args.hf_only:
            previous_hf = None
            endpoint = None
            endpoint_thermo = None

            for istep, h in enumerate(schedule, start=1):
                is_zero = bool(np.isclose(h, 0.0))
                h0 = _branch_h0(branch, h, base_h0, params, grid)
                seed_label = "zero-HF" if previous_hf is None else "previous-HF"
                print(
                    f"[{branch} {istep}/{len(schedule)}] h={h:g}, "
                    f"tol={args.hf_tol:.1e}, seed={seed_label}"
                )
                t0 = time.perf_counter()
                hf = solve_supercell_hf(
                    h0,
                    Vq,
                    grid,
                    target_N,
                    mu0=float(meta["mu"]),
                    initial=previous_hf,
                    max_iter=int(args.hf_max_iter),
                    tol=float(args.hf_tol),
                    mixing=float(args.hf_mixing),
                    mu_tol=float(args.hf_mu_tol),
                    mu_max_iter=int(args.hf_mu_max_iter),
                    momentum_backend="fft",
                    verbose=bool(args.hf_verbose),
                )
                runtime = time.perf_counter() - t0
                charge = charge_order_diagnostics(hf.density)
                currents = _hf_current_diagnostics(hf.rho, grid)
                m_same_pc = currents["same_q0"] / np.sqrt(3.0)
                m_opp_pc = currents["opposite_q0"] / np.sqrt(3.0)
                state_label = _classify(charge, currents, args.order_threshold)

                row = {
                    "method": "HF",
                    "seed_branch": branch,
                    "step": istep,
                    "source": float(h),
                    "seed": seed_label,
                    "converged": bool(hf.converged),
                    "iterations": int(hf.iterations),
                    "final_error": float(hf.final_error),
                    "runtime_s": float(runtime),
                    "mu": float(hf.mu),
                    **_charge_fields(charge),
                    "m_same_pc_abs": float(abs(m_same_pc)),
                    "m_opposite_pc_abs": float(abs(m_opp_pc)),
                    "classification": state_label,
                }
                scan_rows.append(row)
                print(
                    f"    HF conv={hf.converged}, it={hf.iterations}, res={hf.final_error:.3e}, "
                    f"mu={hf.mu:.9f}, time={runtime:.1f}s"
                )
                _print_point("    ", charge, m_same_pc, m_opp_pc, state_label)
                if not hf.converged:
                    print(f"    STOP branch {branch}: HF source point did not converge")
                    break

                previous_hf = hf
                if is_zero:
                    endpoint = hf
                    endpoint_thermo = evaluate_supercell_hf_free_energy(
                        hf,
                        h0,
                        grid,
                        primitive_cells_per_supercell=3,
                    )

            if endpoint is None or endpoint_thermo is None:
                endpoint_rows.append(_failed_endpoint_row(branch, "HF"))
                continue

            charge = charge_order_diagnostics(endpoint.density)
            currents = _hf_current_diagnostics(endpoint.rho, grid)
            m_same_pc = currents["same_q0"] / np.sqrt(3.0)
            m_opp_pc = currents["opposite_q0"] / np.sqrt(3.0)
            label = _classify(charge, currents, args.order_threshold)
            endpoint_rows.append(
                {
                    "method": "HF",
                    "seed_branch": branch,
                    "endpoint_found": True,
                    "classification": label,
                    "mu": float(endpoint.mu),
                    **_charge_fields(charge),
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
                f"    ZERO-SOURCE HF: state={label}, "
                f"F/pc={endpoint_thermo.free_energy_per_primitive_cell:+.12e}, "
                f"Omega/sc={endpoint_thermo.grand_potential:+.12e}"
            )
            _print_point("      ", charge, m_same_pc, m_opp_pc, label)
            continue

        # Full-GW mode: keep the selected Q-CO checkpoint for the co branch.
        # The other branches start from its primitive-translation projection.
        previous = seed_original if branch == "co" else seed_deco
        endpoint = None
        endpoint_thermo = None

        for istep, h in enumerate(schedule, start=1):
            is_zero = bool(np.isclose(h, 0.0))
            tol = float(args.gw_tol if is_zero else args.ramp_tol)
            h0 = _branch_h0(branch, h, base_h0, params, grid)
            seed_label = (
                "original-checkpoint"
                if istep == 1 and branch == "co"
                else "primitive-periodic-checkpoint"
                if istep == 1
                else "previous-GW"
            )
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

            charge = charge_order_diagnostics(gw.density)
            currents = current_diagnostics(gw.G, grid)
            m_same_pc = currents["same_q0"] / np.sqrt(3.0)
            m_opp_pc = currents["opposite_q0"] / np.sqrt(3.0)
            state_label = _classify(charge, currents, args.order_threshold)

            row = {
                "method": "GW",
                "seed_branch": branch,
                "step": istep,
                "source": float(h),
                "seed": seed_label,
                "converged": bool(gw.converged),
                "iterations": int(gw.iterations),
                "final_error": float(gw.final_error),
                "runtime_s": float(runtime),
                "mu": float(gw.mu),
                **_charge_fields(charge),
                "m_same_pc_abs": float(abs(m_same_pc)),
                "m_opposite_pc_abs": float(abs(m_opp_pc)),
                "classification": state_label,
            }
            scan_rows.append(row)
            print(
                f"    GW conv={gw.converged}, it={gw.iterations}, res={gw.final_error:.3e}, "
                f"mu={gw.mu:.9f}, time={runtime:.1f}s"
            )
            _print_point("    ", charge, m_same_pc, m_opp_pc, state_label)
            if not gw.converged:
                print(f"    STOP branch {branch}: GW source point did not converge")
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
            endpoint_rows.append(_failed_endpoint_row(branch, "GW"))
            continue

        charge = charge_order_diagnostics(endpoint.density)
        currents = current_diagnostics(endpoint.G, grid)
        m_same_pc = currents["same_q0"] / np.sqrt(3.0)
        m_opp_pc = currents["opposite_q0"] / np.sqrt(3.0)
        label = _classify(charge, currents, args.order_threshold)
        endpoint_rows.append(
            {
                "method": "GW",
                "seed_branch": branch,
                "endpoint_found": True,
                "classification": label,
                "mu": float(endpoint.mu),
                **_charge_fields(charge),
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
            f"    ZERO-SOURCE GW: state={label}, "
            f"F/pc={endpoint_thermo.free_energy_per_primitive_cell:+.12e}, "
            f"Omega/sc={endpoint_thermo.grand_potential:+.12e}"
        )
        _print_point("      ", charge, m_same_pc, m_opp_pc, label)

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

    print("\n" + "=" * 112)
    print("ZERO-SOURCE BRANCH ATLAS")
    print("=" * 112)
    if not valid_sorted:
        print("No converged zero-source endpoint was obtained.")
    else:
        for rank, row in enumerate(valid_sorted, start=1):
            print(
                f"#{rank} {row['method']:<2s} seed={row['seed_branch']:<8s} "
                f"-> {row['classification']:<32s} "
                f"F/pc={row['F_per_primitive_cell']:+.12e}  "
                f"DeltaF/pc={row['DeltaF_per_primitive_cell']:+.6e}"
            )
            print(
                f"    |Phi|={row['Phi_abs']:.3e}  Delta_Q={row['Delta_Q']:.3e}  "
                f"Delta_A={row['Delta_A']:.3e}  Delta_B={row['Delta_B']:.3e}  "
                f"Delta_AB={row['Delta_AB']:+.3e}  "
                f"|m_same|/pc={row['m_same_pc_abs']:.3e}  "
                f"|m_opp|/pc={row['m_opposite_pc_abs']:.3e}"
            )
        winner = valid_sorted[0]
        print(
            f"\nlowest found basin ({winner['method']}): "
            f"seed={winner['seed_branch']} -> {winner['classification']}"
        )
        print(
            "Important: this is the lowest among the searched charge/LC-focused branches, "
            "not a proof that no other symmetry-breaking basin exists."
        )

    print("\noutput:", outdir)
    print("source scan:", outdir / "branch_scan.csv")
    print("zero-source atlas:", outdir / "branch_atlas.csv")
    if not args.hf_only:
        print("GW branch checkpoints:", ckpt_dir)


if __name__ == "__main__":
    main()