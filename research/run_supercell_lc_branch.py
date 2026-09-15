#!/usr/bin/env python3
"""Search an independent loop-current GW branch from a charge-ordered checkpoint.

Workflow
--------
1. load a converged 18-site checkpoint;
2. project Sigma_H and Sigma_GW onto the primitive-translation-invariant q=0
   subspace, removing the period-three CO component from the *initial seed*;
3. add a finite uniform loop-current source H_source=-h K_channel,q0;
4. solve GW while reducing h to zero;
5. report both physical current channels and the charge-order amplitude at every
   source step;
6. after every converged source point, evaluate the split-GW Luttinger-Ward
   grand potential and the fixed-filling Helmholtz free energy F=Omega+mu*N.

The projection is only an initialization step.  During every subsequent GW
solve charge order is fully allowed to regenerate, so a zero-source endpoint can
be pure LC, CO+LC, CO, or the symmetric state.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
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
    seed_projection_diagnostics,
)
from rubycgw.model import RubyParameters
from rubycgw.supercell import (
    build_supercell_h0,
    build_supercell_interaction,
    charge_order_parameter,
)
from rubycgw.supercell_gw_bootstrap import AndersonOptions, solve_matrix_gw_anderson


def _source_schedule(values: list[float]) -> list[float]:
    out = [float(x) for x in values]
    if not out:
        raise ValueError("current-source sequence must not be empty")
    if not np.isclose(out[-1], 0.0):
        out.append(0.0)
    return out


def _branch_checkpoint_name(
    V: float,
    primitive_filling: float,
    grid: MatsubaraGrid,
    channel: str,
    h: float,
) -> str:
    base = checkpoint_filename(V, primitive_filling, grid)
    stem = base[:-4] if base.endswith(".npz") else base
    return f"{stem}_LC-{channel}_h{float(h):.6g}.npz"


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument(
        "--current-channel",
        choices=["same", "opposite"],
        default="same",
        help=(
            "Physical current channel. Project convention: same=eta_minus, "
            "opposite=eta_plus."
        ),
    )
    p.add_argument(
        "--current-source-sequence",
        nargs="+",
        type=float,
        default=[0.10, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0],
        help="Finite current source ladder; zero is appended automatically.",
    )
    p.add_argument(
        "--V",
        type=float,
        default=None,
        help="Target V. Default: use V stored in the input checkpoint.",
    )
    p.add_argument("--gw-max-iter", type=int, default=500)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--ramp-tol", type=float, default=1e-6)
    p.add_argument("--mu-tol", type=float, default=5e-12)
    p.add_argument("--mu-max-iter", type=int, default=60)
    p.add_argument("--verbose-iterations", action="store_true")
    p.add_argument("--lc-threshold", type=float, default=1e-4)
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help=(
            "Output directory. Default: results/supercell18/lc_branches/"
            "<channel>/<timestamp>."
        ),
    )
    return p.parse_args()


def main():
    args = _parse_args()
    if args.gw_tol <= 0.0 or args.ramp_tol <= 0.0:
        raise ValueError("GW tolerances must be positive")
    if args.gw_tol > args.ramp_tol:
        raise ValueError("Require --gw-tol <= --ramp-tol")
    if args.mu_tol <= 0.0:
        raise ValueError("--mu-tol must be positive")
    if args.gw_max_iter < 1 or args.mu_max_iter < 1:
        raise ValueError("iteration limits must be positive")

    checkpoint = Path(args.checkpoint)
    meta = read_checkpoint_metadata(checkpoint)
    primitive_filling = float(meta["primitive_filling"])
    V = float(meta["V"] if args.V is None else args.V)
    params = RubyParameters(
        ti=float(meta["ti"]),
        t1=float(meta["t1"]),
        t2=float(meta["t2"]),
        V=V,
    )
    grid = MatsubaraGrid(
        nk1=int(meta["nk1"]),
        nk2=int(meta["nk2"]),
        nw=int(meta["nw"]),
        nOmega=int(meta["nOmega"]),
        T=float(meta["T"]),
    )
    target_supercell = 3.0 * primitive_filling

    original_seed, loaded_meta, original_density = load_supercell_checkpoint(
        checkpoint,
        params,
        grid,
        primitive_filling,
    )
    seed = remove_charge_order_from_seed(original_seed, grid)
    projection = seed_projection_diagnostics(original_seed, seed)

    if args.outdir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = (
            Path("results")
            / "supercell18"
            / "lc_branches"
            / args.current_channel
            / stamp
        )
    else:
        outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = outdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sources = _source_schedule(args.current_source_sequence)
    settings = {
        "input_checkpoint": str(checkpoint),
        "input_metadata": loaded_meta,
        "target_V": V,
        "current_channel": args.current_channel,
        "current_source_sequence": sources,
        "primitive_filling": primitive_filling,
        "target_supercell_filling": target_supercell,
        "gw_max_iter": int(args.gw_max_iter),
        "gw_tol": float(args.gw_tol),
        "ramp_tol": float(args.ramp_tol),
        "mu_tol": float(args.mu_tol),
        "mu_max_iter": int(args.mu_max_iter),
        "seed_projection": asdict(projection),
        "thermodynamics": (
            "split-GW Luttinger-Ward Omega; fixed-filling comparison uses F=Omega+mu*N"
        ),
    }
    with (outdir / "settings.json").open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    phi_input = charge_order_parameter(original_density)
    print("=" * 92)
    print("18-site Ruby GW loop-current branch search")
    print(
        f"input CO checkpoint: {checkpoint}\n"
        f"V={V:g}, T={grid.T:g}, grid={grid.nk1}x{grid.nk2}, "
        f"nw={grid.nw}, nOmega={grid.nOmega}, primitive filling={primitive_filling:g}"
    )
    print(
        f"input |Phi_CO|={abs(phi_input):.6e}; remove primitive +/-Q components "
        "from Sigma_H and Sigma_GW before the first solve"
    )
    print(
        "removed seed component: "
        f"max|dSigma_H|={projection.sigma_h_removed_max:.3e}, "
        f"max|dSigma_GW|={projection.sigma_gw_removed_max:.3e}"
    )
    print(
        f"current channel={args.current_channel}; physical label="
        f"{'same circulation' if args.current_channel == 'same' else 'opposite circulation'}"
    )
    print("current-source ladder:", " -> ".join(f"{h:g}" for h in sources))
    print(
        "Important: CO is removed only from the initial seed; it is NOT constrained "
        "away during GW self-consistency."
    )
    print(
        "Thermodynamics: each converged point gets Omega_LW and F=Omega+mu*N; "
        "only h=0 is the physical basin free energy for phase comparison."
    )
    print("=" * 92)

    base_h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    previous = seed
    rows: list[dict] = []
    final_gw = None
    final_thermo = None

    for istep, h in enumerate(sources, start=1):
        is_final = np.isclose(h, 0.0)
        tol = float(args.gw_tol if is_final else args.ramp_tol)
        h0 = add_current_source(base_h0, h, args.current_channel)
        opts = GWOptions(
            mu=float(previous.mu),
            target_filling=target_supercell,
            max_iter=int(args.gw_max_iter),
            tol=tol,
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

        print(
            f"\n[current source {istep}/{len(sources)}] "
            f"h={h:g}, tol={tol:.1e}, seed={'de-CO checkpoint' if istep == 1 else 'previous h'}"
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
        m_same = currents["same_q0"]
        m_opp = currents["opposite_q0"]
        selected = currents[f"{args.current_channel}_q0"]

        thermo = None
        if gw.converged:
            thermo = evaluate_gw_free_energy(
                gw,
                h0,
                Vq,
                grid,
                target_particles=target_supercell,
                primitive_cells_per_supercell=3,
                momentum_backend="fft",
            )

        row = {
            "step": int(istep),
            "V": V,
            "current_source": float(h),
            "current_channel": args.current_channel,
            "converged": bool(gw.converged),
            "iterations": int(gw.iterations),
            "final_error": float(gw.final_error),
            "runtime_s": float(runtime),
            "mu": float(gw.mu),
            "primitive_filling": float(np.sum(gw.density) / 3.0),
            "charge_order_re": float(phi.real),
            "charge_order_im": float(phi.imag),
            "charge_order_abs": float(abs(phi)),
            "m_same_q0_re": float(m_same.real),
            "m_same_q0_im": float(m_same.imag),
            "m_same_q0_abs": float(abs(m_same)),
            "m_opposite_q0_re": float(m_opp.real),
            "m_opposite_q0_im": float(m_opp.imag),
            "m_opposite_q0_abs": float(abs(m_opp)),
            "m_selected_q0_abs": float(abs(selected)),
            "m_selected_per_primitive_cell_abs": float(abs(selected) / np.sqrt(3.0)),
            "screening_smin": float(gw.min_screening_singular_value),
            "Omega_supercell": np.nan if thermo is None else thermo.grand_potential,
            "F_supercell": np.nan if thermo is None else thermo.helmholtz_free_energy,
            "F_per_primitive_cell": np.nan if thermo is None else thermo.free_energy_per_primitive_cell,
            "Omega0": np.nan if thermo is None else thermo.omega0,
            "LW_fermionic": np.nan if thermo is None else thermo.fermionic_lw,
            "Phi_H": np.nan if thermo is None else thermo.phi_hartree,
            "Phi_F": np.nan if thermo is None else thermo.phi_fock,
            "Phi_corr": np.nan if thermo is None else thermo.phi_correlation,
            "free_energy_density_mismatch": np.nan if thermo is None else thermo.density_mismatch_max,
        }
        rows.append(row)
        with (outdir / "lc_branch.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"  conv={gw.converged}, it={gw.iterations}, res={gw.final_error:.3e}, "
            f"mu={gw.mu:.9f}, |Phi|={abs(phi):.6e}, "
            f"m_same(q0)={m_same.real:+.6e}{m_same.imag:+.1e}i, "
            f"m_opposite(q0)={m_opp.real:+.6e}{m_opp.imag:+.1e}i, "
            f"smin={gw.min_screening_singular_value:.3e}, time={runtime:.1f}s"
        )
        if thermo is not None:
            print(
                f"  free energy: Omega/sc={thermo.grand_potential:+.12e}, "
                f"F/sc={thermo.helmholtz_free_energy:+.12e}, "
                f"F/primitive={thermo.free_energy_per_primitive_cell:+.12e}; "
                f"Phi_H={thermo.phi_hartree:+.6e}, "
                f"Phi_F={thermo.phi_fock:+.6e}, "
                f"Phi_corr={thermo.phi_correlation:+.6e}"
            )
            if thermo.density_mismatch_max > 1e-7:
                print(
                    "  WARNING: stored density and tail-reconstructed density differ by "
                    f"{thermo.density_mismatch_max:.3e}; free-energy comparison should use "
                    "a more tightly converged state."
                )

        if not gw.converged:
            print("STOP: this current-source point did not converge; keep the last converged seed.")
            break

        ckpt_name = _branch_checkpoint_name(
            V, primitive_filling, grid, args.current_channel, h
        )
        ckpt_path = checkpoint_dir / ckpt_name
        save_supercell_checkpoint(
            ckpt_path,
            gw,
            params,
            grid,
            primitive_filling,
            source=float(h),
        )
        print("  branch checkpoint:", ckpt_path)
        previous = gw
        if is_final:
            final_gw = gw
            final_thermo = thermo

    print("\n=== LC branch search finished ===")
    print("output:", outdir)
    print("scan:", outdir / "lc_branch.csv")
    print("branch checkpoints:", checkpoint_dir)

    if final_gw is None:
        print("No converged zero-current endpoint was obtained.")
        return

    final_phi = charge_order_parameter(final_gw.density)
    final_currents = current_diagnostics(final_gw.G, grid)
    selected = final_currents[f"{args.current_channel}_q0"]
    per_cell = abs(selected) / np.sqrt(3.0)
    print(
        f"zero-source endpoint: |Phi|={abs(final_phi):.6e}, "
        f"|m_{args.current_channel},q0|={abs(selected):.6e}, "
        f"per-primitive-cell={per_cell:.6e}"
    )
    if final_thermo is not None:
        print(
            f"zero-source thermodynamics: Omega/sc={final_thermo.grand_potential:+.12e}, "
            f"F/sc={final_thermo.helmholtz_free_energy:+.12e}, "
            f"F/primitive={final_thermo.free_energy_per_primitive_cell:+.12e}"
        )
    if per_cell > float(args.lc_threshold):
        if abs(final_phi) > 1e-4:
            print("RESULT: finite zero-source current survives together with CO -> candidate CO+LC branch.")
        else:
            print("RESULT: finite zero-source current survives with negligible period-3 CO -> candidate LC branch.")
    else:
        print(
            "RESULT: the selected current collapses when h->0 for this seed/source ladder; "
            "no spontaneous LC branch was found by this run."
        )


if __name__ == "__main__":
    main()
