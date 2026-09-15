#!/usr/bin/env python3
"""Compare zero-source 18-site GW basins by fixed-filling Helmholtz free energy.

Each input must be a converged zero-source checkpoint at the same physical and
numerical point (V, T, filling, hopping parameters and grids).  The script
reconstructs G, P and W from the stored self-energies, evaluates the split-GW
Luttinger-Ward grand potential, performs the fixed-N Legendre transform

    F = Omega + mu N,

and reports Delta F relative to the lowest supplied basin.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rubycgw.checkpoint import (
    load_supercell_checkpoint,
    read_checkpoint_metadata,
)
from rubycgw.free_energy import evaluate_gw_free_energy
from rubycgw.grids import MatsubaraGrid
from rubycgw.lc_branch import current_diagnostics
from rubycgw.model import RubyParameters
from rubycgw.supercell import (
    build_supercell_h0,
    build_supercell_interaction,
    charge_order_parameter,
)
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    dyson_from_sigma_matrix,
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", nargs="+", help="Two or more zero-source .npz checkpoints.")
    p.add_argument("--csv", type=str, default=None, help="Optional output CSV path.")
    return p.parse_args()


def _same_point(reference: dict, current: dict) -> str | None:
    exact_keys = ["nk1", "nk2", "nw", "nOmega", "matrix_dimension"]
    float_keys = ["V", "T", "primitive_filling", "ti", "t1", "t2"]
    for key in exact_keys:
        if int(reference.get(key, -999)) != int(current.get(key, -998)):
            return f"{key} differs"
    for key in float_keys:
        if not np.isclose(
            float(reference.get(key, np.nan)),
            float(current.get(key, np.nan)),
            rtol=0.0,
            atol=1e-12,
        ):
            return f"{key} differs"
    return None


def main():
    args = _parse_args()
    paths = [Path(x) for x in args.checkpoints]
    if len(paths) < 2:
        raise ValueError("Provide at least two checkpoints to compare.")

    metas = [read_checkpoint_metadata(path) for path in paths]
    ref = metas[0]
    for path, meta in zip(paths, metas):
        if not bool(meta.get("converged", False)):
            raise ValueError(f"Checkpoint is not marked converged: {path}")
        if not np.isclose(float(meta.get("source", np.nan)), 0.0, atol=1e-14):
            raise ValueError(
                f"Checkpoint is not zero-source and cannot be compared as a physical basin: {path}"
            )
        mismatch = _same_point(ref, meta)
        if mismatch is not None:
            raise ValueError(f"Checkpoints are not at the same thermodynamic point: {mismatch}: {path}")

    params = RubyParameters(
        ti=float(ref["ti"]),
        t1=float(ref["t1"]),
        t2=float(ref["t2"]),
        V=float(ref["V"]),
    )
    grid = MatsubaraGrid(
        nk1=int(ref["nk1"]),
        nk2=int(ref["nk2"]),
        nw=int(ref["nw"]),
        nOmega=int(ref["nOmega"]),
        T=float(ref["T"]),
    )
    primitive_filling = float(ref["primitive_filling"])
    target_N = 3.0 * primitive_filling
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    Vq = build_supercell_interaction(grid.qmesh(), params)

    rows = []
    for path, meta in zip(paths, metas):
        seed, _, stored_density = load_supercell_checkpoint(
            path,
            params,
            grid,
            primitive_filling,
        )
        G = dyson_from_sigma_matrix(
            h0,
            grid,
            float(seed.mu),
            np.asarray(seed.Sigma_H, dtype=complex),
            np.asarray(seed.Sigma_GW, dtype=complex),
        )
        P = compute_polarization_matrix(G, grid, backend="fft")
        W = compute_screened_interaction_matrix(P, Vq)
        state = SimpleNamespace(
            G=G,
            P=P,
            W=W,
            Sigma_H=np.asarray(seed.Sigma_H, dtype=complex),
            Sigma_GW=np.asarray(seed.Sigma_GW, dtype=complex),
            mu=float(seed.mu),
            density=np.asarray(stored_density, dtype=float),
        )
        thermo = evaluate_gw_free_energy(
            state,
            h0,
            Vq,
            grid,
            target_particles=target_N,
            primitive_cells_per_supercell=3,
            momentum_backend="fft",
        )
        currents = current_diagnostics(G, grid)
        phi = charge_order_parameter(stored_density)
        rows.append(
            {
                "checkpoint": str(path),
                "mu": float(seed.mu),
                "stored_final_error": float(meta.get("final_error", np.nan)),
                "N_actual": thermo.particle_number_actual,
                "density_mismatch_max": thermo.density_mismatch_max,
                "Phi_abs": float(abs(phi)),
                "m_same_q0_abs": float(abs(currents["same_q0"])),
                "m_opposite_q0_abs": float(abs(currents["opposite_q0"])),
                "Omega_supercell": thermo.grand_potential,
                "F_supercell": thermo.helmholtz_free_energy,
                "F_per_primitive_cell": thermo.free_energy_per_primitive_cell,
                "Omega0": thermo.omega0,
                "LW_fermionic": thermo.fermionic_lw,
                "Phi_H": thermo.phi_hartree,
                "Phi_F": thermo.phi_fock,
                "Phi_corr": thermo.phi_correlation,
            }
        )

    fmin = min(row["F_per_primitive_cell"] for row in rows)
    for row in rows:
        row["DeltaF_per_primitive_cell"] = row["F_per_primitive_cell"] - fmin

    print("=" * 116)
    print("18-site fixed-filling split-GW free-energy comparison")
    print(
        f"V={params.V:g}, T={grid.T:g}, primitive filling={primitive_filling:g}, "
        f"grid={grid.nk1}x{grid.nk2}, nw={grid.nw}, nOmega={grid.nOmega}"
    )
    print("Compare F=Omega+mu*N; DeltaF is per primitive cell.")
    print("=" * 116)
    for i, row in enumerate(rows, start=1):
        print(
            f"[{i}] {row['checkpoint']}\n"
            f"    mu={row['mu']:.10f}, |Phi|={row['Phi_abs']:.6e}, "
            f"|m_same|={row['m_same_q0_abs']:.6e}, "
            f"|m_opposite|={row['m_opposite_q0_abs']:.6e}\n"
            f"    Omega/sc={row['Omega_supercell']:+.12e}, "
            f"F/sc={row['F_supercell']:+.12e}, "
            f"F/pc={row['F_per_primitive_cell']:+.12e}, "
            f"DeltaF/pc={row['DeltaF_per_primitive_cell']:+.6e}\n"
            f"    density mismatch={row['density_mismatch_max']:.3e}, "
            f"stored residual={row['stored_final_error']:.3e}"
        )

    winner = min(rows, key=lambda x: x["F_per_primitive_cell"])
    print("\nlowest supplied basin:", winner["checkpoint"])

    if args.csv is not None:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("CSV:", out)


if __name__ == "__main__":
    main()
