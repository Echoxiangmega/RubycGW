#!/usr/bin/env python3
"""Analyze local Ruby plaquette orbital moments from an 18-site GW checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rubycgw.orbital_moment import (
    TRIANGLE_LABELS,
    analyze_checkpoint_orbital_moments,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Reconstruct G from a zero-source 18-site GW checkpoint and report "
            "triangle bond currents and local plaquette orbital moments."
        )
    )
    p.add_argument("checkpoint", type=Path, help="GW checkpoint .npz")
    p.add_argument(
        "--allow-nonconverged",
        action="store_true",
        help="Analyze a checkpoint marked nonconverged (diagnostic use only).",
    )
    p.add_argument(
        "--energy-unit-ev",
        type=float,
        default=None,
        help=(
            "Model hopping-energy unit E0 in eV. Physical mu_B/A output is "
            "enabled only when --lattice-constant-angstrom is also supplied."
        ),
    )
    p.add_argument(
        "--lattice-constant-angstrom",
        type=float,
        default=None,
        help="Length represented by one lattice-coordinate unit, in Angstrom.",
    )
    p.add_argument(
        "--spin-degeneracy",
        type=float,
        default=1.0,
        help="Multiplicity factor for physical-unit conversion; default 1 (spinless model).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional triangle-resolved CSV output path.",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional compact summary JSON output path.",
    )
    return p.parse_args()


def _fmt(x: float) -> str:
    return f"{float(x): .9e}"


def _write_csv(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    physical = result.plaquette_moments_muB is not None
    fields = [
        "sector",
        "triangle",
        "bond_current_01_code",
        "bond_current_12_code",
        "bond_current_20_code",
        "loop_current_code",
        "loop_current_spread_code",
        "signed_area_a2",
        "plaquette_moment_code",
        "cell_net_moment_code",
        "cell_staggered_moment_code",
    ]
    if physical:
        fields += [
            "loop_charge_current_A",
            "plaquette_moment_muB",
            "cell_net_moment_muB",
            "cell_staggered_moment_muB",
        ]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for s in range(result.plaquette_moments_code.shape[0]):
            for itri, tri in enumerate(TRIANGLE_LABELS):
                row = {
                    "sector": s,
                    "triangle": tri,
                    "bond_current_01_code": result.phase_bond_currents[s, itri, 0],
                    "bond_current_12_code": result.phase_bond_currents[s, itri, 1],
                    "bond_current_20_code": result.phase_bond_currents[s, itri, 2],
                    "loop_current_code": result.loop_phase_currents[s, itri],
                    "loop_current_spread_code": result.loop_current_spread[s, itri],
                    "signed_area_a2": result.signed_triangle_areas[s, itri],
                    "plaquette_moment_code": result.plaquette_moments_code[s, itri],
                    "cell_net_moment_code": result.cell_net_moments_code[s],
                    "cell_staggered_moment_code": result.cell_staggered_moments_code[s],
                }
                if physical:
                    row.update(
                        {
                            "loop_charge_current_A": result.loop_charge_currents_A[s, itri],
                            "plaquette_moment_muB": result.plaquette_moments_muB[s, itri],
                            "cell_net_moment_muB": result.cell_net_moments_muB[s],
                            "cell_staggered_moment_muB": result.cell_staggered_moments_muB[s],
                        }
                    )
                writer.writerow(row)


def _write_json(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_metadata": result.metadata,
        "triangle_labels": list(TRIANGLE_LABELS),
        "loop_phase_currents_code": result.loop_phase_currents.tolist(),
        "loop_current_spread_code": result.loop_current_spread.tolist(),
        "signed_triangle_areas_a2": result.signed_triangle_areas.tolist(),
        "plaquette_moments_code": result.plaquette_moments_code.tolist(),
        "cell_net_moments_code": result.cell_net_moments_code.tolist(),
        "cell_staggered_moments_code": result.cell_staggered_moments_code.tolist(),
        "supercell_net_moment_code": result.supercell_net_moment_code,
    }
    if result.plaquette_moments_muB is not None:
        payload.update(
            {
                "loop_charge_currents_A": result.loop_charge_currents_A.tolist(),
                "plaquette_moments_muB": result.plaquette_moments_muB.tolist(),
                "cell_net_moments_muB": result.cell_net_moments_muB.tolist(),
                "cell_staggered_moments_muB": result.cell_staggered_moments_muB.tolist(),
                "supercell_net_moment_muB": result.supercell_net_moment_muB,
            }
        )
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    args = _parse_args()
    result = analyze_checkpoint_orbital_moments(
        args.checkpoint,
        require_converged=not args.allow_nonconverged,
        energy_unit_ev=args.energy_unit_ev,
        lattice_constant_angstrom=args.lattice_constant_angstrom,
        spin_degeneracy=args.spin_degeneracy,
    )

    meta = result.metadata
    status = "converged" if bool(meta.get("converged", False)) else "NONCONVERGED"
    print(
        f"checkpoint={args.checkpoint}\n"
        f"V={float(meta['V']):g}, filling={float(meta['primitive_filling']):g}, "
        f"T={float(meta['T']):g}, status={status}"
    )
    if status != "converged":
        print(
            "WARNING: this is a nonconverged diagnostic state; orbital moments "
            "must not be interpreted as final physical results."
        )

    print(
        "\nCode-unit convention: j_loop=<dH/dphi> [model energy], "
        "m_p=j_loop*A_p [model energy * a^2]."
    )
    print(
        "The A/B signed areas have opposite signs because the project's eta_A "
        "and eta_B algebraic loops have opposite geometric handedness.\n"
    )

    for s in range(result.plaquette_moments_code.shape[0]):
        print(f"primitive sector s={s}")
        for itri, tri in enumerate(TRIANGLE_LABELS):
            bonds = result.phase_bond_currents[s, itri]
            line = (
                f"  {tri}: j_bonds=[{_fmt(bonds[0])}, {_fmt(bonds[1])}, "
                f"{_fmt(bonds[2])}], "
                f"j_loop={_fmt(result.loop_phase_currents[s, itri])}, "
                f"spread={_fmt(result.loop_current_spread[s, itri])}, "
                f"A_signed={_fmt(result.signed_triangle_areas[s, itri])}, "
                f"m_p={_fmt(result.plaquette_moments_code[s, itri])}"
            )
            if result.plaquette_moments_muB is not None:
                line += (
                    f", I={_fmt(result.loop_charge_currents_A[s, itri])} A, "
                    f"m_p={_fmt(result.plaquette_moments_muB[s, itri])} mu_B"
                )
            print(line)

        summary = (
            f"  cell: m_net={_fmt(result.cell_net_moments_code[s])}, "
            f"m_staggered={_fmt(result.cell_staggered_moments_code[s])}"
        )
        if result.cell_net_moments_muB is not None:
            summary += (
                f", m_net={_fmt(result.cell_net_moments_muB[s])} mu_B, "
                f"m_staggered={_fmt(result.cell_staggered_moments_muB[s])} mu_B"
            )
        print(summary)

    print(
        f"\nsupercell net moment = {_fmt(result.supercell_net_moment_code)} "
        "[model energy * a^2]"
    )
    if result.supercell_net_moment_muB is not None:
        print(f"supercell net moment = {_fmt(result.supercell_net_moment_muB)} mu_B")

    max_loop = float(np.max(np.abs(result.loop_phase_currents)))
    max_spread = float(np.max(result.loop_current_spread))
    if max_loop > 0.0:
        print(f"max bond-current spread / max |loop current| = {max_spread / max_loop:.3e}")
    else:
        print("all loop currents are zero within the reconstructed Matsubara box")

    if args.csv is not None:
        _write_csv(args.csv, result)
        print("wrote:", args.csv)
    if args.json is not None:
        _write_json(args.json, result)
        print("wrote:", args.json)


if __name__ == "__main__":
    main()
