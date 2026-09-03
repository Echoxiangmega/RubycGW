#!/usr/bin/env python3
"""Evaluate bulk orbital magnetization from an 18-site GW checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rubycgw.bulk_orbital_magnetization import (
    analyze_checkpoint_bulk_orbital_magnetization,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--allow-nonconverged-checkpoint",
        action="store_true",
        help="Permit diagnostic use of a checkpoint marked nonconverged.",
    )
    p.add_argument(
        "--sigma-b-npz",
        type=Path,
        default=None,
        help=(
            "Optional NPZ containing d Sigma_tilde / db for the second term of "
            "Nourafkan Eq. (2), where b=(e a^2/hbar) B_z."
        ),
    )
    p.add_argument(
        "--sigma-b-key",
        default="Sigma_B_code",
        help="Array key inside --sigma-b-npz (default: Sigma_B_code).",
    )
    p.add_argument("--energy-unit-ev", type=float, default=None)
    p.add_argument("--lattice-constant-angstrom", type=float, default=None)
    p.add_argument("--spin-degeneracy", type=float, default=1.0)
    p.add_argument("--npz", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args()


def _fmt_optional(value: float | None, fmt: str = "+.12e") -> str:
    if value is None:
        return "not available"
    return format(float(value), fmt)


def main() -> None:
    args = _parse_args()

    sigma_b = None
    if args.sigma_b_npz is not None:
        with np.load(args.sigma_b_npz, allow_pickle=False) as data:
            if args.sigma_b_key not in data:
                raise KeyError(
                    f"{args.sigma_b_npz} has no key {args.sigma_b_key!r}; "
                    f"available keys: {list(data.keys())}"
                )
            sigma_b = np.asarray(data[args.sigma_b_key], dtype=complex)

    result = analyze_checkpoint_bulk_orbital_magnetization(
        args.checkpoint,
        sigma_b=sigma_b,
        require_converged=not args.allow_nonconverged_checkpoint,
        energy_unit_ev=args.energy_unit_ev,
        lattice_constant_angstrom=args.lattice_constant_angstrom,
        spin_degeneracy=args.spin_degeneracy,
    )

    meta = result.metadata
    print(
        f"checkpoint={args.checkpoint}\n"
        f"V={float(meta['V']):g}, filling={float(meta['primitive_filling']):g}, "
        f"T={float(meta['T']):g}, nk={int(meta['nk1'])}x{int(meta['nk2'])}, "
        f"nw={int(meta['nw'])}"
    )
    print(
        "physical derivative norms: "
        f"|Dx H0|={result.max_abs_Dx_H0:.3e}, "
        f"|Dy H0|={result.max_abs_Dy_H0:.3e}, "
        f"|Dx Sigma|={result.max_abs_Dx_Sigma:.3e}, "
        f"|Dy Sigma|={result.max_abs_Dy_Sigma:.3e}"
    )
    print(
        "Nourafkan Eq.(2) first term:\n"
        f"  M1(supercell)          = {result.main_term_code:+.12e} [code]\n"
        f"  M1(primitive cell)     = {result.main_term_per_primitive_code:+.12e} [code]\n"
        f"  M1/area                = {result.main_term_2d_density_code:+.12e} [code]\n"
        f"  imaginary residual     = {result.main_term_imag_residual:+.3e}"
    )
    print(
        "field-self-energy term:\n"
        f"  status                 = {result.field_self_energy_status}\n"
        f"  M2(supercell)          = {_fmt_optional(result.field_self_energy_term_code)}"
    )

    if result.complete:
        print(
            "complete bulk orbital magnetization:\n"
            f"  M(supercell)           = {result.total_code:+.12e} [code]\n"
            f"  M(primitive cell)      = {result.total_per_primitive_code:+.12e} [code]\n"
            f"  M/area                 = {result.total_2d_density_code:+.12e} [code]"
        )
    else:
        print(
            "complete bulk M is NOT reported: for a nonlocal interacting GW "
            "self-energy, Nourafkan Eq.(2) also needs "
            "d Sigma_tilde^(B)/dB at B=0. M1 above is the first term only."
        )

    if result.main_term_muB is not None:
        print(
            "physical units:\n"
            f"  M1(supercell)          = {result.main_term_muB:+.12e} mu_B\n"
            f"  M1(primitive cell)     = {result.main_term_per_primitive_muB:+.12e} mu_B"
        )
        if result.total_muB is not None:
            print(
                f"  M(total, supercell)    = {result.total_muB:+.12e} mu_B\n"
                f"  M(total, primitive)    = {result.total_per_primitive_muB:+.12e} mu_B"
            )

    if args.npz is not None:
        args.npz.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "M1_code": np.asarray(result.main_term_code),
            "M1_per_primitive_code": np.asarray(result.main_term_per_primitive_code),
            "M1_2d_density_code": np.asarray(result.main_term_2d_density_code),
            "M1_imag_residual": np.asarray(result.main_term_imag_residual),
            "k_resolved_M1_code": result.k_resolved_main_code,
            "supercell_area_code": np.asarray(result.supercell_area_code),
            "primitive_cell_area_code": np.asarray(result.primitive_cell_area_code),
        }
        if result.field_self_energy_term_code is not None:
            payload["M2_code"] = np.asarray(result.field_self_energy_term_code)
        if result.k_resolved_field_code is not None:
            payload["k_resolved_M2_code"] = result.k_resolved_field_code
        if result.total_code is not None:
            payload["M_total_code"] = np.asarray(result.total_code)
            payload["M_total_per_primitive_code"] = np.asarray(
                result.total_per_primitive_code
            )
        np.savez_compressed(args.npz, **payload)
        print(f"wrote: {args.npz}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "checkpoint": str(args.checkpoint),
            "V": float(meta["V"]),
            "primitive_filling": float(meta["primitive_filling"]),
            "T": float(meta["T"]),
            "nk1": int(meta["nk1"]),
            "nk2": int(meta["nk2"]),
            "nw": int(meta["nw"]),
            "M1_code": result.main_term_code,
            "M1_per_primitive_code": result.main_term_per_primitive_code,
            "M1_2d_density_code": result.main_term_2d_density_code,
            "M1_imag_residual": result.main_term_imag_residual,
            "M2_code": result.field_self_energy_term_code,
            "M_total_code": result.total_code,
            "M_total_per_primitive_code": result.total_per_primitive_code,
            "complete": result.complete,
            "field_self_energy_status": result.field_self_energy_status,
            "max_abs_Dx_H0": result.max_abs_Dx_H0,
            "max_abs_Dy_H0": result.max_abs_Dy_H0,
            "max_abs_Dx_Sigma": result.max_abs_Dx_Sigma,
            "max_abs_Dy_Sigma": result.max_abs_Dy_Sigma,
            "M1_muB": result.main_term_muB,
            "M1_per_primitive_muB": result.main_term_per_primitive_muB,
            "M_total_muB": result.total_muB,
            "M_total_per_primitive_muB": result.total_per_primitive_muB,
        }
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote: {args.json}")


if __name__ == "__main__":
    main()
