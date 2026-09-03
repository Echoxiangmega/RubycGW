#!/usr/bin/env python3
"""Evaluate complete bulk orbital magnetization from an 18-site GW checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rubycgw.bulk_orbital_magnetization import (
    analyze_checkpoint_bulk_orbital_magnetization,
)
from rubycgw.checkpoint import read_checkpoint_metadata
from rubycgw.magnetic_self_energy import (
    solve_checkpoint_uniform_B_self_energy_derivative,
)
from rubycgw.supercell_cgw import SupercellVertexOptions


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
            "Optional precomputed NPZ containing d Sigma_tilde / db for the "
            "second term of Nourafkan Eq. (2). If omitted, nonzero-V checkpoints "
            "compute this response automatically with the uniform-B GW solver."
        ),
    )
    p.add_argument(
        "--sigma-b-key",
        default="Sigma_B_code",
        help="Array key inside --sigma-b-npz (default: Sigma_B_code).",
    )
    p.add_argument(
        "--no-auto-sigma-b",
        action="store_true",
        help="Do not solve the uniform-B GW self-energy derivative automatically.",
    )
    p.add_argument("--field-tol", type=float, default=1e-8)
    p.add_argument("--field-max-iter", type=int, default=150)
    p.add_argument("--field-gmres-restart", type=int, default=12)
    p.add_argument("--field-mixing", type=float, default=0.25)
    p.add_argument("--field-solver", choices=["gmres", "linear"], default="gmres")
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--field-verbose", action="store_true")
    p.add_argument("--no-field-hartree", action="store_true")
    p.add_argument("--no-field-fock", action="store_true")
    p.add_argument("--no-field-mt", action="store_true")
    p.add_argument("--no-field-al", action="store_true")
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
    require_converged = not args.allow_nonconverged_checkpoint
    meta0 = read_checkpoint_metadata(args.checkpoint)

    sigma_b = None
    field_response = None
    sigma_b_origin = "none"

    if args.sigma_b_npz is not None:
        with np.load(args.sigma_b_npz, allow_pickle=False) as data:
            if args.sigma_b_key not in data:
                raise KeyError(
                    f"{args.sigma_b_npz} has no key {args.sigma_b_key!r}; "
                    f"available keys: {list(data.keys())}"
                )
            sigma_b = np.asarray(data[args.sigma_b_key], dtype=complex)
        sigma_b_origin = f"loaded:{args.sigma_b_npz}"
    elif not args.no_auto_sigma_b and abs(float(meta0["V"])) >= 1e-15:
        fopts = SupercellVertexOptions(
            max_iter=args.field_max_iter,
            tol=args.field_tol,
            mixing=args.field_mixing,
            solver=args.field_solver,
            gmres_restart=args.field_gmres_restart,
            include_hartree=not args.no_field_hartree,
            include_fock=not args.no_field_fock,
            include_mt=not args.no_field_mt,
            include_al=not args.no_field_al,
            verbose=args.field_verbose,
            momentum_backend=args.momentum_backend,
        )
        field_response = solve_checkpoint_uniform_B_self_energy_derivative(
            args.checkpoint,
            require_converged=require_converged,
            opts=fopts,
        )
        if not field_response.converged:
            raise RuntimeError(
                "uniform-B GW self-energy derivative did not converge: "
                f"iterations={field_response.iterations}, "
                f"residual={field_response.final_error:.3e}. "
                "Increase --field-max-iter or adjust the field-response solver."
            )
        sigma_b = field_response.Sigma_B_code
        sigma_b_origin = "auto_uniform_B_GW"

    result = analyze_checkpoint_bulk_orbital_magnetization(
        args.checkpoint,
        sigma_b=sigma_b,
        require_converged=require_converged,
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

    if field_response is not None:
        print(
            "uniform-B gauge-invariant GW self-energy derivative:\n"
            f"  origin                 = {sigma_b_origin}\n"
            f"  converged              = {field_response.converged}\n"
            f"  iterations             = {field_response.iterations}\n"
            f"  equation residual      = {field_response.final_error:.3e}\n"
            f"  |Y_B|                  = {field_response.geometric_G_source_max:.3e}\n"
            f"  |Sigma_B source|       = {field_response.source_sigma_max:.3e}\n"
            f"  |Sigma_B|              = {field_response.sigma_B_max:.3e}\n"
            f"  |G_B|                  = {field_response.G_B_max:.3e}"
        )
    elif sigma_b is not None:
        print(f"uniform-B self-energy derivative: origin={sigma_b_origin}")

    print(
        "field-self-energy term:\n"
        f"  status                 = {result.field_self_energy_status}\n"
        f"  M2(supercell)          = {_fmt_optional(result.field_self_energy_term_code)}\n"
        f"  imaginary residual     = {_fmt_optional(result.field_self_energy_term_imag_residual, '+.3e')}"
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
            "complete bulk M is NOT reported. For nonlocal interacting GW, "
            "Nourafkan Eq.(2) requires the uniform-B gauge-invariant "
            "d Sigma_tilde^(B)/dB term."
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
        if sigma_b is not None:
            payload["Sigma_B_code"] = np.asarray(sigma_b, dtype=complex)
        if field_response is not None:
            payload["G_B_code"] = field_response.G_B_code
            payload["geometric_G_source_code"] = field_response.geometric_G_source_code
            payload["Sigma_H_B_code"] = field_response.Sigma_H_B_code
            payload["Sigma_F_B_code"] = field_response.Sigma_F_B_code
            payload["Sigma_MT_B_code"] = field_response.Sigma_MT_B_code
            payload["Sigma_AL1_B_code"] = field_response.Sigma_AL1_B_code
            payload["Sigma_AL2_B_code"] = field_response.Sigma_AL2_B_code
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
            "M2_imag_residual": result.field_self_energy_term_imag_residual,
            "M_total_code": result.total_code,
            "M_total_per_primitive_code": result.total_per_primitive_code,
            "complete": result.complete,
            "field_self_energy_status": result.field_self_energy_status,
            "sigma_b_origin": sigma_b_origin,
            "max_abs_Dx_H0": result.max_abs_Dx_H0,
            "max_abs_Dy_H0": result.max_abs_Dy_H0,
            "max_abs_Dx_Sigma": result.max_abs_Dx_Sigma,
            "max_abs_Dy_Sigma": result.max_abs_Dy_Sigma,
            "M1_muB": result.main_term_muB,
            "M1_per_primitive_muB": result.main_term_per_primitive_muB,
            "M_total_muB": result.total_muB,
            "M_total_per_primitive_muB": result.total_per_primitive_muB,
        }
        if field_response is not None:
            out["uniform_B_response"] = {
                "converged": field_response.converged,
                "iterations": field_response.iterations,
                "final_error": field_response.final_error,
                "geometric_G_source_max": field_response.geometric_G_source_max,
                "source_sigma_max": field_response.source_sigma_max,
                "sigma_B_max": field_response.sigma_B_max,
                "G_B_max": field_response.G_B_max,
                "momentum_backend": field_response.momentum_backend,
            }
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote: {args.json}")


if __name__ == "__main__":
    main()
