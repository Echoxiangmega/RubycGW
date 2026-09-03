#!/usr/bin/env python3
"""Compute and validate electromagnetic covariant GW response from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rubycgw.electromagnetic import (
    EM_CHANNELS,
    compare_covariant_to_finite_difference,
    finite_difference_electromagnetic_response,
    load_electromagnetic_background,
    solve_electromagnetic_response,
)
from rubycgw.gw import GWOptions
from rubycgw.supercell_cgw import SupercellVertexOptions


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--channel",
        choices=EM_CHANNELS,
        default="same",
        help=(
            "Periodic triangle-flux channel. 'opposite'=(A+B)/sqrt(2), "
            "'same'=(A-B)/sqrt(2), following the project's physical-current labels."
        ),
    )
    p.add_argument(
        "--fixed-mu",
        action="store_true",
        help="Hold chemical potential fixed instead of enforcing the checkpoint filling.",
    )
    p.add_argument(
        "--allow-nonconverged-checkpoint",
        action="store_true",
        help="Permit diagnostic use of a checkpoint marked nonconverged.",
    )
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")

    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-max-iter", type=int, default=150)
    p.add_argument("--gmres-restart", type=int, default=12)
    p.add_argument("--vertex-verbose", action="store_true")
    p.add_argument("--no-hartree", action="store_true")
    p.add_argument("--no-fock", action="store_true")
    p.add_argument("--no-mt", action="store_true")
    p.add_argument("--no-al", action="store_true")

    p.add_argument(
        "--finite-difference",
        type=float,
        default=None,
        metavar="DELTA_PHI",
        help=(
            "Also run fully self-consistent GW at +/-DELTA_PHI and compare "
            "the centered finite difference with the covariant derivative."
        ),
    )
    p.add_argument("--fd-gw-tol", type=float, default=1e-9)
    p.add_argument("--fd-gw-max-iter", type=int, default=250)
    p.add_argument("--fd-gw-mixing", type=float, default=0.20)
    p.add_argument("--fd-mu-tol", type=float, default=1e-10)
    p.add_argument("--fd-mu-max-iter", type=int, default=80)
    p.add_argument("--fd-verbose", action="store_true")
    p.add_argument(
        "--allow-unconverged-fd",
        action="store_true",
        help="Return finite-difference diagnostics even if +/- GW misses its tolerance.",
    )

    p.add_argument("--npz", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args()


def _maxabs(x) -> float:
    return float(np.max(np.abs(np.asarray(x))))


def _metric_line(name: str, metric: dict[str, float]) -> str:
    return (
        f"{name:10s} abs_max={metric['abs_max']:.3e}  "
        f"rms={metric['rms']:.3e}  rel_max={metric['rel_max']:.3e}"
    )


def main() -> None:
    args = _parse_args()
    fixed_filling = not args.fixed_mu

    background = load_electromagnetic_background(
        args.checkpoint,
        require_converged=not args.allow_nonconverged_checkpoint,
        momentum_backend=args.momentum_backend,
    )
    vopts = SupercellVertexOptions(
        max_iter=args.vertex_max_iter,
        tol=args.vertex_tol,
        solver="gmres",
        gmres_restart=args.gmres_restart,
        include_hartree=not args.no_hartree,
        include_fock=not args.no_fock,
        include_mt=not args.no_mt,
        include_al=not args.no_al,
        verbose=args.vertex_verbose,
        momentum_backend=args.momentum_backend,
    )
    response = solve_electromagnetic_response(
        background,
        args.channel,
        fixed_filling=fixed_filling,
        vertex_options=vopts,
    )

    meta = background.metadata
    ensemble = "fixed filling" if fixed_filling else "fixed mu"
    print(
        f"checkpoint={args.checkpoint}\n"
        f"V={float(meta['V']):g}, filling={float(meta['primitive_filling']):g}, "
        f"T={float(meta['T']):g}, channel={response.channel}, ensemble={ensemble}"
    )
    print(
        f"flux weights (A,B)=({response.flux_weights[0]:+.9f},"
        f"{response.flux_weights[1]:+.9f})"
    )
    print(
        f"vertex: converged={response.vertex_converged}, "
        f"iterations={response.vertex_iterations}, residual={response.vertex_final_error:.3e}"
    )
    if response.mu_vertex_converged is not None:
        print(
            f"mu vertex: converged={response.mu_vertex_converged}, "
            f"iterations={response.mu_vertex_iterations}, "
            f"residual={response.mu_vertex_final_error:.3e}"
        )
    print(f"dmu/dphi = {response.mu_phi:+.12e}")
    print(
        f"sum_a dn_a/dphi = {response.density_constraint_residual:+.3e}, "
        f"vertex equation residual = {response.equation_residual:.3e}"
    )
    print(
        "max norms: "
        f"|Gamma|={_maxabs(response.Gamma):.3e}, "
        f"|G_phi|={_maxabs(response.G_phi):.3e}, "
        f"|Sigma_H,phi|={_maxabs(response.Sigma_H_phi):.3e}, "
        f"|Sigma_GW,phi|={_maxabs(response.Sigma_GW_phi):.3e}, "
        f"|W_phi|={_maxabs(response.W_phi):.3e}"
    )

    fd = None
    metrics = None
    if args.finite_difference is not None:
        fd_opts = GWOptions(
            mu=background.mu,
            target_filling=(3.0 * float(meta["primitive_filling"]) if fixed_filling else None),
            max_iter=args.fd_gw_max_iter,
            tol=args.fd_gw_tol,
            mixing=args.fd_gw_mixing,
            mixing_method="pulay",
            pulay_history=6,
            pulay_start=3,
            pulay_regularization=1e-10,
            mu_tol=args.fd_mu_tol,
            mu_max_iter=args.fd_mu_max_iter,
            verbose=args.fd_verbose,
            momentum_backend=args.momentum_backend,
        )
        fd = finite_difference_electromagnetic_response(
            background,
            args.channel,
            args.finite_difference,
            fixed_filling=fixed_filling,
            gw_options=fd_opts,
            require_converged=not args.allow_unconverged_fd,
        )
        metrics = compare_covariant_to_finite_difference(response, fd)

        print(
            "\nfinite difference: "
            f"delta_phi={fd.delta_phi:.3e}, "
            f"+GW(conv={fd.plus_converged}, it={fd.plus_iterations}, err={fd.plus_error:.3e}), "
            f"-GW(conv={fd.minus_converged}, it={fd.minus_iterations}, err={fd.minus_error:.3e})"
        )
        print(
            f"FD dmu/dphi={fd.mu_phi:+.12e}; "
            f"analytic-FD={response.mu_phi - fd.mu_phi:+.3e}"
        )
        print("covariant vs centered finite difference:")
        for name in ("Gamma", "G", "Sigma_H", "Sigma_GW", "P", "W", "density", "mu"):
            print("  " + _metric_line(name, metrics[name]))
        print(
            "\nCentral-difference error should decrease ~delta_phi^2 until GW "
            "convergence and Matsubara-cutoff errors dominate. Increase nw if the "
            "error plateaus above the requested accuracy."
        )

    if args.npz is not None:
        args.npz.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "K": response.K,
            "Gamma": response.Gamma,
            "G_phi": response.G_phi,
            "P_phi": response.P_phi,
            "W_phi": response.W_phi,
            "Sigma_H_phi": response.Sigma_H_phi,
            "Sigma_GW_phi": response.Sigma_GW_phi,
            "density_phi": response.density_phi,
            "mu_phi": np.asarray(response.mu_phi),
        }
        if fd is not None:
            payload.update(
                {
                    "fd_Gamma": fd.Gamma_phi,
                    "fd_G_phi": fd.G_phi,
                    "fd_P_phi": fd.P_phi,
                    "fd_W_phi": fd.W_phi,
                    "fd_Sigma_H_phi": fd.Sigma_H_phi,
                    "fd_Sigma_GW_phi": fd.Sigma_GW_phi,
                    "fd_density_phi": fd.density_phi,
                    "fd_mu_phi": np.asarray(fd.mu_phi),
                    "fd_delta_phi": np.asarray(fd.delta_phi),
                }
            )
        np.savez_compressed(args.npz, **payload)
        print("wrote:", args.npz)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "checkpoint": str(args.checkpoint),
            "V": float(meta["V"]),
            "primitive_filling": float(meta["primitive_filling"]),
            "T": float(meta["T"]),
            "channel": response.channel,
            "flux_weights": list(response.flux_weights),
            "fixed_filling": fixed_filling,
            "mu_phi": response.mu_phi,
            "density_constraint_residual": response.density_constraint_residual,
            "equation_residual": response.equation_residual,
            "vertex_converged": response.vertex_converged,
            "vertex_iterations": response.vertex_iterations,
            "vertex_final_error": response.vertex_final_error,
            "mu_vertex_converged": response.mu_vertex_converged,
            "mu_vertex_iterations": response.mu_vertex_iterations,
            "mu_vertex_final_error": response.mu_vertex_final_error,
        }
        if fd is not None:
            summary["finite_difference"] = {
                "delta_phi": fd.delta_phi,
                "plus_converged": fd.plus_converged,
                "minus_converged": fd.minus_converged,
                "plus_error": fd.plus_error,
                "minus_error": fd.minus_error,
                "mu_phi": fd.mu_phi,
                "metrics": metrics,
            }
        with args.json.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print("wrote:", args.json)


if __name__ == "__main__":
    main()
