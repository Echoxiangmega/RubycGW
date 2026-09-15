#!/usr/bin/env python3
"""Compare GG, cGW and self-consistent cGW+SOX in primitive-cell V scans.

The three production labels have a fixed meaning:

* GG: particle-hole bubble built from the converged ordinary SC-GW Green function;
* cGW: tail-consistent covariant GW vertex on that same SC-GW background;
* cGW+SOX: covariant H/F/MT/AL/SOX vertex on a self-consistent GW+SOX background.

A fixed-GW-background SOX response can optionally be saved as a diagnostic, but
it is deliberately not promoted to a production method label.

Both the historical represented-Matsubara-box susceptibility and the analytic
static-tail-completed susceptibility are stored.  The latter is the appropriate
quantity for comparison to an exact static response.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.gw_sox import solve_primitive_gw_sox
from rubycgw.model import RubyParameters
from rubycgw.primitive_gw import primitive_background_arrays, solve_gw
from rubycgw.production_cgw import solve_vertex_q0_tail
from rubycgw.production_cgw_sox import (
    solve_vertex_q0_tail_sox,
    static_gg_tail_completed,
    static_response_tail_completed,
)
from rubycgw.pseudospin import canonical_channel_name, primitive_pseudospin_vertex
from rubycgw.response_tail import build_tail_reference
from rubycgw.sox_covariant import SOXOptions
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw_split import compute_sigma_gw_split_components


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75, 1.0])
    p.add_argument(
        "--channels", nargs="+", default=["x_even", "z_same", "z_opposite"]
    )
    p.add_argument("--filling", type=float, default=3.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nk1", type=int, default=3)
    p.add_argument("--nk2", type=int, default=3)
    p.add_argument("--nw", type=int, default=40)
    p.add_argument("--nomega", type=int, default=10)
    p.add_argument("--gw-max-iter", type=int, default=120)
    p.add_argument("--gw-tol", type=float, default=2e-8)
    p.add_argument("--mixing", type=float, default=0.25)
    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--vertex-max-iter", type=int, default=160)
    p.add_argument("--vertex-tol", type=float, default=2e-8)
    p.add_argument("--gmres-restart", type=int, default=12)
    p.add_argument("--sox-nquad", type=int, default=128)
    p.add_argument("--sox-interaction-tol", type=float, default=1e-13)
    p.add_argument("--tail-edge-points", type=int, default=2)
    p.add_argument("--backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--fixed-gw-sox-diagnostic", action="store_true")
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/primitive_gw_cgw_sox_vscan"))
    return p.parse_args()


def _require(ok, message, allow):
    if not ok and not allow:
        raise RuntimeError(message)
    if not ok:
        print("WARNING:", message)


def _diag(matrix):
    return np.real(np.diag(np.asarray(matrix)))


def main():
    args = _args()
    channels = [canonical_channel_name(x) for x in args.channels]
    vertices = np.stack([primitive_pseudospin_vertex(x) for x in channels])
    Vvalues = np.asarray(args.V, dtype=float)
    if np.any(Vvalues < 0):
        raise ValueError("this scan expects non-negative V")

    grid = MatsubaraGrid(
        nk1=args.nk1,
        nk2=args.nk2,
        nw=args.nw,
        nOmega=args.nomega,
        T=args.T,
    )
    gw_opts = GWOptions(
        target_filling=args.filling,
        max_iter=args.gw_max_iter,
        tol=args.gw_tol,
        mixing=args.mixing,
        mixing_method=args.mixing_method,
        verbose=args.verbose,
        momentum_backend=args.backend,
    )
    vertex_opts = SupercellVertexOptions(
        max_iter=args.vertex_max_iter,
        tol=args.vertex_tol,
        solver="gmres",
        gmres_restart=args.gmres_restart,
        verbose=args.verbose,
        momentum_backend=args.backend,
    )
    sox_opts = SOXOptions(
        n_quad=args.sox_nquad,
        interaction_tol=args.sox_interaction_tol,
        tail_complete=True,
        tail_edge_points=args.tail_edge_points,
    )

    nV = len(Vvalues)
    nc = len(channels)
    shape = (nV, nc, nc)
    data = {
        "chi_gg_raw": np.full(shape, np.nan + 0j),
        "chi_gg_completed": np.full(shape, np.nan + 0j),
        "chi_cgw_raw": np.full(shape, np.nan + 0j),
        "chi_cgw_completed": np.full(shape, np.nan + 0j),
        "chi_cgw_sox_raw": np.full(shape, np.nan + 0j),
        "chi_cgw_sox_completed": np.full(shape, np.nan + 0j),
        "tail_gg": np.full(shape, np.nan + 0j),
        "tail_cgw": np.full(shape, np.nan + 0j),
        "tail_cgw_sox": np.full(shape, np.nan + 0j),
        "gw_converged": np.zeros(nV, dtype=bool),
        "gwsox_converged": np.zeros(nV, dtype=bool),
        "gw_error": np.full(nV, np.nan),
        "gwsox_error": np.full(nV, np.nan),
        "gw_mu": np.full(nV, np.nan),
        "gwsox_mu": np.full(nV, np.nan),
        "max_sigma_sox": np.full(nV, np.nan),
        "cgw_vertex_converged": np.zeros((nV, nc), dtype=bool),
        "cgw_sox_vertex_converged": np.zeros((nV, nc), dtype=bool),
    }
    if args.fixed_gw_sox_diagnostic:
        data["chi_fixed_gw_sox_raw"] = np.full(shape, np.nan + 0j)
        data["chi_fixed_gw_sox_completed"] = np.full(shape, np.nan + 0j)

    gw_initial = None
    gwsox_initial = None
    for iv, V in enumerate(Vvalues):
        print(f"\n=== V={V:g} ({iv+1}/{nV}) ===")
        params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=float(V))
        h0, Vq = primitive_background_arrays(params, grid)

        gw = solve_gw(params, grid, opts=gw_opts, initial=gw_initial)
        gw_initial = gw
        data["gw_converged"][iv] = gw.converged
        data["gw_error"][iv] = gw.final_error
        data["gw_mu"][iv] = gw.mu
        _require(
            gw.converged,
            f"ordinary GW did not converge at V={V:g}: residual={gw.final_error:.3e}",
            args.allow_unconverged,
        )

        _, sigma_f_gw, _, _ = compute_sigma_gw_split_components(
            gw.G, gw.W, Vq, grid, h0, gw.mu, gw.Sigma_H, backend=args.backend
        )
        h_static_gw = h0 + gw.Sigma_H[None, None] + sigma_f_gw
        reference_gw = build_tail_reference(h0, gw.mu, gw.Sigma_H, grid)

        gg = static_gg_tail_completed(
            gw.G,
            vertices,
            grid,
            h_static_gw,
            gw.mu,
            edge_points=args.tail_edge_points,
        )
        data["chi_gg_raw"][iv] = gg["raw"]
        data["chi_gg_completed"][iv] = gg["completed"]
        data["tail_gg"][iv] = gg["tail_correction"]

        cgw_results = []
        for ic, (channel, K) in enumerate(zip(channels, vertices)):
            print(f"  cGW vertex: {channel}")
            vr = solve_vertex_q0_tail(
                gw.G, gw.W, Vq, K, grid, reference_gw, opts=vertex_opts
            )
            cgw_results.append(vr)
            data["cgw_vertex_converged"][iv, ic] = vr.converged
            _require(
                vr.converged,
                f"cGW vertex {channel} did not converge at V={V:g}: {vr.final_error:.3e}",
                args.allow_unconverged,
            )
        cgw_resp = static_response_tail_completed(
            gw.G,
            vertices,
            [x.Gamma for x in cgw_results],
            grid,
            h_static_gw,
            gw.mu,
            edge_points=args.tail_edge_points,
        )
        data["chi_cgw_raw"][iv] = cgw_resp["raw"]
        data["chi_cgw_completed"][iv] = cgw_resp["completed"]
        data["tail_cgw"][iv] = cgw_resp["tail_correction"]

        if args.fixed_gw_sox_diagnostic:
            fixed = []
            for channel, K in zip(channels, vertices):
                print(f"  fixed-GW + SOX diagnostic: {channel}")
                fixed.append(
                    solve_vertex_q0_tail_sox(
                        gw.G,
                        gw.W,
                        Vq,
                        K,
                        grid,
                        reference_gw,
                        h_static_gw,
                        gw.mu,
                        vertex_opts=vertex_opts,
                        sox_opts=sox_opts,
                    )
                )
            fixed_resp = static_response_tail_completed(
                gw.G,
                vertices,
                [x.Gamma for x in fixed],
                grid,
                h_static_gw,
                gw.mu,
                edge_points=args.tail_edge_points,
            )
            data["chi_fixed_gw_sox_raw"][iv] = fixed_resp["raw"]
            data["chi_fixed_gw_sox_completed"][iv] = fixed_resp["completed"]

        print("  solving self-consistent GW+SOX background")
        gwsox = solve_primitive_gw_sox(
            params,
            grid,
            opts=gw_opts,
            sox_opts=sox_opts,
            initial=gwsox_initial if gwsox_initial is not None else gw,
        )
        gwsox_initial = gwsox
        data["gwsox_converged"][iv] = gwsox.converged
        data["gwsox_error"][iv] = gwsox.final_error
        data["gwsox_mu"][iv] = gwsox.mu
        data["max_sigma_sox"][iv] = float(np.max(np.abs(gwsox.Sigma_SOX)))
        _require(
            gwsox.converged,
            f"GW+SOX did not converge at V={V:g}: residual={gwsox.final_error:.3e}",
            args.allow_unconverged,
        )

        reference_sox = build_tail_reference(h0, gwsox.mu, gwsox.Sigma_H, grid)
        h_static_sox = h0 + gwsox.Sigma_H[None, None] + gwsox.Sigma_F
        sox_vertices = []
        for ic, (channel, K) in enumerate(zip(channels, vertices)):
            print(f"  cGW+SOX vertex: {channel}")
            vr = solve_vertex_q0_tail_sox(
                gwsox.G,
                gwsox.W,
                Vq,
                K,
                grid,
                reference_sox,
                h_static_sox,
                gwsox.mu,
                vertex_opts=vertex_opts,
                sox_opts=sox_opts,
            )
            sox_vertices.append(vr)
            data["cgw_sox_vertex_converged"][iv, ic] = vr.converged
            _require(
                vr.converged,
                f"cGW+SOX vertex {channel} did not converge at V={V:g}: {vr.final_error:.3e}",
                args.allow_unconverged,
            )
        sox_resp = static_response_tail_completed(
            gwsox.G,
            vertices,
            [x.Gamma for x in sox_vertices],
            grid,
            h_static_sox,
            gwsox.mu,
            edge_points=args.tail_edge_points,
        )
        data["chi_cgw_sox_raw"][iv] = sox_resp["raw"]
        data["chi_cgw_sox_completed"][iv] = sox_resp["completed"]
        data["tail_cgw_sox"][iv] = sox_resp["tail_correction"]

        print("  completed diagonal susceptibilities")
        for name, key in (
            ("GG", "chi_gg_completed"),
            ("cGW", "chi_cgw_completed"),
            ("cGW+SOX", "chi_cgw_sox_completed"),
        ):
            vals = _diag(data[key][iv])
            print("   ", name.ljust(9), " ".join(
                f"{ch}={val:+.8f}" for ch, val in zip(channels, vals)
            ))

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "scan.npz",
        V=Vvalues,
        channels=np.asarray(channels),
        **data,
    )
    config = {
        "methods": {
            "GG": "bubble on converged ordinary SC-GW background",
            "cGW": "tail-consistent H/F/MT/AL covariant response on SC-GW background",
            "cGW+SOX": "H/F/MT/AL/SOX covariant response on self-consistent GW+SOX background",
            "fixed_GW_plus_SOX": bool(args.fixed_gw_sox_diagnostic),
        },
        "parameters": vars(args).copy(),
        "channels": channels,
        "note": (
            "Both raw finite-box and analytic-tail-completed static responses are saved. "
            "Use *_completed for ED/static thermodynamic comparisons."
        ),
    }
    config["parameters"]["out"] = str(args.out)
    with (args.out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nwrote {args.out/'scan.npz'}")
    print(f"wrote {args.out/'config.json'}")


if __name__ == "__main__":
    main()
