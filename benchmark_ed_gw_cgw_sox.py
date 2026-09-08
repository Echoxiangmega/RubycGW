#!/usr/bin/env python3
"""Same-torus ED / GG / cGW / cGW+SOX static-response benchmark.

The finite Ruby torus is represented in two exactly matching ways:

* ED uses the full many-body Hilbert space of ``ExactSmallRubyThermal``;
* GW and GW+SOX treat every physical torus site as an orbital and use nk=1.

Thus the comparison contains no lattice-size or momentum-grid mismatch.  The
production method labels are ED, GG, cGW and cGW+SOX.  Both raw finite-box and
analytic-tail-completed diagrammatic responses are saved; ED is intrinsically a
full static thermodynamic response and should be compared to the completed
values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.gw_sox import solve_matrix_gw_sox
from rubycgw.model import RubyParameters
from rubycgw.production_cgw import solve_vertex_q0_tail
from rubycgw.production_cgw_sox import (
    solve_vertex_q0_tail_sox,
    static_gg_tail_completed,
    static_response_tail_completed,
)
from rubycgw.response_tail import build_tail_reference
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.sox_covariant import SOXOptions
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast
from rubycgw.supercell_gw_split import compute_sigma_gw_split_components


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--V", nargs="+", type=float, default=[0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument(
        "--channels", nargs="+", default=["x_even", "z_same", "z_opposite"]
    )
    p.add_argument("--filling", type=float, default=3.0, help="particles per primitive cell")
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--gw-max-iter", type=int, default=140)
    p.add_argument("--gw-tol", type=float, default=2e-8)
    p.add_argument("--mixing", type=float, default=0.22)
    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--vertex-max-iter", type=int, default=180)
    p.add_argument("--vertex-tol", type=float, default=2e-8)
    p.add_argument("--gmres-restart", type=int, default=12)
    p.add_argument("--sox-nquad", type=int, default=128)
    p.add_argument("--tail-edge-points", type=int, default=2)
    p.add_argument("--backend", choices=["fft", "direct"], default="direct")
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/ed_gw_cgw_sox_benchmark"))
    return p.parse_args()


def _require(ok, message, allow):
    if not ok and not allow:
        raise RuntimeError(message)
    if not ok:
        print("WARNING:", message)


def main():
    args = _args()
    if 6 * args.L1 * args.L2 > 16:
        raise ValueError("ExactSmallRubyThermal requires at most 16 sites")
    ncell = int(args.L1) * int(args.L2)
    target = float(args.filling) * ncell
    Vvalues = np.asarray(args.V, dtype=float)
    channels = list(args.channels)
    nc = len(channels)
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=args.nw, nOmega=args.nomega, T=args.T)
    gw_opts = GWOptions(
        target_filling=target,
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
        tail_complete=True,
        tail_edge_points=args.tail_edge_points,
    )

    nV = len(Vvalues)
    shape = (nV, nc, nc)
    ed_chi = np.full(shape, np.nan)
    chi_gg_raw = np.full(shape, np.nan + 0j)
    chi_gg_completed = np.full(shape, np.nan + 0j)
    chi_cgw_raw = np.full(shape, np.nan + 0j)
    chi_cgw_completed = np.full(shape, np.nan + 0j)
    chi_sox_raw = np.full(shape, np.nan + 0j)
    chi_sox_completed = np.full(shape, np.nan + 0j)
    tail_gg = np.full(shape, np.nan + 0j)
    tail_cgw = np.full(shape, np.nan + 0j)
    tail_sox = np.full(shape, np.nan + 0j)
    mu_ed = np.full(nV, np.nan)
    mu_gw = np.full(nV, np.nan)
    mu_sox = np.full(nV, np.nan)
    gw_converged = np.zeros(nV, dtype=bool)
    sox_converged = np.zeros(nV, dtype=bool)
    cgw_converged = np.zeros((nV, nc), dtype=bool)
    cgw_sox_converged = np.zeros((nV, nc), dtype=bool)
    max_sigma_sox = np.full(nV, np.nan)

    gw_initial = None
    sox_initial = None
    for iv, V in enumerate(Vvalues):
        print(f"\n=== same-torus benchmark V={V:g} ({iv+1}/{nV}) ===")
        params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=float(V))
        exact = ExactSmallRubyThermal(args.L1, args.L2, params)
        exact.diagonalize(float(V))
        mu_ed[iv] = exact.solve_mu(target, args.T)
        operators = np.stack([
            exact.pseudospin_operator(ch, (0.0, 0.0)) for ch in channels
        ])
        ed_chi[iv], _ = exact.static_susceptibility_matrix(
            operators, mu_ed[iv], args.T
        )
        print("  ED complete")

        h0 = np.asarray(exact.h0, dtype=complex)[None, None]
        Vq = (float(V) * np.asarray(exact.Vunit, dtype=complex))[None, None]
        gw = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts, initial=gw_initial)
        gw_initial = gw
        gw_converged[iv] = gw.converged
        mu_gw[iv] = gw.mu
        _require(
            gw.converged,
            f"GW did not converge at V={V:g}: {gw.final_error:.3e}",
            args.allow_unconverged,
        )
        _, sigma_f_gw, _, _ = compute_sigma_gw_split_components(
            gw.G, gw.W, Vq, grid, h0, gw.mu, gw.Sigma_H, backend=args.backend
        )
        h_static_gw = h0 + gw.Sigma_H[None, None] + sigma_f_gw
        reference_gw = build_tail_reference(h0, gw.mu, gw.Sigma_H, grid)

        gg = static_gg_tail_completed(
            gw.G,
            operators,
            grid,
            h_static_gw,
            gw.mu,
            edge_points=args.tail_edge_points,
        )
        chi_gg_raw[iv] = gg["raw"]
        chi_gg_completed[iv] = gg["completed"]
        tail_gg[iv] = gg["tail_correction"]

        cgw_results = []
        for ic, (ch, K) in enumerate(zip(channels, operators)):
            print(f"  cGW vertex: {ch}")
            vr = solve_vertex_q0_tail(
                gw.G, gw.W, Vq, K, grid, reference_gw, opts=vertex_opts
            )
            cgw_results.append(vr)
            cgw_converged[iv, ic] = vr.converged
            _require(
                vr.converged,
                f"cGW vertex {ch} did not converge at V={V:g}: {vr.final_error:.3e}",
                args.allow_unconverged,
            )
        resp = static_response_tail_completed(
            gw.G,
            operators,
            [x.Gamma for x in cgw_results],
            grid,
            h_static_gw,
            gw.mu,
            edge_points=args.tail_edge_points,
        )
        chi_cgw_raw[iv] = resp["raw"]
        chi_cgw_completed[iv] = resp["completed"]
        tail_cgw[iv] = resp["tail_correction"]

        print("  solving self-consistent GW+SOX")
        gwsox = solve_matrix_gw_sox(
            h0,
            Vq,
            grid,
            opts=gw_opts,
            sox_opts=sox_opts,
            initial=sox_initial if sox_initial is not None else gw,
        )
        sox_initial = gwsox
        sox_converged[iv] = gwsox.converged
        mu_sox[iv] = gwsox.mu
        max_sigma_sox[iv] = float(np.max(np.abs(gwsox.Sigma_SOX)))
        _require(
            gwsox.converged,
            f"GW+SOX did not converge at V={V:g}: {gwsox.final_error:.3e}",
            args.allow_unconverged,
        )
        reference_sox = build_tail_reference(h0, gwsox.mu, gwsox.Sigma_H, grid)
        h_static_sox = h0 + gwsox.Sigma_H[None, None] + gwsox.Sigma_F
        sox_results = []
        for ic, (ch, K) in enumerate(zip(channels, operators)):
            print(f"  cGW+SOX vertex: {ch}")
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
            sox_results.append(vr)
            cgw_sox_converged[iv, ic] = vr.converged
            _require(
                vr.converged,
                f"cGW+SOX vertex {ch} did not converge at V={V:g}: {vr.final_error:.3e}",
                args.allow_unconverged,
            )
        resp_sox = static_response_tail_completed(
            gwsox.G,
            operators,
            [x.Gamma for x in sox_results],
            grid,
            h_static_sox,
            gwsox.mu,
            edge_points=args.tail_edge_points,
        )
        chi_sox_raw[iv] = resp_sox["raw"]
        chi_sox_completed[iv] = resp_sox["completed"]
        tail_sox[iv] = resp_sox["tail_correction"]

        print("  diagonal static susceptibilities (completed where applicable)")
        for ic, ch in enumerate(channels):
            print(
                f"    {ch:12s} ED={ed_chi[iv,ic,ic]:+.8f} "
                f"GG={chi_gg_completed[iv,ic,ic].real:+.8f} "
                f"cGW={chi_cgw_completed[iv,ic,ic].real:+.8f} "
                f"cGW+SOX={chi_sox_completed[iv,ic,ic].real:+.8f}"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "benchmark.npz",
        V=Vvalues,
        channels=np.asarray(channels),
        ed=ed_chi,
        gg_raw=chi_gg_raw,
        gg_completed=chi_gg_completed,
        cgw_raw=chi_cgw_raw,
        cgw_completed=chi_cgw_completed,
        cgw_sox_raw=chi_sox_raw,
        cgw_sox_completed=chi_sox_completed,
        tail_gg=tail_gg,
        tail_cgw=tail_cgw,
        tail_cgw_sox=tail_sox,
        mu_ed=mu_ed,
        mu_gw=mu_gw,
        mu_gw_sox=mu_sox,
        gw_converged=gw_converged,
        gw_sox_converged=sox_converged,
        cgw_vertex_converged=cgw_converged,
        cgw_sox_vertex_converged=cgw_sox_converged,
        max_sigma_sox=max_sigma_sox,
    )
    config = {
        "geometry": {"L1": args.L1, "L2": args.L2, "n_sites": 6*ncell},
        "methods": ["ED", "GG", "cGW", "cGW+SOX"],
        "comparison_rule": "Compare ED to *_completed, not to raw finite-box values.",
        "parameters": vars(args).copy(),
    }
    config["parameters"]["out"] = str(args.out)
    with (args.out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nwrote {args.out/'benchmark.npz'}")
    print(f"wrote {args.out/'config.json'}")


if __name__ == "__main__":
    main()
