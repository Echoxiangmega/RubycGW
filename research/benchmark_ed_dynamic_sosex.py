#!/usr/bin/env python3
"""Compare ED, SC-GW, static oneW-sym, dynamic SOSEX, and full G3W2 backgrounds.

The intended production case is the exactly geometry-matched 2x1 Ruby torus
represented as one 12-orbital cell (nk=1).  ``dynamic-mode=g3w2`` keeps two
fully frequency-dependent screened W lines.  No covariant/post correction is
performed here: this script tests the background Green function itself.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from rubycgw.dynamic_sosex import DynamicSOSEXOptions
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.gw_dynamic_sosex import solve_matrix_gw_dynamic_sosex
from rubycgw.gw_ssosex import solve_matrix_gw_ssosex
from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.ssosex_static import ScreenedSOSEXOptions
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--low-count", type=int, default=8)
    p.add_argument("--max-iter", type=int, default=1200)
    p.add_argument("--tol", type=float, default=2e-8)
    p.add_argument("--mixing", type=float, default=0.08)
    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--pulay-history", type=int, default=8)
    p.add_argument("--pulay-start", type=int, default=4)
    p.add_argument("--pulay-regularization", type=float, default=1e-9)
    p.add_argument("--backend", choices=["fft", "direct"], default="direct")
    p.add_argument("--nquad", type=int, default=128)
    p.add_argument(
        "--dynamic-mode", choices=["sosex", "2sosex", "g3w2"], default="sosex",
        help="sosex=one dynamic W line; 2sosex=both mixed terms; g3w2=full dynamic W,W",
    )
    p.add_argument("--max-full-sites", type=int, default=36)
    p.add_argument("--initial", choices=["static", "gw", "zero"], default="static")
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/ed_dynamic_sosex_background"))
    return p.parse_args()


def _relerr(a, b):
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    den = max(float(np.linalg.norm(b.ravel())), 1e-300)
    return float(np.linalg.norm((a-b).ravel())/den)


def _green_errors(G, Ged, grid, low_count):
    a = np.asarray(G, dtype=complex)
    if a.ndim == 5:
        a = a[:, 0, 0]
    idx = np.argsort(np.abs(np.asarray(grid.omega)))[:min(int(low_count), grid.nf)]
    return _relerr(a, Ged), _relerr(a[idx], np.asarray(Ged)[idx])


def _delta_sigma_to_ed(G, mu, Ged, mu_ed):
    a = np.asarray(G, dtype=complex)
    if a.ndim == 5:
        a = a[:, 0, 0]
    Ged = np.asarray(Ged, dtype=complex)
    eye = np.eye(a.shape[-1], dtype=complex)
    out = np.empty_like(a)
    for n in range(a.shape[0]):
        out[n] = ((float(mu_ed)-float(mu))*eye
                  + np.linalg.inv(a[n]) - np.linalg.inv(Ged[n]))
    return out


def main():
    args = _args()
    if 6*args.L1*args.L2 > 16:
        raise ValueError("ExactSmallRubyThermal supports at most 16 sites")
    ncell = int(args.L1)*int(args.L2)
    target = float(args.filling)*ncell
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    exact = ExactSmallRubyThermal(args.L1, args.L2, params)
    exact.diagonalize(float(args.V))
    mu_ed = exact.solve_mu(target, args.T)

    grid = MatsubaraGrid(nk1=1, nk2=1, nw=args.nw, nOmega=args.nomega, T=args.T)
    h0 = np.asarray(exact.h0, dtype=complex)[None, None]
    Vunit = np.asarray(exact.Vunit, dtype=complex)
    Vq = (float(args.V)*Vunit)[None, None]
    gw_opts = GWOptions(
        target_filling=target, max_iter=args.max_iter, tol=args.tol,
        mixing=args.mixing, mixing_method=args.mixing_method,
        pulay_history=args.pulay_history, pulay_start=args.pulay_start,
        pulay_regularization=args.pulay_regularization,
        verbose=args.verbose, momentum_backend=args.backend,
    )
    static_opts = ScreenedSOSEXOptions(
        n_quad=args.nquad, tail_complete=True, mode="oneW-sym"
    )
    dyn_opts = DynamicSOSEXOptions(
        n_quad=args.nquad, tail_complete=True, mode=args.dynamic_mode,
        max_full_sites=args.max_full_sites,
    )

    print("=== ED / GW / static oneW / dynamic exchange background benchmark ===")
    print(f"V={args.V:g}, filling={args.filling:g}, T={args.T:g}, "
          f"torus={args.L1}x{args.L2}, nw={args.nw}, nOmega={args.nomega}, "
          f"mode={args.dynamic_mode}")

    print("\nsolving ordinary SC-GW ...")
    gw = solve_matrix_gw_fast(h0, Vq, grid, opts=gw_opts)
    if not gw.converged and not args.allow_unconverged:
        raise RuntimeError(f"GW failed: {gw.final_error:.3e}")
    print(f"  GW: {'OK' if gw.converged else 'FAIL'} iter={gw.iterations} "
          f"r={gw.final_error:.3e} mu={gw.mu:+.10f}")

    static = None
    if args.initial == "static":
        print("\nsolving static oneW-sym for comparison/initialization ...")
        static = solve_matrix_gw_ssosex(
            h0, Vq, grid, opts=gw_opts, ssosex_opts=static_opts, initial=gw
        )
        if not static.converged and not args.allow_unconverged:
            raise RuntimeError(f"static oneW failed: {static.final_error:.3e}")
        print(f"  static: {'OK' if static.converged else 'FAIL'} iter={static.iterations} "
              f"r={static.final_error:.3e} mu={static.mu:+.10f}")

    initial = static if args.initial == "static" else (gw if args.initial == "gw" else None)
    print(f"\nsolving fully frequency-dependent {args.dynamic_mode} ...")
    dyn = solve_matrix_gw_dynamic_sosex(
        h0, Vq, grid, opts=gw_opts, dsosex_opts=dyn_opts, initial=initial
    )
    if not dyn.converged and not args.allow_unconverged:
        raise RuntimeError(f"dynamic calculation failed: {dyn.final_error:.3e}")
    print(f"  dynamic: {'OK' if dyn.converged else 'FAIL'} iter={dyn.iterations} "
          f"r={dyn.final_error:.3e} mu={dyn.mu:+.10f}")
    print(f"  max|Sigma_exchange|={np.max(np.abs(dyn.Sigma_dSOSEX)):.6e}, "
          f"max|Sigma_WpWp|={np.max(np.abs(dyn.Sigma_WpWp)):.6e}, "
          f"mixed-line transpose asym={dyn.mixed_line_relative_difference:.3e}")

    Ged, _ = exact.green_iomega(1j*np.asarray(grid.omega), mu_ed, args.T)
    ggw, lgw = _green_errors(gw.G, Ged, grid, args.low_count)
    gdyn, ldyn = _green_errors(dyn.G, Ged, grid, args.low_count)
    print("\nGreen-function error vs ED:")
    print(f"  GW:      Gerr={ggw:.6e}, Gerr_low={lgw:.6e}")
    gs = ls = np.nan
    if static is not None:
        gs, ls = _green_errors(static.G, Ged, grid, args.low_count)
        print(f"  static:  Gerr={gs:.6e}, Gerr_low={ls:.6e}")
    print(f"  dynamic: Gerr={gdyn:.6e}, Gerr_low={ldyn:.6e}")
    if np.isfinite(gs):
        print(f"  improvement dynamic vs static: full={(gs-gdyn)/gs:+.3%}, "
              f"low={(ls-ldyn)/ls:+.3%}")

    delta = _delta_sigma_to_ed(dyn.G, dyn.mu, Ged, mu_ed)
    bond = np.abs(Vunit) > 1e-13
    np.fill_diagonal(bond, False)
    diag = np.eye(Vunit.shape[0], dtype=bool)
    other = ~(bond | diag)
    print("\nED-required residual Sigma on dynamic background:")
    print("  omega       diag          V-bond        other         total")
    for n in np.flatnonzero(np.asarray(grid.omega) > 0)[:6]:
        D = delta[n]
        print(f"  {grid.omega[n]:8.4f}  {np.linalg.norm(D[diag]):11.4e}  "
              f"{np.linalg.norm(D[bond]):11.4e}  {np.linalg.norm(D[other]):11.4e}  "
              f"{np.linalg.norm(D):11.4e}")

    args.out.mkdir(parents=True, exist_ok=True)
    outfile = args.out / f"V{args.V:g}_fill{args.filling:g}_{args.L1}x{args.L2}_dynamic_{args.dynamic_mode}.npz"
    np.savez_compressed(
        outfile, V=args.V, filling=args.filling, T=args.T,
        L1=args.L1, L2=args.L2, nw=args.nw, nOmega=args.nomega,
        omega=np.asarray(grid.omega), Omega=np.asarray(grid.Omega),
        mu_ed=mu_ed, mu_gw=gw.mu, mu_dynamic=dyn.mu,
        mu_static=np.nan if static is None else static.mu,
        G_ed=np.asarray(Ged), G_gw=np.asarray(gw.G), G_dynamic=np.asarray(dyn.G),
        G_static=np.full_like(dyn.G, np.nan+0j) if static is None else np.asarray(static.G),
        W_dynamic=np.asarray(dyn.W), Sigma_H_dynamic=np.asarray(dyn.Sigma_H),
        Sigma_GW_dynamic=np.asarray(dyn.Sigma_GW),
        Sigma_dSOSEX=np.asarray(dyn.Sigma_dSOSEX),
        Sigma_SOX=np.asarray(dyn.Sigma_SOX), Sigma_WpV=np.asarray(dyn.Sigma_WpV),
        Sigma_VWp=np.asarray(dyn.Sigma_VWp), Sigma_WpWp=np.asarray(dyn.Sigma_WpWp),
        delta_sigma_ed_minus_dynamic=delta,
        mixed_line_relative_difference=dyn.mixed_line_relative_difference,
        gerr_gw=ggw, gerr_low_gw=lgw, gerr_static=gs, gerr_low_static=ls,
        gerr_dynamic=gdyn, gerr_low_dynamic=ldyn,
        dynamic_mode=args.dynamic_mode,
        dynamic_converged=dyn.converged, dynamic_residual=dyn.final_error,
    )
    print(f"\nsaved {outfile}")


if __name__ == "__main__":
    main()
