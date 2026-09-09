#!/usr/bin/env python3
"""Finite-source no-Gamma comparison: ED / GW / GW+SOX / GW+sSOSEX.

This driver deliberately contains no covariant density vertex and no Gamma_P
feedback.  Both exchange-corrected branches use ordinary self-consistent GW
screening P=GG, W=(1-VP)^(-1)V.

The screened-SOSEX branch is fully self-consistent:

    G -> P=GG -> W -> Sigma_GW + Sigma_sSOSEX[G,V,W(0)] -> G.

It therefore isolates whether screening the crossed-exchange topology itself
improves J and the ED Green-function benchmark, without any Gamma correction.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from benchmark_finite_source_current import _bilinear_expectation_tail_completed
from rubycgw.ed_green_compare import (
    primitive_local_green,
    relative_green_error,
    solve_exact_finite_source_local_green,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.gw_sox_fast import solve_matrix_gw_sox_fast
from rubycgw.gw_ssosex import solve_matrix_gw_ssosex
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import canonical_channel_name, primitive_pseudospin_vertex
from rubycgw.sox_covariant import SOXOptions
from rubycgw.ssosex_static import ScreenedSOSEXOptions
from scan_primitive_finite_source_gw_gamma import (
    _fit,
    _gw_current,
    _gw_retry,
    _hvalues,
    _low_frequency_indices,
    _parse_mesh,
)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--source", choices=["same", "opposite", "z_same", "z_opposite"], default="same")
    p.add_argument("--mesh", default="3x3")
    p.add_argument("--h", nargs="+", type=float, default=[.2, .1, .05, .02, .01])
    p.add_argument("--reference-ncell", type=int, default=2)
    p.add_argument("--T", type=float, default=.08)
    p.add_argument("--ti", type=float, default=.4)
    p.add_argument("--t1", type=float, default=.2)
    p.add_argument("--t2", type=float, default=.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)

    p.add_argument("--gw-max-iter", type=int, default=1000)
    p.add_argument("--gw-mixing", type=float, default=.18)
    p.add_argument("--gw-tol", type=float, default=2e-8)
    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--pulay-history", type=int, default=8)
    p.add_argument("--pulay-start", type=int, default=4)
    p.add_argument("--pulay-regularization", type=float, default=1e-9)
    p.add_argument("--backend", choices=["fft", "direct"], default="fft")

    p.add_argument("--exchange-max-iter", type=int, default=1600)
    p.add_argument("--exchange-mixing", type=float, default=.08)
    p.add_argument("--sox-nquad", type=int, default=128)
    p.add_argument("--ssosex-nquad", type=int, default=128)
    p.add_argument("--ssosex-mode", choices=["oneW-sym", "twoW"], default="oneW-sym")
    p.add_argument("--interaction-tol", type=float, default=1e-13)
    p.add_argument("--tail-edge-points", type=int, default=2)

    p.add_argument("--ed", dest="ed", action="store_true", default=True)
    p.add_argument("--no-ed", dest="ed", action="store_false")
    p.add_argument("--ed-L1", type=int, default=2)
    p.add_argument("--ed-L2", type=int, default=1)
    p.add_argument("--ed-discard-weight-tol", type=float, default=1e-12)
    p.add_argument("--ed-low-nfreq", type=int, default=8)

    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fit-points", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("results/primitive_gw_ssosex"))
    return p.parse_args()


def _exchange_current(result, K, h0, grid):
    h_static = h0 + result.Sigma_H[None, None] + result.Sigma_F
    return float(
        _bilinear_expectation_tail_completed(
            result.G, K, grid, h_static, result.mu
        ).real
    )


def _retry_exchange(label, solver, h0, Vq, grid, opts, extra_name, extra_opts,
                    initial=None, gw_fallback=None):
    plans = [
        ("base", opts),
        ("pulay-0.05", replace(opts, mixing=min(float(opts.mixing), .05),
                               mixing_method="pulay", max_iter=max(int(opts.max_iter), 2200))),
        ("pulay-0.03", replace(opts, mixing=min(float(opts.mixing), .03),
                               mixing_method="pulay", max_iter=max(int(opts.max_iter), 2800),
                               pulay_regularization=max(float(opts.pulay_regularization), 1e-8))),
        ("linear-0.02", replace(opts, mixing=min(float(opts.mixing), .02),
                                mixing_method="linear", max_iter=max(int(opts.max_iter), 3200))),
    ]
    seed = initial if initial is not None else gw_fallback
    best = None
    best_name = "none"
    for ia, (name, trial) in enumerate(plans):
        kwargs = {"opts": trial, extra_name: extra_opts, "initial": seed}
        out = solver(h0, Vq, grid, **kwargs)
        sigma_x = getattr(out, "Sigma_SOX", getattr(out, "Sigma_sSOSEX", None))
        scale = float(np.max(np.abs(sigma_x))) if sigma_x is not None else np.nan
        print(
            f"      {label} retry[{ia}] {name:12s}: "
            f"{'OK' if out.converged else 'FAIL':4s} iter={out.iterations:4d} "
            f"r={out.final_error:.3e} max|Sx|={scale:.3e} "
            f"smin={out.min_screening_singular_value:.3e} "
            f"q*=({out.min_screening_q1:.4f},{out.min_screening_q2:.4f})"
        )
        if best is None or out.final_error < best.final_error:
            best, best_name = out, name
        if out.converged:
            return out, name
        seed = out
    return best, best_name


def main():
    args = _args()
    nk1, nk2 = _parse_mesh(args.mesh)
    h_ref = _hvalues(args.h)
    source = canonical_channel_name(args.source)
    K = np.asarray(primitive_pseudospin_vertex(source), dtype=complex)
    root = float(np.sqrt(args.reference_ncell))
    h_cell = h_ref / root

    if args.ed:
        if int(args.ed_L1) * int(args.ed_L2) != int(args.reference_ncell):
            raise ValueError("ed-L1*ed-L2 must equal reference-ncell")

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(nk1=nk1, nk2=nk2, nw=args.nw, nOmega=args.nomega, T=args.T)
    h0_base = np.asarray(build_h0(grid.kmesh(), params), dtype=complex)
    Vq = np.asarray(build_interaction(grid.qmesh(), params), dtype=complex)

    gw_opts = GWOptions(
        target_filling=args.filling, max_iter=args.gw_max_iter,
        tol=args.gw_tol, mixing=args.gw_mixing,
        mixing_method=args.mixing_method,
        pulay_history=args.pulay_history, pulay_start=args.pulay_start,
        pulay_regularization=args.pulay_regularization,
        verbose=args.verbose, momentum_backend=args.backend,
    )
    exchange_opts = replace(
        gw_opts, max_iter=args.exchange_max_iter, mixing=args.exchange_mixing
    )
    sox_opts = SOXOptions(
        n_quad=args.sox_nquad, interaction_tol=args.interaction_tol,
        tail_complete=True, tail_edge_points=args.tail_edge_points,
    )
    ssosex_opts = ScreenedSOSEXOptions(
        n_quad=args.ssosex_nquad, interaction_tol=args.interaction_tol,
        tail_complete=True, tail_edge_points=args.tail_edge_points,
        mode=args.ssosex_mode,
    )

    n = len(h_ref)
    keys = ("ed", "gw", "sox", "ssosex")
    J = {k: np.full(n, np.nan) for k in keys}
    Gerr = {k: np.full(n, np.nan) for k in ("gw", "sox", "ssosex")}
    Gerr_low = {k: np.full(n, np.nan) for k in ("gw", "sox", "ssosex")}
    ok = {k: np.zeros(n, dtype=bool) for k in ("gw", "sox", "ssosex")}
    err = {k: np.full(n, np.nan) for k in ("gw", "sox", "ssosex")}
    iters = {k: np.zeros(n, dtype=int) for k in ("gw", "sox", "ssosex")}
    scale = {k: np.full(n, np.nan) for k in ("sox", "ssosex")}
    G_local = {
        k: np.full((n, grid.nf, 6, 6), np.nan + 0j, dtype=complex)
        for k in keys
    }

    print("=== no-Gamma: ED / GW / GW+SOX / GW+sSOSEX ===")
    print(f"V={args.V:g}, filling={args.filling:g}, T={args.T:g}, source={source}")
    print(f"mesh={nk1}x{nk2}, h_ref={h_ref.tolist()}")
    print(f"sSOSEX mode={args.ssosex_mode}; screening is ordinary P=GG only")

    last_gw = last_sox = last_ssosex = None
    low_idx = _low_frequency_indices(grid.omega, args.ed_low_nfreq)
    rows = []

    for ih, (href, hpc) in enumerate(zip(h_ref, h_cell)):
        print(f"\n-- h_ref={href:.8g}, h_cell={hpc:.8g} --")
        h0 = h0_base - float(hpc) * K[None, None]
        h0 = .5 * (h0 + np.swapaxes(h0.conj(), -1, -2))

        gw, gw_retry = _gw_retry(h0, Vq, grid, gw_opts, initial=last_gw)
        ok["gw"][ih] = bool(gw.converged)
        err["gw"][ih] = gw.final_error
        iters["gw"][ih] = gw.iterations
        J["gw"][ih] = root * _gw_current(gw, K, Vq, h0, grid, args.backend)
        G_local["gw"][ih] = primitive_local_green(gw.G)
        if gw.converged:
            last_gw = gw
        elif not args.allow_unconverged:
            raise RuntimeError(f"GW failed at h={href:g}")

        if args.ed:
            ed = solve_exact_finite_source_local_green(
                L1=args.ed_L1, L2=args.ed_L2, params=params, V=args.V,
                source_channel=source, h_ref=href,
                filling_per_cell=args.filling, T=args.T,
                omega=grid.omega,
                discard_weight_tol=args.ed_discard_weight_tol,
            )
            J["ed"][ih] = ed.J_ref
            G_local["ed"][ih] = ed.G_local
            Gerr["gw"][ih] = relative_green_error(G_local["gw"][ih], G_local["ed"][ih])
            Gerr_low["gw"][ih] = relative_green_error(
                G_local["gw"][ih, low_idx], G_local["ed"][ih, low_idx]
            )

        seed_sox = last_sox if last_sox is not None else gw
        sox, sox_retry = _retry_exchange(
            "SOX", solve_matrix_gw_sox_fast, h0, Vq, grid, exchange_opts,
            "sox_opts", sox_opts, initial=seed_sox, gw_fallback=gw,
        )
        ok["sox"][ih] = bool(sox.converged)
        err["sox"][ih] = sox.final_error
        iters["sox"][ih] = sox.iterations
        scale["sox"][ih] = float(np.max(np.abs(sox.Sigma_SOX)))
        J["sox"][ih] = root * _exchange_current(sox, K, h0, grid)
        G_local["sox"][ih] = primitive_local_green(sox.G)
        if sox.converged:
            last_sox = sox
        elif not args.allow_unconverged:
            raise RuntimeError(f"GW+SOX failed at h={href:g}")

        seed_sx = last_ssosex if last_ssosex is not None else gw
        sx, sx_retry = _retry_exchange(
            "sSOSEX", solve_matrix_gw_ssosex, h0, Vq, grid, exchange_opts,
            "ssosex_opts", ssosex_opts, initial=seed_sx, gw_fallback=gw,
        )
        ok["ssosex"][ih] = bool(sx.converged)
        err["ssosex"][ih] = sx.final_error
        iters["ssosex"][ih] = sx.iterations
        scale["ssosex"][ih] = float(np.max(np.abs(sx.Sigma_sSOSEX)))
        J["ssosex"][ih] = root * _exchange_current(sx, K, h0, grid)
        G_local["ssosex"][ih] = primitive_local_green(sx.G)
        if sx.converged:
            last_ssosex = sx
        elif not args.allow_unconverged:
            raise RuntimeError(f"GW+sSOSEX failed at h={href:g}")

        if args.ed:
            for k in ("sox", "ssosex"):
                Gerr[k][ih] = relative_green_error(G_local[k][ih], G_local["ed"][ih])
                Gerr_low[k][ih] = relative_green_error(
                    G_local[k][ih, low_idx], G_local["ed"][ih, low_idx]
                )

        print(f"    GW:       J={J['gw'][ih]:+.9f} r={err['gw'][ih]:.3e} retry={gw_retry}")
        print(
            f"    GW+SOX:   J={J['sox'][ih]:+.9f} dJ={J['sox'][ih]-J['gw'][ih]:+.9f} "
            f"r={err['sox'][ih]:.3e} max|SOX|={scale['sox'][ih]:.3e} retry={sox_retry}"
        )
        print(
            f"    GW+sSOSEX:J={J['ssosex'][ih]:+.9f} dJ={J['ssosex'][ih]-J['gw'][ih]:+.9f} "
            f"dJ_vs_SOX={J['ssosex'][ih]-J['sox'][ih]:+.9f} "
            f"r={err['ssosex'][ih]:.3e} max|sSOSEX|={scale['ssosex'][ih]:.3e} retry={sx_retry}"
        )
        if args.ed:
            print(
                f"    ED:       J={J['ed'][ih]:+.9f}; Jerr GW={J['gw'][ih]-J['ed'][ih]:+.9f}, "
                f"SOX={J['sox'][ih]-J['ed'][ih]:+.9f}, "
                f"sSOSEX={J['ssosex'][ih]-J['ed'][ih]:+.9f}"
            )
            print(
                f"    Gerr:     GW={Gerr['gw'][ih]:.6e}, SOX={Gerr['sox'][ih]:.6e}, "
                f"sSOSEX={Gerr['ssosex'][ih]:.6e}"
            )
            print(
                f"    Gerr low: GW={Gerr_low['gw'][ih]:.6e}, SOX={Gerr_low['sox'][ih]:.6e}, "
                f"sSOSEX={Gerr_low['ssosex'][ih]:.6e}"
            )

        rows.append({
            "h_ref": href, "h_cell": hpc,
            "J_ed": J["ed"][ih], "J_gw": J["gw"][ih],
            "J_sox": J["sox"][ih], "J_ssosex": J["ssosex"][ih],
            "Gerr_gw": Gerr["gw"][ih], "Gerr_sox": Gerr["sox"][ih],
            "Gerr_ssosex": Gerr["ssosex"][ih],
            "Gerr_gw_low": Gerr_low["gw"][ih],
            "Gerr_sox_low": Gerr_low["sox"][ih],
            "Gerr_ssosex_low": Gerr_low["ssosex"][ih],
            "gw_ok": int(ok["gw"][ih]), "sox_ok": int(ok["sox"][ih]),
            "ssosex_ok": int(ok["ssosex"][ih]),
            "gw_error": err["gw"][ih], "sox_error": err["sox"][ih],
            "ssosex_error": err["ssosex"][ih],
            "max_sigma_sox": scale["sox"][ih],
            "max_sigma_ssosex": scale["ssosex"][ih],
        })

    J0 = {}
    for k in ("gw", "sox", "ssosex"):
        J0[k] = _fit(h_ref, J[k], ok[k], args.fit_points)
    print("\n=== small-h fits ===")
    for k, name in (("gw", "GW"), ("sox", "GW+SOX"), ("ssosex", "GW+sSOSEX")):
        j0, slope, nfit = J0[k]
        print(f"{name:10s}: J0={j0:+.9f}, slope={slope:+.9f}, n={nfit}")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"V{args.V:g}_fill{args.filling:g}_{source}_{nk1}x{nk2}_{args.ssosex_mode}_noGamma"
    csv_path = args.out / f"{stem}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    npz_path = args.out / f"{stem}.npz"
    np.savez_compressed(
        npz_path, h_ref=h_ref, h_cell=h_cell,
        J_ed=J["ed"], J_gw=J["gw"], J_sox=J["sox"], J_ssosex=J["ssosex"],
        Gerr_gw=Gerr["gw"], Gerr_sox=Gerr["sox"], Gerr_ssosex=Gerr["ssosex"],
        Gerr_gw_low=Gerr_low["gw"], Gerr_sox_low=Gerr_low["sox"],
        Gerr_ssosex_low=Gerr_low["ssosex"],
        G_ed_local=G_local["ed"], G_gw_local=G_local["gw"],
        G_sox_local=G_local["sox"], G_ssosex_local=G_local["ssosex"],
        omega=grid.omega, low_frequency_indices=low_idx,
        gw_ok=ok["gw"], sox_ok=ok["sox"], ssosex_ok=ok["ssosex"],
        gw_error=err["gw"], sox_error=err["sox"], ssosex_error=err["ssosex"],
        max_sigma_sox=scale["sox"], max_sigma_ssosex=scale["ssosex"],
        J0_gw=J0["gw"][0], J0_sox=J0["sox"][0], J0_ssosex=J0["ssosex"][0],
        slope_gw=J0["gw"][1], slope_sox=J0["sox"][1], slope_ssosex=J0["ssosex"][1],
        ssosex_mode=args.ssosex_mode,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
