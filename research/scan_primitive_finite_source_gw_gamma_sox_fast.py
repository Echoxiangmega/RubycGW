#!/usr/bin/env python3
"""Finite-source comparison: ED / GW / GW-Gamma_P / GW-Gamma_P+SOX.

The first three branches use the existing production/fast solvers.  The fourth
branch adds the bare second-order crossed-exchange self-energy to the same
Gamma_P-screening fixed point.  By default SOX is *not* included in the density
vertex kernel, so the comparison cleanly separates screening-side Gamma_P from
the lowest-order self-energy exchange vertex topology.  Use ``--sox-in-vertex``
to also differentiate Sigma_SOX in the covariant density vertex.
"""
from __future__ import annotations

import argparse
import csv
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
from rubycgw.hedin_gamma_fast import (
    GammaPFeedbackOptions,
    solve_matrix_gw_gamma_feedback,
)
from rubycgw.hedin_gamma_sox_fast import solve_matrix_gw_gamma_sox_feedback
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import canonical_channel_name, primitive_pseudospin_vertex
from rubycgw.sox_covariant import SOXOptions
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw import compute_polarization_matrix
from scan_primitive_finite_source_gw_gamma import (
    _finite_mask,
    _fit,
    _gw_current,
    _gw_retry,
    _hvalues,
    _low_frequency_indices,
    _parse_mesh,
    _relative_frobenius,
    _static_gamma_soft_mode,
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

    p.add_argument("--vertex-max-iter", type=int, default=180)
    p.add_argument("--vertex-tol", type=float, default=2e-8)
    p.add_argument("--gmres-restart", type=int, default=14)
    p.add_argument("--m-max", type=int, default=0)

    p.add_argument("--gamma-max-iter", type=int, default=30)
    p.add_argument("--gamma-tol", type=float, default=2e-6)
    p.add_argument("--gamma-mixing", type=float, default=.10)
    p.add_argument("--gamma-w-mixing", type=float, default=.10)
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fit-points", type=int, default=3)

    p.add_argument("--sox-nquad", type=int, default=128)
    p.add_argument("--sox-in-vertex", action="store_true",
                   help="also include delta Sigma_SOX / delta G in the density vertex")

    p.add_argument("--ed", dest="ed", action="store_true", default=True)
    p.add_argument("--no-ed", dest="ed", action="store_false")
    p.add_argument("--ed-L1", type=int, default=2)
    p.add_argument("--ed-L2", type=int, default=1)
    p.add_argument("--ed-discard-weight-tol", type=float, default=1e-12)
    p.add_argument("--ed-low-nfreq", type=int, default=8)

    p.add_argument("--out", type=Path, default=Path("results/primitive_gw_gamma_sox"))
    return p.parse_args()


def _current_from_state(G, sigma_h, sigma_f, mu, K, h0, grid):
    h_static = h0 + sigma_h[None, None] + sigma_f
    return float(_bilinear_expectation_tail_completed(G, K, grid, h_static, mu).real)


def main():
    args = _args()
    nk1, nk2 = _parse_mesh(args.mesh)
    h_ref = _hvalues(args.h)
    m_max = None if int(args.m_max) < 0 else int(args.m_max)
    source = canonical_channel_name(args.source)
    K = np.asarray(primitive_pseudospin_vertex(source), dtype=complex)
    root = float(np.sqrt(args.reference_ncell))
    h_cell = h_ref / root

    if args.ed:
        ed_ncell = int(args.ed_L1) * int(args.ed_L2)
        if ed_ncell != int(args.reference_ncell):
            raise ValueError("ed-L1*ed-L2 must equal reference-ncell")
        if 6 * ed_ncell > 16:
            raise ValueError("ExactSmallRubyThermal requires at most 16 sites")

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(nk1=nk1, nk2=nk2, nw=args.nw, nOmega=args.nomega, T=args.T)
    h0_base = np.asarray(build_h0(grid.kmesh(), params), dtype=complex)
    Vq = np.asarray(build_interaction(grid.qmesh(), params), dtype=complex)

    gw_opts = GWOptions(
        target_filling=args.filling,
        max_iter=args.gw_max_iter,
        tol=args.gw_tol,
        mixing=args.gw_mixing,
        mixing_method=args.mixing_method,
        pulay_history=args.pulay_history,
        pulay_start=args.pulay_start,
        pulay_regularization=args.pulay_regularization,
        verbose=args.verbose,
        momentum_backend=args.backend,
    )
    vertex_opts = SupercellVertexOptions(
        max_iter=args.vertex_max_iter,
        tol=args.vertex_tol,
        solver="gmres",
        gmres_restart=args.gmres_restart,
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=True,
        verbose=args.verbose,
        momentum_backend=args.backend,
    )
    gamma_opts = GammaPFeedbackOptions(
        max_iter=args.gamma_max_iter,
        tol=args.gamma_tol,
        mixing=args.gamma_mixing,
        w_mixing=args.gamma_w_mixing,
        m_max=m_max,
        allow_unconverged_vertex=args.allow_unconverged,
        verbose=True,
    )
    sox_opts = SOXOptions(n_quad=args.sox_nquad, tail_complete=True)

    n = len(h_ref)
    names = ("ed", "gw", "gamma", "gamma_sox")
    J = {x: np.full(n, np.nan) for x in names}
    Gerr = {x: np.full(n, np.nan) for x in ("gw", "gamma", "gamma_sox")}
    Gerr_low = {x: np.full(n, np.nan) for x in ("gw", "gamma", "gamma_sox")}
    ok_gw = np.zeros(n, dtype=bool)
    ok_gamma = np.zeros(n, dtype=bool)
    ok_gamma_sox = np.zeros(n, dtype=bool)
    err_gamma = np.full(n, np.nan)
    err_gamma_sox = np.full(n, np.nan)
    iter_gamma = np.zeros(n, dtype=int)
    iter_gamma_sox = np.zeros(n, dtype=int)
    sox_max = np.full(n, np.nan)
    relW_gamma = np.full(n, np.nan)
    relW_gamma_sox = np.full(n, np.nan)
    relP_gamma = np.full(n, np.nan)
    relP_gamma_sox = np.full(n, np.nan)
    mu_ed = np.full(n, np.nan)

    G_local = {
        x: np.full((n, grid.nf, 6, 6), np.nan + 0j, dtype=complex)
        for x in names
    }

    print("=== ED / GW / GWGamma_P / GWGamma_P+SOX ===")
    print(f"V={args.V:g}, filling={args.filling:g}, T={args.T:g}, source={source}")
    print(f"mesh={nk1}x{nk2}, h_ref={h_ref.tolist()}, m_max={m_max}")
    print(
        "SOX enters the self-energy fixed point; "
        f"SOX derivative in density vertex={'ON' if args.sox_in_vertex else 'OFF'}"
    )

    last_gw = None
    last_gamma = None
    last_gamma_sox = None
    low_idx = _low_frequency_indices(grid.omega, args.ed_low_nfreq)
    rows = []

    for ih, (href, hpc) in enumerate(zip(h_ref, h_cell)):
        print(f"\n-- h_ref={href:.8g}, h_cell={hpc:.8g} --")
        h0 = h0_base - float(hpc) * K[None, None]
        h0 = .5 * (h0 + np.swapaxes(h0.conj(), -1, -2))

        gw, retry = _gw_retry(h0, Vq, grid, gw_opts, initial=last_gw)
        ok_gw[ih] = bool(gw.converged)
        J["gw"][ih] = root * _gw_current(gw, K, Vq, h0, grid, args.backend)
        G_local["gw"][ih] = primitive_local_green(gw.G)
        if gw.converged:
            last_gw = gw
        if not gw.converged and not args.allow_unconverged:
            raise RuntimeError(f"GW did not converge at h_ref={href:g}")

        if args.ed:
            print("      solving exact ED Green function ...")
            ed = solve_exact_finite_source_local_green(
                L1=args.ed_L1,
                L2=args.ed_L2,
                params=params,
                V=args.V,
                source_channel=source,
                h_ref=href,
                filling_per_cell=args.filling,
                T=args.T,
                omega=grid.omega,
                discard_weight_tol=args.ed_discard_weight_tol,
            )
            J["ed"][ih] = ed.J_ref
            mu_ed[ih] = ed.mu
            G_local["ed"][ih] = ed.G_local
            Gerr["gw"][ih] = relative_green_error(G_local["gw"][ih], G_local["ed"][ih])
            Gerr_low["gw"][ih] = relative_green_error(
                G_local["gw"][ih, low_idx], G_local["ed"][ih, low_idx]
            )

        print("      iterating GWGamma_P ...")
        gamma = solve_matrix_gw_gamma_feedback(
            h0, Vq, grid,
            gw_opts=gw_opts,
            vertex_opts=vertex_opts,
            feedback_opts=gamma_opts,
            background=gw,
            initial_state=last_gamma,
        )
        ok_gamma[ih] = bool(gamma.converged)
        err_gamma[ih] = gamma.final_error
        iter_gamma[ih] = gamma.iterations
        J["gamma"][ih] = root * _current_from_state(
            gamma.G, gamma.Sigma_H, gamma.Sigma_F, gamma.mu, K, h0, grid
        )
        G_local["gamma"][ih] = primitive_local_green(gamma.G)
        if gamma.converged:
            last_gamma = gamma
        elif not args.allow_unconverged:
            raise RuntimeError(f"GWGamma_P failed at h_ref={href:g}")

        print("      iterating GWGamma_P+SOX ...")
        # At the first h, start directly from the converged Gamma_P state so the
        # new outer loop only has to turn on SOX.  Subsequent h values use the
        # previous combined fixed point for branch continuation.
        combined_seed = last_gamma_sox if last_gamma_sox is not None else gamma
        gamma_sox = solve_matrix_gw_gamma_sox_feedback(
            h0, Vq, grid,
            gw_opts=gw_opts,
            vertex_opts=vertex_opts,
            feedback_opts=gamma_opts,
            sox_opts=sox_opts,
            include_sox_vertex=args.sox_in_vertex,
            background=gw,
            initial_state=combined_seed,
        )
        ok_gamma_sox[ih] = bool(gamma_sox.converged)
        err_gamma_sox[ih] = gamma_sox.final_error
        iter_gamma_sox[ih] = gamma_sox.iterations
        sox_max[ih] = float(np.max(np.abs(gamma_sox.Sigma_SOX)))
        J["gamma_sox"][ih] = root * _current_from_state(
            gamma_sox.G, gamma_sox.Sigma_H, gamma_sox.Sigma_F,
            gamma_sox.mu, K, h0, grid
        )
        G_local["gamma_sox"][ih] = primitive_local_green(gamma_sox.G)
        if gamma_sox.converged:
            last_gamma_sox = gamma_sox
        elif not args.allow_unconverged:
            raise RuntimeError(f"GWGamma_P+SOX failed at h_ref={href:g}")

        for label, state, destW, destP in (
            ("gamma", gamma, relW_gamma, relP_gamma),
            ("gamma_sox", gamma_sox, relW_gamma_sox, relP_gamma_sox),
        ):
            mask = _finite_mask(state.P_gamma)
            destW[ih] = _relative_frobenius(state.W, gw.W, mask)
            pb = compute_polarization_matrix(state.G, grid, backend=args.backend)
            destP[ih] = _relative_frobenius(state.P_gamma, pb, mask)

        if args.ed:
            for key in ("gamma", "gamma_sox"):
                Gerr[key][ih] = relative_green_error(G_local[key][ih], G_local["ed"][ih])
                Gerr_low[key][ih] = relative_green_error(
                    G_local[key][ih, low_idx], G_local["ed"][ih, low_idx]
                )

        print(f"    GW:              J={J['gw'][ih]:+.9f}  retry={retry}")
        print(
            f"    GWGamma_P:        J={J['gamma'][ih]:+.9f}  "
            f"dJ={J['gamma'][ih]-J['gw'][ih]:+.9f}  r={err_gamma[ih]:.3e}"
        )
        print(
            f"    GWGamma_P+SOX:    J={J['gamma_sox'][ih]:+.9f}  "
            f"dJ_vs_Gamma={J['gamma_sox'][ih]-J['gamma'][ih]:+.9f}  "
            f"dJ_vs_GW={J['gamma_sox'][ih]-J['gw'][ih]:+.9f}  "
            f"r={err_gamma_sox[ih]:.3e}  max|SOX|={sox_max[ih]:.3e}"
        )
        if args.ed:
            rg = Gerr["gamma"][ih] / Gerr["gw"][ih] if Gerr["gw"][ih] > 0 else np.nan
            rxs = Gerr["gamma_sox"][ih] / Gerr["gw"][ih] if Gerr["gw"][ih] > 0 else np.nan
            print(
                f"    ED:              J={J['ed'][ih]:+.9f}; "
                f"Jerr GW={J['gw'][ih]-J['ed'][ih]:+.9f}, "
                f"Gamma={J['gamma'][ih]-J['ed'][ih]:+.9f}, "
                f"Gamma+SOX={J['gamma_sox'][ih]-J['ed'][ih]:+.9f}"
            )
            print(
                f"    Gerr local:      GW={Gerr['gw'][ih]:.6e}, "
                f"Gamma={Gerr['gamma'][ih]:.6e} ({rg:.4f}x), "
                f"Gamma+SOX={Gerr['gamma_sox'][ih]:.6e} ({rxs:.4f}x)"
            )
            print(
                f"    Gerr low:        GW={Gerr_low['gw'][ih]:.6e}, "
                f"Gamma={Gerr_low['gamma'][ih]:.6e}, "
                f"Gamma+SOX={Gerr_low['gamma_sox'][ih]:.6e}"
            )
        print(
            f"    screening relW:  Gamma={relW_gamma[ih]:.4e}, "
            f"Gamma+SOX={relW_gamma_sox[ih]:.4e}; "
            f"relP={relP_gamma[ih]:.4e}/{relP_gamma_sox[ih]:.4e}"
        )

        rows.append({
            "h_ref": href,
            "h_cell": hpc,
            "J_ed": J["ed"][ih],
            "J_gw": J["gw"][ih],
            "J_gamma": J["gamma"][ih],
            "J_gamma_sox": J["gamma_sox"][ih],
            "Gerr_gw": Gerr["gw"][ih],
            "Gerr_gamma": Gerr["gamma"][ih],
            "Gerr_gamma_sox": Gerr["gamma_sox"][ih],
            "Gerr_gw_low": Gerr_low["gw"][ih],
            "Gerr_gamma_low": Gerr_low["gamma"][ih],
            "Gerr_gamma_sox_low": Gerr_low["gamma_sox"][ih],
            "gamma_ok": int(ok_gamma[ih]),
            "gamma_sox_ok": int(ok_gamma_sox[ih]),
            "gamma_error": err_gamma[ih],
            "gamma_sox_error": err_gamma_sox[ih],
            "gamma_iter": iter_gamma[ih],
            "gamma_sox_iter": iter_gamma_sox[ih],
            "max_sigma_sox": sox_max[ih],
            "relW_gamma": relW_gamma[ih],
            "relW_gamma_sox": relW_gamma_sox[ih],
            "relP_gamma": relP_gamma[ih],
            "relP_gamma_sox": relP_gamma_sox[ih],
        })

    J0g, bg, ng = _fit(h_ref, J["gw"], ok_gw, args.fit_points)
    J0x, bx, nx = _fit(h_ref, J["gamma"], ok_gamma, args.fit_points)
    J0s, bs, ns = _fit(h_ref, J["gamma_sox"], ok_gamma_sox, args.fit_points)
    print("\n=== small-h fits ===")
    print(f"GW:             J0={J0g:+.9f}, slope={bg:+.9f}, n={ng}")
    print(f"GWGamma_P:      J0={J0x:+.9f}, slope={bx:+.9f}, n={nx}")
    print(f"GWGamma_P+SOX:  J0={J0s:+.9f}, slope={bs:+.9f}, n={ns}")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = (
        f"V{args.V:g}_fill{args.filling:g}_{source}_{nk1}x{nk2}_"
        f"m{('all' if m_max is None else m_max)}_gamma_sox"
    )
    csv_path = args.out / f"{stem}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    npz_path = args.out / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        h_ref=h_ref,
        h_cell=h_cell,
        J_ed=J["ed"],
        J_gw=J["gw"],
        J_gamma=J["gamma"],
        J_gamma_sox=J["gamma_sox"],
        Gerr_gw=Gerr["gw"],
        Gerr_gamma=Gerr["gamma"],
        Gerr_gamma_sox=Gerr["gamma_sox"],
        Gerr_gw_low=Gerr_low["gw"],
        Gerr_gamma_low=Gerr_low["gamma"],
        Gerr_gamma_sox_low=Gerr_low["gamma_sox"],
        G_ed_local=G_local["ed"],
        G_gw_local=G_local["gw"],
        G_gamma_local=G_local["gamma"],
        G_gamma_sox_local=G_local["gamma_sox"],
        omega=grid.omega,
        low_frequency_indices=low_idx,
        gamma_ok=ok_gamma,
        gamma_sox_ok=ok_gamma_sox,
        gamma_error=err_gamma,
        gamma_sox_error=err_gamma_sox,
        gamma_iter=iter_gamma,
        gamma_sox_iter=iter_gamma_sox,
        max_sigma_sox=sox_max,
        relW_gamma=relW_gamma,
        relW_gamma_sox=relW_gamma_sox,
        relP_gamma=relP_gamma,
        relP_gamma_sox=relP_gamma_sox,
        J0_gw=J0g,
        J0_gamma=J0x,
        J0_gamma_sox=J0s,
        slope_gw=bg,
        slope_gamma=bx,
        slope_gamma_sox=bs,
        sox_in_vertex=bool(args.sox_in_vertex),
        sox_nquad=int(args.sox_nquad),
    )
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
