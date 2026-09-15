#!/usr/bin/env python3
"""Finite-source ED/GW/Gamma_P/bare-SOX/screened-SOSEX comparison.

Branches:

    ED
    GW
    GWGamma_P
    GWGamma_P + bare SOX
    GWGamma_P + static screened SOSEX

The screened branch uses W0=W_Gamma(q,Omega=0) inside the crossed-exchange
skeleton.  Default ``oneW-sym`` means

    1/2 [ S[V,W0] + S[W0,V] ],

while ``twoW`` evaluates S[W0,W0].  The density vertex remains the GW
H/F/MT/AL kernel in the screened-SOSEX branch; this scan therefore isolates the
self-energy effect of screening the crossed exchange topology.
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
from rubycgw.hedin_gamma_ssosex_fast import solve_matrix_gw_gamma_ssosex_feedback
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import canonical_channel_name, primitive_pseudospin_vertex
from rubycgw.sox_covariant import SOXOptions
from rubycgw.ssosex_static import ScreenedSOSEXOptions
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
    p.add_argument("--skip-bare-sox", action="store_true")
    p.add_argument(
        "--ssosex-mode", choices=["oneW-sym", "twoW"], default="oneW-sym",
        help="one screened line symmetrized (default) or two statically screened lines",
    )
    p.add_argument("--ssosex-nquad", type=int, default=128)

    p.add_argument("--ed", dest="ed", action="store_true", default=True)
    p.add_argument("--no-ed", dest="ed", action="store_false")
    p.add_argument("--ed-L1", type=int, default=2)
    p.add_argument("--ed-L2", type=int, default=1)
    p.add_argument("--ed-discard-weight-tol", type=float, default=1e-12)
    p.add_argument("--ed-low-nfreq", type=int, default=8)

    p.add_argument("--out", type=Path, default=Path("results/primitive_gw_gamma_ssosex"))
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
    ssosex_opts = ScreenedSOSEXOptions(
        n_quad=args.ssosex_nquad,
        tail_complete=True,
        mode=args.ssosex_mode,
    )

    n = len(h_ref)
    method_names = ["ed", "gw", "gamma", "gamma_ssosex"]
    if not args.skip_bare_sox:
        method_names.insert(3, "gamma_sox")
    J = {x: np.full(n, np.nan) for x in method_names}
    approx = [x for x in method_names if x not in {"ed"}]
    Gerr = {x: np.full(n, np.nan) for x in approx}
    Gerr_low = {x: np.full(n, np.nan) for x in approx}
    ok = {x: np.zeros(n, dtype=bool) for x in approx}
    err = {x: np.full(n, np.nan) for x in approx if x != "gw"}
    nit = {x: np.zeros(n, dtype=int) for x in approx if x != "gw"}
    max_x = {"gamma_sox": np.full(n, np.nan), "gamma_ssosex": np.full(n, np.nan)}
    relW = {x: np.full(n, np.nan) for x in ("gamma", "gamma_sox", "gamma_ssosex")}
    relP = {x: np.full(n, np.nan) for x in ("gamma", "gamma_sox", "gamma_ssosex")}
    mu_ed = np.full(n, np.nan)
    G_local = {
        x: np.full((n, grid.nf, 6, 6), np.nan + 0j, dtype=complex)
        for x in method_names
    }

    print("=== ED / GW / Gamma_P / bare SOX / screened SOSEX ===")
    print(f"V={args.V:g}, filling={args.filling:g}, T={args.T:g}, source={source}")
    print(f"mesh={nk1}x{nk2}, h_ref={h_ref.tolist()}, m_max={m_max}")
    print(
        f"screened SOSEX: mode={args.ssosex_mode}, W0=W_Gamma(q,Omega=0), "
        "density vertex remains GW H/F/MT/AL"
    )

    last_gw = last_gamma = last_sox = last_ssosex = None
    low_idx = _low_frequency_indices(grid.omega, args.ed_low_nfreq)
    rows = []

    for ih, (href, hpc) in enumerate(zip(h_ref, h_cell)):
        print(f"\n-- h_ref={href:.8g}, h_cell={hpc:.8g} --")
        h0 = h0_base - float(hpc) * K[None, None]
        h0 = .5 * (h0 + np.swapaxes(h0.conj(), -1, -2))

        gw, retry = _gw_retry(h0, Vq, grid, gw_opts, initial=last_gw)
        ok["gw"][ih] = bool(gw.converged)
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

        print("      iterating GWGamma_P ...")
        gamma = solve_matrix_gw_gamma_feedback(
            h0, Vq, grid,
            gw_opts=gw_opts,
            vertex_opts=vertex_opts,
            feedback_opts=gamma_opts,
            background=gw,
            initial_state=last_gamma,
        )
        ok["gamma"][ih] = bool(gamma.converged)
        err["gamma"][ih] = gamma.final_error
        nit["gamma"][ih] = gamma.iterations
        J["gamma"][ih] = root * _current_from_state(
            gamma.G, gamma.Sigma_H, gamma.Sigma_F, gamma.mu, K, h0, grid
        )
        G_local["gamma"][ih] = primitive_local_green(gamma.G)
        if gamma.converged:
            last_gamma = gamma
        elif not args.allow_unconverged:
            raise RuntimeError(f"GWGamma_P failed at h_ref={href:g}")

        gamma_sox = None
        if not args.skip_bare_sox:
            print("      iterating GWGamma_P+bare SOX ...")
            seed = last_sox if last_sox is not None else gamma
            gamma_sox = solve_matrix_gw_gamma_sox_feedback(
                h0, Vq, grid,
                gw_opts=gw_opts,
                vertex_opts=vertex_opts,
                feedback_opts=gamma_opts,
                sox_opts=sox_opts,
                include_sox_vertex=False,
                background=gw,
                initial_state=seed,
            )
            ok["gamma_sox"][ih] = bool(gamma_sox.converged)
            err["gamma_sox"][ih] = gamma_sox.final_error
            nit["gamma_sox"][ih] = gamma_sox.iterations
            max_x["gamma_sox"][ih] = float(np.max(np.abs(gamma_sox.Sigma_SOX)))
            J["gamma_sox"][ih] = root * _current_from_state(
                gamma_sox.G, gamma_sox.Sigma_H, gamma_sox.Sigma_F,
                gamma_sox.mu, K, h0, grid
            )
            G_local["gamma_sox"][ih] = primitive_local_green(gamma_sox.G)
            if gamma_sox.converged:
                last_sox = gamma_sox
            elif not args.allow_unconverged:
                raise RuntimeError(f"GWGamma_P+SOX failed at h_ref={href:g}")

        print("      iterating GWGamma_P+screened SOSEX ...")
        # If the bare-SOX branch was evaluated, use it as the closest seed so
        # this loop mainly turns on screening of the crossed interaction line.
        seed = last_ssosex
        if seed is None:
            seed = gamma_sox if gamma_sox is not None else gamma
        gamma_ssosex = solve_matrix_gw_gamma_ssosex_feedback(
            h0, Vq, grid,
            gw_opts=gw_opts,
            vertex_opts=vertex_opts,
            feedback_opts=gamma_opts,
            ssosex_opts=ssosex_opts,
            background=gw,
            initial_state=seed,
        )
        ok["gamma_ssosex"][ih] = bool(gamma_ssosex.converged)
        err["gamma_ssosex"][ih] = gamma_ssosex.final_error
        nit["gamma_ssosex"][ih] = gamma_ssosex.iterations
        max_x["gamma_ssosex"][ih] = float(np.max(np.abs(gamma_ssosex.Sigma_SSOSEX)))
        J["gamma_ssosex"][ih] = root * _current_from_state(
            gamma_ssosex.G, gamma_ssosex.Sigma_H, gamma_ssosex.Sigma_F,
            gamma_ssosex.mu, K, h0, grid
        )
        G_local["gamma_ssosex"][ih] = primitive_local_green(gamma_ssosex.G)
        if gamma_ssosex.converged:
            last_ssosex = gamma_ssosex
        elif not args.allow_unconverged:
            raise RuntimeError(f"GWGamma_P+sSOSEX failed at h_ref={href:g}")

        states = {"gamma": gamma, "gamma_ssosex": gamma_ssosex}
        if gamma_sox is not None:
            states["gamma_sox"] = gamma_sox
        for label, state in states.items():
            mask = _finite_mask(state.P_gamma)
            relW[label][ih] = _relative_frobenius(state.W, gw.W, mask)
            pb = compute_polarization_matrix(state.G, grid, backend=args.backend)
            relP[label][ih] = _relative_frobenius(state.P_gamma, pb, mask)

        if args.ed:
            for key in approx:
                Gerr[key][ih] = relative_green_error(G_local[key][ih], G_local["ed"][ih])
                Gerr_low[key][ih] = relative_green_error(
                    G_local[key][ih, low_idx], G_local["ed"][ih, low_idx]
                )

        print(f"    GW:                  J={J['gw'][ih]:+.9f}  retry={retry}")
        print(
            f"    GWGamma_P:            J={J['gamma'][ih]:+.9f}  "
            f"dJ_GW={J['gamma'][ih]-J['gw'][ih]:+.9f}  r={err['gamma'][ih]:.3e}"
        )
        if gamma_sox is not None:
            print(
                f"    GWGamma_P+SOX:        J={J['gamma_sox'][ih]:+.9f}  "
                f"dJ_Gamma={J['gamma_sox'][ih]-J['gamma'][ih]:+.9f}  "
                f"r={err['gamma_sox'][ih]:.3e}  max|SOX|={max_x['gamma_sox'][ih]:.3e}"
            )
        refj = J["gamma_sox"][ih] if gamma_sox is not None else J["gamma"][ih]
        print(
            f"    GWGamma_P+sSOSEX:     J={J['gamma_ssosex'][ih]:+.9f}  "
            f"dJ_vs_prev={J['gamma_ssosex'][ih]-refj:+.9f}  "
            f"dJ_Gamma={J['gamma_ssosex'][ih]-J['gamma'][ih]:+.9f}  "
            f"r={err['gamma_ssosex'][ih]:.3e}  max|sSOSEX|={max_x['gamma_ssosex'][ih]:.3e}"
        )
        if args.ed:
            parts = [f"GW={Gerr['gw'][ih]:.6e}", f"Gamma={Gerr['gamma'][ih]:.6e}"]
            if gamma_sox is not None:
                parts.append(f"Gamma+SOX={Gerr['gamma_sox'][ih]:.6e}")
            parts.append(f"Gamma+sSOSEX={Gerr['gamma_ssosex'][ih]:.6e}")
            print(f"    ED:                  J={J['ed'][ih]:+.9f}")
            print("    Gerr local:          " + ", ".join(parts))
            parts_low = [f"GW={Gerr_low['gw'][ih]:.6e}", f"Gamma={Gerr_low['gamma'][ih]:.6e}"]
            if gamma_sox is not None:
                parts_low.append(f"Gamma+SOX={Gerr_low['gamma_sox'][ih]:.6e}")
            parts_low.append(f"Gamma+sSOSEX={Gerr_low['gamma_ssosex'][ih]:.6e}")
            print("    Gerr low:            " + ", ".join(parts_low))
        print(
            f"    screening relW/P:    Gamma={relW['gamma'][ih]:.4e}/{relP['gamma'][ih]:.4e}, "
            f"sSOSEX={relW['gamma_ssosex'][ih]:.4e}/{relP['gamma_ssosex'][ih]:.4e}"
        )

        row = {
            "h_ref": href,
            "h_cell": hpc,
            "J_ed": J["ed"][ih],
            "J_gw": J["gw"][ih],
            "J_gamma": J["gamma"][ih],
            "J_gamma_ssosex": J["gamma_ssosex"][ih],
            "Gerr_gw": Gerr["gw"][ih],
            "Gerr_gamma": Gerr["gamma"][ih],
            "Gerr_gamma_ssosex": Gerr["gamma_ssosex"][ih],
            "Gerr_gw_low": Gerr_low["gw"][ih],
            "Gerr_gamma_low": Gerr_low["gamma"][ih],
            "Gerr_gamma_ssosex_low": Gerr_low["gamma_ssosex"][ih],
            "gamma_error": err["gamma"][ih],
            "gamma_ssosex_error": err["gamma_ssosex"][ih],
            "gamma_iter": nit["gamma"][ih],
            "gamma_ssosex_iter": nit["gamma_ssosex"][ih],
            "max_sigma_ssosex": max_x["gamma_ssosex"][ih],
            "relW_gamma": relW["gamma"][ih],
            "relW_gamma_ssosex": relW["gamma_ssosex"][ih],
            "relP_gamma": relP["gamma"][ih],
            "relP_gamma_ssosex": relP["gamma_ssosex"][ih],
        }
        if gamma_sox is not None:
            row.update({
                "J_gamma_sox": J["gamma_sox"][ih],
                "Gerr_gamma_sox": Gerr["gamma_sox"][ih],
                "Gerr_gamma_sox_low": Gerr_low["gamma_sox"][ih],
                "gamma_sox_error": err["gamma_sox"][ih],
                "gamma_sox_iter": nit["gamma_sox"][ih],
                "max_sigma_sox": max_x["gamma_sox"][ih],
                "relW_gamma_sox": relW["gamma_sox"][ih],
                "relP_gamma_sox": relP["gamma_sox"][ih],
            })
        rows.append(row)

    J0 = {}
    for key in ("gw", "gamma", "gamma_ssosex"):
        J0[key] = _fit(h_ref, J[key], ok[key], args.fit_points)
    if not args.skip_bare_sox:
        J0["gamma_sox"] = _fit(h_ref, J["gamma_sox"], ok["gamma_sox"], args.fit_points)

    print("\n=== small-h fits ===")
    for key, label in (
        ("gw", "GW"),
        ("gamma", "GWGamma_P"),
        ("gamma_sox", "GWGamma_P+SOX"),
        ("gamma_ssosex", "GWGamma_P+sSOSEX"),
    ):
        if key not in J0:
            continue
        intercept, slope, count = J0[key]
        print(f"{label:20s} J0={intercept:+.9f}, slope={slope:+.9f}, n={count}")

    args.out.mkdir(parents=True, exist_ok=True)
    stem = (
        f"V{args.V:g}_fill{args.filling:g}_{source}_{nk1}x{nk2}_"
        f"m{('all' if m_max is None else m_max)}_ssosex_{args.ssosex_mode.lower()}"
    )
    csv_path = args.out / f"{stem}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = dict(
        h_ref=h_ref,
        h_cell=h_cell,
        omega=grid.omega,
        low_frequency_indices=low_idx,
        ssosex_mode=str(args.ssosex_mode),
        ssosex_nquad=int(args.ssosex_nquad),
        J_ed=J["ed"],
        J_gw=J["gw"],
        J_gamma=J["gamma"],
        J_gamma_ssosex=J["gamma_ssosex"],
        Gerr_gw=Gerr["gw"],
        Gerr_gamma=Gerr["gamma"],
        Gerr_gamma_ssosex=Gerr["gamma_ssosex"],
        Gerr_gw_low=Gerr_low["gw"],
        Gerr_gamma_low=Gerr_low["gamma"],
        Gerr_gamma_ssosex_low=Gerr_low["gamma_ssosex"],
        G_ed_local=G_local["ed"],
        G_gw_local=G_local["gw"],
        G_gamma_local=G_local["gamma"],
        G_gamma_ssosex_local=G_local["gamma_ssosex"],
        gamma_error=err["gamma"],
        gamma_ssosex_error=err["gamma_ssosex"],
        gamma_iter=nit["gamma"],
        gamma_ssosex_iter=nit["gamma_ssosex"],
        max_sigma_ssosex=max_x["gamma_ssosex"],
        relW_gamma=relW["gamma"],
        relW_gamma_ssosex=relW["gamma_ssosex"],
        relP_gamma=relP["gamma"],
        relP_gamma_ssosex=relP["gamma_ssosex"],
        J0_gw=J0["gw"][0],
        J0_gamma=J0["gamma"][0],
        J0_gamma_ssosex=J0["gamma_ssosex"][0],
    )
    if not args.skip_bare_sox:
        payload.update(
            J_gamma_sox=J["gamma_sox"],
            Gerr_gamma_sox=Gerr["gamma_sox"],
            Gerr_gamma_sox_low=Gerr_low["gamma_sox"],
            G_gamma_sox_local=G_local["gamma_sox"],
            gamma_sox_error=err["gamma_sox"],
            gamma_sox_iter=nit["gamma_sox"],
            max_sigma_sox=max_x["gamma_sox"],
            relW_gamma_sox=relW["gamma_sox"],
            relP_gamma_sox=relP["gamma_sox"],
            J0_gamma_sox=J0["gamma_sox"][0],
        )
    npz_path = args.out / f"{stem}.npz"
    np.savez_compressed(npz_path, **payload)
    print(f"wrote {csv_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
