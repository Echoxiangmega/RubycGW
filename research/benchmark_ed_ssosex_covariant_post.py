#!/usr/bin/env python3
"""Same-torus ED benchmark for static covariant GW+sSOSEX and one-shot G_post.

This benchmark answers two questions on an exactly geometry-matched finite Ruby
torus.

1. If the *complete self-consistent* GW+sSOSEX closure is differentiated, how
   close are the static z_same/z_opposite susceptibilities to exact ED?
2. If the resulting full static orbital density response is inserted into

       W_post(0) = V - V chi_nn,cov(0) V,

   and both the GW correlation self-energy and the screened-SOSEX correction
   are recomputed once with W_post, how much does the Matsubara Green function
   move toward ED?

The covariant derivative is evaluated by a central finite difference of the
fully converged GW+sSOSEX fixed point at fixed chemical potential.  Therefore
it automatically contains dSigma_H/dG, dSigma_F/dG, the GW MT/AL paths through
dW/dG, all three explicit G derivatives of sSOSEX, and the screened-line
(dSigma_sSOSEX/dW)(dW/dG) contribution.  It is a numerical tangent of the same
approximate functional, not a frozen-W SOSEX vertex.

For the first diagnostic G_post only the static bosonic sector is replaced by
the covariant W_post; all m!=0 sectors retain the self-consistent ordinary-GW
screening of the background.  This matches the inexpensive m_max=0 post-GW
diagnostic used elsewhere in the project.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.gw_ssosex import solve_matrix_gw_ssosex
from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.ssosex_covariant_fd import (
    build_static_post_w_same_torus,
    finite_difference_density_response_same_torus,
    finite_difference_static_observable,
    one_shot_post_ssosex_same_torus,
    onebody_expectation_tail_completed,
)
from rubycgw.ssosex_static import ScreenedSOSEXOptions


CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--filling", type=float, default=2.0, help="particles per primitive cell")
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--low-count", type=int, default=8)

    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--tol", type=float, default=2e-9)
    p.add_argument("--mixing", type=float, default=0.08)
    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--pulay-history", type=int, default=8)
    p.add_argument("--pulay-start", type=int, default=4)
    p.add_argument("--pulay-regularization", type=float, default=1e-9)
    p.add_argument("--backend", choices=["fft", "direct"], default="direct")

    p.add_argument("--ssosex-mode", choices=["oneW-sym", "twoW"], default="oneW-sym")
    p.add_argument("--ssosex-nquad", type=int, default=128)
    p.add_argument("--ssosex-interaction-tol", type=float, default=1e-13)
    p.add_argument("--tail-edge-points", type=int, default=2)

    p.add_argument(
        "--fd-eps", type=float, default=1e-4,
        help="central-difference source amplitude for the static covariant tangent",
    )
    p.add_argument(
        "--fd-tol", type=float, default=5e-10,
        help="self-consistency tolerance for +/- finite-difference solves",
    )
    p.add_argument("--fd-max-iter", type=int, default=2400)
    p.add_argument("--fd-mixing", type=float, default=0.06)
    p.add_argument(
        "--skip-density-post", action="store_true",
        help="only compute current susceptibilities; skip 12 density derivatives and G_post",
    )
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--out", type=Path,
        default=Path("results/ed_ssosex_covariant_post"),
    )
    return p.parse_args()


def _relerr(a, b) -> float:
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    den = max(float(np.linalg.norm(bb.ravel())), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def _green_errors(G, Ged, grid, low_count):
    arr = np.asarray(G, dtype=complex)
    if arr.ndim == 5:
        arr = arr[:, 0, 0]
    exact = np.asarray(Ged, dtype=complex)
    full = _relerr(arr, exact)
    idx = np.argsort(np.abs(np.asarray(grid.omega)))[: min(int(low_count), grid.nf)]
    low = _relerr(arr[idx], exact[idx])
    return full, low


def _current_expectation(result, K, h0, grid):
    return float(onebody_expectation_tail_completed(result, K, h0, grid).real)


def main():
    args = _args()
    if args.L1 * args.L2 != 2:
        print(
            "NOTE: the code is generic for <=16 exact sites, but the intended "
            "production benchmark is the 2x1 (12-site) torus."
        )
    if 6 * args.L1 * args.L2 > 16:
        raise ValueError("ExactSmallRubyThermal supports at most 16 sites")
    if args.fd_eps <= 0 or args.T <= 0:
        raise ValueError("need fd_eps>0 and T>0")

    ncell = int(args.L1) * int(args.L2)
    target = float(args.filling) * ncell
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    exact = ExactSmallRubyThermal(args.L1, args.L2, params)
    exact.diagonalize(float(args.V))
    mu_ed = exact.solve_mu(target, args.T)
    norb = int(exact.n_sites)

    # Same-torus representation: the full finite torus is one orbital cell.
    # All primitive finite-q structure is carried by the 12x12 orbital matrix.
    grid = MatsubaraGrid(
        nk1=1, nk2=1, nw=args.nw, nOmega=args.nomega, T=args.T
    )
    h0 = np.asarray(exact.h0, dtype=complex)[None, None]
    Vq = (float(args.V) * np.asarray(exact.Vunit, dtype=complex))[None, None]

    gw_opts = GWOptions(
        target_filling=target,
        max_iter=args.max_iter,
        tol=args.tol,
        mixing=args.mixing,
        mixing_method=args.mixing_method,
        pulay_history=args.pulay_history,
        pulay_start=args.pulay_start,
        pulay_regularization=args.pulay_regularization,
        verbose=args.verbose,
        momentum_backend=args.backend,
    )
    sx_opts = ScreenedSOSEXOptions(
        n_quad=args.ssosex_nquad,
        interaction_tol=args.ssosex_interaction_tol,
        tail_complete=True,
        tail_edge_points=args.tail_edge_points,
        mode=args.ssosex_mode,
    )

    operators = np.stack([
        np.asarray(exact.pseudospin_operator(ch, (0.0, 0.0)), dtype=complex)
        for ch in CHANNELS
    ])
    chi_ed, _ = exact.static_susceptibility_matrix(operators, mu_ed, args.T)

    print("=== covariant screened-SOSEX same-torus benchmark ===")
    print(
        f"torus={args.L1}x{args.L2} ({norb} sites), V={args.V:g}, "
        f"filling={args.filling:g}, T={args.T:g}"
    )
    print(
        f"sSOSEX={args.ssosex_mode}, fd_eps={args.fd_eps:g}; "
        "response derivative is at fixed mu"
    )
    print("solving symmetric self-consistent GW+sSOSEX background ...")
    bg = solve_matrix_gw_ssosex(h0, Vq, grid, opts=gw_opts, ssosex_opts=sx_opts)
    if not bg.converged and not args.allow_unconverged:
        raise RuntimeError(
            f"GW+sSOSEX background did not converge: residual={bg.final_error:.3e}"
        )
    print(
        f"  background: {'OK' if bg.converged else 'FAIL'} iter={bg.iterations} "
        f"r={bg.final_error:.3e} mu={bg.mu:+.10f} "
        f"max|sSOSEX|={np.max(np.abs(bg.Sigma_sSOSEX)):.3e}"
    )

    J0 = np.asarray([
        _current_expectation(bg, K, h0, grid) for K in operators
    ])
    print(
        "  zero-source currents: "
        + " ".join(f"{ch}={j:+.3e}" for ch, j in zip(CHANNELS, J0))
    )

    chi_fd = np.full((len(CHANNELS), len(CHANNELS)), np.nan + 0j)
    fd_conv = np.zeros(len(CHANNELS), dtype=bool)
    fd_res_plus = np.full(len(CHANNELS), np.nan)
    fd_res_minus = np.full(len(CHANNELS), np.nan)
    print("\nstatic current covariant response from full fixed-point derivative:")
    for b, (ch, Ksrc) in enumerate(zip(CHANNELS, operators)):
        # Solve once per source and measure both current observables from the same
        # +/- pair.  finite_difference_static_observable returns the source-source
        # element; cross elements are obtained from the stored +/- states below.
        out = finite_difference_static_observable(
            bg,
            h0,
            Vq,
            Ksrc,
            Ksrc,
            grid,
            eps=args.fd_eps,
            gw_opts=gw_opts,
            ssosex_opts=sx_opts,
            tol=args.fd_tol,
            max_iter=args.fd_max_iter,
            mixing=args.fd_mixing,
        )
        fd_conv[b] = bool(out.plus.converged and out.minus.converged)
        fd_res_plus[b] = float(out.plus.final_error)
        fd_res_minus[b] = float(out.minus.final_error)
        if not fd_conv[b] and not args.allow_unconverged:
            raise RuntimeError(
                f"current FD source {ch} failed: "
                f"r+={fd_res_plus[b]:.3e}, r-={fd_res_minus[b]:.3e}"
            )
        hp = h0 - float(args.fd_eps) * Ksrc[None, None]
        hm = h0 + float(args.fd_eps) * Ksrc[None, None]
        for a, Kobs in enumerate(operators):
            vp = onebody_expectation_tail_completed(out.plus, Kobs, hp, grid)
            vm = onebody_expectation_tail_completed(out.minus, Kobs, hm, grid)
            chi_fd[a, b] = (vp - vm) / (2.0 * float(args.fd_eps))

        print(
            f"  source={ch:10s}: r+={fd_res_plus[b]:.2e} r-={fd_res_minus[b]:.2e} "
            f"chi_diag={chi_fd[b,b].real:+.8f} "
            f"ED={chi_ed[b,b]:+.8f} "
            f"relerr={(chi_fd[b,b].real-chi_ed[b,b])/max(abs(chi_ed[b,b]),1e-300):+.3%}"
        )

    chi_fd_sym = 0.5 * (chi_fd + chi_fd.conj().T)
    current_matrix_relerr = _relerr(chi_fd_sym, chi_ed)
    print(f"  current chi matrix relative Frobenius error = {current_matrix_relerr:.6e}")

    print("\nexact ED Matsubara Green function ...")
    G_ed, _ = exact.green_iomega(1j * np.asarray(grid.omega), mu_ed, args.T)
    gerr_bg, gerr_bg_low = _green_errors(bg.G, G_ed, grid, args.low_count)
    print(
        f"  background G: Gerr={gerr_bg:.6e}, Gerr_low={gerr_bg_low:.6e}"
    )

    # Allocate post outputs even when skipped so NPZ schema is stable.
    chi_nn_fd_raw = np.full((norb, norb), np.nan + 0j)
    chi_nn_fd = np.full((norb, norb), np.nan + 0j)
    chi_nn_ed = np.full((norb, norb), np.nan)
    density_chi_relerr = np.nan
    density_reciprocity_error = np.nan
    density_fd_converged = np.zeros(norb, dtype=bool)
    density_fd_res_plus = np.full(norb, np.nan)
    density_fd_res_minus = np.full(norb, np.nan)
    W_post = np.full_like(bg.W, np.nan + 0j)
    mu_post = np.nan
    G_post = np.full_like(bg.G, np.nan + 0j)
    gerr_post = np.nan
    gerr_post_low = np.nan
    max_delta_W0 = np.nan
    max_delta_ssosex = np.nan

    if not args.skip_density_post:
        print("\nexact ED static orbital density response ...")
        Kdens = np.zeros((norb, norb, norb), dtype=complex)
        idx = np.arange(norb)
        Kdens[idx, idx, idx] = 1.0
        chi_nn_ed, _ = exact.static_susceptibility_matrix(Kdens, mu_ed, args.T)

        print(
            f"full {norb}x{norb} covariant density response by +/- fixed-point solves ..."
        )
        dens = finite_difference_density_response_same_torus(
            bg,
            h0,
            Vq,
            grid,
            eps=args.fd_eps,
            gw_opts=gw_opts,
            ssosex_opts=sx_opts,
            tol=args.fd_tol,
            max_iter=args.fd_max_iter,
            mixing=args.fd_mixing,
            allow_unconverged=args.allow_unconverged,
        )
        chi_nn_fd_raw = dens.chi_raw
        chi_nn_fd = dens.chi_sym
        density_reciprocity_error = dens.reciprocity_error
        density_fd_converged = dens.converged
        density_fd_res_plus = dens.plus_residual
        density_fd_res_minus = dens.minus_residual
        density_chi_relerr = _relerr(chi_nn_fd, chi_nn_ed)
        print(
            f"  density chi: relerr_vs_ED={density_chi_relerr:.6e}, "
            f"reciprocity_err(raw)={density_reciprocity_error:.3e}, "
            f"max_fd_res={max(np.nanmax(density_fd_res_plus),np.nanmax(density_fd_res_minus)):.3e}"
        )

        W_post = build_static_post_w_same_torus(Vq, bg.W, chi_nn_fd, grid)
        im0 = int(np.flatnonzero(np.asarray(grid.m_values) == 0)[0])
        max_delta_W0 = float(np.max(np.abs(W_post[im0] - bg.W[im0])))
        print(f"  max|W_post(0)-W_bg(0)|={max_delta_W0:.6e}")

        step = one_shot_post_ssosex_same_torus(
            bg,
            h0,
            Vq,
            W_post,
            grid,
            target_filling=target,
            gw_opts=gw_opts,
            ssosex_opts=sx_opts,
        )
        G_post = np.asarray(step["G"])
        mu_post = float(step["mu"])
        gerr_post, gerr_post_low = _green_errors(G_post, G_ed, grid, args.low_count)
        max_delta_ssosex = float(
            np.max(
                np.abs(
                    np.asarray(step["Sigma_sSOSEX_post"])
                    - np.asarray(bg.Sigma_sSOSEX)
                )
            )
        )
        print("\none-shot static post result:")
        print(
            f"  mu_post={mu_post:+.10f}, Gerr={gerr_post:.6e}, "
            f"Gerr_low={gerr_post_low:.6e}"
        )
        print(
            f"  improvement full={(gerr_bg-gerr_post)/max(gerr_bg,1e-300):+.3%}, "
            f"low={(gerr_bg_low-gerr_post_low)/max(gerr_bg_low,1e-300):+.3%}, "
            f"max|Delta Sigma_sSOSEX|={max_delta_ssosex:.3e}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    stem = (
        f"V{args.V:g}_fill{args.filling:g}_{args.L1}x{args.L2}_"
        f"{args.ssosex_mode}_covariant_post"
    )
    npz_path = args.out / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        V=float(args.V),
        filling=float(args.filling),
        T=float(args.T),
        L1=int(args.L1),
        L2=int(args.L2),
        channels=np.asarray(CHANNELS),
        fd_eps=float(args.fd_eps),
        mu_ed=float(mu_ed),
        mu_bg=float(bg.mu),
        mu_post=float(mu_post),
        chi_current_ed=np.asarray(chi_ed),
        chi_current_ssosex_fd=np.asarray(chi_fd),
        chi_current_ssosex_fd_sym=np.asarray(chi_fd_sym),
        current_chi_matrix_relerr=float(current_matrix_relerr),
        J0_background=np.asarray(J0),
        current_fd_converged=np.asarray(fd_conv),
        current_fd_res_plus=np.asarray(fd_res_plus),
        current_fd_res_minus=np.asarray(fd_res_minus),
        chi_nn_ed=np.asarray(chi_nn_ed),
        chi_nn_ssosex_fd_raw=np.asarray(chi_nn_fd_raw),
        chi_nn_ssosex_fd=np.asarray(chi_nn_fd),
        density_chi_relerr=float(density_chi_relerr),
        density_reciprocity_error=float(density_reciprocity_error),
        density_fd_converged=np.asarray(density_fd_converged),
        density_fd_res_plus=np.asarray(density_fd_res_plus),
        density_fd_res_minus=np.asarray(density_fd_res_minus),
        G_ed=np.asarray(G_ed),
        G_background=np.asarray(bg.G),
        G_post=np.asarray(G_post),
        Gerr_background=float(gerr_bg),
        Gerr_background_low=float(gerr_bg_low),
        Gerr_post=float(gerr_post),
        Gerr_post_low=float(gerr_post_low),
        W_background=np.asarray(bg.W),
        W_post=np.asarray(W_post),
        max_delta_W0=float(max_delta_W0),
        max_delta_ssosex=float(max_delta_ssosex),
        background_converged=bool(bg.converged),
        background_residual=float(bg.final_error),
        background_iterations=int(bg.iterations),
        Sigma_sSOSEX_background=np.asarray(bg.Sigma_sSOSEX),
    )
    config = vars(args).copy()
    config["out"] = str(args.out)
    config["method"] = {
        "background": "self-consistent GW+sSOSEX with ordinary P=GG screening",
        "covariant": (
            "central finite difference of the fully converged fixed point at fixed mu; "
            "includes explicit G and screened-line W[G] derivatives"
        ),
        "post": (
            "replace Omega=0 W by V-V chi_nn,cov V, retain background W at m!=0, "
            "recompute GW and sSOSEX once on G_background, then one fixed-filling Dyson step"
        ),
    }
    json_path = args.out / f"{stem}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nwrote {npz_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
