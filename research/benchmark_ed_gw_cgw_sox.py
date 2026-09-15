#!/usr/bin/env python3
"""Same-torus ED / GG / cGW / cGW+SOX benchmark with optional post corrections.

The finite Ruby torus is represented in two exactly matching ways:

* ED uses the full many-body Hilbert space of ``ExactSmallRubyThermal``;
* GW and GW+SOX treat every physical torus site as an orbital and use nk=1.

The original static-response benchmark remains

    ED / GG[GW] / cGW[GW] / cGW+SOX[GW+SOX].

Post corrections are independently selectable:

* ``--post-gw`` applies post-GW to the ordinary GW background only;
* ``--post-gw-sox`` applies post-GW to the GW+SOX background only.

For a selected post path, post-GW first constructs the full covariant density
response chi_nn(q,iOmega), updates the screened interaction W_post, evaluates
the one-shot post self-energy and Dyson equation to obtain G_post, and then
recomputes the q=0 static pseudospin susceptibility using the updated pair

    (G_post, W_post).

Thus the post response shown by this benchmark always uses both updated G and
updated W.  We do not report GG[post] or a fixed-G / W_post-only curve.
Green-function errors relative to exact ED are saved and plotted separately
from the susceptibility comparison.

The q=0 response on the post state should be read as the covariant response of
the updated post background.  It is not a second functional differentiation of
the entire one-shot post construction (which would additionally differentiate
the chi_nn entering W_post).

Plots are generated automatically next to ``benchmark.npz`` unless
``--no-plots`` is supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rubycgw.benchmark_plot import plot_benchmark_npz
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.gw_sox import solve_matrix_gw_sox
from rubycgw.model import RubyParameters
from rubycgw.post_gw import run_post_gw
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
    p.add_argument(
        "--post-gw",
        action="store_true",
        help="post-correct the ordinary GW background only",
    )
    p.add_argument(
        "--post-gw-sox",
        action="store_true",
        help="post-correct the self-consistent GW+SOX background only",
    )
    p.add_argument(
        "--post-mmax",
        type=int,
        default=None,
        help="diagnostic only: post-correct |m|<=M and keep background W outside",
    )
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/ed_gw_cgw_sox_benchmark"))
    return p.parse_args()


def _require(ok, message, allow):
    if not ok and not allow:
        raise RuntimeError(message)
    if not ok:
        print("WARNING:", message)


def _relative_green_error(G_approx, G_exact):
    arr = np.asarray(G_approx, dtype=complex)
    if arr.ndim == 5:
        arr = arr[:, 0, 0]
    exact = np.asarray(G_exact, dtype=complex)
    denom = float(np.linalg.norm(exact.ravel()))
    return float(np.linalg.norm((arr - exact).ravel()) / max(denom, 1e-300))


def _lowfreq_green_error(G_approx, G_exact, grid, count=4):
    arr = np.asarray(G_approx, dtype=complex)
    if arr.ndim == 5:
        arr = arr[:, 0, 0]
    exact = np.asarray(G_exact, dtype=complex)
    idx = np.argsort(np.abs(np.asarray(grid.omega)))[: min(int(count), grid.nf)]
    denom = float(np.linalg.norm(exact[idx].ravel()))
    return float(np.linalg.norm((arr[idx] - exact[idx]).ravel()) / max(denom, 1e-300))


def _solve_static_response(
    G,
    W,
    Vq,
    operators,
    channels,
    grid,
    reference,
    h_static,
    mu,
    vertex_opts,
    sox_opts,
    tail_edge_points,
    *,
    include_sox,
    allow_unconverged,
    label,
):
    results = []
    converged = np.zeros(len(channels), dtype=bool)
    for ic, (ch, K) in enumerate(zip(channels, operators)):
        print(f"  {label} vertex: {ch}")
        if include_sox:
            vr = solve_vertex_q0_tail_sox(
                G,
                W,
                Vq,
                K,
                grid,
                reference,
                h_static,
                mu,
                vertex_opts=vertex_opts,
                sox_opts=sox_opts,
            )
        else:
            vr = solve_vertex_q0_tail(
                G, W, Vq, K, grid, reference, opts=vertex_opts
            )
        results.append(vr)
        converged[ic] = vr.converged
        _require(
            vr.converged,
            f"{label} vertex {ch} did not converge: {vr.final_error:.3e}",
            allow_unconverged,
        )
    resp = static_response_tail_completed(
        G,
        operators,
        [x.Gamma for x in results],
        grid,
        h_static,
        mu,
        edge_points=tail_edge_points,
    )
    return resp, converged


def main():
    args = _args()
    if 6 * args.L1 * args.L2 > 16:
        raise ValueError("ExactSmallRubyThermal requires at most 16 sites")
    any_post = bool(args.post_gw or args.post_gw_sox)
    if args.post_mmax is not None and not any_post:
        raise ValueError("--post-mmax requires --post-gw and/or --post-gw-sox")
    if args.post_mmax is not None and args.post_mmax < 0:
        raise ValueError("--post-mmax must be non-negative")

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

    # Optional ordinary post-GW results: final chi uses (G_post, W_post).
    chi_post_gw_raw = np.full(shape, np.nan + 0j)
    chi_post_gw_completed = np.full(shape, np.nan + 0j)
    post_gw_chi_converged = np.zeros((nV, nc), dtype=bool)
    mu_post_gw = np.full(nV, np.nan)
    g_relerr_gw = np.full(nV, np.nan)
    g_relerr_post_gw = np.full(nV, np.nan)
    g_lowfreq_relerr_gw = np.full(nV, np.nan)
    g_lowfreq_relerr_post_gw = np.full(nV, np.nan)
    max_delta_w_post_gw = np.full(nV, np.nan)
    post_fallback_fraction_gw = np.full(nV, np.nan)

    # Optional post-(GW+SOX): final chi uses (G_post, W_post) and SOX vertex.
    chi_post_gw_sox_raw = np.full(shape, np.nan + 0j)
    chi_post_gw_sox_completed = np.full(shape, np.nan + 0j)
    post_gw_sox_chi_converged = np.zeros((nV, nc), dtype=bool)
    mu_post_gw_sox = np.full(nV, np.nan)
    g_relerr_gw_sox = np.full(nV, np.nan)
    g_relerr_post_gw_sox = np.full(nV, np.nan)
    g_lowfreq_relerr_gw_sox = np.full(nV, np.nan)
    g_lowfreq_relerr_post_gw_sox = np.full(nV, np.nan)
    max_delta_w_post_gw_sox = np.full(nV, np.nan)
    post_fallback_fraction_gw_sox = np.full(nV, np.nan)

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
        print("  ED static response complete")

        h0 = np.asarray(exact.h0, dtype=complex)[None, None]
        Vq = (float(V) * np.asarray(exact.Vunit, dtype=complex))[None, None]

        # ---------------- ordinary GW / cGW ----------------
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

        resp, conv = _solve_static_response(
            gw.G,
            gw.W,
            Vq,
            operators,
            channels,
            grid,
            reference_gw,
            h_static_gw,
            gw.mu,
            vertex_opts,
            sox_opts,
            args.tail_edge_points,
            include_sox=False,
            allow_unconverged=args.allow_unconverged,
            label="cGW",
        )
        chi_cgw_raw[iv] = resp["raw"]
        chi_cgw_completed[iv] = resp["completed"]
        tail_cgw[iv] = resp["tail_correction"]
        cgw_converged[iv] = conv

        # ---------------- GW+SOX / cGW+SOX ----------------
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

        resp_sox, conv = _solve_static_response(
            gwsox.G,
            gwsox.W,
            Vq,
            operators,
            channels,
            grid,
            reference_sox,
            h_static_sox,
            gwsox.mu,
            vertex_opts,
            sox_opts,
            args.tail_edge_points,
            include_sox=True,
            allow_unconverged=args.allow_unconverged,
            label="cGW+SOX",
        )
        chi_sox_raw[iv] = resp_sox["raw"]
        chi_sox_completed[iv] = resp_sox["completed"]
        tail_sox[iv] = resp_sox["tail_correction"]
        cgw_sox_converged[iv] = conv

        print("  diagonal static susceptibilities")
        for ic, ch in enumerate(channels):
            print(
                f"    {ch:12s} ED={ed_chi[iv,ic,ic]:+.8f} "
                f"GG={chi_gg_completed[iv,ic,ic].real:+.8f} "
                f"cGW={chi_cgw_completed[iv,ic,ic].real:+.8f} "
                f"cGW+SOX={chi_sox_completed[iv,ic,ic].real:+.8f}"
            )

        G_ed = None
        if any_post:
            print("  exact ED Matsubara Green function")
            G_ed, _ = exact.green_iomega(
                1j * np.asarray(grid.omega), mu_ed[iv], args.T
            )

        # ---------------- ordinary post-GW only ----------------
        if args.post_gw:
            window = (
                "full dynamic"
                if args.post_mmax is None
                else f"|m|<={args.post_mmax} windowed"
            )
            print(f"  solving post-GW from ordinary GW background ({window})")
            post = run_post_gw(
                gw,
                h0,
                Vq,
                grid,
                gw_opts=gw_opts,
                vertex_opts=vertex_opts,
                include_sox=False,
                sox_opts=sox_opts,
                m_max=args.post_mmax,
                allow_unconverged=args.allow_unconverged,
            )
            mu_post_gw[iv] = post.mu
            max_delta_w_post_gw[iv] = float(np.max(np.abs(post.W_post - gw.W)))
            post_fallback_fraction_gw[iv] = float(np.mean(post.used_background_w_mask))

            h_static_post = h0 + post.Sigma_H[None, None] + post.Sigma_F
            reference_post = build_tail_reference(
                h0, post.mu, post.Sigma_H, grid
            )
            print("  static chi on updated post-GW state (G_post, W_post)")
            post_resp, conv = _solve_static_response(
                post.G,
                post.W_post,
                Vq,
                operators,
                channels,
                grid,
                reference_post,
                h_static_post,
                post.mu,
                vertex_opts,
                sox_opts,
                args.tail_edge_points,
                include_sox=False,
                allow_unconverged=args.allow_unconverged,
                label="post-GW chi",
            )
            chi_post_gw_raw[iv] = post_resp["raw"]
            chi_post_gw_completed[iv] = post_resp["completed"]
            post_gw_chi_converged[iv] = conv

            g_relerr_gw[iv] = _relative_green_error(gw.G, G_ed)
            g_relerr_post_gw[iv] = _relative_green_error(post.G, G_ed)
            g_lowfreq_relerr_gw[iv] = _lowfreq_green_error(gw.G, G_ed, grid)
            g_lowfreq_relerr_post_gw[iv] = _lowfreq_green_error(post.G, G_ed, grid)
            print(
                "    full-G relerr: "
                f"GW={g_relerr_gw[iv]:.4e} "
                f"post-GW={g_relerr_post_gw[iv]:.4e}"
            )
            for ic, ch in enumerate(channels):
                print(
                    f"    {ch:12s} cGW={chi_cgw_completed[iv,ic,ic].real:+.8f} "
                    f"post-GW chi={chi_post_gw_completed[iv,ic,ic].real:+.8f} "
                    f"ED={ed_chi[iv,ic,ic]:+.8f}"
                )

        # ---------------- post-(GW+SOX) only ----------------
        if args.post_gw_sox:
            window = (
                "full dynamic"
                if args.post_mmax is None
                else f"|m|<={args.post_mmax} windowed"
            )
            print(f"  solving post-(GW+SOX) background ({window})")
            post_sox = run_post_gw(
                gwsox,
                h0,
                Vq,
                grid,
                gw_opts=gw_opts,
                vertex_opts=vertex_opts,
                include_sox=True,
                sox_opts=sox_opts,
                m_max=args.post_mmax,
                allow_unconverged=args.allow_unconverged,
            )
            mu_post_gw_sox[iv] = post_sox.mu
            max_delta_w_post_gw_sox[iv] = float(
                np.max(np.abs(post_sox.W_post - gwsox.W))
            )
            post_fallback_fraction_gw_sox[iv] = float(
                np.mean(post_sox.used_background_w_mask)
            )

            h_static_post_sox = (
                h0 + post_sox.Sigma_H[None, None] + post_sox.Sigma_F
            )
            reference_post_sox = build_tail_reference(
                h0, post_sox.mu, post_sox.Sigma_H, grid
            )
            print("  static chi on updated post-(GW+SOX) state")
            post_sox_resp, conv = _solve_static_response(
                post_sox.G,
                post_sox.W_post,
                Vq,
                operators,
                channels,
                grid,
                reference_post_sox,
                h_static_post_sox,
                post_sox.mu,
                vertex_opts,
                sox_opts,
                args.tail_edge_points,
                include_sox=True,
                allow_unconverged=args.allow_unconverged,
                label="post-(GW+SOX) chi",
            )
            chi_post_gw_sox_raw[iv] = post_sox_resp["raw"]
            chi_post_gw_sox_completed[iv] = post_sox_resp["completed"]
            post_gw_sox_chi_converged[iv] = conv

            g_relerr_gw_sox[iv] = _relative_green_error(gwsox.G, G_ed)
            g_relerr_post_gw_sox[iv] = _relative_green_error(post_sox.G, G_ed)
            g_lowfreq_relerr_gw_sox[iv] = _lowfreq_green_error(
                gwsox.G, G_ed, grid
            )
            g_lowfreq_relerr_post_gw_sox[iv] = _lowfreq_green_error(
                post_sox.G, G_ed, grid
            )
            print(
                "    full-G relerr: "
                f"GW+SOX={g_relerr_gw_sox[iv]:.4e} "
                f"post-(GW+SOX)={g_relerr_post_gw_sox[iv]:.4e}"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    save = dict(
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
    if args.post_gw:
        save.update(
            mu_post_gw=mu_post_gw,
            post_gw_chi_raw=chi_post_gw_raw,
            post_gw_chi_completed=chi_post_gw_completed,
            post_gw_chi_converged=post_gw_chi_converged,
            g_relerr_gw=g_relerr_gw,
            g_relerr_post_gw=g_relerr_post_gw,
            g_lowfreq_relerr_gw=g_lowfreq_relerr_gw,
            g_lowfreq_relerr_post_gw=g_lowfreq_relerr_post_gw,
            max_delta_w_post_gw=max_delta_w_post_gw,
            post_fallback_fraction_gw=post_fallback_fraction_gw,
        )
    if args.post_gw_sox:
        save.update(
            mu_post_gw_sox=mu_post_gw_sox,
            post_gw_sox_chi_raw=chi_post_gw_sox_raw,
            post_gw_sox_chi_completed=chi_post_gw_sox_completed,
            post_gw_sox_chi_converged=post_gw_sox_chi_converged,
            g_relerr_gw_sox=g_relerr_gw_sox,
            g_relerr_post_gw_sox=g_relerr_post_gw_sox,
            g_lowfreq_relerr_gw_sox=g_lowfreq_relerr_gw_sox,
            g_lowfreq_relerr_post_gw_sox=g_lowfreq_relerr_post_gw_sox,
            max_delta_w_post_gw_sox=max_delta_w_post_gw_sox,
            post_fallback_fraction_gw_sox=post_fallback_fraction_gw_sox,
        )
    npz_path = args.out / "benchmark.npz"
    np.savez_compressed(npz_path, **save)

    post_methods = []
    if args.post_gw:
        post_methods.append("post-GW")
    if args.post_gw_sox:
        post_methods.append("post-(GW+SOX)")
    config = {
        "geometry": {"L1": args.L1, "L2": args.L2, "n_sites": 6*ncell},
        "response_methods": ["ED", "GG", "cGW", "cGW+SOX"],
        "post_methods": post_methods,
        "comparison_rule": "Compare ED static response to *_completed, not raw finite-box values.",
        "post_screening_identity": "W_post = V - V chi_nn,cov V in the RubycGW sign convention.",
        "post_response": (
            "For every selected post method, the reported pseudospin chi is recomputed "
            "on the updated pair (G_post,W_post); no GG[post] or fixed-G/W_post-only curve is used."
        ),
        "post_response_caveat": (
            "This is the covariant q=0 response of the updated post background, not a second "
            "functional derivative of the complete one-shot post map."
        ),
        "parameters": vars(args).copy(),
    }
    config["parameters"]["out"] = str(args.out)
    with (args.out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nwrote {npz_path}")
    print(f"wrote {args.out/'config.json'}")
    if not args.no_plots:
        for path in plot_benchmark_npz(npz_path):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
