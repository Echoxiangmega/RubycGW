#!/usr/bin/env python3
"""Diagnose covariant density-vertex corrections to screening on a primitive mesh.

This script keeps the ordinary self-consistent GW background fixed, computes the
full orbital density covariant response on selected bosonic Matsubara transfers,
constructs

    W_post(Q) = V(q) - V(q) chi_nn,cov(Q) V(q),

and performs the existing one-shot post-GW Dyson update.  It is intended as a
cheap diagnostic before attempting a fully self-consistent GWGamma_P loop.

By default only the static bosonic sector (m=0) is corrected; higher bosonic
frequencies fall back to the background GW screened interaction.  Use
--m-max -1 for the full represented dynamic response.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from benchmark_finite_source_current import _bilinear_expectation_tail_completed
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.post_gw import run_post_gw
from rubycgw.pseudospin import canonical_channel_name, primitive_pseudospin_vertex
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast
from rubycgw.supercell_gw_split import compute_sigma_gw_split_components


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--source", choices=["same", "opposite", "z_same", "z_opposite"], default="same")
    p.add_argument("--mesh", default="4x4", help="primitive k/q mesh, e.g. 4x4")
    p.add_argument("--h", nargs="+", type=float, default=[0.05, 0.02, 0.01])
    p.add_argument("--reference-ncell", type=int, default=2)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)

    p.add_argument("--gw-max-iter", type=int, default=1000)
    p.add_argument("--gw-mixing", type=float, default=0.18)
    p.add_argument("--gw-tol", type=float, default=2e-8)
    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--pulay-history", type=int, default=8)
    p.add_argument("--pulay-start", type=int, default=4)
    p.add_argument("--pulay-regularization", type=float, default=1e-9)
    p.add_argument("--backend", choices=["fft", "direct"], default="fft")

    p.add_argument("--vertex-max-iter", type=int, default=180)
    p.add_argument("--vertex-tol", type=float, default=2e-8)
    p.add_argument("--gmres-restart", type=int, default=14)
    p.add_argument(
        "--m-max", type=int, default=0,
        help="post-correct |m|<=m_max; use -1 for all represented bosonic frequencies",
    )
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fit-points", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("results/covariant_screening_primitive"))
    return p.parse_args()


def _parse_mesh(token: str) -> tuple[int, int]:
    raw = str(token).strip().lower().replace("×", "x")
    parts = raw.split("x")
    if len(parts) != 2:
        raise ValueError(f"invalid mesh {token!r}; expected NxM")
    n1, n2 = int(parts[0]), int(parts[1])
    if n1 < 1 or n2 < 1:
        raise ValueError("mesh dimensions must be positive")
    return n1, n2


def _hvalues(values) -> np.ndarray:
    arr = np.asarray([float(x) for x in values], dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)) or np.any(arr < -1e-15):
        raise ValueError("--h must contain finite non-negative values")
    arr[np.abs(arr) < 1e-15] = 0.0
    return np.asarray(sorted(set(arr.tolist()), reverse=True), dtype=float)


def _gw_retry(h0, Vq, grid, opts, initial=None):
    plans = [
        ("base", opts),
        ("pulay-0.10", replace(opts, mixing=min(float(opts.mixing), 0.10),
                                mixing_method="pulay", max_iter=max(int(opts.max_iter), 1400))),
        ("pulay-0.05", replace(opts, mixing=min(float(opts.mixing), 0.05),
                                mixing_method="pulay", max_iter=max(int(opts.max_iter), 1800))),
        ("linear-0.03", replace(opts, mixing=min(float(opts.mixing), 0.03),
                                 mixing_method="linear", max_iter=max(int(opts.max_iter), 2200))),
    ]
    seed = initial
    best = None
    label_best = "none"
    for ia, (label, trial) in enumerate(plans):
        out = solve_matrix_gw_fast(h0, Vq, grid, opts=trial, initial=seed)
        print(
            f"      GW retry[{ia}] {label:12s}: {'OK' if out.converged else 'FAIL':4s} "
            f"iter={out.iterations:4d} r={out.final_error:.3e} "
            f"smin={out.min_screening_singular_value:.3e} "
            f"q*=({out.min_screening_q1:.4f},{out.min_screening_q2:.4f})"
        )
        if best is None or out.final_error < best.final_error:
            best, label_best = out, label
        if out.converged:
            return out, label
        seed = out
    return best, label_best


def _gw_current(result, K, Vq, grid, h0, backend) -> float:
    _, sigma_f, _, _ = compute_sigma_gw_split_components(
        result.G, result.W, Vq, grid, h0, result.mu, result.Sigma_H,
        backend=backend,
    )
    h_static = h0 + result.Sigma_H[None, None] + sigma_f
    return float(
        _bilinear_expectation_tail_completed(
            result.G, K, grid, h_static, result.mu
        ).real
    )


def _post_current(post, K, grid, h0) -> float:
    h_static = h0 + post.Sigma_H[None, None] + post.Sigma_F
    return float(
        _bilinear_expectation_tail_completed(
            post.G, K, grid, h_static, post.mu
        ).real
    )


def _screening_metrics(Wbg, Wpost, fallback, grid):
    Wbg = np.asarray(Wbg, dtype=complex)
    Wpost = np.asarray(Wpost, dtype=complex)
    solved = ~np.asarray(fallback, dtype=bool)
    if not np.any(solved):
        return {
            "rel_frob": np.nan,
            "max_abs": np.nan,
            "static_rel_frob": np.nan,
            "static_max_abs": np.nan,
            "static_q1": np.nan,
            "static_q2": np.nan,
        }

    dW = Wpost - Wbg
    flat_mask = solved[..., None, None]
    num = np.linalg.norm(dW[flat_mask.repeat(Wbg.shape[-2] * Wbg.shape[-1], axis=-1).reshape(Wbg.shape)])
    den = np.linalg.norm(Wbg[flat_mask.repeat(Wbg.shape[-2] * Wbg.shape[-1], axis=-1).reshape(Wbg.shape)])
    rel = float(num / den) if den > 0 else np.nan
    max_abs = float(np.max(np.abs(dW[solved])))

    m0_match = np.flatnonzero(np.asarray(grid.m_values) == 0)
    if not m0_match.size:
        return {
            "rel_frob": rel,
            "max_abs": max_abs,
            "static_rel_frob": np.nan,
            "static_max_abs": np.nan,
            "static_q1": np.nan,
            "static_q2": np.nan,
        }
    im0 = int(m0_match[0])
    smask = solved[im0]
    if not np.any(smask):
        return {
            "rel_frob": rel,
            "max_abs": max_abs,
            "static_rel_frob": np.nan,
            "static_max_abs": np.nan,
            "static_q1": np.nan,
            "static_q2": np.nan,
        }
    ds = dW[im0]
    num_s = np.linalg.norm(ds[smask])
    den_s = np.linalg.norm(Wbg[im0][smask])
    rel_s = float(num_s / den_s) if den_s > 0 else np.nan
    qnorm = np.linalg.norm(ds.reshape(grid.nk1, grid.nk2, -1), axis=-1)
    qnorm = np.where(smask, qnorm, -np.inf)
    iq1, iq2 = np.unravel_index(int(np.argmax(qnorm)), qnorm.shape)
    return {
        "rel_frob": rel,
        "max_abs": max_abs,
        "static_rel_frob": rel_s,
        "static_max_abs": float(np.max(np.abs(ds[smask]))),
        "static_q1": float(iq1 / grid.nk1),
        "static_q2": float(iq2 / grid.nk2),
    }


def _fit(h, J, ok, npoints):
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    idx = np.flatnonzero(np.asarray(ok, dtype=bool) & np.isfinite(J) & (h > 0))
    if idx.size < 2:
        return np.nan, np.nan, 0
    idx = idx[np.argsort(h[idx])][:max(2, int(npoints))]
    slope, intercept = np.polyfit(h[idx], J[idx], 1)
    return float(intercept), float(slope), int(idx.size)


def main():
    args = _args()
    nk1, nk2 = _parse_mesh(args.mesh)
    h_ref = _hvalues(args.h)
    if args.reference_ncell < 1 or args.T <= 0:
        raise ValueError("need reference_ncell>=1 and T>0")
    m_max = None if int(args.m_max) < 0 else int(args.m_max)

    source = canonical_channel_name(args.source)
    K = np.asarray(primitive_pseudospin_vertex(source), dtype=complex)
    root = float(np.sqrt(args.reference_ncell))
    h_cell = h_ref / root
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

    n = len(h_ref)
    Jgw = np.full(n, np.nan)
    Jpost = np.full(n, np.nan)
    gw_ok = np.zeros(n, dtype=bool)
    post_ok = np.zeros(n, dtype=bool)
    relW = np.full(n, np.nan)
    maxdW = np.full(n, np.nan)
    relW0 = np.full(n, np.nan)
    maxdW0 = np.full(n, np.nan)
    qW1 = np.full(n, np.nan)
    qW2 = np.full(n, np.nan)
    vertex_err = np.full(n, np.nan)
    n_explicit = np.zeros(n, dtype=int)
    n_fallback = np.zeros(n, dtype=int)

    print("=== primitive dense-k covariant screening diagnostic ===")
    print(f"V={args.V:g}, filling={args.filling:g}, T={args.T:g}, source={source}")
    print(f"mesh={nk1}x{nk2}, h_ref={h_ref.tolist()}, m_max={m_max}")
    print("W_post = V - V chi_nn,cov V; background is ordinary self-consistent GW")

    last_gw = None
    for ih, (href, hpc) in enumerate(zip(h_ref, h_cell)):
        print(f"\n-- h_ref={href:.8g}, h_cell={hpc:.8g} --")
        h0 = h0_base - float(hpc) * K[None, None]
        h0 = 0.5 * (h0 + np.swapaxes(h0.conj(), -1, -2))

        gw, retry = _gw_retry(h0, Vq, grid, gw_opts, initial=last_gw)
        gw_ok[ih] = bool(gw.converged)
        Jgw[ih] = root * _gw_current(gw, K, Vq, grid, h0, args.backend)
        if gw.converged:
            last_gw = gw
        if not gw.converged and not args.allow_unconverged:
            raise RuntimeError(f"GW did not converge at h_ref={href:g}")

        print("      solving covariant density vertices ...")
        post = run_post_gw(
            gw,
            h0,
            Vq,
            grid,
            gw_opts=gw_opts,
            vertex_opts=vertex_opts,
            include_sox=False,
            m_max=m_max,
            allow_unconverged=args.allow_unconverged,
        )
        dens = post.density_response
        explicit = np.asarray(dens.solved_explicitly, dtype=bool)
        conv = np.asarray(dens.transfer_converged, dtype=bool)
        errs = np.asarray(dens.transfer_max_error, dtype=float)
        n_explicit[ih] = int(np.count_nonzero(explicit))
        n_fallback[ih] = int(np.count_nonzero(post.used_background_w_mask))
        post_ok[ih] = bool(np.all(conv[explicit])) if np.any(explicit) else False
        if np.any(explicit):
            vertex_err[ih] = float(np.nanmax(errs[explicit]))

        metrics = _screening_metrics(gw.W, post.W_post, post.used_background_w_mask, grid)
        relW[ih] = metrics["rel_frob"]
        maxdW[ih] = metrics["max_abs"]
        relW0[ih] = metrics["static_rel_frob"]
        maxdW0[ih] = metrics["static_max_abs"]
        qW1[ih] = metrics["static_q1"]
        qW2[ih] = metrics["static_q2"]
        Jpost[ih] = root * _post_current(post, K, grid, h0)

        print(
            f"    GW:        J_ref={Jgw[ih]:+.9f}, smin={gw.min_screening_singular_value:.4e}, "
            f"q*=({gw.min_screening_q1:.4f},{gw.min_screening_q2:.4f}), retry={retry}"
        )
        print(
            f"    covariant: vertices={'OK' if post_ok[ih] else 'FAIL'} "
            f"n_explicit={n_explicit[ih]} maxerr={vertex_err[ih]:.3e}"
        )
        print(
            f"    W shift:   relF={relW[ih]:.4e}, max|dW|={maxdW[ih]:.4e}; "
            f"static relF={relW0[ih]:.4e}, static max|dW|={maxdW0[ih]:.4e} "
            f"at/near q=({qW1[ih]:.4f},{qW2[ih]:.4f})"
        )
        print(
            f"    post-Dyson: J_ref={Jpost[ih]:+.9f}, dJ={Jpost[ih]-Jgw[ih]:+.9f}, "
            f"fallback transfers={n_fallback[ih]}"
        )

    J0_gw, slope_gw, ngw = _fit(h_ref, Jgw, gw_ok, args.fit_points)
    J0_post, slope_post, npost = _fit(h_ref, Jpost, post_ok, args.fit_points)
    print("\n=== small-h fits ===")
    print(f"GW:       J0={J0_gw:+.9f}, slope={slope_gw:+.9f}, n={ngw}")
    print(f"post-GW:  J0={J0_post:+.9f}, slope={slope_post:+.9f}, n={npost}")
    print(f"dJ0={J0_post-J0_gw:+.9f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out
    npz_path = stem.with_suffix(".npz")
    csv_path = stem.with_suffix(".csv")
    np.savez_compressed(
        npz_path,
        V=float(args.V), filling=float(args.filling), T=float(args.T),
        source=source, nk1=nk1, nk2=nk2, m_max=-1 if m_max is None else m_max,
        h_ref=h_ref, h_cell=h_cell,
        J_gw=Jgw, J_post=Jpost, gw_ok=gw_ok, post_ok=post_ok,
        W_rel_frob=relW, W_max_abs_shift=maxdW,
        W_static_rel_frob=relW0, W_static_max_abs_shift=maxdW0,
        W_static_shift_q1=qW1, W_static_shift_q2=qW2,
        vertex_max_error=vertex_err,
        n_explicit=n_explicit, n_fallback=n_fallback,
        J0_gw=J0_gw, slope_gw=slope_gw,
        J0_post=J0_post, slope_post=slope_post,
    )
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "h_ref", "h_cell", "gw_ok", "post_ok", "J_gw", "J_post", "dJ",
            "W_rel_frob", "W_max_abs_shift", "W_static_rel_frob",
            "W_static_max_abs_shift", "W_static_shift_q1", "W_static_shift_q2",
            "vertex_max_error", "n_explicit", "n_fallback",
        ])
        for i in range(n):
            w.writerow([
                h_ref[i], h_cell[i], int(gw_ok[i]), int(post_ok[i]), Jgw[i], Jpost[i],
                Jpost[i] - Jgw[i], relW[i], maxdW[i], relW0[i], maxdW0[i],
                qW1[i], qW2[i], vertex_err[i], n_explicit[i], n_fallback[i],
            ])
    print(f"saved {npz_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
