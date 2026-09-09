#!/usr/bin/env python3
"""Compare primitive-cell dense-k finite-source SC-GW and SC-GW+SOX.

The calculation uses the same six-orbital primitive Hamiltonian, genuine k/q
meshes, fixed filling and source normalization for both approximations.  The
GW+SOX branch is fully self-consistent:

    Sigma_corr = Sigma_GW + Sigma_SOX,

and is not a post-processing correction on a frozen GW Green function.

By default the input source h and reported J are normalized to the existing
2x1 ED benchmark,

    h_cell = h_ref/sqrt(Nref),
    J_ref  = sqrt(Nref) j_cell,

with Nref=2.
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
from rubycgw.gw_sox import solve_matrix_gw_sox
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import canonical_channel_name, primitive_pseudospin_vertex
from rubycgw.sox_covariant import SOXOptions
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast
from rubycgw.supercell_gw_split import compute_sigma_gw_split_components


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--source", choices=["same", "opposite", "z_same", "z_opposite"], default="same")
    p.add_argument("--mesh", nargs="+", default=["3x3", "4x4"],
                   help="primitive k/q meshes, e.g. --mesh 3x3 4x4 5x5")
    p.add_argument("--h", nargs="+", type=float,
                   default=[0.2, 0.1, 0.05, 0.02, 0.01])
    p.add_argument("--reference-ncell", type=int, default=2)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)

    p.add_argument("--gw-max-iter", type=int, default=800)
    p.add_argument("--gw-mixing", type=float, default=0.22)
    p.add_argument("--gw-tol", type=float, default=2e-8)

    p.add_argument("--sox-max-iter", type=int, default=1600)
    p.add_argument("--sox-mixing", type=float, default=0.08)
    p.add_argument("--sox-nquad", type=int, default=128)
    p.add_argument("--sox-interaction-tol", type=float, default=1e-13)
    p.add_argument("--sox-tail-edge-points", type=int, default=2)

    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--pulay-history", type=int, default=6)
    p.add_argument("--pulay-start", type=int, default=3)
    p.add_argument("--pulay-regularization", type=float, default=1e-10)
    p.add_argument("--backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--fit-points", type=int, default=3)
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/primitive_finite_source_gw_sox_nk"))
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


def _gw_current(result, K, Vq, grid, h0, backend) -> float:
    _, sigma_f, _, _ = compute_sigma_gw_split_components(
        result.G, result.W, Vq, grid, h0, result.mu, result.Sigma_H,
        backend=backend,
    )
    h_static = h0 + result.Sigma_H[None, None] + sigma_f
    return float(_bilinear_expectation_tail_completed(
        result.G, K, grid, h_static, result.mu
    ).real)


def _sox_current(result, K, grid, h0) -> float:
    h_static = h0 + result.Sigma_H[None, None] + result.Sigma_F
    return float(_bilinear_expectation_tail_completed(
        result.G, K, grid, h_static, result.mu
    ).real)


def _gw_retry(h0, Vq, grid, opts, initial=None):
    plans = [
        ("base", opts),
        ("pulay-0.12", replace(opts, mixing=min(float(opts.mixing), 0.12),
                               mixing_method="pulay", max_iter=max(int(opts.max_iter), 1200),
                               pulay_history=max(int(opts.pulay_history), 8),
                               pulay_start=max(int(opts.pulay_start), 4),
                               pulay_regularization=max(float(opts.pulay_regularization), 1e-9))),
        ("pulay-0.06", replace(opts, mixing=min(float(opts.mixing), 0.06),
                               mixing_method="pulay", max_iter=max(int(opts.max_iter), 1600),
                               pulay_history=max(int(opts.pulay_history), 8),
                               pulay_start=max(int(opts.pulay_start), 5),
                               pulay_regularization=max(float(opts.pulay_regularization), 1e-8))),
        ("linear-0.03", replace(opts, mixing=min(float(opts.mixing), 0.03),
                                mixing_method="linear", max_iter=max(int(opts.max_iter), 2000))),
    ]
    seed = initial
    best = None
    label_best = "none"
    for ia, (label, trial) in enumerate(plans):
        out = solve_matrix_gw_fast(h0, Vq, grid, opts=trial, initial=seed)
        print(f"      GW retry[{ia}] {label:12s}: {'OK' if out.converged else 'FAIL':4s} "
              f"iter={out.iterations:4d} r={out.final_error:.3e} "
              f"smin={out.min_screening_singular_value:.3e} "
              f"q*=({out.min_screening_q1:.4f},{out.min_screening_q2:.4f})")
        if best is None or out.final_error < best.final_error:
            best, label_best = out, label
        if out.converged:
            return out, label
        seed = out
    return best, label_best


def _sox_retry(h0, Vq, grid, opts, sox_opts, initial=None, gw_fallback=None):
    plans = [
        ("base", opts),
        ("pulay-0.05", replace(opts, mixing=min(float(opts.mixing), 0.05),
                               mixing_method="pulay", max_iter=max(int(opts.max_iter), 2200),
                               pulay_history=max(int(opts.pulay_history), 8),
                               pulay_start=max(int(opts.pulay_start), 4),
                               pulay_regularization=max(float(opts.pulay_regularization), 1e-9))),
        ("pulay-0.03", replace(opts, mixing=min(float(opts.mixing), 0.03),
                               mixing_method="pulay", max_iter=max(int(opts.max_iter), 2800),
                               pulay_history=max(int(opts.pulay_history), 8),
                               pulay_start=max(int(opts.pulay_start), 5),
                               pulay_regularization=max(float(opts.pulay_regularization), 1e-8))),
        ("linear-0.02", replace(opts, mixing=min(float(opts.mixing), 0.02),
                                mixing_method="linear", max_iter=max(int(opts.max_iter), 3200))),
    ]
    seed = initial if initial is not None else gw_fallback
    best = None
    label_best = "none"
    for ia, (label, trial) in enumerate(plans):
        out = solve_matrix_gw_sox(h0, Vq, grid, opts=trial, sox_opts=sox_opts, initial=seed)
        sox_scale = float(np.max(np.abs(out.Sigma_SOX)))
        print(f"      SOX retry[{ia}] {label:12s}: {'OK' if out.converged else 'FAIL':4s} "
              f"iter={out.iterations:4d} r={out.final_error:.3e} "
              f"max|Ssox|={sox_scale:.3e} smin={out.min_screening_singular_value:.3e} "
              f"q*=({out.min_screening_q1:.4f},{out.min_screening_q2:.4f})")
        if best is None or out.final_error < best.final_error:
            best, label_best = out, label
        if out.converged:
            return out, label
        seed = out
    return best, label_best


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
    meshes = []
    for token in args.mesh:
        m = _parse_mesh(token)
        if m not in meshes:
            meshes.append(m)
    h_ref = _hvalues(args.h)
    if args.reference_ncell < 1 or args.T <= 0:
        raise ValueError("need reference_ncell>=1 and T>0")

    source = canonical_channel_name(args.source)
    K = np.asarray(primitive_pseudospin_vertex(source), dtype=complex)
    root = float(np.sqrt(args.reference_ncell))
    h_cell = h_ref / root
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    labels = [f"{a}x{b}" for a, b in meshes]
    shape = (len(meshes), len(h_ref))

    Jgw = np.full(shape, np.nan)
    Jsox = np.full(shape, np.nan)
    jgw = np.full(shape, np.nan)
    jsox = np.full(shape, np.nan)
    gw_ok = np.zeros(shape, dtype=bool)
    sox_ok = np.zeros(shape, dtype=bool)
    gw_iter = np.zeros(shape, dtype=int)
    sox_iter = np.zeros(shape, dtype=int)
    gw_res = np.full(shape, np.nan)
    sox_res = np.full(shape, np.nan)
    gw_smin = np.full(shape, np.nan)
    sox_smin = np.full(shape, np.nan)
    gw_q1 = np.full(shape, np.nan)
    gw_q2 = np.full(shape, np.nan)
    sox_q1 = np.full(shape, np.nan)
    sox_q2 = np.full(shape, np.nan)
    sox_scale = np.full(shape, np.nan)
    mu_gw = np.full(shape, np.nan)
    mu_sox = np.full(shape, np.nan)

    print("=== primitive finite-source: SC-GW vs self-consistent GW+SOX ===")
    print(f"V={args.V:g}, filling={args.filling:g}, T={args.T:g}, source={source}")
    print(f"meshes={labels}")
    print(f"h_ref={h_ref.tolist()}")
    print(f"h_cell={h_cell.tolist()} (reference_ncell={args.reference_ncell})")

    sox_opts = SOXOptions(
        n_quad=args.sox_nquad,
        interaction_tol=args.sox_interaction_tol,
        tail_complete=True,
        tail_edge_points=args.sox_tail_edge_points,
    )

    for im, (nk1, nk2) in enumerate(meshes):
        print(f"\n### mesh={labels[im]} ###")
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
        sox_gw_opts = replace(
            gw_opts, max_iter=args.sox_max_iter, mixing=args.sox_mixing
        )

        last_gw = None
        last_sox = None
        for ih, (href, hpc) in enumerate(zip(h_ref, h_cell)):
            print(f"  -- h_ref={href:.8g}, h_cell={hpc:.8g} --")
            h0 = h0_base - float(hpc) * K[None, None]
            h0 = 0.5 * (h0 + np.swapaxes(h0.conj(), -1, -2))

            gw, gw_retry = _gw_retry(h0, Vq, grid, gw_opts, initial=last_gw)
            gw_ok[im, ih] = bool(gw.converged)
            gw_iter[im, ih] = int(gw.iterations)
            gw_res[im, ih] = float(gw.final_error)
            gw_smin[im, ih] = float(gw.min_screening_singular_value)
            gw_q1[im, ih] = float(gw.min_screening_q1)
            gw_q2[im, ih] = float(gw.min_screening_q2)
            mu_gw[im, ih] = float(gw.mu)
            jgw[im, ih] = _gw_current(gw, K, Vq, grid, h0, args.backend)
            Jgw[im, ih] = root * jgw[im, ih]
            if gw.converged:
                last_gw = gw

            seed_sox = last_sox if last_sox is not None else (gw if gw.converged else None)
            sox, sox_retry = _sox_retry(
                h0, Vq, grid, sox_gw_opts, sox_opts,
                initial=seed_sox,
                gw_fallback=gw if gw.converged else None,
            )
            sox_ok[im, ih] = bool(sox.converged)
            sox_iter[im, ih] = int(sox.iterations)
            sox_res[im, ih] = float(sox.final_error)
            sox_smin[im, ih] = float(sox.min_screening_singular_value)
            sox_q1[im, ih] = float(sox.min_screening_q1)
            sox_q2[im, ih] = float(sox.min_screening_q2)
            mu_sox[im, ih] = float(sox.mu)
            sox_scale[im, ih] = float(np.max(np.abs(sox.Sigma_SOX)))
            jsox[im, ih] = _sox_current(sox, K, grid, h0)
            Jsox[im, ih] = root * jsox[im, ih]
            if sox.converged:
                last_sox = sox

            print(
                f"    summary GW:     {'OK' if gw.converged else 'FAIL':4s} "
                f"J_ref={Jgw[im,ih]:+.9f} r={gw.final_error:.3e} retry={gw_retry}"
            )
            print(
                f"    summary GW+SOX: {'OK' if sox.converged else 'FAIL':4s} "
                f"J_ref={Jsox[im,ih]:+.9f} r={sox.final_error:.3e} "
                f"max|Ssox|={sox_scale[im,ih]:.3e} retry={sox_retry}"
            )

            if (not args.allow_unconverged) and (not gw.converged or not sox.converged):
                raise RuntimeError(
                    f"unconverged point mesh={labels[im]}, h_ref={href:g}: "
                    f"GW={gw.final_error:.3e}, GW+SOX={sox.final_error:.3e}"
                )

        gJ0, gslope, gn = _fit(h_ref, Jgw[im], gw_ok[im], args.fit_points)
        sJ0, sslope, sn = _fit(h_ref, Jsox[im], sox_ok[im], args.fit_points)
        print(f"  fit GW:     J0={gJ0:+.9f}, slope={gslope:+.9f}, n={gn}")
        print(f"  fit GW+SOX: J0={sJ0:+.9f}, slope={sslope:+.9f}, n={sn}")
        if np.isfinite(gJ0) and np.isfinite(sJ0):
            print(f"  SOX shift in intercept: dJ0={sJ0-gJ0:+.9f}")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "primitive_finite_source_gw_sox_nk.npz",
        mesh_labels=np.asarray(labels),
        nk1=np.asarray([x[0] for x in meshes]),
        nk2=np.asarray([x[1] for x in meshes]),
        h_reference=h_ref, h_cell=h_cell, reference_ncell=args.reference_ncell,
        V=args.V, filling=args.filling, T=args.T, source=source,
        j_gw=jgw, j_gw_sox=jsox, J_gw=Jgw, J_gw_sox=Jsox,
        gw_converged=gw_ok, gw_sox_converged=sox_ok,
        gw_iterations=gw_iter, gw_sox_iterations=sox_iter,
        gw_residual=gw_res, gw_sox_residual=sox_res,
        gw_smin=gw_smin, gw_sox_smin=sox_smin,
        gw_q1=gw_q1, gw_q2=gw_q2, gw_sox_q1=sox_q1, gw_sox_q2=sox_q2,
        max_sigma_sox=sox_scale, mu_gw=mu_gw, mu_gw_sox=mu_sox,
    )

    with (args.out / "primitive_finite_source_gw_sox_nk.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "mesh","h_reference","h_cell",
            "J_gw","J_gw_sox","j_gw","j_gw_sox",
            "gw_converged","gw_sox_converged","gw_residual","gw_sox_residual",
            "gw_iterations","gw_sox_iterations","gw_smin","gw_sox_smin",
            "gw_q1","gw_q2","gw_sox_q1","gw_sox_q2","max_sigma_sox",
            "mu_gw","mu_gw_sox",
        ])
        for im, label in enumerate(labels):
            for ih, href in enumerate(h_ref):
                w.writerow([
                    label, float(href), float(h_cell[ih]),
                    float(Jgw[im,ih]), float(Jsox[im,ih]),
                    float(jgw[im,ih]), float(jsox[im,ih]),
                    bool(gw_ok[im,ih]), bool(sox_ok[im,ih]),
                    float(gw_res[im,ih]), float(sox_res[im,ih]),
                    int(gw_iter[im,ih]), int(sox_iter[im,ih]),
                    float(gw_smin[im,ih]), float(sox_smin[im,ih]),
                    float(gw_q1[im,ih]), float(gw_q2[im,ih]),
                    float(sox_q1[im,ih]), float(sox_q2[im,ih]),
                    float(sox_scale[im,ih]), float(mu_gw[im,ih]), float(mu_sox[im,ih]),
                ])

    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
