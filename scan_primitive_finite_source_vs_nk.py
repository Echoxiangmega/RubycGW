#!/usr/bin/env python3
"""Dense-k primitive-cell finite-source SC-GW scan.

This script tests whether the large current found on the 2x1 finite torus is a
coarse momentum-transfer artifact.  It solves the translationally invariant
six-orbital primitive-cell SC-GW problem on genuine k/q meshes with

    H0(k;h) = H0(k) - h_cell K_source.

The default mesh list begins with 2x1.  That point has exactly the primitive
momenta/momentum transfers of the old 12-site 2x1 periodic torus, so it is a
useful representation bridge before increasing to 2x2, 4x4, 6x6, ... meshes.

The default source convention is chosen to compare directly with the existing
2x1 ED finite-source benchmark.  There the normalized q=0 operator is

    K_ref = (K_1 + K_2) / sqrt(2),

so an input benchmark field h corresponds to a primitive-cell field

    h_cell = h / sqrt(reference_ncell).

The reported ``J_reference`` is converted back to the same normalized operator,

    J_reference = sqrt(reference_ncell) * j_per_cell.

Set --reference-ncell 1 to use the ordinary primitive-cell convention directly.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark_finite_source_current import _bilinear_expectation_tail_completed
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import canonical_channel_name, primitive_pseudospin_vertex
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast
from rubycgw.supercell_gw_split import compute_sigma_gw_split_components


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--source", choices=["same", "opposite", "z_same", "z_opposite"], default="same")
    p.add_argument(
        "--mesh", nargs="+", default=["2x1", "2x2", "4x4", "6x6", "8x8", "12x12"],
        help="explicit primitive k/q meshes, e.g. --mesh 2x1 2x2 4x4 8x8",
    )
    p.add_argument(
        "--nk", nargs="+", type=int, default=None,
        help="square-mesh shortcut; if supplied, overrides --mesh",
    )
    p.add_argument("--h", nargs="+", type=float,
                   default=[0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0])
    p.add_argument("--reference-ncell", type=int, default=2,
                   help="normalization used for input h and reported J; use 2 to match the 2x1 ED benchmark")
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--gw-max-iter", type=int, default=600)
    p.add_argument("--gw-tol", type=float, default=2e-8)
    p.add_argument("--mixing", type=float, default=0.22)
    p.add_argument("--mixing-method", choices=["linear", "pulay"], default="pulay")
    p.add_argument("--pulay-history", type=int, default=6)
    p.add_argument("--pulay-start", type=int, default=3)
    p.add_argument("--pulay-regularization", type=float, default=1e-10)
    p.add_argument("--backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--fit-points", type=int, default=3,
                   help="smallest positive converged h points used for a diagnostic linear intercept")
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/primitive_finite_source_nk"))
    return p.parse_args()


def _descending_nonnegative(values) -> np.ndarray:
    arr = np.asarray([float(x) for x in values], dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)) or np.any(arr < -1e-15):
        raise ValueError("--h must contain finite non-negative values")
    arr[np.abs(arr) < 1e-15] = 0.0
    return np.asarray(sorted(set(arr.tolist()), reverse=True), dtype=float)


def _parse_mesh_token(token: str) -> tuple[int, int]:
    raw = str(token).strip().lower().replace("×", "x")
    parts = raw.split("x")
    if len(parts) != 2:
        raise ValueError(f"invalid mesh {token!r}; expected form NxM, e.g. 2x1 or 8x8")
    n1, n2 = int(parts[0]), int(parts[1])
    if n1 < 1 or n2 < 1:
        raise ValueError("mesh dimensions must be positive")
    return n1, n2


def _meshes(args) -> list[tuple[int, int]]:
    if args.nk is not None:
        raw = [(int(n), int(n)) for n in args.nk]
    else:
        raw = [_parse_mesh_token(x) for x in args.mesh]
    out = []
    for item in raw:
        if item not in out:
            out.append(item)
    return out


def _current(gw, K, Vq, grid, h0, backend) -> float:
    _, sigma_f, _, _ = compute_sigma_gw_split_components(
        gw.G, gw.W, Vq, grid, h0, gw.mu, gw.Sigma_H, backend=backend
    )
    h_static = h0 + gw.Sigma_H[None, None] + sigma_f
    return float(
        _bilinear_expectation_tail_completed(gw.G, K, grid, h_static, gw.mu).real
    )


def _fit_intercept(h, J, converged, npoints):
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    ok = np.asarray(converged, dtype=bool) & np.isfinite(J) & (h > 0.0)
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        return np.nan, np.nan, 0
    idx = idx[np.argsort(h[idx])][: max(2, int(npoints))]
    slope, intercept = np.polyfit(h[idx], J[idx], 1)
    return float(intercept), float(slope), int(idx.size)


def main():
    args = _args()
    h_reference = _descending_nonnegative(args.h)
    meshes = _meshes(args)
    if args.reference_ncell < 1:
        raise ValueError("--reference-ncell must be positive")
    if args.T <= 0.0:
        raise ValueError("--T must be positive")

    source = canonical_channel_name(args.source)
    K = np.asarray(primitive_pseudospin_vertex(source), dtype=complex)
    ref_root = float(np.sqrt(args.reference_ncell))
    h_cell = h_reference / ref_root
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)

    shape = (len(meshes), len(h_reference))
    j_cell = np.full(shape, np.nan)
    J_reference = np.full(shape, np.nan)
    mu = np.full(shape, np.nan)
    converged = np.zeros(shape, dtype=bool)
    iterations = np.zeros(shape, dtype=int)
    residual = np.full(shape, np.nan)
    screening_smin = np.full(shape, np.nan)
    screening_q1 = np.full(shape, np.nan)
    screening_q2 = np.full(shape, np.nan)
    fit_J0 = np.full(len(meshes), np.nan)
    fit_slope = np.full(len(meshes), np.nan)
    fit_n = np.zeros(len(meshes), dtype=int)

    labels = [f"{n1}x{n2}" for n1, n2 in meshes]
    print("=== primitive dense-k finite-source SC-GW scan ===")
    print(f"V={args.V:g}, filling={args.filling:g}, T={args.T:g}, source={source}")
    print(f"meshes={labels}")
    print(f"reference normalized h={h_reference.tolist()}")
    print(f"reference_ncell={args.reference_ncell}; primitive h_cell=h/sqrt(Nref)")
    print(f"primitive h_cell={h_cell.tolist()}")

    for imesh, (nk1, nk2) in enumerate(meshes):
        label = labels[imesh]
        print(f"\n### mesh={label} ###")
        grid = MatsubaraGrid(
            nk1=nk1, nk2=nk2, nw=args.nw, nOmega=args.nomega, T=args.T
        )
        h0_base = np.asarray(build_h0(grid.kmesh(), params), dtype=complex)
        Vq = np.asarray(build_interaction(grid.qmesh(), params), dtype=complex)
        opts = GWOptions(
            target_filling=args.filling,
            max_iter=args.gw_max_iter,
            tol=args.gw_tol,
            mixing=args.mixing,
            mixing_method=args.mixing_method,
            pulay_history=args.pulay_history,
            pulay_start=args.pulay_start,
            pulay_regularization=args.pulay_regularization,
            verbose=args.verbose,
            momentum_backend=args.backend,
        )

        # Keep the most recent converged point even if an intermediate smaller-h
        # solve fails.  It is still a much better branch seed than restarting from zero.
        last_converged = None
        for ih, (href, hpc) in enumerate(zip(h_reference, h_cell)):
            h0 = h0_base - float(hpc) * K[None, None, :, :]
            h0 = 0.5 * (h0 + np.swapaxes(h0.conj(), -1, -2))
            gw = solve_matrix_gw_fast(h0, Vq, grid, opts=opts, initial=last_converged)

            converged[imesh, ih] = bool(gw.converged)
            iterations[imesh, ih] = int(gw.iterations)
            residual[imesh, ih] = float(gw.final_error)
            mu[imesh, ih] = float(gw.mu)
            screening_smin[imesh, ih] = float(gw.min_screening_singular_value)
            screening_q1[imesh, ih] = float(gw.min_screening_q1)
            screening_q2[imesh, ih] = float(gw.min_screening_q2)

            j = _current(gw, K, Vq, grid, h0, args.backend)
            j_cell[imesh, ih] = j
            J_reference[imesh, ih] = ref_root * j

            status = "OK" if gw.converged else "FAIL"
            print(
                f"  h_ref={href:.8g} h_cell={hpc:.8g} {status:4s} "
                f"iter={gw.iterations:4d} r={gw.final_error:.3e} "
                f"j_cell={j:+.9f} J_ref={J_reference[imesh,ih]:+.9f} "
                f"smin={gw.min_screening_singular_value:.3e} "
                f"q*=({gw.min_screening_q1:.4f},{gw.min_screening_q2:.4f})"
            )

            if gw.converged:
                last_converged = gw
            elif not args.allow_unconverged:
                raise RuntimeError(
                    f"GW failed for mesh={label}, h_ref={href:g}: residual={gw.final_error:.3e}"
                )

        J0, slope, nfit = _fit_intercept(
            h_reference, J_reference[imesh], converged[imesh], args.fit_points
        )
        fit_J0[imesh], fit_slope[imesh], fit_n[imesh] = J0, slope, nfit
        j0 = J0 / ref_root if np.isfinite(J0) else np.nan
        print(
            f"  small-h linear diagnostic ({nfit} points): "
            f"J_ref(h->0)={J0:+.9f}, j_cell(h->0)={j0:+.9f}, slope_ref={slope:+.9f}"
        )

    nk1_values = np.asarray([x[0] for x in meshes], dtype=int)
    nk2_values = np.asarray([x[1] for x in meshes], dtype=int)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "primitive_finite_source_nk.npz",
        mesh_labels=np.asarray(labels), nk1=nk1_values, nk2=nk2_values,
        h_reference=h_reference, h_cell=h_cell,
        reference_ncell=args.reference_ncell, source=source,
        V=args.V, filling=args.filling, T=args.T,
        j_per_cell=j_cell, J_reference=J_reference,
        mu=mu, converged=converged, iterations=iterations, residual=residual,
        screening_smin=screening_smin,
        screening_q1=screening_q1, screening_q2=screening_q2,
        fit_J0_reference=fit_J0,
        fit_J0_per_cell=fit_J0 / ref_root,
        fit_slope_reference=fit_slope, fit_npoints=fit_n,
    )

    with (args.out / "primitive_finite_source_nk.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "mesh", "nk1", "nk2", "h_reference", "h_cell",
            "j_per_cell", "J_reference", "converged", "iterations", "residual",
            "mu", "screening_smin", "screening_q1", "screening_q2",
        ])
        for imesh, (nk1, nk2) in enumerate(meshes):
            for ih, href in enumerate(h_reference):
                w.writerow([
                    labels[imesh], nk1, nk2, float(href), float(h_cell[ih]),
                    float(j_cell[imesh, ih]), float(J_reference[imesh, ih]),
                    bool(converged[imesh, ih]), int(iterations[imesh, ih]),
                    float(residual[imesh, ih]), float(mu[imesh, ih]),
                    float(screening_smin[imesh, ih]),
                    float(screening_q1[imesh, ih]), float(screening_q2[imesh, ih]),
                ])

    with (args.out / "nk_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mesh", "nk1", "nk2", "fit_J0_reference", "fit_J0_per_cell", "fit_slope_reference", "fit_npoints"])
        for imesh, (nk1, nk2) in enumerate(meshes):
            w.writerow([
                labels[imesh], nk1, nk2,
                float(fit_J0[imesh]), float(fit_J0[imesh] / ref_root),
                float(fit_slope[imesh]), int(fit_n[imesh]),
            ])

    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
