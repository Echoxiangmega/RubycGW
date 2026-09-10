#!/usr/bin/env python3
"""Benchmark primitive-cell CPT against exact finite Ruby tori.

The CPT reference system is one isolated six-site primitive cell containing all
intra-cell interactions exactly.  Its exact Green function is embedded back
onto the L1 x L2 primitive momentum mesh through standard CPT.  The exact
reference is the matching L1 x L2 interacting periodic torus.

For finite current source h_ref the exact torus uses the normalized operator
K_N=(1/sqrt(Ncell))*sum_R K_R.  CPT therefore uses the primitive per-cell source
h_cell=h_ref/sqrt(Ncell), and the reported CPT order parameter is multiplied by
sqrt(Ncell) so that it is directly comparable with ED.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from benchmark_finite_source_current import _bilinear_expectation_tail_completed
from rubycgw.cpt import solve_primitive_cpt_fixed_filling
from rubycgw.ed_green_compare import (
    primitive_local_green,
    relative_green_error,
    solve_exact_finite_source_local_green,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.pseudospin import canonical_channel_name, primitive_pseudospin_vertex


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", nargs="+", type=float, default=[1.0])
    p.add_argument("--h", nargs="+", type=float, default=[0.1])
    p.add_argument("--source", choices=["same", "opposite", "z_same", "z_opposite"], default="same")
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--ed-discard-weight-tol", type=float, default=1e-12)
    p.add_argument("--ed-low-nfreq", type=int, default=8)
    p.add_argument("--cpt-cluster-discard-weight-tol", type=float, default=0.0)
    p.add_argument("--cpt-mu-tol", type=float, default=1e-10)
    p.add_argument("--cpt-mu-max-iter", type=int, default=100)
    p.add_argument("--tail-edge-points", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results/cpt_vs_ed"))
    return p.parse_args()


def _low_indices(omega: np.ndarray, count: int) -> np.ndarray:
    n = min(max(int(count), 1), len(omega))
    return np.asarray(np.argsort(np.abs(np.asarray(omega, dtype=float)))[:n], dtype=int)


def _current_cpt(result, K, grid, ncell: int) -> float:
    h_static = result.h0_lattice + result.Sigma_static_tail[None, None, :, :]
    j_cell = _bilinear_expectation_tail_completed(
        result.G,
        K,
        grid,
        h_static,
        result.mu,
    ).real
    return float(np.sqrt(float(ncell)) * j_cell)


def main():
    args = _args()
    if args.L1 < 1 or args.L2 < 1:
        raise ValueError("L1 and L2 must be positive")
    ncell = int(args.L1) * int(args.L2)
    if 6 * ncell > 16:
        raise ValueError("exact benchmark uses ExactSmallRubyThermal and requires <=16 sites")
    if args.T <= 0.0:
        raise ValueError("T must be positive")

    Vvalues = np.asarray([float(x) for x in args.V], dtype=float)
    hvalues = np.asarray([float(x) for x in args.h], dtype=float)
    if np.any(~np.isfinite(Vvalues)) or np.any(~np.isfinite(hvalues)):
        raise ValueError("V and h values must be finite")
    if np.any(hvalues < -1e-15):
        raise ValueError("this benchmark expects non-negative source h")
    hvalues[np.abs(hvalues) < 1e-15] = 0.0

    source = canonical_channel_name(args.source)
    K = np.asarray(primitive_pseudospin_vertex(source), dtype=complex)
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    grid = MatsubaraGrid(
        nk1=int(args.L1), nk2=int(args.L2), nw=int(args.nw),
        nOmega=int(args.nomega), T=float(args.T)
    )
    low_idx = _low_indices(grid.omega, args.ed_low_nfreq)

    shape = (len(Vvalues), len(hvalues))
    J_ed = np.full(shape, np.nan)
    J_cpt = np.full(shape, np.nan)
    Gerr = np.full(shape, np.nan)
    Gerr_low = np.full(shape, np.nan)
    mu_ed = np.full(shape, np.nan)
    mu_cpt = np.full(shape, np.nan)
    filling_cpt = np.full(shape, np.nan)
    cluster_filling = np.full(shape, np.nan)
    mu_residual = np.full(shape, np.nan)
    mu_iterations = np.zeros(shape, dtype=int)
    sigma_static_norm = np.full(shape, np.nan)

    G_ed_local = np.full((len(Vvalues), len(hvalues), grid.nf, 6, 6), np.nan + 0j, dtype=complex)
    G_cpt_local = np.full_like(G_ed_local, np.nan + 0j)
    G_cpt_k = np.full(
        (len(Vvalues), len(hvalues), grid.nf, grid.nk1, grid.nk2, 6, 6),
        np.nan + 0j,
        dtype=complex,
    )

    print("=== CPT / exact-torus benchmark ===")
    print(
        f"ED torus={args.L1}x{args.L2} ({6*ncell} sites), "
        f"CPT cluster=1x1 isolated primitive cell (6 sites)"
    )
    print(f"T={args.T:g}, filling={args.filling:g}, source={source}")
    print(f"V={Vvalues.tolist()}, h_ref={hvalues.tolist()}")
    print("CPT: exact cluster V, no many-body self-consistency; only mu is solved at fixed filling")

    rows = []
    for iv, V in enumerate(Vvalues):
        for ih, href in enumerate(hvalues):
            hcell = float(href) / np.sqrt(float(ncell))
            print(f"\n-- V={V:g}, h_ref={href:.8g}, h_cell={hcell:.8g} --")

            ed = solve_exact_finite_source_local_green(
                L1=int(args.L1), L2=int(args.L2), params=params, V=float(V),
                source_channel=source, h_ref=float(href),
                filling_per_cell=float(args.filling), T=float(args.T),
                omega=grid.omega,
                discard_weight_tol=float(args.ed_discard_weight_tol),
            )
            cpt = solve_primitive_cpt_fixed_filling(
                params,
                grid,
                V=float(V),
                filling=float(args.filling),
                source_vertex=K,
                source_per_cell=hcell,
                mu0=float(ed.mu),
                mu_tol=float(args.cpt_mu_tol),
                mu_max_iter=int(args.cpt_mu_max_iter),
                discard_weight_tol=float(args.cpt_cluster_discard_weight_tol),
                tail_edge_points=int(args.tail_edge_points),
            )

            Gloc = primitive_local_green(cpt.G)
            J_ed[iv, ih] = float(ed.J_ref)
            J_cpt[iv, ih] = _current_cpt(cpt, K, grid, ncell)
            G_ed_local[iv, ih] = ed.G_local
            G_cpt_local[iv, ih] = Gloc
            G_cpt_k[iv, ih] = cpt.G
            Gerr[iv, ih] = relative_green_error(Gloc, ed.G_local)
            Gerr_low[iv, ih] = relative_green_error(Gloc[low_idx], ed.G_local[low_idx])
            mu_ed[iv, ih] = float(ed.mu)
            mu_cpt[iv, ih] = float(cpt.mu)
            filling_cpt[iv, ih] = float(cpt.filling)
            cluster_filling[iv, ih] = float(cpt.cluster_filling)
            mu_residual[iv, ih] = float(cpt.mu_residual)
            mu_iterations[iv, ih] = int(cpt.mu_iterations)
            sigma_static_norm[iv, ih] = float(np.max(np.abs(cpt.Sigma_static_tail)))

            reconstruct = cpt.h_cluster[None, None, :, :] + cpt.intercluster_hopping
            reconstruct_err = float(np.max(np.abs(reconstruct - cpt.h0_lattice)))
            print(
                f"    ED:  mu={ed.mu:+.9f} J={ed.J_ref:+.9f}"
            )
            print(
                f"    CPT: mu={cpt.mu:+.9f} J={J_cpt[iv, ih]:+.9f} "
                f"Jerr={J_cpt[iv, ih]-J_ed[iv, ih]:+.9f}"
            )
            print(
                f"         Gerr={Gerr[iv, ih]:.6e}, Gerr_low={Gerr_low[iv, ih]:.6e}, "
                f"filling={cpt.filling:.10f}, cluster_n={cpt.cluster_filling:.8f}"
            )
            print(
                f"         mu_iter={cpt.mu_iterations}, mu_res={cpt.mu_residual:.3e}, "
                f"max|Sigma_inf|={sigma_static_norm[iv, ih]:.3e}, "
                f"h0_reconstruct_err={reconstruct_err:.3e}"
            )

            rows.append({
                "V": float(V), "h_ref": float(href), "h_cell": hcell,
                "J_ed": J_ed[iv, ih], "J_cpt": J_cpt[iv, ih],
                "Jerr_cpt": J_cpt[iv, ih] - J_ed[iv, ih],
                "Gerr_cpt": Gerr[iv, ih], "Gerr_cpt_low": Gerr_low[iv, ih],
                "mu_ed": mu_ed[iv, ih], "mu_cpt": mu_cpt[iv, ih],
                "filling_cpt": filling_cpt[iv, ih],
                "cluster_filling": cluster_filling[iv, ih],
                "mu_residual": mu_residual[iv, ih],
                "mu_iterations": mu_iterations[iv, ih],
                "max_sigma_static_tail": sigma_static_norm[iv, ih],
                "h0_reconstruct_err": reconstruct_err,
            })

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"CPT_ED_{args.L1}x{args.L2}_fill{args.filling:g}_{source}"
    csv_path = args.out / f"{stem}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    npz_path = args.out / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        V=Vvalues, h_ref=hvalues, omega=grid.omega,
        J_ed=J_ed, J_cpt=J_cpt,
        Gerr_cpt=Gerr, Gerr_cpt_low=Gerr_low,
        mu_ed=mu_ed, mu_cpt=mu_cpt,
        filling_cpt=filling_cpt, cluster_filling=cluster_filling,
        mu_residual=mu_residual, mu_iterations=mu_iterations,
        max_sigma_static_tail=sigma_static_norm,
        G_ed_local=G_ed_local, G_cpt_local=G_cpt_local, G_cpt_k=G_cpt_k,
        low_frequency_indices=low_idx,
        L1=int(args.L1), L2=int(args.L2), filling=float(args.filling),
        T=float(args.T), source=source,
    )
    print(f"\nwrote {csv_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
