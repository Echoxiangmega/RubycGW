#!/usr/bin/env python3
"""Diagnose which part of W causes strong nOmega dependence in supercell GW.

The script keeps one fermionic state (Sigma_H, Sigma_GW, mu) fixed, constructs
G from that state, and for several bosonic cutoffs decomposes

    W = V + Wc,   Wc = W - V,

so that the finite-window GW map is written as

    Sigma_full = Sigma_bare,trunc + Sigma_corr,trunc.

For each nOmega it compares the resulting map with the map at --base-nomega.
If the cutoff variation is dominated by Sigma_bare,trunc while the Wc part is
much less sensitive, the instantaneous bare interaction is being truncated by
the bosonic Matsubara window and should be treated separately (static Fock +
dynamic W-V convolution).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from diagnose_nomega_seam import _secant_or_latest_seed
from rubycgw.checkpoint import find_recent_compatible_checkpoints
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    compute_sigma_gw_matrix,
    dyson_from_sigma_matrix,
)
from rubycgw.supercell_gw_anderson import AndersonOptions, solve_supercell_gw_anderson


def _maxabs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def _map_components(state, params, base_grid, nOmega: int, backend: str):
    grid = MatsubaraGrid(
        nk1=base_grid.nk1,
        nk2=base_grid.nk2,
        nw=base_grid.nw,
        nOmega=int(nOmega),
        T=base_grid.T,
    )
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    G = dyson_from_sigma_matrix(h0, grid, state.mu, state.Sigma_H, state.Sigma_GW)
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)

    Vinst = np.broadcast_to(Vq[None, :, :, :, :], W.shape)
    Wcorr = W - Vinst
    sigma_full = compute_sigma_gw_matrix(G, W, grid, backend=backend)
    sigma_bare = compute_sigma_gw_matrix(G, Vinst, grid, backend=backend)
    sigma_corr = compute_sigma_gw_matrix(G, Wcorr, grid, backend=backend)

    split_error = _maxabs(sigma_full - sigma_bare - sigma_corr)
    raw_residual = _maxabs(sigma_full - state.Sigma_GW)
    return {
        "nOmega": int(nOmega),
        "raw_residual": raw_residual,
        "sigma_full_norm": _maxabs(sigma_full),
        "sigma_bare_trunc_norm": _maxabs(sigma_bare),
        "sigma_corr_trunc_norm": _maxabs(sigma_corr),
        "split_identity_error": split_error,
    }, sigma_full, sigma_bare, sigma_corr


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", type=float, required=True)
    p.add_argument("--T", type=float, default=0.05)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--primitive-filling", type=float, default=3.0)
    p.add_argument("--nk1", type=int, default=3)
    p.add_argument("--nk2", type=int, default=3)
    p.add_argument("--nw", type=int, default=47)
    p.add_argument("--base-nomega", type=int, default=10)
    p.add_argument("--scan-nomega", nargs="+", type=int, default=[8, 10, 12, 14])
    p.add_argument("--relax-steps", type=int, default=100)
    p.add_argument("--mu-tol", type=float, default=1e-9)
    p.add_argument("--mu-max-iter", type=int, default=40)
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--checkpoint-dir", default="results/supercell18/checkpoints")
    p.add_argument("--predictor-damping", type=float, default=0.8)
    p.add_argument("--predictor-max-ratio", type=float, default=2.0)
    p.add_argument("--verbose-relax", action="store_true")
    p.add_argument("--out", default="w_split_cutoff_scan.csv")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.relax_steps < 0:
        raise ValueError("--relax-steps must be non-negative")
    if args.base_nomega < 0 or any(x < 0 for x in args.scan_nomega):
        raise ValueError("nOmega values must be non-negative")

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    base_grid = MatsubaraGrid(
        nk1=args.nk1,
        nk2=args.nk2,
        nw=args.nw,
        nOmega=args.base_nomega,
        T=args.T,
    )
    paths = find_recent_compatible_checkpoints(
        args.checkpoint_dir,
        args.V,
        params,
        base_grid,
        args.primitive_filling,
        limit=2,
    )
    if not paths:
        raise FileNotFoundError(
            "No compatible checkpoint found. Check --checkpoint-dir and numerical grid."
        )

    seed, seed_label = _secant_or_latest_seed(
        paths,
        params,
        base_grid,
        args.primitive_filling,
        args.V,
        args.predictor_damping,
        args.predictor_max_ratio,
    )
    print("restart:", " -> ".join(str(p) for p in paths))
    print("seed:", seed_label)

    if args.relax_steps > 0:
        opts = GWOptions(
            mu=float(seed.mu),
            target_filling=3.0 * float(args.primitive_filling),
            max_iter=int(args.relax_steps),
            tol=1e-30,
            mixing=0.7,
            mixing_method="anderson",
            mu_tol=float(args.mu_tol),
            mu_max_iter=int(args.mu_max_iter),
            verbose=bool(args.verbose_relax),
            momentum_backend=args.momentum_backend,
        )
        state = solve_supercell_gw_anderson(
            params,
            base_grid,
            opts=opts,
            source_strength=0.0,
            initial=seed,
            anderson=AndersonOptions(),
        )
        print(
            f"relaxed {state.iterations} step(s) at base nOmega={args.base_nomega}: "
            f"residual={state.final_error:.3e}, mu={state.mu:.10f}"
        )
    else:
        state = seed
        print("relaxation skipped; scanning the restart seed directly")

    scan_values = list(dict.fromkeys([int(args.base_nomega)] + [int(x) for x in args.scan_nomega]))
    maps = {}
    rows_by_no = {}
    for no in scan_values:
        row, full, bare, corr = _map_components(
            state, params, base_grid, no, args.momentum_backend
        )
        rows_by_no[no] = row
        maps[no] = (full, bare, corr)

    full0, bare0, corr0 = maps[int(args.base_nomega)]
    rows = []
    for no in [int(x) for x in args.scan_nomega]:
        row = dict(rows_by_no[no])
        full, bare, corr = maps[no]
        d_full = _maxabs(full - full0)
        d_bare = _maxabs(bare - bare0)
        d_corr = _maxabs(corr - corr0)
        row.update(
            {
                "delta_full_vs_base": d_full,
                "delta_bare_vs_base": d_bare,
                "delta_corr_vs_base": d_corr,
                "bare_to_full_cutoff_ratio": d_bare / max(d_full, 1e-300),
                "corr_to_full_cutoff_ratio": d_corr / max(d_full, 1e-300),
            }
        )
        rows.append(row)

    print("\nFixed-state W=V+(W-V) cutoff scan")
    print(
        " nOmega   raw_res    dFull(base)  dBare(base)  dCorr(base)  "
        "bare/full  corr/full  split_err"
    )
    for r in rows:
        print(
            f" {r['nOmega']:6d}  {r['raw_residual']:.3e}  "
            f"{r['delta_full_vs_base']:.3e}  {r['delta_bare_vs_base']:.3e}  "
            f"{r['delta_corr_vs_base']:.3e}  {r['bare_to_full_cutoff_ratio']:.3f}      "
            f"{r['corr_to_full_cutoff_ratio']:.3f}      {r['split_identity_error']:.1e}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nCSV:", out)

    nonbase = [r for r in rows if r["nOmega"] != int(args.base_nomega)]
    if nonbase:
        bare_dom = sum(
            r["delta_bare_vs_base"] > 3.0 * r["delta_corr_vs_base"]
            for r in nonbase
        )
        if bare_dom >= max(1, len(nonbase) - 1):
            print(
                "diagnosis: cutoff dependence is dominated by the finite-window bare-V "
                "convolution. This supports replacing it by a static Fock term and "
                "summing only W-V over bosonic Matsubara frequencies."
            )
        else:
            print(
                "diagnosis: W-V contributes substantially to the cutoff dependence too; "
                "do not change the production GW decomposition yet."
            )


if __name__ == "__main__":
    main()
