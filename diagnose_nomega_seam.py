#!/usr/bin/env python3
"""Diagnose whether a slow supercell-GW residual is tied to the bosonic cutoff seam.

The script first loads the two nearest compatible zero-source checkpoints below
``--V`` and uses the same damped secant predictor as the main continuation
workflow when possible.  It then relaxes that state for a short number of GW
iterations at the requested V.  Finally, keeping the resulting
``Sigma_H``, ``Sigma_GW`` and ``mu`` *fixed*, it evaluates one GW map for several
bosonic cutoffs ``nOmega``.

If the largest residual follows

    n = nOmega  or  n = -nOmega-1,

so that

    |omega| = (2*nOmega+1)*pi*T,

as ``nOmega`` is changed, the plateau is a numerical bosonic-cutoff seam rather
than a physical low-frequency fixed-point mode.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from rubycgw.checkpoint import (
    GWCheckpointSeed,
    find_recent_compatible_checkpoints,
    load_supercell_checkpoint,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.supercell_gw_anderson import AndersonOptions, solve_supercell_gw_anderson
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    compute_sigma_gw_matrix,
    dyson_from_sigma_matrix,
)


def _expected_seam_abs_omega(nOmega: int, T: float) -> float:
    return float((2 * int(nOmega) + 1) * np.pi * float(T))


def _seam_distance(n: int, nOmega: int) -> int:
    """Distance from the two fermionic indices adjacent to the bosonic cutoff."""
    n = int(n)
    nOmega = int(nOmega)
    return min(abs(n - nOmega), abs(n - (-nOmega - 1)))


def _same_branch(meta1: dict, meta2: dict, threshold: float = 1e-4) -> bool:
    p1 = complex(
        float(meta1.get("charge_order_re", 0.0)),
        float(meta1.get("charge_order_im", 0.0)),
    )
    p2 = complex(
        float(meta2.get("charge_order_re", 0.0)),
        float(meta2.get("charge_order_im", 0.0)),
    )
    a1, a2 = abs(p1), abs(p2)
    if (a1 > threshold) != (a2 > threshold):
        return False
    if a1 <= threshold:
        return True
    overlap = (p2 * np.conj(p1)).real / max(a1 * a2, 1e-30)
    return bool(overlap > 0.5)


def _secant_or_latest_seed(
    paths: list[Path],
    params: RubyParameters,
    grid: MatsubaraGrid,
    primitive_filling: float,
    target_V: float,
    damping: float,
    max_ratio: float,
) -> tuple[GWCheckpointSeed, str]:
    if not paths:
        raise FileNotFoundError("No compatible zero-source checkpoint found.")

    seed2, meta2, _ = load_supercell_checkpoint(
        paths[-1], params, grid, primitive_filling
    )
    if len(paths) < 2:
        return seed2, f"latest checkpoint V={float(meta2['V']):g}"

    seed1, meta1, _ = load_supercell_checkpoint(
        paths[-2], params, grid, primitive_filling
    )
    V1 = float(meta1["V"])
    V2 = float(meta2["V"])
    denom = V2 - V1
    if (
        abs(denom) < 1e-14
        or not _same_branch(meta1, meta2)
    ):
        return seed2, f"latest checkpoint V={V2:g}"

    ratio = (float(target_V) - V2) / denom
    if ratio <= 0.0 or ratio > float(max_ratio):
        return seed2, f"latest checkpoint V={V2:g}"

    factor = float(damping) * ratio
    sigma_h = seed2.Sigma_H + factor * (seed2.Sigma_H - seed1.Sigma_H)
    sigma_gw = seed2.Sigma_GW + factor * (seed2.Sigma_GW - seed1.Sigma_GW)
    mu = float(seed2.mu + factor * (seed2.mu - seed1.mu))
    return (
        GWCheckpointSeed(
            Sigma_H=np.asarray(sigma_h, dtype=complex),
            Sigma_GW=np.asarray(sigma_gw, dtype=complex),
            mu=mu,
        ),
        f"secant V={V1:g},{V2:g} -> {target_V:g} (factor={factor:.3f})",
    )


def _max_over_frequency_indices(residual: np.ndarray, indices: list[int]) -> float:
    valid = [i for i in indices if 0 <= int(i) < residual.shape[0]]
    if not valid:
        return float("nan")
    return float(np.max(np.abs(residual[valid])))


def _one_map_scan(state, params: RubyParameters, base_grid: MatsubaraGrid, nOmega: int,
                  backend: str) -> dict:
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
    sigma_out = compute_sigma_gw_matrix(G, W, grid, backend=backend)
    residual = sigma_out - state.Sigma_GW

    abs_res = np.abs(residual)
    flat = int(np.argmax(abs_res))
    iw, ik1, ik2, ia, ib = np.unravel_index(flat, residual.shape)
    n = int(grid.n_values[iw])
    omega = float(grid.omega[iw])
    value = complex(residual[iw, ik1, ik2, ia, ib])

    low_order = np.argsort(np.abs(grid.omega))[: min(4, grid.nf)]
    edge_indices = list(range(min(2, grid.nf))) + list(
        range(max(0, grid.nf - 2), grid.nf)
    )
    seam_iw = []
    for target_n in (int(nOmega), -int(nOmega) - 1):
        matches = np.where(grid.n_values == target_n)[0]
        if matches.size:
            seam_iw.append(int(matches[0]))

    k = grid.kmesh()[ik1, ik2]
    return {
        "nOmega": int(nOmega),
        "raw_residual": float(abs_res.flat[flat]),
        "lowfreq_residual": _max_over_frequency_indices(
            residual, [int(i) for i in low_order]
        ),
        "fermion_edge_residual": _max_over_frequency_indices(residual, edge_indices),
        "seam_residual": _max_over_frequency_indices(residual, seam_iw),
        "max_iw": int(iw),
        "max_n": n,
        "max_omega": omega,
        "max_k1_index": int(ik1),
        "max_k2_index": int(ik2),
        "max_k1": float(k[0]),
        "max_k2": float(k[1]),
        "max_orb_a": int(ia),
        "max_orb_b": int(ib),
        "max_residual_re": float(value.real),
        "max_residual_im": float(value.imag),
        "expected_seam_abs_omega": _expected_seam_abs_omega(nOmega, grid.T),
        "seam_distance_in_n": int(_seam_distance(n, nOmega)),
        "seam_match": bool(_seam_distance(n, nOmega) == 0),
    }


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
    p.add_argument(
        "--base-nomega",
        type=int,
        default=10,
        help="nOmega used by the restart checkpoint and short relaxation.",
    )
    p.add_argument(
        "--scan-nomega",
        nargs="+",
        type=int,
        default=[8, 10, 12, 14],
        help="Bosonic cutoffs for fixed-state one-map scans.",
    )
    p.add_argument("--relax-steps", type=int, default=100)
    p.add_argument("--mu-tol", type=float, default=1e-9)
    p.add_argument("--mu-max-iter", type=int, default=40)
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--checkpoint-dir", default="results/supercell18/checkpoints")
    p.add_argument("--predictor-damping", type=float, default=0.8)
    p.add_argument("--predictor-max-ratio", type=float, default=2.0)
    p.add_argument("--verbose-relax", action="store_true")
    p.add_argument("--out", default="nomega_seam_scan.csv")
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

    print("\nFixed-state nOmega seam scan")
    print(
        " nOmega    raw_res       seam_res      lowfreq       f-edge       "
        "max_n   max_omega     seam?   max(k,a,b)"
    )
    rows = []
    for nOmega in args.scan_nomega:
        row = _one_map_scan(
            state, params, base_grid, int(nOmega), args.momentum_backend
        )
        rows.append(row)
        print(
            f" {row['nOmega']:6d}  {row['raw_residual']:.3e}  "
            f"{row['seam_residual']:.3e}  {row['lowfreq_residual']:.3e}  "
            f"{row['fermion_edge_residual']:.3e}  {row['max_n']:6d}  "
            f"{row['max_omega']:+.6f}   {str(row['seam_match']):5s}   "
            f"({row['max_k1_index']},{row['max_k2_index']},"
            f"{row['max_orb_a']},{row['max_orb_b']})"
        )
        print(
            f"          expected |omega_seam|={row['expected_seam_abs_omega']:.6f}, "
            f"Rmax={row['max_residual_re']:+.3e}{row['max_residual_im']:+.3e}i"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nCSV:", out)

    matches = sum(bool(r["seam_match"]) for r in rows)
    if matches >= max(2, len(rows) - 1):
        print(
            "diagnosis: the maximum residual follows the bosonic cutoff seam; "
            "this strongly supports a finite-nOmega convolution artifact."
        )
    else:
        print(
            "diagnosis: the maximum residual does not consistently follow the seam; "
            "inspect the CSV before changing the GW decomposition."
        )


if __name__ == "__main__":
    main()
