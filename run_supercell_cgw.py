#!/usr/bin/env python3
"""Compute the 18-site current susceptibility matrix and R=chi^{-1}.

The script loads a converged zero-source supercell SC-GW checkpoint at exactly
the requested V, verifies that it is a fixed point of the current static-Fock +
dynamic-(W-V) self-energy map, reconstructs G,P,W, and then solves six q_sc=0
current vertices:

    opposite_s0, opposite_s1, opposite_s2,
    same_s0,     same_s1,     same_s2.

The resulting 6x6 susceptibility is transformed to the real harmonic basis
(q0,Qc,Qs) in each physical current channel and inverted to obtain the Landau
curvature matrix R.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from rubycgw.checkpoint import (
    checkpoint_filename,
    load_supercell_checkpoint,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_cgw import (
    SupercellVertexOptions,
    current_harmonic_transform,
    curvature_from_susceptibility,
    physical_symmetric_susceptibility,
    solve_vertex_q0,
    supercell_current_vertices,
    susceptibility_matrix_q0,
)
from rubycgw.supercell_gw import (
    compute_polarization_matrix,
    compute_screened_interaction_matrix,
    density_from_G_matrix,
    dyson_from_sigma_matrix,
    hartree_self_energy_matrix,
)
from rubycgw.supercell_gw_split import compute_sigma_gw_split_matrix


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", type=float, required=True)
    p.add_argument("--primitive-filling", type=float, default=3.0)
    p.add_argument("--T", type=float, default=0.05)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nk1", type=int, default=3)
    p.add_argument("--nk2", type=int, default=3)
    p.add_argument("--nw", type=int, default=47)
    p.add_argument("--nomega", type=int, default=10)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-dir", default="results/supercell18/checkpoints")
    p.add_argument(
        "--stage",
        choices=["gg", "split-mt", "full"],
        default="full",
        help="gg: dressed bubble only; split-mt: H+F+MT(W-V); full: add AL1/AL2.",
    )
    p.add_argument("--vertex-max-iter", type=int, default=150)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-mixing", type=float, default=0.25)
    p.add_argument("--vertex-verbose", action="store_true")
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument(
        "--max-scgw-residual",
        type=float,
        default=1e-6,
        help="Reject checkpoints that are not fixed points of the split SC-GW map.",
    )
    p.add_argument("--out-prefix", default="supercell_cgw")
    return p.parse_args()


def _exact_checkpoint(args, params, grid):
    if args.checkpoint is not None:
        path = Path(args.checkpoint)
    else:
        path = Path(args.checkpoint_dir) / checkpoint_filename(
            args.V, args.primitive_filling, grid
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Exact V={args.V:g} checkpoint not found: {path}. "
            "Run and converge SC-GW at this V first."
        )
    seed, meta, density_saved = load_supercell_checkpoint(
        path, params, grid, args.primitive_filling
    )
    if abs(float(meta["V"]) - float(args.V)) > 1e-12:
        raise ValueError(
            f"cGW response requires the same physical V: checkpoint has "
            f"V={meta['V']}, requested V={args.V}."
        )
    if not bool(meta.get("converged", False)):
        raise ValueError("cGW response requires a converged SC-GW checkpoint")
    if abs(float(meta.get("source", 0.0))) > 1e-14:
        raise ValueError("cGW response requires a zero-source SC-GW checkpoint")
    return path, seed, meta, density_saved


def _verify_and_rebuild(seed, params, grid, backend):
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    G = dyson_from_sigma_matrix(h0, grid, seed.mu, seed.Sigma_H, seed.Sigma_GW)
    density = density_from_G_matrix(
        G, grid, h0=h0, mu=seed.mu, sigma_h=seed.Sigma_H
    )
    sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_split_matrix(
        G, W, Vq, grid, h0, seed.mu, seed.Sigma_H, backend=backend
    )
    rH = float(np.max(np.abs(sigma_h_out - seed.Sigma_H)))
    rGW = float(np.max(np.abs(sigma_gw_out - seed.Sigma_GW)))
    return h0, Vq, G, P, W, density, rH, rGW


def _format_matrix(mat, fmt="+.6e"):
    arr = np.asarray(mat)
    rows = []
    for row in arr:
        rows.append("  " + " ".join(format(float(x), fmt) for x in row))
    return "\n".join(rows)


def main():
    args = _parse_args()
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(
        nk1=args.nk1,
        nk2=args.nk2,
        nw=args.nw,
        nOmega=args.nomega,
        T=args.T,
    )

    checkpoint, seed, meta, density_saved = _exact_checkpoint(args, params, grid)
    print("checkpoint:", checkpoint)
    print(
        f"SC-GW metadata: V={float(meta['V']):g}, mu={float(meta['mu']):.10f}, "
        f"|Phi|={float(meta.get('charge_order_abs', np.nan)):.6e}, "
        f"stored_residual={float(meta.get('final_error', np.nan)):.3e}"
    )

    h0, Vq, G, P, W, density, rH, rGW = _verify_and_rebuild(
        seed, params, grid, args.momentum_backend
    )
    sc_res = max(rH, rGW)
    print(
        f"split-map checkpoint verification: rH={rH:.3e}, rGW={rGW:.3e}, "
        f"max={sc_res:.3e}, n={np.sum(density):.10f}"
    )
    if sc_res > float(args.max_scgw_residual):
        raise RuntimeError(
            "Checkpoint is not a fixed point of the current static-Fock + W-V "
            f"SC-GW map (residual={sc_res:.3e}). Do not compute r from it."
        )

    Klocal, local_labels = supercell_current_vertices()
    bare_gammas = [np.broadcast_to(K, G.shape).copy() for K in Klocal]
    chi_gg = susceptibility_matrix_q0(G, Klocal, bare_gammas, grid)

    vertex_results = []
    if args.stage == "gg":
        gammas = bare_gammas
    else:
        include_al = args.stage == "full"
        vopts = SupercellVertexOptions(
            max_iter=args.vertex_max_iter,
            tol=args.vertex_tol,
            mixing=args.vertex_mixing,
            include_hartree=True,
            include_fock=True,
            include_mt=True,
            include_al=include_al,
            verbose=args.vertex_verbose,
            momentum_backend=args.momentum_backend,
        )
        gammas = []
        for ich, (label, K) in enumerate(zip(local_labels, Klocal), start=1):
            print(f"\n--- vertex {ich}/6: {label} ({args.stage}) ---")
            result = solve_vertex_q0(G, W, Vq, K, grid, opts=vopts)
            vertex_results.append(result)
            gammas.append(result.Gamma)
            norms = {
                "H": np.max(np.abs(result.Gamma_H)),
                "F": np.max(np.abs(result.Gamma_F)),
                "MTc": np.max(np.abs(result.Gamma_MT)),
                "AL1": np.max(np.abs(result.Gamma_AL1)),
                "AL2": np.max(np.abs(result.Gamma_AL2)),
            }
            print(
                f"{label}: converged={result.converged}, it={result.iterations}, "
                f"err={result.final_error:.3e}, "
                + ", ".join(f"|{k}|max={v:.3e}" for k, v in norms.items())
            )
        bad = [
            local_labels[i]
            for i, result in enumerate(vertex_results)
            if not result.converged
        ]
        if bad:
            raise RuntimeError(
                "Vertex solve did not converge for: " + ", ".join(bad)
                + ". Increase --vertex-max-iter or adjust --vertex-mixing before trusting r."
            )

    chi = susceptibility_matrix_q0(G, Klocal, gammas, grid)
    analysis = curvature_from_susceptibility(chi)
    gg_sym, gg_imag = physical_symmetric_susceptibility(chi_gg)
    Tmat, harmonic_labels = current_harmonic_transform()
    chi_gg_harm = Tmat @ gg_sym @ Tmat.T

    print("\nLocal basis:", ", ".join(local_labels))
    print("Harmonic basis:", ", ".join(harmonic_labels))
    print(f"max discarded Im(chi) = {analysis['chi_imag_max']:.3e}")
    print("\nchi_cGW in harmonic basis:")
    print(_format_matrix(analysis["chi_harmonic"]))
    print("\nR=chi^{-1} in harmonic basis:")
    print(_format_matrix(analysis["R_harmonic"]))

    print("\nUniform q=0 block [opposite, same]")
    print("chi_uniform:")
    print(_format_matrix(analysis["chi_uniform"]))
    print("R_uniform_relaxed = inv(chi_uniform):")
    print(_format_matrix(analysis["R_uniform_relaxed"]))
    print("R_uniform_constrained = full-R q0 block:")
    print(_format_matrix(analysis["R_uniform_constrained"]))

    rvals = analysis["R_eigenvalues"]
    soft_local = analysis["soft_vector_local"]
    soft_harm = analysis["soft_vector_harmonic"]
    print("\nFull 6x6 curvature eigenvalues:")
    print("  " + " ".join(f"{x:+.8e}" for x in rvals))
    print(f"softest r = {rvals[0]:+.10e}")
    print(
        "soft-mode weights: "
        f"opposite={analysis['soft_weight_opposite']:.6f}, "
        f"same={analysis['soft_weight_same']:.6f}, "
        f"q0={analysis['soft_weight_q0']:.6f}, "
        f"Q={analysis['soft_weight_Q']:.6f}"
    )
    print("soft vector local:")
    for label, x in zip(local_labels, soft_local):
        print(f"  {label:16s} {x:+.8f}")
    print("soft vector harmonic:")
    for label, x in zip(harmonic_labels, soft_harm):
        print(f"  {label:16s} {x:+.8f}")

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        V=float(args.V),
        primitive_filling=float(args.primitive_filling),
        stage=np.asarray(args.stage),
        local_labels=np.asarray(local_labels),
        harmonic_labels=np.asarray(harmonic_labels),
        density=np.asarray(density),
        chi_gg=np.asarray(chi_gg),
        chi_gg_harmonic=np.asarray(chi_gg_harm),
        chi_raw=np.asarray(chi),
        chi_symmetric=np.asarray(analysis["chi_symmetric"]),
        chi_harmonic=np.asarray(analysis["chi_harmonic"]),
        R=np.asarray(analysis["R"]),
        R_harmonic=np.asarray(analysis["R_harmonic"]),
        R_eigenvalues=np.asarray(analysis["R_eigenvalues"]),
        R_eigenvectors=np.asarray(analysis["R_eigenvectors"]),
        chi_uniform=np.asarray(analysis["chi_uniform"]),
        R_uniform_relaxed=np.asarray(analysis["R_uniform_relaxed"]),
        R_uniform_constrained=np.asarray(analysis["R_uniform_constrained"]),
        soft_vector_local=np.asarray(soft_local),
        soft_vector_harmonic=np.asarray(soft_harm),
    )

    csv_path = prefix.with_name(prefix.name + "_summary.csv")
    Ru = analysis["R_uniform_relaxed"]
    Rc = analysis["R_uniform_constrained"]
    row = {
        "V": float(args.V),
        "primitive_filling": float(args.primitive_filling),
        "stage": args.stage,
        "scgw_residual": sc_res,
        "chi_imag_max": analysis["chi_imag_max"],
        "r_soft": float(rvals[0]),
        "r_uniform_opposite_relaxed": float(Ru[0, 0]),
        "r_uniform_same_relaxed": float(Ru[1, 1]),
        "r_uniform_mix_relaxed": float(Ru[0, 1]),
        "r_uniform_opposite_constrained": float(Rc[0, 0]),
        "r_uniform_same_constrained": float(Rc[1, 1]),
        "r_uniform_mix_constrained": float(Rc[0, 1]),
        "soft_weight_opposite": analysis["soft_weight_opposite"],
        "soft_weight_same": analysis["soft_weight_same"],
        "soft_weight_q0": analysis["soft_weight_q0"],
        "soft_weight_Q": analysis["soft_weight_Q"],
    }
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print("\nNPZ:", npz_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
