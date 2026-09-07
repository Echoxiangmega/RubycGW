#!/usr/bin/env python3
"""Compute cGW susceptibilities for Ruby chirality-pseudospin channels.

Examples
--------
Diagonal TR-even orbital/charge susceptibility at primitive q=0::

    python analyze_pseudospin_susceptibility.py --V 1.4 --primitive-filling 2 \
        --chi x_even,x_even --harmonic q0

Loop-current pseudospin susceptibility (physical same circulation)::

    python analyze_pseudospin_susceptibility.py --V 1.4 --primitive-filling 2 \
        --chi z_same,z_same --harmonic q0

Cross susceptibility::

    python analyze_pseudospin_susceptibility.py --V 1.4 --primitive-filling 2 \
        --chi x_even,y_even --harmonic q0

Or request a full matrix among several channels::

    python analyze_pseudospin_susceptibility.py --V 1.4 --primitive-filling 2 \
        --channels x_even y_even z_same z_opposite --harmonic q0

The x/y vertices are the TR-even E-type intra-triangle charge/orbital order
parameters.  z is the TR-odd loop chirality.  z is normalized as a Pauli
pseudospin, so chi_zz is 1/3 of the legacy eta-current susceptibility for the
same physical channel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.checkpoint import checkpoint_filename, load_supercell_checkpoint
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.pseudospin import (
    available_pseudospin_channels,
    canonical_channel_name,
    harmonic_block_indices,
    pseudospin_harmonic_labels,
    pseudospin_harmonic_transform,
    supercell_pseudospin_vertices,
)
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_cgw import (
    SupercellVertexOptions,
    physical_symmetric_susceptibility,
    solve_vertex_q0,
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
    p.add_argument("--primitive-filling", type=float, default=2.0)
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
        "--chi",
        default=None,
        help=(
            "One requested pair, e.g. x_even,x_even or z_same,z_opposite. "
            "The required channels are solved automatically."
        ),
    )
    p.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help=(
            "Channels for a full susceptibility matrix. Examples: Ax Ay Az, "
            "x_even y_even z_same."
        ),
    )
    p.add_argument(
        "--list-channels",
        action="store_true",
        help="Print available channel names and exit.",
    )
    p.add_argument(
        "--harmonic",
        choices=["q0", "Qc", "Qs", "all"],
        default="q0",
        help="Primitive-cell harmonic folded into supercell q_sc=0.",
    )
    p.add_argument(
        "--stage",
        choices=["gg", "split-mt", "full"],
        default="full",
        help="gg=bubble; split-mt=H+F+MT; full=add AL1/AL2.",
    )
    p.add_argument("--vertex-max-iter", type=int, default=150)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-solver", choices=["gmres", "linear"], default="gmres")
    p.add_argument("--vertex-gmres-restart", type=int, default=12)
    p.add_argument("--vertex-mixing", type=float, default=0.25)
    p.add_argument("--vertex-verbose", action="store_true")
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--max-scgw-residual", type=float, default=1e-6)
    p.add_argument("--out", default="pseudospin_susceptibility.npz")
    return p.parse_args()


def _select_channels(args):
    if args.chi and args.channels:
        raise ValueError("use either --chi or --channels, not both")
    requested_pair = None
    if args.chi:
        fields = [x.strip() for x in str(args.chi).split(",") if x.strip()]
        if len(fields) != 2:
            raise ValueError("--chi expects exactly two comma-separated channel names")
        left, right = [canonical_channel_name(x) for x in fields]
        requested_pair = (left, right)
        channels = []
        for ch in (left, right):
            if ch not in channels:
                channels.append(ch)
    elif args.channels:
        channels = []
        for x in args.channels:
            ch = canonical_channel_name(x)
            if ch not in channels:
                channels.append(ch)
    else:
        channels = ["x_even", "y_even", "z_same", "z_opposite"]
    return channels, requested_pair


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
            "Run and converge zero-source SC-GW at this point first."
        )
    seed, meta, density_saved = load_supercell_checkpoint(
        path, params, grid, args.primitive_filling
    )
    if abs(float(meta["V"]) - float(args.V)) > 1e-12:
        raise ValueError("checkpoint V does not match requested V")
    if not bool(meta.get("converged", False)):
        raise ValueError("pseudospin cGW requires a converged SC-GW checkpoint")
    if abs(float(meta.get("source", 0.0))) > 1e-14:
        raise ValueError("pseudospin cGW requires a zero-source SC-GW checkpoint")
    return path, seed, meta, density_saved


def _verify_and_rebuild(seed, params, grid, backend):
    h0 = build_supercell_h0(grid.kmesh(), params, source_strength=0.0)
    Vq = build_supercell_interaction(grid.qmesh(), params)
    G = dyson_from_sigma_matrix(h0, grid, seed.mu, seed.Sigma_H, seed.Sigma_GW)
    density = density_from_G_matrix(G, grid, h0=h0, mu=seed.mu, sigma_h=seed.Sigma_H)
    sigma_h_out = hartree_self_energy_matrix(density, Vq[0, 0])
    P = compute_polarization_matrix(G, grid, backend=backend)
    W = compute_screened_interaction_matrix(P, Vq)
    sigma_gw_out = compute_sigma_gw_split_matrix(
        G, W, Vq, grid, h0, seed.mu, seed.Sigma_H, backend=backend
    )
    rH = float(np.max(np.abs(sigma_h_out - seed.Sigma_H)))
    rGW = float(np.max(np.abs(sigma_gw_out - seed.Sigma_GW)))
    return h0, Vq, G, P, W, density, rH, rGW


def _format_matrix(mat):
    arr = np.asarray(mat)
    return "\n".join(
        "  " + " ".join(f"{float(x):+.7e}" for x in row)
        for row in arr
    )


def main():
    args = _parse_args()
    if args.list_channels:
        print("Available pseudospin channels:")
        for ch in available_pseudospin_channels():
            print(" ", ch)
        print("Aliases: same -> z_same, opposite -> z_opposite")
        return

    channels, requested_pair = _select_channels(args)
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(
        nk1=args.nk1, nk2=args.nk2, nw=args.nw,
        nOmega=args.nomega, T=args.T,
    )

    checkpoint, seed, meta, _ = _exact_checkpoint(args, params, grid)
    print("checkpoint:", checkpoint)
    print("channels:", ", ".join(channels))
    print(
        "normalization: x,y,z all project to Pauli pseudospins; "
        "z susceptibility = legacy eta-current susceptibility / 3"
    )

    h0, Vq, G, P, W, density, rH, rGW = _verify_and_rebuild(
        seed, params, grid, args.momentum_backend
    )
    sc_res = max(rH, rGW)
    print(
        f"split-map verification: rH={rH:.3e}, rGW={rGW:.3e}, "
        f"max={sc_res:.3e}, n={np.sum(density):.10f}"
    )
    if sc_res > float(args.max_scgw_residual):
        raise RuntimeError(
            f"checkpoint is not a fixed point of the current SC-GW map: {sc_res:.3e}"
        )

    Klocal, local_labels, canonical = supercell_pseudospin_vertices(channels)
    bare = [np.broadcast_to(K, G.shape).copy() for K in Klocal]
    chi_gg = susceptibility_matrix_q0(G, Klocal, bare, grid)

    if args.stage == "gg":
        gammas = bare
        vertex_results = []
    else:
        vopts = SupercellVertexOptions(
            max_iter=args.vertex_max_iter,
            tol=args.vertex_tol,
            mixing=args.vertex_mixing,
            solver=args.vertex_solver,
            gmres_restart=args.vertex_gmres_restart,
            include_hartree=True,
            include_fock=True,
            include_mt=True,
            include_al=(args.stage == "full"),
            verbose=args.vertex_verbose,
            momentum_backend=args.momentum_backend,
        )
        gammas = []
        vertex_results = []
        for i, (label, K) in enumerate(zip(local_labels, Klocal), start=1):
            print(f"--- vertex {i}/{len(local_labels)}: {label} ({args.stage}) ---")
            result = solve_vertex_q0(G, W, Vq, K, grid, opts=vopts)
            vertex_results.append(result)
            gammas.append(result.Gamma)
            print(
                f"{label}: converged={result.converged}, it={result.iterations}, "
                f"residual={result.final_error:.3e}"
            )
        bad = [
            local_labels[i] for i, r in enumerate(vertex_results) if not r.converged
        ]
        if bad:
            raise RuntimeError("vertex solve did not converge for: " + ", ".join(bad))

    chi_raw = susceptibility_matrix_q0(G, Klocal, gammas, grid)
    chi_sym, imag_max = physical_symmetric_susceptibility(chi_raw)
    chi_gg_sym, _ = physical_symmetric_susceptibility(chi_gg)

    Tmat = pseudospin_harmonic_transform(len(canonical))
    harm_labels = pseudospin_harmonic_labels(canonical)
    chi_harm = Tmat @ chi_sym @ Tmat.T
    chi_gg_harm = Tmat @ chi_gg_sym @ Tmat.T

    print(f"max discarded Im(chi)={imag_max:.3e}")
    harmonics = ["q0", "Qc", "Qs"] if args.harmonic == "all" else [args.harmonic]
    blocks = {}
    for harm in harmonics:
        idx = harmonic_block_indices(len(canonical), harm)
        block = chi_harm[np.ix_(idx, idx)]
        blocks[harm] = block
        print(f"\nchi_{harm} basis [{', '.join(canonical)}]:")
        print(_format_matrix(block))

    if requested_pair is not None:
        il = canonical.index(requested_pair[0])
        ir = canonical.index(requested_pair[1])
        for harm in harmonics:
            idx = harmonic_block_indices(len(canonical), harm)
            val = chi_harm[idx[il], idx[ir]]
            print(
                f"\nrequested chi[{requested_pair[0]},{requested_pair[1]}]_{harm} "
                f"= {val:+.12e}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "V": float(args.V),
        "primitive_filling": float(args.primitive_filling),
        "stage": np.asarray(args.stage),
        "channels": np.asarray(canonical),
        "local_labels": np.asarray(local_labels),
        "harmonic_labels": np.asarray(harm_labels),
        "chi_raw": np.asarray(chi_raw),
        "chi_symmetric": np.asarray(chi_sym),
        "chi_harmonic": np.asarray(chi_harm),
        "chi_gg_harmonic": np.asarray(chi_gg_harm),
        "density": np.asarray(density),
        "scgw_residual": float(sc_res),
        "chi_imag_max": float(imag_max),
        "z_is_pauli_normalized": np.asarray(True),
    }
    for harm, block in blocks.items():
        payload[f"chi_{harm}"] = np.asarray(block)
    if vertex_results:
        payload["vertex_iterations"] = np.asarray([r.iterations for r in vertex_results])
        payload["vertex_residuals"] = np.asarray([r.final_error for r in vertex_results])
    np.savez_compressed(out, **payload)
    print("saved:", out)


if __name__ == "__main__":
    main()
