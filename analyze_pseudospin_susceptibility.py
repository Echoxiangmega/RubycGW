#!/usr/bin/env python3
"""Direct q=0 cGW susceptibilities for Ruby chirality-pseudospin channels.

This driver evaluates linear response on a converged zero-field 18-site
supercell SC-GW fixed point.  It does NOT perform finite-field calculations and
does NOT solve three sector-local perturbations first.  Instead it constructs
the requested normalized q=0 bare operator vertex directly,

    K_{mu,q0} = diag(K_mu,K_mu,K_mu) / sqrt(3),

and solves the usual covariant-GW equation

    (I-L) Gamma_{mu,q0} = K_{mu,q0}.

Thus one requested diagonal susceptibility such as chi_xx requires one cGW
vertex solve.  A cross response chi_ab requires only the right/driven vertex
Gamma_b.  A full N-channel matrix requires N cGW solves.

Examples
--------

    python analyze_pseudospin_susceptibility.py --V 1.4 --primitive-filling 2 \
        --chi x_even,x_even --harmonic q0

    python analyze_pseudospin_susceptibility.py --V 1.4 --primitive-filling 2 \
        --chi y_even,y_even --harmonic q0

    python analyze_pseudospin_susceptibility.py --V 1.4 --primitive-filling 2 \
        --chi z_same,z_same --harmonic q0

    python analyze_pseudospin_susceptibility.py --V 1.4 --primitive-filling 2 \
        --channels x_even y_even z_same z_opposite --harmonic q0

The x/y vertices are TR-even E-type intra-triangle charge/orbital order
parameters.  z is the TR-odd loop chirality.  z is Pauli-normalized, so a
diagonal chi_zz is one third of the legacy eta-current susceptibility for the
same physical current channel.

Finite-Q pseudospin response is intentionally deferred in this driver.
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
    supercell_pseudospin_harmonic_vertices,
)
from rubycgw.supercell import build_supercell_h0, build_supercell_interaction
from rubycgw.supercell_cgw import (
    SupercellVertexOptions,
    physical_symmetric_susceptibility,
    solve_vertex_q0,
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
            "One response chi[left,right], e.g. x_even,x_even or x_even,y_even. "
            "Only the right/driven cGW vertex is solved."
        ),
    )
    p.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channels for a full q=0 susceptibility matrix.",
    )
    p.add_argument("--list-channels", action="store_true")
    p.add_argument(
        "--harmonic",
        choices=["q0"],
        default="q0",
        help="Only primitive q=0 is enabled in the current production driver.",
    )
    p.add_argument(
        "--stage",
        choices=["gg", "split-mt", "full"],
        default="full",
        help="gg=bubble; split-mt=H+F+MT(W-V); full=also AL1/AL2.",
    )
    p.add_argument("--vertex-max-iter", type=int, default=150)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-solver", choices=["gmres", "linear"], default="gmres")
    p.add_argument("--vertex-gmres-restart", type=int, default=12)
    p.add_argument("--vertex-mixing", type=float, default=0.25)
    p.add_argument("--vertex-verbose", action="store_true")
    p.add_argument("--momentum-backend", choices=["fft", "direct"], default="fft")
    p.add_argument("--max-scgw-residual", type=float, default=1e-6)
    p.add_argument("--out", default="pseudospin_susceptibility_q0.npz")
    return p.parse_args()


def _select_response(args):
    if args.chi and args.channels:
        raise ValueError("use either --chi or --channels, not both")

    if args.chi:
        fields = [x.strip() for x in str(args.chi).split(",") if x.strip()]
        if len(fields) != 2:
            raise ValueError("--chi expects exactly two comma-separated channel names")
        left = canonical_channel_name(fields[0])
        right = canonical_channel_name(fields[1])
        return [left], [right], (left, right)

    if args.channels:
        channels = []
        for raw in args.channels:
            ch = canonical_channel_name(raw)
            if ch not in channels:
                channels.append(ch)
    else:
        channels = ["x_even", "y_even", "z_same", "z_opposite"]
    return channels, channels, None


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


def _susceptibility_rect_q0(
    G: np.ndarray,
    left_vertices: np.ndarray,
    right_gammas: list[np.ndarray],
    grid: MatsubaraGrid,
) -> np.ndarray:
    """Return chi_ab=-int Tr[K_left,a G Gamma_right,b G]."""
    K = np.asarray(left_vertices, dtype=complex)
    chi = np.zeros((K.shape[0], len(right_gammas)), dtype=complex)
    pref = -(grid.T / grid.nk)
    for b, gamma in enumerate(right_gammas):
        chi[:, b] = pref * np.einsum(
            "iab,nxybc,nxycd,nxyda->i",
            K,
            G,
            np.asarray(gamma, dtype=complex),
            G,
            optimize=True,
        )
    return chi


def _format_real_matrix(mat):
    arr = np.asarray(mat, dtype=float)
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

    left_channels, right_channels, requested_pair = _select_response(args)
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(
        nk1=args.nk1,
        nk2=args.nk2,
        nw=args.nw,
        nOmega=args.nomega,
        T=args.T,
    )

    checkpoint, seed, meta, _ = _exact_checkpoint(args, params, grid)
    print("checkpoint:", checkpoint)
    print("left operators :", ", ".join(left_channels))
    print("cGW derivatives:", ", ".join(right_channels))
    print(
        "response mode: direct zero-field cGW derivative at primitive q=0; "
        "no finite perturbation and no sector-local precursor solves"
    )
    print(
        "normalization: x,y,z all project to Pauli pseudospins; "
        "diagonal z susceptibility = legacy eta-current susceptibility / 3"
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

    Kleft, left_labels, left_canonical = supercell_pseudospin_harmonic_vertices(
        left_channels, harmonic="q0"
    )
    Kright, right_labels, right_canonical = supercell_pseudospin_harmonic_vertices(
        right_channels, harmonic="q0"
    )

    bare_right = [np.broadcast_to(K, G.shape).copy() for K in Kright]
    chi_gg = _susceptibility_rect_q0(G, Kleft, bare_right, grid)

    vertex_results = []
    if args.stage == "gg":
        gammas = bare_right
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
        for i, (label, K) in enumerate(zip(right_labels, Kright), start=1):
            print(f"--- direct cGW vertex {i}/{len(right_labels)}: {label} ({args.stage}) ---")
            result = solve_vertex_q0(G, W, Vq, K, grid, opts=vopts)
            vertex_results.append(result)
            if not result.converged:
                raise RuntimeError(
                    f"vertex {label} did not converge: residual={result.final_error:.3e}"
                )
            gammas.append(result.Gamma)
            print(
                f"{label}: solver={result.solver}, it={result.iterations}, "
                f"residual={result.final_error:.3e}, "
                f"|H|={np.max(np.abs(result.Gamma_H)):.3e}, "
                f"|F|={np.max(np.abs(result.Gamma_F)):.3e}, "
                f"|MTc|={np.max(np.abs(result.Gamma_MT)):.3e}, "
                f"|AL1|={np.max(np.abs(result.Gamma_AL1)):.3e}, "
                f"|AL2|={np.max(np.abs(result.Gamma_AL2)):.3e}"
            )

    chi_raw = _susceptibility_rect_q0(G, Kleft, gammas, grid)

    square_same_basis = left_canonical == right_canonical
    if square_same_basis:
        chi_sym, imag_max = physical_symmetric_susceptibility(chi_raw)
        chi_gg_sym, gg_imag_max = physical_symmetric_susceptibility(chi_gg)
        print("\nchi_GG(q=0), symmetric static matrix:")
        print(_format_real_matrix(chi_gg_sym))
        print(f"\nchi_{args.stage}(q=0), symmetric static matrix:")
        print(_format_real_matrix(chi_sym))
        print(
            f"discarded imaginary scales: GG={gg_imag_max:.3e}, "
            f"{args.stage}={imag_max:.3e}"
        )
    else:
        chi_sym = None
        chi_gg_sym = None
        imag_max = float(np.max(np.abs(chi_raw.imag)))
        gg_imag_max = float(np.max(np.abs(chi_gg.imag)))

    if requested_pair is not None:
        val_gg = complex(chi_gg[0, 0])
        val = complex(chi_raw[0, 0])
        print(
            f"\nrequested chi[{requested_pair[0]},{requested_pair[1]}]_q0"
        )
        print(f"  GG   = {val_gg.real:+.12e} {val_gg.imag:+.12e}j")
        print(f"  {args.stage:<4s} = {val.real:+.12e} {val.imag:+.12e}j")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "V": float(args.V),
        "primitive_filling": float(args.primitive_filling),
        "harmonic": np.asarray("q0"),
        "stage": np.asarray(args.stage),
        "left_channels": np.asarray(left_canonical),
        "right_channels": np.asarray(right_canonical),
        "left_vertex_labels": np.asarray(left_labels),
        "right_vertex_labels": np.asarray(right_labels),
        "chi_raw": np.asarray(chi_raw),
        "chi_gg_raw": np.asarray(chi_gg),
        "density": np.asarray(density),
        "scgw_residual": float(sc_res),
        "chi_imag_max": float(imag_max),
        "chi_gg_imag_max": float(gg_imag_max),
        "z_is_pauli_normalized": np.asarray(True),
        "direct_harmonic_vertex": np.asarray(True),
        "finite_field_used": np.asarray(False),
    }
    if chi_sym is not None:
        payload["chi_symmetric"] = np.asarray(chi_sym)
        payload["chi_gg_symmetric"] = np.asarray(chi_gg_sym)
    if vertex_results:
        payload["vertex_iterations"] = np.asarray([r.iterations for r in vertex_results])
        payload["vertex_residuals"] = np.asarray([r.final_error for r in vertex_results])
    np.savez_compressed(out, **payload)
    print("saved:", out)


if __name__ == "__main__":
    main()
