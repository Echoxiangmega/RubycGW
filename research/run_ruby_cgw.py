#!/usr/bin/env python3
"""Production 6-site primitive-cell GW/cGW response at external q=0.

The background SC-GW uses the same static-Fock + dynamic-(W-V) kernel as the
18-site production solver.  The response uses H/F/MTc/AL covariant vertices and
restarted matrix-free GMRES by default.

Examples
--------
Current channels (legacy physical labels, Pauli-normalized z)::

    python run_ruby_cgw.py --V 1.0 --filling 2 --chi z_opposite,z_opposite
    python run_ruby_cgw.py --V 1.0 --filling 2 --chi z_same,z_same

TR-even intra-triangle orbital/charge channels::

    python run_ruby_cgw.py --V 1.0 --filling 2 --chi x_even,x_even
    python run_ruby_cgw.py --V 1.0 --filling 2 --chi y_even,y_even

Full matrix among several q=0 order parameters::

    python run_ruby_cgw.py --V 1.0 --filling 2 \
        --channels x_even y_even z_same z_opposite

Finite external q is intentionally not part of this driver.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw import (
    GWOptions,
    MatsubaraGrid,
    RubyParameters,
    VertexOptions,
    available_pseudospin_channels,
    canonical_channel_name,
    physical_symmetric_susceptibility,
    primitive_pseudospin_vertex,
    rebuild_primitive_fixed_point,
    solve_gw,
    solve_noninteracting,
    solve_vertex_q0,
    susceptibility_matrix_q0,
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V", type=float, required=True)
    p.add_argument("--filling", type=float, default=2.0)
    p.add_argument("--T", type=float, default=0.05)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--nk1", type=int, default=4)
    p.add_argument("--nk2", type=int, default=4)
    p.add_argument("--nw", type=int, default=47)
    p.add_argument("--nomega", type=int, default=10)

    p.add_argument(
        "--chi",
        default=None,
        help="One requested pair, e.g. x_even,x_even or z_same,z_opposite.",
    )
    p.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channels for a full susceptibility matrix.",
    )
    p.add_argument("--list-channels", action="store_true")
    p.add_argument(
        "--stage",
        choices=["gg", "split-mt", "full"],
        default="full",
        help="gg=bubble; split-mt=H+F+MT(W-V); full=also AL1/AL2.",
    )

    p.add_argument("--gw-max-iter", type=int, default=300)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--gw-mixing", type=float, default=0.20)
    p.add_argument(
        "--gw-mixing-method", choices=["linear", "pulay"], default="pulay"
    )
    p.add_argument("--gw-pulay-history", type=int, default=6)
    p.add_argument("--gw-pulay-start", type=int, default=3)
    p.add_argument("--mu-tol", type=float, default=1e-10)
    p.add_argument("--mu-max-iter", type=int, default=100)
    p.add_argument("--gw-verbose", action="store_true")

    p.add_argument(
        "--vertex-solver", choices=["gmres", "linear"], default="gmres"
    )
    p.add_argument("--vertex-max-iter", type=int, default=150)
    p.add_argument("--vertex-tol", type=float, default=1e-8)
    p.add_argument("--vertex-gmres-restart", type=int, default=12)
    p.add_argument("--vertex-mixing", type=float, default=0.25)
    p.add_argument("--vertex-verbose", action="store_true")
    p.add_argument(
        "--momentum-backend", choices=["fft", "direct"], default="fft"
    )
    p.add_argument(
        "--max-background-residual",
        type=float,
        default=1e-6,
        help="Reject a background that is not a fixed point of the split GW map.",
    )
    p.add_argument("--out", default=None, help="Optional .npz output path.")
    return p.parse_args()


def _requested_channels(args) -> tuple[list[str], tuple[str, str] | None]:
    pair = None
    if args.chi is not None:
        fields = [x.strip() for x in str(args.chi).split(",") if x.strip()]
        if len(fields) != 2:
            raise ValueError("--chi must contain exactly two channels separated by a comma")
        pair = (canonical_channel_name(fields[0]), canonical_channel_name(fields[1]))
        requested = list(pair)
    elif args.channels:
        requested = [canonical_channel_name(x) for x in args.channels]
    else:
        requested = ["z_opposite", "z_same"]

    # Preserve order while avoiding duplicate vertex solves.
    unique = []
    for name in requested:
        if name not in unique:
            unique.append(name)
    return unique, pair


def _format_matrix(mat: np.ndarray) -> str:
    arr = np.asarray(mat)
    return "\n".join(
        "  " + " ".join(f"{float(x):+.8e}" for x in row) for row in arr
    )


def main():
    args = _parse_args()
    if args.list_channels:
        print("\n".join(available_pseudospin_channels()))
        return

    channels, requested_pair = _requested_channels(args)
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(
        nk1=args.nk1,
        nk2=args.nk2,
        nw=args.nw,
        nOmega=args.nomega,
        T=args.T,
    )

    bare = solve_noninteracting(
        params,
        grid,
        mu=0.0,
        target_filling=args.filling,
        mu_tol=args.mu_tol,
        mu_max_iter=args.mu_max_iter,
    )
    print(
        f"primitive G0: mu={bare.mu:.10f}, filling={np.sum(bare.density):.10f}"
    )

    gw_opts = GWOptions(
        mu=bare.mu,
        target_filling=args.filling,
        max_iter=args.gw_max_iter,
        tol=args.gw_tol,
        mixing=args.gw_mixing,
        mixing_method=args.gw_mixing_method,
        pulay_history=args.gw_pulay_history,
        pulay_start=args.gw_pulay_start,
        mu_tol=args.mu_tol,
        mu_max_iter=args.mu_max_iter,
        verbose=args.gw_verbose,
        momentum_backend=args.momentum_backend,
    )
    gw = solve_gw(params, grid, gw_opts)
    rebuilt = rebuild_primitive_fixed_point(
        gw, params, grid, backend=args.momentum_backend
    )
    print(
        f"primitive split SC-GW: converged={gw.converged}, it={gw.iterations}, "
        f"mu={gw.mu:.10f}, filling={np.sum(gw.density):.10f}, "
        f"stored_residual={gw.final_error:.3e}, rebuilt_residual={rebuilt['residual']:.3e}"
    )
    if float(rebuilt["residual"]) > args.max_background_residual:
        raise RuntimeError(
            "Primitive background is not a sufficiently accurate fixed point of "
            f"the static-Fock + W-V map: residual={rebuilt['residual']:.3e}"
        )

    K = np.stack([primitive_pseudospin_vertex(ch) for ch in channels], axis=0)
    bare_gammas = [np.broadcast_to(k, bare.G0.shape).copy() for k in K]
    gg_gammas = [np.broadcast_to(k, gw.G.shape).copy() for k in K]
    chi_g0 = susceptibility_matrix_q0(bare.G0, K, bare_gammas, grid)
    chi_gg = susceptibility_matrix_q0(gw.G, K, gg_gammas, grid)

    results = []
    if args.stage == "gg":
        gammas = gg_gammas
    else:
        opts = VertexOptions(
            max_iter=args.vertex_max_iter,
            tol=args.vertex_tol,
            mixing=args.vertex_mixing,
            solver=args.vertex_solver,
            gmres_restart=args.vertex_gmres_restart,
            include_hartree=True,
            include_fock=True,
            include_mt=True,
            include_al=args.stage == "full",
            verbose=args.vertex_verbose,
            momentum_backend=args.momentum_backend,
        )
        gammas = []
        for i, (ch, k) in enumerate(zip(channels, K), start=1):
            print(f"solve vertex {i}/{len(channels)}: {ch} [{args.stage}]")
            result = solve_vertex_q0(
                gw.G, gw.W, rebuilt["Vq"], k, grid, opts=opts
            )
            results.append(result)
            if not result.converged:
                raise RuntimeError(
                    f"vertex {ch} did not converge: residual={result.final_error:.3e}"
                )
            gammas.append(result.Gamma)
            print(
                f"  solver={result.solver}, it={result.iterations}, "
                f"residual={result.final_error:.3e}, "
                f"|H|={np.max(np.abs(result.Gamma_H)):.3e}, "
                f"|F|={np.max(np.abs(result.Gamma_F)):.3e}, "
                f"|MTc|={np.max(np.abs(result.Gamma_MT)):.3e}, "
                f"|AL1|={np.max(np.abs(result.Gamma_AL1)):.3e}, "
                f"|AL2|={np.max(np.abs(result.Gamma_AL2)):.3e}"
            )

    chi = susceptibility_matrix_q0(gw.G, K, gammas, grid)
    chi_g0_sym, g0_imag = physical_symmetric_susceptibility(chi_g0)
    chi_gg_sym, gg_imag = physical_symmetric_susceptibility(chi_gg)
    chi_sym, imag = physical_symmetric_susceptibility(chi)

    print("\nchannels:", ", ".join(channels))
    print("\nchi_G0G0(q=0):")
    print(_format_matrix(chi_g0_sym))
    print("\nchi_GG(q=0):")
    print(_format_matrix(chi_gg_sym))
    print(f"\nchi_{args.stage}(q=0):")
    print(_format_matrix(chi_sym))
    print(
        f"discarded imaginary scales: G0={g0_imag:.3e}, GG={gg_imag:.3e}, "
        f"{args.stage}={imag:.3e}"
    )

    if requested_pair is not None:
        ia = channels.index(requested_pair[0])
        ib = channels.index(requested_pair[1])
        print(
            f"\nrequested chi[{requested_pair[0]},{requested_pair[1]}] "
            f"= {chi_sym[ia, ib]:+.12e}"
        )

    if any(ch.startswith("z_") or ch in ("Az", "Bz") for ch in channels):
        print(
            "\nnormalization: z is Pauli-normalized; diagonal chi_zz is 1/3 of "
            "the legacy eta-current susceptibility for the same physical channel."
        )

    if args.out:
        path = Path(args.out)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            V=float(args.V),
            filling=float(args.filling),
            channels=np.asarray(channels),
            stage=np.asarray(args.stage),
            mu=float(gw.mu),
            density=np.asarray(gw.density),
            background_residual=float(rebuilt["residual"]),
            chi_g0=np.asarray(chi_g0),
            chi_gg=np.asarray(chi_gg),
            chi_raw=np.asarray(chi),
            chi_symmetric=np.asarray(chi_sym),
        )
        print("saved:", path)


if __name__ == "__main__":
    main()
