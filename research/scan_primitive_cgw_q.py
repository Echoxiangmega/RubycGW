#!/usr/bin/env python3
"""Scan static primitive-cell cGW susceptibilities over external momentum q.

The six-site SC-GW background is solved once.  For each requested external
momentum p on the same reciprocal mesh, the code evaluates

    X(k;p) = G(k+p) Gamma(k;p) G(k)

and solves the finite-q covariant equation

    (I-L_p) Gamma_p = K_p.

No supercell is required.  No finite perturbation is applied.  The calculation
is a direct zero-field covariant derivative of the translationally invariant
primitive-cell GW fixed point.

Examples
--------
One momentum Q=(1/3,1/3), requiring nk1,nk2 divisible by 3::

    python scan_primitive_cgw_q.py --V 1.0 --filling 2 --nk1 6 --nk2 6 \
        --chi x_even,x_even --q 0.3333333333333333 0.3333333333333333

Scan the full primitive reciprocal mesh::

    python scan_primitive_cgw_q.py --V 1.0 --filling 2 --nk1 6 --nk2 6 \
        --chi x_even,x_even --all-q

Compare several pseudospin channels at every q::

    python scan_primitive_cgw_q.py --V 1.0 --filling 2 --nk1 6 --nk2 6 \
        --channels x_even y_even z_same z_opposite --all-q

Use ``--stage gg`` for a cheap dressed-bubble momentum map before running full
cGW.  ``--stage split-mt`` includes H/F/MT(W-V); ``--stage full`` also includes
AL1/AL2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw import (
    GWOptions,
    MatsubaraGrid,
    RubyParameters,
    available_pseudospin_channels,
    canonical_channel_name,
    primitive_pseudospin_vertex,
    rebuild_primitive_fixed_point,
    solve_gw,
    solve_noninteracting,
)
from rubycgw.finite_q_cgw import (
    FiniteQVertexOptions,
    hermitianize_q_pair,
    negative_q_index,
    q_index_from_reduced,
    q_reduced_from_index,
    solve_vertex_finite_q,
    susceptibility_matrix_finite_q,
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

    resp = p.add_mutually_exclusive_group()
    resp.add_argument(
        "--chi",
        default=None,
        help="One response pair left,right, e.g. x_even,x_even or x_even,y_even.",
    )
    resp.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channels for a complete susceptibility matrix at every q.",
    )
    p.add_argument("--list-channels", action="store_true")

    qsel = p.add_mutually_exclusive_group()
    qsel.add_argument(
        "--q-index",
        nargs=2,
        type=int,
        metavar=("IQ1", "IQ2"),
        help="External momentum as integer indices on the k mesh.",
    )
    qsel.add_argument(
        "--q",
        nargs=2,
        type=float,
        metavar=("Q1", "Q2"),
        help="External momentum in reduced reciprocal coordinates; must lie on mesh.",
    )
    qsel.add_argument(
        "--all-q",
        action="store_true",
        help="Scan every external momentum on the primitive reciprocal mesh.",
    )

    p.add_argument(
        "--stage",
        choices=["gg", "split-mt", "full"],
        default="full",
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
    p.add_argument("--max-background-residual", type=float, default=1e-6)
    p.add_argument(
        "--no-q-warm-start",
        action="store_true",
        help="Do not initialize each q vertex from the previous q solution.",
    )
    p.add_argument("--top", type=int, default=8, help="Number of soft/large-q points to print.")
    p.add_argument("--out", default="primitive_finite_q_cgw.npz")
    return p.parse_args()


def _response_channels(args):
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


def _q_points(args, grid):
    if args.all_q:
        return [(i, j) for i in range(grid.nk1) for j in range(grid.nk2)]
    if args.q_index is not None:
        return [(int(args.q_index[0]) % grid.nk1, int(args.q_index[1]) % grid.nk2)]
    if args.q is not None:
        return [q_index_from_reduced((args.q[0], args.q[1]), grid)]
    return [(0, 0)]


def _centered_q(index, grid):
    i, j = index
    x = i if i <= grid.nk1 // 2 else i - grid.nk1
    y = j if j <= grid.nk2 // 2 else j - grid.nk2
    return float(x) / grid.nk1, float(y) / grid.nk2


def _format_complex(z):
    z = complex(z)
    return f"{z.real:+.8e}{z.imag:+.3e}j"


def main():
    args = _parse_args()
    if args.list_channels:
        print("\n".join(available_pseudospin_channels()))
        return

    left_channels, right_channels, requested_pair = _response_channels(args)
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(
        nk1=args.nk1,
        nk2=args.nk2,
        nw=args.nw,
        nOmega=args.nomega,
        T=args.T,
    )
    q_points = _q_points(args, grid)

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
        f"residual={float(rebuilt['residual']):.3e}"
    )
    if float(rebuilt["residual"]) > float(args.max_background_residual):
        raise RuntimeError(
            "Primitive background is not a sufficiently accurate split-GW fixed point: "
            f"residual={float(rebuilt['residual']):.3e}"
        )

    Kleft = np.stack([primitive_pseudospin_vertex(ch) for ch in left_channels])
    Kright = np.stack([primitive_pseudospin_vertex(ch) for ch in right_channels])

    vopts = FiniteQVertexOptions(
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

    nq = len(q_points)
    chi_raw = np.zeros((nq, len(left_channels), len(right_channels)), dtype=complex)
    chi_gg = np.zeros_like(chi_raw)
    iterations = np.zeros((nq, len(right_channels)), dtype=int)
    residuals = np.zeros((nq, len(right_channels)), dtype=float)
    previous_gamma = [None for _ in right_channels]

    print(
        f"response: stage={args.stage}, q-points={nq}, "
        f"left={left_channels}, driven={right_channels}"
    )
    for iq, pidx in enumerate(q_points, start=1):
        qred = q_reduced_from_index(pidx, grid)
        qctr = _centered_q(pidx, grid)
        print(
            f"\n=== q {iq}/{nq}: index={pidx}, reduced={qred}, centered={qctr} ==="
        )
        bare_gammas = [np.broadcast_to(K, gw.G.shape).copy() for K in Kright]
        chi_gg[iq - 1] = susceptibility_matrix_finite_q(
            gw.G, Kleft, bare_gammas, pidx, grid
        )

        if args.stage == "gg":
            gammas = bare_gammas
        else:
            gammas = []
            for j, (ch, K) in enumerate(zip(right_channels, Kright)):
                initial = None if args.no_q_warm_start else previous_gamma[j]
                print(f"solve finite-q vertex {j+1}/{len(right_channels)}: {ch}")
                result = solve_vertex_finite_q(
                    gw.G,
                    gw.W,
                    rebuilt["Vq"],
                    K,
                    pidx,
                    grid,
                    opts=vopts,
                    initial_gamma=initial,
                )
                iterations[iq - 1, j] = int(result.iterations)
                residuals[iq - 1, j] = float(result.final_error)
                if not result.converged:
                    raise RuntimeError(
                        f"finite-q vertex {ch} at q={qred} did not converge: "
                        f"residual={result.final_error:.3e}"
                    )
                gammas.append(result.Gamma)
                previous_gamma[j] = result.Gamma
                print(
                    f"  it={result.iterations}, residual={result.final_error:.3e}, "
                    f"|H|={np.max(np.abs(result.Gamma_H)):.3e}, "
                    f"|F|={np.max(np.abs(result.Gamma_F)):.3e}, "
                    f"|MTc|={np.max(np.abs(result.Gamma_MT)):.3e}, "
                    f"|AL1|={np.max(np.abs(result.Gamma_AL1)):.3e}, "
                    f"|AL2|={np.max(np.abs(result.Gamma_AL2)):.3e}"
                )

        chi_raw[iq - 1] = susceptibility_matrix_finite_q(
            gw.G, Kleft, gammas, pidx, grid
        )
        if requested_pair is not None:
            print(
                f"chi[{requested_pair[0]},{requested_pair[1]}](q) = "
                f"{_format_complex(chi_raw[iq-1,0,0])}"
            )

    q_to_pos = {tuple(q): i for i, q in enumerate(q_points)}
    chi_herm = None
    if left_channels == right_channels:
        chi_herm = np.zeros_like(chi_raw)
        for i, pidx in enumerate(q_points):
            pm = negative_q_index(pidx, grid)
            if pm in q_to_pos:
                chi_herm[i] = hermitianize_q_pair(
                    chi_raw[i], chi_raw[q_to_pos[pm]]
                )
            else:
                chi_herm[i] = 0.5 * (chi_raw[i] + chi_raw[i].conj().T)

    if nq > 1:
        if chi_herm is not None:
            leading = np.asarray(
                [np.max(np.linalg.eigvalsh(x).real) for x in chi_herm], dtype=float
            )
            order = np.argsort(leading)[::-1]
            print("\n=== largest susceptibility eigenvalues over q ===")
            for rank, pos in enumerate(order[: max(int(args.top), 1)], start=1):
                print(
                    f"{rank:2d}: q_index={q_points[pos]}, "
                    f"q_centered={_centered_q(q_points[pos], grid)}, "
                    f"lambda_max={leading[pos]:+.10e}"
                )
        elif requested_pair is not None and requested_pair[0] == requested_pair[1]:
            values = chi_raw[:, 0, 0].real
            order = np.argsort(values)[::-1]
            print("\n=== largest diagonal susceptibility over q ===")
            for rank, pos in enumerate(order[: max(int(args.top), 1)], start=1):
                print(
                    f"{rank:2d}: q_index={q_points[pos]}, "
                    f"q_centered={_centered_q(q_points[pos], grid)}, "
                    f"chi={values[pos]:+.10e}"
                )

    out = Path(args.out)
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "V": float(args.V),
        "filling": float(args.filling),
        "T": float(args.T),
        "nk1": int(grid.nk1),
        "nk2": int(grid.nk2),
        "stage": np.asarray(args.stage),
        "left_channels": np.asarray(left_channels),
        "right_channels": np.asarray(right_channels),
        "q_indices": np.asarray(q_points, dtype=int),
        "q_reduced": np.asarray([q_reduced_from_index(q, grid) for q in q_points]),
        "q_centered": np.asarray([_centered_q(q, grid) for q in q_points]),
        "chi_raw": np.asarray(chi_raw),
        "chi_gg": np.asarray(chi_gg),
        "vertex_iterations": iterations,
        "vertex_residuals": residuals,
        "mu": float(gw.mu),
        "density": np.asarray(gw.density),
        "background_residual": float(rebuilt["residual"]),
        "z_is_pauli_normalized": np.asarray(True),
    }
    if chi_herm is not None:
        payload["chi_hermitian_qpair"] = np.asarray(chi_herm)
        payload["lambda_max"] = np.asarray(
            [np.max(np.linalg.eigvalsh(x).real) for x in chi_herm], dtype=float
        )
    np.savez_compressed(out, **payload)
    print("saved:", out)


if __name__ == "__main__":
    main()
