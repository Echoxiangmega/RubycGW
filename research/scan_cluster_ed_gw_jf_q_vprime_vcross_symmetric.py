#!/usr/bin/env python3
"""Scan pseudospin response of the three-orientation symmetric parent branch."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from rubycgw.cluster_ed_gw import BathParameters
from rubycgw.cluster_ed_gw_jf import BathTangentOptions, ClusterJFOptions
from rubycgw.cluster_ed_gw_jf_symmetric import (
    build_symmetrized_embedded_jacobian,
    response_matrix_symmetrized,
)
from rubycgw.finite_q_cgw import (
    hermitianize_q_pair,
    negative_q_index,
    q_reduced_from_index,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import build_h0
from rubycgw.pseudospin import primitive_pseudospin_vertex
from vprime_study.cross_model import (
    VPrimeCrossParameters,
    build_vprime_vcross_interaction,
)


CHANNELS = ("Ax", "Ay", "Az", "Bx", "By", "Bz")
PROJECTED_NAMES = (
    "x_even", "x_odd", "y_even", "y_odd", "z_even", "z_odd"
)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    q = p.add_mutually_exclusive_group()
    q.add_argument("--q-index", nargs=2, type=int, metavar=("IQ1", "IQ2"))
    q.add_argument("--all-q", action="store_true")
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-svd-rcond", type=float, default=1e-7)
    p.add_argument("--bath-rank", type=int, default=24)
    p.add_argument("--bath-fd-step", type=float, default=2e-4)
    p.add_argument(
        "--bath-fd-scheme", choices=("centered", "forward"), default="centered"
    )
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)
    p.add_argument("--solver", choices=("gcrotmk", "gmres"), default="gcrotmk")
    p.add_argument("--jf-tol", type=float, default=1e-8)
    p.add_argument("--jf-maxiter", type=int, default=120)
    p.add_argument("--krylov-m", type=int, default=32)
    p.add_argument("--recycle-k", type=int, default=16)
    p.add_argument("--stage", choices=("split-mt", "full"), default="full")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/vprime_vcross_cluster_symmetric_response"),
    )
    return p.parse_args()


def _scalar(d, key):
    return np.asarray(d[key]).reshape(()).item()


def _grid(d):
    G = np.asarray(d["G"])
    omega = np.asarray(d["omega"])
    Omega = np.asarray(d["Omega"])
    g = MatsubaraGrid(
        nk1=G.shape[1],
        nk2=G.shape[2],
        nw=len(omega) // 2,
        nOmega=(len(Omega) - 1) // 2,
        T=float(_scalar(d, "T")),
    )
    if not np.allclose(g.omega, omega, atol=1e-12, rtol=0.0):
        raise ValueError("saved fermionic grid mismatch")
    if not np.allclose(g.Omega, Omega, atol=1e-12, rtol=0.0):
        raise ValueError("saved bosonic grid mismatch")
    return g


def _q_points(args, grid):
    if args.all_q:
        return [
            (i, j)
            for i in range(grid.nk1)
            for j in range(grid.nk2)
        ]
    if args.q_index is None:
        return [(0, 0)]
    return [
        (
            int(args.q_index[0]) % grid.nk1,
            int(args.q_index[1]) % grid.nk2,
        )
    ]


def _hermitian_eigensystem(a):
    h = 0.5 * (np.asarray(a, dtype=complex) + np.asarray(a).conj().T)
    vals, vecs = np.linalg.eigh(h)
    order = np.argsort(vals.real)[::-1]
    return vals[order].real, vecs[:, order]


def _project_mode(vec):
    c = {CHANNELS[i]: complex(vec[i]) for i in range(len(CHANNELS))}
    out = np.asarray(
        [
            (c["Ax"] + c["Bx"]) / np.sqrt(2.0),
            (c["Ax"] - c["Bx"]) / np.sqrt(2.0),
            (c["Ay"] + c["By"]) / np.sqrt(2.0),
            (c["Ay"] - c["By"]) / np.sqrt(2.0),
            (c["Az"] + c["Bz"]) / np.sqrt(2.0),
            (c["Az"] - c["Bz"]) / np.sqrt(2.0),
        ],
        dtype=complex,
    )
    return out


def _mode_label(projected):
    w = np.abs(np.asarray(projected)) ** 2
    i = int(np.argmax(w))
    return PROJECTED_NAMES[i], float(w[i] / max(np.sum(w), 1e-300))


def main():
    args = _args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    with np.load(args.input, allow_pickle=False) as z:
        d = {k: np.asarray(z[k]).copy() for k in z.files}

    expected = "three_orientation_physical_pair_average_no_intercell_collapse"
    projection = str(_scalar(d, "cluster_projection"))
    if projection != expected:
        raise ValueError(
            f"expected symmetric physical-pair checkpoint {expected!r}, "
            f"got {projection!r}"
        )
    if not bool(_scalar(d, "converged")):
        raise RuntimeError("background checkpoint is not converged")

    grid = _grid(d)
    params = VPrimeCrossParameters(
        ti=float(_scalar(d, "ti")),
        t1=float(_scalar(d, "t1")),
        t2=float(_scalar(d, "t2")),
        V=float(_scalar(d, "V")),
        Vprime=float(_scalar(d, "Vprime")),
        Vcross=float(_scalar(d, "Vcross")),
    )
    G = np.asarray(d["G"], dtype=complex)
    sigma_h = np.asarray(d["Sigma_H"], dtype=complex)
    mu = float(_scalar(d, "mu"))
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_vprime_vcross_interaction(grid.qmesh(), params)

    eps = np.asarray(d["bath_energies_orientation"], dtype=float)
    hyb = np.asarray(d["bath_couplings_orientation"], dtype=complex)
    if eps.shape[0] != 3 or hyb.shape[0] != 3:
        raise ValueError("checkpoint must contain three orientation baths")
    last_bath_error = np.asarray(d["bath_fit_history"], dtype=float)[-1]
    baths = tuple(
        BathParameters(eps[r], hyb[r], float(last_bath_error[r]), 0)
        for r in range(3)
    )
    static_shifts = np.asarray(
        d["impurity_static_shift_orientation"], dtype=complex
    )

    max_rank = None if int(args.bath_rank) <= 0 else int(args.bath_rank)
    bath_opts = BathTangentOptions(
        fit_metric=str(_scalar(d, "bath_fit_metric")),
        nfit=int(args.bath_fit_nfreq),
        svd_rcond=float(args.bath_svd_rcond),
        max_rank=max_rank,
        fd_step=float(args.bath_fd_step),
        fd_scheme=str(args.bath_fd_scheme),
        discard_weight_tol=float(args.discard_weight_tol),
        verbose=not bool(args.quiet),
    )
    jf_opts = ClusterJFOptions(
        solver=str(args.solver),
        tol=float(args.jf_tol),
        maxiter=int(args.jf_maxiter),
        restart=int(args.krylov_m),
        recycle_dim=int(args.recycle_k),
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=(args.stage == "full"),
        momentum_backend="fft",
        verbose=not bool(args.quiet),
    )

    print(
        "=== three-orientation symmetric cluster-JF ===\n"
        f"input={args.input}\n"
        f"V={params.V:g}, V'={params.Vprime:g}, Vx={params.Vcross:g}, "
        f"T={grid.T:g}, nk={grid.nk1}x{grid.nk2}, mu={mu:+.9f}\n"
        "linearizing the UNPROJECTED 1/3-symmetrized functional; "
        "C3/TR parent constraints are not differentiated",
        flush=True,
    )

    built = build_symmetrized_embedded_jacobian(
        G,
        h0,
        Vq,
        baths,
        static_shifts,
        params,
        grid,
        mu,
        sigma_h,
        bath_opts=bath_opts,
        jf_opts=jf_opts,
    )
    op = built.operator
    vertices = np.stack([primitive_pseudospin_vertex(ch) for ch in CHANNELS])
    q_points = _q_points(args, grid)

    nq = len(q_points)
    chi_raw = np.zeros((nq, 6, 6), dtype=complex)
    iterations = np.zeros((nq, 6), dtype=int)
    residuals = np.zeros((nq, 6), dtype=float)
    previous = [None] * 6

    for iq, qidx in enumerate(q_points):
        print(
            f"\n=== q {iq+1}/{nq}: index={qidx}, "
            f"reduced={q_reduced_from_index(qidx, grid)} ===",
            flush=True,
        )
        chi, results = response_matrix_symmetrized(
            op,
            vertices,
            qidx,
            initial_gammas=previous,
            recycle=True,
        )
        chi_raw[iq] = chi
        for j, res in enumerate(results):
            iterations[iq, j] = int(res.iterations)
            residuals[iq, j] = float(res.final_error)
            previous[j] = res.Gamma

    # Use q/-q reciprocity whenever both members were explicitly computed.
    q_to_pos = {tuple(q): i for i, q in enumerate(q_points)}
    chi_h = np.empty_like(chi_raw)
    for i, qidx in enumerate(q_points):
        qm = negative_q_index(qidx, grid)
        j = q_to_pos.get(tuple(qm))
        if j is None:
            chi_h[i] = 0.5 * (chi_raw[i] + chi_raw[i].conj().T)
        else:
            chi_h[i] = hermitianize_q_pair(chi_raw[i], chi_raw[j])

    eigvals = np.empty((nq, 6), dtype=float)
    eigvecs = np.empty((nq, 6, 6), dtype=complex)
    projections = np.empty((nq, 6), dtype=complex)
    co_max = np.empty(nq, dtype=float)
    lc_max = np.empty(nq, dtype=float)
    co_lc_cross = np.empty(nq, dtype=float)
    labels = []
    weights = []
    co_idx = np.asarray([0, 1, 3, 4], dtype=int)  # Ax,Ay,Bx,By
    lc_idx = np.asarray([2, 5], dtype=int)        # Az,Bz
    for i in range(nq):
        eigvals[i], eigvecs[i] = _hermitian_eigensystem(chi_h[i])
        projections[i] = _project_mode(eigvecs[i, :, 0])
        label, weight = _mode_label(projections[i])
        labels.append(label)
        weights.append(weight)

        co_block = chi_h[i][np.ix_(co_idx, co_idx)]
        lc_block = chi_h[i][np.ix_(lc_idx, lc_idx)]
        co_max[i] = np.max(np.linalg.eigvalsh(
            0.5 * (co_block + co_block.conj().T)
        )).real
        lc_max[i] = np.max(np.linalg.eigvalsh(
            0.5 * (lc_block + lc_block.conj().T)
        )).real
        co_lc_cross[i] = np.max(
            np.abs(chi_h[i][np.ix_(co_idx, lc_idx)]),
            initial=0.0,
        )

    order = np.argsort(eigvals[:, 0])[::-1]
    print("\n=== leading physical modes ===", flush=True)
    for rank, pos in enumerate(order, 1):
        winner = "CO" if co_max[pos] > lc_max[pos] else "LC"
        print(
            f"{rank:2d}: q={q_reduced_from_index(q_points[pos], grid)}, "
            f"chi_max={eigvals[pos,0]:+.10e}, "
            f"mode={labels[pos]}, weight={weights[pos]:.4f}; "
            f"CO={co_max[pos]:+.10e}, LC={lc_max[pos]:+.10e}, "
            f"larger={winner}, cross={co_lc_cross[pos]:.2e}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    outfile = args.out / (
        f"{args.input.stem}_sym_jf_"
        + ("allq" if args.all_q else f"q{q_points[0][0]}_{q_points[0][1]}")
        + ".npz"
    )
    np.savez_compressed(
        outfile,
        source_file=np.asarray(str(args.input)),
        response_functional=np.asarray(
            "unprojected_GW_plus_one_third_sum_three_physical_pair_corrections"
        ),
        channels=np.asarray(CHANNELS),
        q_indices=np.asarray(q_points, dtype=int),
        q_reduced=np.asarray(
            [q_reduced_from_index(q, grid) for q in q_points], dtype=float
        ),
        chi_raw=chi_raw,
        chi_hermitian=chi_h,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        leading_projection_names=np.asarray(PROJECTED_NAMES),
        leading_projection=projections,
        leading_mode_label=np.asarray(labels),
        leading_mode_weight=np.asarray(weights),
        chi_co_max=co_max,
        chi_lc_max=lc_max,
        chi_co_lc_cross_max=co_lc_cross,
        iterations=iterations,
        response_residuals=residuals,
        bath_tangent_rank=np.asarray(
            [t.rank for t in built.tangents], dtype=int
        ),
        bath_tangent_condition=np.asarray(
            [t.condition_number for t in built.tangents], dtype=float
        ),
        V=float(params.V),
        Vprime=float(params.Vprime),
        Vcross=float(params.Vcross),
        T=float(grid.T),
        mu=float(mu),
    )
    print(f"\nsaved {outfile}", flush=True)


if __name__ == "__main__":
    main()
