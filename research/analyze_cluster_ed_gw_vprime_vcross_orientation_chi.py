#!/usr/bin/env python3
"""Direct all-q physical susceptibility of a physical-pair cluster-ED+GW state.

This is the susceptibility-first counterpart of the full JF eigenspectrum
analysis.  It never diagonalizes the enormous Jacobian.  Instead it drives only
six physically relevant Ruby pseudospin sources,

    x_even, y_even, x_odd, y_odd, z_even, z_odd,

solves the covariant response equation, and forms the small static physical
susceptibility matrix

    chi_ab(q) = d <O_a(q)> / d h_b(q).

For q=0 an additional uniform-density source is solved and eliminated by a
Schur complement so the reported 6x6 matrix is at fixed total filling rather
than fixed chemical potential.  For q!=0 the perturbation carries no uniform
momentum and no such correction is needed.

The exact equilibrium identity chi(q)=chi(-q)^dagger is enforced by q/-q pair
Hermitianization before the CO (4x4), LC (2x2), and full (6x6) physical
eigenmodes are analyzed.  A continuous instability is diagnosed by a
susceptibility eigenvalue diverging, equivalently its signed inverse tending to
zero.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

import rubycgw.cluster_ed_gw_jf as cluster_jf
from rubycgw.cluster_ed_gw_jf_consistent import install_tail_consistent_cluster_jf
from rubycgw.finite_q_cgw import negative_q_index
from rubycgw.pseudospin import primitive_cell_pseudospin_channels

from research.analyze_cluster_ed_gw_vprime_vcross_orientation_lambda import (
    _build_operator,
)


CHANNELS = (
    "x_even",
    "y_even",
    "x_odd",
    "y_odd",
    "z_even",
    "z_odd",
)
CO_CHANNELS = CHANNELS[:4]
LC_CHANNELS = CHANNELS[4:]


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--q", nargs=2, type=int, default=(0, 0), metavar=("IQ1", "IQ2"))
    p.add_argument(
        "--all-q", action="store_true",
        help="scan every q on the saved primitive momentum mesh",
    )
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-svd-rcond", type=float, default=1e-7)
    p.add_argument("--bath-rank", type=int, default=24)
    p.add_argument("--bath-fd-step", type=float, default=2e-4)
    p.add_argument("--bath-fd-scheme", choices=("centered", "forward"), default="centered")
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)
    p.add_argument("--stage", choices=("mt", "full"), default="full")
    p.add_argument("--jf-tol", type=float, default=1e-8)
    p.add_argument("--jf-maxiter", type=int, default=100)
    p.add_argument("--jf-restart", type=int, default=28)
    p.add_argument("--jf-recycle-dim", type=int, default=14)
    p.add_argument("--no-recycle", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _q_list(grid, args):
    if args.all_q:
        return [
            (i, j)
            for i in range(int(grid.nk1))
            for j in range(int(grid.nk2))
        ]
    return [
        (
            int(args.q[0]) % int(grid.nk1),
            int(args.q[1]) % int(grid.nk2),
        )
    ]


def _vertices(include_uniform=False):
    ps = primitive_cell_pseudospin_channels()
    mats = [np.asarray(ps[name], dtype=complex) for name in CHANNELS]
    labels = list(CHANNELS)
    if include_uniform:
        mats.append(np.eye(6, dtype=complex) / np.sqrt(6.0))
        labels.append("uniform")
    return np.stack(mats, axis=0), labels


def _hermitianize_pair(chi_q, chi_mq):
    """Combine q/-q reciprocity and enforce the static Hermitian response."""
    a = np.asarray(chi_q, dtype=complex)
    b = np.asarray(chi_mq, dtype=complex)
    pair = 0.5 * (a + b.conj().T)
    return 0.5 * (pair + pair.conj().T)


def _fixed_filling_schur(chi7):
    """Eliminate the uniform-density response at q=0."""
    x = np.asarray(chi7, dtype=complex)
    if x.shape != (7, 7):
        raise ValueError("fixed-filling Schur complement expects a 7x7 matrix")
    phys = x[:6, :6]
    aN = x[:6, 6]
    Na = x[6, :6]
    NN = complex(x[6, 6])
    if abs(NN) < 1e-12:
        raise RuntimeError(
            "uniform compressibility is too small for a stable fixed-filling Schur complement"
        )
    return phys - np.outer(aN, Na) / NN, NN


def _eigen_block(block, names):
    h = 0.5 * (np.asarray(block, dtype=complex) + np.asarray(block, dtype=complex).conj().T)
    vals, vecs = np.linalg.eigh(h)
    vals = np.asarray(vals.real, dtype=float)
    # Softness is divergence in either sign direction on a continued reference
    # branch, i.e. the eigenvalue with the largest absolute susceptibility.
    isoft = int(np.argmax(np.abs(vals)))
    chi_soft = float(vals[isoft])
    inv_soft = float(1.0 / chi_soft) if abs(chi_soft) > 1e-300 else np.copysign(np.inf, chi_soft or 1.0)
    vec = np.asarray(vecs[:, isoft], dtype=complex)
    weights = np.abs(vec) ** 2
    weights /= max(float(np.sum(weights)), 1e-300)
    order = np.argsort(np.abs(vals))[::-1]
    inv_vals = np.asarray([
        1.0 / v if abs(v) > 1e-300 else np.copysign(np.inf, v or 1.0)
        for v in vals
    ], dtype=float)
    # Eigenvalues of the physical inverse susceptibility ("masses").  Sorting
    # these algebraically gives the clean stability diagnostic: on a stable
    # reference branch every mass is positive and the first continuous
    # instability is min(mass)->0.  After crossing, the negative-mode count
    # records how many physical directions are already unstable.
    finite_mass = np.asarray(inv_vals, dtype=float)
    mass_order = np.argsort(finite_mass)
    mass_eigvals = finite_mass[mass_order]
    mass_eigvecs = np.asarray(vecs[:, mass_order], dtype=complex)
    closest_mass_index = int(np.argmin(np.abs(mass_eigvals)))
    return dict(
        eigvals=vals,
        inverse_eigvals=inv_vals,
        mass_eigvals=mass_eigvals,
        mass_eigvecs=mass_eigvecs,
        min_mass=float(mass_eigvals[0]),
        closest_mass=float(mass_eigvals[closest_mass_index]),
        negative_mass_count=int(np.sum(mass_eigvals < 0.0)),
        eigvecs=vecs,
        soft_index=isoft,
        chi_soft=chi_soft,
        inv_chi_soft=inv_soft,
        soft_vector=vec,
        soft_weights=weights,
        soft_order=order,
        labels=tuple(names),
    )


def _analyze_physical_chi(chi6):
    chi6 = 0.5 * (np.asarray(chi6, dtype=complex) + np.asarray(chi6, dtype=complex).conj().T)
    co = _eigen_block(chi6[:4, :4], CO_CHANNELS)
    lc = _eigen_block(chi6[4:, 4:], LC_CHANNELS)
    full = _eigen_block(chi6, CHANNELS)
    cross = chi6[:4, 4:]
    cross_norm = float(np.linalg.norm(cross))
    diag_norm = max(
        float(np.linalg.norm(chi6[:4, :4])),
        float(np.linalg.norm(chi6[4:, 4:])),
        1e-300,
    )
    return co, lc, full, cross_norm / diag_norm


def _solve_all_q(op, grid, qlist, args):
    raw = {}
    solver_rows = {}
    for q in qlist:
        q = tuple(int(x) for x in q)
        include_uniform = q == (0, 0)
        K, labels = _vertices(include_uniform=include_uniform)
        print(
            f"\n--- q={q}: solving {len(labels)} physical sources "
            f"({', '.join(labels)}) ---",
            flush=True,
        )
        chi, results = cluster_jf.response_matrix(
            op,
            K,
            q,
            recycle=not bool(args.no_recycle),
        )
        raw[q] = np.asarray(chi, dtype=complex)
        solver_rows[q] = results
    return raw, solver_rows


def _ensure_q_pairs(op, grid, qlist, args, raw, solver_rows):
    requested = set(tuple(int(x) for x in q) for q in qlist)
    for q in list(requested):
        mq = negative_q_index(q, grid)
        if mq in raw:
            continue
        include_uniform = mq == (0, 0)
        K, labels = _vertices(include_uniform=include_uniform)
        print(
            f"\n--- q-pair partner {mq}: solving {len(labels)} physical sources ---",
            flush=True,
        )
        chi, results = cluster_jf.response_matrix(
            op,
            K,
            mq,
            recycle=not bool(args.no_recycle),
        )
        raw[mq] = np.asarray(chi, dtype=complex)
        solver_rows[mq] = results


def _output_for_q(raw, q, grid):
    mq = negative_q_index(q, grid)
    pair_residual_den = max(
        float(np.linalg.norm(raw[q])),
        float(np.linalg.norm(raw[mq])),
        1e-300,
    )
    pair_residual = float(
        np.linalg.norm(raw[q] - raw[mq].conj().T) / pair_residual_den
    )
    pair = _hermitianize_pair(raw[q], raw[mq])
    compressibility = np.nan + 0.0j
    if q == (0, 0):
        pair, compressibility = _fixed_filling_schur(pair)
    co, lc, full, cross_ratio = _analyze_physical_chi(pair)
    return pair, compressibility, pair_residual, co, lc, full, cross_ratio


def _fmt_soft(name, block):
    weights = ", ".join(
        f"{lab}={w:.3f}" for lab, w in zip(block["labels"], block["soft_weights"])
    )
    return (
        f"{name}: chi_soft={block['chi_soft']:+.8e}, "
        f"1/chi={block['inv_chi_soft']:+.8e}, {weights}"
    )


def main():
    args = _args()
    install_tail_consistent_cluster_jf()

    all_rows = []
    file_meta = []
    for path in args.files:
        print(f"\n=== susceptibility: {path} ===", flush=True)
        d, orientation, grid, params, op, tangent = _build_operator(path, args)
        op.opts = replace(
            op.opts,
            tol=float(args.jf_tol),
            maxiter=int(args.jf_maxiter),
            restart=int(args.jf_restart),
            recycle_dim=int(args.jf_recycle_dim),
            verbose=not bool(args.quiet),
        )

        qlist = _q_list(grid, args)
        raw, solver_rows = _solve_all_q(op, grid, qlist, args)
        _ensure_q_pairs(op, grid, qlist, args, raw, solver_rows)

        print(
            f"[bath tangent] rank={tangent.rank}, cond={tangent.condition_number:.3e}, "
            f"build={tangent.build_seconds:.1f}s",
            flush=True,
        )

        for q in qlist:
            q = tuple(int(x) for x in q)
            chi, compressibility, pair_residual, co, lc, full, cross_ratio = _output_for_q(
                raw, q, grid
            )
            print(
                f"q={q}, q-pair residual={pair_residual:.3e}, "
                f"CO-LC cross/diag={cross_ratio:.3e}"
            )
            print("  " + _fmt_soft("CO", co))
            print("  " + _fmt_soft("LC", lc))
            print("  " + _fmt_soft("FULL", full))

            results = solver_rows[q]
            all_rows.append(dict(
                file=str(path),
                orientation=int(orientation),
                q=q,
                chi=np.asarray(chi, dtype=complex),
                uniform_compressibility=complex(compressibility),
                q_pair_residual=float(pair_residual),
                co=co,
                lc=lc,
                full=full,
                co_lc_cross_ratio=float(cross_ratio),
                solver_iterations=np.asarray([r.iterations for r in results], dtype=int),
                solver_residual=np.asarray([r.final_error for r in results], dtype=float),
            ))

        file_meta.append((str(path), int(orientation)))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out,
            schema=np.asarray(1, dtype=int),
            channels=np.asarray(CHANNELS),
            co_channels=np.asarray(CO_CHANNELS),
            lc_channels=np.asarray(LC_CHANNELS),
            source_files=np.asarray([r["file"] for r in all_rows]),
            orientation=np.asarray([r["orientation"] for r in all_rows], dtype=int),
            q_index=np.asarray([r["q"] for r in all_rows], dtype=int),
            chi_matrix=np.stack([r["chi"] for r in all_rows], axis=0),
            uniform_compressibility=np.asarray(
                [r["uniform_compressibility"] for r in all_rows], dtype=complex
            ),
            q_pair_residual=np.asarray([r["q_pair_residual"] for r in all_rows], dtype=float),
            co_lc_cross_ratio=np.asarray(
                [r["co_lc_cross_ratio"] for r in all_rows], dtype=float
            ),
            chi_co_soft=np.asarray([r["co"]["chi_soft"] for r in all_rows], dtype=float),
            inv_chi_co_soft=np.asarray(
                [r["co"]["inv_chi_soft"] for r in all_rows], dtype=float
            ),
            chi_lc_soft=np.asarray([r["lc"]["chi_soft"] for r in all_rows], dtype=float),
            inv_chi_lc_soft=np.asarray(
                [r["lc"]["inv_chi_soft"] for r in all_rows], dtype=float
            ),
            chi_full_soft=np.asarray([r["full"]["chi_soft"] for r in all_rows], dtype=float),
            inv_chi_full_soft=np.asarray(
                [r["full"]["inv_chi_soft"] for r in all_rows], dtype=float
            ),
            co_eigvals=np.stack([r["co"]["eigvals"] for r in all_rows], axis=0),
            lc_eigvals=np.stack([r["lc"]["eigvals"] for r in all_rows], axis=0),
            full_eigvals=np.stack([r["full"]["eigvals"] for r in all_rows], axis=0),
            co_inverse_eigvals=np.stack(
                [r["co"]["inverse_eigvals"] for r in all_rows], axis=0
            ),
            lc_inverse_eigvals=np.stack(
                [r["lc"]["inverse_eigvals"] for r in all_rows], axis=0
            ),
            full_inverse_eigvals=np.stack(
                [r["full"]["inverse_eigvals"] for r in all_rows], axis=0
            ),
            co_mass_eigvals=np.stack(
                [r["co"]["mass_eigvals"] for r in all_rows], axis=0
            ),
            lc_mass_eigvals=np.stack(
                [r["lc"]["mass_eigvals"] for r in all_rows], axis=0
            ),
            full_mass_eigvals=np.stack(
                [r["full"]["mass_eigvals"] for r in all_rows], axis=0
            ),
            co_min_mass=np.asarray([r["co"]["min_mass"] for r in all_rows], dtype=float),
            lc_min_mass=np.asarray([r["lc"]["min_mass"] for r in all_rows], dtype=float),
            full_min_mass=np.asarray([r["full"]["min_mass"] for r in all_rows], dtype=float),
            co_closest_mass=np.asarray(
                [r["co"]["closest_mass"] for r in all_rows], dtype=float
            ),
            lc_closest_mass=np.asarray(
                [r["lc"]["closest_mass"] for r in all_rows], dtype=float
            ),
            full_closest_mass=np.asarray(
                [r["full"]["closest_mass"] for r in all_rows], dtype=float
            ),
            co_negative_mass_count=np.asarray(
                [r["co"]["negative_mass_count"] for r in all_rows], dtype=int
            ),
            lc_negative_mass_count=np.asarray(
                [r["lc"]["negative_mass_count"] for r in all_rows], dtype=int
            ),
            full_negative_mass_count=np.asarray(
                [r["full"]["negative_mass_count"] for r in all_rows], dtype=int
            ),
            co_soft_vector=np.stack(
                [r["co"]["soft_vector"] for r in all_rows], axis=0
            ),
            lc_soft_vector=np.stack(
                [r["lc"]["soft_vector"] for r in all_rows], axis=0
            ),
            full_soft_vector=np.stack(
                [r["full"]["soft_vector"] for r in all_rows], axis=0
            ),
            co_soft_weights=np.stack(
                [r["co"]["soft_weights"] for r in all_rows], axis=0
            ),
            lc_soft_weights=np.stack(
                [r["lc"]["soft_weights"] for r in all_rows], axis=0
            ),
            full_soft_weights=np.stack(
                [r["full"]["soft_weights"] for r in all_rows], axis=0
            ),
            solver_iterations=np.asarray(
                [np.max(r["solver_iterations"]) for r in all_rows], dtype=int
            ),
            solver_residual=np.asarray(
                [np.max(r["solver_residual"]) for r in all_rows], dtype=float
            ),
            stage=np.asarray(str(args.stage)),
            fixed_filling_q0=np.asarray(True),
            bath_rank=np.asarray(int(args.bath_rank)),
            bath_svd_rcond=np.asarray(float(args.bath_svd_rcond)),
            bath_fd_step=np.asarray(float(args.bath_fd_step)),
        )
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
