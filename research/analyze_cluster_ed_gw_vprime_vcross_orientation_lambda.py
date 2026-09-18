#!/usr/bin/env python3
"""Leading JF-kernel eigenvalues for orientation-resolved physical-pair checkpoints.

This analyzes already-converged ori0/ori1/ori2 cluster-ED+GW solutions without
constructing a new symmetrized nonlinear branch.  For each checkpoint it builds
the response tangent of the *same* physical-pair impurity functional and solves

    L v = lambda v

for the leading real-linear Jacobian modes.  The q=0 problem is split into
spinless time-reversal even and odd sectors.  The odd sector is the clean LC
stability diagnostic on a TR-even CO background: lambda_LC > 1 means that CO
branch is linearly unstable to a loop-current fluctuation.
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
from scipy.sparse.linalg import LinearOperator, eigs

from rubycgw.cluster_ed_gw import BathParameters
from rubycgw.cluster_ed_gw_jf import (
    BathTangentOptions,
    ClusterJFOptions,
    _pack_complex,
    _unpack_complex,
    build_embedded_jacobian,
)
import rubycgw.cluster_ed_gw_jf_consistent as jf_consistent
from rubycgw.cluster_orientation import build_oriented_lattice_fields
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import build_h0
from rubycgw.models.ruby import physical_pair_cluster_interactions
from rubycgw.pseudospin import primitive_cell_pseudospin_channels
from rubycgw.response_tail import build_tail_reference
from rubycgw.supercell_gw_split import one_body_density_matrix_tail
from vprime_study.cross_model import (
    VPrimeCrossParameters,
    build_vprime_vcross_interaction,
)
from vprime_study.patches import (
    install_cluster_interaction_hooks,
    restore_vprime_cluster_hooks,
)


CHANNELS = (
    "x_even", "y_even", "x_odd", "y_odd", "z_even", "z_odd"
)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--q", nargs=2, type=int, default=(0, 0), metavar=("IQ1", "IQ2"))
    p.add_argument(
        "--all-q", action="store_true",
        help="scan every momentum on the saved nk1 x nk2 mesh; the bath tangent is built only once",
    )
    p.add_argument("--nev", type=int, default=6, help="eigenmodes retained per sector/full-q solve")
    p.add_argument("--arpack-tol", type=float, default=2e-6)
    p.add_argument("--arpack-maxiter", type=int, default=350)
    p.add_argument("--ncv", type=int, default=28)
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-svd-rcond", type=float, default=1e-7)
    p.add_argument("--bath-rank", type=int, default=24)
    p.add_argument("--bath-fd-step", type=float, default=2e-4)
    p.add_argument("--bath-fd-scheme", choices=("centered", "forward"), default="centered")
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)
    p.add_argument("--stage", choices=("mt", "full"), default="full")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _scalar(d, key, default=None, cast=float):
    if key not in d:
        if default is None:
            raise KeyError(key)
        return cast(default)
    return cast(np.asarray(d[key]).reshape(()))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _grid_from_saved(d):
    omega = np.asarray(d["omega"], dtype=float)
    Omega = np.asarray(d["Omega"], dtype=float)
    nw = len(omega) // 2
    nOmega = (len(Omega) - 1) // 2
    return MatsubaraGrid(
        nk1=_scalar(d, "Lx", cast=int),
        nk2=_scalar(d, "Ly", cast=int),
        nw=nw,
        nOmega=nOmega,
        T=_scalar(d, "T"),
    )


def _tr_transform(field: np.ndarray) -> np.ndarray:
    """Spinless TR for a q=0 vertex field: X(k,iw)->X*(-k,-iw)."""
    x = np.asarray(field, dtype=complex)
    nf, nk1, nk2 = x.shape[:3]
    ineg = np.arange(nf - 1, -1, -1)
    k1neg = (-np.arange(nk1)) % nk1
    k2neg = (-np.arange(nk2)) % nk2
    return np.conj(x[ineg][:, k1neg][:, :, k2neg])


def _tr_project(field: np.ndarray, parity: str) -> np.ndarray:
    tx = _tr_transform(field)
    if parity == "even":
        return 0.5 * (field + tx)
    if parity == "odd":
        return 0.5 * (field - tx)
    raise ValueError("TR parity must be even or odd")


def _tr_residual(field: np.ndarray, parity: str) -> float:
    x = np.asarray(field, dtype=complex)
    px = _tr_project(x, parity)
    den = max(float(np.max(np.abs(x))), 1e-300)
    return float(np.max(np.abs(x - px)) / den)


def _sector_operator(op, q_index, parity: str) -> LinearOperator:
    shape = op.G.shape
    n = 2 * int(np.prod(shape))

    def matvec(x):
        gamma = _unpack_complex(np.asarray(x, dtype=float), shape)
        gamma = _tr_project(gamma, parity)
        out = jf_consistent.homogeneous_linear_kernel(op, gamma, q_index)
        out = _tr_project(out, parity)
        return _pack_complex(out)

    return LinearOperator((n, n), matvec=matvec, dtype=float)


def _seed(op, parity: str) -> np.ndarray:
    ps = primitive_cell_pseudospin_channels()
    K = ps["z_even"] if parity == "odd" else ps["x_even"]
    field = np.broadcast_to(K, op.G.shape).copy()
    field = _tr_project(field, parity)
    v = _pack_complex(field)
    nrm = np.linalg.norm(v)
    if nrm <= 0:
        raise RuntimeError("zero ARPACK seed")
    return v / nrm


def _physical_field_from_packed(vec: np.ndarray, shape) -> np.ndarray:
    """Choose a real direction from a possibly complex ARPACK eigenvector."""
    z = np.asarray(vec)
    vr = np.asarray(z.real, dtype=float)
    vi = np.asarray(z.imag, dtype=float)
    packed = vr if np.linalg.norm(vr) >= np.linalg.norm(vi) else vi
    return _unpack_complex(packed, shape)


def _mode_projection(op, field, q_index):
    """Project the equal-time mode density matrix onto CO/LC pseudospins."""
    p = tuple(int(x) for x in q_index)
    zero = np.zeros((6, 6), dtype=complex)
    ctx = jf_consistent._tail_context(op, zero, p)
    X = op._x_field(np.asarray(field, dtype=complex), p)
    _, Rk = jf_consistent._static_net_field(op, X, ctx)
    Rc = np.mean(np.asarray(Rk, dtype=complex), axis=(0, 1))

    ps = primitive_cell_pseudospin_channels()
    mats = {name: np.asarray(ps[name], dtype=complex) for name in CHANNELS}
    mats["uniform"] = np.eye(6, dtype=complex)

    amp = {}
    for name, K in mats.items():
        den = max(float(np.linalg.norm(K)), 1e-300)
        amp[name] = np.vdot(K, Rc) / den

    power = {name: float(abs(val) ** 2) for name, val in amp.items()}
    total = max(sum(power.values()), 1e-300)
    weight = {name: power[name] / total for name in power}
    weight["CO_even"] = weight["x_even"] + weight["y_even"]
    weight["CO_odd"] = weight["x_odd"] + weight["y_odd"]
    weight["LC_same"] = weight["z_even"]
    weight["LC_opposite"] = weight["z_odd"]
    return amp, weight


def _solve_sector(op, q_index, parity, args):
    L = _sector_operator(op, q_index, parity)
    n = L.shape[0]
    k = min(max(int(args.nev), 1), max(n - 2, 1))
    ncv = min(max(int(args.ncv), 2 * k + 2), n)
    vals, vecs = eigs(
        L,
        k=k,
        which="LR",
        v0=_seed(op, parity),
        tol=float(args.arpack_tol),
        maxiter=int(args.arpack_maxiter),
        ncv=ncv,
    )
    # Static softness is proximity to the pole lambda=1.  Sort all retained
    # Arnoldi modes by |1-lambda| rather than by Re(lambda).
    order = np.argsort(np.abs(1.0 - vals))
    vals = vals[order]
    vecs = vecs[:, order]
    modes = []
    for j, lam in enumerate(vals):
        field = _physical_field_from_packed(vecs[:, j], op.G.shape)
        field = _tr_project(field, parity)
        amp, weight = _mode_projection(op, field, q_index)
        modes.append(
            {
                "lambda": complex(lam),
                "weight": weight,
                "amp": amp,
                "tr_residual": _tr_residual(field, parity),
            }
        )
    return modes



def _full_seed(op) -> np.ndarray:
    ps = primitive_cell_pseudospin_channels()
    K = (
        np.asarray(ps["x_even"], dtype=complex)
        + np.asarray(ps["z_even"], dtype=complex)
        + 0.5 * np.asarray(ps["x_odd"], dtype=complex)
        + 0.5 * np.asarray(ps["z_odd"], dtype=complex)
    )
    field = np.broadcast_to(K, op.G.shape).copy()
    v = _pack_complex(field)
    nrm = np.linalg.norm(v)
    if nrm <= 0:
        raise RuntimeError("zero full-kernel ARPACK seed")
    return v / nrm


def _solve_full_q(op, q_index, args):
    """Solve the complete real-linear kernel at a finite momentum q.

    Time reversal maps a generic q to -q, so one must not impose a TR-even/odd
    projector within a single finite-q problem.  We therefore diagonalize the
    full packed-real kernel and classify the resulting modes by their equal-time
    CO/LC projection weights.
    """
    L = jf_consistent.homogeneous_kernel_operator(op, q_index)
    n = L.shape[0]
    k = min(max(int(args.nev), 1), max(n - 2, 1))
    ncv = min(max(int(args.ncv), 2 * k + 2), n)
    vals, vecs = eigs(
        L,
        k=k,
        which="LR",
        v0=_full_seed(op),
        tol=float(args.arpack_tol),
        maxiter=int(args.arpack_maxiter),
        ncv=ncv,
    )
    order = np.argsort(np.abs(1.0 - vals))
    vals = vals[order]
    vecs = vecs[:, order]
    modes = []
    for j, lam in enumerate(vals):
        field = _physical_field_from_packed(vecs[:, j], op.G.shape)
        amp, weight = _mode_projection(op, field, q_index)
        modes.append(
            {
                "lambda": complex(lam),
                "weight": weight,
                "amp": amp,
                "tr_residual": np.nan,
            }
        )
    return modes


def _select_candidates(modes, *, gamma_split=False, even_modes=None, odd_modes=None):
    """Return the softest LC and CO candidates by min |1-lambda|."""
    if gamma_split:
        if even_modes is None or odd_modes is None:
            raise ValueError("gamma_split requires even_modes and odd_modes")
        lc_pool = [
            m for m in odd_modes
            if (m["weight"]["LC_same"] + m["weight"]["LC_opposite"])
            >= max(
                m["weight"]["CO_even"] + m["weight"]["CO_odd"],
                m["weight"]["uniform"],
            )
        ]
        co_pool = [
            m for m in even_modes
            if (m["weight"]["CO_even"] + m["weight"]["CO_odd"])
            >= m["weight"]["uniform"]
        ]
        lc = min(
            lc_pool if lc_pool else odd_modes,
            key=lambda m: abs(1.0 - m["lambda"]),
        )
        co = min(
            co_pool if co_pool else even_modes,
            key=lambda m: abs(1.0 - m["lambda"]),
        )
        return lc, co

    lc_pool = [
        m for m in modes
        if (m["weight"]["LC_same"] + m["weight"]["LC_opposite"])
        >= max(
            m["weight"]["CO_even"] + m["weight"]["CO_odd"],
            m["weight"]["uniform"],
        )
    ]
    co_pool = [
        m for m in modes
        if (m["weight"]["CO_even"] + m["weight"]["CO_odd"])
        >= max(
            m["weight"]["LC_same"] + m["weight"]["LC_opposite"],
            m["weight"]["uniform"],
        )
    ]
    lc = min(
        lc_pool if lc_pool else modes,
        key=lambda m: abs(1.0 - m["lambda"]),
    )
    co = min(
        co_pool if co_pool else modes,
        key=lambda m: abs(1.0 - m["lambda"]),
    )
    return lc, co


def _q_list(grid, args):
    if bool(args.all_q):
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


def _analyze_one_q(op, qn, args):
    """Analyze one q and return (LC candidate, CO candidate, all modes)."""
    qn = tuple(int(x) for x in qn)
    if qn == (0, 0):
        even_modes = _solve_sector(op, qn, "even", args)
        odd_modes = _solve_sector(op, qn, "odd", args)
        print("  TR-even leading modes:")
        for i, m in enumerate(even_modes, 1):
            print(f"    {i}: {_fmt_mode(m)}")
        print("  TR-odd leading modes:")
        for i, m in enumerate(odd_modes, 1):
            print(f"    {i}: {_fmt_mode(m)}")
        lc, co = _select_candidates(
            [], gamma_split=True, even_modes=even_modes, odd_modes=odd_modes
        )
        return lc, co, even_modes + odd_modes, "gamma_TR_split"

    modes = _solve_full_q(op, qn, args)
    print("  full finite-q leading modes:")
    for i, m in enumerate(modes, 1):
        print(f"    {i}: {_fmt_mode(m)}")
    lc, co = _select_candidates(modes)
    return lc, co, modes, "full_finite_q"


def _build_operator(path: Path, args):
    d = _load_npz(path)
    if not bool(_scalar(d, "converged", default=True, cast=bool)):
        raise RuntimeError(f"{path}: checkpoint is marked unconverged")
    projection = str(np.asarray(d.get("cluster_projection", "")).reshape(()))
    if projection != "physical_pair_no_intercell_collapse":
        raise RuntimeError(
            f"{path}: expected physical_pair_no_intercell_collapse, got {projection!r}"
        )

    orientation = _scalar(d, "cluster_orientation", cast=int)
    grid = _grid_from_saved(d)
    params = VPrimeCrossParameters(
        ti=_scalar(d, "ti"),
        t1=_scalar(d, "t1"),
        t2=_scalar(d, "t2"),
        V=_scalar(d, "V"),
        Vprime=_scalar(d, "Vprime", default=_scalar(d, "Vp", default=0.0)),
        Vcross=_scalar(d, "Vcross", default=_scalar(d, "Vx", default=0.0)),
    )

    h0_base = build_h0(grid.kmesh(), params)
    Vq_base = build_vprime_vcross_interaction(grid.qmesh(), params)
    h0, Vq = build_oriented_lattice_fields(
        h0_base, Vq_base, orientation
    )

    G = np.asarray(d["G"], dtype=complex)
    sigma_h = np.asarray(d["Sigma_H"], dtype=complex)
    mu = _scalar(d, "mu")
    rho_k = one_body_density_matrix_tail(
        G, grid, h0, mu, sigma_h
    )
    rho_c = np.mean(rho_k, axis=(0, 1))

    h_cluster = np.mean(h0, axis=(0, 1))
    h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)
    if "impurity_static_shift" in d:
        h_cluster = h_cluster + np.asarray(
            d["impurity_static_shift"], dtype=complex
        )

    bath = BathParameters(
        np.asarray(d["bath_energies"], dtype=float),
        np.asarray(d["bath_couplings"], dtype=complex),
        _scalar(d, "bath_fit_error", default=np.nan),
        0,
    )
    metric = str(
        np.asarray(d.get("bath_fit_metric", "delta")).reshape(())
    )

    max_rank = None if int(args.bath_rank) <= 0 else int(args.bath_rank)
    bath_opts = BathTangentOptions(
        fit_metric=metric,
        nfit=int(args.bath_fit_nfreq),
        svd_rcond=float(args.bath_svd_rcond),
        max_rank=max_rank,
        fd_step=float(args.bath_fd_step),
        fd_scheme=str(args.bath_fd_scheme),
        discard_weight_tol=float(args.discard_weight_tol),
        verbose=not bool(args.quiet),
    )
    jf_opts = ClusterJFOptions(
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=(args.stage == "full"),
        momentum_backend="fft",
        verbose=not bool(args.quiet),
    )

    old = install_cluster_interaction_hooks(
        lambda p, r=orientation: physical_pair_cluster_interactions(p, r)
    )
    try:
        op, tangent = build_embedded_jacobian(
            G,
            Vq,
            bath,
            h_cluster,
            params,
            grid,
            mu,
            rho_c,
            bath_opts=bath_opts,
            jf_opts=jf_opts,
        )
    finally:
        restore_vprime_cluster_hooks(old)

    # The generic consistency wrapper historically reconstructs a canonical
    # lattice tail reference.  For an oriented checkpoint use the actual saved
    # lattice Hartree field and the actual oriented h0.
    op._jf_tail_reference = build_tail_reference(
        h0, mu, sigma_h, grid
    )
    return d, orientation, grid, params, op, tangent


def _fmt_mode(m):
    lam = m["lambda"]
    w = m["weight"]
    d1 = abs(1.0 - lam)
    mod = abs(lam)
    return (
        f"lambda={lam.real:+.8f}{lam.imag:+.2e}i, "
        f"|1-lambda|={d1:.6e}, |lambda|={mod:.6e}, "
        f"COe={w['CO_even']:.3f}, COo={w['CO_odd']:.3f}, "
        f"LCsame={w['LC_same']:.3f}, LCo={w['LC_opposite']:.3f}, "
        f"uniform={w['uniform']:.3f}"
    )



def main():
    args = _args()
    all_rows = []

    for path in args.files:
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"\n=== {path} ===", flush=True)
        d, ori, grid, params, op, tangent = _build_operator(path, args)
        q_points = _q_list(grid, args)
        print(
            f"orientation={ori}, q-count={len(q_points)}, "
            f"background residual={_scalar(d,'final_error',default=np.nan):.3e}, "
            f"bath={_scalar(d,'bath_fit_error',default=np.nan):.3e}, "
            f"tangent rank={tangent.rank}, cond={tangent.condition_number:.3e}",
            flush=True,
        )

        file_rows = []
        for iq, qn in enumerate(q_points, 1):
            print(
                f"\n--- q {iq}/{len(q_points)}: {qn} ---",
                flush=True,
            )
            lc, co, modes, solve_kind = _analyze_one_q(op, qn, args)
            print(
                "  candidates: "
                f"LC {_fmt_mode(lc)} | CO {_fmt_mode(co)}",
                flush=True,
            )
            row = dict(
                file=str(path),
                orientation=int(ori),
                q=tuple(qn),
                solve_kind=solve_kind,
                lambda_lc=complex(lc["lambda"]),
                lambda_co=complex(co["lambda"]),
                lc_same=float(lc["weight"]["LC_same"]),
                lc_opposite=float(lc["weight"]["LC_opposite"]),
                co_even=float(co["weight"]["CO_even"]),
                co_odd=float(co["weight"]["CO_odd"]),
                uniform_lc=float(lc["weight"]["uniform"]),
                uniform_co=float(co["weight"]["uniform"]),
                tangent_rank=int(tangent.rank),
                tangent_condition=float(tangent.condition_number),
                distance_lc=float(abs(1.0 - lc["lambda"])),
                distance_co=float(abs(1.0 - co["lambda"])),
                modulus_lc=float(abs(lc["lambda"])),
                modulus_co=float(abs(co["lambda"])),
            )
            file_rows.append(row)
            all_rows.append(row)

        best_lc = min(file_rows, key=lambda r: r["distance_lc"])
        best_co = min(file_rows, key=lambda r: r["distance_co"])
        print("\n=== all-q summary for this checkpoint ===")
        print(
            f"LC*: q={best_lc['q']}, lambda={best_lc['lambda_lc'].real:+.8f}"
            f"{best_lc['lambda_lc'].imag:+.2e}i, "
            f"|1-lambda|={best_lc['distance_lc']:.6e}, "
            f"same/opp={best_lc['lc_same']:.3f}/{best_lc['lc_opposite']:.3f}"
        )
        print(
            f"CO*: q={best_co['q']}, lambda={best_co['lambda_co'].real:+.8f}"
            f"{best_co['lambda_co'].imag:+.2e}i, "
            f"|1-lambda|={best_co['distance_co']:.6e}, "
            f"even/odd={best_co['co_even']:.3f}/{best_co['co_odd']:.3f}"
        )
        print(
            f"softness difference d_CO-d_LC="
            f"{best_co['distance_co']-best_lc['distance_lc']:+.6e}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out,
            source_files=np.asarray([r["file"] for r in all_rows]),
            orientation=np.asarray([r["orientation"] for r in all_rows], dtype=int),
            q_index=np.asarray([r["q"] for r in all_rows], dtype=int),
            solve_kind=np.asarray([r["solve_kind"] for r in all_rows]),
            lambda_lc=np.asarray([r["lambda_lc"] for r in all_rows]),
            lambda_co=np.asarray([r["lambda_co"] for r in all_rows]),
            lc_same_weight=np.asarray([r["lc_same"] for r in all_rows]),
            lc_opposite_weight=np.asarray([r["lc_opposite"] for r in all_rows]),
            co_even_weight=np.asarray([r["co_even"] for r in all_rows]),
            co_odd_weight=np.asarray([r["co_odd"] for r in all_rows]),
            uniform_lc_weight=np.asarray([r["uniform_lc"] for r in all_rows]),
            uniform_co_weight=np.asarray([r["uniform_co"] for r in all_rows]),
            tangent_rank=np.asarray([r["tangent_rank"] for r in all_rows], dtype=int),
            tangent_condition=np.asarray([r["tangent_condition"] for r in all_rows]),
            distance_to_one_lc=np.asarray([r["distance_lc"] for r in all_rows]),
            distance_to_one_co=np.asarray([r["distance_co"] for r in all_rows]),
            lambda_modulus_lc=np.asarray([r["modulus_lc"] for r in all_rows]),
            lambda_modulus_co=np.asarray([r["modulus_co"] for r in all_rows]),
            all_q=np.asarray(bool(args.all_q)),
            stage=np.asarray(str(args.stage)),
        )
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
