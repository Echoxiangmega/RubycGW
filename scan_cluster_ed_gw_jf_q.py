#!/usr/bin/env python3
"""Scan the Jacobian-free cluster-ED+GW pseudospin response over momentum q.

This driver starts from a converged ``run_cluster_ed_gw.py`` NPZ.  The expensive
zero-field embedding is therefore not repeated.  It builds one low-rank
finite-bath impurity tangent and reuses it for every external q and every
pseudospin channel.

The default basis is the six local Pauli-normalized pseudospins

    Ax Ay Az Bx By Bz,

rather than preselecting same/opposite combinations.  Diagonalizing the full
6x6 static response at each q therefore lets the calculation choose

* tau_x / tau_y / tau_z character,
* the relative A/B phase (even or odd), and
* the ordering momentum q

simultaneously.  For z, even/odd are the physical same/opposite loop-current
patterns used elsewhere in this repository.

The full q mesh is traversed in a serpentine path so the previous-q vertex is a
good warm start.  At each q GCROT(m,k) recycles a Krylov subspace across the six
right-hand sides.  This is substantially faster than six independent restarted
GMRES solves when the response becomes soft.

Example
-------

    python scan_cluster_ed_gw_jf_q.py \
        results/cluster_ed_gw/cluster_ed_gw_L6x6_V2_fill2.npz \
        --all-q --solver gcrotmk --out results/jf_V2.npz

For a quick coarse search, use a lower bath tangent rank, then repeat only the
leading q points with a larger rank::

    python scan_cluster_ed_gw_jf_q.py INPUT.npz --all-q --bath-rank 12
    python scan_cluster_ed_gw_jf_q.py INPUT.npz --q-index IQ1 IQ2 --bath-rank 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.cluster_ed_gw import BathParameters
from rubycgw.cluster_ed_gw_jf import (
    BathTangentOptions,
    ClusterJFOptions,
    build_embedded_jacobian,
    response_matrix,
)
from rubycgw.finite_q_cgw import (
    hermitianize_q_pair,
    negative_q_index,
    q_index_from_reduced,
    q_reduced_from_index,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import (
    available_pseudospin_channels,
    canonical_channel_name,
    primitive_pseudospin_vertex,
)
from rubycgw.supercell_gw_split import one_body_density_matrix_tail


DEFAULT_CHANNELS = ("Ax", "Ay", "Az", "Bx", "By", "Bz")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="converged cluster_ed_gw_*.npz")
    p.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    p.add_argument("--list-channels", action="store_true")

    qsel = p.add_mutually_exclusive_group()
    qsel.add_argument("--q-index", nargs=2, type=int, metavar=("IQ1", "IQ2"))
    qsel.add_argument("--q", nargs=2, type=float, metavar=("Q1", "Q2"))
    qsel.add_argument("--all-q", action="store_true")
    p.add_argument(
        "--no-add-minus-q",
        action="store_true",
        help="for a single requested q, do not automatically also solve -q",
    )

    # One-time bath tangent construction.
    p.add_argument(
        "--bath-metric",
        choices=("delta", "g0"),
        default="delta",
        help="must match the objective used for the saved zero-field bath fit",
    )
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-svd-rcond", type=float, default=1e-7)
    p.add_argument(
        "--bath-rank",
        type=int,
        default=24,
        help="maximum retained bath tangent rank; 0 means all SVD-retained modes",
    )
    p.add_argument("--bath-fd-step", type=float, default=2e-4)
    p.add_argument("--bath-fd-scheme", choices=("centered", "forward"), default="centered")
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)

    # Matrix-free outer solve.
    p.add_argument("--solver", choices=("gcrotmk", "gmres"), default="gcrotmk")
    p.add_argument("--jf-tol", type=float, default=1e-8)
    p.add_argument("--jf-maxiter", type=int, default=80)
    p.add_argument("--krylov-m", type=int, default=24)
    p.add_argument("--recycle-k", type=int, default=12)
    p.add_argument("--stage", choices=("split-mt", "full"), default="full")
    p.add_argument("--no-q-warm-start", action="store_true")
    p.add_argument("--no-rhs-recycle", action="store_true")
    p.add_argument("--quiet", action="store_true")

    p.add_argument(
        "--max-background-residual",
        type=float,
        default=5e-5,
        help="refuse a saved embedding whose reported residual is larger",
    )
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("cluster_ed_gw_jf_q.npz"))
    return p.parse_args()


def _scalar(z, key, default=None):
    if key not in z:
        if default is not None:
            return default
        raise KeyError(f"missing {key!r} in input NPZ")
    a = np.asarray(z[key])
    if a.size != 1:
        raise ValueError(f"{key!r} is not scalar")
    return a.reshape(()).item()


def _load_background(path: Path):
    with np.load(path, allow_pickle=False) as z:
        required = (
            "Lx", "Ly", "V", "filling", "T", "ti", "t1", "t2",
            "omega", "Omega", "G", "Sigma_H", "mu",
            "bath_energies", "bath_couplings",
        )
        missing = [k for k in required if k not in z]
        if missing:
            raise KeyError(f"input NPZ is missing keys: {missing}")
        data = {k: np.asarray(z[k]).copy() for k in z.files}
    return data


def _grid_from_saved(d) -> MatsubaraGrid:
    G = np.asarray(d["G"])
    omega = np.asarray(d["omega"]).reshape(-1)
    Omega = np.asarray(d["Omega"]).reshape(-1)
    if G.ndim != 5 or G.shape[-2:] != (6, 6):
        raise ValueError("saved G must have shape (nf,nk1,nk2,6,6)")
    if len(omega) != G.shape[0] or len(omega) % 2:
        raise ValueError("saved fermionic Matsubara grid is inconsistent with G")
    if len(Omega) % 2 != 1:
        raise ValueError("saved bosonic Matsubara grid must have odd length")
    grid = MatsubaraGrid(
        nk1=int(G.shape[1]),
        nk2=int(G.shape[2]),
        nw=int(len(omega) // 2),
        nOmega=int((len(Omega) - 1) // 2),
        T=float(np.asarray(d["T"]).reshape(())),
    )
    if not np.allclose(grid.omega, omega, rtol=0.0, atol=1e-12):
        raise ValueError("saved omega values do not match MatsubaraGrid")
    if not np.allclose(grid.Omega, Omega, rtol=0.0, atol=1e-12):
        raise ValueError("saved Omega values do not match MatsubaraGrid")
    return grid


def _canonical_channels(raw):
    out = []
    for name in raw:
        ch = canonical_channel_name(name)
        if ch not in out:
            out.append(ch)
    if not out:
        raise ValueError("at least one channel is required")
    return out


def _serpentine_all_q(grid):
    out = []
    for i in range(grid.nk1):
        js = range(grid.nk2) if i % 2 == 0 else range(grid.nk2 - 1, -1, -1)
        out.extend((i, j) for j in js)
    return out


def _q_points(args, grid):
    if args.all_q:
        return _serpentine_all_q(grid)
    if args.q_index is not None:
        q = (int(args.q_index[0]) % grid.nk1, int(args.q_index[1]) % grid.nk2)
    elif args.q is not None:
        q = q_index_from_reduced((float(args.q[0]), float(args.q[1])), grid)
    else:
        q = (0, 0)
    out = [q]
    qm = negative_q_index(q, grid)
    if not args.no_add_minus_q and qm != q:
        out.append(qm)
    return out


def _centered_q(q, grid):
    i, j = q
    a = i if i <= grid.nk1 // 2 else i - grid.nk1
    b = j if j <= grid.nk2 // 2 else j - grid.nk2
    return float(a) / grid.nk1, float(b) / grid.nk2


def _sorted_eigensystem(mat):
    a = 0.5 * (np.asarray(mat, dtype=complex) + np.asarray(mat, dtype=complex).conj().T)
    vals, vecs = np.linalg.eigh(a)
    order = np.argsort(vals.real)[::-1]
    return np.asarray(vals[order].real), np.asarray(vecs[:, order])


def _local_to_even_odd(vec, channels):
    """Return Pauli component weights and A/B parity amplitudes when available."""
    c = {ch: complex(vec[i]) for i, ch in enumerate(channels)}
    out = {}
    for comp in "xyz":
        A = c.get(f"A{comp}")
        B = c.get(f"B{comp}")
        if A is None or B is None:
            continue
        out[f"{comp}_even"] = (A + B) / np.sqrt(2.0)
        out[f"{comp}_odd"] = (A - B) / np.sqrt(2.0)
        out[f"{comp}_weight"] = abs(A) ** 2 + abs(B) ** 2
    return out


def _mode_summary(vec, channels):
    eo = _local_to_even_odd(vec, channels)
    weights = [(comp, float(eo.get(f"{comp}_weight", 0.0))) for comp in "xyz"]
    dominant = max(weights, key=lambda x: x[1]) if weights else ("?", np.nan)
    if dominant[0] in "xyz":
        even = eo.get(f"{dominant[0]}_even", 0.0j)
        odd = eo.get(f"{dominant[0]}_odd", 0.0j)
        parity = "even" if abs(even) >= abs(odd) else "odd"
    else:
        parity = "?"
    if dominant[0] == "z":
        parity = "same" if parity == "even" else "opposite"
    return dominant[0], dominant[1], parity, eo


def main():
    args = _args()
    if args.list_channels:
        print("\n".join(available_pseudospin_channels()))
        return
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    d = _load_background(args.input)
    converged = bool(np.asarray(d.get("converged", True)).reshape(()))
    residual = float(np.asarray(d.get("final_error", np.nan)).reshape(()))
    if not converged:
        raise RuntimeError("saved cluster-ED+GW background is marked unconverged")
    if np.isfinite(residual) and residual > float(args.max_background_residual):
        raise RuntimeError(
            f"saved embedding residual {residual:.3e} exceeds "
            f"--max-background-residual={args.max_background_residual:.3e}"
        )

    grid = _grid_from_saved(d)
    params = RubyParameters(
        ti=float(np.asarray(d["ti"]).reshape(())),
        t1=float(np.asarray(d["t1"]).reshape(())),
        t2=float(np.asarray(d["t2"]).reshape(())),
        V=float(np.asarray(d["V"]).reshape(())),
    )
    G = np.asarray(d["G"], dtype=complex)
    sigma_h = np.asarray(d["Sigma_H"], dtype=complex)
    mu = float(np.asarray(d["mu"]).reshape(()))
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_interaction(grid.qmesh(), params)
    rho_k = one_body_density_matrix_tail(G, grid, h0, mu, sigma_h)
    rho_c = np.mean(rho_k, axis=(0, 1))
    h_cluster = np.mean(h0, axis=(0, 1))
    h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)
    bath = BathParameters(
        np.asarray(d["bath_energies"], dtype=float),
        np.asarray(d["bath_couplings"], dtype=complex),
        float(np.asarray(d.get("bath_fit_error", np.nan)).reshape(())),
        0,
    )

    channels = _canonical_channels(args.channels)
    vertices = np.stack([primitive_pseudospin_vertex(ch) for ch in channels])
    q_points = _q_points(args, grid)

    max_rank = None if int(args.bath_rank) <= 0 else int(args.bath_rank)
    bath_opts = BathTangentOptions(
        fit_metric=args.bath_metric,
        nfit=int(args.bath_fit_nfreq),
        svd_rcond=float(args.bath_svd_rcond),
        max_rank=max_rank,
        fd_step=float(args.bath_fd_step),
        fd_scheme=args.bath_fd_scheme,
        discard_weight_tol=float(args.discard_weight_tol),
        verbose=not args.quiet,
    )
    jf_opts = ClusterJFOptions(
        solver=args.solver,
        tol=float(args.jf_tol),
        maxiter=int(args.jf_maxiter),
        restart=int(args.krylov_m),
        recycle_dim=int(args.recycle_k),
        include_hartree=True,
        include_fock=True,
        include_mt=True,
        include_al=(args.stage == "full"),
        momentum_backend="fft",
        verbose=not args.quiet,
    )

    print(
        "=== cluster ED+GW Jacobian-free response ===\n"
        f"input={args.input}\n"
        f"V={params.V:g}, filling={float(np.asarray(d['filling']).reshape(())):g}, "
        f"T={grid.T:g}, nk={grid.nk1}x{grid.nk2}, nw={grid.nw}, nOmega={grid.nOmega}\n"
        f"channels={channels}\n"
        f"q-points={len(q_points)}, solver={args.solver}, stage={args.stage}\n"
        f"nbath={len(bath.energies)}, bath_metric={args.bath_metric}, "
        f"bath_rank={'all' if max_rank is None else max_rank}, fd={args.bath_fd_scheme}",
        flush=True,
    )

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

    nq = len(q_points)
    nc = len(channels)
    chi_raw = np.zeros((nq, nc, nc), dtype=complex)
    iterations = np.zeros((nq, nc), dtype=int)
    response_residuals = np.zeros((nq, nc), dtype=float)
    previous = [None] * nc

    for iq, qidx in enumerate(q_points):
        print(
            f"\n=== q {iq+1}/{nq}: index={qidx}, "
            f"reduced={q_reduced_from_index(qidx, grid)}, centered={_centered_q(qidx, grid)} ===",
            flush=True,
        )
        initial = [None] * nc if args.no_q_warm_start else previous
        chi, results = response_matrix(
            op,
            vertices,
            qidx,
            initial_gammas=initial,
            recycle=not args.no_rhs_recycle,
        )
        chi_raw[iq] = chi
        for j, result in enumerate(results):
            iterations[iq, j] = int(result.iterations)
            response_residuals[iq, j] = float(result.final_error)
            previous[j] = result.Gamma

    # Equilibrium reciprocity is chi_ab(q)=chi_ba(-q)^*.  Hermitianize using the
    # explicitly computed q/-q pair whenever available.
    q_to_pos = {tuple(q): i for i, q in enumerate(q_points)}
    chi_herm = np.empty_like(chi_raw)
    pair_complete = np.zeros(nq, dtype=bool)
    for i, qidx in enumerate(q_points):
        qm = negative_q_index(qidx, grid)
        if qm in q_to_pos:
            chi_herm[i] = hermitianize_q_pair(chi_raw[i], chi_raw[q_to_pos[qm]])
            pair_complete[i] = True
        else:
            chi_herm[i] = 0.5 * (chi_raw[i] + chi_raw[i].conj().T)

    eigenvalues = np.zeros((nq, nc), dtype=float)
    eigenvectors = np.zeros((nq, nc, nc), dtype=complex)
    for i in range(nq):
        eigenvalues[i], eigenvectors[i] = _sorted_eigensystem(chi_herm[i])

    leading = eigenvalues[:, 0]
    order = np.argsort(leading)[::-1]
    print("\n=== leading pseudospin modes over q ===", flush=True)
    for rank, pos in enumerate(order[: max(int(args.top), 1)], start=1):
        vec = eigenvectors[pos, :, 0]
        comp, weight, parity, _ = _mode_summary(vec, channels)
        print(
            f"{rank:2d}: q={_centered_q(q_points[pos], grid)}, "
            f"lambda={leading[pos]:+.10e}, tau_{comp} weight={weight:.4f}, "
            f"A/B={parity}",
            flush=True,
        )

    # Store even/odd projections of every leading mode when the full local basis
    # is present.  This makes z_same/z_opposite and x/y parity immediately visible.
    projected_names = np.asarray(
        ["x_even", "x_odd", "y_even", "y_odd", "z_even", "z_odd"]
    )
    projected = np.full((nq, 6), np.nan + 1j * np.nan, dtype=complex)
    if all(ch in channels for ch in DEFAULT_CHANNELS):
        for i in range(nq):
            _, _, _, eo = _mode_summary(eigenvectors[i, :, 0], channels)
            projected[i] = np.asarray([eo[name] for name in projected_names])

    out = args.out
    if out.suffix.lower() != ".npz":
        out = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        source_file=str(args.input),
        V=float(params.V),
        filling=float(np.asarray(d["filling"]).reshape(())),
        T=float(grid.T),
        nk1=int(grid.nk1),
        nk2=int(grid.nk2),
        nw=int(grid.nw),
        nOmega=int(grid.nOmega),
        channels=np.asarray(channels),
        q_indices=np.asarray(q_points, dtype=int),
        q_reduced=np.asarray([q_reduced_from_index(q, grid) for q in q_points]),
        q_centered=np.asarray([_centered_q(q, grid) for q in q_points]),
        chi_raw=chi_raw,
        chi_hermitian=chi_herm,
        q_pair_complete=pair_complete,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        lambda_max=leading,
        leading_projection_names=projected_names,
        leading_projection=projected,
        iterations=iterations,
        response_residuals=response_residuals,
        solver=str(args.solver),
        stage=str(args.stage),
        background_residual=float(residual),
        background_bath_fit_error=float(bath.fit_error),
        bath_metric=str(args.bath_metric),
        bath_tangent_rank=int(tangent.rank),
        bath_tangent_singular_values=np.asarray(tangent.singular_values),
        bath_tangent_condition=float(tangent.condition_number),
        bath_tangent_build_seconds=float(tangent.build_seconds),
        bath_fd_step=float(tangent.fd_step),
        bath_fit_nfreq=int(args.bath_fit_nfreq),
        bath_svd_rcond=float(args.bath_svd_rcond),
        z_is_pauli_normalized=np.asarray(True),
    )
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
