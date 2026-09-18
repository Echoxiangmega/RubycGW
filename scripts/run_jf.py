#!/usr/bin/env python3
"""Scan the Jacobian-free cluster-ED+GW pseudospin response over momentum q.

The expensive zero-field embedding and finite-bath tangent are built once.  The
six local Pauli-normalized channels ``Ax Ay Az Bx By Bz`` are then solved at
each external q, so the leading Pauli character, A/B parity and ordering
momentum are selected by the response itself.

Long all-q scans are deliberately fault tolerant.  A failed Krylov solve is
retried from a fresh vertex with a larger Krylov space, followed (for GCROT) by
a GMRES fallback.  Every completed q is written to ``*.partial.npz`` by
default, so a later soft-mode failure or interruption does not discard earlier
q points.  A successful scan writes the requested output and removes the
partial checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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
PROJECTED_NAMES = np.asarray(
    ["x_even", "x_odd", "y_even", "y_odd", "z_even", "z_odd"]
)


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

    # Automatic soft-mode retry.  The retry deliberately throws away q/RHS
    # recycled vectors so a poisoned Krylov subspace cannot make the diagnosis
    # ambiguous.
    p.add_argument(
        "--retry-maxiter",
        type=int,
        default=300,
        help="maximum iterations for the fresh retry/fallback solve",
    )
    p.add_argument(
        "--retry-krylov-m",
        type=int,
        default=48,
        help="Krylov restart dimension for the fresh retry/fallback solve",
    )
    p.add_argument("--no-auto-retry", action="store_true")
    p.add_argument(
        "--no-gmres-fallback",
        action="store_true",
        help="after a failed GCROT retry, do not try fresh GMRES",
    )

    # Crash/interruption-safe partial output.
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="write the partial NPZ after this many newly completed q points",
    )
    p.add_argument("--no-partial-checkpoint", action="store_true")

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
        return {k: np.asarray(z[k]).copy() for k in z.files}


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


def _normalise_out(path: Path) -> Path:
    path = Path(path)
    return path if path.suffix.lower() == ".npz" else path.with_suffix(".npz")


def _partial_path(path: Path) -> Path:
    out = _normalise_out(path)
    return out.with_name(out.stem + ".partial.npz")


def _analyse_completed(chi_raw, completed_q, q_points, grid, channels):
    """Build reciprocal/Hermitian analysis without treating missing q as zero."""
    chi_raw = np.asarray(chi_raw, dtype=complex)
    completed_q = np.asarray(completed_q, dtype=bool)
    nq, nc, _ = chi_raw.shape
    nan_c = np.nan + 1j * np.nan
    chi_herm = np.full_like(chi_raw, nan_c)
    pair_complete = np.zeros(nq, dtype=bool)
    q_to_pos = {tuple(q): i for i, q in enumerate(q_points)}

    for i, qidx in enumerate(q_points):
        if not completed_q[i]:
            continue
        qm = negative_q_index(qidx, grid)
        j = q_to_pos.get(qm)
        if j is not None and completed_q[j]:
            chi_herm[i] = hermitianize_q_pair(chi_raw[i], chi_raw[j])
            pair_complete[i] = True
        else:
            # Still useful in a partial checkpoint, but explicitly marked as
            # missing its reciprocity partner when q != -q.
            chi_herm[i] = 0.5 * (chi_raw[i] + chi_raw[i].conj().T)

    eigenvalues = np.full((nq, nc), np.nan, dtype=float)
    eigenvectors = np.full((nq, nc, nc), nan_c, dtype=complex)
    for i in np.flatnonzero(completed_q):
        eigenvalues[i], eigenvectors[i] = _sorted_eigensystem(chi_herm[i])
    leading = eigenvalues[:, 0]

    projected = np.full((nq, 6), nan_c, dtype=complex)
    if all(ch in channels for ch in DEFAULT_CHANNELS):
        for i in np.flatnonzero(completed_q):
            _, _, _, eo = _mode_summary(eigenvectors[i, :, 0], channels)
            projected[i] = np.asarray([eo[name] for name in PROJECTED_NAMES])
    return chi_herm, pair_complete, eigenvalues, eigenvectors, leading, projected


def _atomic_savez(path: Path, **payload) -> None:
    path = _normalise_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def _save_scan(
    path,
    *,
    args,
    d,
    params,
    grid,
    bath,
    tangent,
    channels,
    q_points,
    chi_raw,
    completed_q,
    iterations,
    response_residuals,
    retry_counts,
    solver_used,
    scan_complete,
    failed_q_index=(-1, -1),
    failure_message="",
):
    chi_herm, pair_complete, eigenvalues, eigenvectors, leading, projected = (
        _analyse_completed(chi_raw, completed_q, q_points, grid, channels)
    )
    _atomic_savez(
        Path(path),
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
        completed_q=np.asarray(completed_q, dtype=bool),
        completed_q_count=int(np.count_nonzero(completed_q)),
        scan_complete=np.asarray(bool(scan_complete)),
        failed_q_index=np.asarray(failed_q_index, dtype=int),
        failure_message=np.asarray(str(failure_message)),
        chi_raw=np.asarray(chi_raw),
        chi_hermitian=chi_herm,
        q_pair_complete=pair_complete,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        lambda_max=leading,
        leading_projection_names=PROJECTED_NAMES,
        leading_projection=projected,
        iterations=np.asarray(iterations),
        response_residuals=np.asarray(response_residuals),
        retry_counts=np.asarray(retry_counts, dtype=int),
        solver_used=np.asarray(solver_used),
        solver=str(args.solver),
        stage=str(args.stage),
        jf_maxiter=int(args.jf_maxiter),
        krylov_m=int(args.krylov_m),
        retry_maxiter=int(args.retry_maxiter),
        retry_krylov_m=int(args.retry_krylov_m),
        auto_retry=np.asarray(not bool(args.no_auto_retry)),
        gmres_fallback=np.asarray(not bool(args.no_gmres_fallback)),
        background_residual=float(np.asarray(d.get("final_error", np.nan)).reshape(())),
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
    return leading, eigenvectors


def _solve_q_with_retry(op, vertices, qidx, initial, args):
    """Solve one q, retrying the whole six-RHS block only after failure."""
    original_opts = op.opts
    recycle = not bool(args.no_rhs_recycle)
    try:
        try:
            chi, results = response_matrix(
                op,
                vertices,
                qidx,
                initial_gammas=initial,
                recycle=recycle,
            )
            return chi, results, 0, str(original_opts.solver)
        except RuntimeError as first_error:
            if args.no_auto_retry:
                raise
            retry_max = max(int(args.retry_maxiter), int(original_opts.maxiter))
            retry_m = max(int(args.retry_krylov_m), int(original_opts.restart))
            retry_opts = replace(
                original_opts,
                maxiter=retry_max,
                restart=retry_m,
            )
            op.opts = retry_opts
            print(
                f"[JF retry] q={qidx}: primary solve failed ({first_error}); "
                f"fresh {retry_opts.solver} with maxiter={retry_max}, m={retry_m}",
                flush=True,
            )
            try:
                chi, results = response_matrix(
                    op,
                    vertices,
                    qidx,
                    initial_gammas=[None] * len(vertices),
                    recycle=False,
                )
                return chi, results, 1, str(retry_opts.solver)
            except RuntimeError as retry_error:
                if str(original_opts.solver).lower() != "gcrotmk" or args.no_gmres_fallback:
                    raise RuntimeError(
                        f"JF q={qidx} failed primary and fresh retry; "
                        f"last error: {retry_error}"
                    ) from retry_error

                gmres_opts = replace(retry_opts, solver="gmres")
                op.opts = gmres_opts
                print(
                    f"[JF retry] q={qidx}: enlarged GCROT failed ({retry_error}); "
                    f"fresh GMRES fallback with maxiter={retry_max}, restart={retry_m}",
                    flush=True,
                )
                try:
                    chi, results = response_matrix(
                        op,
                        vertices,
                        qidx,
                        initial_gammas=[None] * len(vertices),
                        recycle=False,
                    )
                    return chi, results, 2, "gmres"
                except RuntimeError as gmres_error:
                    raise RuntimeError(
                        f"JF q={qidx} failed primary, enlarged GCROT, and GMRES; "
                        f"last error: {gmres_error}"
                    ) from gmres_error
    finally:
        op.opts = original_opts


def main():
    args = _args()
    if args.list_channels:
        print("\n".join(available_pseudospin_channels()))
        return
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if int(args.checkpoint_every) < 1:
        raise ValueError("--checkpoint-every must be >= 1")
    if int(args.retry_maxiter) < 1 or int(args.retry_krylov_m) < 1:
        raise ValueError("retry iteration/Krylov sizes must be positive")

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
    if "impurity_static_shift" in d:
        static_shift = np.asarray(d["impurity_static_shift"], dtype=complex)
        if static_shift.shape != (6, 6):
            raise ValueError("saved impurity_static_shift must have shape (6,6)")
        h_cluster = h_cluster + 0.5 * (static_shift + static_shift.conj().T)
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
        f"bath_rank={'all' if max_rank is None else max_rank}, fd={args.bath_fd_scheme}\n"
        f"retry={'off' if args.no_auto_retry else f'{args.retry_maxiter}/{args.retry_krylov_m}'}; "
        f"partial-checkpoint={'off' if args.no_partial_checkpoint else f'every {args.checkpoint_every} q'}",
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
    nan_c = np.nan + 1j * np.nan
    chi_raw = np.full((nq, nc, nc), nan_c, dtype=complex)
    iterations = np.full((nq, nc), -1, dtype=int)
    response_residuals = np.full((nq, nc), np.nan, dtype=float)
    retry_counts = np.full(nq, -1, dtype=int)
    solver_used = np.full(nq, "", dtype="U16")
    completed_q = np.zeros(nq, dtype=bool)
    previous = [None] * nc

    out = _normalise_out(args.out)
    partial = _partial_path(out)

    for iq, qidx in enumerate(q_points):
        print(
            f"\n=== q {iq+1}/{nq}: index={qidx}, "
            f"reduced={q_reduced_from_index(qidx, grid)}, centered={_centered_q(qidx, grid)} ===",
            flush=True,
        )
        initial = [None] * nc if args.no_q_warm_start else previous
        try:
            chi, results, nretry, used_solver = _solve_q_with_retry(
                op, vertices, qidx, initial, args
            )
        except BaseException as exc:
            if not args.no_partial_checkpoint:
                _save_scan(
                    partial,
                    args=args,
                    d=d,
                    params=params,
                    grid=grid,
                    bath=bath,
                    tangent=tangent,
                    channels=channels,
                    q_points=q_points,
                    chi_raw=chi_raw,
                    completed_q=completed_q,
                    iterations=iterations,
                    response_residuals=response_residuals,
                    retry_counts=retry_counts,
                    solver_used=solver_used,
                    scan_complete=False,
                    failed_q_index=qidx,
                    failure_message=str(exc) or exc.__class__.__name__,
                )
                print(
                    f"[checkpoint] failure/interruption saved to {partial} "
                    f"({np.count_nonzero(completed_q)}/{nq} q complete)",
                    flush=True,
                )
            raise

        chi_raw[iq] = chi
        completed_q[iq] = True
        retry_counts[iq] = int(nretry)
        solver_used[iq] = str(used_solver)
        for j, result in enumerate(results):
            iterations[iq, j] = int(result.iterations)
            response_residuals[iq, j] = float(result.final_error)
            previous[j] = result.Gamma

        if (
            not args.no_partial_checkpoint
            and ((int(np.count_nonzero(completed_q)) % int(args.checkpoint_every) == 0) or iq == nq - 1)
        ):
            _save_scan(
                partial,
                args=args,
                d=d,
                params=params,
                grid=grid,
                bath=bath,
                tangent=tangent,
                channels=channels,
                q_points=q_points,
                chi_raw=chi_raw,
                completed_q=completed_q,
                iterations=iterations,
                response_residuals=response_residuals,
                retry_counts=retry_counts,
                solver_used=solver_used,
                scan_complete=False,
            )
            print(
                f"[checkpoint] saved {partial} "
                f"({np.count_nonzero(completed_q)}/{nq} q complete)",
                flush=True,
            )

    leading, eigenvectors = _save_scan(
        out,
        args=args,
        d=d,
        params=params,
        grid=grid,
        bath=bath,
        tangent=tangent,
        channels=channels,
        q_points=q_points,
        chi_raw=chi_raw,
        completed_q=completed_q,
        iterations=iterations,
        response_residuals=response_residuals,
        retry_counts=retry_counts,
        solver_used=solver_used,
        scan_complete=True,
    )

    order = np.argsort(leading)[::-1]
    print("\n=== leading pseudospin modes over q ===", flush=True)
    shown = 0
    for pos in order:
        if not np.isfinite(leading[pos]):
            continue
        vec = eigenvectors[pos, :, 0]
        comp, weight, parity, _ = _mode_summary(vec, channels)
        shown += 1
        print(
            f"{shown:2d}: q={_centered_q(q_points[pos], grid)}, "
            f"lambda={leading[pos]:+.10e}, tau_{comp} weight={weight:.4f}, "
            f"A/B={parity}",
            flush=True,
        )
        if shown >= max(int(args.top), 1):
            break

    if partial.exists():
        try:
            partial.unlink()
        except OSError:
            pass
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
