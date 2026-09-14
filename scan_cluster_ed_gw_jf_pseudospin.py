#!/usr/bin/env python3
"""Finite-q pseudospin stability scan for a converged cluster-ED+GW embedding.

The scan uses the six local Pauli-normalized triangle operators

    Ax, Ay, Az, Bx, By, Bz

by default. At each reciprocal-mesh momentum q it solves the Jacobian-free
linear response of the embedded approximation, builds the complete channel
susceptibility matrix, pairs q with -q, and diagonalizes the Hermitian static
response. The leading eigenvector therefore determines both the pseudospin
orientation (tau_x/tau_y/tau_z) and the relative A/B pattern instead of assuming
an even/odd channel in advance.

This is a linear-stability calculation. The largest susceptibility / softest
inverse-susceptibility mode identifies the channel that wins a continuous
instability of the chosen zero-field branch. It cannot by itself prove the
global winner across a first-order transition; competing broken-symmetry
branches must still be compared by free energy in that case.
"""
from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np

from rubycgw.cluster_ed_gw import BathParameters, ruby_cluster_interactions
from rubycgw.cluster_ed_gw_jf import ClusterJFOptions, scan_pseudospin_q
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import primitive_pseudospin_vertex


DEFAULT_CHANNELS = ("Ax", "Ay", "Az", "Bx", "By", "Bz")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="converged cluster_ed_gw_*.npz")
    p.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    p.add_argument(
        "--q-index",
        nargs=2,
        type=int,
        action="append",
        default=None,
        metavar=("IQ1", "IQ2"),
        help="mesh q index; repeat for multiple q. Default scans the full mesh.",
    )
    p.add_argument("--allow-unconverged", action="store_true")

    p.add_argument("--jf-rtol", type=float, default=2e-5)
    p.add_argument("--jf-atol", type=float, default=0.0)
    p.add_argument("--jf-maxiter", type=int, default=80)
    p.add_argument("--gcrot-m", type=int, default=20)
    p.add_argument("--gcrot-k", type=int, default=8)
    p.add_argument("--preconditioner-order", type=int, default=1)

    p.add_argument(
        "--bath-derivative",
        choices=("linearized", "refit"),
        default="linearized",
        help="linearized is much faster; refit is intended as a validation mode",
    )
    p.add_argument(
        "--difference-scheme",
        choices=("forward", "central"),
        default="forward",
        help="forward costs one ED solve per Jv; central costs two and is more accurate",
    )
    p.add_argument("--fd-rel-step", type=float, default=2e-5)
    p.add_argument("--fd-min-step", type=float, default=1e-8)
    p.add_argument("--fd-max-step", type=float, default=5e-3)

    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-fit-max-nfev", type=int, default=400)
    p.add_argument("--bath-fit-xtol", type=float, default=1e-10)
    p.add_argument("--bath-energy-window", type=float, default=4.0)
    p.add_argument("--bath-coupling-bound", type=float, default=4.0)
    p.add_argument("--bath-svd-rcond", type=float, default=1e-9)
    p.add_argument("--bath-tikhonov", type=float, default=0.0)
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)

    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _full_q_order(nk1: int, nk2: int) -> list[tuple[int, int]]:
    """Serpentine order keeps successive q points close for warm starts."""
    out = []
    for i in range(int(nk1)):
        row = list(range(int(nk2)))
        if i % 2:
            row.reverse()
        out.extend((i, j) for j in row)
    return out


def _qneg(q: tuple[int, int], nk1: int, nk2: int) -> tuple[int, int]:
    return ((-int(q[0])) % int(nk1), (-int(q[1])) % int(nk2))


def _component_report(vec: np.ndarray, labels: list[str]) -> str:
    v = np.asarray(vec, dtype=complex).reshape(-1)
    w = np.abs(v) ** 2
    if float(np.sum(w)) > 0.0:
        w = w / np.sum(w)
    order = np.argsort(w)[::-1]
    chunks = [f"{labels[i]}:{w[i]:.3f}" for i in order[: min(4, len(order))]]
    if labels == list(DEFAULT_CHANNELS):
        vv = {lab: v[i] for i, lab in enumerate(labels)}
        eo = {}
        for comp in "xyz":
            A = vv[f"A{comp}"]
            B = vv[f"B{comp}"]
            eo[f"{comp}_even"] = (A + B) / np.sqrt(2.0)
            eo[f"{comp}_odd"] = (A - B) / np.sqrt(2.0)
        norm = sum(abs(x) ** 2 for x in eo.values())
        eo_order = sorted(eo, key=lambda x: abs(eo[x]) ** 2, reverse=True)
        eo_text = ", ".join(
            f"{x}:{(abs(eo[x])**2/max(norm,1e-300)):.3f}" for x in eo_order[:3]
        )
        return ", ".join(chunks) + " | A/B basis -> " + eo_text
    return ", ".join(chunks)


def main():
    args = _args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    with np.load(args.input, allow_pickle=False) as z:
        required = (
            "Lx", "Ly", "V", "filling", "T", "ti", "t1", "t2",
            "omega", "Omega", "G", "W", "Sigma_H", "Sigma_ED_cluster",
            "G_cluster", "bath_energies", "bath_couplings", "mu", "converged",
        )
        missing = [k for k in required if k not in z]
        if missing:
            raise KeyError(f"embedding file is missing keys: {missing}")
        Lx, Ly = int(z["Lx"]), int(z["Ly"])
        V = float(z["V"])
        filling = float(z["filling"])
        T = float(z["T"])
        ti, t1, t2 = float(z["ti"]), float(z["t1"]), float(z["t2"])
        omega = np.asarray(z["omega"], dtype=float)
        Omega = np.asarray(z["Omega"], dtype=float)
        G = np.asarray(z["G"], dtype=complex)
        W = np.asarray(z["W"], dtype=complex)
        sigma_h = np.asarray(z["Sigma_H"], dtype=complex)
        sigma_imp = np.asarray(z["Sigma_ED_cluster"], dtype=complex)
        Gc = np.asarray(z["G_cluster"], dtype=complex)
        mu = float(z["mu"])
        converged = bool(z["converged"])
        base_residual = float(z["final_error"]) if "final_error" in z else np.nan
        saved_bath_error = float(z["bath_fit_error"]) if "bath_fit_error" in z else np.nan
        bath = BathParameters(
            np.asarray(z["bath_energies"], dtype=float),
            np.asarray(z["bath_couplings"], dtype=complex),
            saved_bath_error,
            0,
        )

    if not converged and not args.allow_unconverged:
        raise RuntimeError(
            f"input embedding is not converged (residual={base_residual:.3e}); "
            "rerun it or pass --allow-unconverged for diagnostics only"
        )
    if len(omega) % 2 != 0 or len(Omega) % 2 != 1:
        raise ValueError("stored Matsubara grids have unexpected lengths")
    grid = MatsubaraGrid(
        nk1=Lx,
        nk2=Ly,
        nw=len(omega) // 2,
        nOmega=(len(Omega) - 1) // 2,
        T=T,
    )
    if np.max(np.abs(grid.omega - omega)) > 1e-10:
        raise ValueError("stored fermion grid does not match reconstructed MatsubaraGrid")

    params = RubyParameters(ti=ti, t1=t1, t2=t2, V=V)
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_interaction(grid.qmesh(), params)
    h_cluster = np.mean(h0, axis=(0, 1))
    h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)
    interactions = ruby_cluster_interactions(params)

    labels = [str(x) for x in args.channels]
    vertices = np.stack([primitive_pseudospin_vertex(x) for x in labels], axis=0)
    q_indices = (
        _full_q_order(Lx, Ly)
        if args.q_index is None
        else [((int(q[0]) % Lx), (int(q[1]) % Ly)) for q in args.q_index]
    )
    q_indices = list(dict.fromkeys(q_indices))

    opts = ClusterJFOptions(
        rtol=float(args.jf_rtol),
        atol=float(args.jf_atol),
        maxiter=int(args.jf_maxiter),
        gcrot_m=int(args.gcrot_m),
        gcrot_k=int(args.gcrot_k),
        preconditioner_order=int(args.preconditioner_order),
        bath_derivative_mode=str(args.bath_derivative),
        difference_scheme=str(args.difference_scheme),
        fd_rel_step=float(args.fd_rel_step),
        fd_min_step=float(args.fd_min_step),
        fd_max_step=float(args.fd_max_step),
        bath_fit_nfreq=int(args.bath_fit_nfreq),
        bath_fit_max_nfev=int(args.bath_fit_max_nfev),
        bath_fit_xtol=float(args.bath_fit_xtol),
        bath_energy_window=float(args.bath_energy_window),
        bath_coupling_bound=float(args.bath_coupling_bound),
        bath_svd_rcond=float(args.bath_svd_rcond),
        bath_tikhonov=float(args.bath_tikhonov),
        discard_weight_tol=float(args.discard_weight_tol),
        verbose=not bool(args.quiet),
    )

    nbath = len(bath.energies)
    norb_imp = 6 + nbath
    max_sector = comb(norb_imp, norb_imp // 2)
    print("=== cluster ED+GW Jacobian-free pseudospin scan ===", flush=True)
    print(
        f"mesh={Lx}x{Ly}, V={V:g}, filling={filling:g}, T={T:g}, "
        f"channels={labels}",
        flush=True,
    )
    print(
        f"bath: nbath={nbath} -> impurity orbitals={norb_imp}, Fock_dim={2**norb_imp}, "
        f"largest fixed-N sector={max_sector}, saved_fit_error={saved_bath_error:.3e}, "
        f"derivative={args.bath_derivative}/{args.difference_scheme}",
        flush=True,
    )
    print(
        f"Krylov: GCROT(m={args.gcrot_m},k={args.gcrot_k}), rtol={args.jf_rtol:.1e}, "
        f"preconditioner_order={args.preconditioner_order}; q-points={len(q_indices)}",
        flush=True,
    )

    points = scan_pseudospin_q(
        G=G,
        W=W,
        Vq=Vq,
        G_cluster=Gc,
        Sigma_imp=sigma_imp,
        bath=bath,
        h_cluster=h_cluster,
        interactions=interactions,
        mu=mu,
        grid=grid,
        vertices=vertices,
        q_indices=q_indices,
        opts=opts,
    )

    point_map = {tuple(p.q_index): p for p in points}
    nq, nch = len(points), len(labels)
    q_arr = np.asarray([p.q_index for p in points], dtype=int)
    q_red = np.asarray([[q[0] / Lx, q[1] / Ly] for q in q_arr], dtype=float)
    chi_raw = np.stack([p.chi for p in points])
    chi_pair = np.empty_like(chi_raw)
    evals = np.empty((nq, nch), dtype=float)
    evecs = np.empty((nq, nch, nch), dtype=complex)
    residuals = np.empty((nq, nch), dtype=float)
    matvecs = np.empty((nq, nch), dtype=int)
    elapsed = np.empty((nq, nch), dtype=float)
    infos = np.empty((nq, nch), dtype=int)
    dmu = np.empty((nq, nch), dtype=complex)

    for iq, p in enumerate(points):
        q = tuple(p.q_index)
        qm = _qneg(q, Lx, Ly)
        pm = point_map.get(qm)
        if pm is None:
            hp = 0.5 * (p.chi + p.chi.conj().T)
        else:
            hp = 0.5 * (p.chi + pm.chi.conj().T)
        chi_pair[iq] = hp
        vals, vecs = np.linalg.eigh(hp)
        order = np.argsort(vals.real)[::-1]
        evals[iq] = vals[order].real
        evecs[iq] = vecs[:, order]
        for a, r in enumerate(p.results):
            residuals[iq, a] = r.residual_max
            matvecs[iq, a] = r.matvecs
            elapsed[iq, a] = r.elapsed
            infos[iq, a] = r.info
            dmu[iq, a] = r.dmu

    best_iq = int(np.nanargmax(evals[:, 0]))
    best_q = tuple(q_arr[best_iq])
    best_v = evecs[best_iq, :, 0]
    print("\n=== leading linear instability ===", flush=True)
    print(
        f"q_index={best_q}, q_reduced=({q_red[best_iq,0]:.6g},{q_red[best_iq,1]:.6g}), "
        f"lambda_max(chi)={evals[best_iq,0]:+.10e}",
        flush=True,
    )
    print("mode weights: " + _component_report(best_v, labels), flush=True)
    print(
        f"worst JF residual={np.nanmax(residuals):.3e}, total matvecs={int(np.sum(matvecs))}, "
        f"total Krylov walltime={np.sum(elapsed):.1f}s",
        flush=True,
    )

    outfile = args.out
    if outfile is None:
        outfile = args.input.with_name(args.input.stem + "_jf_pseudospin_q.npz")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outfile,
        source_file=str(args.input),
        channels=np.asarray(labels, dtype="U32"),
        q_index=q_arr,
        q_reduced=q_red,
        chi_raw=chi_raw,
        chi_hermitian=chi_pair,
        eigenvalues=evals,
        eigenvectors=evecs,
        leading_q_index=np.asarray(best_q, dtype=int),
        leading_q_reduced=q_red[best_iq],
        leading_eigenvalue=float(evals[best_iq, 0]),
        leading_eigenvector=best_v,
        solver_residuals=residuals,
        solver_matvecs=matvecs,
        solver_elapsed=elapsed,
        solver_info=infos,
        dmu=dmu,
        base_embedding_residual=float(base_residual),
        saved_bath_fit_error=float(saved_bath_error),
        nbath=int(nbath),
        bath_derivative_mode=str(args.bath_derivative),
        difference_scheme=str(args.difference_scheme),
        fd_rel_step=float(args.fd_rel_step),
        jf_rtol=float(args.jf_rtol),
        gcrot_m=int(args.gcrot_m),
        gcrot_k=int(args.gcrot_k),
        preconditioner_order=int(args.preconditioner_order),
    )
    print(f"saved {outfile}", flush=True)


if __name__ == "__main__":
    main()
