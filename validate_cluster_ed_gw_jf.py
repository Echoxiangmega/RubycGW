#!/usr/bin/env python3
"""Validate q=0 cluster-ED+GW Jacobian-free response against finite sources.

The finite-source calculation is deliberately expensive and is not intended for
production q scans.  Its purpose is to validate the SVD-compressed finite-bath
impurity tangent used by :mod:`rubycgw.cluster_ed_gw_jf`.

The saved embedding is first relaxed once with the complex-bath source map at
h=0.  Both JF and +/-h finite-source derivatives are then evaluated around that
same refined state.  For symmetry-traceless pseudospin channels on an unbroken
background, the fixed-filling finite-source derivative should agree with the
JF derivative up to bath-tangent, Krylov and finite-h errors.

Example
-------

    python validate_cluster_ed_gw_jf.py INPUT.npz \
        --channels Ax Ay Az --h 1e-3 5e-4 \
        --bath-rank 24 --out results/jf_validation.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.cluster_ed_gw import BathParameters
from rubycgw.cluster_ed_gw_covariant import (
    ClusterCovariantOptions,
    ClusterCovariantState,
    onebody_expectation_from_lattice_G,
    solve_cluster_source_warm,
)
from rubycgw.cluster_ed_gw_jf import (
    BathTangentOptions,
    ClusterJFOptions,
    build_embedded_jacobian,
    response_matrix,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import canonical_channel_name, primitive_pseudospin_vertex
from rubycgw.supercell_gw_split import one_body_density_matrix_tail


DEFAULT_CHANNELS = ("Ax", "Ay", "Az")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    p.add_argument("--h", nargs="+", type=float, default=[1e-3, 5e-4])

    p.add_argument("--response-max", type=int, default=80)
    p.add_argument("--response-tol", type=float, default=1e-7)
    p.add_argument("--response-mixing", type=float, default=0.70)
    p.add_argument("--bath-fit-max-nfev", type=int, default=500)
    p.add_argument("--bath-fit-xtol", type=float, default=1e-10)

    p.add_argument("--bath-metric", choices=("delta", "g0"), default="delta")
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-svd-rcond", type=float, default=1e-7)
    p.add_argument("--bath-rank", type=int, default=24)
    p.add_argument("--bath-fd-step", type=float, default=2e-4)
    p.add_argument("--bath-fd-scheme", choices=("centered", "forward"), default="centered")
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)

    p.add_argument("--solver", choices=("gcrotmk", "gmres"), default="gcrotmk")
    p.add_argument("--jf-tol", type=float, default=1e-8)
    p.add_argument("--jf-maxiter", type=int, default=80)
    p.add_argument("--krylov-m", type=int, default=24)
    p.add_argument("--recycle-k", type=int, default=12)
    p.add_argument("--stage", choices=("split-mt", "full"), default="full")
    p.add_argument("--allow-unconverged-source", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out", type=Path, default=Path("cluster_ed_gw_jf_validation.npz"))
    return p.parse_args()


def _copy_state(s: ClusterCovariantState) -> ClusterCovariantState:
    return ClusterCovariantState(
        Sigma_H=np.asarray(s.Sigma_H, dtype=complex).copy(),
        Sigma_emb=np.asarray(s.Sigma_emb, dtype=complex).copy(),
        Sigma_imp=np.asarray(s.Sigma_imp, dtype=complex).copy(),
        G=np.asarray(s.G, dtype=complex).copy(),
        mu=float(s.mu),
        bath=BathParameters(
            np.asarray(s.bath.energies, dtype=float).copy(),
            np.asarray(s.bath.couplings, dtype=complex).copy(),
            float(s.bath.fit_error),
            int(s.bath.nfev),
        ),
    )


def _extrapolate_h2(h, mats):
    h = np.asarray(h, dtype=float)
    a = np.asarray(mats, dtype=float)
    if len(h) == 1:
        return np.asarray(a[0])
    x = h * h
    out = np.empty(a.shape[1:], dtype=float)
    for idx in np.ndindex(out.shape):
        out[idx] = np.polyfit(x, a[(slice(None),) + idx], 1)[1]
    return out


def _relerr(a, b):
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    den = max(float(np.linalg.norm(bb.ravel())), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def main():
    args = _args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    hvalues = np.asarray(sorted(set(abs(float(x)) for x in args.h if abs(float(x)) > 0)))
    if hvalues.size == 0:
        raise ValueError("--h needs at least one nonzero magnitude")

    with np.load(args.input, allow_pickle=False) as z:
        if "converged" in z and not bool(z["converged"]):
            raise RuntimeError("input embedding is not converged")
        Lx, Ly = int(z["Lx"]), int(z["Ly"])
        V = float(z["V"])
        filling, T = float(z["filling"]), float(z["T"])
        params = RubyParameters(
            ti=float(z["ti"]), t1=float(z["t1"]), t2=float(z["t2"]), V=V
        )
        omega = np.asarray(z["omega"], dtype=float)
        Omega = np.asarray(z["Omega"], dtype=float)
        base = ClusterCovariantState(
            Sigma_H=np.asarray(z["Sigma_H"], dtype=complex),
            Sigma_emb=np.asarray(z["Sigma_emb"], dtype=complex),
            Sigma_imp=np.asarray(z["Sigma_ED_cluster"], dtype=complex),
            G=np.asarray(z["G"], dtype=complex),
            mu=float(z["mu"]),
            bath=BathParameters(
                np.asarray(z["bath_energies"], dtype=float),
                np.asarray(z["bath_couplings"], dtype=complex),
                float(z["bath_fit_error"]),
                0,
            ),
        )

    grid = MatsubaraGrid(
        nk1=Lx, nk2=Ly, nw=len(omega) // 2, nOmega=(len(Omega) - 1) // 2, T=T
    )
    if not np.allclose(grid.omega, omega, rtol=0, atol=1e-12):
        raise ValueError("saved fermionic Matsubara grid mismatch")
    if not np.allclose(grid.Omega, Omega, rtol=0, atol=1e-12):
        raise ValueError("saved bosonic Matsubara grid mismatch")
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_interaction(grid.qmesh(), params)

    channels = []
    for raw in args.channels:
        ch = canonical_channel_name(raw)
        if ch not in channels:
            channels.append(ch)
    K = np.stack([primitive_pseudospin_vertex(ch) for ch in channels])
    nc = len(channels)
    nbath = len(base.bath.energies)

    src_opts = ClusterCovariantOptions(
        max_iter=int(args.response_max),
        tol=float(args.response_tol),
        mixing=float(args.response_mixing),
        nbath=nbath,
        bath_fit_nfreq=int(args.bath_fit_nfreq),
        bath_fit_max_nfev=int(args.bath_fit_max_nfev),
        bath_fit_xtol=float(args.bath_fit_xtol),
        discard_weight_tol=float(args.discard_weight_tol),
        verbose=not args.quiet,
    )

    print("=== refine the common h=0 complex-bath fixed point ===", flush=True)
    zero = solve_cluster_source_warm(
        h0, Vq, params, grid, K[0], filling, _copy_state(base), src_opts
    )
    if not zero.converged and not args.allow_unconverged_source:
        raise RuntimeError(f"zero-source refinement failed: {zero.final_error:.3e}")
    state = zero.state
    rho_k = one_body_density_matrix_tail(
        state.G, grid, h0, state.mu, state.Sigma_H
    )
    rho_c = np.mean(rho_k, axis=(0, 1))
    h_cluster = np.mean(h0, axis=(0, 1))
    h_cluster = 0.5 * (h_cluster + h_cluster.conj().T)

    max_rank = None if int(args.bath_rank) <= 0 else int(args.bath_rank)
    jf, tangent = build_embedded_jacobian(
        state.G,
        Vq,
        state.bath,
        h_cluster,
        params,
        grid,
        state.mu,
        rho_c,
        bath_opts=BathTangentOptions(
            fit_metric=args.bath_metric,
            nfit=int(args.bath_fit_nfreq),
            svd_rcond=float(args.bath_svd_rcond),
            max_rank=max_rank,
            fd_step=float(args.bath_fd_step),
            fd_scheme=args.bath_fd_scheme,
            discard_weight_tol=float(args.discard_weight_tol),
            verbose=not args.quiet,
        ),
        jf_opts=ClusterJFOptions(
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
        ),
    )
    chi_jf_raw, jf_results = response_matrix(jf, K, (0, 0), recycle=True)
    chi_jf = 0.5 * (chi_jf_raw + chi_jf_raw.conj().T)

    chi_h = np.empty((len(hvalues), nc, nc), dtype=float)
    source_residual = np.empty((len(hvalues), nc, 2), dtype=float)
    source_iterations = np.empty((len(hvalues), nc, 2), dtype=int)
    print("\n=== central finite-source derivatives ===", flush=True)
    for b, ch in enumerate(channels):
        plus_seed = _copy_state(state)
        minus_seed = _copy_state(state)
        for ih, h in enumerate(hvalues):
            hp = h0 - h * K[b][None, None]
            hm = h0 + h * K[b][None, None]
            rp = solve_cluster_source_warm(
                hp, Vq, params, grid, K[b], filling, plus_seed, src_opts
            )
            rm = solve_cluster_source_warm(
                hm, Vq, params, grid, K[b], filling, minus_seed, src_opts
            )
            plus_seed, minus_seed = _copy_state(rp.state), _copy_state(rm.state)
            source_residual[ih, b] = (rp.final_error, rm.final_error)
            source_iterations[ih, b] = (rp.iterations, rm.iterations)
            if (not rp.converged or not rm.converged) and not args.allow_unconverged_source:
                raise RuntimeError(
                    f"finite-source solve failed for {ch}, h={h:g}: "
                    f"+ {rp.final_error:.3e}, - {rm.final_error:.3e}"
                )
            for a in range(nc):
                op = onebody_expectation_from_lattice_G(
                    rp.state.G, K[a], hp, rp.state.mu, rp.state.Sigma_H, grid
                )
                om = onebody_expectation_from_lattice_G(
                    rm.state.G, K[a], hm, rm.state.mu, rm.state.Sigma_H, grid
                )
                chi_h[ih, a, b] = (op - om) / (2.0 * h)
            print(
                f"{ch}, h={h:.2e}: max residual="
                f"{max(rp.final_error, rm.final_error):.2e}",
                flush=True,
            )

    chi_fs_raw = _extrapolate_h2(hvalues, chi_h)
    chi_fs = 0.5 * (chi_fs_raw + chi_fs_raw.T)
    err = _relerr(chi_jf, chi_fs)
    print("\nchannels:", channels, flush=True)
    print("JF chi:\n", np.array2string(chi_jf.real, precision=8), flush=True)
    print("finite-source chi(h->0):\n", np.array2string(chi_fs, precision=8), flush=True)
    print(f"matrix relative difference = {err:.6e}", flush=True)

    out = args.out.with_suffix(".npz") if args.out.suffix.lower() != ".npz" else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        source_file=str(args.input),
        channels=np.asarray(channels),
        h_values=hvalues,
        chi_jf_raw=chi_jf_raw,
        chi_jf=chi_jf,
        chi_finite_source_h=chi_h,
        chi_finite_source_raw_extrapolated=chi_fs_raw,
        chi_finite_source=chi_fs,
        matrix_relerr=float(err),
        jf_iterations=np.asarray([r.iterations for r in jf_results], dtype=int),
        jf_residuals=np.asarray([r.final_error for r in jf_results], dtype=float),
        source_iterations=source_iterations,
        source_residuals=source_residual,
        zero_source_residual=float(zero.final_error),
        zero_source_bath_fit=float(zero.bath_fit_error),
        bath_tangent_rank=int(tangent.rank),
        bath_tangent_singular_values=tangent.singular_values,
        bath_tangent_condition=float(tangent.condition_number),
        bath_fd_step=float(tangent.fd_step),
    )
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
