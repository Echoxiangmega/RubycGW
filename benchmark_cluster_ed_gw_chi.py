#!/usr/bin/env python3
"""Compare 2x1 cluster-ED+GW covariant current susceptibility with exact ED.

The only required input is a converged ``cluster_ed_gw_*.npz`` file.  The script
reconstructs the model and Matsubara grid from that file, refines the saved
zero-source embedding with the complex-bath response map, then evaluates the
full fixed-point derivative by symmetric source solves,

    chi_ab = d <K_a> / d h_b,
    H(h_b) = H(0) - h_b K_b,

at fixed filling.  This is a numerical covariant derivative of the complete
cluster-ED+GW approximation: lattice GW, finite bath, impurity ED,
Sigma_ED-Sigma_GW,C double-counting correction and chemical potential are all
allowed to respond.

For the 2x1 torus the exact comparison uses full-spectrum ED and the exact
Lehmann static susceptibility.  The number operator is included explicitly so
the ED result is converted from fixed-mu to fixed-filling response through the
Schur complement.  For the TR-odd current channels the correction should be
numerically negligible, but computing it removes any ambiguity.
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
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import primitive_pseudospin_vertex
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


CHANNELS = ("z_same", "z_opposite")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="converged cluster_ed_gw_L2x1_*.npz")
    p.add_argument("--h", nargs="+", type=float, default=[2e-3, 1e-3])
    p.add_argument("--response-max", type=int, default=60)
    p.add_argument("--response-tol", type=float, default=2e-6)
    p.add_argument("--response-mixing", type=float, default=0.70)
    p.add_argument("--bath-fit-max-nfev", type=int, default=500)
    p.add_argument("--bath-fit-xtol", type=float, default=1e-10)
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out", type=Path, default=None)
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


def _matrix_relerr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    den = max(float(np.linalg.norm(bb.ravel())), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def _extrapolate_h2(hvalues: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    h = np.asarray(hvalues, dtype=float)
    mats = np.asarray(matrices, dtype=float)
    if len(h) == 1:
        return np.asarray(mats[0])
    x = h * h
    out = np.empty(mats.shape[1:], dtype=float)
    for idx in np.ndindex(out.shape):
        y = mats[(slice(None),) + idx]
        out[idx] = np.polyfit(x, y, deg=1)[1]
    return out


def _print_matrix(name: str, mat: np.ndarray):
    a = np.asarray(mat, dtype=float)
    print(name)
    print(
        f"  [[{a[0,0]:+.9f}, {a[0,1]:+.9f}],\n"
        f"   [{a[1,0]:+.9f}, {a[1,1]:+.9f}]]",
        flush=True,
    )


def main():
    args = _args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    hvalues = np.asarray(sorted(set(abs(float(x)) for x in args.h if abs(float(x)) > 0.0), reverse=True))
    if hvalues.size == 0:
        raise ValueError("--h must contain at least one nonzero magnitude")

    with np.load(args.input, allow_pickle=False) as z:
        Lx = int(z["Lx"])
        Ly = int(z["Ly"])
        if (Lx, Ly) not in ((2, 1), (1, 2)):
            raise ValueError("this first covariant chi benchmark is intentionally limited to the 2x1 torus")
        if not bool(z["converged"]):
            raise ValueError("input cluster-ED+GW result is not converged")
        V = float(z["V"])
        filling = float(z["filling"])
        T = float(z["T"])
        ti = float(z["ti"])
        t1 = float(z["t1"])
        t2 = float(z["t2"])
        omega = np.asarray(z["omega"], dtype=float)
        Omega = np.asarray(z["Omega"], dtype=float)
        sigma_h0 = np.asarray(z["Sigma_H"], dtype=complex)
        sigma_emb0 = np.asarray(z["Sigma_emb"], dtype=complex)
        sigma_imp0 = np.asarray(z["Sigma_ED_cluster"], dtype=complex)
        G0 = np.asarray(z["G"], dtype=complex)
        mu0 = float(z["mu"])
        bath_e = np.asarray(z["bath_energies"], dtype=float)
        bath_v = np.asarray(z["bath_couplings"], dtype=complex)
        bath_err = float(z["bath_fit_error"])
        nbath = int(len(bath_e))

    nw = int(len(omega) // 2)
    nOmega = int((len(Omega) - 1) // 2)
    grid = MatsubaraGrid(nk1=Lx, nk2=Ly, nw=nw, nOmega=nOmega, T=T)
    if np.max(np.abs(grid.omega - omega)) > 1e-12 or np.max(np.abs(grid.Omega - Omega)) > 1e-12:
        raise ValueError("saved Matsubara arrays do not match reconstructed grid")

    params = RubyParameters(ti=ti, t1=t1, t2=t2, V=V)
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_interaction(grid.qmesh(), params)
    K = np.stack([primitive_pseudospin_vertex(ch) for ch in CHANNELS], axis=0)

    base = ClusterCovariantState(
        Sigma_H=sigma_h0,
        Sigma_emb=sigma_emb0,
        Sigma_imp=sigma_imp0,
        G=G0,
        mu=mu0,
        bath=BathParameters(bath_e, bath_v, bath_err, 0),
    )
    opts = ClusterCovariantOptions(
        max_iter=int(args.response_max),
        tol=float(args.response_tol),
        mixing=float(args.response_mixing),
        nbath=nbath,
        bath_fit_max_nfev=int(args.bath_fit_max_nfev),
        bath_fit_xtol=float(args.bath_fit_xtol),
        verbose=not bool(args.quiet),
    )

    # Refine once at h=0 with the same complex-bath functional used by the
    # source solves.  This prevents an O(h^0) mismatch between the saved
    # historical real-bath fixed point and the response functional.
    print("=== zero-source covariant-map refinement ===", flush=True)
    zero = solve_cluster_source_warm(
        h0,
        Vq,
        params,
        grid,
        K[0],
        filling,
        _copy_state(base),
        opts,
    )
    if not zero.converged and not args.allow_unconverged:
        raise RuntimeError(
            f"zero-source response-map refinement did not converge: {zero.final_error:.3e}"
        )
    zero_state = zero.state
    zero_obs = np.asarray([
        onebody_expectation_from_lattice_G(
            zero_state.G, K[a], h0, zero_state.mu, zero_state.Sigma_H, grid
        )
        for a in range(len(CHANNELS))
    ])
    print(
        f"zero: converged={zero.converged}, iter={zero.iterations}, "
        f"residual={zero.final_error:.3e}, bath={zero.bath_fit_error:.3e}, "
        f"mu={zero_state.mu:+.10f}, J={zero_obs.tolist()}",
        flush=True,
    )

    nh = len(hvalues)
    nc = len(CHANNELS)
    chi_h = np.full((nh, nc, nc), np.nan, dtype=float)
    plus_obs = np.full((nh, nc, nc), np.nan, dtype=float)
    minus_obs = np.full_like(plus_obs, np.nan)
    source_converged = np.zeros((nh, nc, 2), dtype=bool)
    source_iterations = np.zeros((nh, nc, 2), dtype=int)
    source_residual = np.full((nh, nc, 2), np.nan)
    source_bath_fit = np.full((nh, nc, 2), np.nan)
    source_mu = np.full((nh, nc, 2), np.nan)

    print("\n=== cluster-ED+GW numerical covariant response ===", flush=True)
    for ih, h in enumerate(hvalues):
        for b, ch in enumerate(CHANNELS):
            print(f"\n--- source {ch}, |h|={h:.3e} ---", flush=True)
            hp = h0 - float(h) * K[b][None, None, :, :]
            hm = h0 + float(h) * K[b][None, None, :, :]
            rp = solve_cluster_source_warm(
                hp, Vq, params, grid, K[b], filling, _copy_state(zero_state), opts
            )
            rm = solve_cluster_source_warm(
                hm, Vq, params, grid, K[b], filling, _copy_state(zero_state), opts
            )
            for isign, r in enumerate((rp, rm)):
                source_converged[ih, b, isign] = r.converged
                source_iterations[ih, b, isign] = r.iterations
                source_residual[ih, b, isign] = r.final_error
                source_bath_fit[ih, b, isign] = r.bath_fit_error
                source_mu[ih, b, isign] = r.state.mu
            if (not rp.converged or not rm.converged) and not args.allow_unconverged:
                raise RuntimeError(
                    f"source solve failed for {ch}, h={h:g}: "
                    f"+ residual={rp.final_error:.3e}, - residual={rm.final_error:.3e}"
                )
            for a in range(nc):
                jp = onebody_expectation_from_lattice_G(
                    rp.state.G, K[a], hp, rp.state.mu, rp.state.Sigma_H, grid
                )
                jm = onebody_expectation_from_lattice_G(
                    rm.state.G, K[a], hm, rm.state.mu, rm.state.Sigma_H, grid
                )
                plus_obs[ih, a, b] = jp
                minus_obs[ih, a, b] = jm
                chi_h[ih, a, b] = (jp - jm) / (2.0 * float(h))
            print(
                f"chi column {ch}: {chi_h[ih,:,b].tolist()} "
                f"(+res={rp.final_error:.2e}, -res={rm.final_error:.2e})",
                flush=True,
            )

    # The exact equilibrium response is symmetric.  Numerical source solves can
    # carry tiny antisymmetric solver noise, so report both raw and symmetric
    # matrices; the latter is the physical comparison.
    chi_cluster_raw = _extrapolate_h2(hvalues, chi_h)
    chi_cluster = 0.5 * (chi_cluster_raw + chi_cluster_raw.T)

    print("\n=== exact full-spectrum ED response ===", flush=True)
    exact = ExactSmallRubyThermal(Lx, Ly, params)
    exact.diagonalize(V)
    target_particles = filling * Lx * Ly
    mu_ed = exact.solve_mu(target_particles, T)
    ops = [exact.pseudospin_operator(ch, (0.0, 0.0)) for ch in CHANNELS]
    # Total-number operator is needed for the fixed-filling Schur complement.
    ops.append(np.eye(exact.n_sites, dtype=complex))
    chi_aug, means_aug = exact.static_susceptibility_matrix(np.stack(ops), mu_ed, T)
    chi_ed_mu = np.asarray(chi_aug[:nc, :nc], dtype=float)
    chi_kn = np.asarray(chi_aug[:nc, nc], dtype=float)
    chi_nn = float(chi_aug[nc, nc])
    if chi_nn > 1e-14:
        chi_ed = chi_ed_mu - np.outer(chi_kn, chi_kn) / chi_nn
    else:
        chi_ed = chi_ed_mu.copy()

    relerr = _matrix_relerr(chi_cluster, chi_ed)
    diag_relerr = np.abs(np.diag(chi_cluster) - np.diag(chi_ed)) / np.maximum(
        np.abs(np.diag(chi_ed)), 1e-300
    )

    print("\n=== comparison ===", flush=True)
    _print_matrix("ED fixed-filling chi", chi_ed)
    _print_matrix("cluster-ED+GW covariant chi", chi_cluster)
    print(
        f"matrix relerr={relerr:.6e}; "
        f"diag relerr: same={diag_relerr[0]:.6e}, opposite={diag_relerr[1]:.6e}",
        flush=True,
    )
    print(
        f"ED fixed-mu -> fixed-filling correction: chi_KN={chi_kn.tolist()}, "
        f"chi_NN={chi_nn:.6e}",
        flush=True,
    )
    for ih, h in enumerate(hvalues):
        sym = 0.5 * (chi_h[ih] + chi_h[ih].T)
        print(
            f"h={h:.3e}: same={sym[0,0]:+.9f}, "
            f"opposite={sym[1,1]:+.9f}, cross={sym[0,1]:+.9f}",
            flush=True,
        )

    outfile = args.out
    if outfile is None:
        outfile = args.input.with_name(args.input.stem + "_covariant_chi.npz")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outfile,
        source_file=str(args.input),
        channels=np.asarray(CHANNELS),
        h_values=hvalues,
        chi_cluster_h=chi_h,
        chi_cluster_raw_extrapolated=chi_cluster_raw,
        chi_cluster_covariant=chi_cluster,
        chi_ed_fixed_mu=chi_ed_mu,
        chi_ed_fixed_filling=chi_ed,
        chi_ed_KN=chi_kn,
        chi_ed_NN=chi_nn,
        means_ed=np.asarray(means_aug[:nc]),
        mu_ed=float(mu_ed),
        matrix_relerr=float(relerr),
        diag_relerr=diag_relerr,
        zero_expectation=zero_obs,
        zero_converged=bool(zero.converged),
        zero_iterations=int(zero.iterations),
        zero_residual=float(zero.final_error),
        zero_bath_fit=float(zero.bath_fit_error),
        source_plus_expectation=plus_obs,
        source_minus_expectation=minus_obs,
        source_converged=source_converged,
        source_iterations=source_iterations,
        source_residual=source_residual,
        source_bath_fit=source_bath_fit,
        source_mu=source_mu,
    )
    print(f"saved {outfile}", flush=True)


if __name__ == "__main__":
    main()
