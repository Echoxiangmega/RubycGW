#!/usr/bin/env python3
"""Covariant current susceptibility for selectable weak+ED embedding NPZ files."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from benchmark_cluster_ed_gw_chi import (
    CHANNELS,
    _copy_state,
    _empty_ed_result,
    _exact_full_ed_response,
    _extrapolate_h2,
    _full_ed_supported,
    _matrix_relerr,
    _print_matrix,
    _thermal_defaults,
    _thermal_ed_supported,
    _thermal_rectangular_ed_response,
)
from rubycgw.cluster_ed_gw import BathParameters
from rubycgw.cluster_ed_gw_covariant import (
    ClusterCovariantOptions,
    ClusterCovariantState,
    onebody_expectation_from_lattice_G,
)
from rubycgw.cluster_ed_weak_covariant import solve_cluster_source_warm_weak
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import primitive_pseudospin_vertex
from rubycgw.sox_covariant import SOXOptions


_ED_MODES = ("auto", "full", "thermal", "none")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--h", nargs="+", type=float, default=[2e-3, 1e-3])
    p.add_argument("--response-max", type=int, default=60)
    p.add_argument("--response-tol", type=float, default=2e-6)
    p.add_argument("--response-mixing", type=float, default=0.70)
    p.add_argument("--bath-fit-max-nfev", type=int, default=500)
    p.add_argument("--bath-fit-xtol", type=float, default=1e-10)
    p.add_argument("--gf2-nquad", type=int, default=None)
    p.add_argument("--ed-mode", choices=_ED_MODES, default="auto")
    p.add_argument("--ed-thermal-samples", type=int, default=None)
    p.add_argument("--ed-tau-points", type=int, default=None)
    p.add_argument("--ed-seed", type=int, default=12345)
    p.add_argument("--ed-eig-tol", type=float, default=1e-10)
    p.add_argument("--ed-maxiter", type=int, default=5000)
    p.add_argument("--tr-reduce", action="store_true")
    p.add_argument("--allow-unconverged", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _scalar_string(z, key: str, default: str) -> str:
    if key not in z.files:
        return str(default)
    return str(np.asarray(z[key]).reshape(()).item())


def _choose_ed_method(args, Lx, Ly, filling, mesh_match):
    if args.ed_mode == "none":
        return "none"
    if not mesh_match:
        if args.ed_mode != "auto":
            raise ValueError(
                "forced ED comparison requires nk1=Lx and nk2=Ly; otherwise the finite torus differs"
            )
        return "none"
    full = _full_ed_supported(Lx, Ly)
    thermal = _thermal_ed_supported(Lx, Ly, filling)
    if args.ed_mode == "full":
        if not full:
            raise ValueError("--ed-mode full requires <=16 sites")
        return "full"
    if args.ed_mode == "thermal":
        if not thermal:
            raise ValueError("--ed-mode thermal requires <=24 sites and integer N")
        return "thermal"
    if full:
        return "full"
    if thermal:
        return "thermal"
    return "none"


def main():
    args = _args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    hvalues = np.asarray(sorted(set(abs(x) for x in args.h if abs(x) > 0.0)))
    if len(hvalues) == 0:
        raise ValueError("--h needs at least one nonzero magnitude")

    with np.load(args.input, allow_pickle=False) as z:
        Lx = int(z["Lx"])
        Ly = int(z["Ly"])
        nk1 = int(z["nk1"]) if "nk1" in z.files else Lx
        nk2 = int(z["nk2"]) if "nk2" in z.files else Ly
        weak_solver = _scalar_string(z, "weak_solver", "gw").lower()
        saved_gf2_nquad = int(z["gf2_nquad"]) if "gf2_nquad" in z.files else 64
        if not bool(z["converged"]):
            raise ValueError("input embedding result is not converged")
        V = float(z["V"])
        filling = float(z["filling"])
        T = float(z["T"])
        ti, t1, t2 = float(z["ti"]), float(z["t1"]), float(z["t2"])
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

    gf2_nquad = saved_gf2_nquad if args.gf2_nquad is None else int(args.gf2_nquad)
    nbath = len(bath_e)
    nsites = 6 * Lx * Ly
    mesh_match = (nk1, nk2) == (Lx, Ly)
    print(
        f"=== cluster ED + {weak_solver.upper()} chi: benchmark L={Lx}x{Ly}, "
        f"kmesh={nk1}x{nk2}, V={V:g}, filling={filling:g}, T={T:g}, nbath={nbath} ==="
    )

    grid = MatsubaraGrid(
        nk1=nk1,
        nk2=nk2,
        nw=len(omega) // 2,
        nOmega=(len(Omega) - 1) // 2,
        T=T,
    )
    if np.max(np.abs(grid.omega - omega)) > 1e-12 or np.max(np.abs(grid.Omega - Omega)) > 1e-12:
        raise ValueError("saved Matsubara arrays do not match reconstructed grid")
    if G0.shape[1:3] != (nk1, nk2):
        raise ValueError("saved G shape does not match saved/resolved k mesh")

    params = RubyParameters(ti=ti, t1=t1, t2=t2, V=V)
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_interaction(grid.qmesh(), params)
    K = np.stack([primitive_pseudospin_vertex(ch) for ch in CHANNELS])
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
        nbath=int(nbath),
        bath_fit_max_nfev=int(args.bath_fit_max_nfev),
        bath_fit_xtol=float(args.bath_fit_xtol),
        verbose=not bool(args.quiet),
    )
    gf2_opts = SOXOptions(n_quad=gf2_nquad, tail_complete=True, tail_edge_points=2)

    def solve(source_h0, source_K, seed):
        return solve_cluster_source_warm_weak(
            source_h0,
            Vq,
            params,
            grid,
            source_K,
            filling,
            seed,
            opts,
            weak_solver=weak_solver,
            gf2_sox_opts=gf2_opts,
        )

    print("\n=== zero-source covariant-map refinement ===")
    zero = solve(h0, K[0], _copy_state(base))
    if not zero.converged and not args.allow_unconverged:
        raise RuntimeError(f"zero-source refinement failed: {zero.final_error:.3e}")
    zero_state = zero.state
    zero_obs = np.asarray([
        onebody_expectation_from_lattice_G(
            zero_state.G, K[a], h0, zero_state.mu, zero_state.Sigma_H, grid
        )
        for a in range(len(CHANNELS))
    ])
    print(
        f"zero: converged={zero.converged}, iter={zero.iterations}, residual={zero.final_error:.3e}, "
        f"bath={zero.bath_fit_error:.3e}, mu={zero_state.mu:+.10f}, J={zero_obs.tolist()}"
    )

    nh, nc = len(hvalues), len(CHANNELS)
    chi_h = np.full((nh, nc, nc), np.nan)
    plus_obs = np.full((nh, nc, nc), np.nan)
    minus_obs = np.full((nh, nc, nc), np.nan)
    source_converged = np.zeros((nh, nc, 2), dtype=bool)
    source_iterations = np.zeros((nh, nc, 2), dtype=int)
    source_residual = np.full((nh, nc, 2), np.nan)

    print("\n=== numerical covariant response ===")
    for b, ch in enumerate(CHANNELS):
        plus_seed = _copy_state(zero_state)
        minus_seed = _copy_state(zero_state)
        for ih, h in enumerate(hvalues):
            print(f"\n--- source {ch}, |h|={h:.3e} ---")
            hp = h0 - float(h) * K[b][None, None]
            rp = solve(hp, K[b], plus_seed)
            plus_seed = _copy_state(rp.state)
            source_converged[ih, b, 0] = rp.converged
            source_iterations[ih, b, 0] = rp.iterations
            source_residual[ih, b, 0] = rp.final_error
            if args.tr_reduce:
                rm = None
                source_converged[ih, b, 1] = rp.converged
                source_iterations[ih, b, 1] = 0
                source_residual[ih, b, 1] = rp.final_error
            else:
                hm = h0 + float(h) * K[b][None, None]
                rm = solve(hm, K[b], minus_seed)
                minus_seed = _copy_state(rm.state)
                source_converged[ih, b, 1] = rm.converged
                source_iterations[ih, b, 1] = rm.iterations
                source_residual[ih, b, 1] = rm.final_error
            if (not rp.converged or (rm is not None and not rm.converged)) and not args.allow_unconverged:
                raise RuntimeError(f"source solve failed for {ch}, h={h:g}")

            for a in range(nc):
                jp = onebody_expectation_from_lattice_G(
                    rp.state.G, K[a], hp, rp.state.mu, rp.state.Sigma_H, grid
                )
                if rm is None:
                    jm = -jp
                else:
                    hm = h0 + float(h) * K[b][None, None]
                    jm = onebody_expectation_from_lattice_G(
                        rm.state.G, K[a], hm, rm.state.mu, rm.state.Sigma_H, grid
                    )
                plus_obs[ih, a, b] = jp
                minus_obs[ih, a, b] = jm
                chi_h[ih, a, b] = (jp - jm) / (2.0 * h)
            print(f"chi column {ch}: {chi_h[ih,:,b].tolist()}")

    chi_raw = _extrapolate_h2(hvalues, chi_h)
    chi_cluster = 0.5 * (chi_raw + chi_raw.T)
    print("\n=== cluster result ===")
    _print_matrix(f"cluster-ED+{weak_solver.upper()} covariant chi", chi_cluster)
    for ih, h in enumerate(hvalues):
        sym = 0.5 * (chi_h[ih] + chi_h[ih].T)
        print(
            f"h={h:.3e}: same={sym[0,0]:+.9f}, opposite={sym[1,1]:+.9f}, cross={sym[0,1]:+.9f}"
        )

    ed = _empty_ed_result()
    ed_method = _choose_ed_method(args, Lx, Ly, filling, mesh_match)
    if ed_method == "full":
        print("\n=== exact full-spectrum ED response ===")
        ed = _exact_full_ed_response(Lx, Ly, params, V, filling, T)
    elif ed_method == "thermal":
        samples0, tau0 = _thermal_defaults(nsites)
        samples = samples0 if args.ed_thermal_samples is None else int(args.ed_thermal_samples)
        tau_points = tau0 if args.ed_tau_points is None else int(args.ed_tau_points)
        print("\n=== rectangular thermal-Krylov ED response ===")
        ed = _thermal_rectangular_ed_response(
            Lx,
            Ly,
            params,
            V,
            filling,
            T,
            samples=samples,
            tau_points=tau_points,
            seed=int(args.ed_seed),
            eig_tol=float(args.ed_eig_tol),
            maxiter=int(args.ed_maxiter),
            verbose=True,
        )
    elif not mesh_match:
        print("\n=== ED comparison skipped: nk mesh differs from finite-torus Lx,Ly ===")
    else:
        print("\n=== ED comparison unavailable/skipped ===")

    if ed_method != "none":
        chi_ed = np.asarray(ed["chi_reference"], dtype=float)
        relerr = _matrix_relerr(chi_cluster, chi_ed)
        diag_relerr = np.abs(np.diag(chi_cluster) - np.diag(chi_ed)) / np.maximum(
            np.abs(np.diag(chi_ed)), 1e-300
        )
        print("\n=== comparison ===")
        _print_matrix(f"ED chi [{ed['ensemble']}]", chi_ed)
        _print_matrix(f"cluster-ED+{weak_solver.upper()} covariant chi", chi_cluster)
        print(
            f"matrix relerr={relerr:.6e}; diag relerr: same={diag_relerr[0]:.6e}, "
            f"opposite={diag_relerr[1]:.6e}"
        )
    else:
        relerr = np.nan
        diag_relerr = np.full(nc, np.nan)

    outfile = args.out or args.input.with_name(args.input.stem + "_covariant_chi.npz")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outfile,
        source_file=str(args.input),
        Lx=Lx,
        Ly=Ly,
        nk1=nk1,
        nk2=nk2,
        weak_solver=weak_solver,
        gf2_nquad=gf2_nquad,
        channels=np.asarray(CHANNELS),
        h_values=hvalues,
        tr_reduced=bool(args.tr_reduce),
        chi_cluster_h=chi_h,
        chi_cluster_raw_extrapolated=chi_raw,
        chi_cluster_covariant=chi_cluster,
        ed_available=bool(ed_method != "none"),
        ed_method=str(ed["method"]),
        ed_ensemble=str(ed["ensemble"]),
        chi_ed_reference=np.asarray(ed["chi_reference"]),
        chi_ed_fixed_mu=np.asarray(ed["chi_fixed_mu"]),
        chi_ed_fixed_filling=np.asarray(ed["chi_fixed_filling"]),
        means_ed=np.asarray(ed["means"]),
        ed_n_particles=int(ed["n_particles"]),
        ed_hilbert_dimension=int(ed["hilbert_dimension"]),
        ed_thermal_samples=int(ed["thermal_samples"]),
        ed_tau_points=int(ed["tau_points"]),
        ed_seed=int(ed["seed"]),
        ed_tau_grid=np.asarray(ed["tau_grid"]),
        ed_correlation_tau=np.asarray(ed["correlation_tau"]),
        matrix_relerr=float(relerr),
        diag_relerr=diag_relerr,
        zero_expectation=zero_obs,
        zero_converged=bool(zero.converged),
        zero_iterations=int(zero.iterations),
        zero_residual=float(zero.final_error),
        source_plus_expectation=plus_obs,
        source_minus_expectation=minus_obs,
        source_converged=source_converged,
        source_iterations=source_iterations,
        source_residual=source_residual,
    )
    print(f"saved {outfile}")


if __name__ == "__main__":
    main()
