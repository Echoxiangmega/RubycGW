#!/usr/bin/env python3
"""Covariant current susceptibility from a saved cluster-ED+GW NPZ.

For any primitive rectangular ``Lx x Ly`` mesh this driver reconstructs the
saved cluster-ED+GW fixed point, applies small static sources in the two q=0
loop-current channels, and evaluates the full numerical fixed-point derivative

    chi_ab = d <K_a> / d h_b,
    H(h_b) = H(0) - h_b K_b,

at fixed average filling.  Lattice GW, finite bath, impurity ED,
Sigma_ED-Sigma_GW,C double counting and chemical potential are all allowed to
relax.

ED comparison is selected automatically:

* <=16 physical sites: full-spectrum grand-canonical ED, converted to
  fixed-filling response;
* 18 or 24 physical sites: canonical fixed-N thermal typicality + Krylov
  imaginary-time response on the *same rectangular torus*;
* >24 physical sites: cluster-ED+GW response only.

The 18/24-site ED response is stochastic/numerical rather than full-spectrum.
Its trace-vector count and tau grid are saved so convergence can be checked by
rerunning with tighter ``--ed-thermal-samples`` / ``--ed-tau-points``.
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
from rubycgw.ed_cluster import RubyEDClusterSolver
from rubycgw.ed_cluster_thermal import thermal_response as rectangular_thermal_response
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.pseudospin import primitive_pseudospin_vertex
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


CHANNELS = ("z_same", "z_opposite")
_ED_MODES = ("auto", "full", "thermal", "none")


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="converged cluster_ed_gw_*.npz")
    p.add_argument("--h", nargs="+", type=float, default=[2e-3, 1e-3])
    p.add_argument("--response-max", type=int, default=60)
    p.add_argument("--response-tol", type=float, default=2e-6)
    p.add_argument("--response-mixing", type=float, default=0.70)
    p.add_argument("--bath-fit-max-nfev", type=int, default=500)
    p.add_argument("--bath-fit-xtol", type=float, default=1e-10)
    p.add_argument(
        "--ed-mode",
        choices=_ED_MODES,
        default="auto",
        help=(
            "auto: full ED <=16 sites, rectangular thermal-Krylov ED <=24, "
            "otherwise cluster-only; full/thermal force that ED method; none skips ED"
        ),
    )
    p.add_argument("--ed-thermal-samples", type=int, default=None)
    p.add_argument("--ed-tau-points", type=int, default=None)
    p.add_argument("--ed-seed", type=int, default=12345)
    p.add_argument("--ed-eig-tol", type=float, default=1e-10)
    p.add_argument("--ed-maxiter", type=int, default=5000)
    p.add_argument(
        "--tr-reduce",
        action="store_true",
        help=(
            "solve only +h and infer -h by time reversal for the TR-odd current "
            "channels; default solves both signs explicitly"
        ),
    )
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
        out[idx] = np.polyfit(x, mats[(slice(None),) + idx], deg=1)[1]
    return out


def _print_matrix(name: str, mat: np.ndarray):
    a = np.asarray(mat, dtype=float)
    print(name)
    print(
        f"  [[{a[0,0]:+.9f}, {a[0,1]:+.9f}],\n"
        f"   [{a[1,0]:+.9f}, {a[1,1]:+.9f}]]",
        flush=True,
    )


def _full_ed_supported(Lx: int, Ly: int) -> bool:
    return 6 * int(Lx) * int(Ly) <= 16


def _thermal_ed_supported(Lx: int, Ly: int, filling: float) -> bool:
    nsites = 6 * int(Lx) * int(Ly)
    target = float(filling) * int(Lx) * int(Ly)
    return (
        2 <= nsites <= 24
        and abs(target - round(target)) <= 1e-12
        and 1 <= int(round(target)) <= nsites - 1
    )


def _thermal_defaults(nsites: int) -> tuple[int, int]:
    if int(nsites) <= 18:
        return 8, 17
    return 4, 9


def _empty_ed_result() -> dict:
    nc = len(CHANNELS)
    return dict(
        method="unavailable",
        ensemble="unavailable",
        chi_reference=np.full((nc, nc), np.nan, dtype=float),
        chi_fixed_mu=np.full((nc, nc), np.nan, dtype=float),
        chi_fixed_filling=np.full((nc, nc), np.nan, dtype=float),
        chi_kn=np.full(nc, np.nan, dtype=float),
        chi_nn=np.nan,
        means=np.full(nc, np.nan + 0.0j, dtype=complex),
        mu=np.nan,
        n_particles=-1,
        hilbert_dimension=-1,
        thermal_samples=0,
        tau_points=0,
        seed=-1,
        energy_shift=np.nan,
        shifted_partition_estimate=np.nan,
        tau_grid=np.empty(0, dtype=float),
        correlation_tau=np.empty((0, nc, nc), dtype=complex),
        structure_matrix=np.full((nc, nc), np.nan + 0j, dtype=complex),
    )


def _exact_full_ed_response(
    Lx: int,
    Ly: int,
    params: RubyParameters,
    V: float,
    filling: float,
    T: float,
) -> dict:
    exact = ExactSmallRubyThermal(int(Lx), int(Ly), params)
    exact.diagonalize(float(V))
    target_particles = float(filling) * int(Lx) * int(Ly)
    mu_ed = exact.solve_mu(target_particles, float(T))
    ops = [exact.pseudospin_operator(ch, (0.0, 0.0)) for ch in CHANNELS]
    ops.append(np.eye(exact.n_sites, dtype=complex))
    chi_aug, means_aug = exact.static_susceptibility_matrix(
        np.stack(ops), mu_ed, float(T)
    )
    nc = len(CHANNELS)
    chi_mu = np.asarray(chi_aug[:nc, :nc], dtype=float)
    chi_kn = np.asarray(chi_aug[:nc, nc], dtype=float)
    chi_nn = float(chi_aug[nc, nc])
    if chi_nn > 1e-14:
        chi_fixed = chi_mu - np.outer(chi_kn, chi_kn) / chi_nn
    else:
        chi_fixed = chi_mu.copy()
    out = _empty_ed_result()
    out.update(
        method="full_spectrum_ed",
        ensemble="grand_canonical_fixed_filling",
        chi_reference=chi_fixed,
        chi_fixed_mu=chi_mu,
        chi_fixed_filling=chi_fixed,
        chi_kn=chi_kn,
        chi_nn=chi_nn,
        means=np.asarray(means_aug[:nc]),
        mu=float(mu_ed),
    )
    return out


def _thermal_rectangular_ed_response(
    Lx: int,
    Ly: int,
    params: RubyParameters,
    V: float,
    filling: float,
    T: float,
    *,
    samples: int,
    tau_points: int,
    seed: int,
    eig_tol: float,
    maxiter: int,
    verbose: bool,
) -> dict:
    n_particles = int(round(float(filling) * int(Lx) * int(Ly)))
    solver = RubyEDClusterSolver(
        int(Lx),
        int(Ly),
        params,
        n_particles=n_particles,
    )
    mem = solver.memory_estimate()
    if verbose:
        print(
            f"[ED thermal] rectangular {Lx}x{Ly}: N={n_particles}, "
            f"dim={solver.dimension}, core-memory-est={mem['core_estimate_GiB']:.3f} GiB, "
            f"samples={samples}, tau_points={tau_points}, seed={seed}",
            flush=True,
        )
    resp = rectangular_thermal_response(
        solver,
        float(V),
        float(T),
        np.array([0.0, 0.0]),
        channels=CHANNELS,
        n_trace_vectors=int(samples),
        seed=int(seed),
        tau_points=int(tau_points),
        eig_tol=float(eig_tol),
        eig_maxiter=int(maxiter),
    )
    chi = np.asarray(resp.susceptibility_matrix.real, dtype=float)
    chi = 0.5 * (chi + chi.T)
    out = _empty_ed_result()
    out.update(
        method="thermal_typicality_krylov_rectangular",
        ensemble="canonical_fixed_N",
        chi_reference=chi,
        chi_fixed_filling=chi,
        means=np.asarray(resp.means),
        n_particles=int(n_particles),
        hilbert_dimension=int(solver.dimension),
        thermal_samples=int(resp.n_trace_vectors),
        tau_points=int(resp.tau_points),
        seed=int(seed),
        energy_shift=float(resp.energy_shift),
        shifted_partition_estimate=float(resp.shifted_partition_estimate),
        tau_grid=np.asarray(resp.tau_grid),
        correlation_tau=np.asarray(resp.correlation_tau_matrix),
        structure_matrix=np.asarray(resp.structure_matrix),
    )
    return out


def _choose_ed_method(args, Lx: int, Ly: int, filling: float) -> str:
    if args.ed_mode == "none":
        return "none"
    full = _full_ed_supported(Lx, Ly)
    thermal = _thermal_ed_supported(Lx, Ly, filling)
    if args.ed_mode == "full":
        if not full:
            raise ValueError("--ed-mode full requires <=16 physical sites")
        return "full"
    if args.ed_mode == "thermal":
        if not thermal:
            raise ValueError(
                "--ed-mode thermal requires <=24 sites and integer fixed particle number"
            )
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
    hvalues = np.asarray(
        sorted(set(abs(float(x)) for x in args.h if abs(float(x)) > 0.0))
    )
    if hvalues.size == 0:
        raise ValueError("--h must contain at least one nonzero magnitude")

    with np.load(args.input, allow_pickle=False) as z:
        Lx = int(z["Lx"])
        Ly = int(z["Ly"])
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

    nsites = 6 * Lx * Ly
    print(
        f"=== cluster-ED+GW chi: Lx={Lx}, Ly={Ly}, sites={nsites}, "
        f"V={V:g}, filling={filling:g}, T={T:g}, nbath={nbath} ===",
        flush=True,
    )

    nw = int(len(omega) // 2)
    nOmega = int((len(Omega) - 1) // 2)
    grid = MatsubaraGrid(nk1=Lx, nk2=Ly, nw=nw, nOmega=nOmega, T=T)
    if (
        np.max(np.abs(grid.omega - omega)) > 1e-12
        or np.max(np.abs(grid.Omega - Omega)) > 1e-12
    ):
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

    print("\n=== zero-source covariant-map refinement ===", flush=True)
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
    zero_obs = np.asarray(
        [
            onebody_expectation_from_lattice_G(
                zero_state.G,
                K[a],
                h0,
                zero_state.mu,
                zero_state.Sigma_H,
                grid,
            )
            for a in range(len(CHANNELS))
        ]
    )
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
    if args.tr_reduce:
        print(
            "using TR reduction: only +h solves are performed; -h is inferred",
            flush=True,
        )

    for b, ch in enumerate(CHANNELS):
        plus_seed = _copy_state(zero_state)
        minus_seed = _copy_state(zero_state)
        for ih, h in enumerate(hvalues):
            print(f"\n--- source {ch}, |h|={h:.3e} ---", flush=True)
            hp = h0 - float(h) * K[b][None, None, :, :]
            rp = solve_cluster_source_warm(
                hp, Vq, params, grid, K[b], filling, plus_seed, opts
            )
            plus_seed = _copy_state(rp.state)
            source_converged[ih, b, 0] = rp.converged
            source_iterations[ih, b, 0] = rp.iterations
            source_residual[ih, b, 0] = rp.final_error
            source_bath_fit[ih, b, 0] = rp.bath_fit_error
            source_mu[ih, b, 0] = rp.state.mu

            if args.tr_reduce:
                rm = None
                source_converged[ih, b, 1] = rp.converged
                source_iterations[ih, b, 1] = 0
                source_residual[ih, b, 1] = rp.final_error
                source_bath_fit[ih, b, 1] = rp.bath_fit_error
                source_mu[ih, b, 1] = rp.state.mu
            else:
                hm = h0 + float(h) * K[b][None, None, :, :]
                rm = solve_cluster_source_warm(
                    hm, Vq, params, grid, K[b], filling, minus_seed, opts
                )
                minus_seed = _copy_state(rm.state)
                source_converged[ih, b, 1] = rm.converged
                source_iterations[ih, b, 1] = rm.iterations
                source_residual[ih, b, 1] = rm.final_error
                source_bath_fit[ih, b, 1] = rm.bath_fit_error
                source_mu[ih, b, 1] = rm.state.mu

            if not rp.converged and not args.allow_unconverged:
                raise RuntimeError(
                    f"+h source solve failed for {ch}, h={h:g}: residual={rp.final_error:.3e}"
                )
            if rm is not None and (not rm.converged) and not args.allow_unconverged:
                raise RuntimeError(
                    f"-h source solve failed for {ch}, h={h:g}: residual={rm.final_error:.3e}"
                )

            for a in range(nc):
                jp = onebody_expectation_from_lattice_G(
                    rp.state.G, K[a], hp, rp.state.mu, rp.state.Sigma_H, grid
                )
                if rm is None:
                    jm = -jp
                else:
                    hm = h0 + float(h) * K[b][None, None, :, :]
                    jm = onebody_expectation_from_lattice_G(
                        rm.state.G, K[a], hm, rm.state.mu, rm.state.Sigma_H, grid
                    )
                plus_obs[ih, a, b] = jp
                minus_obs[ih, a, b] = jm
                chi_h[ih, a, b] = (jp - jm) / (2.0 * float(h))

            note = (
                f"+res={rp.final_error:.2e}, TR-inferred -h"
                if rm is None
                else f"+res={rp.final_error:.2e}, -res={rm.final_error:.2e}"
            )
            print(f"chi column {ch}: {chi_h[ih,:,b].tolist()} ({note})", flush=True)

    chi_cluster_raw = _extrapolate_h2(hvalues, chi_h)
    chi_cluster = 0.5 * (chi_cluster_raw + chi_cluster_raw.T)
    print("\n=== cluster-ED+GW result ===", flush=True)
    _print_matrix("cluster-ED+GW covariant chi", chi_cluster)
    for ih, h in enumerate(hvalues):
        sym = 0.5 * (chi_h[ih] + chi_h[ih].T)
        print(
            f"h={h:.3e}: same={sym[0,0]:+.9f}, "
            f"opposite={sym[1,1]:+.9f}, cross={sym[0,1]:+.9f}",
            flush=True,
        )

    ed = _empty_ed_result()
    selected = _choose_ed_method(args, Lx, Ly, filling)
    if selected == "full":
        print("\n=== exact full-spectrum ED response ===", flush=True)
        ed = _exact_full_ed_response(Lx, Ly, params, V, filling, T)
    elif selected == "thermal":
        default_samples, default_tau = _thermal_defaults(nsites)
        samples = default_samples if args.ed_thermal_samples is None else int(args.ed_thermal_samples)
        tau_points = default_tau if args.ed_tau_points is None else int(args.ed_tau_points)
        if samples < 1:
            raise ValueError("--ed-thermal-samples must be positive")
        if tau_points < 3 or tau_points % 2 != 1:
            raise ValueError("--ed-tau-points must be an odd integer >=3")
        print("\n=== rectangular thermal-Krylov ED response ===", flush=True)
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
            verbose=not bool(args.quiet),
        )
        print(
            "[ED thermal] NOTE: canonical fixed-N ensemble; compare sample/tau "
            "convergence before interpreting small finite-size differences.",
            flush=True,
        )

    if selected != "none":
        chi_ed = np.asarray(ed["chi_reference"], dtype=float)
        relerr = _matrix_relerr(chi_cluster, chi_ed)
        diag_relerr = np.abs(np.diag(chi_cluster) - np.diag(chi_ed)) / np.maximum(
            np.abs(np.diag(chi_ed)), 1e-300
        )
        print("\n=== comparison ===", flush=True)
        _print_matrix(f"ED chi [{ed['ensemble']}]", chi_ed)
        _print_matrix("cluster-ED+GW covariant chi", chi_cluster)
        print(
            f"matrix relerr={relerr:.6e}; diag relerr: "
            f"same={diag_relerr[0]:.6e}, opposite={diag_relerr[1]:.6e}",
            flush=True,
        )
        if ed["method"] == "full_spectrum_ed":
            print(
                f"ED fixed-mu -> fixed-filling correction: "
                f"chi_KN={np.asarray(ed['chi_kn']).tolist()}, "
                f"chi_NN={float(ed['chi_nn']):.6e}",
                flush=True,
            )
    else:
        relerr = np.nan
        diag_relerr = np.full(nc, np.nan, dtype=float)
        reason = (
            "disabled by --ed-mode none"
            if args.ed_mode == "none"
            else f"no validated rectangular ED chi path above 24 sites ({nsites} sites)"
        )
        print(f"\n=== ED comparison skipped: {reason} ===", flush=True)

    outfile = args.out
    if outfile is None:
        outfile = args.input.with_name(args.input.stem + "_covariant_chi.npz")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outfile,
        source_file=str(args.input),
        Lx=int(Lx),
        Ly=int(Ly),
        nsites=int(nsites),
        V=float(V),
        filling=float(filling),
        T=float(T),
        nbath=int(nbath),
        channels=np.asarray(CHANNELS),
        h_values=hvalues,
        tr_reduced=bool(args.tr_reduce),
        chi_cluster_h=chi_h,
        chi_cluster_raw_extrapolated=chi_cluster_raw,
        chi_cluster_covariant=chi_cluster,
        ed_available=bool(selected != "none"),
        ed_method=str(ed["method"]),
        ed_ensemble=str(ed["ensemble"]),
        chi_ed_reference=np.asarray(ed["chi_reference"]),
        chi_ed_fixed_mu=np.asarray(ed["chi_fixed_mu"]),
        chi_ed_fixed_filling=np.asarray(ed["chi_fixed_filling"]),
        chi_ed_KN=np.asarray(ed["chi_kn"]),
        chi_ed_NN=float(ed["chi_nn"]),
        means_ed=np.asarray(ed["means"]),
        mu_ed=float(ed["mu"]),
        ed_n_particles=int(ed["n_particles"]),
        ed_hilbert_dimension=int(ed["hilbert_dimension"]),
        ed_thermal_samples=int(ed["thermal_samples"]),
        ed_tau_points=int(ed["tau_points"]),
        ed_seed=int(ed["seed"]),
        ed_energy_shift=float(ed["energy_shift"]),
        ed_shifted_partition_estimate=float(ed["shifted_partition_estimate"]),
        ed_tau_grid=np.asarray(ed["tau_grid"]),
        ed_correlation_tau=np.asarray(ed["correlation_tau"]),
        ed_structure_matrix=np.asarray(ed["structure_matrix"]),
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
