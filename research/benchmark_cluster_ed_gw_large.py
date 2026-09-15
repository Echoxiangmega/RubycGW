#!/usr/bin/env python3
"""Benchmark a saved cluster-ED+GW result against finite-torus ED.

For <=16 sites this uses the full-spectrum ExactSmallRubyThermal solver.  For
18 or 24 sites it uses the low-temperature thermal-Lanczos Green-function
solver, which retains low-energy *initial* states by grand-canonical weight and
evaluates the N+/-1 final-state resolvents with Lanczos.  The expensive
cluster-ED+GW embedding is never rerun by this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.thermal_lanczos_green import (
    ThermalLanczosOptions,
    thermal_lanczos_green,
)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="saved cluster_ed_gw_*.npz result")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--sector-padding", type=int, default=None)
    p.add_argument("--thermal-eigs", type=int, default=None)
    p.add_argument("--thermal-discard-weight-tol", type=float, default=None)
    p.add_argument("--krylov-dim", type=int, default=None)
    p.add_argument("--krylov-tol", type=float, default=1e-12)
    p.add_argument("--eig-tol", type=float, default=1e-9)
    p.add_argument("--eig-maxiter", type=int, default=6000)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _relerr(a, b):
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    den = max(float(np.linalg.norm(bb.ravel())), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def _lattice_to_realspace(Gk: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    arr = np.asarray(Gk, dtype=complex)
    nf = int(arr.shape[0])
    ncell = int(Lx) * int(Ly)
    out = np.zeros((nf, 6 * ncell, 6 * ncell), dtype=complex)
    cells = [(r1, r2) for r1 in range(int(Lx)) for r2 in range(int(Ly))]
    for c, (r1, r2) in enumerate(cells):
        for d, (s1, s2) in enumerate(cells):
            block = np.zeros((nf, 6, 6), dtype=complex)
            for i in range(int(Lx)):
                for j in range(int(Ly)):
                    phase = np.exp(
                        2j * np.pi * (
                            (i / float(Lx)) * (r1 - s1)
                            + (j / float(Ly)) * (r2 - s2)
                        )
                    )
                    block += phase * arr[:, i, j]
            block /= float(ncell)
            out[:, 6*c:6*(c+1), 6*d:6*(d+1)] = block
    return out


def _adaptive_options(args, nsites: int) -> ThermalLanczosOptions:
    if nsites <= 18:
        defaults = dict(
            sector_padding=2,
            thermal_eigs=12,
            thermal_discard_weight_tol=1e-7,
            krylov_dim=160,
        )
    else:
        # 2x2 = 24 sites is much more expensive: the N=8 sector has 735471
        # states and neighboring resolvent sectors exceed one million states.
        # Start with a conservative benchmark that exposes truncation diagnostics;
        # users can tighten each control after seeing sector-edge weights.
        defaults = dict(
            sector_padding=1,
            thermal_eigs=6,
            thermal_discard_weight_tol=1e-5,
            krylov_dim=80,
        )
    return ThermalLanczosOptions(
        sector_padding=(defaults["sector_padding"] if args.sector_padding is None else int(args.sector_padding)),
        thermal_eigs=(defaults["thermal_eigs"] if args.thermal_eigs is None else int(args.thermal_eigs)),
        thermal_discard_weight_tol=(
            defaults["thermal_discard_weight_tol"]
            if args.thermal_discard_weight_tol is None
            else float(args.thermal_discard_weight_tol)
        ),
        krylov_dim=(defaults["krylov_dim"] if args.krylov_dim is None else int(args.krylov_dim)),
        krylov_tol=float(args.krylov_tol),
        eig_tol=float(args.eig_tol),
        eig_maxiter=int(args.eig_maxiter),
        verbose=not bool(args.quiet),
    )


def main():
    args = _args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    outfile = args.out
    if outfile is None:
        outfile = args.input.with_name(args.input.stem + "_ed_benchmark.npz")

    if outfile.exists() and not args.refresh:
        with np.load(outfile, allow_pickle=False) as z:
            print(
                f"benchmark cache hit: {outfile}\n"
                f"method={str(z['benchmark_method'])}, "
                f"Gerr_GW={float(z['Gerr_GW']):.6e}, "
                f"Gerr_clusterED+GW={float(z['Gerr_clusterED_GW']):.6e}",
                flush=True,
            )
        return

    with np.load(args.input, allow_pickle=False) as z:
        Lx = int(z["Lx"])
        Ly = int(z["Ly"])
        V = float(z["V"])
        filling = float(z["filling"])
        T = float(z["T"])
        ti = float(z["ti"])
        t1 = float(z["t1"])
        t2 = float(z["t2"])
        omega = np.asarray(z["omega"], dtype=float)
        G_emb_k = np.asarray(z["G"], dtype=complex)
        G_gw_k = np.asarray(z["G_background"], dtype=complex)
        converged = bool(z["converged"])
        residual = float(z["final_error"])

    if not converged:
        print(
            f"warning: embedding input is not converged (residual={residual:.3e}); "
            "benchmark will still be computed",
            flush=True,
        )

    nsites = 6 * Lx * Ly
    if nsites > 24:
        raise ValueError("this benchmark currently supports at most 24 physical sites")
    params = RubyParameters(ti=ti, t1=t1, t2=t2, V=V)
    target_particles = filling * Lx * Ly

    tl = None
    if nsites <= 16:
        method = "full_spectrum_ed"
        print(f"[benchmark] full-spectrum ED on {nsites} sites ...", flush=True)
        exact = ExactSmallRubyThermal(Lx, Ly, params)
        exact.diagonalize(V)
        mu_ed = exact.solve_mu(target_particles, T)
        G_ed, _ = exact.green_iomega(1j * omega, mu_ed, T)
        diag = dict(
            kept_thermal_weight=1.0,
            sector_numbers=np.empty((0,), dtype=int),
            sector_probabilities=np.empty((0,), dtype=float),
            sector_ground_energies=np.empty((0,), dtype=float),
            sector_last_gap=np.empty((0,), dtype=float),
            retained_states_per_sector=np.empty((0,), dtype=int),
            krylov_steps_mean=0.0,
            krylov_steps_max=0,
            translation_residual=0.0,
        )
    else:
        method = "thermal_lanczos"
        opts = _adaptive_options(args, nsites)
        print(
            f"[benchmark] thermal Lanczos on {nsites} sites: "
            f"padding={opts.sector_padding}, thermal_eigs={opts.thermal_eigs}, "
            f"discard={opts.thermal_discard_weight_tol:.1e}, krylov={opts.krylov_dim}",
            flush=True,
        )
        tl = thermal_lanczos_green(
            Lx,
            Ly,
            params,
            V=V,
            T=T,
            target_particles=target_particles,
            omega=omega,
            opts=opts,
        )
        mu_ed = float(tl.mu)
        G_ed = np.asarray(tl.G)
        diag = dict(
            kept_thermal_weight=float(tl.kept_thermal_weight),
            sector_numbers=np.asarray(tl.sector_numbers),
            sector_probabilities=np.asarray(tl.sector_probabilities),
            sector_ground_energies=np.asarray(tl.sector_ground_energies),
            sector_last_gap=np.asarray(tl.sector_last_gap),
            retained_states_per_sector=np.asarray(tl.retained_states_per_sector),
            krylov_steps_mean=float(tl.krylov_steps_mean),
            krylov_steps_max=int(tl.krylov_steps_max),
            translation_residual=float(tl.translation_residual),
        )

    G_gw = _lattice_to_realspace(G_gw_k, Lx, Ly)
    G_emb = _lattice_to_realspace(G_emb_k, Lx, Ly)
    gerr_gw = _relerr(G_gw, G_ed)
    gerr_emb = _relerr(G_emb, G_ed)
    ratio = gerr_emb / gerr_gw if gerr_gw != 0.0 else np.nan

    edge_weight = (
        float(diag["sector_probabilities"][0] + diag["sector_probabilities"][-1])
        if len(diag["sector_probabilities"]) else 0.0
    )
    print(
        f"[benchmark] method={method}, mu={mu_ed:+.10f}, "
        f"Gerr_GW={gerr_gw:.6e}, Gerr_clusterED+GW={gerr_emb:.6e}, ratio={ratio:.6f}",
        flush=True,
    )
    if method == "thermal_lanczos":
        print(
            f"[benchmark] kept_weight={diag['kept_thermal_weight']:.9f}, "
            f"sector_edge_weight={edge_weight:.3e}, "
            f"krylov_steps(mean,max)=({diag['krylov_steps_mean']:.1f},{diag['krylov_steps_max']}), "
            f"translation_residual={diag['translation_residual']:.3e}",
            flush=True,
        )
        if edge_weight > 1e-3:
            print(
                "[benchmark] WARNING: thermal sector edge weight is not small; "
                "increase --sector-padding before treating this as a converged benchmark.",
                flush=True,
            )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outfile,
        source_file=str(args.input),
        benchmark_method=method,
        Lx=Lx,
        Ly=Ly,
        nsites=nsites,
        V=V,
        filling=filling,
        T=T,
        omega=omega,
        mu_ed=float(mu_ed),
        G_ed=np.asarray(G_ed),
        Gerr_GW=float(gerr_gw),
        Gerr_clusterED_GW=float(gerr_emb),
        Gerr_ratio=float(ratio),
        sector_edge_weight=float(edge_weight),
        **diag,
    )
    print(f"saved {outfile}", flush=True)


if __name__ == "__main__":
    main()
