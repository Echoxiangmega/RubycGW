#!/usr/bin/env python3
"""Run self-consistent 6-site cluster ED + GW on an Lx x Ly Ruby torus."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rubycgw.cluster_ed_gw import ClusterEDGWOptions, solve_cluster_ed_gw
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=1)
    p.add_argument("--V", type=float, default=1.0)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument("--filling", type=float, default=2.0, help="particles per primitive cell")
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--nw", type=int, default=55)
    p.add_argument("--nomega", type=int, default=12)
    p.add_argument("--gw-max", type=int, default=160)
    p.add_argument("--gw-tol", type=float, default=1e-8)
    p.add_argument("--gw-mixing", type=float, default=0.25)
    p.add_argument("--gw-mixing-method", choices=("linear", "pulay"), default="pulay")
    p.add_argument("--embed-max", type=int, default=12)
    p.add_argument("--embed-tol", type=float, default=2e-5)
    p.add_argument("--embed-mixing", type=float, default=0.30)
    p.add_argument("--impurity-mixing", type=float, default=0.70)
    p.add_argument("--nbath", type=int, default=6)
    p.add_argument("--bath-fit-nfreq", type=int, default=12)
    p.add_argument("--bath-fit-max-nfev", type=int, default=300)
    p.add_argument("--bath-energy-window", type=float, default=4.0)
    p.add_argument("--bath-coupling-bound", type=float, default=4.0)
    p.add_argument("--discard-weight-tol", type=float, default=1e-11)
    p.add_argument("--benchmark-ed", action="store_true")
    p.add_argument("--quiet-gw", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/cluster_ed_gw"))
    return p.parse_args()


def _lattice_to_realspace(Gk: np.ndarray, Lx: int, Ly: int) -> np.ndarray:
    """Fourier-transform 6x6 G(k) to the finite-torus site basis."""
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


def _relerr(a, b):
    aa = np.asarray(a, dtype=complex)
    bb = np.asarray(b, dtype=complex)
    den = max(float(np.linalg.norm(bb.ravel())), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / den)


def main():
    args = _args()
    if args.Lx < 1 or args.Ly < 1:
        raise ValueError("Lx and Ly must be positive")
    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=args.V)
    grid = MatsubaraGrid(
        nk1=int(args.Lx),
        nk2=int(args.Ly),
        nw=int(args.nw),
        nOmega=int(args.nomega),
        T=float(args.T),
    )
    h0 = build_h0(grid.kmesh(), params)
    Vq = build_interaction(grid.qmesh(), params)

    gw_opts = GWOptions(
        target_filling=float(args.filling),
        max_iter=int(args.gw_max),
        tol=float(args.gw_tol),
        mixing=float(args.gw_mixing),
        mixing_method=str(args.gw_mixing_method),
        verbose=not bool(args.quiet_gw),
        momentum_backend="fft",
    )
    embed_opts = ClusterEDGWOptions(
        max_iter=int(args.embed_max),
        tol=float(args.embed_tol),
        mixing=float(args.embed_mixing),
        impurity_mixing=float(args.impurity_mixing),
        nbath=int(args.nbath),
        bath_fit_nfreq=int(args.bath_fit_nfreq),
        bath_fit_max_nfev=int(args.bath_fit_max_nfev),
        bath_energy_window=float(args.bath_energy_window),
        bath_coupling_bound=float(args.bath_coupling_bound),
        discard_weight_tol=float(args.discard_weight_tol),
        verbose=True,
    )

    print("=== Ruby 6-site cluster ED + lattice GW ===", flush=True)
    print(
        f"L={args.Lx}x{args.Ly}, V={args.V:g}, filling={args.filling:g}, "
        f"T={args.T:g}, nw={args.nw}, nOmega={args.nomega}, nbath={args.nbath}",
        flush=True,
    )
    result = solve_cluster_ed_gw(
        h0,
        Vq,
        params,
        grid,
        gw_opts=gw_opts,
        embed_opts=embed_opts,
    )
    print(
        f"final: converged={result.converged}, iterations={result.iterations}, "
        f"residual={result.final_error:.3e}, impurity_mismatch={result.impurity_mismatch:.3e}, "
        f"bath_fit={result.bath_fit_error:.3e}, mu={result.mu:+.10f}",
        flush=True,
    )

    Gerr_bg = Gerr_emb = np.nan
    mu_ed = np.nan
    G_ed = np.empty((0,), dtype=complex)
    if args.benchmark_ed:
        nsites = 6 * int(args.Lx) * int(args.Ly)
        if nsites > 16:
            raise ValueError("--benchmark-ed is limited to <=16 sites by ExactSmallRubyThermal")
        print("[benchmark] exact finite-torus ED ...", flush=True)
        exact = ExactSmallRubyThermal(int(args.Lx), int(args.Ly), params)
        exact.diagonalize(float(args.V))
        mu_ed = exact.solve_mu(float(args.filling) * args.Lx * args.Ly, float(args.T))
        G_ed, _ = exact.green_iomega(1j * grid.omega, mu_ed, float(args.T))
        G_bg_real = _lattice_to_realspace(result.background.G, args.Lx, args.Ly)
        G_emb_real = _lattice_to_realspace(result.G, args.Lx, args.Ly)
        Gerr_bg = _relerr(G_bg_real, G_ed)
        Gerr_emb = _relerr(G_emb_real, G_ed)
        print(
            f"[benchmark] mu_ED={mu_ed:+.10f}, Gerr_GW={Gerr_bg:.6e}, "
            f"Gerr_clusterED+GW={Gerr_emb:.6e}, ratio={Gerr_emb/Gerr_bg:.6f}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    outfile = args.out / (
        f"cluster_ed_gw_L{args.Lx}x{args.Ly}_V{args.V:.6g}_fill{args.filling:.6g}.npz"
    )
    np.savez_compressed(
        outfile,
        Lx=int(args.Lx),
        Ly=int(args.Ly),
        V=float(args.V),
        filling=float(args.filling),
        T=float(args.T),
        ti=float(args.ti),
        t1=float(args.t1),
        t2=float(args.t2),
        omega=np.asarray(grid.omega),
        Omega=np.asarray(grid.Omega),
        converged=bool(result.converged),
        iterations=int(result.iterations),
        final_error=float(result.final_error),
        impurity_mismatch=float(result.impurity_mismatch),
        bath_fit_error=float(result.bath_fit_error),
        mu=float(result.mu),
        density=np.asarray(result.density),
        G=np.asarray(result.G),
        W=np.asarray(result.W),
        P=np.asarray(result.P),
        Sigma_H=np.asarray(result.Sigma_H),
        Sigma_emb=np.asarray(result.Sigma_emb),
        Sigma_GW_lattice=np.asarray(result.Sigma_GW_lattice),
        Sigma_GW_cluster=np.asarray(result.Sigma_GW_cluster),
        Sigma_ED_cluster=np.asarray(result.Sigma_ED_cluster),
        G_cluster=np.asarray(result.G_cluster),
        G_impurity=np.asarray(result.G_impurity),
        bath_energies=np.asarray(result.bath.energies),
        bath_couplings=np.asarray(result.bath.couplings),
        G_background=np.asarray(result.background.G),
        mu_background=float(result.background.mu),
        Gerr_background=float(Gerr_bg),
        Gerr_embedded=float(Gerr_emb),
        mu_ed=float(mu_ed),
        G_ed=np.asarray(G_ed),
    )
    print(f"saved {outfile}", flush=True)


if __name__ == "__main__":
    main()
