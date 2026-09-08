#!/usr/bin/env python3
"""Exact finite-temperature ED scan of q=0 pseudospin susceptibilities versus V.

This script intentionally matches the ED definition used by
``benchmark_ed_gw_cgw_sox.py``:

* ``ExactSmallRubyThermal`` on the same finite periodic Ruby torus;
* grand-canonical finite-temperature susceptibility;
* fixed average filling set through the exact chemical potential;
* q=0 pseudospin operators.

The default 2x1 torus has 12 sites, so all particle-number sectors together
contain only 2**12 = 4096 states and can be fully diagonalized exactly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal


_DEFAULT_CHANNELS = ("x_even", "z_same", "z_opposite")
_CHANNEL_TEX = {
    "x_even": r"$x_{\rm even}$",
    "z_same": r"$z_{\rm same}$",
    "z_opposite": r"$z_{\rm opposite}$",
}


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L1", type=int, default=2)
    p.add_argument("--L2", type=int, default=1)
    p.add_argument("--Vmin", type=float, default=0.0)
    p.add_argument("--Vmax", type=float, default=2.0)
    p.add_argument("--dV", type=float, default=0.05)
    p.add_argument(
        "--V",
        nargs="*",
        type=float,
        default=None,
        help="explicit V values; overrides --Vmin/--Vmax/--dV",
    )
    p.add_argument("--channels", nargs="+", default=list(_DEFAULT_CHANNELS))
    p.add_argument("--filling", type=float, default=3.0, help="particles per primitive cell")
    p.add_argument("--T", type=float, default=0.08)
    p.add_argument("--ti", type=float, default=0.4)
    p.add_argument("--t1", type=float, default=0.2)
    p.add_argument("--t2", type=float, default=0.2)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/ed_chi_vs_V"),
        help="output directory",
    )
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def _v_grid(args) -> np.ndarray:
    if args.V:
        return np.asarray(sorted(set(float(x) for x in args.V)), dtype=float)
    if args.dV <= 0.0:
        raise ValueError("--dV must be positive")
    if args.Vmax < args.Vmin:
        raise ValueError("--Vmax must be >= --Vmin")
    n = int(np.floor((args.Vmax - args.Vmin) / args.dV + 1e-12))
    vals = args.Vmin + args.dV * np.arange(n + 1, dtype=float)
    if vals.size == 0 or vals[-1] < args.Vmax - 1e-12:
        vals = np.append(vals, float(args.Vmax))
    return vals


def _plot(V, channels, chi_diag, T, path, dpi):
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for ic, ch in enumerate(channels):
        ax.plot(
            V,
            chi_diag[:, ic],
            marker="o",
            ms=3.2,
            label=_CHANNEL_TEX.get(ch, ch),
        )
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"static susceptibility $\chi$")
    ax.set_title(rf"Exact ED, $q=0$, $T={T:g}$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def main():
    args = _args()
    if args.T <= 0.0:
        raise ValueError("--T must be positive")
    if 6 * args.L1 * args.L2 > 16:
        raise ValueError("ExactSmallRubyThermal requires at most 16 sites")

    Vvalues = _v_grid(args)
    channels = [str(ch) for ch in args.channels]
    ncell = int(args.L1) * int(args.L2)
    target_particles = float(args.filling) * ncell

    params = RubyParameters(ti=args.ti, t1=args.t1, t2=args.t2, V=0.0)
    exact = ExactSmallRubyThermal(args.L1, args.L2, params)
    operators = np.stack(
        [exact.pseudospin_operator(ch, (0.0, 0.0)) for ch in channels]
    )

    nV = len(Vvalues)
    nc = len(channels)
    chi = np.full((nV, nc, nc), np.nan, dtype=float)
    mu = np.full(nV, np.nan, dtype=float)

    print("=== exact finite-T ED q=0 susceptibility scan ===")
    print(
        f"cluster={args.L1}x{args.L2} primitive cells, sites={exact.n_sites}, "
        f"target particles={target_particles:g}"
    )
    print(
        f"T={args.T:g}, filling={args.filling:g}, "
        f"ti={args.ti:g}, t1={args.t1:g}, t2={args.t2:g}"
    )
    print("channels=" + ", ".join(channels))

    for iv, V in enumerate(Vvalues):
        exact.diagonalize(float(V))
        mu[iv] = exact.solve_mu(target_particles, args.T)
        chi[iv], _ = exact.static_susceptibility_matrix(
            operators,
            mu[iv],
            args.T,
        )
        diag = np.diag(chi[iv])
        values = "  ".join(f"{ch}={diag[ic]:.8f}" for ic, ch in enumerate(channels))
        print(f"V={V:8.5f}  mu={mu[iv]:+.10f}  {values}")

    chi_diag = np.real(np.diagonal(chi, axis1=1, axis2=2))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "ed_chi_scan.npz"
    plot_path = outdir / "ed_chi_vs_V.png"

    np.savez_compressed(
        npz_path,
        V=Vvalues,
        channels=np.asarray(channels),
        chi=chi,
        chi_diag=chi_diag,
        mu=mu,
        L1=int(args.L1),
        L2=int(args.L2),
        n_sites=int(exact.n_sites),
        filling=float(args.filling),
        target_particles=float(target_particles),
        T=float(args.T),
        ti=float(args.ti),
        t1=float(args.t1),
        t2=float(args.t2),
        q=np.asarray([0.0, 0.0]),
        ensemble=np.asarray("grand_canonical_exact_fixed_average_filling"),
    )
    _plot(Vvalues, channels, chi_diag, args.T, plot_path, args.dpi)

    print("saved:", npz_path)
    print("saved:", plot_path)


if __name__ == "__main__":
    main()
