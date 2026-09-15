#!/usr/bin/env python3
"""Plot the hex18 uniform-current spectral scan saved by scan_hex18_uniform_spectrum.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--nplot", type=int, default=8,
                   help="number of lowest excited-state branches to display")
    p.add_argument("--weight-scale", type=float, default=900.0,
                   help="scatter marker area multiplier for spectral weights")
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    d = np.load(args.input, allow_pickle=False)
    V = np.asarray(d["V"], dtype=float)
    gaps = np.asarray(d["gaps"], dtype=float)
    weights = np.asarray(d["weights"], dtype=float)
    first_gap = np.asarray(d["first_gap"], dtype=float)
    current_gap = np.asarray(d["current_gap"], dtype=float)
    current_weight = np.asarray(d["current_weight"], dtype=float)
    delta_eff = np.asarray(d["delta_eff_low"], dtype=float)
    chi_low = np.asarray(d["chi_low"], dtype=float)
    S_low = np.asarray(d["S_low"], dtype=float)

    outdir = args.outdir if args.outdir is not None else args.input.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    # 1) Low-energy spectrum, with marker size proportional to uniform-current weight.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    nplot = min(int(args.nplot), gaps.shape[1] - 1)
    for n in range(1, nplot + 1):
        ax.plot(V, gaps[:, n], marker="o", linewidth=1.2, markersize=3.5,
                label=fr"$n={n}$")
        sizes = np.nan_to_num(weights[:, n], nan=0.0, posinf=0.0, neginf=0.0)
        ax.scatter(V, gaps[:, n], s=np.maximum(8.0, args.weight_scale * sizes), alpha=0.45)
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$E_n-E_0$")
    ax.set_title("Hex18 low-energy spectrum\nmarker size = uniform-current spectral weight")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    f1 = outdir / f"{stem}_spectrum.png"
    fig.savefig(f1, dpi=180)
    plt.close(fig)

    # 2) Direct gap diagnostics.
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(V, first_gap, marker="o", label=r"full first gap $\Delta_1$")
    ax.plot(V, current_gap, marker="s", label=r"dominant-current gap $\Delta_{\rm current}$")
    ax.plot(V, delta_eff, marker="^", label=r"$\Delta_{J}^{\rm eff}=2S_{\rm low}/\chi_{\rm low}$")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel("gap")
    ax.set_title("Hex18 gap diagnostics for the uniform loop-current channel")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    f2 = outdir / f"{stem}_gaps.png"
    fig.savefig(f2, dpi=180)
    plt.close(fig)

    # 3) Current spectral weight carried by each low-energy state.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for n in range(1, nplot + 1):
        ax.plot(V, weights[:, n], marker="o", linewidth=1.2, markersize=3.5,
                label=fr"$n={n}$")
    ax.plot(V, current_weight, linewidth=2.2, linestyle="--",
            label=r"largest $|\langle n|J_u|0\rangle|^2$")
    ax.set_xlabel(r"$V$")
    ax.set_ylabel(r"$|\langle n|J_u|0\rangle|^2$")
    ax.set_title("Uniform-current spectral weights")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    f3 = outdir / f"{stem}_weights.png"
    fig.savefig(f3, dpi=180)
    plt.close(fig)

    # 4) Low-energy partial susceptibility and equal-time weight.
    fig, ax1 = plt.subplots(figsize=(7.0, 5.0))
    line1 = ax1.plot(V, chi_low, marker="o", label=r"$\chi_u^{\rm low}$")
    ax1.set_xlabel(r"$V$")
    ax1.set_ylabel(r"$\chi_u^{\rm low}$")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    line2 = ax2.plot(V, S_low, marker="s", linestyle="--", label=r"$S_u^{\rm low}$")
    ax2.set_ylabel(r"$S_u^{\rm low}$")
    lines = line1 + line2
    ax1.legend(lines, [x.get_label() for x in lines], loc="best")
    ax1.set_title("Low-energy uniform-current response")
    fig.tight_layout()
    f4 = outdir / f"{stem}_response.png"
    fig.savefig(f4, dpi=180)
    plt.close(fig)

    print("saved:")
    for f in (f1, f2, f3, f4):
        print(f"  {f}")


if __name__ == "__main__":
    main()
