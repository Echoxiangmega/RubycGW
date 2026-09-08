#!/usr/bin/env python3
"""Plot an existing 12-site four-layer benchmark NPZ without rerunning ED/GW/cGW."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_CHANNEL_TEX = {
    "x_even": r"x_{\rm even}",
    "x_odd": r"x_{\rm odd}",
    "y_even": r"y_{\rm even}",
    "y_odd": r"y_{\rm odd}",
    "z_same": r"z_{\rm same}",
    "z_opposite": r"z_{\rm opposite}",
}


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def main():
    args = _args()
    d = np.load(args.npz, allow_pickle=False)
    tau = np.asarray(d["tau"], dtype=float)
    beta = float(d["beta"])
    channels = tuple(str(x) for x in np.asarray(d["channels"]).tolist())
    qlabel = str(np.asarray(d["qlabel"]).item())
    exactC = np.asarray(d["exact_C_tau"])
    bubbleED = np.asarray(d["bubble_ed_C_tau"])
    bubbleGW = np.asarray(d["bubble_gw_C_tau"])
    fullC = np.asarray(d["full_cgw_C_tau"])

    x = tau / beta
    nc = len(channels)
    fig, axes = plt.subplots(nc, 2, figsize=(13.5, max(3.4 * nc, 4.0)), squeeze=False)
    for ic, ch in enumerate(channels):
        label = _CHANNEL_TEX.get(ch, ch.replace("_", r"\_"))
        ax = axes[ic, 0]
        ax.plot(x, np.real(exactC[:, ic]), label="exact ED")
        ax.plot(x, np.real(bubbleED[:, ic]), linestyle="--", label=r"bubble[$G_{ED}$]")
        ax.plot(x, np.real(bubbleGW[:, ic]), linestyle="-.", label=r"bubble[$G_{GW}$]")
        ax.plot(x, np.real(fullC[:, ic]), linestyle=":", linewidth=2.2, label="full cGW")
        ax.axvline(0.5, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(rf"$C_{{{label},{label}}}(\tau)$")
        if ic == nc - 1:
            ax.set_xlabel(r"$\tau/\beta$")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

        ax = axes[ic, 1]
        ax.plot(x, np.real(bubbleED[:, ic] - bubbleGW[:, ic]), label="background: bubble ED-GW")
        ax.plot(x, np.real(exactC[:, ic] - bubbleED[:, ic]), label="exact vertex: ED-bubbleED")
        ax.plot(x, np.real(fullC[:, ic] - bubbleGW[:, ic]), label="cGW vertex: full-bubbleGW")
        ax.plot(x, np.real(fullC[:, ic] - exactC[:, ic]), linestyle="--", label="total cGW-ED")
        ax.axhline(0.0, linewidth=0.8)
        ax.axvline(0.5, linestyle=":", linewidth=0.8, alpha=0.5)
        if ic == nc - 1:
            ax.set_xlabel(r"$\tau/\beta$")
        ax.set_ylabel("difference")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(f"12-site 2x1 torus four-layer diagnostic at {qlabel}", y=0.995)
    fig.tight_layout()
    out = args.out if args.out is not None else args.npz.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi))
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
