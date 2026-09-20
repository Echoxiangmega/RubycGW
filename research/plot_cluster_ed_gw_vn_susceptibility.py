#!/usr/bin/env python3
"""Plot direct physical-susceptibility diagnostics for a V-filling scan."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _finite_limits(a):
    x = np.asarray(a, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return -1.0, 1.0
    m = float(np.max(np.abs(x)))
    return (-m, m) if m > 0 else (-1.0, 1.0)


def _plot_single_filling(outdir, d):
    Vs = np.asarray(d["V_values"], dtype=float)
    n = float(np.asarray(d["fillings"], dtype=float)[0])
    Lx = int(np.asarray(d["Lx"]).reshape(()))
    Ly = int(np.asarray(d["Ly"]).reshape(()))

    for family, key in (("CO", "inv_chi_co_q"), ("LC", "inv_chi_lc_q")):
        fig, ax = plt.subplots()
        arr = np.asarray(d[key], dtype=float)[0]
        for q1 in range(Lx):
            for q2 in range(Ly):
                ax.plot(
                    Vs,
                    arr[:, q1, q2],
                    marker="o",
                    label=f"q=({q1},{q2})",
                )
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("V")
        ax.set_ylabel(r"$1/\chi_{\mathrm{soft}}$")
        ax.set_title(f"{family} inverse susceptibility, filling={n:g}")
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(outdir / f"inv_chi_{family.lower()}_vs_V_fill{n:g}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(Vs, np.asarray(d["inv_chi_co"], dtype=float)[0], marker="o", label="CO closest-to-zero mass")
    ax.plot(Vs, np.asarray(d["inv_chi_lc"], dtype=float)[0], marker="s", label="LC closest-to-zero mass")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("V")
    ax.set_ylabel(r"$1/\chi_{\mathrm{soft}}$")
    ax.set_title(f"Physical instability masses, filling={n:g}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"inv_chi_co_lc_vs_V_fill{n:g}.png", dpi=180)
    plt.close(fig)

    if "co_mass_eigvals_q" in d and "lc_mass_eigvals_q" in d:
        co_mass = np.asarray(d["co_mass_eigvals_q"], dtype=float)[0]
        lc_mass = np.asarray(d["lc_mass_eigvals_q"], dtype=float)[0]
        fig, ax = plt.subplots()
        for q1 in range(Lx):
            for q2 in range(Ly):
                for im in range(co_mass.shape[-1]):
                    ax.plot(
                        Vs,
                        co_mass[:, q1, q2, im],
                        marker="o",
                        linewidth=1.0,
                        alpha=0.75,
                        label=f"CO q=({q1},{q2}) m{im}" if im == 0 else None,
                    )
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("V")
        ax.set_ylabel(r"$m_\nu=\mathrm{eig}(\chi_{CO}^{-1})$")
        ax.set_title(f"All CO physical masses, filling={n:g}")
        ax.legend(fontsize="x-small")
        fig.tight_layout()
        fig.savefig(outdir / f"co_mass_spectrum_vs_V_fill{n:g}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots()
        for q1 in range(Lx):
            for q2 in range(Ly):
                for im in range(lc_mass.shape[-1]):
                    ax.plot(
                        Vs,
                        lc_mass[:, q1, q2, im],
                        marker="s",
                        linewidth=1.0,
                        alpha=0.75,
                        label=f"LC q=({q1},{q2}) m{im}" if im == 0 else None,
                    )
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xlabel("V")
        ax.set_ylabel(r"$m_\nu=\mathrm{eig}(\chi_{LC}^{-1})$")
        ax.set_title(f"All LC physical masses, filling={n:g}")
        ax.legend(fontsize="x-small")
        fig.tight_layout()
        fig.savefig(outdir / f"lc_mass_spectrum_vs_V_fill{n:g}.png", dpi=180)
        plt.close(fig)


def _plot_2d_field(path, Vs, fillings, field, title, cbar):
    fig, ax = plt.subplots()
    lo, hi = _finite_limits(field)
    mesh = ax.pcolormesh(Vs, fillings, field, shading="auto", vmin=lo, vmax=hi)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(cbar)
    ax.set_xlabel("V")
    ax.set_ylabel("filling n")
    ax.set_title(title)
    if len(Vs) >= 2 and len(fillings) >= 2:
        try:
            ax.contour(Vs, fillings, field, levels=[0.0], linewidths=1.5)
        except ValueError:
            pass
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_multi_filling(outdir, d):
    Vs = np.asarray(d["V_values"], dtype=float)
    fillings = np.asarray(d["fillings"], dtype=float)
    invco = np.asarray(d["inv_chi_co"], dtype=float)
    invlc = np.asarray(d["inv_chi_lc"], dtype=float)

    _plot_2d_field(
        outdir / "inv_chi_co_map.png",
        Vs,
        fillings,
        invco,
        "Softest physical CO inverse susceptibility",
        r"$1/\chi_{CO}$",
    )
    _plot_2d_field(
        outdir / "inv_chi_lc_map.png",
        Vs,
        fillings,
        invlc,
        "Softest physical LC inverse susceptibility",
        r"$1/\chi_{LC}$",
    )

    # Positive values mean LC is closer to a pole in inverse-susceptibility
    # distance; negative values mean CO is closer.
    competition = np.abs(invco) - np.abs(invlc)
    _plot_2d_field(
        outdir / "chi_softness_competition.png",
        Vs,
        fillings,
        competition,
        "Physical softness: |1/chi_CO| - |1/chi_LC|",
        r"$|1/\chi_{CO}|-|1/\chi_{LC}|$",
    )


def _write_csv(path, d):
    Vs = np.asarray(d["V_values"], dtype=float)
    ns = np.asarray(d["fillings"], dtype=float)
    fields = [
        "V", "filling",
        "chi_co", "inv_chi_co", "q_co_1", "q_co_2",
        "co_even_weight", "co_odd_weight",
        "chi_lc", "inv_chi_lc", "q_lc_1", "q_lc_2",
        "lc_same_weight", "lc_opposite_weight",
        "global_co_min_mass", "global_lc_min_mass",
        "total_co_negative_modes", "total_lc_negative_modes",
        "max_q_pair_residual", "max_co_lc_cross_ratio", "max_solver_residual",
        "failed",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for inn, n in enumerate(ns):
            for iv, V in enumerate(Vs):
                qco = np.asarray(d["q_co"][inn, iv], dtype=int)
                qlc = np.asarray(d["q_lc"][inn, iv], dtype=int)
                w.writerow(dict(
                    V=float(V),
                    filling=float(n),
                    chi_co=float(d["chi_co"][inn, iv]),
                    inv_chi_co=float(d["inv_chi_co"][inn, iv]),
                    q_co_1=int(qco[0]),
                    q_co_2=int(qco[1]),
                    co_even_weight=float(d["co_even_weight"][inn, iv]),
                    co_odd_weight=float(d["co_odd_weight"][inn, iv]),
                    chi_lc=float(d["chi_lc"][inn, iv]),
                    inv_chi_lc=float(d["inv_chi_lc"][inn, iv]),
                    q_lc_1=int(qlc[0]),
                    q_lc_2=int(qlc[1]),
                    lc_same_weight=float(d["lc_same_weight"][inn, iv]),
                    lc_opposite_weight=float(d["lc_opposite_weight"][inn, iv]),
                    global_co_min_mass=float(
                        d["global_co_min_mass"][inn, iv]
                    ) if "global_co_min_mass" in d else np.nan,
                    global_lc_min_mass=float(
                        d["global_lc_min_mass"][inn, iv]
                    ) if "global_lc_min_mass" in d else np.nan,
                    total_co_negative_modes=int(
                        d["total_co_negative_modes"][inn, iv]
                    ) if "total_co_negative_modes" in d else 0,
                    total_lc_negative_modes=int(
                        d["total_lc_negative_modes"][inn, iv]
                    ) if "total_lc_negative_modes" in d else 0,
                    max_q_pair_residual=float(d["max_q_pair_residual"][inn, iv]),
                    max_co_lc_cross_ratio=float(d["max_co_lc_cross_ratio"][inn, iv]),
                    max_solver_residual=float(d["max_solver_residual"][inn, iv]),
                    failed=bool(d["point_failed"][inn, iv]),
                ))


def main():
    args = _args()
    d = _load(args.summary)
    outdir = args.out or args.summary.parent
    outdir.mkdir(parents=True, exist_ok=True)
    fillings = np.asarray(d["fillings"], dtype=float)
    if len(fillings) == 1:
        _plot_single_filling(outdir, d)
    else:
        _plot_multi_filling(outdir, d)
    _write_csv(outdir / "vn_susceptibility_summary.csv", d)
    print(f"saved susceptibility plots/CSV in {outdir}")


if __name__ == "__main__":
    main()
