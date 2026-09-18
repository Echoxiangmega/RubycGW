#!/usr/bin/env python3
"""Export/plot a V-filling instability summary produced by the grid driver."""
from __future__ import annotations

import argparse
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _save_heatmap(path, Vs, ns, Z, title, cbar_label):
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(Vs, ns, Z, shading="nearest")
    ax.set_xlabel("V")
    ax.set_ylabel("filling n")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = _args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    out = args.out or args.input.with_suffix("").with_name(args.input.stem + "_plots")
    out.mkdir(parents=True, exist_ok=True)

    with np.load(args.input, allow_pickle=False) as z:
        d = {k: np.asarray(z[k]) for k in z.files}
    Vs = np.asarray(d["V_values"], dtype=float)
    ns = np.asarray(d["fillings"], dtype=float)
    dlc = np.asarray(d["d_lc"], dtype=float)
    dco = np.asarray(d["d_co"], dtype=float)
    delta = np.asarray(d["delta_d_co_minus_lc"], dtype=float)
    llc = np.asarray(d["lambda_lc"], dtype=complex)
    lco = np.asarray(d["lambda_co"], dtype=complex)
    qlc = np.asarray(d["q_lc"], dtype=int)
    qco = np.asarray(d["q_co"], dtype=int)

    _save_heatmap(
        out / "d_lc.png", Vs, ns, dlc,
        r"LC softness: $\min_q |1-\lambda_{LC}|$",
        r"$d_{LC}$",
    )
    _save_heatmap(
        out / "d_co.png", Vs, ns, dco,
        r"CO softness: $\min_q |1-\lambda_{CO}|$",
        r"$d_{CO}$",
    )
    _save_heatmap(
        out / "dco_minus_dlc.png", Vs, ns, delta,
        r"Competition: $d_{CO}-d_{LC}$",
        r"$d_{CO}-d_{LC}$",
    )
    _save_heatmap(
        out / "min_distance.png", Vs, ns, np.minimum(dlc, dco),
        r"Nearest JF pole: $\min(d_{CO},d_{LC})$",
        r"$d_{\min}$",
    )

    csv_path = out / "vn_instability_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "V", "filling",
            "d_lc", "lambda_lc_real", "lambda_lc_imag", "q_lc_1", "q_lc_2",
            "d_co", "lambda_co_real", "lambda_co_imag", "q_co_1", "q_co_2",
            "d_co_minus_d_lc",
            "lc_same_weight", "lc_opposite_weight",
            "co_even_weight", "co_odd_weight",
            "background_residual", "bath_error", "impurity_mismatch",
        ])
        for j, n in enumerate(ns):
            for i, V in enumerate(Vs):
                w.writerow([
                    float(V), float(n),
                    float(dlc[j, i]), float(llc[j, i].real), float(llc[j, i].imag),
                    int(qlc[j, i, 0]), int(qlc[j, i, 1]),
                    float(dco[j, i]), float(lco[j, i].real), float(lco[j, i].imag),
                    int(qco[j, i, 0]), int(qco[j, i, 1]),
                    float(delta[j, i]),
                    float(d["lc_same_weight"][j, i]),
                    float(d["lc_opposite_weight"][j, i]),
                    float(d["co_even_weight"][j, i]),
                    float(d["co_odd_weight"][j, i]),
                    float(d["background_residual"][j, i]),
                    float(d["background_bath_error"][j, i]),
                    float(d["background_impurity_mismatch"][j, i]),
                ])

    print(f"saved plots/table in {out}")
    print(f"table: {csv_path}")


if __name__ == "__main__":
    main()
