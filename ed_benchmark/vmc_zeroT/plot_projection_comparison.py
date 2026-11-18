#!/usr/bin/env python3
"""Plot the standard five diagnostics from comparison_all.csv.

Works with both the density-Jastrow and bond-backflow projection benchmarks.

Outputs:
  1) best_energy_vs_ed.png
  2) best_observables_vs_ed.png
  3) best_fidelity.png
  4) pav_vap_rank_consistency.png
  5) pav_vap_energy_fidelity.png

It also writes:
  best_results_for_plotting.csv
  pav_vap_seed_ranking.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "csv",
        nargs="?",
        default="comparison_all.csv",
        help="Path to comparison_all.csv",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory; default is <csv parent>/plots",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
    )
    return p


def detect_methods(df: pd.DataFrame):
    methods = set(df["method"].dropna().astype(str))

    def pick(prefix, contains_backflow=None):
        cand = [m for m in methods if m.startswith(prefix)]
        if contains_backflow is not None:
            cand = [
                m for m in cand
                if ("Backflow" in m) == bool(contains_backflow)
            ]
        return sorted(cand)[0] if cand else None

    has_backflow = any("Backflow" in m for m in methods)

    raw = pick("Raw-Slater-Jastrow", has_backflow)
    pav = pick("PAV-TR-Slater-Jastrow", has_backflow)
    vap = pick("Full-VAP-TR-Slater-Jastrow", has_backflow)

    # Exclude +1LS variants from base PAV/VAP.
    if pav is not None and pav.endswith("+1LS"):
        pav = None
    if vap is not None and vap.endswith("+1LS"):
        vap = None
    if pav is None:
        cand = [
            m for m in methods
            if m.startswith("PAV-TR-Slater-Jastrow")
            and "+1LS" not in m
            and (("Backflow" in m) == has_backflow)
        ]
        pav = sorted(cand)[0] if cand else None
    if vap is None:
        cand = [
            m for m in methods
            if m.startswith("Full-VAP-TR-Slater-Jastrow")
            and "+1LS" not in m
            and (("Backflow" in m) == has_backflow)
        ]
        vap = sorted(cand)[0] if cand else None

    pav_ls = None if pav is None else pav + "+1LS"
    vap_ls = None if vap is None else vap + "+1LS"

    ordered = ["ED", raw, pav, pav_ls, vap, vap_ls]
    ordered = [m for m in ordered if m in methods]

    return {
        "backflow": has_backflow,
        "raw": raw,
        "pav": pav,
        "pav_ls": pav_ls if pav_ls in methods else None,
        "vap": vap,
        "vap_ls": vap_ls if vap_ls in methods else None,
        "ordered": ordered,
    }


def short_name(method: str):
    if method == "ED":
        return "ED"
    bf = " + BF" if "Backflow" in method else ""
    ls = "+1LS" if method.endswith("+1LS") else ""
    if method.startswith("Raw-"):
        return "Raw" + bf
    if method.startswith("PAV-TR"):
        return "PAV-TR" + bf + ls
    if method.startswith("Full-VAP-TR"):
        return "VAP-TR" + bf + ls
    return method


def best_by_method(df: pd.DataFrame, methods):
    rows = []
    for m in methods:
        sub = df[df["method"] == m].copy()
        if sub.empty:
            continue
        if m == "ED":
            rows.append(sub.iloc[0])
        else:
            rows.append(sub.loc[sub["E"].idxmin()])
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["method_short"] = out["method"].map(short_name)
    return out


def prepare_numeric(df):
    numeric = [
        "E",
        "delta_E_vs_ED",
        "fidelity_ground_subspace",
        "exact_energy_variance",
        "CO_qmax",
        "S_BOW_star",
        "S_LC_same",
        "S_LC_opposite",
    ]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_best_energy(best, out, dpi):
    x = np.arange(len(best))
    y = best["E"].to_numpy(float)
    labels = best["method_short"].tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, y)
    ed_e = float(best.loc[best["method"] == "ED", "E"].iloc[0])
    ax.axhline(ed_e, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Energy E")
    ax.set_title("Best energy of each method")
    for xi, yi in zip(x, y):
        ax.text(xi, yi, f"{yi:.5f}", ha="center", va="bottom", fontsize=8, rotation=90)
    fig.tight_layout()
    fig.savefig(out / "best_energy_vs_ed.png", dpi=dpi)
    plt.close(fig)


def save_best_observables(best, out, dpi):
    cols = ["CO_qmax", "S_BOW_star", "S_LC_same", "S_LC_opposite"]
    names = ["CO_qmax", "S_BOW*", "S_LC+", "S_LC-"]
    x = np.arange(len(cols))
    width = min(0.13, 0.8 / max(1, len(best)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(best.iterrows()):
        vals = [float(row[c]) for c in cols]
        shift = (i - (len(best)-1)/2) * width
        ax.bar(x + shift, vals, width=width, label=row["method_short"])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Observable value")
    ax.set_title("Best-result observables compared with ED")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "best_observables_vs_ed.png", dpi=dpi)
    plt.close(fig)


def save_best_fidelity(best, out, dpi):
    sub = best[best["method"] != "ED"].copy()
    x = np.arange(len(sub))
    y = sub["fidelity_ground_subspace"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, y)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["method_short"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Ground-state fidelity F")
    ax.set_ylim(0, 1.05)
    ax.set_title("Best fidelity of each variational method")
    for xi, yi in zip(x, y):
        ax.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "best_fidelity.png", dpi=dpi)
    plt.close(fig)


def ranking_table(df, pav_method, vap_method):
    pieces = []
    for m, label in [(pav_method, "PAV-TR"), (vap_method, "VAP-TR")]:
        if m is None:
            continue
        sub = df[df["method"] == m].copy().sort_values("E").reset_index(drop=True)
        sub["rank_E"] = np.arange(1, len(sub)+1)
        sub["method_short"] = label
        pieces.append(
            sub[
                [
                    "method_short",
                    "rank_E",
                    "seed",
                    "E",
                    "fidelity_ground_subspace",
                    "S_LC_same",
                    "S_LC_opposite",
                ]
            ]
        )
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def save_rank_consistency(df, pav_method, vap_method, out, dpi):
    if pav_method is None or vap_method is None:
        return
    pav = df[df["method"] == pav_method].copy().sort_values("E").reset_index(drop=True)
    vap = df[df["method"] == vap_method].copy().sort_values("E").reset_index(drop=True)
    pav["rank"] = np.arange(1, len(pav)+1)
    vap["rank"] = np.arange(1, len(vap)+1)

    seeds = sorted(set(pav["seed"]).intersection(vap["seed"]))
    rp = dict(zip(pav["seed"], pav["rank"]))
    rv = dict(zip(vap["seed"], vap["rank"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    for seed in seeds:
        ax.plot([0, 1], [rp[seed], rv[seed]], marker="o")
        ax.text(-0.02, rp[seed], str(seed), ha="right", va="center", fontsize=9)
        ax.text(1.02, rv[seed], str(seed), ha="left", va="center", fontsize=9)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(len(seeds)+0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["PAV-TR", "VAP-TR"])
    ax.set_ylabel("Energy rank (1 = best)")
    ax.set_title("Seed ranking consistency: PAV vs VAP")
    fig.tight_layout()
    fig.savefig(out / "pav_vap_rank_consistency.png", dpi=dpi)
    plt.close(fig)


def save_energy_fidelity(df, pav_method, vap_method, out, dpi):
    fig, ax = plt.subplots(figsize=(7, 5))
    any_data = False
    for m, label in [(pav_method, "PAV-TR"), (vap_method, "VAP-TR")]:
        if m is None:
            continue
        sub = df[df["method"] == m].copy()
        if sub.empty:
            continue
        any_data = True
        ax.scatter(sub["E"], sub["fidelity_ground_subspace"], label=label)
        for _, row in sub.iterrows():
            ax.text(
                float(row["E"]),
                float(row["fidelity_ground_subspace"]),
                str(row["seed"]),
                fontsize=8,
                va="bottom",
            )
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("Energy E")
    ax.set_ylabel("Ground-state fidelity F")
    ax.set_title("Energy vs fidelity for PAV and VAP seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pav_vap_energy_fidelity.png", dpi=dpi)
    plt.close(fig)


def main():
    a = parser().parse_args()
    csv_path = Path(a.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out = (
        Path(a.output).resolve()
        if a.output is not None
        else csv_path.parent / "plots"
    )
    out.mkdir(parents=True, exist_ok=True)

    df = prepare_numeric(pd.read_csv(csv_path))
    if "method" not in df or "seed" not in df:
        raise RuntimeError("CSV must contain method and seed columns")

    methods = detect_methods(df)
    best = best_by_method(df, methods["ordered"])
    rank = ranking_table(df, methods["pav"], methods["vap"])

    keep = [
        "method_short",
        "seed",
        "E",
        "delta_E_vs_ED",
        "fidelity_ground_subspace",
        "exact_energy_variance",
        "CO_qmax",
        "S_BOW_star",
        "S_LC_same",
        "S_LC_opposite",
    ]
    best[keep].to_csv(out / "best_results_for_plotting.csv", index=False)
    rank.to_csv(out / "pav_vap_seed_ranking.csv", index=False)

    save_best_energy(best, out, a.dpi)
    save_best_observables(best, out, a.dpi)
    save_best_fidelity(best, out, a.dpi)
    save_rank_consistency(df, methods["pav"], methods["vap"], out, a.dpi)
    save_energy_fidelity(df, methods["pav"], methods["vap"], out, a.dpi)

    print(f"[plot] detected correlator={'backflow' if methods['backflow'] else 'sj'}")
    print("[plot] methods:")
    for m in methods["ordered"]:
        print(" ", m)
    print("[plot] saved to", out)
    for name in (
        "best_energy_vs_ed.png",
        "best_observables_vs_ed.png",
        "best_fidelity.png",
        "pav_vap_rank_consistency.png",
        "pav_vap_energy_fidelity.png",
        "best_results_for_plotting.csv",
        "pav_vap_seed_ranking.csv",
    ):
        print(" ", out / name)


if __name__ == "__main__":
    main()
