#!/usr/bin/env python3
"""Compare the new 1x1 ring3-supercell benchmark with historical Hex18 CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pandas as pd
import numpy as np


def best(df, predicate):
    x=df[predicate(df)].copy()
    if x.empty:
        return None
    x["E"]=pd.to_numeric(x["E"],errors="coerce")
    x=x[np.isfinite(x["E"])]
    if x.empty:
        return None
    return x.loc[x["E"].idxmin()]


def time_per_iter(row):
    if row is None:
        return np.nan
    if "time_per_iteration_s" in row.index and pd.notna(row["time_per_iteration_s"]):
        return float(row["time_per_iteration_s"])
    if (
        "time_solve_or_opt_s" in row.index
        and "n_iter" in row.index
        and pd.notna(row["time_solve_or_opt_s"])
        and pd.notna(row["n_iter"])
        and float(row["n_iter"])>0
    ):
        return float(row["time_solve_or_opt_s"])/float(row["n_iter"])
    return np.nan


def record(label,row,projection_images=None):
    if row is None:
        return None
    keys=[
        "E","delta_E_vs_ED","fidelity_ground_subspace",
        "exact_energy_variance","CO_qmax","S_BOW_star",
        "S_LC_same","S_LC_opposite",
    ]
    out={
        "label":label,
        "method":row.get("method",""),
        "seed":row.get("seed",""),
        "explicit_projection_images":(
            projection_images
            if projection_images is not None
            else row.get("explicit_projection_images",np.nan)
        ),
        "time_per_iteration_s":time_per_iter(row),
    }
    for key in keys:
        out[key]=row.get(key,np.nan)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--old",required=True,help="Historical Hex18 comparison_all.csv")
    p.add_argument("--new",required=True,help="New comparison_supercell.csv")
    p.add_argument("--output",default="comparison_supercell_vs_hex18.csv")
    a=p.parse_args()

    old=pd.read_csv(a.old)
    new=pd.read_csv(a.new)

    old_raw=best(old,lambda d:d["method"].astype(str).str.startswith("Raw-"))
    old_vap=best(old,lambda d:
        d["method"].astype(str).str.startswith("Full-VAP")
        & ~d["method"].astype(str).str.endswith("+1LS")
    )
    old_vap_ls=best(old,lambda d:
        d["method"].astype(str).str.startswith("Full-VAP")
        & d["method"].astype(str).str.endswith("+1LS")
    )
    new_raw=best(new,lambda d:
        d["method"].astype(str).str.startswith("Raw-SupercellBloch")
    )
    new_c3=best(new,lambda d:
        d["method"].astype(str).str.startswith("VAP-C3-SupercellBloch")
        & ~d["method"].astype(str).str.endswith("+1LS")
    )
    new_c3_ls=best(new,lambda d:
        d["method"].astype(str).str.startswith("VAP-C3-SupercellBloch")
        & d["method"].astype(str).str.endswith("+1LS")
    )

    rows=[
        record("old Raw Hex18",old_raw,1),
        # Historical full spatial ring3 projector: 3 primitive translations
        # x 6 C3v operations = 18 base-wavefunction images. TR restoration
        # conjugates the projected amplitude but does not add 18 more base calls.
        record("old full spatial/TR VAP Hex18",old_vap,18),
        record("old full spatial/TR VAP Hex18 +1LS",old_vap_ls,18),
        record("new 1x1 supercell Raw",new_raw,1),
        record("new 1x1 supercell C3 VAP",new_c3,3),
        record("new 1x1 supercell C3 VAP +1LS",new_c3_ls,3),
    ]
    rows=[r for r in rows if r is not None]

    if new_raw is not None and old_raw is not None:
        dE=abs(float(new_raw["E"])-float(old_raw["E"]))
        print(f"Raw implementation energy difference |E_new-E_old| = {dE:.6e}")
    if new_c3 is not None and old_vap is not None:
        tn=time_per_iter(new_c3); to=time_per_iter(old_vap)
        if np.isfinite(tn) and np.isfinite(to) and tn>0:
            print(f"C3-only / old full-VAP time-per-iteration ratio = {tn/to:.4f}")
            print(f"old full-VAP / C3-only speedup = {to/tn:.2f}x")

    fields=list(rows[0].keys()) if rows else []
    with Path(a.output).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print()
    for r in rows:
        print(
            f"{r['label']:38s} "
            f"seed={str(r['seed']):11s} "
            f"dE={float(r['delta_E_vs_ED']):+.4e} "
            f"F={float(r['fidelity_ground_subspace']):.6f} "
            f"var={float(r['exact_energy_variance']):.3e} "
            f"t/it={float(r['time_per_iteration_s']):.3f}s"
        )
    print("Saved:",Path(a.output).resolve())


if __name__=="__main__":
    main()
