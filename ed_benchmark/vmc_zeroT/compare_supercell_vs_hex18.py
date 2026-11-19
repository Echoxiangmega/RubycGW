#!/usr/bin/env python3
"""Compare the new 1x1 ring3-supercell benchmark with historical Hex18 CSV."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_rows(path):
    with Path(path).open("r",newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(row,key):
    try:
        x=float(row.get(key,""))
    except Exception:
        return math.nan
    return x if math.isfinite(x) else math.nan


def best(rows,predicate):
    x=[r for r in rows if predicate(str(r.get("method","")))]
    x=[r for r in x if math.isfinite(number(r,"E"))]
    return min(x,key=lambda r:number(r,"E")) if x else None


def time_per_iter(row):
    if row is None:
        return math.nan
    x=number(row,"time_per_iteration_s")
    if math.isfinite(x):
        return x
    t=number(row,"time_solve_or_opt_s")
    n=number(row,"n_iter")
    if math.isfinite(t) and math.isfinite(n) and n>0:
        return t/n
    return math.nan


def record(label,row,projection_images=None):
    if row is None:
        return None
    keys=[
        "E","delta_E_vs_ED","fidelity_ground_subspace",
        "exact_energy_variance","CO_qmax","S_BOW_star",
        "S_LC_same","S_LC_opposite",
        "ed_primitive_translation_sector_weight",
        "ed_full_spatial_sector_weight",
        "time_reversal_residual",
    ]
    out={
        "label":label,
        "method":row.get("method",""),
        "seed":row.get("seed",""),
        "explicit_projection_images":(
            projection_images
            if projection_images is not None
            else row.get("explicit_projection_images","")
        ),
        "time_per_iteration_s":time_per_iter(row),
    }
    for key in keys:
        out[key]=number(row,key)
    return out


def fmt(x,spec):
    return format(x,spec) if math.isfinite(float(x)) else "nan"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--old",required=True,help="Historical Hex18 comparison_all.csv")
    p.add_argument("--new",required=True,help="New comparison_supercell.csv")
    p.add_argument("--output",default="comparison_supercell_vs_hex18.csv")
    a=p.parse_args()

    old=read_rows(a.old)
    new=read_rows(a.new)

    old_raw=best(old,lambda m:m.startswith("Raw-"))
    old_vap=best(old,lambda m:m.startswith("Full-VAP") and not m.endswith("+1LS"))
    old_vap_ls=best(old,lambda m:m.startswith("Full-VAP") and m.endswith("+1LS"))
    new_raw=best(new,lambda m:m.startswith("Raw-SupercellBloch"))
    new_c3v=best(new,lambda m:m.startswith("VAP-C3v-SupercellBloch") and not m.endswith("+1LS"))
    new_c3v_ls=best(new,lambda m:m.startswith("VAP-C3v-SupercellBloch") and m.endswith("+1LS"))
    new_c3vtr=best(new,lambda m:m.startswith("VAP-C3vTR-SupercellBloch") and not m.endswith("+1LS"))
    new_c3vtr_ls=best(new,lambda m:m.startswith("VAP-C3vTR-SupercellBloch") and m.endswith("+1LS"))
    # Keep old C3-only rows readable for previously completed runs.
    new_c3=best(new,lambda m:m.startswith("VAP-C3-SupercellBloch") and not m.endswith("+1LS"))
    new_c3_ls=best(new,lambda m:m.startswith("VAP-C3-SupercellBloch") and m.endswith("+1LS"))

    rows=[
        record("old Raw Hex18",old_raw,1),
        # Historical full spatial ring3 projector: 3 primitive translations
        # x 6 C3v operations = 18 base-wavefunction images. TR restoration
        # conjugates the projected amplitude but does not add 18 more base calls.
        record("old full spatial/TR VAP Hex18",old_vap,18),
        record("old full spatial/TR VAP Hex18 +1LS",old_vap_ls,18),
        record("new 1x1 supercell Raw",new_raw,1),
        record("new 1x1 supercell C3v VAP",new_c3v,6),
        record("new 1x1 supercell C3v VAP +1LS",new_c3v_ls,6),
        record("new 1x1 supercell C3v+TR VAP",new_c3vtr,6),
        record("new 1x1 supercell C3v+TR VAP +1LS",new_c3vtr_ls,6),
        record("legacy new 1x1 supercell C3 VAP",new_c3,3),
        record("legacy new 1x1 supercell C3 VAP +1LS",new_c3_ls,3),
    ]
    rows=[r for r in rows if r is not None]

    if new_raw is not None and old_raw is not None:
        dE=abs(number(new_raw,"E")-number(old_raw,"E"))
        print(f"Raw implementation energy difference |E_new-E_old| = {dE:.6e}")
    if new_c3v is not None and old_vap is not None:
        tn=time_per_iter(new_c3v); to=time_per_iter(old_vap)
        if math.isfinite(tn) and math.isfinite(to) and tn>0:
            print(f"C3v / old full-VAP time-per-iteration ratio = {tn/to:.4f}")
            print(f"old full-VAP / C3v speedup = {to/tn:.2f}x")

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
            f"dE={fmt(r['delta_E_vs_ED'],'+.4e')} "
            f"F={fmt(r['fidelity_ground_subspace'],'.6f')} "
            f"var={fmt(r['exact_energy_variance'],'.3e')} "
            f"t/it={fmt(r['time_per_iteration_s'],'.3f')}s"
        )
    print("Saved:",Path(a.output).resolve())


if __name__=="__main__":
    main()
