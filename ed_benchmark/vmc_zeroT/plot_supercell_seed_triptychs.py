#!/usr/bin/env python3
"""Batch site/bond/current triptychs for final supercell-VMC states by seed.

For an exact 1x1 benchmark, every optimized stage saves psi_native.  This
script finds the final available state for each seed and makes one 1x3 figure:
site density | bond order | bond current.

If a TR-restored final state exists, --include-pre-tr also plots the preceding
Tint+C3v state, where a loop-current broken branch can remain visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from plot_saved_state_maps import (
    load_geometry,
    load_saved_state,
    select_representation,
    site_positions,
    edge_segments,
    compute_site_density,
    compute_edge_observables,
    export_site_csv,
    export_bond_csv,
    plot_site_bond_current_triptych,
)


DEFAULT_SEEDS=["normal","sd","isd","lc_same","lc_opposite"]


def _settings(root: Path):
    p=root/"settings.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _auto_irrep(settings):
    vals=settings.get("point_group_targets",[])
    if vals:
        return str(vals[0].get("label","A1"))
    return "A1"


def _auto_correlator(settings):
    return str(settings.get("correlator","backflow"))


def _stage_dirs(root,seed,correlator,irrep):
    pre=root/f"supercell_tint_c3v_{correlator}__{irrep}__{seed}"
    tr=root/f"supercell_tint_c3vtr_{correlator}__{irrep}__{seed}"
    tint=root/f"supercell_tint_{correlator}__Gamma__{seed}"
    raw=root/f"supercell_raw_{correlator}__{seed}"
    if tr.exists():
        final=tr
    elif pre.exists():
        final=pre
    elif tint.exists():
        final=tint
    elif raw.exists():
        final=raw
    else:
        final=None
    return final,pre if pre.exists() else None,tr if tr.exists() else None


def _plot_one(root,g,state_dir: Path,out: Path,title,current_threshold):
    state_file=state_dir/"final_state.npz"
    if not state_file.exists():
        raise FileNotFoundError(state_file)
    payload=load_saved_state(state_file)
    if payload["kind"]!="vector":
        raise RuntimeError(f"Expected a saved VMC vector: {state_file}")
    rep=select_representation(payload,"vector",None,g)
    states=np.asarray(rep["states"],np.uint64)

    xy,cells=site_positions(g)
    starts,ends,crossed=edge_segments(g,xy,cells)
    density=compute_site_density(g,states,rep)
    rho,bond,current=compute_edge_observables(g,states,rep)

    out.mkdir(parents=True,exist_ok=True)
    export_site_csv(out/"site_density.csv",g,xy,density)
    export_bond_csv(
        out/"bond_observables.csv",g,starts,ends,crossed,rho,bond,current
    )
    plot_site_bond_current_triptych(
        g,xy,starts,ends,crossed,density,bond,current,out,title,
        threshold_fraction=float(current_threshold),
    )
    summary={
        "state_dir":str(state_dir),
        "state_file":str(state_file),
        "mean_density":float(np.mean(density)),
        "density_min":float(np.min(density)),
        "density_max":float(np.max(density)),
        "max_abs_bond_order":float(np.max(np.abs(bond))),
        "max_abs_current":float(np.max(np.abs(current))),
    }
    (out/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    return summary


def plot_seed_triptychs(
    result_root,
    *,
    seeds=None,
    correlator=None,
    irrep=None,
    include_pre_tr=True,
    current_threshold=0.02,
):
    root=Path(result_root).resolve()
    settings=_settings(root)
    correlator=correlator or _auto_correlator(settings)
    irrep=irrep or _auto_irrep(settings)
    seeds=list(seeds or DEFAULT_SEEDS)
    g=load_geometry(root)

    outroot=root/"seed_final_maps"
    manifest={}
    for seed in seeds:
        final,pre,tr=_stage_dirs(root,seed,correlator,irrep)
        if final is None:
            print(f"[triptych] seed={seed}: no saved stage found; skip")
            continue
        try:
            final_summary=_plot_one(
                root,g,final,outroot/seed/"final",
                f"{g.name} | seed={seed} | final={final.name}",
                current_threshold,
            )
        except (FileNotFoundError,RuntimeError) as exc:
            print(f"[triptych] seed={seed}: final plot skipped: {exc}")
            continue

        entry={"final":final_summary}
        if include_pre_tr and tr is not None and pre is not None and pre!=final:
            try:
                entry["pre_tr"]=_plot_one(
                    root,g,pre,outroot/seed/"pre_tr",
                    f"{g.name} | seed={seed} | pre-TR={pre.name}",
                    current_threshold,
                )
            except (FileNotFoundError,RuntimeError) as exc:
                entry["pre_tr_error"]=str(exc)
        manifest[seed]=entry
        print(
            f"[triptych] seed={seed}: "
            f"final max|J|={entry['final']['max_abs_current']:.6g}"
        )

    outroot.mkdir(parents=True,exist_ok=True)
    (outroot/"manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    print(f"[triptych] output={outroot}")
    return manifest


def parser():
    p=argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--result-root",required=True)
    p.add_argument("--seeds",nargs="+",default=DEFAULT_SEEDS)
    p.add_argument("--correlator",default=None)
    p.add_argument("--irrep",default=None)
    p.add_argument(
        "--include-pre-tr",action=argparse.BooleanOptionalAction,default=True
    )
    p.add_argument("--current-threshold",type=float,default=0.02)
    return p


def main():
    a=parser().parse_args()
    plot_seed_triptychs(
        a.result_root,
        seeds=a.seeds,
        correlator=a.correlator,
        irrep=a.irrep,
        include_pre_tr=a.include_pre_tr,
        current_threshold=a.current_threshold,
    )


if __name__=="__main__":
    main()
