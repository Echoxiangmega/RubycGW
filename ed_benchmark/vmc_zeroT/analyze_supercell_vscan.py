#!/usr/bin/env python3
"""Analyze Raw-only fixed-filling Ruby-lattice V scans.

The script consumes branch_scan.csv files produced by run_supercell_vscan.py
(or an already collected vscan_master.csv) and creates, for every available
supercell size:

  * energy_branches.png
      All seed/continuation branches E(V) on one axis with MC error bars.
  * energy_relative.png
      E_branch(V)-min_branch E(V), useful for locating crossings.
  * phase_diagram.png
      Phase assignment from the lowest-energy optimized branch.  Points where
      the lowest and second-lowest branches differ by less than --tie-sigma
      combined standard deviations are hatched/marked uncertain.
  * error_diagnostics.png
      MC energy error, split-Rhat, autocorrelation time and acceptance.
  * lro_diagnostics.png
      Raw and connected CO/LC correlation diagnostics of the lowest-energy
      branch at each V.
  * ground_state_orders.png
      One-point CO/BOW/LC order amplitudes of the lowest-energy branch.
  * phase_table.csv
      Numerical winner, runner-up, gap, combined error, z-score and phase.

If several LxL sizes are present, finite-size scaling figures are generated
for selected common V values.  Only such multi-size scaling can establish
long-range order.  A single-size LRO plot is diagnostic, not proof.

No pandas/seaborn dependency is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


BRANCH_ORDER=[
    "normal","sd","isd","lc_same","lc_opposite","random",
    "ps_co","mix_co_lc_opposite",
]

BRANCH_LABELS={
    "normal":"normal",
    "sd":"SD seed",
    "isd":"ISD seed",
    "lc_same":"LC same",
    "lc_opposite":"LC opposite",
    "random":"random",
    "ps_co":"pseudospin CO",
    "mix_co_lc_opposite":"CO + LC opposite",
}


def _f(x, default=np.nan):
    try:
        if x is None:
            return float(default)
        s=str(x).strip()
        if s=="" or s.lower() in {"none","nan"}:
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _i(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _s(x, default=""):
    if x is None:
        return default
    s=str(x).strip()
    return s if s else default


def _finite(x):
    return np.isfinite(_f(x))


def _load_rows(root: Path, master: Path | None):
    if master is not None:
        files=[master]
    else:
        candidate=root/"vscan_master.csv"
        if candidate.exists():
            files=[candidate]
        else:
            files=sorted(root.glob("L*x*/seed_*/**/branch_scan.csv"))
    if not files:
        raise FileNotFoundError(
            f"No vscan_master.csv or branch_scan.csv below {root}"
        )

    rows=[]
    for path in files:
        with path.open(newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r=dict(row)
                r["_source_file"]=str(path)
                rows.append(r)
    return rows


def _filter_rows(rows, mode, direction):
    out=[]
    for r in rows:
        rm=_s(r.get("scan_mode"),"continuation")
        rd=_s(r.get("scan_direction"),"up")
        if mode!="any" and rm!=mode:
            continue
        if direction!="any" and rd!=direction:
            continue
        if not np.isfinite(_f(r.get("scan_V",r.get("V")))):
            continue
        if not np.isfinite(_f(r.get("mc_E"))):
            continue
        out.append(r)
    return out


def _deduplicate(rows):
    """Keep the last occurrence of each size/branch/mode/direction/V key."""
    d={}
    for r in rows:
        key=(
            _i(r.get("L1")),
            _i(r.get("L2")),
            _s(r.get("branch_seed",r.get("seed"))),
            _s(r.get("scan_mode"),"continuation"),
            _s(r.get("scan_direction"),"up"),
            round(_f(r.get("scan_V",r.get("V"))),12),
        )
        d[key]=r
    return list(d.values())


def _branch_sort_key(name):
    try:
        return (0,BRANCH_ORDER.index(name))
    except ValueError:
        return (1,name)


def _write_csv(path,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for r in rows for k in r.keys()))
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _phase_rows(size_rows,tie_sigma):
    all_branches={
        _s(r.get("branch_seed",r.get("seed")))
        for r in size_rows
    }
    by_v=defaultdict(list)
    for r in size_rows:
        by_v[round(_f(r.get("scan_V",r.get("V"))),12)].append(r)

    out=[]
    for V in sorted(by_v):
        candidates=sorted(by_v[V],key=lambda r:_f(r.get("mc_E")))
        best=candidates[0]
        second=candidates[1] if len(candidates)>1 else None
        e1=_f(best.get("mc_E"))
        s1=_f(best.get("mc_E_error"))
        if second is None:
            e2=gap=sgap=z=np.nan
            second_seed=""
            second_phase=""
            uncertain=False
        else:
            e2=_f(second.get("mc_E"))
            s2=_f(second.get("mc_E_error"))
            gap=e2-e1
            sgap=math.sqrt(max(0.0,s1*s1+s2*s2))
            z=gap/sgap if sgap>0 else np.inf
            second_seed=_s(second.get("branch_seed",second.get("seed")))
            second_phase=_s(second.get("phase_label"),"unknown")
            uncertain=bool(np.isfinite(z) and z<float(tie_sigma))

        best_seed=_s(best.get("branch_seed",best.get("seed")))
        best_phase=_s(best.get("phase_label"),"unknown")
        present={
            _s(r.get("branch_seed",r.get("seed")))
            for r in candidates
        }
        missing=sorted(all_branches-present,key=_branch_sort_key)
        incomplete=bool(missing)
        uncertain=bool(uncertain or incomplete)
        out.append({
            "V":V,
            "winner_seed":best_seed,
            "winner_phase":best_phase,
            "winner_E":e1,
            "winner_E_error":s1,
            "second_seed":second_seed,
            "second_phase":second_phase,
            "second_E":e2,
            "energy_gap":gap,
            "energy_gap_sigma":sgap,
            "gap_z":z,
            "uncertain":uncertain,
            "incomplete":incomplete,
            "n_candidates":len(candidates),
            "missing_branches":";".join(missing),
            "_winner_row":best,
        })
    return out


def _energy_plot(size_rows,outdir,L1,L2):
    fig,ax=plt.subplots(figsize=(8.5,5.8))
    branches=sorted(
        {_s(r.get("branch_seed",r.get("seed"))) for r in size_rows},
        key=_branch_sort_key,
    )
    for branch in branches:
        rr=sorted(
            [r for r in size_rows
             if _s(r.get("branch_seed",r.get("seed")))==branch],
            key=lambda r:_f(r.get("scan_V",r.get("V"))),
        )
        x=np.asarray([_f(r.get("scan_V",r.get("V"))) for r in rr])
        y=np.asarray([_f(r.get("mc_E")) for r in rr])
        e=np.asarray([_f(r.get("mc_E_error"),0.0) for r in rr])
        ax.errorbar(
            x,y,yerr=e,marker="o",markersize=3.5,
            linewidth=1.2,capsize=2.5,
            label=BRANCH_LABELS.get(branch,branch),
        )
    ax.set_xlabel("V")
    ax.set_ylabel("VMC energy")
    ax.set_title(f"Raw VMC branch energies, {L1}x{L2} ring3 supercell")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout()
    fig.savefig(outdir/"energy_branches.png",dpi=220)
    plt.close(fig)


def _relative_energy_plot(size_rows,outdir,L1,L2):
    by_v=defaultdict(list)
    for r in size_rows:
        by_v[round(_f(r.get("scan_V",r.get("V"))),12)].append(r)
    emin={V:min(_f(r.get("mc_E")) for r in rs) for V,rs in by_v.items()}

    fig,ax=plt.subplots(figsize=(8.5,5.8))
    branches=sorted(
        {_s(r.get("branch_seed",r.get("seed"))) for r in size_rows},
        key=_branch_sort_key,
    )
    for branch in branches:
        rr=sorted(
            [r for r in size_rows
             if _s(r.get("branch_seed",r.get("seed")))==branch],
            key=lambda r:_f(r.get("scan_V",r.get("V"))),
        )
        x=np.asarray([_f(r.get("scan_V",r.get("V"))) for r in rr])
        y=np.asarray([
            _f(r.get("mc_E"))-emin[round(_f(r.get("scan_V",r.get("V"))),12)]
            for r in rr
        ])
        e=np.asarray([_f(r.get("mc_E_error"),0.0) for r in rr])
        ax.errorbar(
            x,y,yerr=e,marker="o",markersize=3.5,
            linewidth=1.2,capsize=2.5,
            label=BRANCH_LABELS.get(branch,branch),
        )
    ax.axhline(0.0,linewidth=1.0)
    ax.set_xlabel("V")
    ax.set_ylabel(r"$E_{branch}-E_{min}$")
    ax.set_title(f"Relative branch energies, {L1}x{L2}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8,ncol=2)
    fig.tight_layout()
    fig.savefig(outdir/"energy_relative.png",dpi=220)
    plt.close(fig)


def _phase_plot(phase_rows,outdir,L1,L2,tie_sigma):
    if not phase_rows:
        return
    V=np.asarray([r["V"] for r in phase_rows],float)
    if len(V)==1:
        bounds=np.asarray([V[0]-0.5,V[0]+0.5])
    else:
        mids=0.5*(V[:-1]+V[1:])
        bounds=np.concatenate((
            [V[0]-0.5*(V[1]-V[0])],
            mids,
            [V[-1]+0.5*(V[-1]-V[-2])],
        ))

    phases=[]
    for r in phase_rows:
        p=r["winner_phase"]
        if p not in phases:
            phases.append(p)
    cmap=plt.get_cmap("tab20")
    phase_color={p:cmap(i%20) for i,p in enumerate(phases)}

    fig,ax=plt.subplots(figsize=(9.0,2.4))
    for i,r in enumerate(phase_rows):
        p=r["winner_phase"]
        uncertain=bool(r["uncertain"])
        rect=ax.axvspan(
            bounds[i],bounds[i+1],
            ymin=0.15,ymax=0.85,
            alpha=0.45 if uncertain else 0.9,
            hatch="///" if uncertain else None,
            facecolor=phase_color[p],
            edgecolor="black" if uncertain else "none",
            linewidth=0.5,
        )
        ax.text(
            r["V"],0.5,
            BRANCH_LABELS.get(r["winner_seed"],r["winner_seed"]),
            ha="center",va="center",fontsize=7,rotation=90,
        )

    handles=[
        Patch(facecolor=phase_color[p],label=p)
        for p in phases
    ]
    handles.append(Patch(
        facecolor="white",edgecolor="black",hatch="///",
        label=f"uncertain (<{tie_sigma:g} sigma gap)"
    ))
    ax.legend(handles=handles,bbox_to_anchor=(1.01,1),loc="upper left",fontsize=8)
    ax.set_xlim(bounds[0],bounds[-1])
    ax.set_ylim(0,1)
    ax.set_yticks([])
    ax.set_xlabel("V")
    ax.set_title(
        f"Lowest-energy Raw phase assignment, {L1}x{L2}\n"
        "text shows the winning continuation seed"
    )
    fig.tight_layout()
    fig.savefig(outdir/"phase_diagram.png",dpi=220)
    plt.close(fig)


def _error_plot(size_rows,outdir,L1,L2):
    branches=sorted(
        {_s(r.get("branch_seed",r.get("seed"))) for r in size_rows},
        key=_branch_sort_key,
    )
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.4),sharex=True)
    panels=[
        ("mc_E_error","Energy error"),
        ("mc_R_hat","split-Rhat"),
        ("mc_tau_corr","Autocorrelation time"),
        ("sampler_acceptance","Sampler acceptance"),
    ]
    for branch in branches:
        rr=sorted(
            [r for r in size_rows
             if _s(r.get("branch_seed",r.get("seed")))==branch],
            key=lambda r:_f(r.get("scan_V",r.get("V"))),
        )
        x=np.asarray([_f(r.get("scan_V",r.get("V"))) for r in rr])
        for ax,(key,label) in zip(axes.ravel(),panels):
            y=np.asarray([_f(r.get(key)) for r in rr])
            mask=np.isfinite(x)&np.isfinite(y)
            if np.any(mask):
                ax.plot(
                    x[mask],y[mask],marker="o",markersize=3,
                    linewidth=1.0,label=BRANCH_LABELS.get(branch,branch),
                )
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("V")
    # Common convergence guides.
    axes[0,1].axhline(1.01,linestyle="--",linewidth=0.9)
    axes[0,1].axhline(1.05,linestyle=":",linewidth=0.9)
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle(f"Monte Carlo error and convergence diagnostics, {L1}x{L2}")
    fig.tight_layout()
    fig.savefig(outdir/"error_diagnostics.png",dpi=220)
    plt.close(fig)


def _winner_metric(phase_rows,key):
    x=[]; y=[]; seeds=[]
    for p in phase_rows:
        r=p["_winner_row"]
        val=_f(r.get(key))
        if np.isfinite(val):
            x.append(p["V"]); y.append(val); seeds.append(p["winner_seed"])
    return np.asarray(x),np.asarray(y),seeds


def _lro_plot(phase_rows,outdir,L1,L2):
    # Raw structure factors/second moments are the relevant order diagnostics
    # for explicitly symmetry-broken Raw VMC states. Connected quantities are
    # shown alongside them as fluctuation diagnostics.
    metrics=[
        ("S_CO_over_Ncell","S_CO_connected_over_Ncell",
         r"$S_{CO}/N_{cell}$",r"$S_{CO}^{conn}/N_{cell}$"),
        ("S_LC_same","S_LC_same_connected",
         r"$S_{LC,same}$",r"$S_{LC,same}^{conn}$"),
        ("S_LC_opposite","S_LC_opposite_connected",
         r"$S_{LC,opp}$",r"$S_{LC,opp}^{conn}$"),
    ]
    fig,axes=plt.subplots(3,1,figsize=(8.5,8.8),sharex=True)
    any_data=False
    for ax,(rawkey,connkey,rawlab,connlab) in zip(axes,metrics):
        xr,yr,_=_winner_metric(phase_rows,rawkey)
        xc,yc,_=_winner_metric(phase_rows,connkey)
        if len(xr):
            ax.plot(xr,yr,marker="o",label=rawlab)
            any_data=True
        if len(xc):
            ax.plot(xc,yc,marker="s",linestyle="--",label=connlab)
            any_data=True
        ax.set_ylabel("Correlation strength")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("V")
    fig.suptitle(
        f"LRO diagnostics of the lowest-energy Raw branch, {L1}x{L2}\n"
        "Single-size nonzero values are diagnostic; long-range order requires finite-size scaling."
    )
    fig.tight_layout()
    if any_data:
        fig.savefig(outdir/"lro_diagnostics.png",dpi=220)
    plt.close(fig)


def _orders_plot(phase_rows,outdir,L1,L2):
    metrics=[
        ("CO_onepoint_rms","CO one-point RMS"),
        ("B_star_mean","B-star"),
        ("J_same_mean","J same"),
        ("J_opposite_mean","J opposite"),
    ]
    fig,ax=plt.subplots(figsize=(8.5,5.8))
    any_data=False
    for key,label in metrics:
        x,y,_=_winner_metric(phase_rows,key)
        if len(x):
            if key.startswith("J_"):
                y=np.abs(y)
                label="|"+label+"|"
            ax.plot(x,y,marker="o",label=label)
            any_data=True
    ax.set_xlabel("V")
    ax.set_ylabel("Order amplitude")
    ax.set_title(f"One-point order parameters of the lowest-energy branch, {L1}x{L2}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if any_data:
        fig.savefig(outdir/"ground_state_orders.png",dpi=220)
    plt.close(fig)


def _phase_table_public(phase_rows):
    out=[]
    for r in phase_rows:
        out.append({
            k:v for k,v in r.items()
            if k!="_winner_row"
        })
    return out


def _common_v_by_size(size_phase):
    if len(size_phase)<2:
        return []
    sets=[]
    for _,rows in sorted(size_phase.items()):
        sets.append({round(float(r["V"]),12) for r in rows})
    common=set.intersection(*sets) if sets else set()
    return sorted(common)


def _choose_fss_v(common,requested,n_default=5):
    if requested:
        out=[]
        for x in requested:
            if common:
                v=min(common,key=lambda y:abs(y-float(x)))
                if v not in out:
                    out.append(v)
        return out
    if not common:
        return []
    if len(common)<=n_default:
        return common
    idx=np.linspace(0,len(common)-1,n_default).round().astype(int)
    return [common[i] for i in sorted(set(idx))]


def _finite_size_plots(size_phase,outroot,fss_v):
    if len(size_phase)<2:
        return
    common=_common_v_by_size(size_phase)
    selected=_choose_fss_v(common,fss_v)
    if not selected:
        return

    metrics=[
        ("S_CO_over_Ncell",r"$S_{CO}/N_{cell}$","finite_size_CO.png"),
        ("S_LC_same",r"$S_{LC,same}$","finite_size_LC_same.png"),
        ("S_LC_opposite",r"$S_{LC,opp}$","finite_size_LC_opposite.png"),
    ]

    by_size_v={}
    for size,phase_rows in size_phase.items():
        by_size_v[size]={
            round(float(p["V"]),12):p for p in phase_rows
        }

    for key,ylabel,filename in metrics:
        fig,ax=plt.subplots(figsize=(7.5,5.5))
        any_data=False
        for V in selected:
            xs=[]; ys=[]
            for (L1,L2),mp in sorted(by_size_v.items()):
                p=mp.get(round(V,12))
                if p is None:
                    continue
                val=_f(p["_winner_row"].get(key))
                if not np.isfinite(val):
                    continue
                ncell=3*L1*L2
                # 1/Ncell -> 0 is the thermodynamic limit for this family.
                xs.append(1.0/ncell)
                ys.append(val)
            if len(xs)>=2:
                order=np.argsort(xs)
                xs=np.asarray(xs)[order]; ys=np.asarray(ys)[order]
                ax.plot(xs,ys,marker="o",label=f"V={V:g}")
                any_data=True
        ax.axvline(0.0,linewidth=0.8)
        ax.set_xlabel(r"$1/N_{cell}$")
        ax.set_ylabel(ylabel)
        ax.set_title(
            "Finite-size LRO scaling of the lowest-energy Raw branch\n"
            "A nonzero intercept as 1/Ncell -> 0 supports stable long-range order."
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8,ncol=2)
        fig.tight_layout()
        if any_data:
            fig.savefig(outroot/filename,dpi=220)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root",required=True,
                   help="V-scan root, e.g. results/vscan_3x3")
    p.add_argument("--master",default=None,
                   help="Optional explicit vscan_master.csv")
    p.add_argument("--output",default=None,
                   help="Analysis output directory; default ROOT/analysis")
    p.add_argument("--mode",default="continuation",
                   choices=["continuation","independent","any"])
    p.add_argument("--direction",default="up",
                   choices=["up","down","any"])
    p.add_argument("--tie-sigma",type=float,default=2.0,
                   help="Below this combined-error z-score, mark the phase assignment uncertain.")
    p.add_argument("--fss-v",nargs="*",type=float,default=None,
                   help="V values for finite-size scaling; nearest common scan values are used.")
    a=p.parse_args()

    root=Path(a.root).expanduser().resolve()
    master=(
        None if a.master is None
        else Path(a.master).expanduser().resolve()
    )
    outroot=(
        Path(a.output).expanduser().resolve()
        if a.output is not None
        else root/"analysis"
    )
    outroot.mkdir(parents=True,exist_ok=True)

    rows=_deduplicate(_filter_rows(_load_rows(root,master),a.mode,a.direction))
    if not rows:
        raise RuntimeError("No usable scan rows after filtering")

    grouped=defaultdict(list)
    for r in rows:
        grouped[(_i(r.get("L1")),_i(r.get("L2")))].append(r)

    size_phase={}
    summary={
        "root":str(root),
        "mode":a.mode,
        "direction":a.direction,
        "tie_sigma":float(a.tie_sigma),
        "sizes":{},
    }

    for (L1,L2),rr in sorted(grouped.items()):
        outdir=outroot/f"L{L1}x{L2}"
        outdir.mkdir(parents=True,exist_ok=True)
        phase=_phase_rows(rr,a.tie_sigma)
        size_phase[(L1,L2)]=phase

        _energy_plot(rr,outdir,L1,L2)
        _relative_energy_plot(rr,outdir,L1,L2)
        _phase_plot(phase,outdir,L1,L2,a.tie_sigma)
        _error_plot(rr,outdir,L1,L2)
        _lro_plot(phase,outdir,L1,L2)
        _orders_plot(phase,outdir,L1,L2)
        _write_csv(outdir/"phase_table.csv",_phase_table_public(phase))

        n_uncertain=sum(bool(x["uncertain"]) for x in phase)
        summary["sizes"][f"{L1}x{L2}"]={
            "n_rows":len(rr),
            "n_V":len(phase),
            "branches":sorted(
                {_s(r.get("branch_seed",r.get("seed"))) for r in rr},
                key=_branch_sort_key,
            ),
            "n_uncertain_phase_points":n_uncertain,
        }
        print(
            f"[analysis] L{L1}x{L2}: rows={len(rr)} V={len(phase)} "
            f"uncertain={n_uncertain} -> {outdir}",
            flush=True,
        )

    _finite_size_plots(size_phase,outroot,a.fss_v)

    (outroot/"analysis_summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[analysis] done -> {outroot}",flush=True)


if __name__=="__main__":
    main()
