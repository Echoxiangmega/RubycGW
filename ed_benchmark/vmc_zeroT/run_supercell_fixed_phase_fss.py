#!/usr/bin/env python3
"""Fixed-phase finite-size scaling for Raw Ruby-lattice VMC.

Use this after a phase/seed has already been identified at a reference size.
The script deliberately does NOT compare competing seeds.  It follows one
chosen Raw branch (for example lc_same) at one fixed V and filling across
square LxL ring3 supercells, then collects the corresponding order diagnostics.

Typical use
-----------
  python run_supercell_fixed_phase_fss.py \
      --shapes 2x2 2x3 3x3 3x4 4x4 \
      --V 3 \
      --filling 2/9 \
      --seed lc_same \
      --output results/fss_nf2over9_V3_lcsame

Extra arguments after "--" are forwarded to run_supercell_vscan.py, e.g.
  ... --output results/fss -- --eval-samples 16384 --anchor-iter 260

For a Slurm array it is often better to run one shape per task:
  ... --shapes 3x4 --V 3 --filling 2/9 --seed lc_same --output results/fss --run-only

Then run once with --collect-only and all desired --shapes to make the
combined CSV/plots.

Important
---------
This is a conditional finite-size study of a chosen variational branch.  It
does not prove that the same branch is the global minimum at every size.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
VSCAN=HERE/"run_supercell_vscan.py"


def _fraction(value):
    try:
        f=Fraction(str(value))
    except (ValueError,ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid filling {value!r}; use a decimal or fraction such as 2/9"
        ) from exc
    if not (0 < f < 1):
        raise argparse.ArgumentTypeError("filling must satisfy 0 < filling < 1")
    return f


def _f(x,default=np.nan):
    try:
        if x is None:
            return float(default)
        s=str(x).strip()
        if not s or s.lower() in {"none","nan"}:
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _resource_row(L1,L2,filling,eval_samples=16384,train_samples=4096,ref_shape=(3,3)):
    area=int(L1)*int(L2)
    ns=18*area
    nf_exact=filling*ns
    if nf_exact.denominator!=1:
        return {
            "L1":int(L1),"L2":int(L2),"Ns":ns,"Nf":None,
            "representable":False,
        }
    nf=int(nf_exact)

    # Current Raw defaults: packed 18 x Nf complex orbital leaf,
    # 7 bond-aware Jastrow channels (3 bond + 4 nonbond shells),
    # and 4 A/B-resolved backflow channels.
    p=18*nf+11
    dense_qgt=p*p*16.0/2**30
    jac_train=train_samples*p*16.0/2**30
    jac_eval=eval_samples*p*16.0/2**30

    ref_nf=int(filling*(18*int(ref_shape[0])*int(ref_shape[1])))
    r2=(nf/ref_nf)**2
    r3=(nf/ref_nf)**3

    if ns<=288:
        chunk,qchunk=256,128
    elif ns<=450:
        chunk,qchunk=128,64
    elif ns<=648:
        chunk,qchunk=64,64
    else:
        chunk,qchunk=32,32

    sweep=max(16,int(round(2*nf/3)))
    return {
        "L1":int(L1),
        "L2":int(L2),
        "aspect_ratio":max(L1,L2)/min(L1,L2),
        "Ns":ns,
        "Nf":nf,
        "representable":True,
        "n_parameters_estimate":p,
        "dense_qgt_estimate_GiB":dense_qgt,
        "jacobian_4096_estimate_GiB":jac_train,
        "jacobian_16384_estimate_GiB":jac_eval,
        "chunk_size_auto":chunk,
        "qgt_chunk_size_auto":qchunk,
        "sweep_size_auto":sweep,
        "sweep_proposals_per_particle":float(sweep)/float(nf),
        "runtime_relative_Nf2":r2,
        "runtime_relative_Nf3":r3,
    }


def _print_resource_table(shapes,filling):
    rows=[_resource_row(L1,L2,filling) for L1,L2 in shapes]
    print("\n[resource estimate] current Raw backflow ansatz",flush=True)
    print(
        " shape    Ns    Nf   params   sweep  prop/Nf  Jac16384"
        "  chunk/qchunk   time rel. ~Nf^2..Nf^3",
        flush=True,
    )
    for r in rows:
        tag=f"{r['L1']}x{r['L2']}"
        if not r["representable"]:
            print(
                f"{tag:>6s} {r['Ns']:5d}   --   filling not representable",
                flush=True,
            )
            continue
        print(
            f"{tag:>6s} {r['Ns']:5d} {r['Nf']:5d} "
            f"{r['n_parameters_estimate']:7d} "
            f"{r['sweep_size_auto']:6d} "
            f"{r['sweep_proposals_per_particle']:8.3f} "
            f"{r['jacobian_16384_estimate_GiB']:9.4f} "
            f"{r['chunk_size_auto']:4d}/{r['qgt_chunk_size_auto']:<4d} "
            f"{r['runtime_relative_Nf2']:7.2f}.."
            f"{r['runtime_relative_Nf3']:<7.2f}",
            flush=True,
        )
    return rows


def _branch_csv(root,L1,L2,seed):
    return (
        root/f"L{L1}x{L2}"/f"seed_{seed}"/"continuation_up"/"branch_scan.csv"
    )


def _load_branch_row(root,L1,L2,seed,V):
    path=_branch_csv(root,L1,L2,seed)
    if not path.exists():
        return None
    with path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if not rows:
        return None
    candidates=[
        r for r in rows
        if np.isfinite(_f(r.get("scan_V",r.get("V"))))
        and abs(_f(r.get("scan_V",r.get("V")))-V)<1e-9
    ]
    if not candidates:
        return None
    return dict(candidates[-1])


def _collect(root,shapes,seed,V,filling,reuse_root=None):
    out=[]
    for L1,L2 in shapes:
        row=_load_branch_row(root,L1,L2,seed,V)
        source_root=root
        if row is None and reuse_root is not None:
            row=_load_branch_row(reuse_root,L1,L2,seed,V)
            source_root=reuse_root
        if row is None:
            print(f"[collect] L1xL2={L1}x{L2}: no completed branch row",flush=True)
            continue

        area=L1*L2
        ns=int(round(_f(row.get("scan_Ns"),18*area)))
        nf=int(round(_f(row.get("scan_Nf"),float(filling)*ns)))
        ncell=3*area
        ntri=6*area
        E=_f(row.get("mc_E"))
        Eerr=_f(row.get("mc_E_error"))
        js=_f(row.get("J_same_mean"))
        jserr=_f(row.get("J_same_mean_error"))
        ss=_f(row.get("S_LC_same"))
        sserr=_f(row.get("S_LC_same_error"))
        sconn=_f(row.get("S_LC_same_connected"))
        jo=_f(row.get("J_opposite_mean"))
        so=_f(row.get("S_LC_opposite"))
        co=_f(row.get("CO_onepoint_rms"))
        sco=_f(row.get("S_CO_over_Ncell"))

        static_fraction=(
            js*js/ss if np.isfinite(js) and np.isfinite(ss) and ss>0
            else np.nan
        )
        out.append({
            "L1":L1,
            "L2":L2,
            "shape":f"{L1}x{L2}",
            "aspect_ratio":max(L1,L2)/min(L1,L2),
            "Ns":ns,
            "Nf":nf,
            "Ncell":ncell,
            "Ntriangle":ntri,
            "inv_Ncell":1.0/ncell,
            "inv_Ntriangle":1.0/ntri,
            "V":V,
            "filling":nf/ns,
            "seed":seed,
            "mc_E":E,
            "mc_E_error":Eerr,
            "E_per_site":E/ns if np.isfinite(E) else np.nan,
            "E_per_site_error":Eerr/ns if np.isfinite(Eerr) else np.nan,
            "J_same_mean":js,
            "J_same_abs":abs(js) if np.isfinite(js) else np.nan,
            "J_same_mean_error":jserr,
            "S_LC_same":ss,
            "S_LC_same_error":sserr,
            "S_LC_same_connected":sconn,
            "LC_same_static_fraction_J2_over_S":static_fraction,
            "J_opposite_mean":jo,
            "S_LC_opposite":so,
            "CO_onepoint_rms":co,
            "S_CO_over_Ncell":sco,
            "n_parameters":_f(row.get("n_parameters")),
            "time_opt_s":_f(row.get("time_opt_s")),
            "time_per_iteration_s":_f(row.get("time_per_iteration_s")),
            "time_final_eval_s":_f(row.get("time_final_eval_s")),
            "mc_tau_corr":_f(row.get("mc_tau_corr")),
            "mc_R_hat":_f(row.get("mc_R_hat")),
            "sampler_acceptance":_f(row.get("sampler_acceptance")),
            "source":str(_branch_csv(source_root,L1,L2,seed)),
        })

    if not out:
        return []

    keys=list(out[0].keys())
    path=root/"fixed_phase_fss.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(out)
    print(f"[collect] wrote {path}",flush=True)

    _plot_fss(root,out,seed,V)
    return out


def _linear_intercept(x,y):
    mask=np.isfinite(x)&np.isfinite(y)
    if np.count_nonzero(mask)<3:
        return None
    p=np.polyfit(x[mask],y[mask],1)
    return float(p[1]),float(p[0])


def _plot_fss(root,rows,seed,V):
    import matplotlib.pyplot as plt

    rows=sorted(rows,key=lambda r:(r["Ntriangle"],r["L1"],r["L2"]))
    x=np.asarray([r["inv_Ntriangle"] for r in rows],float)
    labels=[r["shape"] for r in rows]

    # 1) LC same order amplitude and sqrt(S) on the same physical scale.
    j=np.asarray([r["J_same_abs"] for r in rows],float)
    je=np.asarray([r["J_same_mean_error"] for r in rows],float)
    s=np.asarray([r["S_LC_same"] for r in rows],float)
    se=np.asarray([r["S_LC_same_error"] for r in rows],float)
    sq=np.sqrt(np.maximum(s,0.0))
    sqe=np.where((s>0)&np.isfinite(se),0.5*se/np.sqrt(s),np.nan)

    fig,ax=plt.subplots(figsize=(7.2,5.4))
    ax.errorbar(x,j,yerr=je,marker="o",capsize=2.5,label=r"$|J_{same}|$")
    ax.errorbar(x,sq,yerr=sqe,marker="s",capsize=2.5,label=r"$\sqrt{S_{LC,same}}$")
    for xx,yy,lab in zip(x,j,labels):
        ax.annotate(lab,(xx,yy),xytext=(4,4),textcoords="offset points",fontsize=8)
    fit=_linear_intercept(x,j)
    if fit is not None:
        intercept,slope=fit
        xx=np.linspace(0,max(x)*1.05,100)
        ax.plot(xx,intercept+slope*xx,linestyle="--",linewidth=1.0,
                label=rf"linear $|J|$ intercept={intercept:.4g}")
    ax.set_xlabel(r"$1/N_{\triangle}$")
    ax.set_ylabel("LC same amplitude")
    ax.set_title(f"Fixed-branch LC same scaling | seed={seed}, V={V:g}")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/"fss_lc_same_amplitude.png",dpi=230)
    plt.close(fig)

    # 2) Full and connected second moments.
    conn=np.asarray([r["S_LC_same_connected"] for r in rows],float)
    fig,ax=plt.subplots(figsize=(7.2,5.4))
    ax.errorbar(x,s,yerr=se,marker="o",capsize=2.5,label=r"$S_{LC,same}$")
    ax.plot(x,conn,marker="s",linestyle="--",label=r"$S^{conn}_{LC,same}$")
    fit=_linear_intercept(x,s)
    if fit is not None:
        intercept,slope=fit
        xx=np.linspace(0,max(x)*1.05,100)
        ax.plot(xx,intercept+slope*xx,linestyle=":",linewidth=1.0,
                label=rf"linear $S$ intercept={intercept:.4g}")
    ax.set_xlabel(r"$1/N_{\triangle}$")
    ax.set_ylabel("LC same second moment")
    ax.set_title(f"Fixed-branch LC same structure factor | seed={seed}, V={V:g}")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/"fss_lc_same_structure.png",dpi=230)
    plt.close(fig)

    # 3) Static fraction: in a broken-symmetry state this should approach one
    # when S is dominated by the one-point order rather than fluctuations.
    frac=np.asarray([r["LC_same_static_fraction_J2_over_S"] for r in rows],float)
    fig,ax=plt.subplots(figsize=(7.2,5.0))
    ax.plot(x,frac,marker="o")
    ax.axhline(1.0,linestyle="--",linewidth=0.9)
    ax.set_xlabel(r"$1/N_{\triangle}$")
    ax.set_ylabel(r"$|J_{same}|^2/S_{LC,same}$")
    ax.set_title(f"Static fraction of LC same order | seed={seed}, V={V:g}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(root/"fss_lc_same_static_fraction.png",dpi=230)
    plt.close(fig)


def parser():
    p=argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--shapes",nargs="+",default=None,
        help="Supercell shapes such as --shapes 2x3 3x3 3x4 4x5",
    )
    p.add_argument(
        "--sizes",nargs="+",type=int,default=None,
        help="Backward-compatible shorthand for square shapes, e.g. 2 3 4",
    )
    p.add_argument("--V",type=float,required=True)
    p.add_argument("--filling",type=_fraction,default=Fraction(1,3))
    p.add_argument("--seed",default="lc_same")
    p.add_argument(
        "--sampler",choices=["fast","netket"],default="fast",
        help="Sampler backend forwarded explicitly to run_supercell_vscan.py.",
    )
    p.add_argument(
        "--sampler-chunk-size",type=int,default=None,
        help="Optional sampler chunk size forwarded to run_supercell_vscan.py.",
    )
    p.add_argument(
        "--iter",type=int,default=230,
        help=(
            "Backward-compatible total Raw optimization iterations. Ignored "
            "when --early-iter/--late-iter/--refine-iter are all supplied."
        ),
    )
    p.add_argument(
        "--early-iter",type=int,default=None,
        help="Number of optimization iterations in the early stage.",
    )
    p.add_argument(
        "--late-iter",type=int,default=None,
        help="Number of optimization iterations in the late stage.",
    )
    p.add_argument(
        "--refine-iter",type=int,default=None,
        help="Number of optimization iterations in the refine stage.",
    )
    p.add_argument("--train-samples-early",type=int,default=1024)
    p.add_argument("--train-samples-late",type=int,default=2048)
    p.add_argument("--train-samples-refine",type=int,default=4096)
    p.add_argument("--eval-samples",type=int,default=8192)
    p.add_argument("--n-chains",type=int,default=32)
    p.add_argument("--n-discard",type=int,default=16)
    p.add_argument(
        "--sweep-size",type=int,default=None,
        help="Explicit Metropolis proposals per sweep; omit to use V-scan auto.",
    )
    p.add_argument("--t",type=float,default=1.0)
    p.add_argument("--t1",type=float,default=0.5)
    p.add_argument("--t2",type=float,default=-0.5)
    p.add_argument("--seed-amplitude",type=float,default=0.08)
    p.add_argument("--output",required=True)
    p.add_argument(
        "--reuse-root",default=None,
        help=(
            "Optional existing V-scan root. If the requested size/V/seed is "
            "already present there, reuse it read-only instead of rerunning it."
        ),
    )
    p.add_argument("--collect-only",action="store_true")
    p.add_argument(
        "--run-only",action="store_true",
        help="Run requested tasks but do not write combined FSS CSV/plots. Recommended for Slurm arrays.",
    )
    p.add_argument("--dry-run",action="store_true")
    p.add_argument(
        "vscan_args",nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to run_supercell_vscan.py after --",
    )
    return p


def main():
    a=parser().parse_args()
    shapes=[]
    if a.shapes:
        for token in a.shapes:
            t=str(token).lower().replace("*","x")
            if "x" not in t:
                raise ValueError(f"Invalid shape {token!r}; use e.g. 3x4")
            u,v=t.split("x",1)
            L1,L2=int(u),int(v)
            if L1<1 or L2<1:
                raise ValueError("all shape dimensions must be positive")
            shapes.append((L1,L2))
    if a.sizes:
        shapes.extend((int(L),int(L)) for L in a.sizes)
    shapes=sorted(set(shapes),key=lambda z:(z[0]*z[1],z[0],z[1]))
    if not shapes:
        raise ValueError("provide --shapes and/or --sizes")

    filling=a.filling
    resource_rows=_print_resource_table(shapes,filling)

    stage_iters=[a.early_iter,a.late_iter,a.refine_iter]
    if any(x is not None for x in stage_iters):
        if not all(x is not None for x in stage_iters):
            raise ValueError(
                "When setting stage-specific iterations, provide all of "
                "--early-iter, --late-iter and --refine-iter."
            )
        early_iter=int(a.early_iter)
        late_iter=int(a.late_iter)
        refine_iter=int(a.refine_iter)
        if min(early_iter,late_iter,refine_iter)<0:
            raise ValueError("stage iteration counts must be nonnegative")
        iter_effective=early_iter+late_iter+refine_iter
        switch1_effective=early_iter
        switch2_effective=early_iter+late_iter
    else:
        iter_effective=int(a.iter)
        # Preserve the historical anchor schedule used by run_supercell_vscan.py.
        switch1_effective=min(80,iter_effective)
        switch2_effective=min(180,iter_effective)
        early_iter=switch1_effective
        late_iter=max(0,switch2_effective-switch1_effective)
        refine_iter=max(0,iter_effective-switch2_effective)

    root=Path(a.output).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    reuse_root=(
        None if a.reuse_root is None
        else Path(a.reuse_root).expanduser().resolve()
    )
    settings={
        "shapes":[[int(x),int(y)] for x,y in shapes],
        "V":float(a.V),
        "filling":str(filling),
        "filling_float":float(filling),
        "seed":a.seed,
        "sampler":a.sampler,
        "sampler_chunk_size":a.sampler_chunk_size,
        "iter":int(iter_effective),
        "early_iter":int(early_iter),
        "late_iter":int(late_iter),
        "refine_iter":int(refine_iter),
        "anchor_switch1":int(switch1_effective),
        "anchor_switch2":int(switch2_effective),
        "train_samples_early":int(a.train_samples_early),
        "train_samples_late":int(a.train_samples_late),
        "train_samples_refine":int(a.train_samples_refine),
        "eval_samples":int(a.eval_samples),
        "n_chains":int(a.n_chains),
        "n_discard":int(a.n_discard),
        "sweep_size":a.sweep_size,
        "t":float(a.t),"t1":float(a.t1),"t2":float(a.t2),
        "seed_amplitude":float(a.seed_amplitude),
        "resource_estimates":resource_rows,
        "conditional_phase_fss":True,
        "note":(
            "This follows one chosen variational branch and does not compare "
            "its energy against other phases at every size."
        ),
    }
    shape_tag="_".join(f"L{x}x{y}" for x,y in shapes)
    (root/f"fixed_phase_fss_settings_{shape_tag}.json").write_text(
        json.dumps(settings,indent=2,ensure_ascii=False),encoding="utf-8"
    )

    extra=list(a.vscan_args)
    if extra and extra[0]=="--":
        extra=extra[1:]

    if not a.collect_only:
        for L1,L2 in shapes:
            rr=_resource_row(L1,L2,filling)
            if not rr["representable"]:
                raise ValueError(
                    f"filling={filling} is not exactly representable for {L1}x{L2}"
                )
            if (
                reuse_root is not None
                and _load_branch_row(reuse_root,L1,L2,a.seed,float(a.V)) is not None
            ):
                print(
                    f"\n[fss run] {L1}x{L2}: reuse read-only result from "
                    f"{reuse_root}",
                    flush=True,
                )
                continue

            cmd=[
                sys.executable,str(VSCAN),
                "--L1",str(L1),"--L2",str(L2),
                "--filling",str(filling),
                "--seed",a.seed,
                "--sampler",a.sampler,
                "--anchor-iter",str(int(iter_effective)),
                "--anchor-switch1",str(int(switch1_effective)),
                "--anchor-switch2",str(int(switch2_effective)),
                "--train-samples-early",str(int(a.train_samples_early)),
                "--train-samples-late",str(int(a.train_samples_late)),
                "--train-samples-refine",str(int(a.train_samples_refine)),
                "--eval-samples",str(int(a.eval_samples)),
                "--n-chains",str(int(a.n_chains)),
                "--n-discard",str(int(a.n_discard)),
                "--mode","continuation",
                "--direction","up",
                "--V-values",str(float(a.V)),
                "--t",str(a.t),"--t1",str(a.t1),"--t2",str(a.t2),
                "--seed-amplitude",str(a.seed_amplitude),
                "--output",str(root),
            ]+extra
            if a.sampler_chunk_size is not None:
                cmd.extend([
                    "--sampler-chunk-size",str(int(a.sampler_chunk_size))
                ])
            if a.sweep_size is not None:
                cmd.extend(["--sweep-size",str(int(a.sweep_size))])
            print("\n[fss run] "+" ".join(cmd),flush=True)
            if not a.dry_run:
                subprocess.run(cmd,check=True)

    if not a.dry_run and not a.run_only:
        _collect(
            root,shapes,a.seed,float(a.V),filling,reuse_root=reuse_root
        )


if __name__=="__main__":
    main()
