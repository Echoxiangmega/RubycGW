#!/usr/bin/env python3
"""Compare ED, finite-source cluster-ED+GW and ANALYTIC static cluster chi.

Examples (PowerShell):
    python research/compare_ed_cluster_chi_analytic.py `
      --V 0.2 --n 2 --T 0.125 --nbath 2 `
      --Q 0.333333333333 0.333333333333 `
      --channels x_even y_even z_same z_opposite `
      --out results/hex18_cluster_chi_analytic

For strong coupling --cluster-seed density-gw may be beneficial. The
three results use EXACTLY the same hex18 PBC geometry. Both embedded
responses expand about one independently converged, ZERO-FIELD,
COMPLEX-BATH multi-impurity fixed point, so neither is compared to an
incompatible zero-field real-bath solution.

Analytic method: differentiate the complete finite-bath fixed-point
map (lattice GW/HF, impurity ED Lehmann G, static Weiss split and the
complex bath fit KKT equation) and solve the matrix-free linear
equation by real GMRES. No ±h ED self-consistency is used by the
analytic method. Its smooth interior-fit assumptions and condition
number are checked; it does not silently output a bad chi.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Make the repository importable when called directly without pip install -e.
if __package__ in (None,""):
    root=str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0,root)

import numpy as np

from rubycgw.cluster_ed_gw import (
    ruby_cluster_interactions
)
from rubycgw.cluster_ed_gw_multicell import (
    MultiCellEDGWOptions,MultiCellRestart,
    restart_from_uniform_cluster,solve_multicell_cluster_ed_gw
)
from rubycgw.cluster_ed_gw_linear_response import StaticClusterEDGWLinearizer
from rubycgw.fierz_channel_gw import solve_channel_gw_same_torus
from rubycgw.grand_canonical_ed import grand_canonical_ed_response
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.momentum_ed import MomentumTorusED
from rubycgw.three_torus_benchmark import (
    build_torus_geometry,cluster_seed_from_density_gw,
    hermitian_source_components,noninteracting_cluster_seed,
    uniform_primitive_cluster_ed_gw
)
from rubycgw.three_torus_interaction_channels import build_torus_interaction_channel

from benchmark_three_tori import _cluster_source_solve


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--V",type=float,required=True)
    p.add_argument("--n","--filling",dest="filling",type=float,required=True,
                   help="spinless particles per six-site primitive cell")
    p.add_argument("--T",type=float,default=.125)
    p.add_argument("--t",type=float,default=1.)
    p.add_argument("--t1",type=float,default=-.5)
    p.add_argument("--t2",type=float,default=.5)
    p.add_argument("--Q",nargs=2,type=float,default=(1/3,1/3))
    p.add_argument("--channels",nargs="+",default=[
        "x_even","y_even","z_same","z_opposite"
    ])
    p.add_argument("--nw",type=int,default=49)
    p.add_argument("--nomega",type=int,default=10)
    p.add_argument("--nbath",type=int,default=6)
    p.add_argument("--bath-fit-nfreq",type=int,default=12)
    p.add_argument("--bath-max-nfev",type=int,default=250)
    p.add_argument("--ed-samples",type=int,default=6)
    p.add_argument("--ed-tau-points",type=int,default=17)
    p.add_argument("--ed-charge-sector-padding",type=int,default=1)
    p.add_argument("--cluster-seed",choices=("free","density-gw"),
                   default="free")
    p.add_argument("--gw-max-iter",type=int,default=100)
    p.add_argument("--gw-tol",type=float,default=1e-7)
    p.add_argument("--cluster-max-iter",type=int,default=60)
    p.add_argument("--cluster-tol",type=float,default=3e-5)
    p.add_argument("--baseline-max-iter",type=int,default=80)
    p.add_argument("--baseline-tol",type=float,default=3e-6)
    p.add_argument("--source-max-iter",type=int,default=100)
    p.add_argument("--source-tol",type=float,default=1e-6)
    p.add_argument("--source-step",type=float,default=1e-3)
    p.add_argument("--source-workers",type=int,default=1)
    p.add_argument("--analytic-tol",type=float,default=1e-5)
    p.add_argument("--analytic-maxiter",type=int,default=12)
    p.add_argument("--analytic-restart",type=int,default=8)
    p.add_argument("--bath-stationarity-tol",type=float,default=2e-5)
    p.add_argument("--strict",action="store_true")
    p.add_argument("--verbose",action="store_true")
    p.add_argument("--out",type=Path,required=True)
    return p.parse_args(argv)


def _tojson(x):
    if isinstance(x,dict):
        return {str(k):_tojson(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [_tojson(v) for v in x]
    if isinstance(x,np.ndarray):
        return _tojson(x.tolist())
    if isinstance(x,(np.integer,)):
        return int(x)
    if isinstance(x,(np.floating,float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x,(complex,np.complexfloating)):
        return dict(real=float(x.real),imag=float(x.imag))
    if isinstance(x,(np.bool_,bool)):
        return bool(x)
    if isinstance(x,(int,str)) or x is None:
        return x
    return str(x)


def _as_multi_restart(x):
    return MultiCellRestart(
        Sigma_H=np.asarray(x.Sigma_H,complex).copy(),
        Sigma_emb=np.asarray(x.Sigma_emb,complex).copy(),
        Sigma_imp=np.asarray(x.Sigma_ED_cluster,complex).copy(),
        G=np.asarray(x.G,complex).copy(),
        mu=float(x.mu),
        bath_energies=np.asarray(x.bath_energies,float).copy(),
        bath_couplings=np.asarray(x.bath_couplings,complex).copy(),
    )


def _finite_source_one_channel(geom,grid,interactions,opts,restart,
                                name,Klist,h,target,workers):
    jobs=[]
    for K in Klist:
        for sign in (1.,-1.):
            Hsrc=np.asarray(geom.h0,complex).copy()
            Hsrc[0,0]-=sign*h*K
            jobs.append((
                Hsrc,geom.vq,interactions,grid,K,sign,h,target,opts,restart
            ))
    begin=time.perf_counter()
    if workers==1:
        results=list(map(_cluster_source_solve,jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results=list(pool.map(_cluster_source_solve,jobs))
    value=0.
    for idx in range(0,len(results),2):
        p,m=results[idx],results[idx+1]
        if not p["ok"] or not m["ok"]:
            raise RuntimeError(
                f"source {name} failed: +h={p.get('error','ok')}, "
                f"-h={m.get('error','ok')}"
            )
        value+=(p["value"]-m["value"])/(2*h)
    return float(value),dict(
        wall_seconds=time.perf_counter()-begin,
        n_source_solves=len(results),
        source_residual_max=max(r["residual"] for r in results),
        source_iterations=sum(r["iterations"] for r in results),
        source_wall_total=sum(r["elapsed"] for r in results),
    )


def main(argv=None):
    args=parse_args(argv)
    if not 0<args.filling<6:
        raise ValueError("require filling between 0 and 6")
    if args.source_step<=0 or args.source_workers<1:
        raise ValueError("require positive source-step and source-workers >=1")
    params=RubyParameters(ti=args.t,t1=args.t1,t2=args.t2,V=args.V)
    geom=build_torus_geometry("hex18",params)
    Q=geom.validate_momentum(args.Q)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=args.nw,
                        nOmega=args.nomega,T=args.T)
    target=float(args.filling*geom.ncell)
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    channels=list(dict.fromkeys(args.channels))
    diagnostic=dict(
        V=float(args.V),filling=float(args.filling),target_particles=target,
        T=float(args.T),Q=Q,channels=channels,nbath=int(args.nbath),
        grid=dict(nw=args.nw,nOmega=args.nomega),
        stage={},errors={},
        background="one converged zero-field *complex-bath* three-cell fixed point",
        approximations=[
            "finite Matsubara and finite complex bath",
            "analytic chi assumes smooth interior bath-fit stationary point",
            "analytic response is static Omega=0",
        ],
    )
    row_by_method={}
    def record(method,ch,status,value=None,elapsed=None,details=None):
        row_by_method[(method,ch)]=dict(
            method=method,channel=ch,status=status,
            value=value,time_seconds=elapsed,
            details=details if details is not None else {}
        )
        print(f"[chi] {method:23s} {ch:15s} {status:12s} "
              f"{value if value is not None else 'N/A'}",flush=True)

    def stage(name,fn):
        start=time.perf_counter()
        try:
            result=fn()
            diagnostic["stage"][name]=dict(
                status="ok",seconds=time.perf_counter()-start
            )
            return result
        except Exception as exc:
            error=f"{type(exc).__name__}: {exc}"
            diagnostic["errors"][name]=error
            diagnostic["stage"][name]=dict(
                status="failed",seconds=time.perf_counter()-start,
                error=error
            )
            print(f"[chi] FAILED {name}: {error}",flush=True)
            if args.strict:
                raise
            return None

    def exact_ed():
        Nround=round(target)
        if abs(target-Nround)<1e-10:
            solver=MomentumTorusED(geom,params,int(Nround))
            return solver.thermal_response(
                args.V,args.T,Q,channels,
                n_trace_vectors=args.ed_samples,
                tau_points=args.ed_tau_points
            )
        return grand_canonical_ed_response(
            geom,params,V=args.V,T=args.T,
            target_particles=target,Q=Q,channels=channels,
            n_trace_vectors=args.ed_samples,
            tau_points=args.ed_tau_points,
            padding=args.ed_charge_sector_padding,
            use_momentum=True
        )
    ed=stage("ED",exact_ed)
    if ed is not None:
        for i,ch in enumerate(channels):
            record("ED",ch,"ok",float(ed.susceptibility_matrix[i,i].real),
                   diagnostic["stage"]["ED"]["seconds"],
                   dict(ensemble=getattr(ed,"ensemble","canonical"),
                        stochastic_samples=args.ed_samples))
        diagnostic["ED_susceptibility_matrix"]=np.asarray(
            ed.susceptibility_matrix
        )

    def seed():
        if args.cluster_seed=="free":
            return noninteracting_cluster_seed(geom,grid,target)
        definition=build_torus_interaction_channel(geom,args.V,"density")
        gw=solve_channel_gw_same_torus(
            geom.h0,definition,grid,
            opts=GWOptions(
                target_filling=target,max_iter=args.gw_max_iter,
                tol=args.gw_tol,mixing=.2,mixing_method="pulay",
                verbose=args.verbose,momentum_backend="fft"
            )
        )
        if not np.isfinite(gw.final_error):
            raise RuntimeError("nonfinite preliminary GW")
        return cluster_seed_from_density_gw(gw,geom,grid,target)
    starting=stage("cluster_initialization",seed)

    def uniform():
        result=uniform_primitive_cluster_ed_gw(
            geom,grid,starting,nbath=args.nbath,
            max_iter=args.cluster_max_iter,tol=args.cluster_tol,
            bath_nfit=args.bath_fit_nfreq,
            bath_max_nfev=args.bath_max_nfev,
            verbose=args.verbose,target_particles=target
        )
        if not result.converged:
            raise RuntimeError(
                f"uniform cluster embedding residual={result.final_error:.3e}"
            )
        return result

    first=stage("uniform_cluster",uniform) if starting is not None else None

    def baseline():
        opts=MultiCellEDGWOptions(
            max_iter=args.baseline_max_iter,tol=args.baseline_tol,
            nbath=args.nbath,bath_fit_nfreq=args.bath_fit_nfreq,
            bath_fit_max_nfev=args.bath_max_nfev,
            complex_bath=True,verbose=args.verbose
        )
        state=solve_multicell_cluster_ed_gw(
            geom.h0,geom.vq,ruby_cluster_interactions(params),
            grid,target_particles=target,opts=opts,
            restart=restart_from_uniform_cluster(first,geom.ncell)
        )
        if not state.converged:
            raise RuntimeError(
                f"zero-field complex bath multicell embedding did not "
                f"converge, residual={state.final_error:.3e}"
            )
        return state

    reference=stage("zero_field_complex_bath",baseline) if first is not None else None
    if reference is not None:
        diagnostic["zero_field"]=dict(
            residual=reference.final_error,
            bath_fit_errors=reference.bath_fit_error,
            impurity_mismatches=reference.impurity_mismatch,
            bath_time=reference.timing_seconds,
        )
        opts=MultiCellEDGWOptions(
            max_iter=args.source_max_iter,tol=args.source_tol,
            nbath=args.nbath,bath_fit_nfreq=args.bath_fit_nfreq,
            bath_fit_max_nfev=args.bath_max_nfev,
            complex_bath=True,verbose=args.verbose
        )
        restart=_as_multi_restart(reference)
        interactions=ruby_cluster_interactions(params)

        for ch in channels:
            components=hermitian_source_components(geom,ch,Q)
            outcome=stage(f"finite_source_{ch}",lambda ch=ch,components=components:
                _finite_source_one_channel(
                    geom,grid,interactions,opts,restart,ch,
                    components,float(args.source_step),target,
                    args.source_workers
                )
            )
            if outcome is None:
                record("cluster_finite_source",ch,"failed")
            else:
                val,details=outcome
                record("cluster_finite_source",ch,"ok",val,
                       details["wall_seconds"],details)

        kernel=stage("analytic_kernel",lambda:StaticClusterEDGWLinearizer(
            geom,params,grid,reference,bath_nfit=args.bath_fit_nfreq,
            bath_stationarity_tol=args.bath_stationarity_tol
        ))
        if kernel is not None:
            for ch in channels:
                components=hermitian_source_components(geom,ch,Q)
                def compute():
                    vals=[]
                    info=[]
                    for K in components:
                        result=kernel.response(
                            K,rtol=args.analytic_tol,
                            restart=args.analytic_restart,
                            maxiter=args.analytic_maxiter
                        )
                        vals.append(result.chi)
                        info.append(result.__dict__)
                    return float(sum(vals)),info
                outcome=stage(f"analytic_{ch}",compute)
                if outcome is None:
                    record("cluster_analytic",ch,"failed")
                else:
                    val,details=outcome
                    record("cluster_analytic",ch,"ok",val,
                           diagnostic["stage"][f"analytic_{ch}"]["seconds"],
                           details)
    else:
        for ch in channels:
            record("cluster_finite_source",ch,"unavailable")
            record("cluster_analytic",ch,"unavailable")

    rows=[]
    for ch in channels:
        edrow=row_by_method.get(("ED",ch))
        ref=edrow["value"] if edrow is not None else None
        for method in ("ED","cluster_finite_source","cluster_analytic"):
            r=row_by_method.get((method,ch))
            if r is None:
                r=dict(method=method,channel=ch,status="unavailable",
                       value=None,time_seconds=None,details={})
            val=r["value"]
            error=(abs(val-ref)/max(abs(ref),1.e-12)
                   if val is not None and ref is not None
                   else None)
            other=None
            if method=="cluster_analytic":
                comparison=row_by_method.get(("cluster_finite_source",ch))
                if comparison and val is not None and comparison["value"] is not None:
                    other=abs(val-comparison["value"])/max(
                        abs(comparison["value"]),1.e-12
                    )
            rows.append(dict(
                method=method,channel=ch,status=r["status"],chi=val,
                ED_reference=ref,relative_error_ED=error,
                relative_error_vs_finite_source=other,
                time_seconds=r.get("time_seconds")
            ))
    with (out/"chi_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    diagnostic["results"]=rows
    with (out/"diagnostics.json").open("w",encoding="utf-8") as f:
        json.dump(_tojson(diagnostic),f,ensure_ascii=False,indent=2,allow_nan=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels=channels
        x=np.arange(len(labels))
        fig,ax=plt.subplots(figsize=(max(7,len(labels)*1.6),4.5))
        methods=("ED","cluster_finite_source","cluster_analytic")
        for j,method in enumerate(methods):
            ys=[
                row_by_method.get((method,ch),{}).get("value",np.nan)
                for ch in labels
            ]
            ys=[np.nan if v is None else v for v in ys]
            ax.bar(x+(j-1)*.25,ys,width=.24,label=method.replace("_"," "))
        ax.set_xticks(x,labels)
        ax.set_ylabel("Static susceptibility")
        ax.set_title(f"Hex18 PBC | V={args.V:g}, n={args.filling:g}, T={args.T:g}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out/"chi_comparison.png",dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[chi] plotting unavailable: {exc}",flush=True)
    print(f"[chi] saved {out/'chi_comparison.csv'}, diagnostics.json",flush=True)
    return diagnostic


if __name__=="__main__":
    main()
