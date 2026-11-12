#!/usr/bin/env python3
"""One-command comparison on the SAME finite Ruby torus.

Example
-------
python research/benchmark_three_tori.py --geometry hex18 --V 1.0 --filling 2 \
    --K 0.333333333333 0.333333333333 --Q 0.333333333333 0.333333333333 \
    --T 0.125 --out results/hex18_V1_n2

For a first inexpensive validation use --methods ed gw cgw and --nw 19
--nomega 5 --ed-samples 2 --ed-tau-points 7 --ed-krylov-steps 50.
Use --skip-cluster-chi to avoid repeated source-shifted multi-impurity ED.
The full post-GW density vertex can be costly; --post-mmax 1 is an explicitly
windowed *diagnostic* and not full post-GW.

Finite-T ED G uses truncated-sector thermal Lanczos, and ED chi uses
canonical stochastic trace. ED ground E0 is exact within the fixed-N
Lanczos tolerance, while GW/cluster energies are finite-T estimates and
MUST NOT be interpreted as ground-state energy error without T extrapolation.
cGW has the same G as GW; only its covariant susceptibility differs.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path
from time import perf_counter
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import sys

# Support direct execution: "python research/benchmark_three_tori.py".
# In that mode Python puts "research/" on sys.path, not its parent.  Keep
# the repository root discoverable before importing the sibling rubycgw
# package, including on Windows PowerShell without an editable install.
if __package__ in (None, ""):
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rubycgw.model import RubyParameters
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast
from rubycgw.supercell_gw_split import one_body_density_matrix_tail
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.ed18 import ED18Solver
from rubycgw.momentum_ed import MomentumTorusED
from rubycgw.grand_canonical_ed import (
    grand_canonical_ed_response,selected_charge_sectors
)
from rubycgw.momentum_ed_green import momentum_thermal_lanczos_green
from rubycgw.ed18_thermal import thermal_response as hex_thermal_response
from rubycgw.ed_cluster import RubyEDClusterSolver
from rubycgw.ed_cluster_thermal import thermal_response as rect_thermal_response
from rubycgw.thermal_lanczos_green import thermal_lanczos_green, ThermalLanczosOptions
from rubycgw.three_torus_benchmark import (
    build_torus_geometry, k_projector, project_k_green, green_k_tau,
    hermitian_source_components, channel_static_response,
    max_required_ed_dimension, uniform_primitive_cluster_ed_gw,
    SkewEDSectorAdapter, gw_energy_from_state, embedded_gm_energy,
    cluster_seed_from_density_gw, noninteracting_cluster_seed,
)
from rubycgw.benchmark_report import make_comparison_scorecards
from rubycgw.fierz_channel_gw import solve_channel_gw_same_torus
from rubycgw.fierz_channel_continuation import solve_channel_gw_v_ramp
from rubycgw.three_torus_interaction_channels import (
    build_torus_interaction_channel,
    solve_interaction_channel_response,
    one_shot_post_interaction_channel,
    channel_energy_estimate,
)
from types import SimpleNamespace
from rubycgw.post_gw import run_post_gw
from rubycgw.free_energy import evaluate_gw_free_energy
from rubycgw.cluster_ed_gw_multicell import (
    MultiCellEDGWOptions, solve_multicell_cluster_ed_gw,
    restart_from_uniform_cluster,
)
from rubycgw.cluster_ed_gw import ruby_cluster_interactions


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geometry",required=True,choices=["3x1","2x2","hex18","hex","ring3"])
    p.add_argument("--V",type=float,required=True)
    p.add_argument("--filling",type=float,required=True,
                   help="spinless particles per six-site primitive cell")
    p.add_argument("--K",nargs=2,type=float,metavar=("K1","K2"),default=(0.,0.),
                   help="primitive reciprocal coordinates of single-particle G(k,tau)")
    p.add_argument("--Q",nargs=2,type=float,metavar=("Q1","Q2"),default=None,
                   help="external momentum of pseudospin/charge susceptibility; defaults to K")
    p.add_argument("--t",type=float,default=1.0,help="intra-triangle hopping ti")
    p.add_argument("--t1",type=float,default=-0.5)
    p.add_argument("--t2",type=float,default=0.5)
    p.add_argument("--T",type=float,default=0.125)
    p.add_argument("--nw",type=int,default=49)
    p.add_argument("--nomega",type=int,default=10)
    p.add_argument("--methods",nargs="+",choices=["ed","gw","cgw","postgw","cluster"],
                   default=["ed","gw","cgw","postgw","cluster"])
    p.add_argument("--gw-channels",nargs="+",choices=["density","pseudospin","bj"],
                   default=["density","pseudospin","bj"],
                   help="interaction-screening GW variants: original density, exact A1+E pseudospin, half B plus half J")
    p.add_argument("--channels",nargs="+",
                   default=["x_even","y_even","z_same","z_opposite"])
    p.add_argument("--plot-orbital",type=str,default="trace",
                   help="trace means Tr(G_K)/6; otherwise an orbital index 0..5")
    p.add_argument("--plot-tau-points",type=int,default=101)
    p.add_argument("--gw-max-iter",type=int,default=150)
    p.add_argument("--gw-tol",type=float,default=1e-7)
    p.add_argument("--gw-mixing",type=float,default=.22)
    p.add_argument("--gw-mixing-method",choices=["pulay","linear"],default="pulay")
    p.add_argument("--gw-pulay-history",type=int,default=6)
    p.add_argument("--gw-ramp-steps",type=int,default=0,
                   help="0: direct solve; N>=1: start at V=0 and warm-start N steps to requested V")
    p.add_argument("--gw-ramp-min-step",type=float,default=.02,
                   help="abort ramp if a failed V step must be halved below this value")
    p.add_argument("--gw-ramp-max-stages",type=int,default=100,
                   help="hard limit on all continuation attempts, including retried stages")
    p.add_argument("--skip-failed-gw-channel",action="store_true",
                   help="deprecated compatibility alias: failed GW channels are skipped by default")
    p.add_argument("--strict-method-failures",action="store_true",
                   help="optional fail-fast behavior; default is to record failed/unconverged methods and continue")
    p.add_argument("--vertex-max-iter",type=int,default=120)
    p.add_argument("--vertex-tol",type=float,default=2e-7)
    p.add_argument("--vertex-restart",type=int,default=12)
    p.add_argument("--post-mmax",type=int,default=None,
                   help="only explicitly solve bosonic |m|<=M; 0 keeps GW W at all nonzero frequencies")
    p.add_argument("--post-no-translation",action="store_true",
                   help="disable translationally reduced source solve (reference/diagnostic)")
    p.add_argument("--post-no-recycle",action="store_true",
                   help="disable real-linear Krylov recycling; use legacy GMRES per source")
    p.add_argument("--nbath",type=int,default=6)
    p.add_argument("--cluster-initial-state",choices=["gw","free"],default="gw",
                   help="gw: initialize embedding from preliminary density GW; free: skip GW pre-solve when only cluster is requested")
    p.add_argument("--cluster-max-iter",type=int,default=50)
    p.add_argument("--cluster-tol",type=float,default=3e-5)
    p.add_argument("--cluster-mixing",type=float,default=.35)
    p.add_argument("--cluster-require-converged-gw",action="store_true",
                   help="optional strict gate: do not initialize cluster ED+GW from a nonconverged density-GW iterate")
    p.add_argument("--cluster-bath-fit-nfreq",type=int,default=12)
    p.add_argument("--cluster-bath-max-nfev",type=int,default=200)
    p.add_argument("--source-step",type=float,default=1e-3)
    p.add_argument("--cluster-chi-tol",type=float,default=None,
                   help="optional tighter self-consistency tolerance for finite-field χ; default is --cluster-tol")
    p.add_argument("--cluster-chi-max-iter",type=int,default=None,
                   help="optional extra iterations per ±source solve; default is --cluster-max-iter")
    p.add_argument("--cluster-chi-workers",type=int,default=1,
                   help="number of independent +h/-h source processes (1=serial; 2-4 on CPU with enough RAM)")
    p.add_argument("--cluster-chi-coldstart",action="store_true",
                   help="diagnostic only: recompute a separate initial GW at every source; default reuses uniform converged cluster")
    p.add_argument("--skip-cluster-chi",action="store_true",
                   help="skip expensive finite-source cluster-covariant chi; no false bubble alias")
    p.add_argument("--ed-ensemble",choices=["auto","canonical","grand"],default="auto",
                   help="auto: fixed-N for integer total filling; grand-canonical thermal sector mixture otherwise")
    p.add_argument("--ed-charge-sector-padding",type=int,default=1,
                   help="extra integer N sectors on each side of floor/ceil(<N>) in grand-canonical chi")
    p.add_argument("--ed-symmetry",choices=["momentum","legacy"],default="momentum",
                   help="canonical ED ground and finite-T chi: momentum-block Krylov or legacy full-sector Krylov; ED G still uses thermal Lanczos full-sector")
    p.add_argument("--ed-green-symmetry",choices=["momentum","legacy"],default="legacy",
                   help="momentum: evaluate G(K,iw) with N+/-1 momentum-block resolvents; legacy: full-sector thermal Lanczos")
    p.add_argument("--ed-samples",type=int,default=6)
    p.add_argument("--ed-tau-points",type=int,default=17)
    p.add_argument("--ed-sector-padding",type=int,default=1)
    p.add_argument("--ed-thermal-eigs",type=int,default=5)
    p.add_argument("--ed-krylov-steps",type=int,default=100)
    p.add_argument("--ed-max-dim",type=int,default=3000000,
                   help="abort ED before sector allocations exceed this Hilbert dimension")
    p.add_argument("--ed-eigs",type=int,default=4)
    p.add_argument("--refresh-ed-cache",action="store_true",
                   help="recompute ED E0, chi and G even when a matching ED checkpoint exists")
    p.add_argument("--no-ed-cache",action="store_true",
                   help="disable saving/loading out/ed_reference.npz across failed GW retries")
    p.add_argument("--seed",type=int,default=20261111)
    p.add_argument("--allow-unconverged",action="store_true",
                   help="deprecated: unconverged results are never used as a reference")
    p.add_argument("--verbose",action="store_true")
    p.add_argument("--out",type=Path,default=Path("results/three_torus_benchmark"))
    return p.parse_args(argv)


def onebody_channel_expectation(state, H0, K, grid):
    rho=one_body_density_matrix_tail(state.G,grid,H0,float(state.mu),state.Sigma_H)
    return float(np.einsum("ab,ba->",K,rho[0,0]).real)


def _cluster_source_solve(job):
    """One independent signed source, safe to execute in Windows spawned process.

    Worker arguments contain immutable physical matrices and a zero-field
    restart. The solver copies its restart on entry. No source shares
    the mutable bath state with any other signed field.
    """
    Hsrc,Vq,interactions,grid,K,sign,h,target,opts,restart=job
    t0=perf_counter()
    try:
        # Each Python worker may otherwise start a large multithreaded BLAS
        # eigensolver, leading to severe oversubscription with 2-4 workers.
        try:
            from threadpoolctl import threadpool_limits
            context=threadpool_limits(limits=1)
        except ImportError:
            from contextlib import nullcontext
            context=nullcontext()
        with context:
            solved=solve_multicell_cluster_ed_gw(
                Hsrc,Vq,interactions,grid,target_particles=target,
                opts=opts,restart=restart
            )
            if not solved.converged:
                raise RuntimeError(
                    f"nonconverged finite-source embedding: "
                    f"residual={solved.final_error:.3e}, "
                    f"iterations={solved.iterations}"
                )
            val=onebody_channel_expectation(solved,Hsrc,K,grid)
        if not np.isfinite(val):
            raise ValueError("nonfinite onebody susceptibility expectation")
        return dict(
            ok=True,value=float(val),residual=float(solved.final_error),
            bath_error=float(np.max(solved.bath_fit_error)),
            iterations=int(solved.iterations),
            elapsed=float(perf_counter()-t0),
            substage_seconds=getattr(solved,"timing_seconds",None),
        )
    except Exception as exc:
        return dict(
            ok=False,error=f"{type(exc).__name__}: {exc}",
            value=None,elapsed=float(perf_counter()-t0)
        )


def solve_cluster_source_response(geom, params, grid, channels, Q, args, *,
                                  uniform=None):
    """Full finite-source multi-impurity derivative, optional source parallelism.

    The same physical source field, bath fitting, and self-consistency map
    are used in serial and parallel paths. The only new approximation
    control is the pre-existing finite source step h. Source failures are
    kept separate; a bad sign/component never contributes to a fake chi.
    """
    out={}; times={}; diagnostics={}
    interactions=ruby_cluster_interactions(params)
    opts=MultiCellEDGWOptions(
        max_iter=int(args.cluster_chi_max_iter if args.cluster_chi_max_iter is not None
                     else args.cluster_max_iter),
        tol=float(args.cluster_chi_tol if args.cluster_chi_tol is not None
                  else args.cluster_tol),
        mixing=float(args.cluster_mixing),nbath=int(args.nbath),
        bath_fit_nfreq=int(args.cluster_bath_fit_nfreq),
        bath_fit_max_nfev=int(args.cluster_bath_max_nfev),
        complex_bath=True,verbose=bool(args.verbose)
    )
    h=abs(float(args.source_step))
    if h==0:
        raise ValueError("--source-step must be nonzero")
    workers=int(args.cluster_chi_workers)
    if workers < 1:
        raise ValueError("--cluster-chi-workers must be >=1")
    use_warm=not bool(args.cluster_chi_coldstart)
    restart=restart_from_uniform_cluster(uniform,geom.ncell) if use_warm else None
    spec=[]
    jobs=[]
    target=float(args.filling*geom.ncell)
    for ch in channels:
        components=hermitian_source_components(geom,ch,Q)
        for component_idx,K in enumerate(components):
            for sign in (1.,-1.):
                Hsrc=np.asarray(geom.h0,complex).copy()
                Hsrc[0,0]-=sign*h*K
                jobs.append((
                    Hsrc,np.asarray(geom.vq,complex),interactions,grid,
                    K,sign,h,target,opts,restart
                ))
                spec.append((ch,component_idx,sign))
    t_begin=perf_counter()
    if workers==1:
        results=list(map(_cluster_source_solve,jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results=list(pool.map(_cluster_source_solve,jobs,chunksize=1))
    wall=float(perf_counter()-t_begin)

    grouped={ch:{} for ch in channels}
    for key,value in zip(spec,results):
        ch,component_idx,sign=key
        grouped[ch][(component_idx,sign)]=value

    for ch in channels:
        sources=grouped[ch]
        parts=[]
        expected=hermitian_source_components(geom,ch,Q)
        problems=[]
        maxres=0.
        maxbath=0.
        work=0.
        phases={}
        for comp_idx in range(len(expected)):
            plus=sources[(comp_idx,1.)]
            minus=sources[(comp_idx,-1.)]
            for item in (plus,minus):
                work+=float(item["elapsed"])
                if item["ok"]:
                    maxres=max(maxres,float(item["residual"]))
                    maxbath=max(maxbath,float(item["bath_error"]))
                    for k,v in (item.get("substage_seconds") or {}).items():
                        phases[k]=phases.get(k,0.)+float(v)
                else:
                    problems.append(item["error"])
            if plus["ok"] and minus["ok"]:
                parts.append((plus["value"]-minus["value"])/(2*h))
        if problems or len(parts)!=len(expected):
            out[ch]=None
            diagnostics[ch]=dict(
                status="failed",error="; ".join(problems),
                residual=maxres,bath_error=maxbath,
                source_step=h,n_source_solves=2*len(expected),
                work_seconds=work,wall_total_seconds=wall,
                workers=workers,phase_seconds=phases,
                initial_state="uniform-cluster-restart" if use_warm else "cold-GW"
            )
            if args.strict_method_failures:
                raise RuntimeError(f"cluster-chi {ch}: {problems}")
        else:
            chi=float(sum(parts))
            if not np.isfinite(chi):
                raise ValueError(f"nonfinite susceptibility: {ch}")
            out[ch]=chi
            diagnostics[ch]=dict(
                status="converged",residual=maxres,bath_error=maxbath,
                source_step=h,n_source_solves=2*len(expected),
                work_seconds=work,wall_total_seconds=wall,
                workers=workers,phase_seconds=phases,
                residual_over_h=maxres/h,
                numerical_resolution_warning=(
                    "cluster source residual/h exceeds 0.01; confirm finite-h "
                    "and outer-iteration convergence before interpreting χ"
                    if maxres/h>0.01 else None
                ),
                initial_state="uniform-cluster-restart" if use_warm else "cold-GW"
            )
        # Aggregate independent source compute time, not the wall-clock of
        # concurrent tasks. The overall cluster_chi stage timer is the real
        # elapsed wall time and can be compared to other stages.
        times[ch]=work
        if diagnostics[ch].get("numerical_resolution_warning"):
            print(f"[cluster-chi] WARNING {ch}: {diagnostics[ch]['numerical_resolution_warning']}",flush=True)
        print(
            f"[cluster-chi] {ch}: {diagnostics[ch]['status']}; "
            f"source-work={work:.2f}s, workers={workers}",
            flush=True
        )
    diagnostics["_execution"]=dict(
        workers=workers,source_count=len(jobs),wall_seconds=wall,
        total_source_work_seconds=sum(times.values()),
        warmstart_used=use_warm
    )
    return out,times,diagnostics


def output_rows(rows, out):
    columns=["method","quantity","value","ED_reference","relative_error",
             "runtime_seconds","stage_runtime_seconds","status","notes"]
    with (out/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=columns)
        wr.writeheader()
        for row in rows:
            wr.writerow({key:row.get(key,"") for key in columns})


def json_safe(x):
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        val=float(x)
        return val if np.isfinite(val) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k):json_safe(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)):
        return [json_safe(y) for y in x]
    if isinstance(x, (int,str,bool)) or x is None:
        return x
    return str(x)


def save_plot(out, tau, curves, K, V, filling, beta):
    palette={"density":"#1565c0","pseudospin":"#7b1fa2","bj":"#00897b"}
    fig,ax=plt.subplots(figsize=(7.0,5.0),dpi=130)
    for label,data in curves.items():
        if label=="ED":
            color,ls,lw="red","-",2.25
        elif label=="cluster-ED+GW":
            color,ls,lw="black","-.",1.9
        else:
            mode=label.split("[")[-1].rstrip("]") if "[" in label else "density"
            color=palette.get(mode,"gray")
            ls="--" if label.startswith("post-GW") else ":"
            lw=1.8
        ax.plot(tau/beta,data,label=label,color=color,
                linestyle=ls,linewidth=lw)
    ax.set_xlabel(r"$\tau/\beta$")
    ax.set_ylabel(r"$G(K,\tau)$")
    ax.set_xlim(0,1)
    ax.set_title(f"Ruby {V=:.3g}, filling={filling:g}, K=({K[0]:g},{K[1]:g})")
    ax.legend(loc="best",frameon=True,fontsize=8,ncol=2)
    ax.grid(alpha=.16)
    fig.tight_layout()
    fig.savefig(out/"green_tau.png",dpi=200)
    fig.savefig(out/"green_tau.pdf")
    plt.close(fig)


def main(a=None, *, initial_gw=None, initial_vertices=None,
         return_scan_state=False):
    """Run one V point, optionally reusing certified states from a V scan.

    initial_gw maps mode -> (prior_V, converged ChannelGWResult).
    initial_vertices maps mode -> (prior_V, {(channel,component):Gamma}).
    No numerical state from a nonconverged GW/cGW may be propagated.
    """
    a=parse_args() if a is None else a
    initial_gw={} if initial_gw is None else dict(initial_gw)
    initial_vertices={} if initial_vertices is None else dict(initial_vertices)
    next_gw={}
    next_vertices={}
    if a.T<=0:
        raise ValueError("require T>0")
    if a.nw<3 or a.nomega<1:
        raise ValueError("require nw>=3 and nomega>=1")
    if a.ed_tau_points<3 or a.ed_tau_points%2!=1:
        raise ValueError("ED Simpson tau_points must be odd and >=3")
    if a.plot_tau_points<3:
        raise ValueError("plot needs at least three tau points")
    if a.filling<=0 or a.filling>=6:
        raise ValueError("spinless filling must lie strictly between 0 and 6")
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    channels=tuple(str(x) for x in a.channels)
    params=RubyParameters(ti=a.t,t1=a.t1,t2=a.t2,V=a.V)
    geom=build_torus_geometry(a.geometry,params)
    K=geom.validate_momentum(a.K)
    Q=geom.validate_momentum(a.Q if a.Q is not None else K)
    total=float(a.filling*geom.ncell)
    N_nearest=int(round(total))
    integer_total=bool(abs(total-N_nearest)<1.e-10)
    ensemble=(
        "canonical" if integer_total else "grand"
        if a.ed_ensemble=="auto" else a.ed_ensemble
    )
    if a.ed_ensemble!="auto":
        ensemble=a.ed_ensemble
    if "ed" in a.methods and ensemble=="canonical" and not integer_total:
        raise ValueError(
            f"canonical ED needs integer Ncell*filling; got {total}. "
            "Use --ed-ensemble grand (or auto)."
        )
    N=N_nearest if integer_total and ensemble=="canonical" else None
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=a.nw,nOmega=a.nomega,T=a.T)
    tau=np.linspace(0,1/a.T,int(a.plot_tau_points))
    U=k_projector(geom,K)
    hK=U.conj().T@geom.h0[0,0]@U
    methods=set(a.methods)
    # A cluster embedding can start directly from a noninteracting Dyson G:
    # no preliminary GW call is needed unless a standalone GW/cGW/post
    # method was explicitly requested or the user selects the GW seed.
    if ("cgw" in methods or "postgw" in methods or
            ("cluster" in methods and a.cluster_initial_state=="gw")):
        methods.add("gw")
    if "ed" in methods:
        ed_sectors=(
            [int(N)] if ensemble=="canonical"
            else selected_charge_sectors(
                geom.nsites,total,int(a.ed_charge_sector_padding)
            )
        )
        dmax=max(
            max_required_ed_dimension(geom.nsites,int(n),int(a.ed_sector_padding))
            for n in ed_sectors
        )
        if dmax>int(a.ed_max_dim):
            raise RuntimeError(
                f"ED sector dimension {dmax:,} exceeds "
                f"--ed-max-dim={a.ed_max_dim:,}; lower filling or increase "
                "the explicit cap only with adequate RAM"
            )
    info=dict(geometry=geom.name,V=a.V,filling=a.filling,N=N,
              target_mean_particles=total,ed_ensemble=ensemble,
              ed_charge_sector_padding=int(a.ed_charge_sector_padding),
              sites=geom.nsites,
              K=K,Q=Q,T=a.T,beta=1/a.T,
              hoppings=dict(t=a.t,t1=a.t1,t2=a.t2),
              channels=channels,allowed_K=geom.allowed_momenta(),
              gw_interaction_channels=list(a.gw_channels),
              ed_symmetry=(
                  "many-body translation blocks for ED ground state and canonical finite-T chi; "
                  "ED Green still uses unblocked thermal Lanczos"
                  if a.ed_symmetry=="momentum" else
                  "legacy full fixed-N solver"
              ),
              ED_green=("momentum-sector N+/-1 low-T thermal Lanczos" if a.ed_green_symmetry=="momentum" else
                        "full fixed-N sector low-T thermal Lanczos"),
              ed_green_symmetry=a.ed_green_symmetry,
              ED_chi=("canonical fixed-N stochastic thermal typicality" if ensemble=="canonical"
                      else "grand-canonical selected N-sector stochastic trace at fixed mean N"),
              GW_ensemble="grand canonical, fixed mean N",
              chi="connected static source response at same physical Q",
              post_scope="one-shot Dyson; only G compared",
              cgw_green="identical to ordinary self-consistent GW",
              energy_scope="ED exact E0; GW finite-T Galitskii-Migdal estimate",
              timings={},diagnostics={},errors={})
    # Keep completed ED work when a later strong-coupling GW stage fails.
    # The signature excludes GW mixing/ramp settings so numerical retries
    # reuse exactly the same ED reference, but includes all ED controls.
    cache_signature=json.dumps({
        "geometry":geom.name,"V":float(a.V),"filling":float(a.filling),
        "N":N,"ensemble":ensemble,"ed_charge_sector_padding":a.ed_charge_sector_padding,
        "K":list(map(float,K)),"Q":list(map(float,Q)),
        "T":float(a.T),"nw":int(a.nw),"nomega":int(a.nomega),
        "t":float(a.t),"t1":float(a.t1),"t2":float(a.t2),
        "channels":list(channels),"ed_symmetry":a.ed_symmetry,
        "ed_green_symmetry":a.ed_green_symmetry,
        "ed_samples":a.ed_samples,"ed_tau_points":a.ed_tau_points,
        "ed_eigs":a.ed_eigs,"ed_sector_padding":a.ed_sector_padding,
        "ed_thermal_eigs":a.ed_thermal_eigs,
        "ed_krylov_steps":a.ed_krylov_steps,"ed_seed":a.seed,
    },sort_keys=True)
    ed_cache_path=out/"ed_reference.npz"
    ed_cache_data=None
    if ("ed" in methods and not a.no_ed_cache and not a.refresh_ed_cache
            and ed_cache_path.is_file()):
        try:
            with np.load(ed_cache_path,allow_pickle=False) as saved:
                if str(saved["signature"].item()) == cache_signature:
                    ed_cache_data={
                        key: np.asarray(saved[key]).copy() for key in saved.files
                    }
                    print(f"[benchmark] matching ED cache: {ed_cache_path}",flush=True)
                else:
                    print("[benchmark] ED cache signature differs; recomputing ED",flush=True)
        except (OSError,KeyError,ValueError) as exc:
            print(f"[benchmark] invalid ED cache ({exc}); recomputing ED",flush=True)
    rows=[]; curves={}; arrays={}; green_by_method={}; chi_by_method={}
    times={}; energy={}; ed_scalar=None; ed_full=None; ed_chi={}
    method_mu={}; total_times={}; channel_states={}
    method_status={}
    info["warm_start"]=dict(
        enabled=bool(initial_gw or initial_vertices),
        mode="across_user_V_values" if initial_gw or initial_vertices else "cold",
        previous_GW_V={key:float(val[0]) for key,val in initial_gw.items()},
        previous_cGW_V={key:float(val[0]) for key,val in initial_vertices.items()},
    )
    def mark_method(method, status, reason="", *, dependent=()):
        """Persist explicit method status and a readable CSV row."""
        method_status[method]=dict(status=str(status),reason=str(reason))
        row(method,"method_status",None,status=status,notes=str(reason))
        for name in dependent:
            method_status[name]=dict(status="skipped",reason=f"{method} {status}: {reason}")
            row(name,"method_status",None,status="skipped",
                notes=f"depends on {method}: {reason}")

    def row(method, quantity, val, *, ref=None, runtime=None, status="ok",notes=""):
        if runtime is None: runtime=times.get(method,None)
        err=("" if ref is None or val is None or not np.isfinite(float(ref))
             else float(abs(float(val)-float(ref))/max(abs(float(ref)),1e-12)))
        rows.append(dict(method=method,quantity=quantity,
                         value="" if val is None else float(val),
                         ED_reference="" if ref is None else float(ref),
                         relative_error=err,
                         runtime_seconds="" if runtime is None else float(runtime),
                         status=status,notes=notes))
    def run(label,fn):
        start=perf_counter()
        print(f"[benchmark] {label} ...",flush=True)
        try:
            result=fn()
            times[label]=float(perf_counter()-start)
            print(f"[benchmark] {label} complete in {times[label]:.2f}s",flush=True)
            return result
        except Exception as exc:
            times[label]=float(perf_counter()-start)
            info["errors"][label]=f"{type(exc).__name__}: {exc}"
            print(f"[benchmark] {label} FAILED: {exc}",flush=True)
            nonconv=("converge" in str(exc).lower() or
                     "residual" in str(exc).lower() or
                     "singular" in str(exc).lower())
            mark_method(label,"unconverged" if nonconv else "failed",
                        f"{type(exc).__name__}: {exc}")
            if a.strict_method_failures:
                raise
            return None
    if "ed" in methods and ed_cache_data is None:
        def stage_ed():
            if ensemble=="grand":
                grchi=grand_canonical_ed_response(
                    geom,params,V=a.V,T=a.T,target_particles=total,
                    Q=Q,channels=channels,n_trace_vectors=a.ed_samples,
                    tau_points=a.ed_tau_points,seed=a.seed,
                    padding=a.ed_charge_sector_padding,
                    ed_eigs=a.ed_eigs,
                    use_momentum=(a.ed_symmetry=="momentum")
                )
                from types import SimpleNamespace
                sol=SimpleNamespace(
                    dimension=sum(grchi.dimensions.values())
                )
                spec=SimpleNamespace(
                    energies=np.asarray([np.nan],float),
                    ground_multiplicity=0,gap_above_manifold=float("nan"),
                    sector_dimensions=list(grchi.dimensions.values())
                )
                return sol,spec,grchi
            if a.ed_symmetry=="momentum":
                ed=MomentumTorusED(geom,params,n_particles=N)
                spec=ed.ground_spectrum(a.V,n_eigs=a.ed_eigs)
                chi=ed.thermal_response(
                    a.V,a.T,Q,channels,
                    n_trace_vectors=a.ed_samples,
                    tau_points=a.ed_tau_points,
                    seed=a.seed,
                    energy_shift=float(spec.energies[0])
                )
                return ed,spec,chi
            if geom.name=="hex18":
                ed=ED18Solver(params,n_particles=N)
                spec=ed.solve(a.V,n_eigs=a.ed_eigs)
                chi=hex_thermal_response(
                    ed,a.V,a.T,Q,channels=channels,
                    n_trace_vectors=a.ed_samples,
                    seed=a.seed,tau_points=a.ed_tau_points,
                    energy_shift=float(spec.energies[0])
                )
            else:
                lx,ly=(3,1) if geom.name=="3x1" else (2,2)
                ed=RubyEDClusterSolver(lx,ly,params,n_particles=N)
                spec=ed.solve(a.V,n_eigs=a.ed_eigs)
                chi=rect_thermal_response(
                    ed,a.V,a.T,Q,channels=channels,
                    n_trace_vectors=a.ed_samples,
                    seed=a.seed,tau_points=a.ed_tau_points,
                    energy_shift=float(spec.energies[0])
                )
            return ed,spec,chi
        result=run("ED",stage_ed)
        if result is None:
            for ch in channels:
                row("ED",f"chi[{ch}](Q)",None,status="failed",
                    notes="canonical ED reference did not complete")
            row("ED","ground_state_E0",None,status="failed",
                notes="ED ground/thermal stage did not complete")
            mark_method("ED_Green","skipped","ED reference stage failed")
        if result is not None:
            ed,sp,chi=result
            if ensemble=="canonical":
                energy["ED"]=float(sp.energies[0])
            info["diagnostics"]["ED"]=dict(
                dimension=int(ed.dimension),ensemble=ensemble,
                target_mean_particles=float(total),
                grand_mu=float(chi.mu) if ensemble=="grand" else None,
                grand_charge_sectors=chi.sectors if ensemble=="grand" else None,
                grand_charge_probabilities=chi.sector_probabilities if ensemble=="grand" else None,
                grand_edge_probability=float(chi.edge_probability) if ensemble=="grand" else None,
                ground_multiplicity=int(sp.ground_multiplicity),
                gap=float(sp.gap_above_manifold),samples=a.ed_samples,
                tau_points=a.ed_tau_points,
                symmetry=a.ed_symmetry,
                sector_dimensions=(list(map(int,sp.sector_dimensions))
                                   if hasattr(sp,"sector_dimensions") else None)
            )
            if ensemble=="canonical":
                row("ED","ground_state_E0",energy["ED"],runtime=times["ED"],
                    notes="exact fixed-N ground state up to Lanczos tolerance")
            else:
                row("ED","ground_state_E0",None,status="not-defined",
                    notes="no single canonical ground-state E0 at fractional mean filling")
                row("ED","grand_canonical_mu",float(chi.mu),
                    notes="chemical potential estimated from truncated ED sector traces")
                row("ED","mean_particles",float(chi.average_particles),
                    notes="charge sector mixture at fixed mean particle number")
            for i,ch in enumerate(channels):
                ed_chi[ch]=float(np.real(chi.susceptibility_matrix[i,i]))
                row("ED",f"chi[{ch}](Q)",ed_chi[ch],runtime=times["ED"],
                    notes=("canonical stochastic thermal trace" if ensemble=="canonical"
                           else "grand-canonical selected charge-sector trace; monitor edge probability"))
            arrays["ED_chi_matrix"]=np.asarray(chi.susceptibility_matrix)
            arrays["ED_connected_tau"]=np.asarray(chi.correlation_tau_matrix)
            def stage_ed_green():
                opts=ThermalLanczosOptions(
                    sector_padding=a.ed_sector_padding,
                    thermal_eigs=a.ed_thermal_eigs,
                    krylov_dim=a.ed_krylov_steps,
                    verbose=a.verbose
                )
                if a.ed_green_symmetry=="momentum":
                    return momentum_thermal_lanczos_green(
                        geom,params,V=a.V,T=a.T,target_particles=total,
                        omega=grid.omega,K=K,opts=opts
                    )
                if geom.name=="hex18":
                    return thermal_lanczos_green(
                        3,1,params,V=a.V,T=a.T,target_particles=total,
                        omega=grid.omega,opts=opts,
                        sector_solver_factory=lambda n:SkewEDSectorAdapter(params,n)
                    )
                lx,ly=(3,1) if geom.name=="3x1" else (2,2)
                return thermal_lanczos_green(
                    lx,ly,params,V=a.V,T=a.T,target_particles=total,
                    omega=grid.omega,opts=opts
                )
            gr=run("ED_Green",stage_ed_green)
            if gr is None:
                row("ED","G_rel_Frobenius_K",None,status="failed",
                    notes="ED Green reference did not complete")
            if gr is not None:
                info["diagnostics"]["ED_Green"]=dict(
                    mu=gr.mu,thermal_kept_weight=gr.kept_thermal_weight,
                    sector_edge_weight=float(gr.sector_probabilities[0]+gr.sector_probabilities[-1]),
                    translation_residual=getattr(gr,"translation_residual",None),
                    symmetry=a.ed_green_symmetry,
                    momentum_sector_dimensions=getattr(gr,"sector_dimensions_by_N",None),
                    sector_numbers=gr.sector_numbers,
                    sector_probabilities=gr.sector_probabilities,
                    krylov_steps_max=gr.krylov_steps_max)
                if a.ed_green_symmetry=="momentum":
                    ed_scalar=np.asarray(gr.G,dtype=complex)
                    if ed_scalar.shape!=(grid.nf,6,6):
                        raise RuntimeError("momentum ED Green must be nf x 6 x 6")
                    ed_full=None
                else:
                    ed_full=np.asarray(gr.G)
                    ed_scalar=project_k_green(ed_full,geom,K)
                    arrays["ED_Giw_full"]=ed_full
                green_by_method["ED"]=ed_scalar
                energy["ED_mu"]=float(gr.mu)
                row("ED","G_reference_initialized",0.,runtime=times["ED_Green"],
                    notes="finite-T sector-truncated thermal Lanczos ED")
                if not a.no_ed_cache:
                    # Atomic write: incomplete GW runs must not corrupt ED cache.
                    temp=ed_cache_path.with_name(ed_cache_path.name+".tmp")
                    with open(temp,"wb") as handle:
                        np.savez_compressed(
                            handle,
                            signature=np.asarray(cache_signature),
                            e0=np.asarray(energy.get("ED",float("nan"))),
                            mu=np.asarray(energy["ED_mu"]),
                            chi_matrix=np.asarray(arrays["ED_chi_matrix"]),
                            corr_tau=np.asarray(arrays["ED_connected_tau"]),
                            giw_K=np.asarray(ed_scalar),
                            diagnostics_ed=np.asarray(json.dumps(
                                json_safe(info["diagnostics"]["ED"]))),
                            diagnostics_ed_green=np.asarray(json.dumps(
                                json_safe(info["diagnostics"]["ED_Green"]))),
                            time_ed=np.asarray(times["ED"]),
                            time_ed_green=np.asarray(times["ED_Green"]),
                        )
                    temp.replace(ed_cache_path)
                    print(f"[benchmark] saved ED reference: {ed_cache_path}",flush=True)
    if "ed" in methods and ed_cache_data is not None:
        cached_e0=float(ed_cache_data["e0"].item())
        if np.isfinite(cached_e0):
            energy["ED"]=cached_e0
        energy["ED_mu"]=float(ed_cache_data["mu"].item())
        ed_scalar=np.asarray(ed_cache_data["giw_K"],dtype=complex)
        if ed_scalar.shape!=(grid.nf,6,6):
            raise ValueError("ED cache Green shape is inconsistent with Matsubara grid")
        ed_full=None
        times["ED"]=0.0
        times["ED_Green"]=0.0
        info["diagnostics"]["ED"]=json.loads(
            str(ed_cache_data["diagnostics_ed"].item())
        )
        info["diagnostics"]["ED_Green"]=json.loads(
            str(ed_cache_data["diagnostics_ed_green"].item())
        )
        info["diagnostics"]["ED"]["cache_hit"]=True
        info["diagnostics"]["ED"]["original_runtime_seconds"]=float(
            ed_cache_data["time_ed"].item()
        )
        info["diagnostics"]["ED_Green"]["original_runtime_seconds"]=float(
            ed_cache_data["time_ed_green"].item()
        )
        arrays["ED_chi_matrix"]=ed_cache_data["chi_matrix"]
        arrays["ED_connected_tau"]=ed_cache_data["corr_tau"]
        green_by_method["ED"]=ed_scalar
        for i,ch in enumerate(channels):
            ed_chi[ch]=float(np.real(arrays["ED_chi_matrix"][i,i]))
        row("ED","ground_state_E0",energy.get("ED"),status="cached" if ensemble=="canonical" else "not-defined",
            notes=("fixed-N ED ground state loaded from checkpoint" if ensemble=="canonical"
                   else "not defined at fractional average filling; ED chi is grand canonical"))
        for ch in channels:
            row("ED",f"chi[{ch}](Q)",ed_chi[ch],status="cached",
                notes="canonical ED thermal susceptibility loaded from matching checkpoint")
        row("ED","G_reference_initialized",0.,status="cached",
            notes="finite-T thermal Lanczos ED loaded from matching checkpoint")
    opts_gw=GWOptions(
        target_filling=total,max_iter=a.gw_max_iter,tol=a.gw_tol,
        mixing=a.gw_mixing,mixing_method=a.gw_mixing_method,
        pulay_history=a.gw_pulay_history,
        verbose=a.verbose,momentum_backend="fft",
        mu_tol=1e-10,mu_max_iter=100
    )
    # Each different bosonic/Fierz interaction channel requires its OWN
    # self-consistent GW G; within each channel, later stages reuse that G.
    # The original density-channel GW is also the background for our
    # primitive V-only cluster embedding (not BJ or E-only).
    requested_gw_modes=list(dict.fromkeys(a.gw_channels))
    if ("cluster" in methods and a.cluster_initial_state=="gw" and
            "density" not in requested_gw_modes):
        requested_gw_modes.insert(0,"density")
    gw=None
    if "cluster" in methods and a.cluster_initial_state=="free":
        gw=run("cluster_seed",lambda:noninteracting_cluster_seed(
            geom,grid,total,mu_tol=opts_gw.mu_tol
        ))
        if gw is not None:
            info["diagnostics"]["cluster_seed"]=dict(
                source_method="noninteracting",
                source_converged=False,
                source_residual=None,
                seed_dyson_rebuilt=True,
                seed_density_error=float(gw.seed_density_error),
                target_V=float(a.V),seed_mu=float(gw.mu)
            )
    vertex_opts=SupercellVertexOptions(
        max_iter=a.vertex_max_iter,tol=a.vertex_tol,
        solver="gmres",gmres_restart=a.vertex_restart,
        verbose=a.verbose,momentum_backend="fft"
    )
    if "gw" in methods:
        for mode in requested_gw_modes:
            definition=build_torus_interaction_channel(geom,a.V,mode)
            label_gw=f"GW[{mode}]"
            label_cgw=f"cGW[{mode}]"
            label_post=f"post-GW[{mode}]"
            prev=initial_gw.get(mode)
            previous_V=float(prev[0]) if prev is not None else None
            previous_state=prev[1] if prev is not None else None
            if previous_state is not None and not previous_state.converged:
                # Never continue from a non-fixed-point GW state.
                previous_state=None
                previous_V=None
            if a.gw_ramp_steps > 0:
                ramp=run(label_gw,lambda mode=mode:
                    solve_channel_gw_v_ramp(
                        geom.h0,
                        definition.interaction_pairs,grid,a.V,mode,opts_gw,
                        initial_steps=a.gw_ramp_steps,
                        min_step=a.gw_ramp_min_step,
                        max_stages=a.gw_ramp_max_stages,
                    )
                )
                if ramp is None:
                    mark_method(label_cgw,"skipped",f"{label_gw} failed") if "cgw" in methods else None
                    mark_method(label_post,"skipped",f"{label_gw} failed") if "postgw" in methods else None
                    continue
                info["diagnostics"][f"{label_gw}_ramp"]=ramp.diagnostics()
                scheme=ramp.state
                if not ramp.reached_target:
                    failure=dict(
                        method=label_gw,target_V=float(a.V),
                        last_converged_V=ramp.last_converged_V,
                        ramp=ramp.diagnostics(),
                        recommendation=(
                            "Inspect screening_smin and residual across V. "
                            "If the response is singular, test symmetry-broken "
                            "finite-Q solutions; do not treat the last lower-V "
                            "converged G as the requested target."
                        ),
                    )
                    failure_path=out/f"gw_failure_{mode}.json"
                    failure_path.write_text(
                        json.dumps(json_safe(failure),indent=2),encoding="utf-8"
                    )
                    why=(f"continuation stopped below target V={a.V:g}; "
                         f"last converged V={ramp.last_converged_V}; {failure_path}")
                    mark_method(label_gw,"unconverged",why,
                                dependent=(
                                    ([label_cgw] if "cgw" in methods else [])+
                                    ([label_post] if "postgw" in methods else [])
                                ))
                    print(f"[benchmark] {label_gw}: {why}; continuing",flush=True)
                    if a.strict_method_failures:
                        raise RuntimeError(why)
                    continue
            else:
                scheme=run(label_gw,lambda definition=definition:
                    solve_channel_gw_same_torus(
                        geom.h0,definition,grid,opts=opts_gw,
                        initial=previous_state,
                    )
                )
            if scheme is None:
                if "cgw" in methods:
                    mark_method(label_cgw,"skipped",f"no converged {label_gw}")
                if "postgw" in methods:
                    mark_method(label_post,"skipped",f"no converged {label_gw}")
                continue
            info["diagnostics"][label_gw]=dict(
                converged=scheme.converged,residual=scheme.final_error,
                mu=scheme.mu,n_bosonic_channels=len(definition.labels),
                exact_triangle_A1E=(mode=="pseudospin"),
                residual_A1E_density_retained=(mode=="pseudospin"),
                exact_Fierz_identity=True,
                screening_smin=scheme.min_screening_singular_value,
                warm_started=previous_state is not None,
                warm_started_from_V=previous_V,
                warm_seed_was_converged=(
                    bool(previous_state.converged)
                    if previous_state is not None else None
                ),
            )
            # A separate cluster-ED+GW fixed point does NOT require that the
            # *initial* density GW map has already converged. For a finite but
            # unconverged last GW iterate, rebuild one consistent Dyson G from
            # its output Sigma_H/Sigma_GW, then let the embedding map update
            # lattice W and Sigma on every outer iteration. The preliminary GW
            # result remains "unconverged" for all independent comparisons.
            if mode=="density" and "cluster" in methods and a.cluster_initial_state=="gw":
                if scheme.converged or not a.cluster_require_converged_gw:
                    gw=run("cluster_seed",lambda:
                        cluster_seed_from_density_gw(
                            scheme,geom,grid,total,mu_tol=opts_gw.mu_tol
                        )
                    )
                    if gw is not None:
                        info["diagnostics"]["cluster_seed"]=dict(
                            source_method=label_gw,
                            source_converged=bool(scheme.converged),
                            source_residual=float(scheme.final_error),
                            source_iterations=int(scheme.iterations),
                            seed_dyson_rebuilt=bool(gw.seed_dyson_rebuilt),
                            seed_density_error=float(gw.seed_density_error),
                            target_V=float(a.V),seed_mu=float(gw.mu),
                        )
                        seed_status=("initial-only" if scheme.converged
                                     else "initial-only-unconverged")
                        row("cluster-ED+GW","initial_density_GW_residual",
                            float(scheme.final_error),status=seed_status,
                            notes="GW used ONLY to initialize independent embedding fixed point")
                else:
                    info["diagnostics"]["cluster_seed"]=dict(
                        source_method=label_gw,source_converged=False,
                        source_residual=float(scheme.final_error),
                        rejected_by_strict_option=True
                    )
            if not scheme.converged:
                failure=dict(method=label_gw,target_V=float(a.V),
                             residual=float(scheme.final_error),
                             iterations=int(scheme.iterations),
                             screening_smin=float(scheme.min_screening_singular_value),
                             mixing=float(a.gw_mixing),
                             mixing_method=a.gw_mixing_method,
                             max_iter=int(a.gw_max_iter),
                             tolerance=float(a.gw_tol))
                failure_path=out/f"gw_failure_{mode}.json"
                failure_path.write_text(
                    json.dumps(json_safe(failure),indent=2),encoding="utf-8"
                )
                why=(f"residual={scheme.final_error:.3e} > tol={a.gw_tol:.3e}; "
                     f"smin={scheme.min_screening_singular_value:.3e}; "
                     f"iterations={scheme.iterations}; {failure_path}")
                mark_method(label_gw,"unconverged",why,
                            dependent=(
                                ([label_cgw] if "cgw" in methods else [])+
                                ([label_post] if "postgw" in methods else [])
                            ))
                print(f"[benchmark] {label_gw}: {why}; continuing",flush=True)
                if a.strict_method_failures:
                    raise RuntimeError(f"{label_gw} did not converge: {why}")
                continue
            channel_states[mode]=scheme
            next_gw[mode]=(float(a.V),scheme)
            method_status[label_gw]=dict(status="converged",
                                         residual=float(scheme.final_error))
            green_by_method[label_gw]=project_k_green(scheme.G,geom,K)
            method_mu[label_gw]=float(scheme.mu)
            arrays[f"Giw_full_{mode}"]=np.asarray(scheme.G)
            energy[label_gw]=channel_energy_estimate(scheme,geom.h0,grid)
            row(label_gw,"finite_T_internal_energy_estimate",energy[label_gw],
                notes=("exact A1+E Fierz bare interaction; finite-T GW energy estimate, not E0"
                       if mode=="pseudospin" else
                       "finite-T Galitskii-Migdal estimate; not E0"))
            if "ED" in energy:
                row(label_gw,"energy_estimate_vs_ED_E0_diagnostic",energy[label_gw],
                    ref=energy["ED"],status="mixed-temperature",
                    notes="finite-T estimate compared numerically against zero-T E0; not a controlled E0 error")
            prev_vertex=initial_vertices.get(mode)
            vertex_last_V=(
                float(prev_vertex[0]) if prev_vertex is not None else None
            )
            vertex_seed=prev_vertex[1] if prev_vertex is not None else None
            finished_vertices={}
            scheme_chi=run(label_cgw,lambda scheme=scheme,definition=definition:
                solve_interaction_channel_response(
                    scheme,definition,geom,grid,Q,channels,vertex_opts,
                    strict=True,
                    initial_vertices=vertex_seed,
                    output_vertices=finished_vertices,
                )
            ) if "cgw" in methods else None
            if "cgw" in methods and scheme_chi is None:
                for ch in channels:
                    row(label_cgw,f"chi[{ch}](Q)",None,status="failed",
                        notes="covariant vertex computation failed; GW background remains valid")
            if scheme_chi is not None:
                bubble,cov,valid,ctimes=scheme_chi
                chi_by_method[f"GG[{mode}]"]=bubble
                chi_by_method[label_cgw]={
                    ch:cov[ch] for ch in channels if valid.get(ch,False)
                }
                info["diagnostics"][label_cgw]=dict(
                    vertex_converged=valid,
                    warm_started=vertex_seed is not None,
                    warm_started_from_V=vertex_last_V,
                    certified_vertices=len(finished_vertices),
                )
                if all(valid.values()) and finished_vertices:
                    next_vertices[mode]=(float(a.V),finished_vertices)
                method_status[label_cgw]=dict(
                    status="converged" if all(valid.values()) else "unconverged",
                    channel_status=valid
                )
                for ch in channels:
                    row(label_gw,f"chi_GG[{ch}](Q)",bubble[ch],ref=ed_chi.get(ch),
                        notes="bare GG response on this interaction-channel GW G")
                    row(label_cgw,f"chi[{ch}](Q)",
                        cov[ch] if valid[ch] else None,
                        ref=ed_chi.get(ch),
                        status="ok" if valid[ch] else "unconverged",
                        notes="same physical source; static tail-completed covariant channel vertex")
            if "cgw" not in methods or scheme_chi is None:
                from rubycgw.three_torus_benchmark import hermitian_source_components
                from rubycgw.production_cgw_sox import static_gg_tail_completed
                b={}
                hstat=geom.h0+scheme.Sigma_static[None,None]
                for ch in channels:
                    b[ch]=float(sum(
                        static_gg_tail_completed(scheme.G,src[None],grid,
                                                 hstat,scheme.mu)["completed"][0,0].real
                        for src in hermitian_source_components(geom,ch,Q)
                    ))
                    row(label_gw,f"chi_GG[{ch}](Q)",b[ch],ref=ed_chi.get(ch),
                        notes=("bare bubble; cGW vertex unavailable" if scheme_chi is None and "cgw" in methods
                               else "bare bubble; no cGW vertex requested"))
                chi_by_method[f"GG[{mode}]"]=b
            if "postgw" in methods:
                post=run(label_post,lambda scheme=scheme,definition=definition:
                    one_shot_post_interaction_channel(
                        scheme,definition,geom.h0,grid,total,vertex_opts,
                        m_max=a.post_mmax,allow_unconverged=False,
                        geometry=geom,
                        optimize_translation=not a.post_no_translation,
                        recycle_krylov=not a.post_no_recycle,
                        mu_tol=opts_gw.mu_tol,mu_max_iter=opts_gw.mu_max_iter
                    ))
                if post is not None:
                    method_status[label_post]=dict(status="converged")
                    green_by_method[label_post]=project_k_green(post.G,geom,K)
                    method_mu[label_post]=float(post.mu)
                    arrays[f"Giw_full_post_{mode}"]=np.asarray(post.G)
                    info["diagnostics"][label_post]=dict(
                        mu=post.mu,window_mmax=a.post_mmax,
                        fallback_fraction=post.fallback_fraction,
                        screening_identity_error=post.screening_identity_error,
                        translation_reduced=bool(post.translation_reduced),
                        source_vertices_solved=int(post.source_vertices_solved),
                        recycled_fallbacks=int(post.recycled_fallbacks),
                        post_solver="recycled" if not a.post_no_recycle else "gmres",
                        phase_seconds=post.phase_seconds,
                        transfer_wall_seconds=post.transfer_wall_seconds,
                    )
                    row(label_post,"chi",None,status="not_requested",
                        notes="post-GW compared in G only")
            base=times.get(label_gw,0.)
            cstep=times.get(label_cgw,0.)
            pstep=times.get(label_post,0.)
            total_times[label_gw]=base
            if "cgw" in methods:
                total_times[label_cgw]=base+cstep
            if "postgw" in methods:
                total_times[label_post]=base+cstep+pstep
            # The density GW -> cluster embedding seed is prepared above,
            # even when the density GW itself reached max_iter without
            # converging. Never promote that GW result into GW/cGW/post tables.
    if gw is None and "cluster" in methods:
        mark_method("cluster-ED+GW","skipped",
                    "no finite density-GW iterate at target V for embedding seed "
                    "(or --cluster-require-converged-gw was set)")
    if gw is not None and "cluster" in methods:
        cl=run("cluster-ED+GW",lambda:uniform_primitive_cluster_ed_gw(
            geom,grid,gw,nbath=a.nbath,max_iter=a.cluster_max_iter,
            tol=a.cluster_tol,mixing=a.cluster_mixing,
            bath_nfit=a.cluster_bath_fit_nfreq,
            bath_max_nfev=a.cluster_bath_max_nfev,
            verbose=a.verbose,target_particles=total
        ))
        if cl is not None:
            if not cl.converged:
                why=f"cluster ED+GW residual={cl.final_error:.3e}, tol={a.cluster_tol:.3e}"
                info["diagnostics"]["cluster-ED+GW"]=dict(
                    converged=False, residual=float(cl.final_error),
                    bath_fit_error=float(cl.bath_fit_error),
                    impurity_mismatch=float(cl.impurity_mismatch),
                    iterations=int(cl.iterations),
                    timing_seconds=getattr(cl,"timing_seconds",None),
                )
                mark_method("cluster-ED+GW","unconverged",why)
                print(f"[benchmark] {why}; omitting G and energy",flush=True)
                if a.strict_method_failures:
                    raise RuntimeError(why)
            else:
                method_status["cluster-ED+GW"]=dict(
                    status="converged",
                    seed_GW_converged=bool(gw.converged),
                    seed_GW_residual=json_safe(float(gw.final_error)),
                    seed_origin=str(gw.seed_origin)
                )
        if cl is not None and cl.converged:
            green_by_method["cluster-ED+GW"]=project_k_green(cl.G,geom,K)
            method_mu["cluster-ED+GW"]=float(cl.mu)
            arrays["cluster_Giw_full"]=cl.G
            info["diagnostics"]["cluster-ED+GW"]=dict(
                converged=cl.converged,residual=cl.final_error,
                bath_fit_error=cl.bath_fit_error,
                impurity_mismatch=cl.impurity_mismatch,
                n_impurities=1,correlated_sites_per_impurity=6,
                nbath=a.nbath,
                seed_GW_converged=bool(gw.converged),
                seed_GW_residual=json_safe(float(gw.final_error)),
                seed_origin=str(gw.seed_origin),
                seed_dyson_rebuilt=bool(gw.seed_dyson_rebuilt),
                timing_seconds=getattr(cl,"timing_seconds",None))
            ecl=embedded_gm_energy(cl,geom,grid)
            energy["cluster"]=ecl
            row("cluster-ED+GW","finite_T_GM_proxy",ecl,
                status="nonvariational-estimate",
                notes="not a certified stationary energy or ground E0")
            if "ED" in energy:
                row("cluster-ED+GW","energy_proxy_vs_ED_E0_diagnostic",
                    ecl,ref=energy["ED"],status="mixed-temperature",
                    notes="finite-T GM proxy vs zero-T E0; not strict error")
            if a.skip_cluster_chi:
                for ch in channels:
                    row("cluster-ED+GW",f"chi[{ch}](Q)",None,
                        status="skipped",notes="run without --skip-cluster-chi for expensive full source derivative")
            else:
                chi_cl=run("cluster_chi",lambda:solve_cluster_source_response(
                    geom,params,grid,channels,Q,a,uniform=cl
                ))
                if chi_cl is not None:
                    cvals,ctimes,cdiag=chi_cl
                    chi_by_method["cluster-ED+GW"]={
                        ch:val for ch,val in cvals.items() if val is not None
                    }
                    info["diagnostics"]["cluster_source_chi"]=cdiag
                    for ch in channels:
                        passed=cvals[ch] is not None
                        row("cluster-ED+GW",f"chi[{ch}](Q)",cvals[ch],
                            ref=ed_chi.get(ch),runtime=ctimes[ch],
                            status="ok" if passed else "failed",
                            notes=("full multi-impurity fixed-source derivative, warm started"
                                   if passed else cdiag[ch].get("error","source failed")))
                    info["diagnostics"]["cluster_source_chi"]["warmstart_used"]=(
                        not a.cluster_chi_coldstart
                    )
    if len(green_by_method):
        arrays["omega"]=np.asarray(grid.omega)
        for name,G in green_by_method.items():
            arrays["Giw_K_"+name.replace("-","_").replace("+","_")]=G
            if ed_scalar is not None:
                e=float(np.linalg.norm(G-ed_scalar)/max(np.linalg.norm(ed_scalar),1e-300))
                status="reference" if name=="ED" else "ok"
                if name=="ED": e=0.
                row(name,"G_rel_Frobenius_K",e,ref=None,
                    status=status,
                    notes="||G_method(K,iw)-G_ED(K,iw)||_F/||G_ED||_F")
            mu=(energy.get("ED_mu") if name=="ED" else method_mu.get(name))
            if mu is None: continue
            g_tau=green_k_tau(G,grid.omega,a.T,mu,hK,tau)
            curves_data=np.trace(g_tau,axis1=1,axis2=2).real/6.0
            if a.plot_orbital!="trace":
                orb=int(a.plot_orbital)
                if orb not in range(6): raise ValueError("--plot-orbital must be trace or 0..5")
                curves_data=g_tau[:,orb,orb]
            curves[name]=np.asarray(curves_data,dtype=float)
            arrays["Gtau_"+name.replace("-","_").replace("+","_")]=g_tau
        if "ED" in curves:
            ed_curve=curves["ED"]
            for name,curve in curves.items():
                rel=float(np.linalg.norm(curve-ed_curve)/max(np.linalg.norm(ed_curve),1e-300))
                row(name,"G_tau_curve_relative_L2_error",rel,
                    notes="selected trace/orbital on 0<tau<beta; ED TL approximation")
        save_plot(out,tau,curves,K,a.V,a.filling,1/a.T)
        arrays["tau"]=tau
        arrays["tau_over_beta"]=tau*a.T
    if "ED" in methods:
        total_times["ED"]=times.get("ED",0.0)+times.get("ED_Green",0.0)
    if "cluster" in methods and "cluster-ED+GW" in times:
        total_times["cluster-ED+GW"]=(
            times.get("GW[density]",0.0)+times.get("cluster_seed",0.0)
            +times.get("cluster-ED+GW",0.0)+times.get("cluster_chi",0.0)
        )
    for name,total_s in total_times.items():
        row(name,"total_method_runtime_seconds",float(total_s),
            runtime=total_s,
            notes="cumulative in one interaction-channel pipeline; shared GW counted once")
    info["method_total_runtime_seconds"]=total_times
    info["method_status"]=method_status
    # CSV's primary runtime column is the cumulative time the user requested.
    # Preserve the incremental per-stage/per-channel value separately.
    for rec in rows:
        rec["stage_runtime_seconds"] = rec.get("runtime_seconds", "")
        if rec.get("method") in total_times:
            rec["runtime_seconds"] = float(total_times[rec["method"]])
    for method,timing in times.items():
        info["timings"][method]=float(timing)
    info["energies"]=energy
    info["computed_chi"]=chi_by_method
    info["G_methods"]=list(green_by_method)
    output_rows(rows,out)
    # Keep comparison.csv as a detailed log, but also create one-row-per-
    # method summaries to make the best G and best chi choices visible.
    try:
        make_comparison_scorecards(rows,out)
    except (ValueError,KeyError,OSError) as exc:
        info["errors"]["summary_scorecards"]=f"{type(exc).__name__}: {exc}"
        print(f"[benchmark] scorecard export failed: {exc}",flush=True)
        if a.strict_method_failures:
            raise
    with (out/"summary.json").open("w",encoding="utf-8") as f:
        json.dump(json_safe(info),f,ensure_ascii=False,indent=2,allow_nan=False)
    np.savez_compressed(out/"raw_arrays.npz",**arrays)
    print(f"[benchmark] saved {out/'comparison.csv'}, comparison_wide.csv, ranking_green.csv, ranking_susceptibility.csv, summary.json, raw_arrays.npz",flush=True)
    if curves: print(f"[benchmark] saved {out/'green_tau.png'} and .pdf",flush=True)
    if info["errors"]: print("[benchmark] partial failures:",info["errors"],flush=True)
    if return_scan_state:
        return {
            "converged_GW":next_gw,
            "converged_vertices":next_vertices,
            "summary":info,
            "output":out,
        }
    return None


if __name__=="__main__":
    main()
