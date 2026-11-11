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
from rubycgw.ed18_thermal import thermal_response as hex_thermal_response
from rubycgw.ed_cluster import RubyEDClusterSolver
from rubycgw.ed_cluster_thermal import thermal_response as rect_thermal_response
from rubycgw.thermal_lanczos_green import thermal_lanczos_green, ThermalLanczosOptions
from rubycgw.three_torus_benchmark import (
    build_torus_geometry, k_projector, project_k_green, green_k_tau,
    hermitian_source_components, channel_static_response,
    max_required_ed_dimension, uniform_primitive_cluster_ed_gw,
    SkewEDSectorAdapter, gw_energy_from_state, embedded_gm_energy,
)
from rubycgw.post_gw import run_post_gw
from rubycgw.free_energy import evaluate_gw_free_energy
from rubycgw.cluster_ed_gw_multicell import (
    MultiCellEDGWOptions, solve_multicell_cluster_ed_gw,
)
from rubycgw.cluster_ed_gw import ruby_cluster_interactions


def parse_args():
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
    p.add_argument("--channels",nargs="+",
                   default=["x_even","y_even","z_same","z_opposite"])
    p.add_argument("--plot-orbital",type=str,default="trace",
                   help="trace means Tr(G_K)/6; otherwise an orbital index 0..5")
    p.add_argument("--plot-tau-points",type=int,default=101)
    p.add_argument("--gw-max-iter",type=int,default=150)
    p.add_argument("--gw-tol",type=float,default=1e-7)
    p.add_argument("--gw-mixing",type=float,default=.22)
    p.add_argument("--vertex-max-iter",type=int,default=120)
    p.add_argument("--vertex-tol",type=float,default=2e-7)
    p.add_argument("--vertex-restart",type=int,default=12)
    p.add_argument("--post-mmax",type=int,default=None)
    p.add_argument("--nbath",type=int,default=6)
    p.add_argument("--cluster-max-iter",type=int,default=50)
    p.add_argument("--cluster-tol",type=float,default=3e-5)
    p.add_argument("--cluster-mixing",type=float,default=.35)
    p.add_argument("--cluster-bath-fit-nfreq",type=int,default=12)
    p.add_argument("--cluster-bath-max-nfev",type=int,default=200)
    p.add_argument("--source-step",type=float,default=1e-3)
    p.add_argument("--skip-cluster-chi",action="store_true",
                   help="skip expensive finite-source cluster-covariant chi; no false bubble alias")
    p.add_argument("--ed-samples",type=int,default=6)
    p.add_argument("--ed-tau-points",type=int,default=17)
    p.add_argument("--ed-sector-padding",type=int,default=1)
    p.add_argument("--ed-thermal-eigs",type=int,default=5)
    p.add_argument("--ed-krylov-steps",type=int,default=100)
    p.add_argument("--ed-max-dim",type=int,default=3000000,
                   help="abort ED before sector allocations exceed this Hilbert dimension")
    p.add_argument("--ed-eigs",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261111)
    p.add_argument("--allow-unconverged",action="store_true")
    p.add_argument("--verbose",action="store_true")
    p.add_argument("--out",type=Path,default=Path("results/three_torus_benchmark"))
    return p.parse_args()


def onebody_channel_expectation(state, H0, K, grid):
    rho=one_body_density_matrix_tail(state.G,grid,H0,float(state.mu),state.Sigma_H)
    return float(np.einsum("ab,ba->",K,rho[0,0]).real)


def solve_cluster_source_response(geom, params, grid, channels, Q, args):
    """Finite difference of the *full* multi-impurity self-consistent map.

    This is NOT the ordinary GG bubble on the cluster-modified G. Each
    physical cell in the finite supercell has its own six-site impurity,
    with a complex bath, so a finite-Q source can induce distinct local
    self-energies. Cost: 2*Nchannels*Ncomponents full multi-impurity runs.
    """
    out={}; times={}; diagnostics={}
    interactions=ruby_cluster_interactions(params)
    opts=MultiCellEDGWOptions(
        max_iter=int(args.cluster_max_iter),tol=float(args.cluster_tol),
        mixing=float(args.cluster_mixing),nbath=int(args.nbath),
        bath_fit_nfreq=int(args.cluster_bath_fit_nfreq),
        bath_fit_max_nfev=int(args.cluster_bath_max_nfev),
        complex_bath=True,verbose=bool(args.verbose)
    )
    h=abs(float(args.source_step))
    if h==0:
        raise ValueError("--source-step must be nonzero")
    for ch in channels:
        t0=perf_counter()
        response=0.
        maxres=0.
        for K in hermitian_source_components(geom,ch,Q):
            vals=[]
            for sign in (1.,-1.):
                Hsrc=np.asarray(geom.h0,complex).copy()
                Hsrc[0,0]-=sign*h*K
                solved=solve_multicell_cluster_ed_gw(
                    Hsrc,geom.vq,interactions,grid,
                    target_particles=float(args.filling*geom.ncell),opts=opts
                )
                maxres=max(maxres,float(solved.final_error))
                if not solved.converged and not args.allow_unconverged:
                    raise RuntimeError(
                        f"source clusterED+GW {ch} sign={sign} failed: "
                        f"residual={solved.final_error:.3e}"
                    )
                vals.append(onebody_channel_expectation(solved,Hsrc,K,grid))
            response+=(vals[0]-vals[1])/(2*h)
        out[ch]=float(response)
        times[ch]=float(perf_counter()-t0)
        diagnostics[ch]=dict(residual=maxres,source_step=h)
    return out,times,diagnostics


def output_rows(rows, out):
    columns=["method","quantity","value","ED_reference","relative_error",
             "runtime_seconds","status","notes"]
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
    styles=[
        ("ED","red","-",2.0),
        ("GW","blue",":",2.0),
        ("post-GW","forestgreen","--",2.0),
        ("cluster-ED+GW","black","-.",1.8),
    ]
    fig,ax=plt.subplots(figsize=(6.2,4.65),dpi=130)
    for label,color,ls,lw in styles:
        if label in curves:
            ax.plot(tau/beta,curves[label],label=label,color=color,
                    linestyle=ls,linewidth=lw)
    ax.set_xlabel(r"$\tau/\beta$")
    ax.set_ylabel(r"$G(K,\tau)$")
    ax.set_xlim(0,1)
    ax.set_title(f"Ruby {V=:.3g}, filling={filling:g}, K=({K[0]:g},{K[1]:g})")
    ax.legend(loc="best",frameon=True,fontsize=9)
    ax.grid(alpha=.16)
    fig.tight_layout()
    fig.savefig(out/"green_tau.png",dpi=200)
    fig.savefig(out/"green_tau.pdf")
    plt.close(fig)


def main():
    a=parse_args()
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
    N=int(round(total))
    if abs(total-N)>1e-10:
        raise ValueError(
            f"ED fixed-N requires integer filling*Ncell; got {total}"
        )
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=a.nw,nOmega=a.nomega,T=a.T)
    tau=np.linspace(0,1/a.T,int(a.plot_tau_points))
    U=k_projector(geom,K)
    hK=U.conj().T@geom.h0[0,0]@U
    methods=set(a.methods)
    if ("cgw" in methods or "postgw" in methods or "cluster" in methods):
        methods.add("gw")
    if "ed" in methods:
        dmax=max_required_ed_dimension(geom.nsites,N,int(a.ed_sector_padding))
        if dmax>int(a.ed_max_dim):
            raise RuntimeError(
                f"ED adjacent-sector dimension {dmax:,} exceeds "
                f"--ed-max-dim={a.ed_max_dim:,}; lower filling or allow "
                "a larger explicit cap only with adequate RAM"
            )
    info=dict(geometry=geom.name,V=a.V,filling=a.filling,N=N,sites=geom.nsites,
              K=K,Q=Q,T=a.T,beta=1/a.T,
              hoppings=dict(t=a.t,t1=a.t1,t2=a.t2),
              channels=channels,allowed_K=geom.allowed_momenta(),
              ED_green="truncated-sector finite-T Lanczos (not full thermal trace)",
              ED_chi="canonical fixed-N stochastic thermal typicality",
              GW_ensemble="grand canonical, fixed mean N",
              chi="connected static source response at same physical Q",
              post_scope="one-shot Dyson; only G compared",
              cgw_green="identical to ordinary self-consistent GW",
              energy_scope="ED exact E0; GW finite-T Galitskii-Migdal estimate",
              timings={},diagnostics={},errors={})
    rows=[]; curves={}; arrays={}; green_by_method={}; chi_by_method={}
    times={}; energy={}; ed_scalar=None; ed_full=None; ed_chi={}
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
            if not a.allow_unconverged:
                raise
            return None
    if "ed" in methods:
        def stage_ed():
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
        if result is not None:
            ed,sp,chi=result
            energy["ED"]=float(sp.energies[0])
            info["diagnostics"]["ED"]=dict(
                dimension=int(ed.dimension),ground_multiplicity=int(sp.ground_multiplicity),
                gap=float(sp.gap_above_manifold),samples=a.ed_samples,
                tau_points=a.ed_tau_points
            )
            row("ED","ground_state_E0",energy["ED"],runtime=times["ED"],
                notes="exact fixed-N ground state up to Lanczos tolerance")
            for i,ch in enumerate(channels):
                ed_chi[ch]=float(np.real(chi.susceptibility_matrix[i,i]))
                row("ED",f"chi[{ch}](Q)",ed_chi[ch],runtime=times["ED"],
                    notes="canonical stochastic thermal trace; increase samples/tau points")
            arrays["ED_chi_matrix"]=np.asarray(chi.susceptibility_matrix)
            arrays["ED_connected_tau"]=np.asarray(chi.correlation_tau_matrix)
            def stage_ed_green():
                opts=ThermalLanczosOptions(
                    sector_padding=a.ed_sector_padding,
                    thermal_eigs=a.ed_thermal_eigs,
                    krylov_dim=a.ed_krylov_steps,
                    verbose=a.verbose
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
            if gr is not None:
                info["diagnostics"]["ED_Green"]=dict(
                    mu=gr.mu,thermal_kept_weight=gr.kept_thermal_weight,
                    sector_edge_weight=float(gr.sector_probabilities[0]+gr.sector_probabilities[-1]),
                    translation_residual=gr.translation_residual,
                    sector_numbers=gr.sector_numbers,
                    sector_probabilities=gr.sector_probabilities,
                    krylov_steps_max=gr.krylov_steps_max)
                ed_full=np.asarray(gr.G)
                ed_scalar=project_k_green(ed_full,geom,K)
                green_by_method["ED"]=ed_scalar
                arrays["ED_Giw_full"]=ed_full
                energy["ED_mu"]=float(gr.mu)
                row("ED","G_reference_initialized",0.,runtime=times["ED_Green"],
                    notes="finite-T sector-truncated thermal Lanczos ED")
    opts_gw=GWOptions(
        target_filling=total,max_iter=a.gw_max_iter,tol=a.gw_tol,
        mixing=a.gw_mixing,mixing_method="pulay",
        verbose=a.verbose,momentum_backend="fft",
        mu_tol=1e-10,mu_max_iter=100
    )
    gw=None
    if "gw" in methods:
        gw=run("GW",lambda:solve_matrix_gw_fast(
            geom.h0,geom.vq,grid,opts=opts_gw
        ))
        if gw is not None:
            info["diagnostics"]["GW"]=dict(
                converged=gw.converged,residual=gw.final_error,mu=gw.mu)
            if not gw.converged and not a.allow_unconverged:
                raise RuntimeError(f"GW did not converge: {gw.final_error:.3e}")
            green_by_method["GW"]=project_k_green(gw.G,geom,K)
            arrays["GW_Giw_full"]=np.asarray(gw.G)
            energy["GW"]=run("GW_energy",lambda:gw_energy_from_state(gw,geom,grid))
            if energy["GW"] is not None:
                row("GW","finite_T_internal_energy_estimate",
                    energy["GW"],ref=None,
                    notes="Galitskii-Migdal finite-T estimate; NOT ground-state E0")
                if "ED" in energy:
                    row("GW","energy_estimate_vs_ED_E0_diagnostic",
                        energy["GW"],ref=energy["ED"],
                        status="mixed-temperature",
                        notes="finite-T vs T=0; not a controlled ground-state error")
            try:
                F=evaluate_gw_free_energy(
                    gw,geom.h0,geom.vq,grid,target_particles=total,
                    primitive_cells_per_supercell=geom.ncell
                )
                energy["GW_F"]=float(F.helmholtz_free_energy)
                row("GW","finite_T_Helmholtz_F",energy["GW_F"],
                    notes="stationary GW free energy, not E0")
            except Exception as exc:
                info["errors"]["GW_free_energy"]=str(exc)
    if gw is not None and ("cgw" in methods):
        vertex_opts=SupercellVertexOptions(
            max_iter=a.vertex_max_iter,tol=a.vertex_tol,
            solver="gmres",gmres_restart=a.vertex_restart,
            verbose=a.verbose,momentum_backend="fft"
        )
        resp=run("cGW",lambda:channel_static_response(
            gw,geom,grid,Q,channels,include_cgw=True,
            vertex_opts=vertex_opts,strict=not a.allow_unconverged
        ))
        if resp is not None:
            bubble,cov,valid=resp
            chi_by_method["GW_bubble"]=bubble
            chi_by_method["cGW"]=cov
            info["diagnostics"]["cGW"]=dict(vertex_converged=valid)
            for ch in channels:
                row("GW",f"chi_GG[{ch}](Q)",bubble[ch],ref=ed_chi.get(ch),
                    runtime=times.get("GW"),notes="bare bubble on self-consistent GW G")
                row("cGW",f"chi[{ch}](Q)",cov[ch],ref=ed_chi.get(ch),
                    status="ok" if valid[ch] else "unconverged",
                    notes="tail-consistent covariant vertex at Omega_ext=0")
            row("cGW","G_same_as_GW",0.,notes="cGW does not update Dyson G")
            if "GW" in energy:
                row("cGW","finite_T_internal_energy_estimate",energy["GW"],
                    notes="same GW background; cGW vertex does not recalculate energy")
    elif gw is not None:
        bubble,_,_=channel_static_response(
            gw,geom,grid,Q,channels,include_cgw=False
        )
        chi_by_method["GW_bubble"]=bubble
        for ch in channels:
            row("GW",f"chi_GG[{ch}](Q)",bubble[ch],ref=ed_chi.get(ch),
                notes="bare GW-GG bubble")
    if gw is not None and "postgw" in methods:
        def stage_post():
            opts=SupercellVertexOptions(
                max_iter=a.vertex_max_iter,tol=a.vertex_tol,
                solver="gmres",gmres_restart=a.vertex_restart,
                verbose=a.verbose,momentum_backend="fft"
            )
            return run_post_gw(
                gw,geom.h0,geom.vq,grid,gw_opts=opts_gw,
                vertex_opts=opts,m_max=a.post_mmax,
                allow_unconverged=a.allow_unconverged
            )
        post=run("post-GW",stage_post)
        if post is not None:
            green_by_method["post-GW"]=project_k_green(post.G,geom,K)
            arrays["post_Giw_full"]=post.G
            info["diagnostics"]["post-GW"]=dict(
                mu=post.mu,window_mmax=a.post_mmax,
                fallback_fraction=float(np.mean(post.used_background_w_mask))
            )
            row("post-GW","chi",None,status="not_requested",
                notes="post-GW is compared only through Green function")
    if gw is not None and "cluster" in methods:
        cl=run("cluster-ED+GW",lambda:uniform_primitive_cluster_ed_gw(
            geom,grid,gw,nbath=a.nbath,max_iter=a.cluster_max_iter,
            tol=a.cluster_tol,mixing=a.cluster_mixing,
            bath_nfit=a.cluster_bath_fit_nfreq,
            bath_max_nfev=a.cluster_bath_max_nfev,
            verbose=a.verbose
        ))
        if cl is not None:
            if not cl.converged and not a.allow_unconverged:
                raise RuntimeError(
                    f"uniform cluster ED+GW did not converge: {cl.final_error:.3e}"
                )
            green_by_method["cluster-ED+GW"]=project_k_green(cl.G,geom,K)
            arrays["cluster_Giw_full"]=cl.G
            info["diagnostics"]["cluster-ED+GW"]=dict(
                converged=cl.converged,residual=cl.final_error,
                bath_fit_error=cl.bath_fit_error,
                impurity_mismatch=cl.impurity_mismatch,
                n_impurities=1,correlated_sites_per_impurity=6,
                nbath=a.nbath)
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
                    geom,params,grid,channels,Q,a
                ))
                if chi_cl is not None:
                    cvals,ctimes,cdiag=chi_cl
                    chi_by_method["cluster-ED+GW"]=cvals
                    info["diagnostics"]["cluster_source_chi"]=cdiag
                    for ch in channels:
                        row("cluster-ED+GW",f"chi[{ch}](Q)",cvals[ch],
                            ref=ed_chi.get(ch),runtime=ctimes[ch],
                            notes="fixed-filling finite difference, independent 6-site impurity per cell")
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
            mu=(energy.get("ED_mu") if name=="ED" else
                float(gw.mu) if name=="GW" else
                float(post.mu) if name=="post-GW" and 'post' in locals() and post else
                float(cl.mu) if name=="cluster-ED+GW" and 'cl' in locals() and cl else None)
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
        if "GW" in curves and "cgw" in methods:
            row("cGW","G_rel_Frobenius_K",
                float(np.linalg.norm(green_by_method["GW"]-ed_scalar)/
                      max(np.linalg.norm(ed_scalar),1e-300))
                if ed_scalar is not None else None,
                notes="identical to GW Green by definition")
        save_plot(out,tau,curves,K,a.V,a.filling,1/a.T)
        arrays["tau"]=tau
        arrays["tau_over_beta"]=tau*a.T
    # Shared GW is a prerequisite for cGW, post-GW and cluster embedding.
    # Report both incremental stage times and end-to-end method costs.
    cumulative = {
        "ED": times.get("ED",0.0)+times.get("ED_Green",0.0),
        "GW": times.get("GW",0.0)+times.get("GW_energy",0.0),
        "cGW": times.get("GW",0.0)+times.get("cGW",0.0),
        "post-GW": times.get("GW",0.0)+times.get("post-GW",0.0),
        "cluster-ED+GW": times.get("GW",0.0)+times.get("cluster-ED+GW",0.0)
                         +times.get("cluster_chi",0.0),
    }
    for key in list(cumulative):
        if key not in ("ED" if "ED" in methods else "", "GW" if gw is not None else "",
                       "cGW" if "cgw" in methods else "", "post-GW" if "postgw" in methods else "",
                       "cluster-ED+GW" if "cluster" in methods else ""):
            del cumulative[key]
    for key,total_s in cumulative.items():
        row(key,"total_method_runtime_seconds",float(total_s),
            runtime=total_s,notes="includes shared GW when required; stage times recorded separately")
    info["method_total_runtime_seconds"]=cumulative
    for method,timing in times.items():
        info["timings"][method]=float(timing)
    info["energies"]=energy
    info["computed_chi"]=chi_by_method
    info["G_methods"]=list(green_by_method)
    output_rows(rows,out)
    with (out/"summary.json").open("w",encoding="utf-8") as f:
        json.dump(json_safe(info),f,ensure_ascii=False,indent=2,allow_nan=False)
    np.savez_compressed(out/"raw_arrays.npz",**arrays)
    print(f"[benchmark] saved {out/'comparison.csv'}, summary.json, raw_arrays.npz",flush=True)
    if curves: print(f"[benchmark] saved {out/'green_tau.png'} and .pdf",flush=True)
    if info["errors"]: print("[benchmark] partial failures:",info["errors"],flush=True)


if __name__=="__main__":
    main()
