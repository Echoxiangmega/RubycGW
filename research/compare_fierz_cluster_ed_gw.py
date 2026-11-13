#!/usr/bin/env python3
"""Benchmark ED against three Fierz-consistent cluster-ED+GW embeddings.

Physical lattice: the SAME 18-site skew-periodic Ruby torus for every mode.
ED impurity: the SAME two 3-site triangle density-density V Hamiltonian.
Embedding GW: density, pseudospin A1+E, or symmetric bond+current (BJ).
The matching *cluster GW subtraction* is recomputed in the SAME Fierz mode.

Static LC/CO cluster chi defaults to the full self-consistent +/- source
derivative on the Fierz-matched multi-impurity finite-bath fixed point.
To opt into the analytic fixed-point derivative (NOT a GG bubble),
add --cluster-chi-analytic.

Example (PowerShell from repository root):
    python research/compare_fierz_cluster_ed_gw.py --V 0.2 --n 2 --nbath 2 --nw 9 --nomega 3 --out results/test_fierz_cluster

For the full physical comparison increase nbath/nw and check the
finite-source field-step and self-consistency convergence. Analytic
GMRES response can be slower than finite-source at these parameters;
it remains available as an optional cross-check.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor

if __package__ in (None,""):
    root=str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0,root)

import numpy as np

from rubycgw.model import RubyParameters
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.momentum_ed import MomentumTorusED
from rubycgw.momentum_ed_green import momentum_thermal_lanczos_green
from rubycgw.grand_canonical_ed import grand_canonical_ed_response
from rubycgw.thermal_lanczos_green import ThermalLanczosOptions
from rubycgw.fierz_channel_gw import solve_channel_gw_same_torus
from rubycgw.fierz_cluster_embedding import build_embedding_channels
from rubycgw.cluster_ed_gw import ruby_cluster_interactions
from rubycgw.cluster_ed_gw_linear_response import StaticClusterEDGWLinearizer
from rubycgw.cluster_ed_gw_multicell import (
    MultiCellEDGWOptions,MultiCellRestart,solve_multicell_cluster_ed_gw,
    restart_from_uniform_cluster,
)
from rubycgw.supercell_gw_fast import _solve_mu_matrix_fast
from rubycgw.supercell_gw_split import one_body_density_matrix_tail
from rubycgw.three_torus_benchmark import (
    build_torus_geometry,noninteracting_cluster_seed,
    cluster_seed_from_density_gw,uniform_primitive_cluster_ed_gw,
    project_k_green,k_projector,green_k_tau,hermitian_source_components
)
from benchmark_three_tori import _cluster_source_solve


def parse_args(argv=None):
    p=argparse.ArgumentParser(
        description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--V",type=float,default=None,
                   help="single interaction strength (old single-point mode)")
    p.add_argument("--V-scan",nargs="+",type=float,default=None,metavar="V",
                   help="explicit ordered V values; each point warm-starts from the previous converged point")
    p.add_argument("--V-start",type=float,default=None)
    p.add_argument("--V-stop",type=float,default=None)
    p.add_argument("--V-step",type=float,default=None,
                   help="inclusive arithmetic V grid with --V-start/--V-stop")
    p.add_argument("--scan-on-failure",choices=("stop","continue"),default="stop",
                   help="stop at first incomplete V (default), or continue from last certified per-mode solution")
    p.add_argument("--resume-scan",action="store_true",
                   help="reuse completed V point results/checkpoints from --out; retry the first incomplete point")
    p.add_argument("--n","--filling",dest="filling",type=float,required=True)
    p.add_argument("--T",type=float,default=.125)
    p.add_argument("--t",type=float,default=1.)
    p.add_argument("--t1",type=float,default=-.5)
    p.add_argument("--t2",type=float,default=.5)
    p.add_argument("--K",nargs=2,type=float,default=(1/3,1/3))
    p.add_argument("--Q",nargs=2,type=float,default=(1/3,1/3))
    p.add_argument("--modes",nargs="+",choices=("density","pseudospin","bj"),
                   default=("density","pseudospin","bj"))
    p.add_argument("--channels",nargs="+",default=(
        "x_even","y_even","z_same","z_opposite"
    ))
    p.add_argument("--nw",type=int,default=49)
    p.add_argument("--nomega",type=int,default=10)
    p.add_argument("--nbath",type=int,default=6)
    p.add_argument("--bath-fit-nfreq",type=int,default=12)
    p.add_argument("--bath-max-nfev",type=int,default=250)
    p.add_argument("--initial-state",choices=("free","gw"),default="free",
                   help="free: no initial standalone GW; gw: mode-matched GW seed")
    p.add_argument("--gw-max-iter",type=int,default=80)
    p.add_argument("--gw-tol",type=float,default=1e-7)
    p.add_argument("--gw-mixing",type=float,default=.2)
    p.add_argument("--uniform-max-iter",type=int,default=60)
    p.add_argument("--uniform-tol",type=float,default=3e-5)
    p.add_argument("--uniform-mixing",type=float,default=.35)
    p.add_argument("--uniform-mixing-method",choices=("linear","pulay"),
                   default="pulay")
    p.add_argument("--uniform-bath",choices=("real","complex"),default="real",
                   help="uniform stage: real bath (historical) or complex bath")
    p.add_argument("--uniform-seed-channel",type=str,default=None,
                   help="optional intra-cell Q=0 LC/CO perturbation in initial self-energy")
    p.add_argument("--uniform-seed-amplitude",type=float,default=0.)
    p.add_argument("--pulay-history",type=int,default=6)
    p.add_argument("--pulay-start",type=int,default=3)
    p.add_argument("--pulay-step-cap",type=float,default=2.)
    p.add_argument("--pulay-reset-growth",type=float,default=2.)
    p.add_argument("--bath-energy-window",type=float,default=4.)
    p.add_argument("--bath-coupling-bound",type=float,default=4.)
    p.add_argument("--bath-fit-restarts",type=int,default=1,
                   help="extra bath starting points on failure/bounds/high fit error")
    p.add_argument("--bath-retry-error",type=float,default=.03)
    p.add_argument("--checkpoint-every",type=int,default=5)
    p.add_argument("--resume-uniform",action="store_true",
                   help="resume per-mode uniform state from --out, without V continuation")
    p.add_argument("--resume-baseline",action="store_true",
                   help="resume zero-field multicell state from --out and skip uniform")
    p.add_argument("--no-screening-monitor",dest="screening_monitor",
                   action="store_false",help="disable per-iteration screening singular-value logging")
    p.set_defaults(screening_monitor=True)
    p.add_argument("--baseline-max-iter",type=int,default=80)
    p.add_argument("--baseline-tol",type=float,default=3e-6)
    p.add_argument("--baseline-mixing",type=float,default=.45)
    p.add_argument("--baseline-mixing-method",choices=("linear","pulay"),
                   default="pulay")
    p.add_argument("--source-mixing-method",choices=("linear","pulay"),
                   default="pulay")
    p.add_argument("--cluster-chi-method",choices=("analytic","finite-source"),
                   default="finite-source",
                   help="finite-source (default): self-consistent +/-h derivative; analytic: full fixed-point linear response")
    p.add_argument("--cluster-chi-analytic",
                   action="store_const",dest="cluster_chi_method",
                   const="analytic",
                   help="opt in to the analytic fixed-point cluster chi")
    p.add_argument("--cluster-chi-finite-source",
                   action="store_const",dest="cluster_chi_method",
                   const="finite-source",
                   help="explicitly choose the default +/-h source method (compatibility alias)")
    p.add_argument("--analytic-tol",type=float,default=1e-5,
                   help="relative GMRES tolerance for analytic cluster chi")
    p.add_argument("--analytic-maxiter",type=int,default=12)
    p.add_argument("--analytic-restart",type=int,default=8)
    p.add_argument("--bath-stationarity-tol",type=float,default=2e-5,
                   help="maximum stationarity error of complex bath fit for analytic response")
    p.add_argument("--source-max-iter",type=int,default=100)
    p.add_argument("--source-tol",type=float,default=1e-6)
    p.add_argument("--source-mixing",type=float,default=.45)
    p.add_argument("--source-step",type=float,default=.001)
    p.add_argument("--source-workers",type=int,default=1)
    p.add_argument("--skip-chi",action="store_true",
                   help="only compare ED and cluster Green functions")
    p.add_argument("--ed-samples",type=int,default=6)
    p.add_argument("--ed-tau-points",type=int,default=17)
    p.add_argument("--ed-charge-sector-padding",type=int,default=1)
    p.add_argument("--ed-sector-padding",type=int,default=2)
    p.add_argument("--ed-thermal-eigs",type=int,default=12)
    p.add_argument("--ed-krylov-steps",type=int,default=160)
    p.add_argument("--tau-points",type=int,default=81)
    p.add_argument("--strict",action="store_true")
    p.add_argument("--verbose",action="store_true")
    p.add_argument("--out",type=Path,required=True)
    return p.parse_args(argv)


def _json(x):
    if isinstance(x,dict):
        return {str(k):_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [_json(v) for v in x]
    if isinstance(x,np.ndarray):
        return _json(x.tolist())
    if isinstance(x,(np.floating,float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.bool_,bool)):
        return bool(x)
    if isinstance(x,(np.integer,int)):
        return int(x)
    if isinstance(x,(complex,np.complexfloating)):
        return dict(real=float(x.real),imag=float(x.imag))
    return x


def _save_csv(path,rows,fields):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _fierz_seed(gw,mode,geom,grid,target):
    """Rebuild a Dyson-consistent starting state from a mode-matched GW.

    In the density convention Sigma_H stores the tadpole only and
    Sigma_GW stores exchange plus correlation. In pseudospin/BJ convention
    Sigma_H stores the complete physical static HF contribution, and
    Sigma_GW stores the dynamic correlation. Their SUM is the same
    self-energy split used in that embedding's GW map.
    """
    if mode=="density":
        return cluster_seed_from_density_gw(gw,geom,grid,target)
    hs=np.asarray(gw.Sigma_static,complex)
    sd=np.asarray(gw.Sigma_c,complex)
    mu,G,_,_=_solve_mu_matrix_fast(
        geom.h0,hs,sd,grid,float(target),float(gw.mu),1e-10,100
    )
    rho=one_body_density_matrix_tail(G,grid,geom.h0,mu,hs)
    dens=np.real(np.diag(rho[0,0]))
    if not all(np.all(np.isfinite(x)) for x in (G,hs,sd,dens)):
        raise ValueError("Fierz GW initial self-energy has nonfinite values")
    return SimpleNamespace(
        G=G,Sigma_H=hs,Sigma_GW=sd,mu=float(mu),density=dens,
        converged=bool(gw.converged),final_error=float(gw.final_error)
    )


def _multi_restart(x):
    """Copy a certified multicell result/restart without sharing mutable baths."""
    impurity=(x.Sigma_imp if isinstance(x,MultiCellRestart)
              else x.Sigma_ED_cluster)
    return MultiCellRestart(
        Sigma_H=np.asarray(x.Sigma_H,complex).copy(),
        Sigma_emb=np.asarray(x.Sigma_emb,complex).copy(),
        Sigma_imp=np.asarray(impurity,complex).copy(),
        G=np.asarray(x.G,complex).copy(),mu=float(x.mu),
        bath_energies=np.asarray(x.bath_energies,float).copy(),
        bath_couplings=np.asarray(x.bath_couplings,complex).copy()
    )


def _checkpoint_signature(a,mode,stage):
    """Exclude mixer settings so an unstable iteration can resume with safer mixing."""
    return dict(
        version=1,stage=str(stage),mode=str(mode),geometry="hex18",
        V=float(a.V),filling=float(a.filling),T=float(a.T),
        t=float(a.t),t1=float(a.t1),t2=float(a.t2),
        nw=int(a.nw),nomega=int(a.nomega),nbath=int(a.nbath),
        bath_mode=(str(a.uniform_bath) if stage=="uniform" else "complex"),
    )


def _checkpoint_path(out,mode,stage):
    return out/f"{stage}_cluster_{mode}_checkpoint.npz"


def _save_checkpoint(path,signature,state,it, *,stage):
    payload=dict(
        signature=json.dumps(signature,sort_keys=True),
        iterations=np.asarray(int(it)),
        G=np.asarray(state.G),
        Sigma_H=np.asarray(state.Sigma_H),
        Sigma_emb=np.asarray(state.Sigma_emb),
        mu=np.asarray(float(state.mu)),
    )
    if stage=="uniform":
        payload.update(
            Sigma_imp=np.asarray(state.Sigma_ED_cluster),
            bath_energies=np.asarray(state.bath.energies),
            bath_couplings=np.asarray(state.bath.couplings),
            bath_fit_error=np.asarray(float(state.bath.fit_error)),
            bath_nfev=np.asarray(int(state.bath.nfev)),
        )
    else:
        payload.update(
            Sigma_imp=np.asarray(state.Sigma_imp),
            bath_energies=np.asarray(state.bath_energies),
            bath_couplings=np.asarray(state.bath_couplings),
        )
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle,**payload)
    tmp.replace(path)


def _load_checkpoint(path,signature,*,stage,grid,nsites,nbath):
    if not path.is_file():
        raise FileNotFoundError(f"requested checkpoint is missing: {path}")
    with np.load(path,allow_pickle=False) as z:
        if str(z["signature"].item())!=json.dumps(signature,sort_keys=True):
            raise ValueError(f"checkpoint model/grid/bath parameters do not match: {path}")
        g=np.asarray(z["G"],complex).copy()
        sh=np.asarray(z["Sigma_H"],complex).copy()
        emb=np.asarray(z["Sigma_emb"],complex).copy()
        imp=np.asarray(z["Sigma_imp"],complex).copy()
        energies=np.asarray(z["bath_energies"],float).copy()
        couplings=np.asarray(z["bath_couplings"],complex).copy()
        mu=float(z["mu"].item())
    if (g.shape!=(grid.nf,1,1,nsites,nsites) or
        sh.shape!=(nsites,nsites) or emb.shape!=g.shape or
        not np.isfinite(mu) or
        any(not np.all(np.isfinite(v)) for v in
            (g,sh,emb,imp,energies,couplings))):
        raise ValueError(f"checkpoint has nonfinite values or invalid lattice shape: {path}")
    if stage=="uniform":
        if (imp.shape!=(grid.nf,6,6) or energies.shape!=(nbath,) or
                couplings.shape!=(6,nbath)):
            raise ValueError("uniform checkpoint impurity/bath shape mismatch")
        from rubycgw.cluster_ed_gw import BathParameters
        with np.load(path,allow_pickle=False) as z:
            bath=BathParameters(
                energies,couplings,float(z["bath_fit_error"].item()),
                int(z["bath_nfev"].item())
            )
        return SimpleNamespace(
            G=g,Sigma_H=sh,Sigma_emb=emb,Sigma_ED_cluster=imp,
            bath=bath,mu=mu
        )
    if (imp.shape!=(nsites//6,grid.nf,6,6) or
            energies.shape!=(nsites//6,nbath) or
            couplings.shape!=(nsites//6,6,nbath)):
        raise ValueError("multicell checkpoint impurity/bath shape mismatch")
    return MultiCellRestart(
        Sigma_H=sh,Sigma_emb=emb,Sigma_imp=imp,G=g,mu=mu,
        bath_energies=energies,bath_couplings=couplings
    )


def _sources(geom,grid,interactions,restart,opts,target,K,h,workers):
    jobs=[]
    for component in K:
        for sign in (1.,-1.):
            shifted=np.asarray(geom.h0,complex).copy()
            shifted[0,0]-=sign*h*component
            jobs.append((
                shifted,geom.vq,interactions,grid,component,sign,h,
                target,opts,restart
            ))
    start=time.perf_counter()
    if workers==1:
        results=list(map(_cluster_source_solve,jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results=list(pool.map(_cluster_source_solve,jobs,chunksize=1))
    errors=[r.get("error") for r in results if not r["ok"]]
    if errors:
        raise RuntimeError("; ".join(errors))
    chi=sum(
        (results[j]["value"]-results[j+1]["value"])/(2*h)
        for j in range(0,len(results),2)
    )
    return float(chi),dict(
        wall_seconds=float(time.perf_counter()-start),
        n_source_solves=len(results),
        source_residual=max(float(r["residual"]) for r in results),
        source_work_seconds=sum(float(r["elapsed"]) for r in results),
        source_iterations=sum(int(r["iterations"]) for r in results),
    )



def _requested_scan_values(a):
    """Return an ordered V grid or None for the legacy single-point run."""
    values=getattr(a,"V_scan",None)
    range_args=(getattr(a,"V_start",None),getattr(a,"V_stop",None),
                getattr(a,"V_step",None))
    using_range=any(x is not None for x in range_args)
    if values is not None and using_range:
        raise ValueError("use --V-scan or --V-start/--V-stop/--V-step, not both")
    if a.V is not None and (values is not None or using_range):
        raise ValueError("--V specifies a single point; omit it when scanning V")
    if values is None and not using_range:
        return None
    if using_range:
        if any(x is None for x in range_args):
            raise ValueError("the V range requires --V-start, --V-stop and --V-step")
        try:
            start,stop,step=(Decimal(str(x)) for x in range_args)
        except InvalidOperation as exc:
            raise ValueError("invalid V range") from exc
        if not all(x.is_finite() for x in (start,stop,step)):
            raise ValueError("V range entries must be finite")
        if not step or (stop-start)*step<0:
            raise ValueError("V step must be nonzero and point towards --V-stop")
        count=int(abs((stop-start)/step))
        if count>10000:
            raise ValueError("refusing more than 10001 requested V points")
        dvalues=[start+i*step for i in range(count+1)]
        if dvalues[-1]!=stop:
            dvalues.append(stop)  # include endpoint after a shorter last step
        values=[float(v) for v in dvalues]
    else:
        values=list(map(float,values))
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("V scan must contain finite values")
    if len(values)>10001:
        raise ValueError("refusing more than 10001 V points")
    if len(values)>1:
        differences=np.diff(values)
        if not (np.all(differences>0) or np.all(differences<0)):
            raise ValueError("V scan must be strictly increasing or decreasing")
    return tuple(float(v) for v in values)


def _scan_point_dir(root,V):
    value=Decimal(str(float(V))).normalize()
    tag=format(value,"f")
    tag=tag.replace("-","m").replace(".","p")
    return Path(root)/("V_"+tag)


def _scan_signature(a,values):
    """Only numerical/physical controls that affect already computed data."""
    return dict(
        version=1,values=list(values),filling=float(a.filling),
        T=float(a.T),t=float(a.t),t1=float(a.t1),t2=float(a.t2),
        K=list(map(float,a.K)),Q=list(map(float,a.Q)),
        modes=list(dict.fromkeys(a.modes)),
        channels=list(dict.fromkeys(a.channels)),
        nw=int(a.nw),nomega=int(a.nomega),nbath=int(a.nbath),
        uniform_bath=str(a.uniform_bath),
        bath_fit_nfreq=int(a.bath_fit_nfreq),
        bath_energy_window=float(a.bath_energy_window),
        bath_coupling_bound=float(a.bath_coupling_bound),
        skip_chi=bool(a.skip_chi),
        chi_method=str(a.cluster_chi_method),
        source_step=float(a.source_step),
        ed_samples=int(a.ed_samples),
        ed_tau_points=int(a.ed_tau_points),
        ed_charge_sector_padding=int(a.ed_charge_sector_padding),
        ed_sector_padding=int(a.ed_sector_padding),
        ed_thermal_eigs=int(a.ed_thermal_eigs),
        ed_krylov_steps=int(a.ed_krylov_steps),
        tau_points=int(a.tau_points),
    )


def _scan_read_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _scan_write_outputs(out,signature,entries,green_rows,chi_rows,status_rows,
                        *,state="running"):
    out=Path(out)
    green_fields=[
        "V","method","status","warm_start_from_V",
        "relative_Frobenius_K","low_frequency_error",
        "trace_tau_relative_L2","mu","zero_field_residual",
        "bath_fit_max","impurity_mismatch_max","seconds"
    ]
    chi_fields=[
        "V","method","channel","status","warm_start_from_V",
        "chi","ED_chi","relative_error_ED","chi_method",
        "chi_wall_seconds","analytic_wall_seconds","max_linear_residual",
        "linear_iterations","finite_source_wall_seconds","source_residual"
    ]
    status_fields=[
        "V","method","status","warm_start_from_V","zero_field_residual",
        "bath_fit_max","chi_complete","directory"
    ]
    _save_csv(out/"scan_green_comparison.csv",green_rows,green_fields)
    _save_csv(out/"scan_chi_comparison.csv",chi_rows,chi_fields)
    _save_csv(out/"scan_status.csv",status_rows,status_fields)
    info=dict(signature=signature,status=state,points=entries)
    tmp=out/"scan_manifest.json.tmp"
    tmp.write_text(json.dumps(_json(info),ensure_ascii=False,indent=2,
                              allow_nan=False),encoding="utf-8")
    tmp.replace(out/"scan_manifest.json")


def _scan_make_plots(out,chi_rows,green_rows,channels,modes):
    """Build plots from numeric, successfully computed entries only."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names=["ED"]+["cluster-"+x for x in modes]
        for ch in channels:
            fig,ax=plt.subplots(figsize=(7,4.5))
            for name in names:
                pts=[]
                for row in chi_rows:
                    if (row["method"]==name and row["channel"]==ch and
                            row["status"]=="ok" and row.get("chi") is not None):
                        pts.append((float(row["V"]),float(row["chi"])))
                if pts:
                    pts.sort()
                    ax.plot([v for v,_ in pts],[y for _,y in pts],
                            marker="o",label=name)
            ax.set(xlabel="V",ylabel=r"$\\chi(Q,0)$",title=f"{ch}, Q-fixed")
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(Path(out)/("scan_chi_"+ch+".png"),dpi=170)
            plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4.5))
        for mode in modes:
            pts=[]
            for row in green_rows:
                if (row["method"]=="cluster-"+mode and
                        row["status"]=="converged" and
                        row.get("relative_Frobenius_K") is not None):
                    pts.append((float(row["V"]),
                                float(row["relative_Frobenius_K"])))
            if pts:
                pts.sort()
                ax.plot([v for v,_ in pts],[y for _,y in pts],
                        marker="o",label=mode)
        ax.set(xlabel="V",ylabel=r"Relative $G(K,i\\omega)$ Frobenius error")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(Path(out)/"scan_green_error.png",dpi=170)
        plt.close(fig)
    except (ImportError,ValueError,OSError) as exc:
        print(f"[fierz-scan] summary plotting failed: {exc}",flush=True)


def run_v_scan(a,values):
    """V scan with strictly certified, mode-specific multicell warm starts.

    Each target-V lattice/cluster GW skeleton is recomputed. The first point
    uses the old initialization. Later points bypass the redundant uniform
    stage when a previous certified complex-bath multicell state exists.
    """
    root=Path(a.out).resolve()
    root.mkdir(parents=True,exist_ok=True)
    modes=list(dict.fromkeys(a.modes))
    channels=list(dict.fromkeys(a.channels))
    sig=_scan_signature(a,values)
    successful_states={}
    successful_V={}
    entries=[];green_rows=[];chi_rows=[];status_rows=[]
    first_unfinished=0
    manifest_path=root/"scan_manifest.json"
    if a.resume_scan:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"--resume-scan requires an existing {manifest_path}"
            )
        previous=json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("signature")!=_json(sig):
            raise ValueError("scan signature differs from recorded sweep")
        for i,entry in enumerate(previous.get("points",[])):
            if i>=len(values) or float(entry["V"])!=values[i]:
                raise ValueError("scan manifest contains a mismatched V order")
            if entry.get("status")!="ok":
                break
            v=values[i]
            folder=_scan_point_dir(root,v)
            if not folder.is_dir():
                raise FileNotFoundError(f"missing completed scan folder: {folder}")
            old_diag=json.loads((folder/"diagnostics.json").read_text(
                encoding="utf-8"
            ))
            if any(
                mode not in old_diag.get("methods",{}) or
                "zero_field_residual" not in old_diag["methods"][mode]
                for mode in modes
            ):
                raise ValueError(f"missing certified cluster metadata in {folder}")
            for mode in modes:
                point_args=copy.copy(a)
                point_args.V=v
                state=_load_checkpoint(
                    _checkpoint_path(folder,mode,"baseline"),
                    _checkpoint_signature(point_args,mode,"baseline"),
                    stage="baseline",
                    grid=MatsubaraGrid(nk1=1,nk2=1,nw=a.nw,
                                       nOmega=a.nomega,T=a.T),
                    nsites=18,nbath=a.nbath
                )
                saved_error=float(old_diag["methods"][mode]["zero_field_residual"])
                if not math.isfinite(saved_error) or saved_error>=a.baseline_tol:
                    raise ValueError(
                        f"previous solution for {mode} at V={v:g} is not certified "
                        f"at current baseline tolerance: {saved_error:g}"
                    )
                successful_states[mode]=state
                successful_V[mode]=v
            for row in _scan_read_csv(folder/"green_comparison.csv"):
                warm=entry.get("warm_start_sources",{}).get(
                    row["method"].replace("cluster-",""),None
                )
                green_rows.append(dict(V=v,**row,warm_start_from_V=warm))
            for row in _scan_read_csv(folder/"chi_comparison.csv"):
                warm=entry.get("warm_start_sources",{}).get(
                    row["method"].replace("cluster-",""),None
                )
                chi_rows.append(dict(V=v,**row,warm_start_from_V=warm))
            status_rows.extend(entry.get("method_statuses",[]))
            entries.append(entry)
            first_unfinished=i+1
        print(
            f"[fierz-scan] resumed {first_unfinished}/{len(values)} complete points; "
            f"next V={values[first_unfinished] if first_unfinished<len(values) else 'none'}",
            flush=True
        )
    _scan_write_outputs(root,sig,entries,green_rows,chi_rows,status_rows)
    for i in range(first_unfinished,len(values)):
        v=float(values[i])
        sub=_scan_point_dir(root,v)
        sub.mkdir(parents=True,exist_ok=True)
        args=copy.copy(a)
        args.V=v
        args.V_scan=None
        args.V_start=args.V_stop=args.V_step=None
        args.resume_scan=False
        args.resume_uniform=False
        args.resume_baseline=False
        args.out=sub
        sources={m:successful_V[m] for m in modes if m in successful_states}
        print(
            f"[fierz-scan] point {i+1}/{len(values)}: V={v:g}, "
            f"mode-specific previous V={sources or 'cold'}",flush=True
        )
        result=main(args,warm_states=successful_states,
                    warm_sources=sources,return_scan_state=True)
        for row in result["green_rows"]:
            mode=row["method"].removeprefix("cluster-")
            green_rows.append(dict(
                V=v,**row,warm_start_from_V=sources.get(mode)
            ))
        for row in result["chi_rows"]:
            mode=row["method"].removeprefix("cluster-")
            chi_rows.append(dict(
                V=v,**row,warm_start_from_V=sources.get(mode)
            ))
        ed_green_ok=any(
            row.get("method")=="ED" and row.get("status")=="reference"
            for row in result["green_rows"]
        )
        ed_chi_ok=bool(a.skip_chi) or all(
            any(row.get("method")=="ED" and row.get("channel")==ch and
                row.get("status")=="ok" for row in result["chi_rows"])
            for ch in channels
        )
        statuses=[]
        for mode in modes:
            state=result["states"].get(mode)
            mode_ok=state is not None and bool(getattr(state,"converged",False))
            valid_chi=bool(a.skip_chi) or all(
                any(row.get("method")=="cluster-"+mode and
                    row.get("channel")==ch and row.get("status")=="ok"
                    for row in result["chi_rows"])
                for ch in channels
            )
            method_status=("ok" if mode_ok and valid_chi else
                           "chi-failed" if mode_ok else "unconverged")
            md=result["diagnostics"].get("methods",{}).get(mode,{})
            entry=dict(
                V=v,method="cluster-"+mode,status=method_status,
                warm_start_from_V=sources.get(mode),
                zero_field_residual=(
                    float(state.final_error) if mode_ok else
                    md.get("baseline_diagnostics",{}).get("residual")),
                bath_fit_max=(
                    float(np.max(state.bath_fit_error)) if mode_ok else None
                ),
                chi_complete=valid_chi,directory=str(sub),
            )
            statuses.append(entry)
            status_rows.append(entry)
            if mode_ok:
                # Only a certified zero-source fixed point becomes a warm seed.
                successful_states[mode]=state
                successful_V[mode]=v
        statuses.insert(0,dict(
            V=v,method="ED",status="ok" if ed_green_ok and ed_chi_ok else "failed",
            warm_start_from_V=None,zero_field_residual=None,
            bath_fit_max=None,chi_complete=ed_chi_ok,directory=str(sub)
        ))
        status_rows.append(statuses[0])
        complete=(ed_green_ok and ed_chi_ok and all(
            entry["status"]=="ok" for entry in statuses[1:]
        ))
        point=dict(
            V=v,status="ok" if complete else "partial",
            directory=str(sub),warm_start_sources=sources,
            ed_green_ok=ed_green_ok,ed_chi_ok=ed_chi_ok,
            method_statuses=statuses
        )
        entries.append(point)
        _scan_write_outputs(
            root,sig,entries,green_rows,chi_rows,status_rows,
            state="running" if complete else "partial"
        )
        if not complete and a.scan_on_failure=="stop":
            print(
                f"[fierz-scan] V={v:g} incomplete; keeping partial per-V "
                f"results and stopping. Use --resume-scan to retry.",
                flush=True
            )
            break
    finished=len(entries)==len(values) and all(
        x["status"]=="ok" for x in entries
    )
    _scan_write_outputs(
        root,sig,entries,green_rows,chi_rows,status_rows,
        state="complete" if finished else "partial"
    )
    _scan_make_plots(root,chi_rows,green_rows,channels,modes)
    print(
        f"[fierz-scan] {len(entries)}/{len(values)} V points attempted; "
        f"status={'complete' if finished else 'partial'}; "
        f"results at {root/'scan_manifest.json'}",flush=True
    )
    return dict(points=entries,completed=finished,output=str(root))


def main(argv=None, *, warm_states=None, warm_sources=None,
         return_scan_state=False):
    a=(argv if isinstance(argv,argparse.Namespace) else parse_args(argv))
    scan_values=_requested_scan_values(a)
    if scan_values is not None:
        if warm_states is not None or return_scan_state:
            raise ValueError("nested V scans are not supported")
        if a.resume_uniform or a.resume_baseline:
            raise ValueError("use --resume-scan, not per-point --resume-uniform/--resume-baseline, for V scans")
        return run_v_scan(a,scan_values)
    if a.V is None or not math.isfinite(float(a.V)):
        raise ValueError("single-point runs require a finite --V")
    warm_states={} if warm_states is None else dict(warm_states)
    warm_sources={} if warm_sources is None else dict(warm_sources)
    if not 0<a.filling<6:
        raise ValueError("filling must be between 0 and 6 spinless fermions per primitive cell")
    if a.nbath<1:
        raise ValueError("--nbath must be positive")
    if (a.bath_energy_window<=0 or a.bath_coupling_bound<=0 or
        a.bath_fit_restarts<0 or a.bath_retry_error<0 or
        a.pulay_history<2 or a.pulay_start<1 or
        a.pulay_step_cap<=0 or a.pulay_reset_growth<=1 or
        a.checkpoint_every<1):
        raise ValueError("invalid bath, Pulay, or checkpoint settings")
    if a.resume_uniform and a.resume_baseline:
        raise ValueError("--resume-uniform and --resume-baseline are mutually exclusive")
    if a.cluster_chi_method=="finite-source" and not a.skip_chi:
        if a.source_workers<1 or a.source_step<=0:
            raise ValueError("finite-source workers and source-step must be positive")
    if a.cluster_chi_method=="analytic" and not a.skip_chi:
        if (a.analytic_tol<=0 or a.analytic_maxiter<1 or
                a.analytic_restart<1 or a.bath_stationarity_tol<=0):
            raise ValueError("invalid analytic cluster chi solver parameters")
    params=RubyParameters(ti=a.t,t1=a.t1,t2=a.t2,V=a.V)
    geom=build_torus_geometry("hex18",params)
    K=geom.validate_momentum(a.K)
    Q=geom.validate_momentum(a.Q)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=a.nw,nOmega=a.nomega,T=a.T)
    target=float(a.filling*geom.ncell)
    modes=list(dict.fromkeys(a.modes))
    channels=list(dict.fromkeys(a.channels))
    out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=True)

    diag={
        "geometry":"hex18-pbc","V":a.V,"filling":a.filling,
        "target_particles":target,"T":a.T,"K":list(K),"Q":list(Q),
        "modes":modes,"channels":channels,"nbath":a.nbath,
        "initial_state":a.initial_state,
        "cluster_chi_method":a.cluster_chi_method,
        "uniform_mixing_method":a.uniform_mixing_method,
        "uniform_bath_mode":a.uniform_bath,
        "baseline_mixing_method":a.baseline_mixing_method,
        "screening_monitor":bool(a.screening_monitor),
        "checkpoint_every":int(a.checkpoint_every),
        "continuation":dict(
            mode="scan-warm-start" if warm_states else "single-point",
            input_sources={str(k):float(v) for k,v in warm_sources.items()},
        ),
        "note":"The ED impurity is always the original six intra-triangle density interactions; only the matched lattice-GW and cluster-GW subtraction vary.",
        "stages":{},"errors":{},"methods":{}
    }
    Gref=None
    ed_chi={}
    tau=np.linspace(0,1./a.T,int(a.tau_points))
    h0K=k_projector(geom,K).conj().T@geom.h0[0,0]@k_projector(geom,K)
    data_G={}
    data_tau={}
    green_rows=[]
    chi_rows=[]
    mode_results={}

    def record_stage(name,fn):
        start=time.perf_counter()
        try:
            result=fn()
            diag["stages"][name]={"status":"ok","seconds":time.perf_counter()-start}
            return result
        except Exception as exc:
            message=f"{type(exc).__name__}: {exc}"
            diag["stages"][name]={"status":"failed","seconds":time.perf_counter()-start,"error":message}
            diag["errors"][name]=message
            print(f"[fierz-cluster] FAILED {name}: {message}",flush=True)
            if a.strict:
                raise
            return None

    def save():
        _save_csv(out/"green_comparison.csv",green_rows,[
            "method","status","relative_Frobenius_K","low_frequency_error",
            "trace_tau_relative_L2","mu","zero_field_residual",
            "bath_fit_max","impurity_mismatch_max","seconds"
        ])
        _save_csv(out/"chi_comparison.csv",chi_rows,[
            "method","channel","status","chi","ED_chi","relative_error_ED",
            "chi_method","chi_wall_seconds","analytic_wall_seconds",
            "max_linear_residual","linear_iterations",
            "finite_source_wall_seconds","source_residual"
        ])
        with (out/"diagnostics.json").open("w",encoding="utf-8") as f:
            json.dump(_json(diag),f,ensure_ascii=False,indent=2,allow_nan=False)
        np.savez_compressed(
            out/"green_curves.npz",
            omega=grid.omega,tau=tau,**data_G,**data_tau
        )
        # Plain CSV traces for Origin, without discarding the complex
        # 6x6 matrix information retained in green_curves.npz.
        iw_fields=["omega"]
        it_fields=["tau","tau_over_beta"]
        for label in ("ED",)+tuple("cluster_"+m for m in modes):
            iw_fields.extend([label+"_Re",label+"_Im"])
            it_fields.append(label)
        iw_rows=[]
        for iw,w in enumerate(grid.omega):
            row={"omega":float(w)}
            for label in ("ED",)+tuple("cluster_"+m for m in modes):
                G=data_G.get("Giw_"+label)
                if G is not None:
                    v=np.trace(G[iw])/6.
                    row[label+"_Re"]=float(v.real)
                    row[label+"_Im"]=float(v.imag)
            iw_rows.append(row)
        _save_csv(out/"green_iomega_trace.csv",iw_rows,iw_fields)
        tau_rows=[]
        for j,tt in enumerate(tau):
            row={"tau":float(tt),"tau_over_beta":float(tt*a.T)}
            for label in ("ED",)+tuple("cluster_"+m for m in modes):
                G=data_tau.get("Gtau_"+label)
                if G is not None:
                    row[label]=float(np.trace(G[j]).real/6.)
            tau_rows.append(row)
        _save_csv(out/"green_tau_trace.csv",tau_rows,it_fields)

    def ed_func():
        N=int(round(target))
        if abs(target-N)<1e-10:
            sol=MomentumTorusED(geom,params,N)
            spectrum=sol.ground_spectrum(a.V,n_eigs=2)
            return sol.thermal_response(
                a.V,a.T,Q,channels,
                n_trace_vectors=a.ed_samples,
                tau_points=a.ed_tau_points,
                energy_shift=float(spectrum.energies[0])
            )
        return grand_canonical_ed_response(
            geom,params,V=a.V,T=a.T,target_particles=target,
            Q=Q,channels=channels,n_trace_vectors=a.ed_samples,
            tau_points=a.ed_tau_points,
            padding=a.ed_charge_sector_padding,use_momentum=True
        )
    ED=record_stage("ED_chi",ed_func) if not a.skip_chi else None
    if ED is not None:
        ed_chi={name:float(ED.susceptibility_matrix[i,i].real)
                for i,name in enumerate(channels)}
        for name in channels:
            chi_rows.append(dict(method="ED",channel=name,status="ok",
                                 chi=ed_chi[name],ED_chi=ed_chi[name],
                                 relative_error_ED=0.))
        diag["ED_chi_matrix"]=ED.susceptibility_matrix
        diag["ED_ensemble"]=getattr(ED,"ensemble","canonical")
        if hasattr(ED,"edge_probability"):
            diag["ED_charge_edge_probability"]=float(ED.edge_probability)

    def ed_green_func():
        return momentum_thermal_lanczos_green(
            geom,params,V=a.V,T=a.T,target_particles=target,
            omega=grid.omega,K=K,
            opts=ThermalLanczosOptions(
                sector_padding=a.ed_sector_padding,
                thermal_eigs=a.ed_thermal_eigs,
                krylov_dim=a.ed_krylov_steps,verbose=a.verbose,
            )
        )
    Gr=record_stage("ED_green",ed_green_func)
    if Gr is not None:
        Gref=np.asarray(Gr.G,complex)
        data_G["Giw_ED"]=Gref
        ref_tau=green_k_tau(Gref,grid.omega,a.T,Gr.mu,h0K,tau)
        data_tau["Gtau_ED"]=ref_tau
        diag["ED_green"]=dict(
            mu=Gr.mu,kept_thermal_weight=Gr.kept_thermal_weight,
            sector_edge_weight=float(
                Gr.sector_probabilities[0]+Gr.sector_probabilities[-1]
            ),sector_numbers=Gr.sector_numbers
        )
        green_rows.append(dict(
            method="ED",status="reference",relative_Frobenius_K=0.,
            low_frequency_error=0.,trace_tau_relative_L2=0.,
            mu=Gr.mu,seconds=diag["stages"]["ED_green"]["seconds"]
        ))

    ints=ruby_cluster_interactions(params)
    for mode in modes:
        print(f"[fierz-cluster] start {mode}: V={a.V:g}, n={a.filling:g}, nbath={a.nbath}",flush=True)
        build_embedding_channels(ints,geom.ncell,mode,geom.vq)
        warm=warm_states.get(mode)
        if warm is not None:
            if (not bool(getattr(warm,"converged",True)) or
                    str(getattr(warm,"gw_mode",mode))!=mode):
                raise ValueError(
                    f"refusing a nonconverged or wrong-Fierz warm seed for {mode}"
                )
            from_v=warm_sources.get(mode)
            diag["methods"].setdefault(mode,{})["warm_start"]=dict(
                used=True,source_V=float(from_v) if from_v is not None else None,
                source_stage="converged-multicell-zero-field",
                destination_V=float(a.V),
                new_V_screening_recomputed=True,
            )
            print(
                f"[fierz-cluster] {mode}: warm-start V={from_v} -> {a.V:g} "
                "(rebuilding lattice/cluster GW at target V)",flush=True
            )
        def initial():
            if a.initial_state=="free":
                return noninteracting_cluster_seed(geom,grid,target)
            from rubycgw.fierz_cluster_embedding import build_embedding_channels
            definitions=build_embedding_channels(ints,geom.ncell,mode,geom.vq)
            from rubycgw.three_torus_interaction_channels import build_torus_interaction_channel
            definition=(build_torus_interaction_channel(geom,a.V,"density")
                        if mode=="density" else definitions.lattice)
            gw=solve_channel_gw_same_torus(
                geom.h0,definition,grid,
                opts=GWOptions(
                    target_filling=target,max_iter=a.gw_max_iter,
                    tol=a.gw_tol,mixing=a.gw_mixing,
                    mixing_method="pulay",
                    momentum_backend="fft",verbose=a.verbose
                )
            )
            diag["methods"].setdefault(mode,{})["seed_GW"]=dict(
                converged=gw.converged,residual=gw.final_error,
                iterations=gw.iterations
            )
            return _fierz_seed(gw,mode,geom,grid,target)
        seed=(None if warm is not None else
              record_stage(mode+"_initial",initial))
        if seed is None and warm is None:
            mode_results[mode]=None
            green_rows.append(dict(method="cluster-"+mode,status="initial-failed"))
            for ch in channels:
                if not a.skip_chi:
                    chi_rows.append(dict(method="cluster-"+mode,channel=ch,
                                         status="initial-failed"))
            save()
            continue

        uniform_path=_checkpoint_path(out,mode,"uniform")
        baseline_path=_checkpoint_path(out,mode,"baseline")
        uniform_sig=_checkpoint_signature(a,mode,"uniform")
        baseline_sig=_checkpoint_signature(a,mode,"baseline")
        baseline_restart=(
            _multi_restart(warm) if warm is not None else (
                _load_checkpoint(
                    baseline_path,baseline_sig,stage="baseline",
                    grid=grid,nsites=geom.nsites,nbath=a.nbath
                ) if a.resume_baseline else None
            )
        )
        uniform_restart=(
            _load_checkpoint(
                uniform_path,uniform_sig,stage="uniform",
                grid=grid,nsites=geom.nsites,nbath=a.nbath
            ) if a.resume_uniform else None
        )
        def uniform():
            sol=uniform_primitive_cluster_ed_gw(
                geom,grid,seed,nbath=a.nbath,
                max_iter=a.uniform_max_iter,tol=a.uniform_tol,
                mixing=a.uniform_mixing,
                mixing_method=a.uniform_mixing_method,
                pulay_history=a.pulay_history,pulay_start=a.pulay_start,
                pulay_step_cap=a.pulay_step_cap,
                pulay_reset_growth=a.pulay_reset_growth,
                bath_complex=(a.uniform_bath=="complex"),
                bath_energy_window=a.bath_energy_window,
                bath_coupling_bound=a.bath_coupling_bound,
                bath_fit_restarts=a.bath_fit_restarts,
                bath_retry_error=a.bath_retry_error,
                initial_channel=a.uniform_seed_channel,
                initial_amplitude=a.uniform_seed_amplitude,
                bath_nfit=a.bath_fit_nfreq,
                bath_max_nfev=a.bath_max_nfev,
                target_particles=target,gw_mode=mode,
                restart=uniform_restart,
                screening_monitor=a.screening_monitor,
                checkpoint_every=a.checkpoint_every,
                checkpoint_callback=lambda it,state:_save_checkpoint(
                    uniform_path,uniform_sig,state,it,stage="uniform"
                ),
                verbose=a.verbose
            )
            diag["methods"].setdefault(mode,{})["uniform_diagnostics"]=dict(
                residual_history=sol.residual_history,
                component_residual_history=sol.component_residual_history,
                bath_history=sol.bath_history,
                bath_diagnostics=sol.bath_diagnostics,
                screening_history=sol.screening_history,
                pulay_fallbacks=sol.pulay_fallbacks,
                pulay_resets=sol.pulay_resets,
                bath_mode=sol.bath_mode,iterations=sol.iterations,
                residual=sol.final_error,
                checkpoint=str(uniform_path),
            )
            if not sol.converged:
                raise RuntimeError(
                    f"uniform cluster {mode} not converged: {sol.final_error:.3e}; "
                    f"restart from {uniform_path} with --resume-uniform"
                )
            return sol
        zero_uniform=(
            None if baseline_restart is not None
            else record_stage(mode+"_uniform",uniform)
        )
        if zero_uniform is None and baseline_restart is None:
            green_rows.append(dict(method="cluster-"+mode,status="uniform-failed"))
            if not a.skip_chi:
                chi_rows.extend(dict(method="cluster-"+mode,channel=ch,
                                    status="uniform-failed") for ch in channels)
            save()
            continue
        def baseline():
            opts=MultiCellEDGWOptions(
                max_iter=a.baseline_max_iter,tol=a.baseline_tol,
                mixing=a.baseline_mixing,nbath=a.nbath,
                mixing_method=a.baseline_mixing_method,
                pulay_history=a.pulay_history,pulay_start=a.pulay_start,
                pulay_step_cap=a.pulay_step_cap,
                pulay_reset_growth=a.pulay_reset_growth,
                bath_fit_restarts=a.bath_fit_restarts,
                bath_retry_error=a.bath_retry_error,
                bath_energy_window=a.bath_energy_window,
                bath_coupling_bound=a.bath_coupling_bound,
                bath_fit_nfreq=a.bath_fit_nfreq,
                bath_fit_max_nfev=a.bath_max_nfev,
                screening_monitor=a.screening_monitor,
                complex_bath=True,verbose=a.verbose,gw_mode=mode
            )
            result=solve_multicell_cluster_ed_gw(
                geom.h0,geom.vq,ints,grid,
                target_particles=target,opts=opts,
                restart=(baseline_restart if baseline_restart is not None else
                         restart_from_uniform_cluster(zero_uniform,geom.ncell)),
                checkpoint_every=a.checkpoint_every,
                checkpoint_callback=lambda it,state:_save_checkpoint(
                    baseline_path,baseline_sig,state,it,stage="baseline"
                ),
            )
            diag["methods"].setdefault(mode,{})["baseline_diagnostics"]=dict(
                residual_history=result.residual_history,
                component_residual_history=result.component_residual_history,
                bath_diagnostics=result.bath_diagnostics,
                screening_history=result.screening_history,
                pulay_fallbacks=result.pulay_fallbacks,
                pulay_resets=result.pulay_resets,
                iterations=result.iterations,residual=result.final_error,
                checkpoint=str(baseline_path),
            )
            if not result.converged:
                raise RuntimeError(
                    f"zero-field complex-bath {mode} residual={result.final_error:.3e}; "
                    f"restart from {baseline_path} with --resume-baseline"
                )
            return result
        base=record_stage(mode+"_zero_field",baseline)
        if base is None:
            green_rows.append(dict(method="cluster-"+mode,status="zero-field-failed"))
            if not a.skip_chi:
                chi_rows.extend(dict(method="cluster-"+mode,channel=ch,
                                    status="zero-field-failed") for ch in channels)
            save()
            continue
        mode_results[mode]=base
        group=diag["methods"].setdefault(mode,{})
        group.update(dict(
            zero_field_residual=base.final_error,
            uniform_residual=(zero_uniform.final_error
                              if zero_uniform is not None else None),
            warm_start_from_V=warm_sources.get(mode) if warm is not None else None,
            bath_fit_error=base.bath_fit_error,
            impurity_mismatch=base.impurity_mismatch,
            gw_mode=base.gw_mode,
            timing=base.timing_seconds
        ))
        Gk=project_k_green(base.G,geom,K)
        data_G["Giw_cluster_"+mode]=Gk
        Tk=green_k_tau(Gk,grid.omega,a.T,base.mu,h0K,tau)
        data_tau["Gtau_cluster_"+mode]=Tk
        err=lowerr=taur=None
        if Gref is not None:
            err=float(np.linalg.norm(Gk-Gref)/max(np.linalg.norm(Gref),1e-30))
            ix=np.argsort(abs(grid.omega))[:min(8,len(grid.omega))]
            lowerr=float(np.linalg.norm(Gk[ix]-Gref[ix])/
                         max(np.linalg.norm(Gref[ix]),1e-30))
            a_trace=np.trace(Tk,axis1=1,axis2=2).real/6
            ref_trace=np.trace(ref_tau,axis1=1,axis2=2).real/6
            taur=float(np.linalg.norm(a_trace-ref_trace)/
                       max(np.linalg.norm(ref_trace),1e-30))
        green_rows.append(dict(
            method="cluster-"+mode,status="converged",
            relative_Frobenius_K=err,low_frequency_error=lowerr,
            trace_tau_relative_L2=taur,mu=float(base.mu),
            zero_field_residual=float(base.final_error),
            bath_fit_max=float(np.max(base.bath_fit_error)),
            impurity_mismatch_max=float(np.max(base.impurity_mismatch)),
            seconds=diag["stages"].get(mode+"_initial",{}).get("seconds",0.)+
                    diag["stages"].get(mode+"_uniform",{}).get("seconds",0.)+
                    diag["stages"][mode+"_zero_field"]["seconds"]
        ))
        save()

        if a.skip_chi:
            continue
        group["chi_method"]=a.cluster_chi_method
        kernel=None
        restart=None
        source_opts=None
        if a.cluster_chi_method=="analytic":
            kernel=record_stage(
                mode+"_analytic_kernel",
                lambda: StaticClusterEDGWLinearizer(
                    geom,params,grid,base,bath_nfit=a.bath_fit_nfreq,
                    bath_stationarity_tol=a.bath_stationarity_tol
                )
            )
            if kernel is None:
                for channel in channels:
                    chi_rows.append(dict(
                        method="cluster-"+mode,channel=channel,
                        status="analytic-kernel-failed",chi_method="analytic"
                    ))
                save()
                continue
        else:
            restart=_multi_restart(base)
            source_opts=MultiCellEDGWOptions(
                max_iter=a.source_max_iter,tol=a.source_tol,
                mixing=a.source_mixing,nbath=a.nbath,
                mixing_method=a.source_mixing_method,
                pulay_history=a.pulay_history,pulay_start=a.pulay_start,
                pulay_step_cap=a.pulay_step_cap,
                pulay_reset_growth=a.pulay_reset_growth,
                bath_fit_restarts=a.bath_fit_restarts,
                bath_retry_error=a.bath_retry_error,
                bath_energy_window=a.bath_energy_window,
                bath_coupling_bound=a.bath_coupling_bound,
                bath_fit_nfreq=a.bath_fit_nfreq,
                bath_fit_max_nfev=a.bath_max_nfev,
                complex_bath=True,verbose=a.verbose,gw_mode=mode
            )
        for channel in channels:
            components=hermitian_source_components(geom,channel,Q)
            if a.cluster_chi_method=="analytic":
                def analytic_channel():
                    start=time.perf_counter()
                    answers=[
                        kernel.response(
                            component,rtol=a.analytic_tol,
                            maxiter=a.analytic_maxiter,
                            restart=a.analytic_restart,
                        )
                        for component in components
                    ]
                    return float(sum(item.chi for item in answers)),dict(
                        wall_seconds=float(time.perf_counter()-start),
                        max_linear_residual=max(item.residual for item in answers),
                        linear_iterations=sum(item.iterations for item in answers),
                        n_source_solves=0,
                        components=[dict(item.__dict__) for item in answers],
                        bath_stationarity=float(kernel.bath_stationarity_max),
                        bath_linear_error=float(kernel.bath_linear_max),
                    )
                result=record_stage(mode+"_chi_"+channel,analytic_channel)
            else:
                result=record_stage(
                    mode+"_chi_"+channel,
                    lambda: _sources(
                        geom,grid,ints,restart,source_opts,target,
                        components,float(a.source_step),a.source_workers
                    )
                )
            if result is None:
                chi_rows.append(dict(
                    method="cluster-"+mode,channel=channel,status="failed",
                    chi_method=a.cluster_chi_method
                ))
            else:
                value,details=result
                ref=ed_chi.get(channel)
                is_analytic=a.cluster_chi_method=="analytic"
                chi_rows.append(dict(
                    method="cluster-"+mode,channel=channel,status="ok",
                    chi=value,ED_chi=ref,chi_method=a.cluster_chi_method,
                    relative_error_ED=(
                        abs(value-ref)/max(abs(ref),1e-12)
                        if ref is not None else None
                    ),
                    chi_wall_seconds=details["wall_seconds"],
                    analytic_wall_seconds=(
                        details["wall_seconds"] if is_analytic else None
                    ),
                    max_linear_residual=(
                        details["max_linear_residual"] if is_analytic else None
                    ),
                    linear_iterations=(
                        details["linear_iterations"] if is_analytic else None
                    ),
                    finite_source_wall_seconds=(
                        None if is_analytic else details["wall_seconds"]
                    ),
                    source_residual=(
                        None if is_analytic else details["source_residual"]
                    )
                ))
                group.setdefault("analytic" if is_analytic else "source",{})[
                    channel
                ]=details
            save()

    save()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if Gr is not None:
            fig,ax=plt.subplots(figsize=(8,4.5))
            ax.plot(tau*a.T,np.trace(ref_tau,axis1=1,axis2=2).real/6,
                    label="ED",lw=2.2)
            for mode in modes:
                if "Gtau_cluster_"+mode in data_tau:
                    y=np.trace(data_tau["Gtau_cluster_"+mode],
                               axis1=1,axis2=2).real/6
                    ax.plot(tau*a.T,y,label="cluster "+mode)
            ax.set(xlabel=r"$\tau/\beta$",ylabel=r"$\mathrm{Tr}\,G(K,\tau)/6$",
                   title=f"Hex18 PBC, V={a.V:g}, n={a.filling:g}, K={tuple(K)}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out/"green_tau_comparison.png",dpi=180)
            plt.close(fig)
        if Gr is not None:
            fig,ax=plt.subplots(figsize=(8,4.5))
            positive=np.asarray(grid.omega)>0.
            ax.plot(grid.omega[positive],
                    np.trace(Gref[positive],axis1=1,axis2=2).imag/6,
                    label="ED",lw=2.2)
            for mode in modes:
                item=data_G.get("Giw_cluster_"+mode)
                if item is not None:
                    ax.plot(grid.omega[positive],
                            np.trace(item[positive],axis1=1,axis2=2).imag/6,
                            label="cluster "+mode)
            ax.set(xlabel=r"$\omega_n$",
                   ylabel=r"$\mathrm{Im}\,\mathrm{Tr}G(K,i\omega_n)/6$",
                   title=f"Hex18 PBC, V={a.V:g}, n={a.filling:g}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out/"green_iomega_comparison.png",dpi=180)
            plt.close(fig)
        if chi_rows and not a.skip_chi:
            fig,ax=plt.subplots(figsize=(max(8,2*len(channels)),4.7))
            x=np.arange(len(channels))
            for j,method in enumerate(["ED"]+["cluster-"+m for m in modes]):
                vals=[]
                for ch in channels:
                    row=next((r for r in chi_rows if
                              r["method"]==method and r["channel"]==ch),{})
                    vals.append(np.nan if row.get("chi") is None else row["chi"])
                ax.bar(x+(j-1.5)*.18,vals,.18,label=method)
            ax.set_xticks(x,channels)
            ax.set_ylabel(r"$\chi(Q,0)$")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out/"chi_comparison.png",dpi=180)
            plt.close(fig)
    except Exception as exc:
        print(f"[fierz-cluster] plot failed: {exc}",flush=True)
    print(f"[fierz-cluster] results: {out/'green_comparison.csv'}, "
          f"{out/'chi_comparison.csv'}",flush=True)
    if return_scan_state:
        return dict(
            diagnostics=diag,states=mode_results,
            green_rows=green_rows,chi_rows=chi_rows,
            output=str(out),
        )
    return diag


if __name__=="__main__":
    main()
