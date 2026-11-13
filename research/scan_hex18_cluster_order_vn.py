#!/usr/bin/env python3
"""Find zero-source CO/LC/bond-order solutions on the *periodic hex18* Ruby torus.

One correlated six-site ED impurity per primitive cell (three impurities),
with the full 18x18 lattice GW propagator and Fierz-matched local subtraction.
All order patterns are applied ONLY to the initial self-energy. The subsequent
nonlinear solver runs at exactly zero external source, permitting finite
Q=(1/3,1/3) translation-breaking self-energies and unequal impurity baths.

Scan V (inner loop) and primitive-cell filling n (outer loop). Each candidate
branch uses its own last certified solution at the preceding V, otherwise the
same V of the preceding n, otherwise a zero-field symmetric bootstrap. Each
(V,n) has ONE 18-site ED Green reference reused by all candidate branches;
no susceptibilities are computed.

Free energies are diagnostics of converged branches, not proof of the exact
thermodynamic ground state. The existing physical density-GW Luttinger-Ward
functional is only evaluated for the density Fierz embedding, and endpoints
with large bath/Gimp-Gc mismatches are not certified for phase ranking.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

if __package__ in (None,""):
    root=str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0,root)

import numpy as np

from rubycgw.cluster_ed_gw import ruby_cluster_interactions
from rubycgw.cluster_ed_gw_multicell import (
    MultiCellEDGWOptions,MultiCellRestart,
    solve_multicell_cluster_ed_gw,restart_from_uniform_cluster,
)
from rubycgw.cluster_ed_gw_multicell_free_energy import (
    evaluate_multicell_free_energy,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.hex18_order import (
    FAMILIES,DEFAULT_FAMILIES,QK,seed_matrix,order_observables,
)
from rubycgw.momentum_ed_green import momentum_thermal_lanczos_green
from rubycgw.thermal_lanczos_green import ThermalLanczosOptions
from rubycgw.supercell_gw_fast import _solve_mu_matrix_fast
from rubycgw.supercell_gw_split import one_body_density_matrix_tail
from rubycgw.three_torus_benchmark import (
    build_torus_geometry,noninteracting_cluster_seed,
    uniform_primitive_cluster_ed_gw,project_k_green,k_projector,
)


def _write_csv(path,rows,cols):
    path=Path(path)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=cols,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json(value):
    if isinstance(value,dict):
        return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [_json(v) for v in value]
    if isinstance(value,np.ndarray):
        return _json(value.tolist())
    if isinstance(value,(np.bool_,bool)):
        return bool(value)
    if isinstance(value,(np.integer,int)):
        return int(value)
    if isinstance(value,(np.floating,float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,(complex,np.complexfloating)):
        return dict(real=float(value.real),imag=float(value.imag))
    return value


def _save_json(path,data):
    path=Path(path)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(_json(data),ensure_ascii=False,
                                   indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def _tag(value):
    return format(Decimal(str(float(value))).normalize(),"f").replace("-","m").replace(".","p")


def _grid_explicit_or_range(explicit,start,stop,step,name):
    if explicit is not None:
        if any(x is not None for x in (start,stop,step)):
            raise ValueError(f"{name}: choose explicit values OR start/stop/step")
        values=[float(x) for x in explicit]
    elif any(x is not None for x in (start,stop,step)):
        if any(x is None for x in (start,stop,step)):
            raise ValueError(f"{name}: start, stop and step must all be set")
        a,b,d=[Decimal(str(x)) for x in (start,stop,step)]
        if not all(x.is_finite() for x in (a,b,d)):
            raise ValueError(f"{name}: finite range values required")
        if d==0 or (b-a)*d<0:
            raise ValueError(f"{name}: invalid step direction")
        steps=int(abs((b-a)/d))
        if steps>2000:
            raise ValueError(f"{name}: too many scan points")
        values=[float(a+i*d) for i in range(steps+1)]
        if Decimal(str(values[-1]))!=b and abs(values[-1]-float(b))>1e-11:
            values.append(float(b))
    else:
        values=[0.,.5,1.] if name=="V" else [2.2]
    if not values or not all(np.isfinite(values)):
        raise ValueError(f"{name}: values must be nonempty and finite")
    if len(values)>2001 or len(set(values))!=len(values):
        raise ValueError(f"{name}: duplicate/too many values")
    dif=np.diff(values)
    if dif.size and not (np.all(dif>0) or np.all(dif<0)):
        raise ValueError(f"{name}: grid must be monotonic")
    return tuple(values)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--V-scan",nargs="+",type=float,default=None)
    p.add_argument("--V-start",type=float)
    p.add_argument("--V-stop",type=float)
    p.add_argument("--V-step",type=float)
    p.add_argument("--fillings",nargs="+",type=float,default=None,
                   help="particles per six-site primitive cell, e.g. 2 2.1 2.2")
    p.add_argument("--n-start",type=float)
    p.add_argument("--n-stop",type=float)
    p.add_argument("--n-step",type=float)
    p.add_argument("--T",type=float,default=.125)
    p.add_argument("--t",type=float,default=1.)
    p.add_argument("--t1",type=float,default=-.5)
    p.add_argument("--t2",type=float,default=.5)
    p.add_argument("--K",nargs=2,type=float,default=QK,
                   help="primitive K at which G(K,iomega) is compared to ED; does NOT set the order wave vector Q")
    p.add_argument("--modes",nargs="+",choices=("density","pseudospin","bj"),
                   default=("density",),
                   help="Fierz embedding schemes; physical V and ED impurity remain unchanged. Only density has compatible free-energy ranking")
    p.add_argument("--candidates",nargs="+",choices=FAMILIES,
                   default=DEFAULT_FAMILIES)
    p.add_argument("--seed-amplitudes",nargs="+",type=float,
                   default=(.08,),help="initial self-energy perturbations, NOT physical sources")
    p.add_argument("--random-seed",type=int,default=27)
    p.add_argument("--nw",type=int,default=19)
    p.add_argument("--nomega",type=int,default=10)
    p.add_argument("--cluster-partition",choices=("cell6","triangle3"),
                   default="cell6",
                   help="cell6: 3 six-site impurities; triangle3: 6 independent three-site A/B impurities")
    p.add_argument("--nbath",type=int,default=None,
                   help="bath orbitals per impurity; default 6 for cell6, 3 for triangle3")
    p.add_argument("--max-iter",type=int,default=140)
    p.add_argument("--tol",type=float,default=3e-5)
    p.add_argument("--mixing",type=float,default=.25)
    p.add_argument("--mixing-method",choices=("linear","pulay"),default="pulay")
    p.add_argument("--pulay-history",type=int,default=6)
    p.add_argument("--pulay-start",type=int,default=3)
    p.add_argument("--pulay-step-cap",type=float,default=2.)
    p.add_argument("--bath-fit-nfreq",type=int,default=12)
    p.add_argument("--bath-max-nfev",type=int,default=250)
    p.add_argument("--bath-fit-restarts",type=int,default=1)
    p.add_argument("--bath-energy-window",type=float,default=6.)
    p.add_argument("--bath-coupling-bound",type=float,default=4.)
    p.add_argument("--bootstrap",choices=("uniform","free"),default="uniform",
                   help="first point without any converged prior solution")
    p.add_argument("--uniform-max-iter",type=int,default=100)
    p.add_argument("--uniform-mixing",type=float,default=.2)
    p.add_argument("--free-energy-mismatch-max",type=float,default=.05)
    p.add_argument("--free-energy-bath-max",type=float,default=.15)
    p.add_argument("--free-energy-degeneracy-tol",type=float,default=1e-4)
    p.add_argument("--no-free-energy",action="store_true")
    p.add_argument("--order-threshold",type=float,default=2e-3)
    p.add_argument("--ed-sector-padding",type=int,default=2)
    p.add_argument("--ed-thermal-eigs",type=int,default=12)
    p.add_argument("--ed-krylov-steps",type=int,default=160)
    p.add_argument("--no-ed",action="store_true",
                   help="debug only: skip requested ED Green comparison")
    p.add_argument("--tau-points",type=int,default=81)
    p.add_argument("--checkpoint-every",type=int,default=5)
    p.add_argument("--plot-branches",action="store_true",
                   help="save density/bond/current triptych for each converged branch")
    p.add_argument("--plot-formats",nargs="+",choices=("png","svg","pdf"),
                   default=("png",),help="triptych file formats")
    p.add_argument("--plot-annotate",action="store_true",
                   help="label native site indices on real-space triptych")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--verbose",action="store_true")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    if a.nbath is None:
        a.nbath=3 if a.cluster_partition=="triangle3" else 6
    a.V_values=_grid_explicit_or_range(
        a.V_scan,a.V_start,a.V_stop,a.V_step,"V"
    )
    a.n_values=_grid_explicit_or_range(
        a.fillings,a.n_start,a.n_stop,a.n_step,"n"
    )
    if (not 0<a.T or any(not 0<n<6 for n in a.n_values) or
        a.nw<1 or a.nomega<0 or a.nbath<1 or a.nbath>10 or
        not 0<a.mixing<=1 or a.tol<=0 or a.max_iter<1 or
        a.uniform_max_iter<1 or not 0<a.uniform_mixing<=1 or
        a.bath_fit_nfreq<1 or a.bath_max_nfev<1 or
        a.bath_fit_restarts<0 or a.bath_energy_window<=0 or
        a.bath_coupling_bound<=0 or a.order_threshold<=0 or
        a.checkpoint_every<1 or a.pulay_history<2 or a.pulay_start<1 or
        a.pulay_step_cap<=0 or a.free_energy_mismatch_max<=0 or
        a.free_energy_bath_max<=0 or a.free_energy_degeneracy_tol<0 or
        a.ed_sector_padding<0 or a.ed_thermal_eigs<1 or
        a.ed_krylov_steps<1 or a.tau_points<3):
        raise ValueError("invalid physical, ED or solver parameters")
    if len(a.seed_amplitudes)==0 or any(
        not np.isfinite(x) or x<=0 for x in a.seed_amplitudes
    ):
        raise ValueError("--seed-amplitudes must be positive finite numbers")
    if len(set(a.candidates))!=len(a.candidates):
        raise ValueError("duplicate candidate names")
    return a


def _variants(candidates,amps):
    out=[]
    for name in candidates:
        if name=="symmetric":
            out.append((name,"symmetric",0.))
        else:
            for amp in amps:
                key=f"{name}_h{_tag(amp)}"
                out.append((key,name,float(amp)))
    return out


def _run_signature(a):
    """Config affecting physical results and candidate branch identification."""
    return dict(
        version=2,V=list(a.V_values),fillings=list(a.n_values),
        cluster_partition=a.cluster_partition,
        T=a.T,t=a.t,t1=a.t1,t2=a.t2,K=list(a.K),
        modes=list(a.modes),candidates=list(a.candidates),
        seed_amplitudes=list(a.seed_amplitudes),random_seed=a.random_seed,
        nw=a.nw,nomega=a.nomega,nbath=a.nbath,
        bath_fit_nfreq=a.bath_fit_nfreq,
        bath_energy_window=a.bath_energy_window,
        bath_coupling_bound=a.bath_coupling_bound,
        order_threshold=a.order_threshold,no_ed=a.no_ed,
        no_free_energy=a.no_free_energy,
        free_energy_mismatch_max=a.free_energy_mismatch_max,
        free_energy_bath_max=a.free_energy_bath_max,
        free_energy_degeneracy_tol=a.free_energy_degeneracy_tol,
        ed_sector_padding=a.ed_sector_padding,
        ed_thermal_eigs=a.ed_thermal_eigs,
        ed_krylov_steps=a.ed_krylov_steps,
        tau_points=a.tau_points,
    )


def _restart_copy(state):
    if isinstance(state,MultiCellRestart):
        imp=state.Sigma_imp
    else:
        imp=state.Sigma_ED_cluster
    return MultiCellRestart(
        Sigma_H=np.asarray(state.Sigma_H,complex).copy(),
        Sigma_emb=np.asarray(state.Sigma_emb,complex).copy(),
        Sigma_imp=np.asarray(imp,complex).copy(),
        G=np.asarray(state.G,complex).copy(),mu=float(state.mu),
        bath_energies=np.asarray(state.bath_energies,float).copy(),
        bath_couplings=np.asarray(state.bath_couplings,complex).copy()
    )


def _inject_seed(restart,K,amplitude,h0,grid,target):
    """One-time self-energy seed; target h0 is untouched in all iterations."""
    state=_restart_copy(restart)
    if amplitude:
        state.Sigma_emb[:,0,0]+=float(amplitude)*np.asarray(K)[None]
    state.mu,state.G,_,_=_solve_mu_matrix_fast(
        h0,state.Sigma_H,state.Sigma_emb,grid,float(target),
        float(state.mu),1e-10,100
    )
    return state


def _fresh_bootstrap(geom,params,grid,target,mode,a):
    from rubycgw.fierz_cluster_embedding import build_embedding_channels,cluster_gw_map
    bg=noninteracting_cluster_seed(geom,grid,float(target))
    if a.bootstrap=="uniform" and a.cluster_partition=="cell6":
        u=uniform_primitive_cluster_ed_gw(
            geom,grid,bg,nbath=a.nbath,gw_mode=mode,
            max_iter=a.uniform_max_iter,tol=max(float(a.tol),1e-4),
            mixing=a.uniform_mixing,mixing_method="pulay",
            bath_complex=True,
            bath_energy_window=a.bath_energy_window,
            bath_coupling_bound=a.bath_coupling_bound,
            bath_nfit=a.bath_fit_nfreq,
            bath_max_nfev=a.bath_max_nfev,
            bath_fit_restarts=a.bath_fit_restarts,
            target_particles=float(target),verbose=a.verbose,
        )
        if u.converged:
            return restart_from_uniform_cluster(u,geom.ncell),"uniform-converged"
        print(
            f"[hex18-order] uniform bootstrap for mode={mode} failed "
            f"residual={u.final_error:.3e}; using free fixed-point initializer",
            flush=True
        )
    # Triangle3 uses six three-orbital impurity seeds from the same full
    # 18-site free Green function. No six-site ED warm-up is needed.
    from rubycgw.fierz_cluster_embedding import (
        triangle_cluster_interactions,triangle_embedding_channel,
    )
    cluster_size=3 if a.cluster_partition=="triangle3" else 6
    nimp=geom.nsites//cluster_size
    rho=one_body_density_matrix_tail(
        bg.G,grid,geom.h0,bg.mu,bg.Sigma_H
    )
    defs=build_embedding_channels(
        ruby_cluster_interactions(params),geom.ncell,mode,geom.vq
    )
    sigma_imp=[]
    interactions=ruby_cluster_interactions(params)
    for cell in range(nimp):
        sl=slice(cluster_size*cell,cluster_size*(cell+1))
        terms=(interactions if cluster_size==6 else
               triangle_cluster_interactions(interactions,cell%2))
        local_def=(defs if cluster_size==6 else
                   triangle_embedding_channel(defs,cell%2))
        sc,_,_=cluster_gw_map(
            bg.G[:,0,0,sl,sl],rho[0,0,sl,sl],
            terms,grid,local_def
        )
        sigma_imp.append(sc)
    energies=np.linspace(-a.bath_energy_window/2,
                         a.bath_energy_window/2,a.nbath)+bg.mu
    couplings=np.zeros((nimp,cluster_size,a.nbath),complex)
    for p in range(a.nbath):
        couplings[:,p%cluster_size,p]=1e-3
    return MultiCellRestart(
        Sigma_H=np.asarray(bg.Sigma_H,complex).copy(),
        Sigma_emb=np.asarray(bg.Sigma_GW,complex).copy(),
        Sigma_imp=np.asarray(sigma_imp,complex),
        G=np.asarray(bg.G,complex).copy(),mu=float(bg.mu),
        bath_energies=np.tile(energies[None],(nimp,1)),
        bath_couplings=couplings,
    ),"free"


def _save_state(path,state,meta,rho,Gk,Gmix):
    payload=dict(
        meta=json.dumps(meta,sort_keys=True),G=state.G,
        Sigma_H=state.Sigma_H,Sigma_emb=state.Sigma_emb,
        Sigma_imp=state.Sigma_ED_cluster,mu=float(state.mu),
        bath_energies=state.bath_energies,
        bath_couplings=state.bath_couplings,
        rho=rho,Gk=Gk,G_K_KplusQ=Gmix,
        residual_history=state.residual_history,
        final_error=float(state.final_error),converged=bool(state.converged),
        bath_fit_error=state.bath_fit_error,
        impurity_mismatch=state.impurity_mismatch,
    )
    path=Path(path)
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**payload)
    tmp.replace(path)


def _load_state(path,mode,key,grid,geom,nbath,tol,cluster_size=6):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z["meta"].item()))
        if (meta["mode"]!=mode or meta["candidate"]!=key or
                meta.get("cluster_partition","cell6")!=
                ("triangle3" if cluster_size==3 else "cell6")):
            raise ValueError("wrong candidate/mode/partition in saved ordered-state checkpoint")
        if (not bool(z["converged"].item()) or
                float(z["final_error"].item())>=float(tol)):
            return None
        sh=np.asarray(z["Sigma_H"],complex)
        emb=np.asarray(z["Sigma_emb"],complex)
        imp=np.asarray(z["Sigma_imp"],complex)
        G=np.asarray(z["G"],complex)
        energies=np.asarray(z["bath_energies"],float)
        couplings=np.asarray(z["bath_couplings"],complex)
        mu=float(z["mu"].item())
    nimp=geom.nsites//int(cluster_size)
    if (G.shape!=(grid.nf,1,1,geom.nsites,geom.nsites) or
        emb.shape!=G.shape or sh.shape!=(geom.nsites,geom.nsites) or
        imp.shape!=(nimp,grid.nf,cluster_size,cluster_size) or
        energies.shape!=(nimp,nbath) or
        couplings.shape!=(nimp,cluster_size,nbath) or
        not np.isfinite(mu) or
        not all(np.all(np.isfinite(x)) for x in
                (G,sh,emb,imp,energies,couplings))):
        raise ValueError("corrupt or mismatched ordered-state checkpoint")
    return MultiCellRestart(
        Sigma_H=sh,Sigma_emb=emb,Sigma_imp=imp,G=G,mu=mu,
        bath_energies=energies,bath_couplings=couplings
    )


def _nearest_warm(cache,ni,vi,mode,key):
    """Prefer preceding V at same n; else preceding n at same V.

    A last earlier certified V at the current n is a fallback when an
    intermediate V failed; it is always recorded, not hidden as adjacent.
    """
    for j in range(vi-1,-1,-1):
        item=cache.get((ni,j,mode,key))
        if item is not None:
            return item,(ni,j)
    for j in range(ni-1,-1,-1):
        item=cache.get((j,vi,mode,key))
        if item is not None:
            return item,(j,vi)
    return None,None


def _try_ed_green(geom,params,grid,K,target,a,path):
    path=Path(path)
    if path.exists() and a.resume:
        with np.load(path,allow_pickle=False) as z:
            if int(z["nf"].item())==grid.nf:
                return np.asarray(z["Gk"],complex),"cached"
    if a.no_ed:
        return None,"disabled"
    result=momentum_thermal_lanczos_green(
        geom,params,V=params.V,T=grid.T,target_particles=target,
        omega=grid.omega,K=K,
        opts=ThermalLanczosOptions(
            sector_padding=a.ed_sector_padding,
            thermal_eigs=a.ed_thermal_eigs,
            krylov_dim=a.ed_krylov_steps,
            verbose=a.verbose,
        )
    )
    Gk=np.asarray(result.G,complex)
    np.savez_compressed(
        path,Gk=Gk,omega=grid.omega,nf=grid.nf,
        mu=float(result.mu),
        kept_thermal_weight=float(result.kept_thermal_weight),
        sector_numbers=np.asarray(result.sector_numbers),
    )
    return Gk,"ok"


def _green_errors(Gk,Gref,omega):
    if Gref is None:
        return dict(g_error=np.nan,g_low_error=np.nan)
    refnorm=max(float(np.linalg.norm(Gref)),1e-30)
    idx=np.argsort(abs(omega))[:min(8,len(omega))]
    return dict(
        g_error=float(np.linalg.norm(Gk-Gref)/refnorm),
        g_low_error=float(
            np.linalg.norm(Gk[idx]-Gref[idx])/
            max(float(np.linalg.norm(Gref[idx])),1e-30)
        )
    )


def _classify_winner(rows,mode,free_energy_tol):
    """Only reliable, distinct converged endpoints are eligible for ranking."""
    eligible=[r for r in rows if r["mode"]==mode and
              bool(r.get("free_energy_reliable",False)) and
              np.isfinite(float(r.get("F_per_pc",np.nan)))]
    if not eligible:
        return dict(phase="unresolved",winner="",F_min=np.nan,
                    F_second=np.nan,free_energy_gap=np.nan,
                    near_degenerate=False,n_reliable=0)
    eligible.sort(key=lambda r:r["F_per_pc"])
    distinct=[]
    for row in eligible:
        if any(
            row.get("order_signature") is not None and
            old.get("order_signature") is not None and
            float(np.linalg.norm(
                np.asarray(row["order_signature"],float)-
                np.asarray(old["order_signature"],float)
            ))<1e-3 and
            float(abs(row["F_per_pc"]-old["F_per_pc"]))<1e-4
            for old in distinct
        ):
            continue
        distinct.append(row)
    first=distinct[0]
    second=distinct[1] if len(distinct)>1 else None
    gap=(float(second["F_per_pc"]-first["F_per_pc"])
         if second else np.nan)
    degenerate=(second is not None and gap<=float(free_energy_tol))
    phase=(f"near-degenerate:{first['phase_family']}|{second['phase_family']}"
           if degenerate and second["phase_family"]!=first["phase_family"]
           else first["phase_family"])
    return dict(
        phase=phase,winner=first["candidate"],
        F_min=first["F_per_pc"],
        F_second=second["F_per_pc"] if second else np.nan,
        free_energy_gap=gap,
        near_degenerate=bool(degenerate),n_reliable=len(distinct)
    )


def _plot_phase(out,point_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vals=sorted(set(r["V"] for r in point_rows))
        fills=sorted(set(r["filling"] for r in point_rows))
        labels=("unresolved","symmetric","CO","LC",
                "real_bond_order","mixed_CO_LC")
        colors=np.full((len(fills),len(vals)),0,dtype=int)
        for row in point_rows:
            if row["mode"]!="density":
                continue
            phase=row["phase"]
            if phase in labels:
                colors[fills.index(row["filling"]),vals.index(row["V"])]=labels.index(phase)
        fig,ax=plt.subplots(figsize=(max(6,len(vals)*.85),max(3,len(fills)*.7+2)))
        im=ax.imshow(colors,origin="lower",aspect="auto",interpolation="nearest",
                     vmin=-.5,vmax=len(labels)-.5)
        ax.set(xticks=range(len(vals)),xticklabels=[f"{x:g}" for x in vals],
               yticks=range(len(fills)),yticklabels=[f"{x:g}" for x in fills],
               xlabel="V",ylabel="filling n",
               title="Hex18 cluster ED+GW ordered branches (density mode)")
        bar=fig.colorbar(im,ax=ax,ticks=range(len(labels)))
        bar.ax.set_yticklabels(labels)
        fig.tight_layout()
        fig.savefig(Path(out)/"phase_map.png",dpi=170)
        plt.close(fig)
    except (ImportError,ValueError,OSError) as exc:
        print(f"[hex18-order] plot skipped: {exc}",flush=True)


def _plot_branch(a,ordersfile):
    """Plotting errors must not change a certified ED+GW fixed-point result."""
    if not a.plot_branches:
        return
    try:
        from research.plot_hex18_cluster_order import plot_order_triptych
        images=plot_order_triptych(
            ordersfile,formats=a.plot_formats,annotate=a.plot_annotate
        )
        if a.verbose:
            print("[hex18-order] rendered: "+", ".join(map(str,images)),
                  flush=True)
    except Exception as exc:
        print(
            f"[hex18-order] plot warning for {ordersfile}: "
            f"{type(exc).__name__}: {exc}",flush=True
        )


def run(a):
    signature=_run_signature(a)
    out=Path(a.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/"scan_manifest.json"
    if a.resume:
        if not manifest.exists():
            raise FileNotFoundError("--resume requires previous scan_manifest.json")
        data=json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("signature")!=_json(signature):
            raise ValueError("resume scan physical signature mismatch")
    else:
        if manifest.exists():
            raise FileExistsError(
                "output contains a previous scan; use --resume or a fresh --out"
            )
    variants=_variants(a.candidates,a.seed_amplitudes)
    branch_rows=[]
    point_rows=[]
    cache={}
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=a.nw,nOmega=a.nomega,T=a.T)
    for ni,n in enumerate(a.n_values):
        for vi,V in enumerate(a.V_values):
            params=RubyParameters(ti=a.t,t1=a.t1,t2=a.t2,V=float(V))
            geom=build_torus_geometry("hex18",params)
            K=geom.validate_momentum(a.K)
            target=float(n*geom.ncell)
            point_dir=out/f"n_{_tag(n)}"/f"V_{_tag(V)}"
            point_dir.mkdir(parents=True,exist_ok=True)
            print(f"[hex18-order] partition={a.cluster_partition} "
                  f"n={n:g} V={V:g}, "
                  f"branches={len(variants)*len(a.modes)}",flush=True)
            edfile=point_dir/"ed_Gk.npz"
            try:
                Gref,ed_status=_try_ed_green(
                    geom,params,grid,K,target,a,edfile
                )
            except Exception as exc:
                Gref=None
                ed_status=f"failed:{type(exc).__name__}:{exc}"
                print(f"[hex18-order] ED reference failed: {exc}",flush=True)
            per_point=[]
            cold_by_mode={}
            for mode in a.modes:
                for key,name,amplitude in variants:
                    statefile=point_dir/f"{mode}__{key}.npz"
                    rowfile=point_dir/f"{mode}__{key}.json"
                    cached_state=None
                    if a.resume and statefile.exists() and rowfile.exists():
                        cached_state=_load_state(
                            statefile,mode,key,grid,geom,a.nbath,a.tol,
                            cluster_size=(3 if a.cluster_partition=="triangle3" else 6)
                        )
                    if cached_state is not None:
                        row=json.loads(rowfile.read_text(encoding="utf-8"))
                        row["status"]="converged"
                        # ED may have failed in an earlier interrupted run.
                        # A newly computed ED reference must refresh G errors.
                        with np.load(statefile,allow_pickle=False) as state:
                            stored_Gk=np.asarray(state["Gk"],complex)
                        row.update(_green_errors(
                            stored_Gk,Gref,grid.omega
                        ))
                        row["ed_status"]=ed_status
                        _save_json(rowfile,row)
                        _plot_branch(
                            a,point_dir/f"{mode}__{key}_orders.npz"
                        )
                        cache[(ni,vi,mode,key)]=cached_state
                        branch_rows.append(row)
                        per_point.append(row)
                        print(f"[hex18-order] restored n={n:g} V={V:g} "
                              f"{mode}/{key}",flush=True)
                        continue
                    prior,origin=_nearest_warm(cache,ni,vi,mode,key)
                    origin_type="branch"
                    if prior is None:
                        # A symmetric solution at this same point (or its
                        # earlier parameter neighbors) is a useful branching
                        # seed, but it is never regarded as this branch's own
                        # converged continuation result.
                        prior=cache.get((ni,vi,mode,"symmetric"))
                        origin_type="same-point-symmetric"
                    if prior is None:
                        prior,origin=_nearest_warm(
                            cache,ni,vi,mode,"symmetric"
                        )
                        origin_type="previous-symmetric"
                    if prior is None:
                        if mode not in cold_by_mode:
                            try:
                                cold_by_mode[mode]=_fresh_bootstrap(
                                    geom,params,grid,target,mode,a
                                )
                            except Exception as exc:
                                print(f"[hex18-order] bootstrap failed for {mode}: "
                                      f"{type(exc).__name__}: {exc}",flush=True)
                                cold_by_mode[mode]=(None,"bootstrap-failed")
                        prior,origin_type=cold_by_mode[mode]
                    from_n=(a.n_values[origin[0]] if origin is not None else None)
                    from_V=(a.V_values[origin[1]] if origin is not None else None)
                    row=dict(
                        V=float(V),filling=float(n),mode=mode,candidate=key,
                        cluster_partition=a.cluster_partition,
                        seed_family=name,seed_amplitude=float(amplitude),
                        warm_source=origin_type,warm_from_V=from_V,
                        warm_from_n=from_n,status="failed",
                        ed_status=ed_status,
                        zero_source=True,
                        phase_family="unresolved",F_per_pc=np.nan,
                        free_energy_reliable=False,
                        g_error=np.nan,g_low_error=np.nan,
                        final_error=np.nan,bath_fit_max=np.nan,
                        gimp_gc_max=np.nan,iterations=0,
                    )
                    if prior is None:
                        row["error"]="no usable initial seed"
                        branch_rows.append(row);per_point.append(row)
                        _save_json(rowfile,row)
                        continue
                    start=time.perf_counter()
                    try:
                        direction=seed_matrix(
                            geom,params,name,random_seed=a.random_seed
                        )
                        restart=_inject_seed(
                            prior,direction,amplitude,geom.h0,grid,target
                        )
                        opts=MultiCellEDGWOptions(
                            max_iter=a.max_iter,tol=a.tol,mixing=a.mixing,
                            mixing_method=a.mixing_method,
                            pulay_history=a.pulay_history,
                            pulay_start=a.pulay_start,
                            pulay_step_cap=a.pulay_step_cap,
                            nbath=a.nbath,
                            cluster_size=(3 if a.cluster_partition=="triangle3" else 6),
                            bath_fit_nfreq=a.bath_fit_nfreq,
                            bath_fit_max_nfev=a.bath_max_nfev,
                            bath_fit_restarts=a.bath_fit_restarts,
                            bath_energy_window=a.bath_energy_window,
                            bath_coupling_bound=a.bath_coupling_bound,
                            complex_bath=True,gw_mode=mode,
                            verbose=a.verbose,screening_monitor=True,
                        )
                        result=solve_multicell_cluster_ed_gw(
                            geom.h0,geom.vq,ruby_cluster_interactions(params),
                            grid,target_particles=target,opts=opts,
                            restart=restart,
                        )
                        row["final_error"]=float(result.final_error)
                        row["iterations"]=int(result.iterations)
                        row["bath_fit_max"]=float(
                            np.max(result.bath_fit_error)
                        )
                        row["gimp_gc_max"]=float(
                            np.max(result.impurity_mismatch)
                        )
                        if not result.converged:
                            row["status"]="unconverged"
                            row["error"]=(
                                f"zero-field self-consistency residual="
                                f"{result.final_error:.3e}"
                            )
                        else:
                            rho=one_body_density_matrix_tail(
                                result.G,grid,geom.h0,
                                result.mu,result.Sigma_H
                            )[0,0]
                            rho=.5*(rho+rho.conj().T)
                            obs=order_observables(
                                rho,geom,params,threshold=a.order_threshold
                            )
                            occupations=np.asarray(obs["density"],float).reshape(
                                geom.ncell,2,3
                            ).sum(axis=-1)
                            total_filling=float(occupations.sum())
                            row["N_lattice"]=total_filling
                            row["filling_error"]=total_filling-target
                            row["triangle_A_occupations"]=occupations[:,0].tolist()
                            row["triangle_B_occupations"]=occupations[:,1].tolist()
                            Gk=project_k_green(result.G,geom,K)
                            # Nonzero at finite-Q order: the off-diagonal
                            # primitive momentum Green block K -> K+Q.
                            KplusQ=geom.validate_momentum(
                                (np.asarray(K)+np.asarray(QK))%1.
                            )
                            UK=k_projector(geom,K)
                            UQ=k_projector(geom,KplusQ)
                            Gmix=np.einsum(
                                "ia,nij,jb->nab",UK.conj(),
                                result.G[:,0,0],UQ,optimize=True
                            )
                            Gmix_norm=float(
                                np.linalg.norm(Gmix)/
                                max(float(np.linalg.norm(Gk)),1e-30)
                            )
                            errors=_green_errors(Gk,Gref,grid.omega)
                            row["G_K_KplusQ_norm"]=Gmix_norm
                            row.update(errors)
                            # Duplicate seeds may converge to the same ordered
                            # endpoint; compare normal density matrices, not
                            # their seed names, when ranking free-energy minima.
                            upper=rho[np.triu_indices(geom.nsites)]
                            row["order_signature"]=np.round(
                                np.concatenate((upper.real,upper.imag)),5
                            ).tolist()
                            row.update(dict(
                                status="converged",
                                phase_family=obs["phase_family"],
                                dominant_channel=obs["dominant_channel"],
                                dominant_amplitude=obs["dominant_amplitude"],
                                co_amplitude=obs["co_amplitude"],
                                lc_amplitude=obs["lc_amplitude"],
                                bond_anisotropy=obs["bond_anisotropy"],
                                density_range=obs["density_range"],
                                mu=float(result.mu),
                                amplitudes=obs["amplitudes"],
                            ))
                            if mode=="density" and not a.no_free_energy:
                                try:
                                    free=evaluate_multicell_free_energy(
                                        result,geom.h0,geom.vq,
                                        ruby_cluster_interactions(params),
                                        grid,target_particles=target
                                    )
                                    row["F_per_pc"]=float(
                                        free.helmholtz_free_energy_per_primitive_cell
                                    )
                                    row["free_energy_reliable"]=bool(
                                        row["gimp_gc_max"]<=a.free_energy_mismatch_max
                                        and row["bath_fit_max"]<=a.free_energy_bath_max
                                        and np.isfinite(row["F_per_pc"])
                                    )
                                    row["free_energy_reconstructed_mismatch_max"]=float(
                                        np.max(free.gimp_reconstruction_mismatch)
                                    )
                                except Exception as exc:
                                    row["free_energy_error"]=str(exc)
                            # Save only a converged physical zero-source state.
                            _save_state(
                                statefile,result,
                                dict(mode=mode,candidate=key,V=V,
                                     filling=n,source=origin_type,
                                     cluster_partition=a.cluster_partition),
                                rho,Gk,Gmix,
                            )
                            np.savez_compressed(
                                point_dir/f"{mode}__{key}_orders.npz",
                                rho=rho,density=obs["density"],
                                triangle_occupations=occupations,
                                total_particles=total_filling,
                                target_particles=target,
                                bond_i=[x["i"] for x in obs["bond_rows"]],
                                bond_j=[x["j"] for x in obs["bond_rows"]],
                                bond_current=[x["current"] for x in obs["bond_rows"]],
                                bond_real=[x["real_bond"] for x in obs["bond_rows"]],
                                Gk=Gk,G_K_KplusQ=Gmix,
                                K=np.asarray(K),KplusQ=np.asarray(KplusQ),
                                omega=grid.omega,
                            )
                            _write_csv(
                                point_dir/f"{mode}__{key}_bonds.csv",
                                obs["bond_rows"],
                                ["i","j","t_real","t_imag",
                                 "real_bond","current","triangle"],
                            )
                            cache[(ni,vi,mode,key)]=_restart_copy(result)
                    except Exception as exc:
                        row["error"]=f"{type(exc).__name__}: {exc}"
                        print(f"[hex18-order] failed {mode}/{key}: {exc}",
                              flush=True)
                    row["seconds"]=float(time.perf_counter()-start)
                    branch_rows.append(row)
                    per_point.append(row)
                    _save_json(rowfile,row)
                    if row["status"]=="converged":
                        _plot_branch(
                            a,point_dir/f"{mode}__{key}_orders.npz"
                        )
                    print(
                        f"[hex18-order] {mode}/{key} {row['status']}: "
                        f"res={row['final_error']:.3e}, phase={row['phase_family']}, "
                        f"Gerr={row['g_error']:.3e}",
                        flush=True
                    )
            for mode in a.modes:
                choice=_classify_winner(
                    per_point,mode,a.free_energy_degeneracy_tol
                )
                point_rows.append(dict(
                    V=float(V),filling=float(n),mode=mode,
                    cluster_partition=a.cluster_partition,
                    ed_status=ed_status,
                    n_converged=sum(x["mode"]==mode and
                                    x["status"]=="converged"
                                    for x in per_point),
                    n_attempted=sum(x["mode"]==mode for x in per_point),
                    **choice,
                ))
            _flush(out,signature,branch_rows,point_rows)
    _plot_phase(out,point_rows)
    print(f"[hex18-order] completed {len(point_rows)} mode/grid points; "
          f"outputs in {out}",flush=True)
    return branch_rows,point_rows


BRANCH_COLS=[
    "V","filling","mode","cluster_partition",
    "candidate","seed_family","seed_amplitude",
    "status","phase_family","dominant_channel","dominant_amplitude",
    "co_amplitude","lc_amplitude","bond_anisotropy","density_range",
    "g_error","g_low_error","G_K_KplusQ_norm",
    "mu","N_lattice","filling_error",
    "triangle_A_occupations","triangle_B_occupations",
    "F_per_pc","free_energy_reliable",
    "final_error","bath_fit_max","gimp_gc_max","iterations","seconds",
    "warm_source","warm_from_V","warm_from_n","ed_status","error",
]
PHASE_COLS=[
    "V","filling","mode","cluster_partition",
    "phase","winner","F_min","F_second",
    "free_energy_gap","near_degenerate","n_reliable",
    "n_converged","n_attempted","ed_status"
]


def _flush(out,signature,branches,points):
    _write_csv(out/"branches.csv",branches,BRANCH_COLS)
    _write_csv(out/"phase_summary.csv",points,PHASE_COLS)
    _save_json(out/"scan_manifest.json",dict(
        signature=signature,branch_count=len(branches),points=points
    ))


def main(argv=None):
    a=parse_args(argv)
    return run(a)


if __name__=="__main__":
    main()
