#!/usr/bin/env python3
"""Raw-branch VMC + symmetry PAV diagnostics for large Ruby supercells.

Purpose
-------
Track competing branches cheaply across parameters without expensive
point-group VAP.

Per seed the default workflow is

  Raw VAP
    -> Raw state observables/maps
    -> Tint PAV
    -> Tint x C3v(A1/A2) PAV, when the finite torus supports ordinary C3v

No C3v VAP is performed.

Optionally, --tint-vap-iter N>0 adds a modest 3-image Tint VAP starting from
the optimized Raw branch, followed by the same A1/A2 PAV diagnostics.

Optimized branches (Raw and optional Tint-VAP) are characterized by measured
one-point CO/BOW/LC order parameters and receive an automatic phase label.
PAV states are used only for projected-energy shifts and do not define phases.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NATIVE=REPO/"ed_benchmark"/"native_ed"
sys.path.insert(0,str(NATIVE))

import geometry as native_geometry

from run_benchmark import atomic_json, make_netket_hamiltonian
from vmc_ansatz import jastrow_channels, bond_backflow_channels
from supercell_ansatz import (
    seed_bloch_blocks,
    build_supercell_netket_model,
)
from symmetry_projection import (
    internal_translation_site_maps,
    internal_c3v_space_group_site_maps,
    internal_gamma_c3v_characters,
    c3v_irrep_characters,
)
from run_supercell_large import (
    _stage,
    _load_params,
    _write_csv,
    _current_operators,
)


def _requested_irreps(spec):
    if spec=="both":
        return ["A1","A2"]
    if spec=="all":
        return ["A1","A2","E"]
    return [spec]



def _seed_bstar_reference(g,nf,l1,l2,seed_amplitude):
    """Calibrate the sign of the Ruby star BOW to the sd/isd seed convention."""
    from benchmark_observables import ruby_star_formfactor
    f=np.asarray(ruby_star_formfactor(g),float)
    out={}
    for name in ("sd","isd"):
        dat=seed_bloch_blocks(g,nf,l1,l2,name,seed_amplitude)
        Q=np.asarray(dat["Q0_full"],complex)
        C=Q@Q.conj().T
        bond=np.asarray([
            2.0*np.real(C[int(e.i),int(e.j)]) for e in g.edges
        ])
        out[name]=float(np.dot(f,bond)/len(g.edges))
    return out


def _classify_branch(row, bstar_ref, *, co_thr, bow_thr, current_thr):
    """Classify an optimized unprojected/Tint-VAP branch from measured orders.

    Classification deliberately uses one-point broken-symmetry observables.
    Two-point S_CO/S_LC remain reported as correlation diagnostics but do not
    by themselves name the Raw phase.
    """
    co=float(row.get("CO_onepoint_rms") or 0.0)
    b=float(row.get("B_star_mean") or 0.0)
    js=float(row.get("J_same_mean") or 0.0)
    jo=float(row.get("J_opposite_mean") or 0.0)
    es=float(row.get("J_same_mean_error") or 0.0)
    eo=float(row.get("J_opposite_mean_error") or 0.0)

    active=[]
    scores={}

    if co>=co_thr:
        active.append("CO")
        scores["CO"]=co/max(co_thr,1e-15)

    if abs(b)>=bow_thr:
        sdv=float(bstar_ref.get("sd",0.0))
        isdv=float(bstar_ref.get("isd",0.0))
        if sdv*isdv<0:
            bow_label=(
                "SD" if np.sign(b)==np.sign(sdv) else "ISD"
            )
        else:
            bow_label="BOW_star_pos" if b>=0 else "BOW_star_neg"
        active.append(bow_label)
        scores[bow_label]=abs(b)/max(bow_thr,1e-15)

    if abs(js)>=current_thr:
        active.append("LC_same")
        scores["LC_same"]=abs(js)/max(current_thr,1e-15)
    if abs(jo)>=current_thr:
        active.append("LC_opposite")
        scores["LC_opposite"]=abs(jo)/max(current_thr,1e-15)

    label="normal" if not active else "+".join(active)
    dominant=(
        "normal" if not scores
        else max(scores,key=scores.get)
    )
    return {
        "phase_label":label,
        "dominant_order":dominant,
        "CO_order_amplitude":co,
        "B_star_order_amplitude":b,
        "J_same_order_amplitude":js,
        "J_opposite_order_amplitude":jo,
        "J_same_snr":(
            abs(js)/es if es>0 else None
        ),
        "J_opposite_snr":(
            abs(jo)/eo if eo>0 else None
        ),
        "classification_CO_threshold":float(co_thr),
        "classification_BOW_threshold":float(bow_thr),
        "classification_current_threshold":float(current_thr),
        "seed_sd_Bstar_reference":float(bstar_ref.get("sd",np.nan)),
        "seed_isd_Bstar_reference":float(bstar_ref.get("isd",np.nan)),
    }


def _annotate_pav_energy(row, source_row):
    row["PAV_source_stage"]=source_row["stage"]
    row["PAV_source_phase_label"]=source_row.get("phase_label")
    row["PAV_delta_E"]=float(row["mc_E"])-float(source_row["mc_E"])
    row["PAV_delta_E_per_site"]=row["PAV_delta_E"]/float(row["Ns"])
    e0=float(source_row.get("mc_E_error") or 0.0)
    e1=float(row.get("mc_E_error") or 0.0)
    comb=(e0*e0+e1*e1)**0.5
    row["PAV_delta_E_combined_error"]=comb
    row["PAV_delta_E_sigma"]=(
        row["PAV_delta_E"]/comb if comb>0 else None
    )
    return row



def _local_h1_values(vstate,H,configs,*,config_chunk=256):
    """Local energy E_L(x)=(H psi)(x)/psi(x) for arbitrary configurations."""
    x=np.asarray(configs,float)
    if x.ndim!=2:
        x=x.reshape((-1,x.shape[-1]))
    out=np.empty(len(x),np.complex128)
    step=max(1,int(config_chunk))
    for lo in range(0,len(x),step):
        xx=x[lo:lo+step]
        xp,mels=H.get_conn_padded(xx)
        xp=np.asarray(xp,float)
        mels=np.asarray(mels,np.complex128)
        if xp.ndim!=3 or mels.ndim!=2:
            raise RuntimeError(
                "Unexpected get_conn_padded shapes: "
                f"x'={xp.shape}, mels={mels.shape}"
            )
        b,k=mels.shape
        logx=np.asarray(vstate.log_value(xx),np.complex128).reshape(-1)
        valid=np.abs(mels)>0
        vals=np.zeros((b,k),np.complex128)
        if np.any(valid):
            flat=xp.reshape((b*k,xp.shape[-1]))
            ids=np.flatnonzero(valid.reshape(-1))
            logp=np.asarray(
                vstate.log_value(flat[ids]),np.complex128
            ).reshape(-1)
            parent=np.repeat(logx,k)[ids]
            vals.reshape(-1)[ids]=(
                mels.reshape(-1)[ids]*np.exp(logp-parent)
            )
        out[lo:lo+b]=np.sum(vals,axis=1)
    return out


def _local_h1_h2_values(
    vstate,H,configs,*,root_chunk=16,config_chunk=256
):
    """Return H psi / psi and H^2 psi / psi on sampled root configurations.

    H^2 is evaluated as
        (H^2 psi)(x)/psi(x)
        = sum_y H_xy psi(y)/psi(x) * E_L(y).
    Only connected y with nonzero matrix elements are evaluated.  Unique
    first-shell configurations are reused within each root chunk.
    """
    x=np.asarray(configs,float)
    if x.ndim!=2:
        x=x.reshape((-1,x.shape[-1]))
    e1=np.empty(len(x),np.complex128)
    e2=np.empty(len(x),np.complex128)
    step=max(1,int(root_chunk))
    for lo in range(0,len(x),step):
        xx=x[lo:lo+step]
        xp,mels=H.get_conn_padded(xx)
        xp=np.asarray(xp,float)
        mels=np.asarray(mels,np.complex128)
        if xp.ndim!=3 or mels.ndim!=2:
            raise RuntimeError(
                "Unexpected get_conn_padded shapes: "
                f"x'={xp.shape}, mels={mels.shape}"
            )
        b,k=mels.shape
        logx=np.asarray(vstate.log_value(xx),np.complex128).reshape(-1)
        valid=np.abs(mels)>0
        ib,ik=np.nonzero(valid)
        if len(ib)==0:
            e1[lo:lo+b]=0.0
            e2[lo:lo+b]=0.0
            continue

        y=xp[ib,ik]
        logy=np.asarray(vstate.log_value(y),np.complex128).reshape(-1)
        ratio=mels[ib,ik]*np.exp(logy-logx[ib])

        # Many roots share connected configurations.  Reusing them is
        # especially helpful for the projected multi-image PAV model.
        yu,inv=np.unique(y,axis=0,return_inverse=True)
        e1u=_local_h1_values(
            vstate,H,yu,config_chunk=config_chunk
        )
        e1y=e1u[inv]

        a=np.zeros(b,np.complex128)
        b2=np.zeros(b,np.complex128)
        np.add.at(a,ib,ratio)
        np.add.at(b2,ib,ratio*e1y)
        e1[lo:lo+b]=a
        e2[lo:lo+b]=b2
    return e1,e2


def _lanczos_from_moments(m1,m2,m3,m4,*,tol=1e-12):
    """One Lanczos/Ritz step from spectral moments <H^n>, n=1..4."""
    m1=float(np.real(m1))
    m2=float(np.real(m2))
    m3=float(np.real(m3))
    m4=float(np.real(m4))
    var0=m2-m1*m1
    scale=max(1.0,abs(m2),m1*m1)
    if var0<0 and abs(var0)<=tol*scale:
        var0=0.0
    if var0<0:
        raise RuntimeError(
            f"Negative PAV energy variance from MC moments: {var0:.6e}"
        )
    if var0<=tol*scale:
        return {
            "lanczos_improved":False,
            "lanczos_krylov_dim":1,
            "lanczos_energy_before":m1,
            "lanczos_energy_after":m1,
            "lanczos_delta_E":0.0,
            "lanczos_alpha_H":0.0,
            "lanczos_residual_norm_before":float(np.sqrt(max(var0,0.0))),
            "lanczos_variance_before":var0,
            "lanczos_variance_after":var0,
            "lanczos_variance_after_raw":var0,
            "lanczos_krylov_h00":m1,
            "lanczos_krylov_h01":0.0,
            "lanczos_krylov_h11":None,
        }

    sigma=float(np.sqrt(var0))
    a1=(m3-2.0*m1*m2+m1**3)/var0
    K=np.asarray([[m1,sigma],[sigma,a1]],float)
    evals,vecs=np.linalg.eigh(K)
    c=np.asarray(vecs[:,0],float)
    e1=float(evals[0])

    # <H^2> in the orthonormal Krylov basis
    h2_01=(m3-m1*m2)/sigma
    h2_11=(m4-2.0*m1*m3+m1*m1*m2)/var0
    K2=np.asarray([[m2,h2_01],[h2_01,h2_11]],float)
    second=float(c@K2@c)
    var1_raw=float(second-e1*e1)
    var1=max(0.0,var1_raw)

    denom=c[0]-c[1]*m1/sigma
    alpha=(
        None if abs(denom)<=1e-14
        else float((c[1]/sigma)/denom)
    )
    return {
        "lanczos_improved":True,
        "lanczos_krylov_dim":2,
        "lanczos_energy_before":m1,
        "lanczos_energy_after":e1,
        "lanczos_delta_E":e1-m1,
        "lanczos_alpha_H":alpha,
        "lanczos_residual_norm_before":sigma,
        "lanczos_variance_before":var0,
        "lanczos_variance_after":var1,
        "lanczos_variance_after_raw":var1_raw,
        "lanczos_krylov_h00":m1,
        "lanczos_krylov_h01":sigma,
        "lanczos_krylov_h11":float(a1),
    }


def _chain_blocks(values,sample_shape):
    """Reduce sample estimators to approximately independent chain/block means."""
    values=np.asarray(values)
    lead=tuple(int(x) for x in sample_shape[:-1])
    if int(np.prod(lead))!=len(values):
        raise RuntimeError(
            f"sample shape {sample_shape} is incompatible with {len(values)} values"
        )
    arr=values.reshape(lead)
    if len(lead)>=2 and lead[0]>=2:
        return arr.reshape((lead[0],-1)).mean(axis=1)
    flat=values.reshape(-1)
    nblock=min(32,max(2,len(flat)//32))
    if len(flat)<2*nblock:
        return flat
    nuse=(len(flat)//nblock)*nblock
    return flat[:nuse].reshape((nblock,-1)).mean(axis=1)


def _jackknife_lanczos(e1,e2,sample_shape):
    """Chain/block jackknife for the derived one-step Lanczos energy."""
    z1=np.asarray(e1,np.complex128)
    z2=np.asarray(e2,np.complex128)
    estimators=[
        z1,
        np.abs(z1)**2,
        np.conjugate(z1)*z2,
        np.abs(z2)**2,
    ]
    blocks=[_chain_blocks(v,sample_shape) for v in estimators]
    nb=min(len(x) for x in blocks)
    if nb<2:
        return {}
    blocks=[x[:nb] for x in blocks]

    leave=[]
    for ib in range(nb):
        mm=[]
        for x in blocks:
            mm.append((np.sum(x)-x[ib])/(nb-1))
        try:
            info=_lanczos_from_moments(*mm)
        except RuntimeError:
            continue
        leave.append([
            info["lanczos_energy_after"],
            info["lanczos_delta_E"],
            info["lanczos_variance_before"],
            info["lanczos_variance_after_raw"],
        ])
    if len(leave)<2:
        return {}
    q=np.asarray(leave,float)
    mean=np.mean(q,axis=0)
    err=np.sqrt((len(q)-1)/len(q)*np.sum((q-mean)**2,axis=0))
    return {
        "lanczos_jackknife_blocks":int(len(q)),
        "lanczos_energy_after_error":float(err[0]),
        "lanczos_delta_E_error":float(err[1]),
        "lanczos_variance_before_error":float(err[2]),
        "lanczos_variance_after_error":float(err[3]),
    }


def _make_pav_eval_state(
    *,
    hi,graph,seed_data,params,shell_mats,bf_mats,use_backflow,
    maps,chars,n_samples,n_discard,n_chains,sweep_size,
    sampler_chunk_size,chunk_size,rng_seed,
):
    """Rebuild the fixed PAV wavefunction as an MCState for Lanczos moments."""
    import netket as nk
    model=build_supercell_netket_model(
        params["blocks"],
        seed_data["sc_coords"],
        seed_data["internal_sites"],
        seed_data["kpoints"],
        shell_mats,
        use_jastrow=True,
        backflow_matrices=(bf_mats if use_backflow else None),
        initial_v=params.get("v",None),
        initial_eta=params.get("eta",None),
        point_site_maps=maps,
        point_characters=chars,
        time_reversal_even=False,
        sparse_correlators=True,
        packed_orbitals=True,
    )
    sampler_kwargs={
        "graph":graph,
        "n_chains":int(n_chains),
        "sweep_size":int(sweep_size),
    }
    if sampler_chunk_size is not None:
        sampler_kwargs["chunk_size"]=int(sampler_chunk_size)
    sampler=nk.sampler.MetropolisFermionHop(hi,**sampler_kwargs)
    state=nk.vqs.MCState(
        sampler,model,
        n_samples=int(n_samples),
        n_discard_per_chain=int(n_discard),
        chunk_size=chunk_size,
        seed=int(rng_seed),
        sampler_seed=int(rng_seed)+104729,
    )
    state.sample()
    return state


def _pav_lanczos_row(
    parent_row, *,
    hi,H_fast,graph,seed_data,params,shell_mats,bf_mats,use_backflow,
    maps,chars,args,rng_seed,sweep_size,directory,
):
    """Stochastic one-Lanczos-step energy/variance diagnostic after PAV."""
    nsamp=(
        int(args.lanczos_samples)
        if args.lanczos_samples is not None
        else int(args.eval_samples)
    )
    vstate=_make_pav_eval_state(
        hi=hi,graph=graph,seed_data=seed_data,params=params,
        shell_mats=shell_mats,bf_mats=bf_mats,use_backflow=use_backflow,
        maps=maps,chars=chars,n_samples=nsamp,
        n_discard=args.n_discard,n_chains=args.n_chains,
        sweep_size=sweep_size,
        sampler_chunk_size=args.sampler_chunk_size,
        chunk_size=args.chunk_size,rng_seed=rng_seed,
    )
    samples=np.asarray(vstate.samples)
    flat=samples.reshape((-1,samples.shape[-1]))
    e1,e2=_local_h1_h2_values(
        vstate,H_fast,flat,
        root_chunk=args.lanczos_root_chunk,
        config_chunk=args.lanczos_config_chunk,
    )

    m1=np.mean(e1)
    m2=np.mean(np.abs(e1)**2)
    m2_alt=np.mean(e2)
    m3=np.mean(np.conjugate(e1)*e2)
    m4=np.mean(np.abs(e2)**2)
    info=_lanczos_from_moments(m1,m2,m3,m4)
    info.update(_jackknife_lanczos(e1,e2,samples.shape))

    de_err=info.get("lanczos_delta_E_error")
    info["lanczos_delta_E_sigma"]=(
        info["lanczos_delta_E"]/de_err
        if de_err is not None and de_err>0 else None
    )
    info.update({
        "lanczos_samples":int(len(flat)),
        "lanczos_m1":float(np.real(m1)),
        "lanczos_m1_imag":float(np.imag(m1)),
        "lanczos_m2":float(np.real(m2)),
        "lanczos_m2_from_H2_local":float(np.real(m2_alt)),
        "lanczos_m2_from_H2_local_imag":float(np.imag(m2_alt)),
        "lanczos_m2_consistency":float(np.real(m2_alt)-np.real(m2)),
        "lanczos_m3":float(np.real(m3)),
        "lanczos_m3_imag":float(np.imag(m3)),
        "lanczos_m4":float(np.real(m4)),
        "lanczos_parent_reported_E":float(parent_row["mc_E"]),
        "lanczos_parent_recheck_delta_E":(
            float(np.real(m1))-float(parent_row["mc_E"])
        ),
        "lanczos_estimator":"MC spectral moments m1..m4; 2D Krylov Rayleigh-Ritz",
        "lanczos_state_materialized":False,
    })

    row=dict(parent_row)
    row.update(info)
    row["stage"]=str(parent_row["stage"])+"+1LS"
    row["lanczos_parent_stage"]=parent_row["stage"]
    row["mc_E"]=float(info["lanczos_energy_after"])
    row["mc_E_error"]=info.get("lanczos_energy_after_error")
    row["mc_energy_variance"]=float(info["lanczos_variance_after"])
    row["mc_energy_variance_per_site"]=(
        float(info["lanczos_variance_after"])/float(row["Ns"])
    )
    row["orders_measured"]=False
    row["maps_saved"]=False

    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/"summary.json",row)
    return row


def parser():
    p=argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--L1",type=int,default=2)
    p.add_argument("--L2",type=int,default=2)
    p.add_argument("--nf",type=int,default=None)
    p.add_argument("--t",type=float,default=1.0)
    p.add_argument("--t1",type=float,default=0.5)
    p.add_argument("--t2",type=float,default=-0.5)
    p.add_argument("--V",type=float,default=5.5)
    p.add_argument(
        "--seeds",nargs="+",
        default=["normal","sd","isd","lc_same","lc_opposite"],
    )
    p.add_argument("--seed-amplitude",type=float,default=0.08)
    p.add_argument(
        "--correlator",choices=["sj","backflow"],default="backflow"
    )
    p.add_argument(
        "--jastrow-mode",choices=["bond_aware","distance"],
        default="bond_aware",
    )
    p.add_argument("--jastrow-shells",type=int,default=4)

    p.add_argument("--raw-iter",type=int,default=150)
    p.add_argument(
        "--tint-vap-iter",type=int,default=0,
        help="Optional 3-image Tint VAP iterations. 0 disables Tint VAP.",
    )
    p.add_argument("--raw-learning-rate",type=float,default=0.03)
    p.add_argument("--tint-learning-rate",type=float,default=0.01)

    p.add_argument(
        "--c3v-irrep",
        choices=["A1","A2","E","both","all"],
        default="both",
        help="C3v PAV sectors. both=A1+A2; all=A1+A2+E.",
    )
    p.add_argument(
        "--spatial-symmetry",
        choices=["auto","c3v","none"],
        default="auto",
        help="auto skips C3v PAV on finite tori without ordinary C3.",
    )

    p.add_argument("--n-samples",type=int,default=2048)
    p.add_argument(
        "--eval-samples",type=int,default=None,
        help="Samples for final E/CO/LC/maps. Default=--n-samples.",
    )
    p.add_argument(
        "--pav-warmup-samples",type=int,default=256,
        help="Cheap samples used only to initialise PAV MCState before final evaluation.",
    )
    p.add_argument("--n-discard",type=int,default=8)
    p.add_argument("--n-chains",type=int,default=32)
    p.add_argument(
        "--sweep-size",type=int,default=None,
        help="Raw/Tint-VAP sweep size. Default=Nf.",
    )
    p.add_argument(
        "--pav-sweep-size",type=int,default=None,
        help="PAV evaluation sweep size. Default=max(2,Nf//4).",
    )
    p.add_argument("--sampler-chunk-size",type=int,default=None)
    p.add_argument("--chunk-size",type=int,default=256)
    p.add_argument("--map-edge-chunk",type=int,default=12)
    p.add_argument(
        "--lanczos-samples",type=int,default=None,
        help="PAV samples for stochastic 1-step Lanczos moments. Default=--eval-samples.",
    )
    p.add_argument(
        "--lanczos-root-chunk",type=int,default=16,
        help="Root-sample chunk for H^2 local estimator.",
    )
    p.add_argument(
        "--lanczos-config-chunk",type=int,default=256,
        help="Connected-configuration chunk for local-energy evaluation.",
    )
    p.add_argument(
        "--skip-lanczos",action="store_true",
        help="Disable the PAV -> one-Lanczos-step energy/variance diagnostic.",
    )
    p.add_argument(
        "--classify-co-threshold",type=float,default=0.03,
        help="Raw one-point CO RMS threshold used only for automatic branch labels.",
    )
    p.add_argument(
        "--classify-bow-threshold",type=float,default=0.03,
        help="Absolute Ruby-star bond-order threshold used only for automatic branch labels.",
    )
    p.add_argument(
        "--classify-current-threshold",type=float,default=0.05,
        help="Absolute loop-current one-point threshold used only for automatic branch labels.",
    )

    p.add_argument(
        "--qgt",choices=["auto","pytree","onthefly","dense"],default="auto"
    )
    p.add_argument("--sr-solver",choices=["cg","gmres"],default="cg")
    p.add_argument("--qgt-chunk-size",type=int,default=128)
    p.add_argument("--cg-tol",type=float,default=1e-5)
    p.add_argument("--cg-maxiter",type=int,default=100)
    p.add_argument("--diag-shift",type=float,default=0.05)

    p.add_argument(
        "--reuse-raw",action="store_true",
        help="Reuse large_raw_<correlator>__<seed>/final_params.npz.",
    )
    p.add_argument(
        "--reuse-tint-vap",action="store_true",
        help="Reuse large_tint_<correlator>__Gamma__<seed>/final_params.npz.",
    )
    p.add_argument("--rng-seed",type=int,default=2026)
    p.add_argument(
        "--output",default="results/raw_pav_2x2_V5p5"
    )
    return p


def main():
    a=parser().parse_args()
    if a.L1<1 or a.L2<1:
        raise ValueError("L1/L2 must be positive")

    out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=True)

    g=native_geometry.make_ring3_supercell(
        a.L1,a.L2,t=a.t,t1=a.t1,t2=a.t2
    )
    nf=6*a.L1*a.L2 if a.nf is None else int(a.nf)
    if not 0<nf<g.Ns:
        raise ValueError("nf must satisfy 0<nf<Ns")

    import jax
    jax.config.update("jax_enable_x64",True)

    hi,_,H_fast,graph=make_netket_hamiltonian(g,nf,a.V)
    current_ops=_current_operators(hi,g)
    _,_,shell_mats,_=jastrow_channels(
        g,mode=a.jastrow_mode,n_shells=a.jastrow_shells
    )
    _,bf_mats,_=bond_backflow_channels(g)
    use_backflow=(a.correlator=="backflow")

    tint_ops=internal_translation_site_maps(g)
    tint_maps=np.asarray([p for _,p in tint_ops],np.int32)
    tint_chars=np.ones(3,np.complex128)

    spatial_maps=None
    spatial_chars={}
    requested=_requested_irreps(a.c3v_irrep)
    spatial_effective=False
    if a.spatial_symmetry!="none":
        try:
            ops=internal_c3v_space_group_site_maps(g)
            spatial_maps=np.asarray([p for _,p in ops],np.int32)
            spatial_effective=True
        except RuntimeError as exc:
            if a.spatial_symmetry=="c3v":
                raise
            print(
                f"[PAV] ordinary C3v unavailable for L1={a.L1},L2={a.L2}; "
                "only Raw/Tint diagnostics will be produced.",
                flush=True,
            )

    if spatial_effective:
        for ir in requested:
            if ir in ("A1","A2"):
                spatial_chars[ir]=internal_gamma_c3v_characters(ir)
            else:
                spatial_chars[ir]=np.tile(
                    c3v_irrep_characters("E"),3
                ).astype(np.complex128)
    else:
        requested=[]

    eval_samples=(
        int(a.eval_samples)
        if a.eval_samples is not None
        else int(a.n_samples)
    )
    pav_sweep=(
        int(a.pav_sweep_size)
        if a.pav_sweep_size is not None
        else max(2,nf//4)
    )
    bstar_ref=_seed_bstar_reference(
        g,nf,a.L1,a.L2,a.seed_amplitude
    )

    settings={
        **vars(a),
        "Ns":int(g.Ns),
        "Nf":int(nf),
        "workflow":"Raw optimization -> PAV -> PAV+1LS diagnostics; optional Tint VAP",
        "c3v_available":bool(spatial_effective),
        "c3v_irreps":requested,
        "eval_samples_effective":eval_samples,
        "pav_sweep_size_effective":pav_sweep,
        "Bstar_seed_reference":bstar_ref,
        "notes":[
            "No C3v VAP is performed in this script.",
            "Full residual A1/A2 PAV is applied directly to optimized Raw parameters.",
            "Tint VAP is optional and disabled by default.",
            "Only optimized branches (Raw and optional Tint-VAP) are classified and mapped.",
            "PAV branches are energy-only symmetry diagnostics and report Delta E relative to their source branch.",
            "Unless --skip-lanczos is set, every PAV is followed by a stochastic one-step Lanczos/Ritz correction in span{|PAV>,H|PAV>}.",
            "Large-supercell 1LS uses MC spectral moments m1..m4 and never enumerates the full Hilbert space.",
            "1LS rows are energy/variance diagnostics only and do not define or relabel the Raw phase.",
        ],
    }
    atomic_json(out/"settings_raw_pav.json",settings)
    atomic_json(out/"geometry.json",g.metadata())

    # _stage reads these fields from args.
    a.eval_samples=eval_samples
    rows=[]
    optimized_rows=[]

    for iseed,seed in enumerate(a.seeds):
        a._active_seed=seed
        base_seed=a.rng_seed+10007*iseed
        seed_data=seed_bloch_blocks(
            g,nf,a.L1,a.L2,seed,a.seed_amplitude
        )

        # ---- 1. Raw optimization (or reuse) ----
        raw_dir=out/f"large_raw_{a.correlator}__{seed}"
        if a.reuse_raw:
            _,raw_params=_load_params(raw_dir)
            # Re-evaluate the reused state so this workflow gets a fresh,
            # same-statistics summary and map.
            raw_eval_dir=out/f"raw_eval_{a.correlator}__{seed}"
            raw_row,_=_stage(
                stage_name="Raw",
                directory=raw_eval_dir,
                g=g,nf=nf,V=a.V,
                hi=hi,H_fast=H_fast,graph=graph,current_ops=current_ops,
                seed_data=seed_data,
                shell_mats=shell_mats,bf_mats=bf_mats,
                use_backflow=use_backflow,
                initial_params=raw_params,
                maps=None,chars=None,point_label=None,
                time_reversal_even=False,
                n_iter=0,learning_rate=a.raw_learning_rate,
                args=a,rng_seed=base_seed+1,
                train_samples=a.pav_warmup_samples,
                train_sweep_size=(
                    a.sweep_size if a.sweep_size is not None else nf
                ),
                make_maps=True,map_edge_chunk=a.map_edge_chunk,
            )
        else:
            raw_row,raw_params=_stage(
                stage_name="Raw",
                directory=raw_dir,
                g=g,nf=nf,V=a.V,
                hi=hi,H_fast=H_fast,graph=graph,current_ops=current_ops,
                seed_data=seed_data,
                shell_mats=shell_mats,bf_mats=bf_mats,
                use_backflow=use_backflow,
                initial_params=None,
                maps=None,chars=None,point_label=None,
                time_reversal_even=False,
                n_iter=a.raw_iter,learning_rate=a.raw_learning_rate,
                args=a,rng_seed=base_seed+1,
                make_maps=True,map_edge_chunk=a.map_edge_chunk,
            )
        raw_row["branch_source"]="Raw"
        raw_row.update(_classify_branch(
            raw_row,bstar_ref,
            co_thr=a.classify_co_threshold,
            bow_thr=a.classify_bow_threshold,
            current_thr=a.classify_current_threshold,
        ))
        rows.append(raw_row)
        optimized_rows.append(dict(raw_row))

        # ---- 2. Tint PAV directly from Raw ----
        tint_pav_dir=out/f"raw_tint_pav_{a.correlator}__{seed}"
        tint_pav_row,_=_stage(
            stage_name="Raw->Tint-PAV",
            directory=tint_pav_dir,
            g=g,nf=nf,V=a.V,
            hi=hi,H_fast=H_fast,graph=graph,current_ops=current_ops,
            seed_data=seed_data,
            shell_mats=shell_mats,bf_mats=bf_mats,
            use_backflow=use_backflow,
            initial_params=raw_params,
            maps=tint_maps,chars=tint_chars,point_label="Gamma",
            time_reversal_even=False,
            n_iter=0,learning_rate=a.tint_learning_rate,
            args=a,rng_seed=base_seed+2,
            train_samples=a.pav_warmup_samples,
            train_sweep_size=pav_sweep,
            make_maps=False,map_edge_chunk=a.map_edge_chunk,
            measure_orders=False,
        )
        tint_pav_row["branch_source"]="Raw"
        _annotate_pav_energy(tint_pav_row,raw_row)
        rows.append(tint_pav_row)

        if not a.skip_lanczos:
            tint_l1=_pav_lanczos_row(
                tint_pav_row,
                hi=hi,H_fast=H_fast,graph=graph,
                seed_data=seed_data,params=raw_params,
                shell_mats=shell_mats,bf_mats=bf_mats,
                use_backflow=use_backflow,
                maps=tint_maps,chars=tint_chars,args=a,
                rng_seed=base_seed+102,sweep_size=pav_sweep,
                directory=out/f"raw_tint_pav_1ls_{a.correlator}__{seed}",
            )
            rows.append(tint_l1)

        # ---- 3. Full residual C3v PAV directly from Raw ----
        for ii,ir in enumerate(requested):
            pav_dir=out/f"raw_c3v_pav_{a.correlator}__{ir}__{seed}"
            row,_=_stage(
                stage_name="Raw->Tint+C3v-PAV",
                directory=pav_dir,
                g=g,nf=nf,V=a.V,
                hi=hi,H_fast=H_fast,graph=graph,current_ops=current_ops,
                seed_data=seed_data,
                shell_mats=shell_mats,bf_mats=bf_mats,
                use_backflow=use_backflow,
                initial_params=raw_params,
                maps=spatial_maps,chars=spatial_chars[ir],point_label=ir,
                time_reversal_even=False,
                n_iter=0,learning_rate=a.tint_learning_rate,
                args=a,rng_seed=base_seed+20+ii,
                train_samples=a.pav_warmup_samples,
                train_sweep_size=pav_sweep,
                make_maps=False,map_edge_chunk=a.map_edge_chunk,
                measure_orders=False,
            )
            row["branch_source"]="Raw"
            _annotate_pav_energy(row,raw_row)
            rows.append(row)

            if not a.skip_lanczos:
                l1row=_pav_lanczos_row(
                    row,
                    hi=hi,H_fast=H_fast,graph=graph,
                    seed_data=seed_data,params=raw_params,
                    shell_mats=shell_mats,bf_mats=bf_mats,
                    use_backflow=use_backflow,
                    maps=spatial_maps,chars=spatial_chars[ir],args=a,
                    rng_seed=base_seed+120+ii,sweep_size=pav_sweep,
                    directory=out/f"raw_c3v_pav_1ls_{a.correlator}__{ir}__{seed}",
                )
                rows.append(l1row)

        # ---- 4. Optional Tint VAP, then PAV the Tint-refined branch ----
        if int(a.tint_vap_iter)>0:
            tint_dir=out/f"large_tint_{a.correlator}__Gamma__{seed}"
            if a.reuse_tint_vap:
                _,tint_params=_load_params(tint_dir)
                tint_eval_dir=out/f"tint_vap_eval_{a.correlator}__{seed}"
                tint_row,_=_stage(
                    stage_name="Tint-VAP",
                    directory=tint_eval_dir,
                    g=g,nf=nf,V=a.V,
                    hi=hi,H_fast=H_fast,graph=graph,current_ops=current_ops,
                    seed_data=seed_data,
                    shell_mats=shell_mats,bf_mats=bf_mats,
                    use_backflow=use_backflow,
                    initial_params=tint_params,
                    maps=tint_maps,chars=tint_chars,point_label="Gamma",
                    time_reversal_even=False,
                    n_iter=0,learning_rate=a.tint_learning_rate,
                    args=a,rng_seed=base_seed+40,
                    train_samples=a.pav_warmup_samples,
                    train_sweep_size=(
                        a.sweep_size if a.sweep_size is not None else nf
                    ),
                    make_maps=True,map_edge_chunk=a.map_edge_chunk,
                )
            else:
                tint_row,tint_params=_stage(
                    stage_name="Tint-VAP",
                    directory=tint_dir,
                    g=g,nf=nf,V=a.V,
                    hi=hi,H_fast=H_fast,graph=graph,current_ops=current_ops,
                    seed_data=seed_data,
                    shell_mats=shell_mats,bf_mats=bf_mats,
                    use_backflow=use_backflow,
                    initial_params=raw_params,
                    maps=tint_maps,chars=tint_chars,point_label="Gamma",
                    time_reversal_even=False,
                    n_iter=a.tint_vap_iter,
                    learning_rate=a.tint_learning_rate,
                    args=a,rng_seed=base_seed+40,
                    make_maps=True,map_edge_chunk=a.map_edge_chunk,
                )
            tint_row["branch_source"]="Tint-VAP"
            tint_row.update(_classify_branch(
                tint_row,bstar_ref,
                co_thr=a.classify_co_threshold,
                bow_thr=a.classify_bow_threshold,
                current_thr=a.classify_current_threshold,
            ))
            rows.append(tint_row)
            optimized_rows.append(dict(tint_row))

            for ii,ir in enumerate(requested):
                pav_dir=out/f"tintvap_c3v_pav_{a.correlator}__{ir}__{seed}"
                row,_=_stage(
                    stage_name="Tint-VAP->C3v-PAV",
                    directory=pav_dir,
                    g=g,nf=nf,V=a.V,
                    hi=hi,H_fast=H_fast,graph=graph,current_ops=current_ops,
                    seed_data=seed_data,
                    shell_mats=shell_mats,bf_mats=bf_mats,
                    use_backflow=use_backflow,
                    initial_params=tint_params,
                    maps=spatial_maps,chars=spatial_chars[ir],point_label=ir,
                    time_reversal_even=False,
                    n_iter=0,learning_rate=a.tint_learning_rate,
                    args=a,rng_seed=base_seed+50+ii,
                    train_samples=a.pav_warmup_samples,
                    train_sweep_size=pav_sweep,
                    make_maps=False,map_edge_chunk=a.map_edge_chunk,
                    measure_orders=False,
                )
                row["branch_source"]="Tint-VAP"
                _annotate_pav_energy(row,tint_row)
                rows.append(row)

                if not a.skip_lanczos:
                    l1row=_pav_lanczos_row(
                        row,
                        hi=hi,H_fast=H_fast,graph=graph,
                        seed_data=seed_data,params=tint_params,
                        shell_mats=shell_mats,bf_mats=bf_mats,
                        use_backflow=use_backflow,
                        maps=spatial_maps,chars=spatial_chars[ir],args=a,
                        rng_seed=base_seed+150+ii,sweep_size=pav_sweep,
                        directory=out/f"tintvap_c3v_pav_1ls_{a.correlator}__{ir}__{seed}",
                    )
                    rows.append(l1row)

        _write_csv(out/"raw_pav_summary.csv",rows)
        _write_csv(out/"optimized_branch_summary.csv",optimized_rows)

    print("\n[raw branch characterization]")
    for r in optimized_rows:
        print(
            f" seed={r['seed']:11s} stage={r['stage']:10s} "
            f"phase={r['phase_label']:28s} "
            f"E={r['mc_E']:+.8f} +/- {r['mc_E_error']:.2e} "
            f"CO1={r.get('CO_onepoint_rms',0.0):.4g} "
            f"B*={r.get('B_star_mean',0.0):+.4g} "
            f"Jsame={r.get('J_same_mean',0.0):+.4g} "
            f"Jopp={r.get('J_opposite_mean',0.0):+.4g}"
        )

    print("\n[PAV energy shifts]")
    for r in rows:
        if "PAV_delta_E" not in r or "lanczos_delta_E" in r:
            continue
        print(
            f" seed={r['seed']:11s} stage={r['stage']:22s} "
            f"irrep={str(r.get('point_label','-')):5s} "
            f"dE={r['PAV_delta_E']:+.6f} "
            f"({r.get('PAV_delta_E_sigma',float('nan')):+.2f} sigma)"
        )

    if not a.skip_lanczos:
        print("\n[PAV -> one-Lanczos-step energy shifts]")
        for r in rows:
            if "lanczos_delta_E" not in r:
                continue
            ds=r.get("lanczos_delta_E_sigma")
            dss="nan" if ds is None else f"{ds:+.2f}"
            print(
                f" seed={r['seed']:11s} stage={r['stage']:26s} "
                f"irrep={str(r.get('point_label','-')):5s} "
                f"E={r['mc_E']:+.8f} "
                f"dE_L1={r['lanczos_delta_E']:+.6f} "
                f"({dss} sigma) "
                f"var={r['lanczos_variance_before']:.4g}"
                f"->{r['lanczos_variance_after']:.4g}"
            )
    print(f"Saved: {out}")


if __name__=="__main__":
    main()
