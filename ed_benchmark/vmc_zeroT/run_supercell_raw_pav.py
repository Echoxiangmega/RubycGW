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
from supercell_ansatz import seed_bloch_blocks
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
        "workflow":"Raw optimization -> PAV diagnostics; optional Tint VAP",
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
        if "PAV_delta_E" not in r:
            continue
        print(
            f" seed={r['seed']:11s} stage={r['stage']:22s} "
            f"irrep={str(r.get('point_label','-')):5s} "
            f"dE={r['PAV_delta_E']:+.6f} "
            f"({r.get('PAV_delta_E_sigma',float('nan')):+.2f} sigma)"
        )
    print(f"Saved: {out}")


if __name__=="__main__":
    main()
