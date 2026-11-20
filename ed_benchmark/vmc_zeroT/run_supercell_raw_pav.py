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

Every evaluated branch records energy, CO/LC diagnostics, optimized parameters,
and a site-density | bond-order | bond-current triptych estimated from the same
Monte Carlo state.
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

    settings={
        **vars(a),
        "Ns":int(g.Ns),
        "Nf":int(nf),
        "workflow":"Raw optimization -> PAV diagnostics; optional Tint VAP",
        "c3v_available":bool(spatial_effective),
        "c3v_irreps":requested,
        "eval_samples_effective":eval_samples,
        "pav_sweep_size_effective":pav_sweep,
        "notes":[
            "No C3v VAP is performed in this script.",
            "Full residual A1/A2 PAV is applied directly to optimized Raw parameters.",
            "Tint VAP is optional and disabled by default.",
            "Every evaluated branch saves site/bond/current maps.",
        ],
    }
    atomic_json(out/"settings_raw_pav.json",settings)
    atomic_json(out/"geometry.json",g.metadata())

    # _stage reads these fields from args.
    a.eval_samples=eval_samples
    rows=[]

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
        rows.append(raw_row)

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
            make_maps=True,map_edge_chunk=a.map_edge_chunk,
        )
        tint_pav_row["branch_source"]="Raw"
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
                make_maps=True,map_edge_chunk=a.map_edge_chunk,
            )
            row["branch_source"]="Raw"
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
            rows.append(tint_row)

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
                    make_maps=True,map_edge_chunk=a.map_edge_chunk,
                )
                row["branch_source"]="Tint-VAP"
                rows.append(row)

        _write_csv(out/"raw_pav_summary.csv",rows)

    print("\n[raw+pav branch summary]")
    for r in rows:
        print(
            f" seed={r['seed']:11s} "
            f"stage={r['stage']:22s} "
            f"irrep={str(r.get('point_label','-')):5s} "
            f"E={r['mc_E']:+.8f} +/- {r['mc_E_error']:.2e} "
            f"SCO={r['S_CO']:.5g} "
            f"SLC-={r['S_LC_opposite']:.5g} "
            f"J-={r['J_opposite_mean']:+.5g}"
        )
    print(f"Saved: {out}")


if __name__=="__main__":
    main()
