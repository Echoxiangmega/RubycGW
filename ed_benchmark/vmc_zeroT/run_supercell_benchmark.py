#!/usr/bin/env python3
"""Benchmark the translation-built-in ring3-supercell VMC ansatz.

The 18-site ring3 cluster is treated as the variational unit cell.  A total
L1 x L2 superlattice contains Ns=18*L1*L2 sites.  Ring3-supercell translation
is exact in the Bloch orbital construction and is never explicitly projected.

For L1=L2=1, exact ED is available and this runner compares:
  ED
  Raw-Supercell-Bloch
  VAP-C3-Supercell-Bloch
  VAP-C3-Supercell-Bloch + one exact Lanczos step

This 1x1 case is deliberately designed as a fair comparison with the existing
Hex18 benchmark: the raw one-body variational space is the same full complex
18 x Nf matrix, but C3 VAP sums only 3 point-group images instead of the old
translations x C3v spatial projection (18 images on ring3).

For larger superlattices the runner currently performs MC optimization and
saves the scalable ansatz state/parameters, but exact full-basis postprocessing
is intentionally disabled.
"""
from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from math import comb
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NATIVE=REPO/"ed_benchmark"/"native_ed"
sys.path.insert(0,str(NATIVE))

import geometry as native_geometry
import observables as native_observables

from benchmark_observables import benchmark_observables
from vmc_ansatz import jastrow_channels, bond_backflow_channels
from supercell_ansatz import (
    seed_bloch_blocks,
    build_supercell_netket_model,
    extract_supercell_parameters,
)
from run_benchmark import (
    atomic_json,
    make_netket_hamiltonian,
    reorder_netket_vector_to_native,
    solve_ed,
    stats_error,
    validate_netket_native_hamiltonian,
)
from run_projection_comparison import one_lanczos_step
from symmetry_projection import (
    build_c3_group,
    c3_site_maps,
    characterize_unique_state,
    decompose_ground_manifold_abelian,
    project_to_characters,
    projected_sector_residual,
    c3_from_characters,
)


def write_csv(path: Path, rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def c3_characters(m: int) -> np.ndarray:
    m=int(m)%3
    return np.asarray(
        [np.exp(-2j*np.pi*m*r/3) for r in range(3)],
        np.complex128,
    )


def c3_targets_from_ed(g,H,ed_gs,tol=1e-8):
    group=build_c3_group(g,H.states)
    if ed_gs.shape[1]==1:
        chars=characterize_unique_state(group,ed_gs[:,0],tol=max(1e-8,tol))
        q=c3_from_characters(g,group,chars)
        return group,[{
            "m":int(q["m"]),
            "characters":np.asarray(chars,complex),
            "ground_dimension":1,
        }]
    sectors=decompose_ground_manifold_abelian(
        group,ed_gs,tol=max(1e-8,10*tol)
    )
    targets=[]
    for sec in sectors:
        q=c3_from_characters(g,group,sec["characters"])
        if q["m"] is None:
            raise RuntimeError("Could not identify C3 quantum number in ED ground manifold")
        targets.append({
            "m":int(q["m"]),
            "characters":np.asarray(sec["characters"],complex),
            "ground_dimension":int(sec["dimension"]),
        })
    # Merge accidental multiplicities with the same C3 character.
    merged={}
    for t in targets:
        m=t["m"]
        if m not in merged:
            merged[m]=dict(t)
        else:
            merged[m]["ground_dimension"]+=t["ground_dimension"]
    return group,[merged[m] for m in sorted(merged)]


def exact_row(method,seed,psi,H,ed_gs,ed_row,*,extra=None):
    psi=np.asarray(psi,np.complex128)
    psi/=np.linalg.norm(psi)
    Hpsi=H.operator@psi
    E=float(np.vdot(psi,Hpsi).real)
    var=float(max(0.,np.vdot(Hpsi,Hpsi).real-E**2))
    fidelity=float(np.sum(np.abs(ed_gs.conj().T@psi)**2).real)
    obs=benchmark_observables(H.g,H.states,psi,native_observables)
    row={
        "method":method,
        "seed":seed,
        "E":E,
        "E_per_site":E/H.g.Ns,
        "delta_E_vs_ED":E-float(ed_row["E"]),
        "relative_E_error":abs(E-float(ed_row["E"]))/max(abs(float(ed_row["E"])),1e-15),
        "exact_energy_variance":var,
        "exact_energy_variance_per_site":var/H.g.Ns,
        "fidelity_ground_subspace":fidelity,
        **obs.as_dict(),
    }
    if extra:
        row.update(extra)
    return row


def _save_history(path,log):
    history={}
    try:
        en=log.data["Energy"]
        history["iteration"]=[int(x) for x in np.asarray(en.iters)]
        history["E_mean"]=[float(np.real(x)) for x in np.asarray(en.Mean)]
        history["E_sigma"]=[float(x) for x in np.asarray(en.Sigma)]
    except Exception:
        pass
    atomic_json(path,history)


def _warm_blocks(params,seed_data):
    if params is None:
        return [np.asarray(x,np.complex128) for x in seed_data["blocks"]],None,None
    blocks=[np.asarray(x,np.complex128) for x in params["blocks"]]
    v=np.asarray(params.get("v",[]),float)
    eta=np.asarray(params.get("eta",[]),complex)
    return blocks,(None if v.size==0 else v),(None if eta.size==0 else eta)


def train_one(
    *,
    g,nf,V,
    seed_name,seed_amplitude,
    l1,l2,
    n_shells,jastrow_mode,use_backflow,
    n_samples,n_discard,n_chains,n_iter,learning_rate,diag_shift,
    rng_seed,chunk_size,
    output_dir,dir_name,method_name,
    point_maps=None,point_chars=None,c3_m=None,
    initial_params=None,
    native_H=None,ed_gs=None,ed_row=None,
    validate_hamiltonian=False,
):
    import jax
    jax.config.update("jax_enable_x64",True)
    import netket as nk

    t0=time.perf_counter()
    hi,H_generic,H_fast,graph=make_netket_hamiltonian(g,nf,V)
    jlabels,jdist,shells,jcounts=jastrow_channels(
        g,mode=jastrow_mode,n_shells=n_shells
    )
    bf_labels,bf_mats,bf_counts=bond_backflow_channels(g)
    seed_data=seed_bloch_blocks(
        g,nf,l1,l2,seed_name,seed_amplitude
    )
    blocks0,v0,eta0=_warm_blocks(initial_params,seed_data)

    if point_maps is not None:
        ktot=np.asarray(seed_data["total_supercell_momentum"],float)
        dist=np.linalg.norm(ktot-np.rint(ktot))
        if dist>1e-9:
            raise RuntimeError(
                "C3-only projection requires a C3-compatible supercell-translation "
                "sector. The fixed seed occupation pattern has total supercell "
                f"momentum {ktot.tolist()}, not Gamma. Choose a closed-shell/filling "
                "or run this branch without C3 projection."
            )

    model=build_supercell_netket_model(
        blocks0,
        seed_data["sc_coords"],
        seed_data["internal_sites"],
        seed_data["kpoints"],
        shells,
        use_jastrow=True,
        backflow_matrices=(bf_mats if use_backflow else None),
        initial_v=v0,
        initial_eta=eta0,
        point_site_maps=point_maps,
        point_characters=point_chars,
    )

    sampler=nk.sampler.MetropolisFermionHop(
        hi,graph=graph,n_chains=n_chains
    )
    vstate=nk.vqs.MCState(
        sampler,model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard,
        chunk_size=chunk_size,
        seed=rng_seed,
        sampler_seed=rng_seed+104729,
    )
    setup_s=time.perf_counter()-t0

    if (
        validate_hamiltonian
        and native_H is not None
        and hasattr(native_H.operator,"tocsr")
    ):
        validate_netket_native_hamiltonian(
            hi,H_generic,native_H,g.Ns
        )

    tw=time.perf_counter()
    warm=vstate.expect(H_fast)
    warmup_s=time.perf_counter()-tw

    optimizer=nk.optimizer.Sgd(learning_rate=learning_rate)
    qgt_ctor=partial(nk.optimizer.qgt.QGTJacobianDense,mode="complex")
    sr=nk.optimizer.SR(
        qgt=qgt_ctor,
        solver=nk.optimizer.solver.cholesky_with_fallback,
        diag_shift=diag_shift,
    )
    driver=nk.driver.VMC(
        H_fast,optimizer,variational_state=vstate,preconditioner=sr
    )
    n_images=1 if point_maps is None else int(len(point_maps))
    print(
        f"[optimizer] {method_name}: parameters={vstate.n_parameters}, "
        f"k_blocks={len(seed_data['kpoints'])}, "
        f"nocc(k)={seed_data['occupations_per_k'].tolist()}, "
        f"explicit point images={n_images}",
        flush=True,
    )

    log=nk.logging.RuntimeLog()
    topt=time.perf_counter()
    driver.run(n_iter=n_iter,out=log,show_progress=True)
    opt_s=time.perf_counter()-topt

    te=time.perf_counter()
    vstate.sample()
    stats=vstate.expect(H_fast)
    mc_eval_s=time.perf_counter()-te

    params=extract_supercell_parameters(
        vstate.parameters,len(seed_data["kpoints"])
    )

    psi=None
    exact=None
    post_s=0.
    if native_H is not None and ed_gs is not None and ed_row is not None:
        tp=time.perf_counter()
        psi_nk=np.asarray(vstate.to_array(normalize=True))
        psi=reorder_netket_vector_to_native(
            hi,psi_nk,native_H.states,g.Ns
        )
        extra={
            "family":"ring3_supercell_bloch",
            "L1":int(l1),"L2":int(l2),
            "supercell_translation_built_in":True,
            "explicit_projection_images":n_images,
            "c3_m":c3_m,
            "n_parameters":int(vstate.n_parameters),
            "n_one_body_parameter_entries":int(18*nf),
            "n_k_blocks":int(len(seed_data["kpoints"])),
            "occupations_per_k":";".join(map(str,seed_data["occupations_per_k"].tolist())),
            "seed_shell_gap":seed_data["shell_gap"],
            "total_supercell_momentum":";".join(
                f"{x:.12g}" for x in seed_data["total_supercell_momentum"]
            ),
            "mc_E":float(np.real(stats.mean)),
            "mc_E_error":stats_error(stats),
            "time_setup_s":setup_s,
            "time_jit_warmup_s":warmup_s,
            "time_solve_or_opt_s":opt_s,
            "time_per_iteration_s":opt_s/max(int(n_iter),1),
            "time_mc_final_eval_s":mc_eval_s,
            "jastrow_mode":jastrow_mode,
            "jastrow_labels":";".join(map(str,jlabels)),
            "backflow_enabled":bool(use_backflow),
        }
        if point_maps is not None:
            group=build_c3_group(g,native_H.states)
            sector_weight=float(project_to_characters(
                group,np.asarray(point_chars,complex),psi
            )[1])
            extra["target_sector_weight"]=sector_weight
            extra["projected_sector_residual"]=projected_sector_residual(
                group,np.asarray(point_chars,complex),psi
            )
        exact=exact_row(
            method_name,seed_name,psi,native_H,ed_gs,ed_row,extra=extra
        )
        post_s=time.perf_counter()-tp

    d=output_dir/dir_name
    d.mkdir(parents=True,exist_ok=True)
    save_items={
        "kpoints":seed_data["kpoints"],
        "occupations_per_k":seed_data["occupations_per_k"],
        "supercell_coords":seed_data["sc_coords"],
        "internal_sites":seed_data["internal_sites"],
        "star_formfactor":seed_data["star_formfactor"],
        "total_supercell_momentum":seed_data["total_supercell_momentum"],
        "variational_v":params["v"],
        "variational_eta":params["eta"],
    }
    for ik,block in enumerate(params["blocks"]):
        save_items[f"variational_M_{ik}"]=block
    if psi is not None:
        save_items["psi_native"]=psi
        save_items["native_states"]=native_H.states
    np.savez_compressed(d/"final_state.npz",**save_items)
    _save_history(d/"history.json",log)

    summary={
        "method":method_name,
        "seed":seed_name,
        "family":"ring3_supercell_bloch",
        "L1":int(l1),"L2":int(l2),
        "Ns":int(g.Ns),"Nf":int(nf),"V":float(V),
        "supercell_translation_built_in":True,
        "explicit_projection_images":n_images,
        "c3_m":c3_m,
        "n_parameters":int(vstate.n_parameters),
        "n_one_body_parameter_entries":int(18*nf),
        "n_k_blocks":int(len(seed_data["kpoints"])),
        "occupations_per_k":seed_data["occupations_per_k"].tolist(),
        "seed_shell_gap":seed_data["shell_gap"],
        "total_supercell_momentum":[
            float(x) for x in seed_data["total_supercell_momentum"]
        ],
        "mc_E":float(np.real(stats.mean)),
        "mc_E_error":stats_error(stats),
        "warmup_mc_E":float(np.real(warm.mean)),
        "n_samples":int(vstate.n_samples),
        "n_iter":int(n_iter),
        "learning_rate":float(learning_rate),
        "time_setup_s":setup_s,
        "time_jit_warmup_s":warmup_s,
        "time_solve_or_opt_s":opt_s,
        "time_per_iteration_s":opt_s/max(int(n_iter),1),
        "time_mc_final_eval_s":mc_eval_s,
        "time_postprocess_s":post_s,
        "time_total_s":setup_s+warmup_s+opt_s+mc_eval_s+post_s,
        "jastrow_mode":jastrow_mode,
        "jastrow_labels":jlabels,
        "jastrow_channel_counts":[int(x) for x in jcounts],
        "backflow_enabled":bool(use_backflow),
        "backflow_labels":bf_labels if use_backflow else [],
        "backflow_channel_counts":[int(x) for x in bf_counts] if use_backflow else [],
    }
    if exact is not None:
        summary.update(exact)
    atomic_json(d/"summary.json",summary)
    return summary,params,psi,seed_data


def parser():
    p=argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--L1",type=int,default=1)
    p.add_argument("--L2",type=int,default=1)
    p.add_argument(
        "--nf",type=int,default=None,
        help="Total fermion number. Default is 6*L1*L2 (same 1/3 site filling as Hex18 Nf=6).",
    )
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
        "--jastrow-mode",choices=["bond_aware","distance"],default="bond_aware"
    )
    p.add_argument("--jastrow-shells",type=int,default=4)
    p.add_argument("--n-samples",type=int,default=4096)
    p.add_argument("--n-discard",type=int,default=16)
    p.add_argument("--n-chains",type=int,default=16)
    p.add_argument("--n-iter",type=int,default=300)
    p.add_argument("--rotation-iter",type=int,default=100)
    p.add_argument("--learning-rate",type=float,default=0.03)
    p.add_argument("--rotation-learning-rate",type=float,default=0.01)
    p.add_argument("--diag-shift",type=float,default=0.05)
    p.add_argument("--chunk-size",type=int,default=512)
    p.add_argument("--rng-seed",type=int,default=2026)
    p.add_argument(
        "--c3-m",nargs="*",type=int,default=None,
        help="C3 sectors to optimize. For 1x1 with ED, default uses only C3 sectors present in the exact ground manifold.",
    )
    p.add_argument("--skip-raw",action="store_true")
    p.add_argument("--skip-c3",action="store_true")
    p.add_argument("--skip-lanczos",action="store_true")
    p.add_argument("--ed-k",type=int,default=12)
    p.add_argument("--ed-degeneracy-tol",type=float,default=1e-8)
    p.add_argument(
        "--output",default="results/supercell_1x1_N6_V5p5"
    )
    return p


def main():
    a=parser().parse_args()
    if a.L1<1 or a.L2<1:
        raise ValueError("L1,L2 must be positive")
    out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=True)

    g=native_geometry.make_ring3_supercell(
        a.L1,a.L2,t=a.t,t1=a.t1,t2=a.t2
    )
    nf=6*a.L1*a.L2 if a.nf is None else int(a.nf)
    if not 0<nf<g.Ns:
        raise ValueError("nf must lie between 1 and Ns-1")

    use_backflow=a.correlator=="backflow"
    exact_available=(a.L1,a.L2)==(1,1)

    native_H=evals=ed_gs=ed_row=None
    c3_group=None
    targets=[]
    if exact_available:
        native_H,evals,ed_gs,ed_row=solve_ed(
            g,nf,a.V,k=a.ed_k,
            degeneracy_tol=a.ed_degeneracy_tol,
            backend="full",
        )
        if ed_row.get("ground_manifold_truncated"):
            raise RuntimeError("ED ground manifold truncated; increase --ed-k")
        c3_group,targets=c3_targets_from_ed(
            g,native_H,ed_gs,a.ed_degeneracy_tol
        )
        ed_dir=out/"ed"
        ed_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(
            ed_dir/"final_state.npz",
            psi_ground_manifold=np.asarray(ed_gs,np.complex128),
            native_states=np.asarray(native_H.states,np.uint64),
            eigenvalues=np.asarray(evals,float),
            ground_energy=np.asarray(float(evals[0])),
        )
        atomic_json(ed_dir/"summary.json",ed_row)
    else:
        if a.c3_m is None:
            targets=[
                {"m":m,"characters":c3_characters(m),"ground_dimension":None}
                for m in (0,1,2)
            ]

    if a.c3_m is not None:
        wanted=sorted({int(m)%3 for m in a.c3_m})
        targets=[
            {"m":m,"characters":c3_characters(m),"ground_dimension":None}
            for m in wanted
        ]

    point_maps=None
    if not a.skip_c3:
        try:
            point_maps=np.asarray(
                [p for _,p in c3_site_maps(g)],np.int32
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "This total ring3-supercell torus does not retain ordinary C3. "
                "Use a C3-compatible shape (normally L1=L2) or pass --skip-c3."
            ) from exc
    settings={
        **vars(a),
        "family":"ring3_supercell_bloch",
        "Ns":g.Ns,"Nf":nf,
        "n_supercells":a.L1*a.L2,
        "variational_supercell_sites":18,
        "supercell_translation_built_in":True,
        "explicit_rotation_group":"C3",
        "explicit_rotation_images":3,
        "exact_available":exact_available,
        "c3_targets":[{
            "m":int(t["m"]),
            "ground_dimension":t["ground_dimension"],
        } for t in targets],
    }
    atomic_json(out/"settings.json",settings)
    atomic_json(out/"geometry.json",g.metadata())

    rows=[]
    if ed_row is not None:
        ed=dict(ed_row)
        ed.update({
            "method":"ED",
            "family":"exact",
            "L1":a.L1,"L2":a.L2,
            "supercell_translation_built_in":False,
            "explicit_projection_images":0,
        })
        rows.append(ed)
        print(
            f"[ED] E0={ed['E']:.12g}, ground_dim={ed_gs.shape[1]}, "
            f"C3 targets={[t['m'] for t in targets]}",
            flush=True,
        )

    raw_cache={}
    run_id=0
    for seed in a.seeds:
        run_id+=1
        raw_dir=f"supercell_raw_{a.correlator}__{seed}"
        raw_path=out/raw_dir/"summary.json"
        if a.skip_raw and not raw_path.exists():
            raise RuntimeError(
                f"--skip-raw requested but warm-start file is missing: {raw_path}"
            )
        if not a.skip_raw:
            print(f"\n[supercell raw] seed={seed}",flush=True)
            raw,params,psi,seed_data=train_one(
                g=g,nf=nf,V=a.V,
                seed_name=seed,seed_amplitude=a.seed_amplitude,
                l1=a.L1,l2=a.L2,
                n_shells=a.jastrow_shells,jastrow_mode=a.jastrow_mode,
                use_backflow=use_backflow,
                n_samples=a.n_samples,n_discard=a.n_discard,
                n_chains=a.n_chains,n_iter=a.n_iter,
                learning_rate=a.learning_rate,diag_shift=a.diag_shift,
                rng_seed=a.rng_seed+1009*run_id,chunk_size=a.chunk_size,
                output_dir=out,dir_name=raw_dir,
                method_name=f"Raw-SupercellBloch-{a.correlator}",
                point_maps=None,point_chars=None,c3_m=None,
                initial_params=None,
                native_H=native_H,ed_gs=ed_gs,ed_row=ed_row,
                validate_hamiltonian=(run_id==1 and exact_available),
            )
            if exact_available:
                rows.append(raw)
        else:
            raw=json.loads(raw_path.read_text(encoding="utf-8"))
            with np.load(out/raw_dir/"final_state.npz",allow_pickle=False) as z:
                nkp=len(z["kpoints"])
                params={
                    "blocks":[
                        np.asarray(z[f"variational_M_{ik}"],complex)
                        if f"variational_M_{ik}" in z.files
                        else np.empty((18,0),complex)
                        for ik in range(nkp)
                    ],
                    "v":np.asarray(z["variational_v"],float),
                    "eta":np.asarray(z["variational_eta"],complex),
                }
            seed_data=None
        raw_cache[seed]=(raw,params)

    if not a.skip_c3:
        for target in targets:
            m=int(target["m"])
            chars=np.asarray(target["characters"],complex)
            for seed in a.seeds:
                run_id+=1
                raw,params0=raw_cache[seed]
                print(
                    f"\n[supercell C3-VAP] seed={seed}, m={m}",
                    flush=True,
                )
                method=f"VAP-C3-SupercellBloch-{a.correlator}"
                dirname=f"supercell_c3_{a.correlator}__m{m}__{seed}"
                row,params,psi,_=train_one(
                    g=g,nf=nf,V=a.V,
                    seed_name=seed,seed_amplitude=a.seed_amplitude,
                    l1=a.L1,l2=a.L2,
                    n_shells=a.jastrow_shells,jastrow_mode=a.jastrow_mode,
                    use_backflow=use_backflow,
                    n_samples=a.n_samples,n_discard=a.n_discard,
                    n_chains=a.n_chains,n_iter=a.rotation_iter,
                    learning_rate=a.rotation_learning_rate,
                    diag_shift=a.diag_shift,
                    rng_seed=a.rng_seed+1009*run_id,
                    chunk_size=a.chunk_size,
                    output_dir=out,dir_name=dirname,
                    method_name=method,
                    point_maps=point_maps,point_chars=chars,c3_m=m,
                    initial_params=params0,
                    native_H=native_H,ed_gs=ed_gs,ed_row=ed_row,
                    validate_hamiltonian=False,
                )
                if exact_available:
                    row["target_c3"]=f"C3:m{m}"
                    row["target_sector_ground_dimension"]=target["ground_dimension"]
                    rows.append(row)

                    if not a.skip_lanczos:
                        lspsi,info=one_lanczos_step(psi,native_H)
                        lsrow=exact_row(
                            method+"+1LS",seed,lspsi,native_H,ed_gs,ed_row,
                            extra={
                                "family":"ring3_supercell_bloch",
                                "L1":a.L1,"L2":a.L2,
                                "supercell_translation_built_in":True,
                                "explicit_projection_images":3,
                                "c3_m":m,
                                "target_c3":f"C3:m{m}",
                                "target_sector_ground_dimension":target["ground_dimension"],
                                **info,
                            },
                        )
                        rows.append(lsrow)
                        d=out/f"{dirname}_1ls"
                        d.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(
                            d/"final_state.npz",
                            psi_native=lspsi,
                            native_states=native_H.states,
                        )
                        atomic_json(d/"summary.json",lsrow)

    if rows:
        write_csv(out/"comparison_supercell.csv",rows)

    if exact_available:
        # Compact best-by-method timing/accuracy table for direct comparison
        # with the historical Hex18 comparison_all.csv.
        best=[]
        methods=list(dict.fromkeys(r["method"] for r in rows))
        for method in methods:
            rr=[r for r in rows if r["method"]==method and r.get("seed")!="exact"]
            if method=="ED":
                best.append(next(r for r in rows if r["method"]=="ED"))
                continue
            if not rr:
                continue
            chosen=min(rr,key=lambda r:float(r["E"]))
            best.append(chosen)
        write_csv(out/"best_supercell.csv",best)

        print("\n[1x1 supercell benchmark summary]",flush=True)
        for r in best:
            if r["method"]=="ED":
                print(f"  ED E={r['E']:.10f}",flush=True)
            else:
                print(
                    f"  {r['method']:34s} seed={r['seed']:11s} "
                    f"dE={r['delta_E_vs_ED']:+.4e} "
                    f"F={r['fidelity_ground_subspace']:.6f} "
                    f"var={r['exact_energy_variance']:.3e} "
                    f"t/it={r.get('time_per_iteration_s',float('nan')):.3f}s",
                    flush=True,
                )

    print(f"\nSaved: {out}",flush=True)


if __name__=="__main__":
    main()
