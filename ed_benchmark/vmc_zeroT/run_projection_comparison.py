#!/usr/bin/env python3
"""Compare ED, PAV, full VAP, C3-constrained PAV, and C3-constrained VAP.

All methods use the same Hex18 Hamiltonian and the same density-Jastrow shell
definition. The comparison is designed to isolate the effect of constraining
the Slater occupied subspace to preserve ordinary C3 while allowing primitive
translation symmetry to break.

Methods:
  ED
  PAV-Slater-Jastrow
      optimize a full complex Slater x Jastrow state, then project afterward
      into the exact ED ground symmetry sector(s).
  Full-VAP-Slater-Jastrow
      optimize the same full complex Slater x Jastrow state after projection.
  C3-PAV-Slater-Jastrow
      optimize a C3-constrained but translation-breaking Slater x Jastrow
      state, then project afterward into the same full ED target sector(s).
  C3-VAP-Slater-Jastrow
      optimize that C3-constrained state directly after the full target-sector
      projection.

For a unique ED ground state the full target group is ordinary T x C3v.
For a degenerate ED ground manifold, full target sectors are the ordinary
abelian T x C3 sectors present in the exact ground manifold.

The C3-constrained base state is NOT primitive-translation invariant.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from functools import partial
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NATIVE = REPO / "ed_benchmark" / "native_ed"
sys.path.insert(0, str(NATIVE))

import geometry as native_geometry
import observables as native_observables

from benchmark_observables import benchmark_observables, ruby_star_formfactor
from vmc_ansatz import minimum_image_shells, one_body_seed
from c3_constrained_ansatz import (
    c3_seed_subspace,
    build_c3_constrained_netket_model,
)
from run_benchmark import (
    atomic_json,
    make_netket_hamiltonian,
    reorder_netket_vector_to_native,
    run_vmc_one,
    solve_ed,
    stats_error,
)
from run_parametric import run_full_vap_one
from symmetry_projection import (
    build_ordinary_space_group,
    build_translation_c3_group,
    build_c3_group,
    characterize_unique_state,
    decompose_ground_manifold_abelian,
    project_to_characters,
    projected_sector_residual,
    commutator_error,
    sector_quantum_numbers,
)


METHOD_ORDER = [
    "ED",
    "PAV-Slater-Jastrow",
    "Full-VAP-Slater-Jastrow",
    "C3-PAV-Slater-Jastrow",
    "C3-VAP-Slater-Jastrow",
]


def write_csv(path: Path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def add_ed_differences(row, ed):
    row["delta_E_vs_ED"] = float(row["E"]) - float(ed["E"])
    row["relative_E_error"] = abs(row["delta_E_vs_ED"]) / max(abs(float(ed["E"])), 1e-15)
    for key in ("CO_qmax", "S_BOW_star", "S_LC_same", "S_LC_opposite"):
        ref = float(ed[key]); val = float(row[key])
        row[f"delta_{key}_vs_ED"] = val-ref
        row[f"relative_{key}_error"] = abs(val-ref)/max(abs(ref), 1e-12)
    return row


def build_ed_sectors(g, H, ed_gs, tol):
    if ed_gs.shape[1] == 1:
        group = build_ordinary_space_group(g, H.states)
        err, name = commutator_error(group, H.operator, n_tests=2)
        if err > 1e-9:
            raise RuntimeError(f"full symmetry/H mismatch: {name} {err:.3e}")
        chars = characterize_unique_state(group, ed_gs[:,0], tol=1e-8)
        sectors = [{
            "characters": chars,
            "basis": ed_gs,
            "dimension": 1,
            "residual": 0.0,
        }]
        group_name = "ordinary translations x C3v"
    else:
        group = build_translation_c3_group(g, H.states)
        err, name = commutator_error(group, H.operator, n_tests=2)
        if err > 1e-9:
            raise RuntimeError(f"T x C3/H mismatch: {name} {err:.3e}")
        sectors = decompose_ground_manifold_abelian(
            group, ed_gs, tol=max(1e-8,10*tol)
        )
        group_name = "ordinary translations x C3"

    for sec in sectors:
        sec["quantum_numbers"] = sector_quantum_numbers(
            g, group, sec["characters"]
        )

    # Also characterize the C3 content of the exact ground manifold. This is
    # used only to choose a compatible C3-constrained base Slater branch.
    c3_group = build_c3_group(g, H.states)
    err, name = commutator_error(c3_group, H.operator, n_tests=2)
    if err > 1e-9:
        raise RuntimeError(f"C3/H mismatch: {name} {err:.3e}")
    if ed_gs.shape[1] == 1:
        chars = characterize_unique_state(c3_group, ed_gs[:,0], tol=1e-8)
        c3_sectors = [{
            "characters": chars,
            "basis": ed_gs,
            "dimension": 1,
            "residual": 0.0,
        }]
    else:
        c3_sectors = decompose_ground_manifold_abelian(
            c3_group, ed_gs, tol=max(1e-8,10*tol)
        )
    for sec in c3_sectors:
        sec["quantum_numbers"] = sector_quantum_numbers(
            g, c3_group, sec["characters"]
        )
    return group, group_name, sectors, c3_group, c3_sectors


def sector_record(i, sec):
    qn=sec["quantum_numbers"]
    return {
        "sector_id": int(i),
        "dimension": int(sec["dimension"]),
        "residual": float(sec["residual"]),
        "k": qn["momentum"]["name"],
        "q": qn["momentum"]["q"],
        "c3": qn["c3"]["label"],
        "c3_m": qn["c3"]["m"],
    }


def projected_row(
    *, method, seed, psi, H, ed_gs, ed_row, group, sec,
    sector_id, projection_name, extra=None,
):
    psi_p, weight = project_to_characters(group, sec["characters"], psi)
    if psi_p is None:
        return None
    t0=time.perf_counter()
    Hpsi=H.operator@psi_p
    E=float(np.vdot(psi_p,Hpsi).real)
    var=float(max(0.0,np.vdot(Hpsi,Hpsi).real-E**2))
    obs=benchmark_observables(H.g,H.states,psi_p,native_observables)
    fid=float(np.sum(np.abs(ed_gs.conj().T@psi_p)**2).real)
    qn=sec["quantum_numbers"]
    row={
        "method":method,
        "seed":seed,
        "projection":projection_name,
        "target_sector_id":int(sector_id),
        "target_sector_ground_dimension":int(sec["dimension"]),
        "target_k":qn["momentum"]["name"],
        "target_q":json.dumps(qn["momentum"]["q"]),
        "target_c3":qn["c3"]["label"],
        "target_c3_m":qn["c3"]["m"],
        "symmetry_sector_weight":float(weight),
        "projected_sector_residual":float(
            projected_sector_residual(group,sec["characters"],psi_p)
        ),
        "E":E,
        "E_per_site":E/H.g.Ns,
        "exact_energy_variance":var,
        "exact_energy_variance_per_site":var/H.g.Ns,
        "fidelity_ground_subspace":fid,
        "time_projection_postprocess_s":time.perf_counter()-t0,
        **obs.as_dict(),
    }
    if extra:
        row.update(extra)
    return add_ed_differences(row,ed_row),psi_p


def load_summary_state(directory,H):
    sf=directory/"summary.json"
    wf=directory/"final_state.npz"
    if not sf.exists() or not wf.exists():
        return None,None
    summary=json.loads(sf.read_text(encoding="utf-8"))
    with np.load(wf,allow_pickle=False) as z:
        psi=np.asarray(z["psi_native"],np.complex128)
        states=np.asarray(z["native_states"],np.uint64)
    if not np.array_equal(states,H.states):
        raise RuntimeError(f"basis mismatch in {wf}")
    psi/=np.linalg.norm(psi)
    return summary,psi


def run_c3_base_or_vap(
    g,nf,V,H,ed_gs,ed_E0, *,
    seed_name,seed_amplitude,target_c3_m,
    target_group=None,target_chars=None,target_sector_id=None,
    target_sector_dim=None,target_k="unfixed",target_c3_label=None,
    n_shells,n_samples,n_discard,n_chains,n_iter,learning_rate,
    diag_shift,rng_seed,chunk_size,output_dir,prefix,method,
):
    """Optimize a C3-constrained, translation-breaking Slater x Jastrow."""
    import jax
    jax.config.update("jax_enable_x64",True)
    import netket as nk

    projected=target_group is not None
    t0=time.perf_counter()
    hi,_,H_fast,graph=make_netket_hamiltonian(g,nf,V)
    star=ruby_star_formfactor(g)
    distances,shells,shell_counts=minimum_image_shells(g,n_shells=n_shells)

    Qseed=one_body_seed(g,nf,seed_name,seed_amplitude,star)
    seed_data=c3_seed_subspace(
        g,Qseed,nf,target_manybody_m=target_c3_m
    )

    maps=(
        np.asarray([op.site_map for op in target_group],np.int32)
        if projected else None
    )
    chars=(
        np.asarray(target_chars,complex) if projected else None
    )
    model=build_c3_constrained_netket_model(
        seed_data,shells,True,
        projection_site_maps=maps,
        projection_characters=chars,
    )
    sampler=nk.sampler.MetropolisFermionHop(hi,graph=graph,n_chains=n_chains)
    vstate=nk.vqs.MCState(
        sampler,model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard,
        chunk_size=chunk_size,
        seed=rng_seed,
        sampler_seed=rng_seed+104729,
    )
    setup_s=time.perf_counter()-t0

    tw=time.perf_counter()
    warm_stats=vstate.expect(H_fast)
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
    print(
        f"[C3 ansatz] seed={seed_name} counts={seed_data['counts']} "
        f"C3:m{seed_data['manybody_m']} stored_params={vstate.n_parameters} "
        f"physical_orbital_dof={seed_data['physical_real_dof']}",
        flush=True,
    )
    log=nk.logging.RuntimeLog()
    topt=time.perf_counter()
    driver.run(n_iter=n_iter,out=log,show_progress=True)
    opt_s=time.perf_counter()-topt

    te=time.perf_counter()
    vstate.sample()
    final_stats=vstate.expect(H_fast)
    mc_eval_s=time.perf_counter()-te

    tp=time.perf_counter()
    psi_nk=np.asarray(vstate.to_array(normalize=True))
    psi=reorder_netket_vector_to_native(hi,psi_nk,H.states,g.Ns)
    Hpsi=H.operator@psi
    E=float(np.vdot(psi,Hpsi).real)
    var=float(max(0.0,np.vdot(Hpsi,Hpsi).real-E**2))
    obs=benchmark_observables(g,H.states,psi,native_observables)
    fid=float(np.sum(np.abs(ed_gs.conj().T@psi)**2).real)
    if projected:
        _,weight=project_to_characters(target_group,target_chars,psi)
        residual=projected_sector_residual(target_group,target_chars,psi)
    else:
        weight=None;residual=None
    post_s=time.perf_counter()-tp

    d=output_dir/f"{prefix}__sector{target_sector_id}__{seed_name}"
    if not projected:
        d=output_dir/f"{prefix}__m{target_c3_m}__{seed_name}"
    d.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        d/"final_state.npz",
        psi_native=psi,
        native_states=H.states,
        shell_distances=distances,
        shell_counts=shell_counts,
        c3_counts=np.asarray(seed_data["counts"],int),
    )
    hist={}
    try:
        en=log.data["Energy"]
        hist["iteration"]=[int(x) for x in np.asarray(en.iters)]
        hist["E_mean"]=[float(np.real(x)) for x in np.asarray(en.Mean)]
        hist["E_sigma"]=[float(x) for x in np.asarray(en.Sigma)]
    except Exception:
        pass
    atomic_json(d/"history.json",hist)

    row={
        "method":method,
        "seed":seed_name,
        "family":prefix,
        "projection":(
            "none; C3-constrained base"
            if not projected else "full target sector; C3-constrained base"
        ),
        "target_sector_id":target_sector_id,
        "target_sector_ground_dimension":target_sector_dim,
        "target_k":target_k,
        "target_c3":target_c3_label or f"C3:m{target_c3_m}",
        "target_c3_m":int(target_c3_m),
        "c3_occupation_counts":json.dumps(seed_data["counts"]),
        "c3_seed_overlap_score":seed_data["seed_overlap_score"],
        "physical_orbital_dof":seed_data["physical_real_dof"],
        "n_parameters":int(vstate.n_parameters),
        "n_jastrow_parameters":int(len(distances)),
        "target_sector_weight":None if weight is None else float(weight),
        "projected_sector_residual":None if residual is None else float(residual),
        "E":E,
        "E_per_site":E/g.Ns,
        "delta_E_vs_ED":E-ed_E0,
        "relative_E_error":abs((E-ed_E0)/ed_E0) if ed_E0 else None,
        "fidelity_ground_subspace":fid,
        "mc_E":float(np.real(final_stats.mean)),
        "mc_E_error":stats_error(final_stats),
        "mc_variance":(
            float(np.real(final_stats.variance))
            if hasattr(final_stats,"variance") else None
        ),
        "exact_energy_variance":var,
        "exact_energy_variance_per_site":var/g.Ns,
        "warmup_mc_E":float(np.real(warm_stats.mean)),
        "time_build_s":setup_s,
        "time_jit_warmup_s":warmup_s,
        "time_solve_or_opt_s":opt_s,
        "time_mc_final_eval_s":mc_eval_s,
        "time_postprocess_s":post_s,
        "time_total_s":setup_s+warmup_s+opt_s+mc_eval_s+post_s,
        **obs.as_dict(),
    }
    atomic_json(d/"summary.json",row)
    return row,psi


def parser():
    p=argparse.ArgumentParser(
        description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--nf",type=int,default=6)
    p.add_argument("--t",type=float,default=1.0)
    p.add_argument("--t1",type=float,default=0.5)
    p.add_argument("--t2",type=float,default=-0.5)
    p.add_argument("--V",type=float,default=2.0)
    p.add_argument(
        "--seeds",nargs="+",
        default=["normal","sd","isd","lc_same","lc_opposite"]
    )
    p.add_argument("--seed-amplitude",type=float,default=0.08)
    p.add_argument("--jastrow-shells",type=int,default=4)
    p.add_argument("--n-samples",type=int,default=4096)
    p.add_argument("--n-discard",type=int,default=16)
    p.add_argument("--n-chains",type=int,default=16)
    p.add_argument("--n-iter",type=int,default=300)
    p.add_argument("--learning-rate",type=float,default=0.03)
    p.add_argument("--diag-shift",type=float,default=0.05)
    p.add_argument("--chunk-size",type=int,default=512)
    p.add_argument("--rng-seed",type=int,default=2026)
    p.add_argument("--ed-k",type=int,default=12)
    p.add_argument("--ed-degeneracy-tol",type=float,default=1e-8)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--skip-vap",action="store_true")
    p.add_argument("--output",default="results/projection_compare_hex18")
    return p


def main():
    a=parser().parse_args()
    out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=True)

    g=native_geometry.RubyPBC("ring3",t=a.t,t1=a.t1,t2=a.t2)
    H,evals,ed_gs,ed_row=solve_ed(
        g,a.nf,a.V,k=a.ed_k,degeneracy_tol=a.ed_degeneracy_tol
    )
    if ed_row.get("ground_manifold_truncated"):
        raise RuntimeError("ED ground manifold truncated; increase --ed-k")

    full_group,full_group_name,full_sectors,c3_group,c3_sectors=build_ed_sectors(
        g,H,ed_gs,a.ed_degeneracy_tol
    )

    atomic_json(out/"settings.json",{
        **vars(a),
        "cluster":"ring3",
        "Ns":g.Ns,
        "full_projection_group":full_group_name,
        "full_sector_count":len(full_sectors),
        "c3_sector_count":len(c3_sectors),
    })
    atomic_json(out/"geometry.json",g.metadata())
    atomic_json(out/"ed_projection_sectors.json",{
        "ground_manifold_dimension":int(ed_gs.shape[1]),
        "full_group":full_group_name,
        "full_sectors":[sector_record(i,s) for i,s in enumerate(full_sectors)],
        "c3_content":[sector_record(i,s) for i,s in enumerate(c3_sectors)],
    })

    ed=dict(ed_row)
    ed["method"]="ED"
    ed["target_k"]="ground manifold"
    ed["target_c3"]="ground manifold"
    rows=[ed]

    print(
        f"[ED] E0={ed['E']:.12g}, ground_dim={ed_gs.shape[1]}, "
        f"full_sectors={len(full_sectors)}"
    )
    for i,sec in enumerate(full_sectors):
        qn=sec["quantum_numbers"]
        print(
            f"  [ED sector {i}] k={qn['momentum']['name']} "
            f"q={qn['momentum']['q']} {qn['c3']['label']} "
            f"dim={sec['dimension']}"
        )

    run_id=0

    # A. Full unconstrained SJ -> PAV.
    for seed in a.seeds:
        run_id+=1
        d=out/f"slater_jastrow__{seed}"
        summary,psi=load_summary_state(d,H) if a.resume else (None,None)
        if summary is None:
            print(f"\n[full raw SJ] seed={seed}",flush=True)
            run_vmc_one(
                g,a.nf,a.V,H,ed_gs,float(evals[0]),
                ansatz="slater_jastrow",
                seed_name=seed,
                seed_amplitude=a.seed_amplitude,
                n_shells=a.jastrow_shells,
                n_samples=a.n_samples,
                n_discard=a.n_discard,
                n_chains=a.n_chains,
                n_iter=a.n_iter,
                learning_rate=a.learning_rate,
                diag_shift=a.diag_shift,
                rng_seed=a.rng_seed+1009*run_id,
                chunk_size=a.chunk_size,
                output_dir=out,
                validate_hamiltonian=(run_id==1),
            )
            summary,psi=load_summary_state(d,H)
        else:
            print(f"\n[resume full raw SJ] seed={seed}",flush=True)

        for sid,sec in enumerate(full_sectors):
            result=projected_row(
                method="PAV-Slater-Jastrow",
                seed=seed,psi=psi,H=H,ed_gs=ed_gs,ed_row=ed_row,
                group=full_group,sec=sec,sector_id=sid,
                projection_name=full_group_name,
            )
            if result is None:
                continue
            row,ppsi=result
            rows.append(row)
            pd=out/f"pav__sector{sid}__{seed}"
            pd.mkdir(parents=True,exist_ok=True)
            atomic_json(pd/"summary.json",row)
            np.savez_compressed(
                pd/"final_state.npz",psi_native=ppsi,native_states=H.states
            )
        write_csv(out/"projection_comparison_all.csv",rows)

    # B. C3-constrained raw SJ -> full PAV.
    c3_raw={}
    target_ms=sorted({
        int(sec["quantum_numbers"]["c3"]["m"])
        for sec in full_sectors
        if sec["quantum_numbers"]["c3"]["m"] is not None
    })
    for m in target_ms:
        for seed in a.seeds:
            run_id+=1
            d=out/f"c3_base__m{m}__{seed}"
            summary,psi=load_summary_state(d,H) if a.resume else (None,None)
            if summary is None:
                print(f"\n[C3 raw SJ] seed={seed}, C3:m{m}",flush=True)
                row,psi=run_c3_base_or_vap(
                    g,a.nf,a.V,H,ed_gs,float(evals[0]),
                    seed_name=seed,
                    seed_amplitude=a.seed_amplitude,
                    target_c3_m=m,
                    target_group=None,
                    target_chars=None,
                    target_sector_id=None,
                    target_sector_dim=None,
                    target_k="unfixed",
                    target_c3_label=f"C3:m{m}",
                    n_shells=a.jastrow_shells,
                    n_samples=a.n_samples,
                    n_discard=a.n_discard,
                    n_chains=a.n_chains,
                    n_iter=a.n_iter,
                    learning_rate=a.learning_rate,
                    diag_shift=a.diag_shift,
                    rng_seed=a.rng_seed+1009*run_id,
                    chunk_size=a.chunk_size,
                    output_dir=out,
                    prefix="c3_base",
                    method="C3-Base-Slater-Jastrow",
                )
                summary=row
            else:
                print(f"\n[resume C3 raw SJ] seed={seed}, C3:m{m}",flush=True)
            c3_raw[(m,seed)]=psi

            for sid,sec in enumerate(full_sectors):
                sm=sec["quantum_numbers"]["c3"]["m"]
                if sm is None or int(sm)!=m:
                    continue
                result=projected_row(
                    method="C3-PAV-Slater-Jastrow",
                    seed=seed,psi=psi,H=H,ed_gs=ed_gs,ed_row=ed_row,
                    group=full_group,sec=sec,sector_id=sid,
                    projection_name=full_group_name,
                    extra={
                        "base_constraint":"C3 only; primitive translation free",
                        "base_c3_m":m,
                        "c3_occupation_counts":summary.get("c3_occupation_counts"),
                        "physical_orbital_dof":summary.get("physical_orbital_dof"),
                    },
                )
                if result is None:
                    continue
                row,ppsi=result
                rows.append(row)
                pd=out/f"c3_pav__sector{sid}__{seed}"
                pd.mkdir(parents=True,exist_ok=True)
                atomic_json(pd/"summary.json",row)
                np.savez_compressed(
                    pd/"final_state.npz",psi_native=ppsi,native_states=H.states
                )
            write_csv(out/"projection_comparison_all.csv",rows)

    if not a.skip_vap:
        # C. Full VAP.
        for sid,sec in enumerate(full_sectors):
            qn=sec["quantum_numbers"]
            for seed in a.seeds:
                run_id+=1
                d=out/f"full_vap_sj__sector{sid}__{seed}"
                summary,_=load_summary_state(d,H) if a.resume else (None,None)
                if summary is None:
                    print(
                        f"\n[full VAP] seed={seed}, sector={sid}, "
                        f"k={qn['momentum']['name']} {qn['c3']['label']}",
                        flush=True,
                    )
                    row=run_full_vap_one(
                        g,a.nf,a.V,H,ed_gs,float(evals[0]),
                        full_group,sec["characters"],
                        ground_sectors=full_sectors,
                        sector_id=sid,
                        sector_dim=int(sec["dimension"]),
                        method_name="Full-VAP-Slater-Jastrow",
                        family_name="full_vap_sj",
                        dir_prefix="full_vap_sj",
                        projection_name=full_group_name,
                        seed_name=seed,
                        seed_amplitude=a.seed_amplitude,
                        n_shells=a.jastrow_shells,
                        n_samples=a.n_samples,
                        n_discard=a.n_discard,
                        n_chains=a.n_chains,
                        n_iter=a.n_iter,
                        learning_rate=a.learning_rate,
                        diag_shift=a.diag_shift,
                        rng_seed=a.rng_seed+1009*run_id,
                        chunk_size=a.chunk_size,
                        output_dir=out,
                    )
                else:
                    row=summary
                    print(
                        f"\n[resume full VAP] seed={seed}, sector={sid}",
                        flush=True,
                    )
                row["target_k"]=qn["momentum"]["name"]
                row["target_q"]=json.dumps(qn["momentum"]["q"])
                row["target_c3"]=qn["c3"]["label"]
                row["target_c3_m"]=qn["c3"]["m"]
                rows.append(row)
                write_csv(out/"projection_comparison_all.csv",rows)

        # D. C3-constrained VAP, but the projector is the same full target sector.
        for sid,sec in enumerate(full_sectors):
            qn=sec["quantum_numbers"]
            m=qn["c3"]["m"]
            if m is None:
                continue
            for seed in a.seeds:
                run_id+=1
                d=out/f"c3_vap__sector{sid}__{seed}"
                summary,_=load_summary_state(d,H) if a.resume else (None,None)
                if summary is None:
                    print(
                        f"\n[C3 VAP] seed={seed}, sector={sid}, "
                        f"k={qn['momentum']['name']} {qn['c3']['label']}",
                        flush=True,
                    )
                    row,_=run_c3_base_or_vap(
                        g,a.nf,a.V,H,ed_gs,float(evals[0]),
                        seed_name=seed,
                        seed_amplitude=a.seed_amplitude,
                        target_c3_m=int(m),
                        target_group=full_group,
                        target_chars=sec["characters"],
                        target_sector_id=sid,
                        target_sector_dim=int(sec["dimension"]),
                        target_k=qn["momentum"]["name"],
                        target_c3_label=qn["c3"]["label"],
                        n_shells=a.jastrow_shells,
                        n_samples=a.n_samples,
                        n_discard=a.n_discard,
                        n_chains=a.n_chains,
                        n_iter=a.n_iter,
                        learning_rate=a.learning_rate,
                        diag_shift=a.diag_shift,
                        rng_seed=a.rng_seed+1009*run_id,
                        chunk_size=a.chunk_size,
                        output_dir=out,
                        prefix="c3_vap",
                        method="C3-VAP-Slater-Jastrow",
                    )
                else:
                    row=summary
                    print(
                        f"\n[resume C3 VAP] seed={seed}, sector={sid}",
                        flush=True,
                    )
                rows.append(row)
                write_csv(out/"projection_comparison_all.csv",rows)

    best=[ed]
    for method in METHOD_ORDER[1:]:
        cand=[
            r for r in rows
            if r.get("method")==method and r.get("E") is not None
        ]
        if cand:
            best.append(min(cand,key=lambda r:float(r["E"])))

    write_csv(out/"projection_comparison_best.csv",best)
    atomic_json(out/"projection_comparison_best.json",best)

    print("\n=== ED / PAV / Full-VAP / C3-PAV / C3-VAP ===")
    for r in best:
        print(
            f"{r['method']:28s} seed={str(r.get('seed','')):11s} "
            f"k={str(r.get('target_k','')):7s} "
            f"C3={str(r.get('target_c3','')):7s} "
            f"E={float(r['E']):.12g} "
            f"dE={float(r.get('delta_E_vs_ED',0.0)):.3e} "
            f"F={float(r.get('fidelity_ground_subspace',1.0)):.6f} "
            f"var={float(r.get('exact_energy_variance',0.0)):.4g} "
            f"SLC+={float(r['S_LC_same']):.5g} "
            f"SLC-={float(r['S_LC_opposite']):.5g}"
        )
    print("Saved",out/"projection_comparison_all.csv")
    print("Saved",out/"projection_comparison_best.csv")


if __name__=="__main__":
    main()
