#!/usr/bin/env python3
"""Large-supercell VMC runner: no ED, no full-state enumeration.

This script is the scalable path for L1 x L2 copies of the 18-site ring3
variational supercell. It keeps the staged residual-symmetry optimization

  Raw -> Tint (3 images) -> Tint x C3v (18 images) -> optional TR,

but removes all exact-diagonalization/full-basis work.

Large-system optimizations:
  * packed orbital parameter PyTree: one complex 18 x Nf leaf rather than
    one leaf per superlattice momentum block;
  * sparse Jastrow and sparse nearest-neighbor backflow gather/scatter;
  * matrix-free QGTOnTheFly + iterative CG by default;
  * optional QGTJacobianPyTree for timing comparisons;
  * reduced fermion-hop sweep_size (default Nf instead of Ns);
  * no to_array(), fidelity, exact variance or Lanczos step.

The determinant itself is still evaluated exactly for every proposed
configuration. A true Woodbury determinant-ratio sampler would require
sampler-state caches for every chain and every projected image; that is kept
separate from this safe stateless logpsi optimization path.
"""
from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NATIVE=REPO/"ed_benchmark"/"native_ed"
sys.path.insert(0,str(NATIVE))

import geometry as native_geometry

from run_benchmark import (
    atomic_json,
    make_netket_hamiltonian,
    stats_error,
)
from vmc_ansatz import jastrow_channels, bond_backflow_channels
from supercell_ansatz import (
    seed_bloch_blocks,
    build_supercell_netket_model,
    extract_supercell_parameters,
)
from symmetry_projection import (
    internal_translation_site_maps,
    internal_c3v_space_group_site_maps,
    internal_gamma_c3v_characters,
)


def _history(log):
    out={}
    try:
        en=log.data["Energy"]
        out["iteration"]=[int(x) for x in np.asarray(en.iters)]
        out["E_mean"]=[float(np.real(x)) for x in np.asarray(en.Mean)]
        out["E_sigma"]=[float(x) for x in np.asarray(en.Sigma)]
    except Exception:
        pass
    return out


def _write_csv(path,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _save_params(path,params,seed_data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={
        "kpoints":np.asarray(seed_data["kpoints"],float),
        "occupations_per_k":np.asarray(seed_data["occupations_per_k"],np.int32),
        "total_supercell_momentum":np.asarray(
            seed_data["total_supercell_momentum"],float
        ),
        "variational_v":np.asarray(params.get("v",[]),float),
        "variational_eta":np.asarray(params.get("eta",[]),complex),
    }
    for ik,block in enumerate(params["blocks"]):
        payload[f"variational_M_{ik}"]=np.asarray(block,complex)
    if params.get("tr_phase",None) is not None:
        payload["variational_tr_phase"]=np.asarray(
            params["tr_phase"],np.float64
        )
    np.savez_compressed(path,**payload)


def _load_params(directory):
    directory=Path(directory)
    state=directory/"final_params.npz"
    summary=directory/"summary.json"
    if not state.exists() or not summary.exists():
        raise FileNotFoundError(
            f"Missing saved large-supercell stage: {state} / {summary}"
        )
    with np.load(state,allow_pickle=False) as z:
        counts=np.asarray(z["occupations_per_k"],int)
        params={
            "blocks":[
                np.asarray(z[f"variational_M_{ik}"],complex)
                for ik in range(len(counts))
            ],
            "v":np.asarray(z["variational_v"],float),
            "eta":np.asarray(z["variational_eta"],complex),
            "tr_phase":(
                float(np.asarray(z["variational_tr_phase"]))
                if "variational_tr_phase" in z.files else None
            ),
        }
    return json.loads(summary.read_text(encoding="utf-8")),params


def _memory_estimates(n_parameters,n_samples):
    # Complex-mode Jacobians can split real/imag directions. These are only
    # rough upper-scale diagnostics, not allocations performed by this script.
    p=float(n_parameters)
    s=float(n_samples)
    return {
        "dense_qgt_estimate_gib_complex128":p*p*16.0/2**30,
        "jacobian_estimate_gib_complex128":s*p*16.0/2**30,
    }


def _qgt_and_solver(nk,jax,args):
    if args.qgt=="onthefly":
        qgt=partial(
            nk.optimizer.qgt.QGTOnTheFly,
            chunk_size=args.qgt_chunk_size,
        )
    elif args.qgt=="pytree":
        qgt=partial(
            nk.optimizer.qgt.QGTJacobianPyTree,
            mode="complex",
            chunk_size=args.qgt_chunk_size,
        )
    else:
        raise ValueError(args.qgt)

    if args.sr_solver=="cg":
        solver=partial(
            jax.scipy.sparse.linalg.cg,
            tol=args.cg_tol,
            atol=0.0,
            maxiter=args.cg_maxiter,
        )
    elif args.sr_solver=="gmres":
        solver=partial(
            jax.scipy.sparse.linalg.gmres,
            tol=args.cg_tol,
            atol=0.0,
            maxiter=args.cg_maxiter,
        )
    else:
        raise ValueError(args.sr_solver)
    return qgt,solver


def _stage(
    *,
    stage_name,
    directory,
    g,nf,V,
    hi,H_fast,graph,
    seed_data,
    shell_mats,bf_mats,
    use_backflow,
    initial_params,
    maps,chars,
    point_label,
    time_reversal_even,
    n_iter,learning_rate,
    args,
    rng_seed,
):
    import jax
    jax.config.update("jax_enable_x64",True)
    import netket as nk

    blocks=(
        seed_data["blocks"]
        if initial_params is None else initial_params["blocks"]
    )
    v0=None if initial_params is None else initial_params.get("v",None)
    eta0=None if initial_params is None else initial_params.get("eta",None)
    tr0=(
        0.0 if initial_params is None
        else float(initial_params.get("tr_phase",0.0) or 0.0)
    )

    t0=time.perf_counter()
    model=build_supercell_netket_model(
        blocks,
        seed_data["sc_coords"],
        seed_data["internal_sites"],
        seed_data["kpoints"],
        shell_mats,
        use_jastrow=True,
        backflow_matrices=(bf_mats if use_backflow else None),
        initial_v=v0,
        initial_eta=eta0,
        point_site_maps=maps,
        point_characters=chars,
        time_reversal_even=bool(time_reversal_even),
        initial_tr_phase=tr0,
        sparse_correlators=True,
        packed_orbitals=True,
    )

    sweep_size=(
        int(args.sweep_size)
        if args.sweep_size is not None
        else int(nf)
    )
    sampler_kwargs={
        "graph":graph,
        "n_chains":args.n_chains,
        "sweep_size":sweep_size,
    }
    if args.sampler_chunk_size is not None:
        sampler_kwargs["chunk_size"]=int(args.sampler_chunk_size)

    sampler=nk.sampler.MetropolisFermionHop(hi,**sampler_kwargs)
    vstate=nk.vqs.MCState(
        sampler,model,
        n_samples=args.n_samples,
        n_discard_per_chain=args.n_discard,
        chunk_size=args.chunk_size,
        seed=rng_seed,
        sampler_seed=rng_seed+104729,
    )
    setup_s=time.perf_counter()-t0

    tw=time.perf_counter()
    warm=vstate.expect(H_fast)
    warmup_s=time.perf_counter()-tw

    qgt_ctor,solver=_qgt_and_solver(nk,jax,args)
    optimizer=nk.optimizer.Sgd(learning_rate=learning_rate)
    try:
        sr=nk.optimizer.SR(
            qgt=qgt_ctor,
            solver=solver,
            diag_shift=args.diag_shift,
            solver_restart=True,
        )
    except TypeError:
        # Compatibility with older 3.x releases that do not expose
        # solver_restart in the public constructor.
        sr=nk.optimizer.SR(
            qgt=qgt_ctor,
            solver=solver,
            diag_shift=args.diag_shift,
        )

    driver=nk.driver.VMC(
        H_fast,optimizer,
        variational_state=vstate,
        preconditioner=sr,
    )
    n_images=1 if maps is None else int(len(maps))
    print(
        f"[large {stage_name}] Ns={g.Ns} Nf={nf} "
        f"params={vstate.n_parameters} images={n_images} "
        f"sweep={sweep_size} qgt={args.qgt}/{args.sr_solver}",
        flush=True,
    )

    log=nk.logging.RuntimeLog()
    topt=time.perf_counter()
    driver.run(n_iter=n_iter,out=log,show_progress=True)
    opt_s=time.perf_counter()-topt

    te=time.perf_counter()
    vstate.sample()
    final_stats=vstate.expect(H_fast)
    final_eval_s=time.perf_counter()-te

    params=extract_supercell_parameters(
        vstate.parameters,
        len(seed_data["kpoints"]),
        occupations_per_k=seed_data["occupations_per_k"],
    )

    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    _save_params(directory/"final_params.npz",params,seed_data)
    atomic_json(directory/"history.json",_history(log))

    def _scalar_stat(obj,name):
        try:
            x=getattr(obj,name)
            arr=np.asarray(x)
            return float(np.real(arr.reshape(-1)[0]))
        except Exception:
            return None

    acceptance=None
    try:
        acceptance=float(np.asarray(vstate.sampler_state.acceptance).mean())
    except Exception:
        pass

    summary={
        "stage":stage_name,
        "Ns":int(g.Ns),"Nf":int(nf),"V":float(V),
        "L1":int(args.L1),"L2":int(args.L2),
        "seed":args._active_seed,
        "correlator":args.correlator,
        "point_label":point_label,
        "time_reversal_restored":bool(time_reversal_even),
        "explicit_projection_images":n_images,
        "n_parameters":int(vstate.n_parameters),
        "packed_orbitals":True,
        "sparse_correlators":True,
        "qgt":args.qgt,
        "sr_solver":args.sr_solver,
        "cg_tol":float(args.cg_tol),
        "cg_maxiter":int(args.cg_maxiter),
        "diag_shift":float(args.diag_shift),
        "n_samples":int(args.n_samples),
        "n_chains":int(args.n_chains),
        "sweep_size":int(sweep_size),
        "default_netket_sweep_size_would_be":int(g.Ns),
        "sweep_reduction_vs_default":float(g.Ns/sweep_size),
        "chunk_size":(
            None if args.chunk_size is None else int(args.chunk_size)
        ),
        "qgt_chunk_size":(
            None if args.qgt_chunk_size is None
            else int(args.qgt_chunk_size)
        ),
        "warmup_E":float(np.real(warm.mean)),
        "warmup_E_error":stats_error(warm),
        "mc_E":float(np.real(final_stats.mean)),
        "mc_E_error":stats_error(final_stats),
        "mc_tau_corr":_scalar_stat(final_stats,"tau_corr"),
        "mc_R_hat":_scalar_stat(final_stats,"R_hat"),
        "sampler_acceptance":acceptance,
        "time_setup_s":float(setup_s),
        "time_jit_warmup_s":float(warmup_s),
        "time_opt_s":float(opt_s),
        "time_per_iteration_s":float(opt_s/max(int(n_iter),1)),
        "time_final_eval_s":float(final_eval_s),
        "n_iter":int(n_iter),
        "learning_rate":float(learning_rate),
        "seed_shell_gap":seed_data["shell_gap"],
        "total_supercell_momentum":[
            float(x) for x in seed_data["total_supercell_momentum"]
        ],
        "occupations_per_k":[
            int(x) for x in seed_data["occupations_per_k"]
        ],
        **_memory_estimates(vstate.n_parameters,args.n_samples),
    }
    atomic_json(directory/"summary.json",summary)
    return summary,params


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
        "--seeds",nargs="+",default=["normal","lc_opposite"]
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
    p.add_argument("--c3v-irrep",choices=["A1","A2"],default="A2")

    p.add_argument("--raw-iter",type=int,default=50)
    p.add_argument("--tint-iter",type=int,default=50)
    p.add_argument("--spatial-iter",type=int,default=50)
    p.add_argument("--tr-iter",type=int,default=50)
    p.add_argument("--raw-learning-rate",type=float,default=0.03)
    p.add_argument("--tint-learning-rate",type=float,default=0.01)
    p.add_argument("--spatial-learning-rate",type=float,default=0.01)
    p.add_argument("--tr-learning-rate",type=float,default=0.01)
    p.add_argument("--restore-time-reversal",action="store_true")

    p.add_argument("--n-samples",type=int,default=2048)
    p.add_argument("--n-discard",type=int,default=8)
    p.add_argument("--n-chains",type=int,default=32)
    p.add_argument(
        "--sweep-size",type=int,default=None,
        help="Metropolis proposals between returned samples. Default=Nf; NetKet default would be Ns.",
    )
    p.add_argument("--sampler-chunk-size",type=int,default=None)
    p.add_argument("--chunk-size",type=int,default=256)

    p.add_argument(
        "--qgt",choices=["onthefly","pytree"],default="onthefly"
    )
    p.add_argument(
        "--sr-solver",choices=["cg","gmres"],default="cg"
    )
    p.add_argument("--qgt-chunk-size",type=int,default=128)
    p.add_argument("--cg-tol",type=float,default=1e-5)
    p.add_argument("--cg-maxiter",type=int,default=100)
    p.add_argument("--diag-shift",type=float,default=0.05)

    p.add_argument("--skip-raw",action="store_true")
    p.add_argument("--reuse-raw",action="store_true")
    p.add_argument("--reuse-tint",action="store_true")
    p.add_argument("--reuse-spatial",action="store_true")
    p.add_argument("--rng-seed",type=int,default=2026)
    p.add_argument(
        "--output",default="results/large_supercell_2x2_V5p5"
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
    nf=(6*a.L1*a.L2 if a.nf is None else int(a.nf))
    if not 0<nf<g.Ns:
        raise ValueError("nf must satisfy 0<nf<Ns")

    # Build Hamiltonian, graph and correlator channels exactly once.
    import jax
    jax.config.update("jax_enable_x64",True)
    hi,_,H_fast,graph=make_netket_hamiltonian(g,nf,a.V)
    _,_,shell_mats,_=jastrow_channels(
        g,mode=a.jastrow_mode,n_shells=a.jastrow_shells
    )
    _,bf_mats,_=bond_backflow_channels(g)
    use_backflow=(a.correlator=="backflow")

    tint_ops=internal_translation_site_maps(g)
    tint_maps=np.asarray([p for _,p in tint_ops],np.int32)
    tint_chars=np.ones(3,np.complex128)

    spatial_ops=internal_c3v_space_group_site_maps(g)
    spatial_maps=np.asarray([p for _,p in spatial_ops],np.int32)
    spatial_chars=internal_gamma_c3v_characters(a.c3v_irrep)

    settings={
        **vars(a),
        "Ns":int(g.Ns),"Nf":int(nf),
        "n_supercells":int(a.L1*a.L2),
        "variational_supercell_sites":18,
        "large_system_no_ed":True,
        "sparse_correlators":True,
        "packed_orbitals":True,
        "internal_translation_images":3,
        "full_residual_spatial_images":18,
        "notes":[
            "No ED/full-state enumeration is performed.",
            "QGTOnTheFly+CG is matrix-free and is the default.",
            "Metropolis sweep_size defaults to Nf rather than Ns.",
            "Determinant evaluation remains stateless/exact; no inverse cache is used.",
        ],
    }
    atomic_json(out/"settings.json",settings)
    atomic_json(out/"geometry.json",g.metadata())

    rows=[]
    for iseed,seed in enumerate(a.seeds):
        a._active_seed=seed
        seed_data=seed_bloch_blocks(
            g,nf,a.L1,a.L2,seed,a.seed_amplitude
        )
        ktot=np.asarray(seed_data["total_supercell_momentum"],float)
        if np.linalg.norm(ktot-np.rint(ktot))>1e-9:
            raise RuntimeError(
                f"seed={seed}: fixed k-block occupations have non-Gamma total "
                f"supercell momentum {ktot.tolist()}. Tint x C3v cannot be "
                "applied consistently; choose a closed-shell size/filling."
            )

        base_seed=a.rng_seed+10007*iseed
        params=None

        raw_dir=out/f"large_raw_{a.correlator}__{seed}"
        if a.reuse_raw or a.skip_raw:
            if raw_dir.exists():
                raw_summary,params=_load_params(raw_dir)
                rows.append(raw_summary)
            elif a.skip_raw:
                params=None
            else:
                raise FileNotFoundError(raw_dir)
        if not a.skip_raw and not a.reuse_raw:
            raw_summary,params=_stage(
                stage_name="Raw",
                directory=raw_dir,
                g=g,nf=nf,V=a.V,
                hi=hi,H_fast=H_fast,graph=graph,
                seed_data=seed_data,
                shell_mats=shell_mats,bf_mats=bf_mats,
                use_backflow=use_backflow,
                initial_params=None,
                maps=None,chars=None,
                point_label=None,
                time_reversal_even=False,
                n_iter=a.raw_iter,
                learning_rate=a.raw_learning_rate,
                args=a,rng_seed=base_seed+1,
            )
            rows.append(raw_summary)

        tint_dir=out/f"large_tint_{a.correlator}__Gamma__{seed}"
        if a.reuse_tint:
            tint_summary,params=_load_params(tint_dir)
        else:
            tint_summary,params=_stage(
                stage_name="Tint",
                directory=tint_dir,
                g=g,nf=nf,V=a.V,
                hi=hi,H_fast=H_fast,graph=graph,
                seed_data=seed_data,
                shell_mats=shell_mats,bf_mats=bf_mats,
                use_backflow=use_backflow,
                initial_params=params,
                maps=tint_maps,chars=tint_chars,
                point_label="Gamma",
                time_reversal_even=False,
                n_iter=a.tint_iter,
                learning_rate=a.tint_learning_rate,
                args=a,rng_seed=base_seed+2,
            )
        rows.append(tint_summary)

        spatial_dir=out/(
            f"large_tint_c3v_{a.correlator}__{a.c3v_irrep}__{seed}"
        )
        if a.reuse_spatial:
            spatial_summary,params=_load_params(spatial_dir)
        else:
            spatial_summary,params=_stage(
                stage_name="Tint+C3v",
                directory=spatial_dir,
                g=g,nf=nf,V=a.V,
                hi=hi,H_fast=H_fast,graph=graph,
                seed_data=seed_data,
                shell_mats=shell_mats,bf_mats=bf_mats,
                use_backflow=use_backflow,
                initial_params=params,
                maps=spatial_maps,chars=spatial_chars,
                point_label=a.c3v_irrep,
                time_reversal_even=False,
                n_iter=a.spatial_iter,
                learning_rate=a.spatial_learning_rate,
                args=a,rng_seed=base_seed+3,
            )
        rows.append(spatial_summary)

        if a.restore_time_reversal:
            tr_dir=out/(
                f"large_tint_c3vtr_{a.correlator}__"
                f"{a.c3v_irrep}__{seed}"
            )
            tr_summary,params=_stage(
                stage_name="Tint+C3v+TR",
                directory=tr_dir,
                g=g,nf=nf,V=a.V,
                hi=hi,H_fast=H_fast,graph=graph,
                seed_data=seed_data,
                shell_mats=shell_mats,bf_mats=bf_mats,
                use_backflow=use_backflow,
                initial_params=params,
                maps=spatial_maps,chars=spatial_chars,
                point_label=a.c3v_irrep,
                time_reversal_even=True,
                n_iter=a.tr_iter,
                learning_rate=a.tr_learning_rate,
                args=a,rng_seed=base_seed+4,
            )
            rows.append(tr_summary)

        _write_csv(out/"large_supercell_summary.csv",rows)

    print("\n[large supercell] summary")
    for r in rows:
        print(
            f"  seed={r['seed']:11s} stage={r['stage']:14s} "
            f"E={r['mc_E']:+.8f} +/- {r['mc_E_error']:.2e} "
            f"t/it={r['time_per_iteration_s']:.3f}s "
            f"images={r['explicit_projection_images']}"
        )
    print(f"Saved: {out}")


if __name__=="__main__":
    main()
