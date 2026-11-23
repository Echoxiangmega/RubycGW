#!/usr/bin/env python3
"""Checkpoint-resuming, projected-Woodbury VAP for Ruby 18-site supercells.

Copy BOTH this file and projected_fast_kernels.py into
RubycGW/ed_benchmark/vmc_zeroT, next to run_supercell_large.py.

Reads EXISTING run_supercell_vscan.py Raw final_params.npz; does not modify it.
Trains independent Tint, Tint*C3v and optional TR stages and saves compatible
final_params.npz / summary.json / history.json after EVERY stage.

The projected fast-local energy uses the exact rank-3 Raw Woodbury kernel for
all symmetry images. Optionally the experimental projected inverse-cache
sampler shares the same 18 image inverses between accepted Metropolis hops.
Both optimizations are validated against direct, non-Woodbury amplitudes;
any failed validation automatically falls back to the standard implementation,
or raises with --strict.

Example (after copying the files into ed_benchmark/vmc_zeroT):
  python ed_benchmark/vmc_zeroT/run_supercell_vap_fast.py \\
    --L1 2 --L2 2 --nf 24 --V 5.5 \\
    --seed lc_opposite \\
    --raw-stage /scratch/vscan/L2x2/seed_lc_opposite/continuation_up/V_p5p5/large_raw_backflow__lc_opposite \\
    --irreps A1 A2 --restore-tr --tint-iter 50 --spatial-iter 30 --tr-iter 30 \\
    --output /scratch/fast_vap/L2x2/V_p5p5/seed_lc_opposite

Requirements: the SAME original RubycGW checkout and NetKet/JAX environment
as the Raw scan, SciPy, NumPy, and (for Shared-SR) Optax.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
NATIVE=REPO/'ed_benchmark'/'native_ed'
if not NATIVE.is_dir():
    raise RuntimeError('Place this script at RubycGW/ed_benchmark/vmc_zeroT/')
sys.path.insert(0,str(NATIVE))

import geometry as native_geometry
from run_benchmark import atomic_json,make_netket_hamiltonian,stats_error
from vmc_ansatz import bond_backflow_channels,jastrow_channels
from supercell_ansatz import (
    seed_bloch_blocks,build_supercell_netket_model,
    make_fast_raw_local_energy_kernel,extract_supercell_parameters,
)
from symmetry_projection import (
    internal_translation_site_maps,
    internal_c3v_space_group_site_maps,
    internal_gamma_c3v_characters,
    c3v_irrep_characters,
)
import run_supercell_large as old
from projected_fast_kernels import (
    check_spatial_maps,
    make_projected_fast_local_energy_kernel,
    make_projected_cached_sampler,
)
from projected_fused_kernels import make_fused_projected_kernel


def _vtag(value):
    sign='m' if value<0 else 'p'
    return 'V_'+sign+f'{abs(float(value)):.8f}'.rstrip('0').rstrip('.').replace('.','p')


def _find_raw_stage(opt):
    if opt.raw_stage:
        p=Path(opt.raw_stage).expanduser().resolve()
    elif opt.scan_root:
        p=(Path(opt.scan_root).expanduser().resolve()
           /f'L{opt.L1}x{opt.L2}'/f'seed_{opt.seed}'
           /f'{opt.mode}_{opt.direction}'/_vtag(opt.V)
           /f'large_raw_backflow__{opt.seed}')
    else:
        raise ValueError('Provide --raw-stage or --scan-root')
    if not (p/'final_params.npz').is_file():
        raise FileNotFoundError(f'Raw checkpoint not found: {p/"final_params.npz"}')
    return p


def _check_checkpoint_compatibility(path,seed_data,g,nf,opt):
    with np.load(path/'final_params.npz',allow_pickle=False) as z:
        count=np.asarray(z['occupations_per_k'],int)
        k=np.asarray(z['kpoints'],float)
        expected=np.asarray(seed_data['occupations_per_k'],int)
        if not np.array_equal(count,expected):
            raise ValueError(f'Occupations per k differ: checkpoint={count}, now={expected}. '
                             'Do not mix different Bloch sectors or fillings.')
        if not np.allclose(k,np.asarray(seed_data['kpoints']),atol=1e-13):
            raise ValueError('Bloch k-point convention differs from saved Raw checkpoint.')
        for ik,n in enumerate(count):
            if z[f'variational_M_{ik}'].shape!=(18,int(n)):
                raise ValueError(f'M_{ik} shape mismatch')
    # If original summary exists, reject mismatched parameters instead of
    # silently projecting a state with a different Hamiltonian.
    meta=path/'summary.json'
    if meta.exists():
        d=json.loads(meta.read_text(encoding='utf-8'))
        for key,expected in [('V',opt.V),('Ns',g.Ns),('Nf',nf)]:
            if key in d and d[key] is not None and abs(float(d[key])-expected)>1e-10:
                raise ValueError(f'Raw summary {key}={d[key]} != expected {expected}')
    return old._load_params_file(path)


def _random_fixed_states(ns,nf,n,seed=51):
    rng=np.random.default_rng(seed)
    samples=np.zeros((n,ns),np.float64)
    for row in samples:
        row[rng.choice(ns,size=nf,replace=False)]=1.0
    return samples


def _direct_local_energy(model,pars,samples,g,V,chunk_size=16):
    """Independent reference using full recomputed projected logPsi per hop."""
    states=np.asarray(samples,np.float64)
    big=[]
    sizes=[]
    for x in states:
        connected=[]
        for e in g.edges:
            i,j=int(e.i),int(e.j)
            if x[i]==x[j]:
                continue
            new=x.copy();new[i],new[j]=new[j],new[i]
            parity=int(np.sum(x[min(i,j)+1:max(i,j)]))%2
            connected.append((new,float(g.t[e.kind])*(-1.0 if parity else 1.0)))
        sizes.append(connected)
    def batch_logs(x):
        out=[]
        for pos in range(0,len(x),chunk_size):
            part=np.asarray(x[pos:pos+chunk_size],np.float64)
            res=model.apply({'params':pars},part)
            out.append(np.asarray(res,np.complex128).reshape(-1))
        return np.concatenate(out)
    roots=batch_logs(states)
    ans=[]
    for row,(x,edges) in enumerate(zip(states,sizes)):
        diag=sum(V*x[e.i]*x[e.j] for e in g.edges if e.kind=='t')
        if not edges:
            ans.append(complex(diag));continue
        newlog=batch_logs([a for a,b in edges])
        val=diag+np.sum([amp*np.exp(l-roots[row]) for l,(cfg,amp) in zip(newlog,edges)])
        ans.append(val)
    return np.asarray(ans,dtype=np.complex128)


def _test_local_fast(vstate,fast_kernel,g,V,nf,*,nroots=3,atol=3e-7,rtol=3e-7):
    # Reject exact symmetry-projected nodes: local energy is undefined there.
    candidates=_random_fixed_states(g.Ns,nf,max(8,nroots*8))
    logs=np.asarray(vstate.model.apply({'params':vstate.parameters},candidates),complex)
    good=np.isfinite(np.real(logs)) & (np.real(logs)>np.max(np.real(logs))-45.)
    x=candidates[np.flatnonzero(good)[:nroots]]
    if len(x)<nroots:
        raise RuntimeError('Insufficient non-nodal samples for projected local-energy check')
    t=time.perf_counter()
    fast=np.asarray(fast_kernel(vstate.parameters,x),np.complex128)
    direct=_direct_local_energy(vstate.model,vstate.parameters,x,g,V)
    abs_err=np.abs(fast-direct)
    rel_err=abs_err/np.maximum(1.,np.abs(direct))
    report={
        'max_abs_error':float(abs_err.max()),
        'max_relative_error':float(rel_err.max()),
        'nroots':int(nroots),
        'elapsed_s':time.perf_counter()-t,
        'passed':bool(np.allclose(fast,direct,atol=atol,rtol=rtol)),
    }
    return report


def _test_sampler_cache(sampler,vstate,g,nf,*,nprop=3,atol=2e-7):
    import jax.numpy as jnp
    rng=np.random.default_rng(917)
    candidates=_random_fixed_states(g.Ns,nf,32)
    params=vstate.parameters
    logs=np.asarray(vstate.model.apply({'params':params},candidates),complex)
    good=np.isfinite(np.real(logs)) & (np.real(logs)>np.max(np.real(logs))-45.)
    roots=candidates[np.flatnonzero(good)[:4]]
    if len(roots)<4:
        raise RuntimeError('Insufficient non-nodal samples for projected sampler check')
    cache=sampler._init_fun(params,jnp.asarray(roots))
    up=jnp.asarray(rng.random((len(roots),nprop)))
    ua=jnp.asarray(rng.random((len(roots),nprop)))
    cache1,dlogp,_=sampler._advance_fun(cache,up,ua)
    final=np.asarray(cache1['x'],np.float64)
    orig_l=np.asarray(vstate.model.apply({'params':params},roots),complex)
    final_l=np.asarray(vstate.model.apply({'params':params},final),complex)
    ref=2*(np.real(final_l)-np.real(orig_l))
    err=float(np.max(np.abs(np.asarray(dlogp)-ref)))
    return {'max_log_probability_delta_error':err,'passed':bool(err<atol)}


def _make_group(g,stage,irrep):
    if stage=='tint':
        ops=internal_translation_site_maps(g)
        maps=np.asarray([p for _,p in ops],np.int32)
        chars=np.ones(len(maps),np.complex128)
    else:
        ops=internal_c3v_space_group_site_maps(g)
        maps=np.asarray([p for _,p in ops],np.int32)
        # E is a two-dimensional C3v irrep.  Its central character projector
        # selects the entire E *subspace*; it does not choose a particular
        # component of the doublet.  The common d_E/|G| normalization cancels.
        if irrep=='E':
            chars=np.tile(c3v_irrep_characters('E'),3).astype(np.complex128)
        else:
            chars=np.asarray(internal_gamma_c3v_characters(irrep),np.complex128)
    return check_spatial_maps(g,maps,chars,tr_even=(stage=='tr'))


def _save_params_and_summary(out,params,seed_data,summary,history):
    out.mkdir(parents=True,exist_ok=True)
    old._save_params(out/'final_params.npz',params,seed_data)
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'history.json',history)


def _read_progress_parameters(path):
    with np.load(path,allow_pickle=False) as z:
        counts=np.asarray(z['occupations_per_k'],int)
        return {
            'blocks':[np.asarray(z[f'variational_M_{k}'],complex) for k in range(len(counts))],
            'v':np.asarray(z['variational_v'],float),
            'eta':np.asarray(z['variational_eta'],complex),
            'tr_phase':(float(np.asarray(z['variational_tr_phase']))
                        if 'variational_tr_phase' in z.files else None),
        }


def _fused_kwargs(opt, g, nf, seed_data, shells, bfmat, maps, chars, tr,
                  orthonormalize):
    """Build a single source of truth for energy and LC channel orientation."""
    w_same, w_opp = old._current_channel_edge_weights(g)
    return dict(
        seed_data=seed_data,
        shell_matrices=shells,
        backflow_matrices=bfmat,
        edge_i=np.asarray([e.i for e in g.edges], np.int32),
        edge_j=np.asarray([e.j for e in g.edges], np.int32),
        edge_t=np.asarray([g.t[e.kind] for e in g.edges], np.float64),
        edge_has_interaction=np.asarray([e.kind == 't' for e in g.edges], np.float64),
        interaction_V=opt.V,
        site_maps=maps, characters=chars, w_same=w_same, w_opposite=w_opp,
        time_reversal_even=tr, orthonormalize_orbitals=orthonormalize,
    )


def _direct_current_loc(model, pars, roots, g, w_same, w_opp, batch_size=8):
    """Independent brute-force reference, batching all one-hop configurations.

    This still recomputes the full projected wavefunction, but avoids one
    Python/JAX model.apply() launch PER hopping.  Fixed-size padded batches
    prevent one extra XLA compilation for the final short batch.
    """
    roots=np.asarray(roots,np.float64)
    batch_size=max(1,int(batch_size))
    edge_i=np.asarray([e.i for e in g.edges],dtype=int)
    edge_j=np.asarray([e.j for e in g.edges],dtype=int)
    out=np.zeros((len(roots),2),np.complex128)
    root_logs=np.asarray(model.apply({'params':pars},roots),np.complex128).reshape(-1)
    configs=[]
    terms=[]
    for ib,x in enumerate(roots):
        for ie,(i,j) in enumerate(zip(edge_i,edge_j)):
            if x[i]==x[j]:
                continue
            src=i if x[i]>0.5 else j
            sign_dir=1.0 if src==i else -1.0
            y=x.copy()
            y[i],y[j]=y[j],y[i]
            parity=(-1.0)**int(np.sum(x[min(i,j)+1:max(i,j)]))
            configs.append(y)
            terms.append((ib,ie,sign_dir*parity))
    if not configs:
        return out
    log_hops=np.empty(len(configs),dtype=np.complex128)
    for pos in range(0,len(configs),batch_size):
        part=np.asarray(configs[pos:pos+batch_size],np.float64)
        n=len(part)
        if n<batch_size:
            part=np.concatenate((part,np.repeat(part[-1:],batch_size-n,axis=0)),axis=0)
        values=np.asarray(model.apply({'params':pars},part),np.complex128).reshape(-1)
        log_hops[pos:pos+n]=values[:n]
    for log1,(ib,ie,sgn) in zip(log_hops,terms):
        ratio=sgn*np.exp(log1-root_logs[ib])
        out[ib,0]+=1j*w_same[ie]*ratio
        out[ib,1]+=1j*w_opp[ie]*ratio
    return out


def _measure_fused_observables(vstate,g,hi,nf,kernel,chunk):
    """Evaluate charge S and both LC S via exact one-hop current identity.

    j_loc(x) = <x|J|psi>/<x|psi>. Hermiticity gives:
       <J^2> = < |j_loc|^2 >_{|psi|^2}
    with no two-hop NetKet J@J operator and no approximation.
    """
    import jax.numpy as jnp
    import netket as nk
    samples=np.asarray(vstate.samples)
    shape=samples.shape[:-1]
    roots=np.asarray(samples,np.float64).reshape((-1,g.Ns))
    chunk=max(1,int(chunk))
    js=np.empty(len(roots),np.complex128)
    jo=np.empty(len(roots),np.complex128)
    # Fixed batch shapes prevent repeated JAX compilation for a short tail.
    for start in range(0,len(roots),chunk):
        part=roots[start:start+chunk]
        length=len(part)
        if length<chunk:
            part=np.concatenate((part,np.repeat(part[-1:],chunk-length,axis=0)),axis=0)
        _,x,y=kernel(vstate.parameters,jnp.asarray(part))
        js[start:start+length]=np.asarray(x)[:length]
        jo[start:start+length]=np.asarray(y)[:length]
    sj=nk.stats.statistics(jnp.asarray(js.reshape(shape)))
    oj=nk.stats.statistics(jnp.asarray(jo.reshape(shape)))
    sj2=nk.stats.statistics(jnp.asarray((np.abs(js)**2).reshape(shape)))
    oj2=nk.stats.statistics(jnp.asarray((np.abs(jo)**2).reshape(shape)))
    a=float(np.real(sj.mean));b=float(np.real(oj.mean))
    sa=float(np.real(sj2.mean));sb=float(np.real(oj2.mean))
    out=old._charge_structure_from_samples(g,samples,nf)
    out.update({
        'J_same_mean':a,'J_same_mean_error':stats_error(sj),
        'J_opposite_mean':b,'J_opposite_mean_error':stats_error(oj),
        'S_LC_same':sa,'S_LC_same_error':stats_error(sj2),
        'S_LC_opposite':sb,'S_LC_opposite_error':stats_error(oj2),
        'S_LC_same_connected':float(sa-a*a),
        'S_LC_opposite_connected':float(sb-b*b),
        'LC_estimator':'fused-projector one-hop local currents; E[abs(Jloc)^2]',
        'LC_observable_chunk':int(chunk),
    })
    needed=('S_CO','S_CO_over_Ncell','S_LC_same','S_LC_opposite')
    if not all(np.isfinite(out[k]) for k in needed):
        raise RuntimeError('Nonfinite structure factor in fused estimator')
    return out


def _run_stage(opt,g,nf,hi,Hstandard,graph,seed_data,shells,bfmat,
               initial,stage,irrep,out,rng_seed,n_iter):
    import jax
    import jax.numpy as jnp
    import netket as nk
    jax.config.update('jax_enable_x64',True)
    maps,chars=_make_group(g,stage,irrep)
    tr=(stage=='tr')
    already_done=0
    progress_file=out/'progress.json'
    progress_params=out/'progress_params.npz'
    if opt.resume and int(n_iter)>0 and progress_file.exists() and progress_params.exists():
        progress=json.loads(progress_file.read_text(encoding='utf-8'))
        signature={'stage':stage,'irrep':irrep,'L1':opt.L1,'L2':opt.L2,
                   'nf':nf,'V':opt.V,'t':opt.t,'t1':opt.t1,'t2':opt.t2}
        if any(progress.get(k)!=v for k,v in signature.items()):
            raise RuntimeError(f'Incomplete checkpoint configuration mismatch: {progress_file}')
        already_done=int(progress['completed_iterations'])
        if already_done>int(n_iter):
            raise RuntimeError('Saved iterations exceed requested target; increase --*-iter')
        initial=_read_progress_parameters(progress_params)
        print(f'[resume partial] {stage}/{irrep} start at iteration {already_done}/{n_iter}',flush=True)
    orthonormalize=(not opt.qr_free_spatial or tr)
    base_stage=build_supercell_netket_model(
        initial['blocks'],seed_data['sc_coords'],seed_data['internal_sites'],
        seed_data['kpoints'],shells,use_jastrow=True,backflow_matrices=bfmat,
        initial_v=initial.get('v'),initial_eta=initial.get('eta'),
        sparse_correlators=True,packed_orbitals=True,
        orthonormalize_orbitals=orthonormalize,
    )
    stage_model=build_supercell_netket_model(
        initial['blocks'],seed_data['sc_coords'],seed_data['internal_sites'],
        seed_data['kpoints'],shells,use_jastrow=True,backflow_matrices=bfmat,
        initial_v=initial.get('v'),initial_eta=initial.get('eta'),
        point_site_maps=maps,point_characters=chars,
        time_reversal_even=tr,
        initial_tr_phase=float(initial.get('tr_phase') or 0.0),
        sparse_correlators=True,packed_orbitals=True,
        orthonormalize_orbitals=orthonormalize,
    )
    clusters=np.asarray([(int(e.i),int(e.j)) for e in g.edges],np.int32)
    stage_sweep=(opt.sweep_size if stage=='tint' else opt.spatial_sweep_size)
    sampler_status='standard'
    sampler_report={}
    sampler=nk.sampler.MetropolisFermionHop(
        hi,graph=graph,n_chains=opt.n_chains,sweep_size=stage_sweep,
    )
    if opt.sampler=='projected-cache':
        try:
            candidate=make_projected_cached_sampler(
                nk,jax,hilbert=hi,seed_data=seed_data,
                shell_matrices=shells,backflow_matrices=bfmat,
                clusters=clusters,site_maps=maps,characters=chars,
                time_reversal_even=tr,n_chains=opt.n_chains,
                sweep_size=stage_sweep,chunk_size=None,
                orthonormalize_orbitals=orthonormalize,
            )
            # Validate sampler without trusting its MC samples.
            probe=nk.vqs.MCState(
                sampler,stage_model,n_samples=max(32,opt.n_chains),
                n_discard_per_chain=opt.n_discard,
                chunk_size=opt.chunk_size,seed=rng_seed,
                sampler_seed=rng_seed+104729,
            )
            sampler_report=_test_sampler_cache(candidate,probe,g,nf,
                                               nprop=opt.sampler_check_proposals)
            print(f'[{stage}/{irrep}] projected-cache sampler check: {sampler_report}',flush=True)
            if not sampler_report['passed']:
                raise RuntimeError('projected-cache probability-ratio validation failed')
            sampler=candidate
            sampler_status='projected-cache'
            del probe
        except Exception as exc:
            if opt.strict:
                raise
            sampler_report={'passed':False,'error':str(exc)}
            print(f'[{stage}/{irrep}] sampler fallback to standard: {exc}',flush=True)
            sampler=nk.sampler.MetropolisFermionHop(
                hi,graph=graph,n_chains=opt.n_chains,sweep_size=stage_sweep,
            )

    vstate=nk.vqs.MCState(
        sampler,stage_model,n_samples=opt.samples if stage=='tint' else opt.spatial_samples,
        n_discard_per_chain=opt.n_discard,chunk_size=opt.chunk_size,
        seed=rng_seed,sampler_seed=rng_seed+104729,
    )
    raw_kernel=make_fast_raw_local_energy_kernel(
        sc_coords=seed_data['sc_coords'],
        internal_sites=seed_data['internal_sites'],
        kpoints=seed_data['kpoints'],
        occupations_per_k=seed_data['occupations_per_k'],
        shell_matrices=shells,backflow_matrices=bfmat,
        edge_i=np.asarray([e.i for e in g.edges],np.int32),
        edge_j=np.asarray([e.j for e in g.edges],np.int32),
        edge_t=np.asarray([g.t[e.kind] for e in g.edges],float),
        edge_has_interaction=np.asarray([e.kind=='t' for e in g.edges],float),
        interaction_V=opt.V,return_diagnostics=False,
        orthonormalize_orbitals=orthonormalize,
    )
    fused_parameters=_fused_kwargs(opt,g,nf,seed_data,shells,bfmat,
                                   maps,chars,tr,orthonormalize)
    if opt.local_kernel=='fused':
        training_fused=make_fused_projected_kernel(**fused_parameters,
                                                 return_currents=False)
        projected_kernel=training_fused
    else:
        projected_kernel=make_projected_fast_local_energy_kernel(
            raw_model=base_stage,raw_local_energy_kernel=raw_kernel,
            site_maps=maps,characters=chars,nf=nf,time_reversal_even=tr,
            batch_strategy=opt.projected_batch,
        )
    local_check={}
    Hrun=Hstandard
    fast_status='standard'
    if opt.fast_local:
        try:
            if opt.check_woodbury:
                local_check=_test_local_fast(vstate,projected_kernel,g,opt.V,nf,
                                              nroots=opt.check_roots)
                local_check['performed']=True
                print(f'[{stage}/{irrep}] projected Woodbury check: {local_check}',flush=True)
                if not local_check['passed']:
                    raise RuntimeError('projected local energy != direct determinant recomputation')
            else:
                # Keep the optimized operator, skip only the expensive
                # direct-determinant recomputation used for validation.
                local_check={'performed':False,'status':'skipped'}
                print(f'[{stage}/{irrep}] projected Woodbury direct check skipped '
                      '(enable with --check-woodbury)',flush=True)
            Hcustom=old._fast_raw_operator_class(nk)
            Hrun=Hcustom(hi,projected_kernel)
            fast_status='projected-woodbury'
        except Exception as exc:
            if opt.strict:
                raise
            local_check={'passed':False,'error':str(exc)}
            print(f'[{stage}/{irrep}] local-energy fallback to standard: {exc}',flush=True)
            Hrun=Hstandard

    warm=vstate.expect(Hrun)
    eval_start=float(np.real(warm.mean))
    if opt.check_gradient and fast_status=='projected-woodbury':
        try:
            check=old._validate_fast_raw_gradient(
                vstate=vstate,H_reference=Hstandard,
                H_fast_custom=Hrun,jax=jax,
            )
            if check['relative_l2_error']>1e-6 or check['energy_abs_error']>1e-6:
                raise RuntimeError(f'Projected gradient mismatch: {check}')
            local_check['gradient']=check
        except Exception as exc:
            if opt.strict:
                raise
            print(f'[{stage}/{irrep}] gradient test failed; fallback: {exc}',flush=True)
            Hrun=Hstandard
            fast_status='standard'
            local_check['gradient_error']=str(exc)

    optimizer=nk.optimizer.Sgd(learning_rate=float(opt.tint_lr if stage=='tint' else opt.spatial_lr if stage=='spatial' else opt.tr_lr))
    # This is the same Shared-SR implementation as the Raw scan, now operating
    # on the projected wavefunction and projected fast-local-energy operator.
    solver=partial(jax.scipy.sparse.linalg.cg,tol=opt.cg_tol,atol=0.0,maxiter=opt.cg_maxiter)
    trainer='shared_sr'
    if opt.trainer=='shared_sr':
        try:
            driver=old._make_shared_sr_driver(
                nk,jax,H=Hrun,optimizer=optimizer,vstate=vstate,
                diag_shift=opt.diag_shift,solver=solver,
                chunk_size=opt.qgt_chunk,
            )
        except Exception as exc:
            if opt.strict:raise
            print(f'[{stage}/{irrep}] Shared-SR failed to initialise: {exc}; legacy SR.',flush=True)
            trainer='legacy'
    else:
        trainer='legacy'
    if trainer=='legacy':
        qgt=partial(nk.optimizer.qgt.QGTJacobianPyTree,
                    mode='complex',chunk_size=opt.qgt_chunk)
        sr=nk.optimizer.SR(qgt=qgt,solver=solver,diag_shift=opt.diag_shift)
        driver=nk.driver.VMC(Hrun,optimizer,variational_state=vstate,preconditioner=sr)
    callback=(old._make_orbital_gauge_fix_callback(
        nk,seed_data['occupations_per_k'],opt.gauge_fix_interval
    ) if not orthonormalize else None)
    log=nk.logging.RuntimeLog()
    start=time.perf_counter()
    segment_times=[]
    done=already_done
    while done<int(n_iter):
        nseg=min(int(opt.checkpoint_every),int(n_iter)-done)
        step_started=time.perf_counter()
        driver.run(n_iter=nseg,out=log,show_progress=True,callback=callback)
        segment_seconds=time.perf_counter()-step_started
        done+=nseg
        segment_times.append({'end_iteration':int(done),
                              'iterations':int(nseg),
                              'seconds':float(segment_seconds),
                              'seconds_per_iteration':float(segment_seconds/nseg)})
        checkpoint_params=extract_supercell_parameters(
            vstate.parameters,len(seed_data['kpoints']),
            occupations_per_k=seed_data['occupations_per_k'],
        )
        out.mkdir(parents=True,exist_ok=True)
        old._save_params(progress_params,checkpoint_params,seed_data)
        atomic_json(progress_file,{
            'stage':stage,'irrep':irrep,'L1':opt.L1,'L2':opt.L2,
            'nf':nf,'V':opt.V,'t':opt.t,'t1':opt.t1,'t2':opt.t2,
            'completed_iterations':int(done),'target_iterations':int(n_iter),
        })
        print(f'[checkpoint] {stage}/{irrep} iterations={done}/{n_iter}',flush=True)
    elapsed=time.perf_counter()-start
    # Save BEFORE slow final observable evaluation and maps.
    result_params=extract_supercell_parameters(
        vstate.parameters,len(seed_data['kpoints']),
        occupations_per_k=seed_data['occupations_per_k'],
    )
    out.mkdir(parents=True,exist_ok=True)
    old._save_params(out/'final_params.npz',result_params,seed_data)
    print(f'[{stage}/{irrep}] saved checkpoint {out/"final_params.npz"}',flush=True)
    eval_count=int(opt.eval_samples)
    if stage!='tr' and opt.eval_samples_intermediate is not None:
        eval_count=int(opt.eval_samples_intermediate)
    eval_started=time.perf_counter()
    print(f'[{stage}/{irrep}] final eval: samples={eval_count} '
          f'sampler={sampler_status} local={fast_status} '
          f'lc_estimator={opt.lc_estimator}',flush=True)
    vstate.n_samples=eval_count
    vstate.sample()
    # JAX dispatch is asynchronous; wait so sampling and energy timings
    # represent real elapsed time instead of just launch overhead.
    np.asarray(vstate.samples)
    sampling_elapsed=time.perf_counter()-eval_started
    print(f'[{stage}/{irrep}] eval sampling done: {sampling_elapsed:.2f}s',flush=True)
    energy_started=time.perf_counter()
    stat=vstate.expect(Hrun)
    np.asarray(stat.mean)
    energy_elapsed=time.perf_counter()-energy_started
    print(f'[{stage}/{irrep}] eval energy done: {energy_elapsed:.2f}s',flush=True)
    summary={

        'stage':stage,'seed':opt.seed,'irrep':irrep,'Ns':g.Ns,'Nf':nf,
        'L1':opt.L1,'L2':opt.L2,'V':opt.V,'t':opt.t,'t1':opt.t1,'t2':opt.t2,
        'n_images':len(maps),'time_reversal_restored':tr,
        'mc_E':float(np.real(stat.mean)),
        'mc_E_error':float(stats_error(stat)),
        'initial_mc_E':eval_start,
        'iterations':int(n_iter),
        'resumed_iterations':int(already_done),
        'time_training_s':elapsed,
        'time_per_iteration_s':elapsed/max(int(n_iter)-int(already_done),1),
        'training_segments':segment_times,
        'steady_time_per_iteration_s':(
            float(np.mean([x['seconds_per_iteration'] for x in segment_times[1:]]))
            if len(segment_times)>1 else None
        ),
        'samples_per_iter':int(opt.samples if stage=='tint' else opt.spatial_samples),
        'sweep_size':int(stage_sweep),
        'eval_samples':int(eval_count),
        'time_eval_sampling_s':float(sampling_elapsed),
        'time_eval_energy_s':float(energy_elapsed),
        'fast_local_status':fast_status,'fast_local_check':local_check,
        'woodbury_check_requested':bool(opt.check_woodbury),
        'local_kernel_family':opt.local_kernel,
        'lc_estimator_requested':opt.lc_estimator,
        'sampler_status':sampler_status,'sampler_check':sampler_report,
        'trainer':trainer,'qr_free_orbitals':not orthonormalize,
        'raw_checkpoint':str(_find_raw_stage(opt)),
    }
    observable_failure=None
    obs_started=time.perf_counter()
    if opt.observables:
        print(f'[{stage}/{irrep}] observable evaluation starts '
              f'(LC one-hop={opt.lc_estimator=="onehop"}, '
              f'direct-check={opt.check_currents and opt.lc_estimator=="onehop"})',
              flush=True)
        try:
            if opt.lc_estimator=='onehop':
                observable_kernel=make_fused_projected_kernel(
                    **fused_parameters,return_currents=True)
                if opt.check_currents:
                    candidates=np.asarray(vstate.samples,np.float64).reshape((-1,g.Ns))[:max(8,opt.check_roots*8)]
                    logs=np.asarray(stage_model.apply({'params':vstate.parameters},candidates))
                    okay=np.isfinite(np.real(logs)) & (np.real(logs)>np.max(np.real(logs))-45.)
                    test_roots=candidates[np.flatnonzero(okay)[:opt.check_roots]]
                    if len(test_roots)<opt.check_roots:
                        raise RuntimeError('Not enough non-nodal roots to validate LC estimator')
                    import jax.numpy as jnp
                    _,j_fast1,j_fast2=observable_kernel(vstate.parameters,jnp.asarray(test_roots))
                    weights1,weights2=old._current_channel_edge_weights(g)
                    current_direct_started=time.perf_counter()
                    j_direct=_direct_current_loc(stage_model,vstate.parameters,
                                                 test_roots,g,weights1,weights2,
                                                 batch_size=opt.current_check_batch)
                    current_direct_seconds=time.perf_counter()-current_direct_started
                    summary['time_current_direct_check_s']=float(current_direct_seconds)
                    print(f'[{stage}/{irrep}] direct LC check: '
                          f'{current_direct_seconds:.2f}s in batches of {opt.current_check_batch}',
                          flush=True)
                    currents_err=float(np.max(np.abs(np.column_stack(
                        (np.asarray(j_fast1),np.asarray(j_fast2)))-j_direct)))
                    summary['lc_direct_check_max_abs_error']=currents_err
                    if currents_err>opt.current_tolerance:
                        raise RuntimeError('Projected one-hop LC orientation/sign mismatch: '
                                           f'{currents_err:.3e} > {opt.current_tolerance:.3e}')
                obs=_measure_fused_observables(vstate,g,hi,nf,observable_kernel,
                                               opt.observable_chunk)
            else:
                ops=old._current_operators(hi,g)
                obs=old._mc_order_observables(vstate,g,nf,hi,ops)
            summary.update(obs)
            for key in ('S_CO','S_CO_over_Ncell','S_LC_same','S_LC_opposite'):
                if key not in summary or not np.isfinite(summary[key]):
                    raise RuntimeError(f'Missing or invalid observable: {key}')
        except Exception as exc:
            observable_failure=exc
            summary['observables_error']=str(exc)
            print(f'[{stage}/{irrep}] observables failure: {exc}',flush=True)
    obs_elapsed=time.perf_counter()-obs_started
    eval_elapsed=time.perf_counter()-eval_started
    summary['time_eval_observables_s']=float(obs_elapsed)
    summary['time_eval_total_s']=float(eval_elapsed)
    print(f'[{stage}/{irrep}] eval profile: '
          f'sampling={sampling_elapsed:.2f}s '
          f'energy={energy_elapsed:.2f}s '
          f'obs_including_checks={obs_elapsed:.2f}s '
          f'total={eval_elapsed:.2f}s',flush=True)
    _save_params_and_summary(out,result_params,seed_data,summary,old._history(log))
    if progress_file.exists(): progress_file.unlink()
    if progress_params.exists(): progress_params.unlink()
    if observable_failure is not None and not opt.allow_missing_observables:
        raise RuntimeError('VAP parameters were saved, but a required structure factor failed; '
                           'see observables_error in summary.json') from observable_failure
    return summary,result_params



def _screen_irreps(opt,g,nf,hi,Hstandard,graph,seed_data,shells,bfmat,params_screen,output,
                   source='tint'):
    """PAV screening from the *fixed* Raw or Tint parameters, without SR updates.

    Initial PAV ordering is only a heuristic for final VAP ordering.  The
    screen_selection='best' option is intentionally an explicit opt-in.
    """
    import hashlib
    import jax
    jax.config.update('jax_enable_x64',True)
    import netket as nk
    if source not in ('raw','tint'):
        raise ValueError(f'Invalid PAV screen source {source}')
    path=output/('irrep_pav_screen_raw.json' if source=='raw'
                 else 'irrep_pav_screen.json')

    # Avoid silently reusing a screen computed with different Raw/Tint
    # parameters, samples, geometry, or selection rules.
    digest=hashlib.sha256()
    for block in params_screen['blocks']:
        data=np.ascontiguousarray(np.asarray(block,dtype=np.complex128))
        digest.update(str(data.shape).encode('ascii'))
        digest.update(data.tobytes())
    for name,dtype in (('v',np.float64),('eta',np.complex128)):
        data=np.ascontiguousarray(np.asarray(params_screen[name],dtype=dtype))
        digest.update(name.encode('ascii'))
        digest.update(data.tobytes())
    signature={
        'source':source,'seed':opt.seed,
        'V':float(opt.V),'L1':int(opt.L1),'L2':int(opt.L2),'nf':int(nf),
        'irreps_requested':list(opt.irreps),
        'screen_samples':int(opt.screen_samples),
        'screen_keep_sigma':float(opt.screen_keep_sigma),
        'screen_selection':str(opt.screen_selection),
        'params_sha256':digest.hexdigest(),
    }
    if opt.resume and path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        if all(saved.get(k)==v for k,v in signature.items()):
            print(f'[resume] loaded {source} PAV screening {path}: '
                  f'{saved["selected"]}',flush=True)
            return saved['selected']
    print(f'[PAV screen] source={source} selection={opt.screen_selection} '
          f'samples={opt.screen_samples} irreps={opt.irreps}',flush=True)
    entries=[]
    for ix,irrep in enumerate(opt.irreps):
        maps,chars=_make_group(g,'spatial',irrep)
        model=build_supercell_netket_model(
            params_screen['blocks'],seed_data['sc_coords'],
            seed_data['internal_sites'],seed_data['kpoints'],shells,
            use_jastrow=True,backflow_matrices=bfmat,
            initial_v=params_screen['v'],initial_eta=params_screen['eta'],
            point_site_maps=maps,point_characters=chars,
            sparse_correlators=True,packed_orbitals=True,
            orthonormalize_orbitals=True,
        )
        sampler=nk.sampler.MetropolisFermionHop(
            hi,graph=graph,n_chains=opt.n_chains,sweep_size=opt.spatial_sweep_size)
        vstate=nk.vqs.MCState(
            sampler,model,n_samples=opt.screen_samples,
            n_discard_per_chain=opt.n_discard,
            chunk_size=opt.chunk_size,seed=opt.random_seed+580+ix,
            sampler_seed=opt.random_seed+104729+580+ix,
        )
        stat=vstate.expect(Hstandard)
        energy=float(np.real(stat.mean))
        error=float(stats_error(stat))
        def optional_float(value):
            try:
                result=float(np.asarray(value).mean())
                return result if np.isfinite(result) else None
            except (ValueError,TypeError,AttributeError):
                return None
        try:
            acceptance=optional_float(vstate.sampler_state.acceptance)
        except (ValueError,TypeError,AttributeError):
            acceptance=None
        entry={'irrep':irrep,'E':energy,'error':error,
               'R_hat':optional_float(getattr(stat,'R_hat',None)),
               'tau_corr':optional_float(getattr(stat,'tau_corr',None)),
               'sampler_acceptance':acceptance}
        if not np.isfinite(energy) or not np.isfinite(error):
            raise RuntimeError(
                f'PAV {source}/{irrep} returned a non-finite energy or error. '
                'The projected sector may have near-zero Raw overlap or the '
                'sampler may be mixing poorly; do not select a single irrep '
                'from this screening result.'
            )
        entries.append(entry)
        print('[PAV screen]',entry,flush=True)
    best=min(entries,key=lambda row:row['E'])
    for e in entries:
        sigma=float(np.hypot(e['error'],best['error']))
        e['gap_to_best']=float(e['E']-best['E'])
        e['gap_in_combined_sigma']=(
            float((e['E']-best['E'])/sigma) if sigma>0 else None
        )
    if opt.screen_selection=='best':
        selected=[best['irrep']]
        runners=sorted(entries,key=lambda item:item['E'])
        if len(runners)>1 and (
            runners[1]['gap_in_combined_sigma'] is None
            or runners[1]['gap_in_combined_sigma']<3.
        ):
            print('[PAV screen] WARNING: best-only selection has a '
                  'statistically close competing irrep. Initial PAV rankings '
                  'can reverse after VAP or time-reversal restoration; '
                  'consider --screen-selection confidence.',flush=True)
    else:
        selected=[
            e['irrep'] for e in entries
            if e['irrep']==best['irrep']
            or e['gap_to_best']<=opt.screen_keep_sigma*np.hypot(e['error'],best['error'])
        ]
    atomic_json(path,{**signature,'entries':entries,'selected':selected,
                      'best_initial_irrep':best['irrep'],
                      'best_initial_E':float(best['E']),
                      'warning':('Initial PAV is not a proof of the optimal VAP irrep; '
                                 'near-degenerate or weak-overlap sectors can reorder '
                                 'after variational optimization.')})
    print(f'[PAV screen] source={source} initial best={best["irrep"]} '
          f'E={best["E"]:.6f} +/- {best["error"]:.3g}; '
          f'selected VAP sector(s)={selected}',flush=True)
    return selected


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--L1',type=int,required=True);p.add_argument('--L2',type=int,required=True)
    p.add_argument('--nf',type=int,default=None)
    p.add_argument('--t',type=float,default=1.0);p.add_argument('--t1',type=float,default=0.5)
    p.add_argument('--t2',type=float,default=-0.5);p.add_argument('--V',type=float,required=True)
    p.add_argument('--seed',required=True)
    p.add_argument('--seed-amplitude',type=float,default=0.08,
                   help='Use the same auxiliary seed amplitude as the Raw scan when reconstructing k-block occupation counts.')
    p.add_argument('--raw-stage',type=str,default=None)
    p.add_argument('--scan-root',type=str,default=None)
    p.add_argument('--mode',choices=['continuation','independent'],default='continuation')
    p.add_argument('--direction',choices=['up','down'],default='up')
    p.add_argument('--output',type=str,required=True)
    p.add_argument('--irreps',nargs='+',choices=['A1','A2','E'],
                   default=['A1','A2','E'],
                   help='All ordinary C3v irreps at internal Gamma. Does NOT scan finite-momentum sectors.')
    p.add_argument('--screen-irreps',action=argparse.BooleanOptionalAction,default=False,
                   help='Optional PAV pre-screen of selected irreps; may miss sectors that improve under VAP. Disable for final phase diagrams.')
    p.add_argument('--screen-samples',type=int,default=512)
    p.add_argument('--screen-source',choices=['raw','tint'],default='tint',
                   help='Use fixed Raw checkpoint or Tint-optimized parameters for PAV irrep screening')
    p.add_argument('--screen-selection',choices=['confidence','best'],default='confidence',
                   help='confidence retains statistical ties; best keeps only lowest initial PAV energy')
    p.add_argument('--screen-only',action='store_true',
                   help='Evaluate Raw PAV irrep energies, write irrep_pav_screen_raw.json and exit without VAP')
    p.add_argument('--screen-keep-sigma',type=float,default=2.0,
                   help='Retain sectors statistically tied to the best PAV energy')
    p.add_argument('--restore-tr',action='store_true')
    p.add_argument('--stages',nargs='+',choices=['tint','spatial','tr'],default=['tint','spatial','tr'])
    p.add_argument('--tint-iter',type=int,default=50)
    p.add_argument('--spatial-iter',type=int,default=30)
    p.add_argument('--tr-iter',type=int,default=30)
    p.add_argument('--tint-lr',type=float,default=0.01)
    p.add_argument('--spatial-lr',type=float,default=0.01)
    p.add_argument('--tr-lr',type=float,default=0.01)
    p.add_argument('--samples',type=int,default=2048)
    p.add_argument('--spatial-samples',type=int,default=512)
    p.add_argument('--eval-samples',type=int,default=8192)
    p.add_argument('--eval-samples-intermediate',type=int,default=None,
                   help='Optional smaller final-eval sample count for Tint/Spatial; TR still uses --eval-samples. All stages use --eval-samples by default.')
    p.add_argument('--n-chains',type=int,default=32)
    p.add_argument('--n-discard',type=int,default=8)
    p.add_argument('--sweep-size',type=int,default=None,help='Tint sweep; default Nf')
    p.add_argument('--spatial-sweep-size',type=int,default=None,help='Spatial/TR sweep; default Nf//4')
    p.add_argument('--chunk-size',type=int,default=32,
                   help='Keep small for projected Woodbury intermediates')
    p.add_argument('--qgt-chunk',type=int,default=64)
    p.add_argument('--cg-tol',type=float,default=1e-5)
    p.add_argument('--cg-maxiter',type=int,default=100)
    p.add_argument('--diag-shift',type=float,default=0.05)
    p.add_argument('--trainer',choices=['shared_sr','legacy'],default='shared_sr')
    p.add_argument('--sampler',choices=['standard','projected-cache'],default='projected-cache')
    p.add_argument('--sampler-check-proposals',type=int,default=3)
    p.add_argument('--projected-batch',choices=['scan','flatten'],default='scan')
    p.add_argument('--local-kernel',choices=['fused','legacy'],default='fused',
                   help='Fused kernel assembles Q/W/C once; legacy retains old per-image Raw kernel')
    p.add_argument('--fast-local',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--check-woodbury',action=argparse.BooleanOptionalAction,
                   default=False,
                   help='Verify projected local energy against full determinant recomputation; '
                        'off by default for production scans, fast Woodbury kernel remains enabled.')
    p.add_argument('--check-roots',type=int,default=2)
    p.add_argument('--check-gradient',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--strict',action='store_true',help='Raise instead of fallback if fast path check fails')
    p.add_argument('--checkpoint-every',type=int,default=10,
                   help='Save per-stage progress after this many VAP iterations')
    p.add_argument('--qr-free-spatial',action='store_true')
    p.add_argument('--gauge-fix-interval',type=int,default=20)
    p.add_argument('--observables',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--lc-estimator',choices=['onehop','netket'],default='onehop')
    p.add_argument('--observable-chunk',type=int,default=16)
    p.add_argument('--check-currents',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--current-check-batch',type=int,default=8,
                   help='Batch full projected amplitudes for one-hop current validation')
    p.add_argument('--current-tolerance',type=float,default=3e-7)
    p.add_argument('--allow-missing-observables',action='store_true',
                   help='Allow missing S in summary; default raises but preserves checkpoint')
    p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--random-seed',type=int,default=2026)
    p.add_argument('--dry-run',action='store_true',help='Validate checkpoint geometry and print plan without starting NetKet')
    p.add_argument('--jastrow-shells',type=int,default=4)
    return p


def main():
    opt=parser().parse_args()
    if opt.checkpoint_every<1:
        raise ValueError('--checkpoint-every must be >=1')
    if opt.screen_only and (not opt.screen_irreps or opt.screen_source!='raw'):
        raise ValueError('--screen-only requires --screen-irreps --screen-source raw')
    if opt.observable_chunk<1 or opt.check_roots<1 or opt.current_check_batch<1:
        raise ValueError('--observable-chunk, --check-roots and --current-check-batch must be positive')
    if 'tr' in opt.stages and not opt.restore_tr:
        opt.stages=[s for s in opt.stages if s!='tr']
    if opt.restore_tr and 'tr' in opt.stages and 'spatial' not in opt.stages:
        # Allowed only if spatial checkpoint already exists, checked below.
        pass
    if opt.sweep_size is None:
        opt.sweep_size=6*opt.L1*opt.L2
    if opt.spatial_sweep_size is None:
        opt.spatial_sweep_size=max(2,(6*opt.L1*opt.L2)//4)
    if opt.eval_samples<opt.n_chains or opt.spatial_samples<opt.n_chains:
        raise ValueError('Samples must exceed number of chains')
    if opt.eval_samples_intermediate is not None and opt.eval_samples_intermediate<opt.n_chains:
        raise ValueError('--eval-samples-intermediate must be >= --n-chains')
    raw_dir=_find_raw_stage(opt)
    g=native_geometry.make_ring3_supercell(opt.L1,opt.L2,
                                            t=opt.t,t1=opt.t1,t2=opt.t2)
    nf=opt.nf or 6*opt.L1*opt.L2
    seed_data=seed_bloch_blocks(g,nf,opt.L1,opt.L2,opt.seed,opt.seed_amplitude,
                                random_seed=opt.random_seed)
    params=_check_checkpoint_compatibility(raw_dir,seed_data,g,nf,opt)
    if np.linalg.norm(np.asarray(seed_data['total_supercell_momentum'])
                      -np.rint(seed_data['total_supercell_momentum']))>1e-9:
        raise ValueError('Selected k occupation has non-Gamma total supercell momentum; this Γ-sector projector cannot be used.')
    print('[symmetry] Screening Gamma internal-translation sector in C3v irreps '+','.join(opt.irreps)+'.',
          'Finite-momentum sectors are NOT included in this run.',flush=True)
    if opt.dry_run:
        print(json.dumps({
            'raw_checkpoint':str(raw_dir/'final_params.npz'),
            'output':str(Path(opt.output).expanduser().resolve()),
            'Ns':int(g.Ns),'Nf':int(nf),
            'occupations_per_k':np.asarray(seed_data['occupations_per_k'],int).tolist(),
            'V':float(opt.V),
            'tint_iter':int(opt.tint_iter),'spatial_iter':int(opt.spatial_iter),
            'tr_iter':int(opt.tr_iter) if opt.restore_tr else 0,
            'irreps':list(opt.irreps),
            'irrep_screening':bool(opt.screen_irreps),
        },indent=2,ensure_ascii=False))
        return
    import jax
    jax.config.update('jax_enable_x64',True)
    hi,_,Hstandard,graph=make_netket_hamiltonian(g,nf,opt.V)
    _,_,shells,_=jastrow_channels(g,mode='bond_aware',n_shells=opt.jastrow_shells)
    _,bfmat,_=bond_backflow_channels(g)
    if len(params['v'])!=len(shells):
        raise ValueError(f'Jastrow channel mismatch {len(params["v"])} vs {len(shells)}')
    if len(params['eta'])!=len(bfmat):
        raise ValueError(f'Backflow channel mismatch {len(params["eta"])} vs {len(bfmat)}')
    output=Path(opt.output).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    meta={'raw_dir':str(raw_dir),'args':vars(opt),
          'geometry':g.metadata(),
          'note':'TR-restored structure factors, not one-point phase order, classify finite-size correlations.'}
    atomic_json(output/'run_config.json',meta)
    results=[]
    raw_screened=None
    if opt.screen_irreps and opt.screen_source=='raw':
        raw_screened=_screen_irreps(
            opt,g,nf,hi,Hstandard,graph,seed_data,shells,bfmat,
            params,output,source='raw',
        )
        if opt.screen_only:
            print('[PAV screen] raw-only screen finished; no VAP training was run.',
                  flush=True)
            return
    def load_or_run(stage, irrep, parent, niter, out, offset):
        saved_params=out/'final_params.npz'
        saved_summary=out/'summary.json'
        if opt.resume and saved_params.exists():
            recovered=old._load_params_file(out)
            if saved_summary.exists():
                summary=json.loads(saved_summary.read_text(encoding='utf-8'))
                needed=('S_CO','S_CO_over_Ncell','S_LC_same','S_LC_opposite')
                if not opt.observables or (
                    'observables_error' not in summary
                    and all(k in summary and np.isfinite(summary[k]) for k in needed)
                ):
                    print(f'[resume] {stage}/{irrep} from {out}',flush=True)
                    return summary,recovered
            print(f'[resume] {stage}/{irrep} missing observables; evaluating saved parameters without retraining',flush=True)
            return _run_stage(opt,g,nf,hi,Hstandard,graph,seed_data,shells,bfmat,
                              recovered,stage,irrep,out,opt.random_seed+offset,0)
        return _run_stage(opt,g,nf,hi,Hstandard,graph,seed_data,shells,bfmat,
                          parent,stage,irrep,out,opt.random_seed+offset,niter)

    tint_dir=output/'fast_tint__Gamma'
    if 'tint' in opt.stages:
        row,params_tint=load_or_run('tint','Gamma',params,opt.tint_iter,tint_dir,11)
        results.append(row)
    else:
        if not (tint_dir/'final_params.npz').is_file():
            raise FileNotFoundError('Need finished Tint stage to skip --stages tint')
        params_tint=old._load_params_file(tint_dir)
    selected_irreps=(
        raw_screened if raw_screened is not None
        else _screen_irreps(opt,g,nf,hi,Hstandard,graph,seed_data,shells,bfmat,
                            params_tint,output,source='tint')
        if opt.screen_irreps and len(opt.irreps)>1
        else list(opt.irreps)
    )
    for i,ir in enumerate(selected_irreps):
        spatial_dir=output/f'fast_spatial__{ir}'
        if 'spatial' in opt.stages:
            row,par_spatial=load_or_run('spatial',ir,params_tint,opt.spatial_iter,spatial_dir,100+17*i)
            results.append(row)
        else:
            if not (spatial_dir/'final_params.npz').is_file():
                raise FileNotFoundError(f'Need completed Spatial stage for irrep {ir}: {spatial_dir}')
            par_spatial=old._load_params_file(spatial_dir)
        if opt.restore_tr and 'tr' in opt.stages:
            tr_dir=output/f'fast_tr__{ir}'
            row,_=load_or_run('tr',ir,par_spatial,opt.tr_iter,tr_dir,300+17*i)
            results.append(row)
    if results:
        keys=list(dict.fromkeys(k for item in results for k in item))
        with (output/'fast_vap_summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
            writer.writeheader();writer.writerows(results)
    print('Saved:',output,flush=True)

if __name__=='__main__':
    main()