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
    make_fast_raw_local_energy_kernel,
    make_fast_raw_sampler_sweep_kernel,
    make_fast_raw_sampler_cache_kernels,
    extract_supercell_parameters,
)
from symmetry_projection import (
    internal_translation_site_maps,
    internal_c3v_space_group_site_maps,
    internal_gamma_c3v_characters,
)


_FAST_RAW_OPERATOR_CLASS=None
_FAST_LOCAL_GRADIENT_VALIDATED=False
_ANALYTIC_LOGDERIV_VALIDATED_KEYS=set()
_SHARED_SR_VALIDATED_KEYS=set()


def _make_fast_raw_sampler(
    nk,
    jax,
    *,
    hilbert,
    clusters,
    seed_data,
    shell_mats,
    bf_mats,
    n_chains,
    sweep_size,
    chunk_size,
    orthonormalize_orbitals,
):
    """Persistent-cache Raw sampler using rank-3 Woodbury hop updates.

    The cache is rebuilt by sampler.reset() after every parameter update and
    then persists through both discarded and retained Markov-chain samples.
    The ordinary MetropolisSamplerState is reused; its rule_state field stores
    the Slater/backflow/Jastrow cache.
    """
    import jax.numpy as jnp

    init_cache,advance,_=make_fast_raw_sampler_cache_kernels(
        sc_coords=seed_data["sc_coords"],
        internal_sites=seed_data["internal_sites"],
        kpoints=seed_data["kpoints"],
        occupations_per_k=seed_data["occupations_per_k"],
        shell_matrices=shell_mats,
        backflow_matrices=bf_mats,
        clusters=np.asarray(clusters,np.int32),
        machine_pow=2.0,
        orthonormalize_orbitals=orthonormalize_orbitals,
    )

    class FastRawMetropolisSampler(nk.sampler.MetropolisSampler):
        _fast_init_cache: object = nk.utils.struct.field(
            pytree_node=False,serialize=False,default=None
        )
        _fast_advance: object = nk.utils.struct.field(
            pytree_node=False,serialize=False,default=None
        )

        def __init__(self):
            rule=nk.sampler.rules.FermionHopRule(
                hilbert,
                clusters=[
                    (int(i),int(j))
                    for i,j in np.asarray(clusters,np.int32)
                ],
            )
            super().__init__(
                hilbert,
                rule,
                n_chains=int(n_chains),
                sweep_size=int(sweep_size),
                chunk_size=(
                    None if chunk_size is None else int(chunk_size)
                ),
                machine_pow=2,
            )
            self._fast_init_cache=init_cache
            self._fast_advance=advance

        def _reset(self,machine,parameters,state):
            # Let NetKet recompute the exact current log probability once and
            # reset the standard acceptance counters/RNG bookkeeping.
            state=super()._reset(machine,parameters,state)
            model_pars=(
                parameters["params"]
                if "params" in parameters else parameters
            )
            cache=self._fast_init_cache(model_pars,state.σ)
            return state.replace(rule_state=cache)

        def _sample_next(self,machine,parameters,state):
            del machine,parameters
            new_rng,key_prop,key_acc=jax.random.split(state.rng,3)
            dtype=state.log_prob.dtype
            shape=(self.n_batches,self.sweep_size)
            proposal_uniforms=jax.random.uniform(
                key_prop,shape=shape,dtype=dtype
            )
            accept_uniforms=jax.random.uniform(
                key_acc,shape=shape,dtype=dtype
            )

            cache,logp_delta,naccepted,_=self._fast_advance(
                state.rule_state,
                proposal_uniforms,
                accept_uniforms,
            )
            new_state=state.replace(
                σ=cache["x"],
                log_prob=state.log_prob+logp_delta,
                rng=new_rng,
                rule_state=cache,
                n_accepted_proc=state.n_accepted_proc+naccepted,
                n_steps_proc=(
                    state.n_steps_proc
                    +jnp.asarray(
                        self.n_batches*self.sweep_size,
                        dtype=state.n_steps_proc.dtype,
                    )
                ),
            )
            return new_state,(new_state.σ,new_state.log_prob)

    return FastRawMetropolisSampler()


def _fast_raw_operator_class(nk):
    """Register and return the NetKet custom operator used by fast Raw VMC.

    Registration is lazy so importing this module does not require NetKet.
    The lean local-kernel interface lets NetKet keep its standard expect(),
    expect_and_grad(), statistics and VMC driver while replacing only the
    expensive local-energy evaluation.
    """
    global _FAST_RAW_OPERATOR_CLASS
    if _FAST_RAW_OPERATOR_CLASS is not None:
        return _FAST_RAW_OPERATOR_CLASS

    class FastRawHamiltonian(nk.operator.AbstractOperator):
        def __init__(self, hilbert, kernel):
            super().__init__(hilbert)
            self.kernel=kernel

        @property
        def dtype(self):
            return np.dtype(np.complex128)

        @property
        def is_hermitian(self):
            return True

    def _local_kernel_impl(op):
        def local_kernel(
            logpsi, pars, sigma, extra_args, *, chunk_size=None
        ):
            del logpsi, extra_args
            # NetKet's lean interface passes vstate.variables.  The
            # specialised kernel consumes only the trainable parameter
            # collection.
            model_pars=pars["params"] if "params" in pars else pars
            if chunk_size is None:
                return op.kernel(model_pars,sigma)

            # Mirror NetKet's own chunked operator kernels: split only the
            # sample axis while broadcasting the parameter PyTree.  This
            # keeps the Woodbury intermediates (Nf x Nedge x 3 per root)
            # bounded for large systems and avoids materialising all roots
            # in one XLA program.
            return nk.jax.apply_chunked(
                lambda p,x: op.kernel(p,x),
                in_axes=(None,0),
                chunk_size=int(chunk_size),
            )(model_pars,sigma)
        return local_kernel

    # NetKet 3.22+ dispatches get_local_kernel(vstate, op, chunk_size) when
    # MCState.chunk_size is set. Register both the chunked and unchunked
    # signatures explicitly. Runtime class objects avoid postponed-annotation
    # forward-reference issues with Plum.
    def _get_fast_raw_kernel_chunked(vstate,op,chunk_size):
        del vstate, chunk_size
        return _local_kernel_impl(op)

    _get_fast_raw_kernel_chunked.__annotations__={
        "vstate":nk.vqs.MCState,
        "op":FastRawHamiltonian,
        "chunk_size":int,
    }
    nk.vqs.get_local_kernel.dispatch(_get_fast_raw_kernel_chunked)

    def _get_fast_raw_kernel_unchunked(vstate,op,chunk_size):
        del vstate, chunk_size
        return _local_kernel_impl(op)

    _get_fast_raw_kernel_unchunked.__annotations__={
        "vstate":nk.vqs.MCState,
        "op":FastRawHamiltonian,
        "chunk_size":type(None),
    }
    nk.vqs.get_local_kernel.dispatch(_get_fast_raw_kernel_unchunked)

    def _get_fast_raw_kernel_arguments(vstate,op):
        del op
        return vstate.samples, None

    _get_fast_raw_kernel_arguments.__annotations__={
        "vstate":nk.vqs.MCState,
        "op":FastRawHamiltonian,
    }
    nk.vqs.get_local_kernel_arguments.dispatch(
        _get_fast_raw_kernel_arguments
    )

    _FAST_RAW_OPERATOR_CLASS=FastRawHamiltonian
    return FastRawHamiltonian


def _compare_parameter_trees(a,b):
    import jax
    la=jax.tree_util.tree_leaves(a)
    lb=jax.tree_util.tree_leaves(b)
    if len(la)!=len(lb):
        raise ValueError("Gradient PyTrees have different leaf counts")
    max_abs=0.0
    sq_diff=0.0
    sq_ref=0.0
    for xa,xb in zip(la,lb):
        aa=np.asarray(xa)
        bb=np.asarray(xb)
        if aa.shape!=bb.shape:
            raise ValueError(
                f"Gradient leaf shape mismatch: {aa.shape} != {bb.shape}"
            )
        d=aa-bb
        if d.size:
            max_abs=max(max_abs,float(np.max(np.abs(d))))
            sq_diff+=float(np.sum(np.abs(d)**2))
            sq_ref+=float(np.sum(np.abs(aa)**2))
    rel_l2=float(np.sqrt(sq_diff)/max(1e-15,np.sqrt(sq_ref)))
    return {"max_abs_error":float(max_abs),"relative_l2_error":rel_l2}


def _validate_fast_raw_gradient(
    *,
    vstate,H_reference,H_fast_custom,jax,
):
    """Compare custom/local-energy and standard NetKet energy gradients."""
    vstate.sample()
    ref_stats,ref_grad=vstate.expect_and_grad(H_reference)
    _sync_tree(jax,ref_grad)
    fast_stats,fast_grad=vstate.expect_and_grad(H_fast_custom)
    _sync_tree(jax,fast_grad)
    report=_compare_parameter_trees(ref_grad,fast_grad)
    ref_E=complex(np.asarray(ref_stats.mean))
    fast_E=complex(np.asarray(fast_stats.mean))
    report.update({
        "energy_abs_error":float(abs(ref_E-fast_E)),
        "reference_energy_real":float(ref_E.real),
        "fast_energy_real":float(fast_E.real),
    })
    print(
        "[fast local gradient check] "
        f"dE={report['energy_abs_error']:.3e} "
        f"grad_max={report['max_abs_error']:.3e} "
        f"grad_rel_l2={report['relative_l2_error']:.3e}",
        flush=True,
    )
    return report


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


def _load_params_file(directory):
    """Load final_params.npz even if a later map/summary step was interrupted."""
    directory=Path(directory)
    state=directory/"final_params.npz"
    if not state.exists():
        raise FileNotFoundError(f"Missing saved parameters: {state}")
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
    return params


def _load_params(directory):
    directory=Path(directory)
    summary=directory/"summary.json"
    if not summary.exists():
        raise FileNotFoundError(
            f"Missing saved large-supercell summary: {summary}"
        )
    params=_load_params_file(directory)
    return json.loads(summary.read_text(encoding="utf-8")),params



def _charge_structure_from_samples(g, samples, nf):
    """MC estimate matching the existing small-cluster CO_qmax definition.

    Sites are ordered as primitive cells x six Ruby orbitals. For every
    allowed primitive-lattice momentum q, form the 6x6 density structure
    matrix and return its largest eigenvalue. Both the raw delta-n definition
    (subtracting only the global filling) and the connected covariance are
    reported.
    """
    x=np.asarray(samples,float).reshape((-1,g.Ns))
    if x.shape[1]!=g.Ns:
        raise ValueError("Unexpected VMC sample shape")
    nc=int(g.num_cells)
    if g.Ns!=6*nc:
        raise RuntimeError("Ruby geometry must contain six sites per primitive cell")
    dn=x.reshape((-1,nc,6))-float(nf/g.Ns)
    reps=np.asarray(g.reps,float)
    qs=np.asarray(g.permitted_momenta(),float)
    phase=np.exp(
        -2j*np.pi*np.einsum("qd,cd->qc",qs,reps)
    )/np.sqrt(float(nc))
    z=np.einsum("qc,sca->sqa",phase,dn,optimize=True)

    raw=np.einsum(
        "sqa,sqb->qab",z.conj(),z,optimize=True
    )/float(z.shape[0])
    raw=0.5*(raw+np.swapaxes(raw.conj(),1,2))
    raw_e=np.linalg.eigvalsh(raw).real
    raw_max=raw_e[:,-1]

    zc=z-np.mean(z,axis=0,keepdims=True)
    conn=np.einsum(
        "sqa,sqb->qab",zc.conj(),zc,optimize=True
    )/float(z.shape[0])
    conn=0.5*(conn+np.swapaxes(conn.conj(),1,2))
    conn_e=np.linalg.eigvalsh(conn).real
    conn_max=conn_e[:,-1]

    iraw=int(np.argmax(raw_max))
    iconn=int(np.argmax(conn_max))
    return {
        "CO_qmax":float(raw_max[iraw]),
        "CO_q1":float(qs[iraw,0]),
        "CO_q2":float(qs[iraw,1]),
        "CO_qmax_connected":float(conn_max[iconn]),
        "CO_connected_q1":float(qs[iconn,0]),
        "CO_connected_q2":float(qs[iconn,1]),
        # Explicit S_CO aliases for large-supercell phase-diagram tables.
        "S_CO":float(raw_max[iraw]),
        "S_CO_connected":float(conn_max[iconn]),
        # Conventional intensive finite-size quantity, useful later for scaling.
        "S_CO_over_Ncell":float(raw_max[iraw]/nc),
        "S_CO_connected_over_Ncell":float(conn_max[iconn]/nc),
        "CO_n_momenta":int(len(qs)),
        "CO_n_samples":int(z.shape[0]),
    }


def _as_fixed_number_fermion_operator(op):
    """Convert a NetKet fermion operator to the fastest fixed-N backend available."""
    try:
        from netket.experimental.operator import (
            ParticleNumberAndSpinConservingFermioperator2nd as FastFermion,
        )
        fast=FastFermion.from_fermionoperator2nd(op)
        try:
            fast=fast.collect()
        except Exception:
            pass
        return fast
    except Exception:
        pass
    try:
        from netket.experimental.operator import (
            ParticleNumberConservingFermioperator2nd as FastFermion,
        )
        fast=FastFermion.from_fermionoperator2nd(op)
        try:
            fast=fast.collect()
        except Exception:
            pass
        return fast
    except Exception:
        pass
    try:
        return op.collect()
    except Exception:
        return op


def _current_operators(hi,g):
    """Benchmark-normalized same/opposite triangle loop-current operators.

    This exactly follows the existing benchmark definition:
      J_loop = oriented particle-current sum / 3
      J_same = average over all elementary triangles
      J_opp  = A/B signed average over all elementary triangles
    and S_LC=<J_channel^2>.
    """
    from netket.operator.fermion import create as cdag, destroy as c

    triangles=[lp for lp in g.cycles() if len(lp)==3]
    if len(triangles)!=2*g.num_cells:
        raise RuntimeError(
            f"Expected {2*g.num_cells} elementary triangles, got {len(triangles)}"
        )
    ntri=float(len(triangles))
    same=0.0
    opp=0.0
    for lp in triangles:
        sign_ab=1.0 if lp[0]%6<3 else -1.0
        loopop=0.0
        for i,j in zip(lp,lp[1:]+lp[:1]):
            e,_=g._directed_edge(i,j)
            if e is None:
                raise RuntimeError(f"Nonbond triangle edge {i}-{j}")
            t=float(g.t[e.kind])/len(lp)
            loopop += (
                1j*t*(cdag(hi,i)@c(hi,j))
                -1j*t*(cdag(hi,j)@c(hi,i))
            )
        same += (1.0/ntri)*loopop
        opp += (sign_ab/ntri)*loopop

    # Build the squared channels before converting the operators, then
    # explicitly move all four observables to a fixed-particle-number
    # representation.  This preserves exactly the historical LC definition
    # while reducing the connected-state count in vstate.expect().
    same2=same@same
    opp2=opp@opp

    same=_as_fixed_number_fermion_operator(same)
    opp=_as_fixed_number_fermion_operator(opp)
    same2=_as_fixed_number_fermion_operator(same2)
    opp2=_as_fixed_number_fermion_operator(opp2)
    return same,opp,same2,opp2


def _mc_order_observables(
    vstate, g, nf, hi, current_ops,
    fast_hopping_data=None,
):
    """Evaluate CO and historical LC channels on the current MCState.

    When fast_hopping_data is available, LC means and second moments come from
    the exact local-current identity <J^2>=<|J_loc|^2>, avoiding explicit
    squared-current operators.
    """
    import netket as nk
    import jax.numpy as jnp

    samples=np.asarray(vstate.samples)
    out=_charge_structure_from_samples(g,samples,nf)

    if fast_hopping_data is not None:
        js_loc=jnp.asarray(fast_hopping_data["J_same_loc"])
        jo_loc=jnp.asarray(fast_hopping_data["J_opposite_loc"])
        js_stats=nk.stats.statistics(js_loc)
        jo_stats=nk.stats.statistics(jo_loc)
        ss_stats=nk.stats.statistics(jnp.abs(js_loc)**2)
        os_stats=nk.stats.statistics(jnp.abs(jo_loc)**2)

        j_same=float(np.real(js_stats.mean))
        j_opp=float(np.real(jo_stats.mean))
        s_same=float(np.real(ss_stats.mean))
        s_opp=float(np.real(os_stats.mean))
        out.update({
            "J_same_mean":j_same,
            "J_same_mean_error":stats_error(js_stats),
            "J_opposite_mean":j_opp,
            "J_opposite_mean_error":stats_error(jo_stats),
            "S_LC_same":s_same,
            "S_LC_same_error":stats_error(ss_stats),
            "S_LC_opposite":s_opp,
            "S_LC_opposite_error":stats_error(os_stats),
            "S_LC_same_connected":float(s_same-j_same*j_same),
            "S_LC_opposite_connected":float(s_opp-j_opp*j_opp),
            "LC_estimator":"Woodbury J_loc and |J_loc|^2",
        })
        return out

    same,opp,same2,opp2=current_ops
    sm=vstate.expect(same)
    om=vstate.expect(opp)
    ss=vstate.expect(same2)
    os=vstate.expect(opp2)

    j_same=float(np.real(sm.mean))
    j_opp=float(np.real(om.mean))
    s_same=float(np.real(ss.mean))
    s_opp=float(np.real(os.mean))
    out.update({
        "J_same_mean":j_same,
        "J_same_mean_error":stats_error(sm),
        "J_opposite_mean":j_opp,
        "J_opposite_mean_error":stats_error(om),
        "S_LC_same":s_same,
        "S_LC_same_error":stats_error(ss),
        "S_LC_opposite":s_opp,
        "S_LC_opposite_error":stats_error(os),
        "S_LC_same_connected":float(s_same-j_same*j_same),
        "S_LC_opposite_connected":float(s_opp-j_opp*j_opp),
        "LC_estimator":"NetKet explicit current and squared-current operators",
    })
    return out




def _current_channel_edge_weights(g):
    """Canonical-edge coefficients for the historical same/opposite LC channels.

    For canonical edge e=(i,j), the returned real weights w satisfy
        J_loc contribution = 1j * w[e] * dir * ratio[e],
    where dir=+1 when the root occupied site (fast-kernel src) is e.i and
    dir=-1 when it is e.j.  The ratio already contains the fermionic sign.
    """
    nedge=len(g.edges)
    index={
        tuple(sorted((int(e.i),int(e.j)))):ie
        for ie,e in enumerate(g.edges)
    }
    same=np.zeros(nedge,float)
    opp=np.zeros(nedge,float)
    triangles=[lp for lp in g.cycles() if len(lp)==3]
    if len(triangles)!=2*g.num_cells:
        raise RuntimeError(
            f"Expected {2*g.num_cells} elementary triangles, got {len(triangles)}"
        )
    ntri=float(len(triangles))
    for lp in triangles:
        sign_ab=1.0 if lp[0]%6<3 else -1.0
        for i,j in zip(lp,lp[1:]+lp[:1]):
            e,sgn=g._directed_edge(i,j)
            if e is None:
                raise RuntimeError(f"Nonbond triangle edge {i}-{j}")
            ie=index[tuple(sorted((int(e.i),int(e.j))))]
            amp=float(g.t[e.kind])/float(len(lp))/ntri
            same[ie]+=float(sgn)*amp
            opp[ie]+=sign_ab*float(sgn)*amp
    return same,opp


def _mc_fast_hopping_data(
    vstate,g,fast_map_kernel,chunk_size=None,show_progress=True
):
    """One Woodbury pass producing bond ratios and LC local estimators.

    The same per-edge replacement ratios determine the one-body density matrix
    and the local estimators of both loop-current channels.  For Hermitian J,
        <J^2> = < |J_loc|^2 >,
    so the expensive explicit squared-current operators are unnecessary.
    """
    samples_arr=np.asarray(vstate.samples)
    leading_shape=samples_arr.shape[:-1]
    x=np.asarray(samples_arr,float).reshape((-1,g.Ns))
    nsamp=len(x)
    density=np.mean(x,axis=0)
    edge_i=np.asarray([int(e.i) for e in g.edges],np.int32)
    w_same,w_opp=_current_channel_edge_weights(g)

    if chunk_size is None:
        chunk_size=(
            int(vstate.chunk_size)
            if getattr(vstate,"chunk_size",None) is not None
            else 256
        )
    chunk_size=max(1,int(chunk_size))

    rho_forward_sum=np.zeros(len(g.edges),np.complex128)
    rho_reverse_conj_sum=np.zeros(len(g.edges),np.complex128)
    j_same=np.empty(nsamp,np.complex128)
    j_opp=np.empty(nsamp,np.complex128)
    nchunks=(nsamp+chunk_size-1)//chunk_size
    t0=time.perf_counter()
    if show_progress:
        print(
            "[hopping obs] fast Woodbury pass: "
            f"samples={nsamp} edges={len(g.edges)} chunk={chunk_size} "
            f"chunks={nchunks}",
            flush=True,
        )

    for ichunk,c0 in enumerate(range(0,nsamp,chunk_size),1):
        c1=min(nsamp,c0+chunk_size)
        out=fast_map_kernel(
            vstate.parameters,
            np.asarray(x[c0:c1],dtype=np.float64),
        )
        ratio=np.asarray(out[1],np.complex128)
        active=np.asarray(out[2],bool)
        src=np.asarray(out[3],np.int32)

        forward=active & (src==edge_i[None,:])
        reverse=active & ~forward
        rho_forward_sum += np.sum(
            np.where(forward,ratio,0.0+0.0j),
            axis=0,
        )
        rho_reverse_conj_sum += np.sum(
            np.where(reverse,np.conjugate(ratio),0.0+0.0j),
            axis=0,
        )

        direction=np.where(
            src==edge_i[None,:],1.0,-1.0
        )
        signed_ratio=np.where(
            active,direction*ratio,0.0+0.0j
        )
        j_same[c0:c1]=1j*np.sum(
            signed_ratio*w_same[None,:],axis=1
        )
        j_opp[c0:c1]=1j*np.sum(
            signed_ratio*w_opp[None,:],axis=1
        )

        if show_progress and (
            ichunk==1
            or ichunk==nchunks
            or ichunk % max(1,nchunks//10)==0
        ):
            print(
                f"[hopping obs] chunks {ichunk}/{nchunks}",
                flush=True,
            )

    elapsed=time.perf_counter()-t0
    rho_forward=rho_forward_sum/float(nsamp)
    rho_reverse_conj=rho_reverse_conj_sum/float(nsamp)
    rho=0.5*(rho_forward+rho_reverse_conj)

    # Internal orientation/sign check: reconstruct the mean loop-current
    # channels from the one-body rho values and compare to the direct mean of
    # the per-sample local-current estimators built from the same ratios.
    mean_same_from_rho=0.0
    mean_opp_from_rho=0.0
    index={
        tuple(sorted((int(e.i),int(e.j)))):ie
        for ie,e in enumerate(g.edges)
    }
    triangles=[lp for lp in g.cycles() if len(lp)==3]
    ntri=float(len(triangles))
    for lp in triangles:
        sign_ab=1.0 if lp[0]%6<3 else -1.0
        for i,j in zip(lp,lp[1:]+lp[:1]):
            e,sgn=g._directed_edge(i,j)
            ie=index[tuple(sorted((int(e.i),int(e.j))))]
            z=rho[ie]
            canonical=float(
                (1j*float(g.t[e.kind])*(z-np.conjugate(z))).real
            )
            contrib=float(sgn)*canonical/float(len(lp))/ntri
            mean_same_from_rho+=contrib
            mean_opp_from_rho+=sign_ab*contrib
    mean_same_loc=float(np.real(np.mean(j_same)))
    mean_opp_loc=float(np.real(np.mean(j_opp)))
    check_err=max(
        abs(mean_same_from_rho-mean_same_loc),
        abs(mean_opp_from_rho-mean_opp_loc),
    )
    if check_err>5e-10:
        raise RuntimeError(
            "Fast hopping LC orientation check failed: "
            f"max error={check_err:.3e}"
        )

    if show_progress:
        print(
            f"[hopping obs] fast Woodbury pass done in {elapsed:.2f}s",
            flush=True,
        )
    return {
        "density":density,
        "rho":rho,
        "rho_forward":rho_forward,
        "rho_reverse_conj":rho_reverse_conj,
        "J_same_loc":j_same.reshape(leading_shape),
        "J_opposite_loc":j_opp.reshape(leading_shape),
        "time_s":float(elapsed),
        "n_samples":int(nsamp),
        "chunk_size":int(chunk_size),
    }


def _mc_site_bond_current_fast(
    vstate,g,fast_map_kernel,chunk_size=None,fast_hopping_data=None
):
    """Fast bond/current map from the shared Woodbury hopping data."""
    data=(
        fast_hopping_data
        if fast_hopping_data is not None
        else _mc_fast_hopping_data(
            vstate,g,fast_map_kernel,chunk_size=chunk_size
        )
    )
    density=np.asarray(data["density"],float)
    rho=np.asarray(data["rho"],np.complex128)
    bond=2.0*np.real(rho)
    current=np.asarray([
        float((1j*float(g.t[e.kind])*(z-np.conjugate(z))).real)
        for e,z in zip(g.edges,rho)
    ])
    return density,rho,bond,current



def _mc_site_bond_current(
    vstate,g,edge_chunk=12,log_chunk_size=None
):
    """Estimate density, <c_i^dag c_j>, bond order and current from cached MC samples.

    Off-diagonal one-body density-matrix elements are evaluated from batched
    fermion-hop wavefunction ratios instead of calling expect() once per bond.
    For rho_ij=<c_i^dag c_j>, sampled configurations with i occupied and j
    empty contribute the ratio psi(x with i->j)/psi(x), including the canonical
    fermionic parity between i and j.

    log_value evaluations are explicitly chunked.  MCState.log_value() itself
    does not use MCState.chunk_size, and evaluating tens of thousands of
    backflow-Slater configurations at once can otherwise allocate very large
    temporary Q_BF/Slater tensors during final map generation.
    """
    samples=np.asarray(vstate.samples)
    x=np.asarray(samples,float).reshape((-1,g.Ns))
    nsamp=len(x)
    density=np.mean(x,axis=0)

    if log_chunk_size is None:
        log_chunk_size=(
            int(vstate.chunk_size)
            if getattr(vstate,"chunk_size",None) is not None
            else 256
        )
    log_chunk_size=max(1,int(log_chunk_size))

    def log_value_chunked(configs):
        configs=np.asarray(configs,float)
        vals=[]
        for c0 in range(0,len(configs),log_chunk_size):
            c1=min(len(configs),c0+log_chunk_size)
            vals.append(
                np.asarray(
                    vstate.log_value(configs[c0:c1]),
                    np.complex128,
                ).reshape(-1)
            )
        return (
            np.concatenate(vals)
            if vals else np.empty((0,),np.complex128)
        )

    logx=log_value_chunked(x)
    rho=np.zeros(len(g.edges),np.complex128)

    for lo in range(0,len(g.edges),max(1,int(edge_chunk))):
        hi=min(len(g.edges),lo+max(1,int(edge_chunk)))
        configs=[]
        meta=[]
        for ie in range(lo,hi):
            e=g.edges[ie]
            i,j=int(e.i),int(e.j)
            valid=np.flatnonzero((x[:,i]>0.5)&(x[:,j]<0.5))
            if len(valid)==0:
                continue
            xp=x[valid].copy()
            xp[:,i]=0.0
            xp[:,j]=1.0
            a=min(i,j)+1
            b=max(i,j)
            parity=np.sum(x[valid,a:b],axis=1,dtype=np.int64)&1
            sign=1.0-2.0*parity.astype(float)
            configs.append(xp)
            meta.append((ie,valid,sign,len(xp)))

        if not configs:
            continue
        joined=np.concatenate(configs,axis=0)
        logp=log_value_chunked(joined)
        off=0
        for ie,valid,sign,nv in meta:
            vals=sign*np.exp(logp[off:off+nv]-logx[valid])
            rho[ie]=np.sum(vals)/float(nsamp)
            off+=nv

    bond=2.0*np.real(rho)
    current=np.asarray([
        float((1j*float(g.t[e.kind])*(z-np.conjugate(z))).real)
        for e,z in zip(g.edges,rho)
    ])
    return density,rho,bond,current


def _mc_density_onepoint_meta(vstate,g):
    """Cheap one-point density/CO summary without evaluating bond hops."""
    density=np.mean(
        np.asarray(vstate.samples,float).reshape((-1,g.Ns)),
        axis=0,
    )
    nc=int(g.num_cells)
    filling=float(np.mean(density))
    dcell=density.reshape((nc,6))-filling
    reps=np.asarray(g.reps,float)
    qs=np.asarray(g.permitted_momenta(),float)
    phase=np.exp(
        -2j*np.pi*np.einsum("qd,cd->qc",qs,reps)
    )/np.sqrt(float(nc))
    nq=np.einsum("qc,ca->qa",phase,dcell,optimize=True)
    co1_sq=np.sum(np.abs(nq)**2,axis=1)/(6.0*nc)
    ico=int(np.argmax(co1_sq))
    return {
        "mean_density":float(np.mean(density)),
        "density_min":float(np.min(density)),
        "density_max":float(np.max(density)),
        "density_rms_modulation":float(np.std(density)),
        "CO_onepoint_rms":float(np.sqrt(max(0.0,co1_sq[ico]))),
        "CO_onepoint_q1":float(qs[ico,0]),
        "CO_onepoint_q2":float(qs[ico,1]),
        "B_star_mean":None,
        "max_abs_bond_order":None,
        "max_abs_current":None,
        "n_samples":int(len(np.asarray(vstate.samples).reshape((-1,g.Ns)))),
        "estimator":"density-only one-point summary from cached MC samples",
    }


def _save_mc_maps(
    directory,vstate,g,title,edge_chunk=12,log_chunk_size=None,
    fast_map_kernel=None,fast_map_chunk_size=None,
    fast_hopping_data=None,
):
    """Save site/bond/current CSVs and a single triptych for a large-system MC state."""
    from plot_saved_state_maps import (
        site_positions,edge_segments,export_site_csv,export_bond_csv,
        plot_site_bond_current_triptych,
    )
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    t_map_est=time.perf_counter()
    if fast_map_kernel is not None:
        density,rho,bond,current=_mc_site_bond_current_fast(
            vstate,g,fast_map_kernel,
            chunk_size=fast_map_chunk_size,
            fast_hopping_data=fast_hopping_data,
        )
        map_estimator="rank-3 Woodbury ratios on cached MC roots"
    else:
        print(
            "[maps] using full-logpsi fallback estimator",
            flush=True,
        )
        density,rho,bond,current=_mc_site_bond_current(
            vstate,g,
            edge_chunk=edge_chunk,
            log_chunk_size=log_chunk_size,
        )
        map_estimator=(
            "batched fermion-hop full-logpsi ratios on cached MC samples"
        )
    map_estimator_s=time.perf_counter()-t_map_est

    # Validate the batched fermionic hop/sign convention against NetKet on
    # one representative edge using the exact same cached samples.
    try:
        from netket.operator.fermion import create as cdag, destroy as c
        e0=g.edges[0]
        ref_op=cdag(vstate.hilbert,e0.i)@c(vstate.hilbert,e0.j)
        ref_op=_as_fixed_number_fermion_operator(ref_op)
        ref=vstate.expect(ref_op)
        ref_mean=complex(np.asarray(ref.mean))
        err=abs(rho[0]-ref_mean)
        scale=max(1.0,abs(ref_mean))
        if err>5e-7*scale:
            raise RuntimeError(
                "MC map fermion-hop estimator disagrees with NetKet on the "
                f"validation edge: rho_fast={rho[0]}, rho_netket={ref_mean}, "
                f"|diff|={err:.3e}"
            )
    except RuntimeError:
        raise
    except Exception as exc:
        print(
            f"[map warning] one-edge NetKet rho validation skipped: {exc}",
            flush=True,
        )

    t_map_export=time.perf_counter()
    xy,cells=site_positions(g)
    starts,ends,crossed=edge_segments(g,xy,cells)
    export_site_csv(directory/"site_density.csv",g,xy,density)
    export_bond_csv(
        directory/"bond_observables.csv",
        g,starts,ends,crossed,rho,bond,current,
    )
    plot_site_bond_current_triptych(
        g,xy,starts,ends,crossed,density,bond,current,directory,title,
        threshold_fraction=0.02,
    )
    from benchmark_observables import ruby_star_formfactor
    fstar=np.asarray(ruby_star_formfactor(g),float)
    bstar=float(np.dot(fstar,bond)/len(g.edges))

    nc=int(g.num_cells)
    filling=float(np.mean(density))
    dcell=density.reshape((nc,6))-filling
    reps=np.asarray(g.reps,float)
    qs=np.asarray(g.permitted_momenta(),float)
    phase=np.exp(-2j*np.pi*np.einsum("qd,cd->qc",qs,reps))/np.sqrt(float(nc))
    nq=np.einsum("qc,ca->qa",phase,dcell,optimize=True)
    # RMS density modulation per site in the strongest primitive momentum.
    co1_sq=np.sum(np.abs(nq)**2,axis=1)/(6.0*nc)
    ico=int(np.argmax(co1_sq))

    meta={
        "mean_density":float(np.mean(density)),
        "density_min":float(np.min(density)),
        "density_max":float(np.max(density)),
        "density_rms_modulation":float(np.std(density)),
        "CO_onepoint_rms":float(np.sqrt(max(0.0,co1_sq[ico]))),
        "CO_onepoint_q1":float(qs[ico,0]),
        "CO_onepoint_q2":float(qs[ico,1]),
        "B_star_mean":bstar,
        "max_abs_bond_order":float(np.max(np.abs(bond))),
        "max_abs_current":float(np.max(np.abs(current))),
        "n_samples":int(np.asarray(vstate.samples).reshape((-1,g.Ns)).shape[0]),
        "estimator":map_estimator,
        "time_map_estimator_s":float(map_estimator_s),
        "time_map_export_plot_s":float(
            time.perf_counter()-t_map_export
        ),
    }
    atomic_json(directory/"map_summary.json",meta)
    return meta


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
    """Choose a QGT implementation compatible with the mixed real/complex ansatz.

    The current large-supercell model always has a real Jastrow leaf v and
    complex orbital/backflow leaves.  It is therefore non-holomorphic as a
    parameter PyTree.  QGTOnTheFly + CG can map the real v tangent to complex
    output, which violates JAX CG's fixed-leaf-dtype requirement.  The safe
    memory-scalable choice is QGTJacobianPyTree(mode="complex"): it stores the
    Jacobian but never constructs the dense P x P QGT.
    """
    requested=str(args.qgt).lower()
    if requested=="dense":
        effective="dense"
        qgt=partial(
            nk.optimizer.qgt.QGTJacobianDense,
            mode="complex",
            chunk_size=args.qgt_chunk_size,
        )
    elif requested=="pytree":
        effective="pytree"
        qgt=partial(
            nk.optimizer.qgt.QGTJacobianPyTree,
            mode="complex",
            chunk_size=args.qgt_chunk_size,
        )
    elif requested in ("auto","onthefly"):
        # Even an explicit --qgt onthefly request is redirected because the
        # present parameter tree is mixed float64/complex128.  This preserves
        # the physical real-Jastrow ansatz instead of complexifying v merely
        # to satisfy the iterative solver.
        effective="pytree"
        qgt=partial(
            nk.optimizer.qgt.QGTJacobianPyTree,
            mode="complex",
            chunk_size=args.qgt_chunk_size,
        )
    else:
        raise ValueError(requested)

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
    return qgt,solver,effective


def _sync_tree(jax,tree):
    """Block until all JAX leaves in a PyTree are ready."""
    def _sync(x):
        try:
            return x.block_until_ready()
        except Exception:
            return x
    return jax.tree_util.tree_map(_sync,tree)


def _gauge_fix_packed_orbitals(vstate, occupations_per_k):
    """QR each occupied k block in-place without changing the physical Raw state."""
    import jax.numpy as jnp
    from flax.core import FrozenDict, freeze, unfreeze

    counts=np.asarray(occupations_per_k,int)
    params=vstate.parameters
    if "M_packed" not in params:
        return
    packed=params["M_packed"]
    offsets=np.concatenate(([0],np.cumsum(counts))).astype(int)
    blocks=[]
    for ik,nocc in enumerate(counts):
        nocc=int(nocc)
        if nocc==0:
            continue
        lo=int(offsets[ik]); hi=int(offsets[ik+1])
        q,_=jnp.linalg.qr(packed[:,lo:hi],mode="reduced")
        blocks.append(q)
    if not blocks:
        return
    newpacked=jnp.concatenate(blocks,axis=1)
    mutable=unfreeze(params) if isinstance(params,FrozenDict) else dict(params)
    mutable["M_packed"]=newpacked
    vstate.parameters=(
        freeze(mutable) if isinstance(params,FrozenDict) else mutable
    )


def _make_orbital_gauge_fix_callback(nk, occupations_per_k, interval):
    interval=max(0,int(interval))
    if interval<=0:
        return None

    class OrbitalGaugeFix(nk.callbacks.AbstractCallback):
        @staticmethod
        def _reset_solver_restart(driver):
            if hasattr(driver,"_shared_x0"):
                driver._shared_x0=None
            pre=getattr(driver,"preconditioner",None)
            if pre is not None and hasattr(pre,"x0"):
                pre.x0=None

        def on_step_end(self,step,log_data,driver):
            del log_data
            if (int(step)+1)%interval==0:
                _gauge_fix_packed_orbitals(
                    driver.state,occupations_per_k
                )
                self._reset_solver_restart(driver)

        def on_run_end(self,step,driver):
            del step
            _gauge_fix_packed_orbitals(
                driver.state,occupations_per_k
            )
            self._reset_solver_restart(driver)

    return OrbitalGaugeFix()


def _validate_analytic_logderiv(
    *,
    vstate_analytic,
    vstate_reference,
    H_reference,
    jax,
    timing_repeats=3,
):
    """Compare analytic-JVP and original-autodiff Raw energy gradients."""
    vstate_analytic.sample()
    samples=vstate_analytic.samples
    # Setting parameters calls VariationalState.reset(), which clears the MC
    # sample cache.  Therefore synchronise parameters FIRST and inject the
    # identical cached samples only afterwards.
    vstate_reference.parameters=vstate_analytic.parameters
    # Supported NetKet extension point: inject the same cached samples so the
    # two gradient estimates are strictly comparable.
    vstate_reference._samples=samples

    stats_a,grad_a=vstate_analytic.expect_and_grad(H_reference)
    _sync_tree(jax,grad_a)
    stats_r,grad_r=vstate_reference.expect_and_grad(H_reference)
    _sync_tree(jax,grad_r)

    report=_compare_parameter_trees(grad_r,grad_a)
    Ea=complex(np.asarray(stats_a.mean))
    Er=complex(np.asarray(stats_r.mean))
    report.update({
        "energy_abs_error":float(abs(Ea-Er)),
        "analytic_energy_real":float(Ea.real),
        "reference_energy_real":float(Er.real),
    })

    # The calls above also serve as JIT warmup. Benchmark both derivative
    # paths on the exact same cached samples without resampling.
    nrep=max(1,int(timing_repeats))
    ta=[]
    tr=[]
    for _ in range(nrep):
        t=time.perf_counter()
        st,gr=vstate_analytic.expect_and_grad(H_reference)
        try:
            np.asarray(st.mean)
        except Exception:
            pass
        _sync_tree(jax,gr)
        ta.append(time.perf_counter()-t)

        t=time.perf_counter()
        st,gr=vstate_reference.expect_and_grad(H_reference)
        try:
            np.asarray(st.mean)
        except Exception:
            pass
        _sync_tree(jax,gr)
        tr.append(time.perf_counter()-t)

    report["analytic_expect_and_grad_median_s"]=float(np.median(ta))
    report["autodiff_expect_and_grad_median_s"]=float(np.median(tr))
    report["expect_and_grad_speedup"]=float(
        report["autodiff_expect_and_grad_median_s"]
        / max(1e-15,report["analytic_expect_and_grad_median_s"])
    )

    print(
        "[analytic logderiv check] "
        f"dE={report['energy_abs_error']:.3e} "
        f"grad_max={report['max_abs_error']:.3e} "
        f"grad_rel_l2={report['relative_l2_error']:.3e} "
        f"speedup={report['expect_and_grad_speedup']:.2f}x "
        f"({report['autodiff_expect_and_grad_median_s']:.3f}s->"
        f"{report['analytic_expect_and_grad_median_s']:.3f}s)",
        flush=True,
    )
    return report


def _make_shared_sr_linear_solve(jax,solver):
    """JIT the matrix-free O^T O solve once, with O passed dynamically."""
    import jax.numpy as jnp

    @jax.jit
    def solve(O,dv,diag_shift,x0):
        force=O.T@dv
        shift=jnp.asarray(diag_shift,dtype=force.dtype)

        def matvec(vec):
            return O.T@(O@vec)+shift*vec

        out=solver(matvec,force,x0=x0)
        if isinstance(out,tuple):
            return out
        return out,None

    return solve


def _shared_sr_update(
    *,
    vstate,
    H,
    nk,
    jax,
    diag_shift,
    linear_solve,
    chunk_size,
    x0=None,
):
    """One mixed-parameter SR update from a single explicit Jacobian.

    This mirrors the standard-QGT branch of NetKet's VMC_SR, but first maps the
    mixed real/complex parameter PyTree to a purely real coordinate PyTree.
    The same centered Jacobian is then used both for the energy force and the
    SR matrix-vector products, avoiding the duplicate log-derivative work of
    legacy VMC(expect_and_grad)+SR.
    """
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    local_estimators=vstate.local_estimators(H)
    stats=local_estimators.to_stats()
    local_e=jnp.asarray(local_estimators.data).reshape(-1)

    samples=vstate.samples
    if samples.ndim>=3:
        samples=jax.lax.collapse(samples,0,samples.ndim-1)
    n_samples=int(samples.shape[0])

    params=vstate.parameters
    params_real,reassemble=nk.jax.tree_to_real(params)
    _,unravel_real=ravel_pytree(params_real)

    jac=nk.jax.jacobian(
        vstate._apply_fun,
        params,
        samples,
        vstate.model_state,
        mode="complex",
        dense=True,
        center=True,
        chunk_size=chunk_size,
    )

    inv_sqrt_n=1.0/jnp.sqrt(jnp.asarray(float(n_samples)))
    O=jac*inv_sqrt_n
    # complex mode: (Ns,2,Nreal) -> (2*Ns,Nreal)
    O=jax.lax.collapse(O,0,2)

    de=local_e-jnp.mean(local_e)
    dv=2.0*de*inv_sqrt_n
    dv=jnp.stack((jnp.real(dv),jnp.imag(dv)),axis=-1)
    dv=jax.lax.collapse(dv,0,2)

    update_real,info=linear_solve(
        O,dv,jnp.asarray(diag_shift),x0
    )

    update_tree_real=unravel_real(update_real)
    update_tree=reassemble(update_tree_real)
    update_tree=nk.jax.tree_cast(update_tree,params)

    return stats,update_tree,update_real,info


def _make_shared_sr_driver(
    nk,
    jax,
    *,
    H,
    optimizer,
    vstate,
    diag_shift,
    solver,
    chunk_size,
):
    class SharedJacobianVMC(nk.driver.VMC):
        # NetKet drivers are struct.Pytree subclasses.  Every attribute added
        # by a subclass must therefore be declared in the class body before
        # __init__.  Static Python callables/settings are excluded from the
        # traced PyTree; the previous CG solution is dynamic state.
        _shared_diag_shift: object = nk.utils.struct.field(
            pytree_node=False,serialize=False,default=None
        )
        _shared_linear_solve: object = nk.utils.struct.field(
            pytree_node=False,serialize=False,default=None
        )
        _shared_chunk_size: object = nk.utils.struct.field(
            pytree_node=False,serialize=False,default=None
        )
        _shared_x0: object = nk.utils.struct.field(
            pytree_node=True,default=None
        )

        def __init__(self):
            super().__init__(
                H,optimizer,
                variational_state=vstate,
                preconditioner=None,
            )
            self._shared_diag_shift=diag_shift
            self._shared_linear_solve=_make_shared_sr_linear_solve(
                jax,solver
            )
            self._shared_chunk_size=chunk_size
            self._shared_x0=None
            self.info=None

        def compute_loss_and_update(self):
            self.state.reset()
            shift=self._shared_diag_shift
            if callable(shift):
                shift=shift(self.step_count)
            stats,dp,x0,info=_shared_sr_update(
                vstate=self.state,
                H=self._ham,
                nk=nk,
                jax=jax,
                diag_shift=shift,
                linear_solve=self._shared_linear_solve,
                chunk_size=self._shared_chunk_size,
                x0=self._shared_x0,
            )
            self._loss_stats=stats
            self._dp=dp
            self._shared_x0=x0
            self.info=info
            return stats,dp

    return SharedJacobianVMC()


def _validate_shared_sr(
    *,
    vstate,
    H,
    nk,
    jax,
    qgt_ctor,
    solver,
    diag_shift,
    chunk_size,
    repeats=3,
):
    """Compare shared-Jacobian SR against legacy expect_and_grad + SR."""
    vstate.sample()
    np.asarray(vstate.samples)

    try:
        ref_sr=nk.optimizer.SR(
            qgt=qgt_ctor,
            solver=solver,
            diag_shift=diag_shift,
            solver_restart=False,
        )
    except TypeError:
        ref_sr=nk.optimizer.SR(
            qgt=qgt_ctor,
            solver=solver,
            diag_shift=diag_shift,
        )

    shared_linear_solve=_make_shared_sr_linear_solve(jax,solver)

    # Warmup both paths on the identical cached sample pool.
    st_ref,grad_ref=vstate.expect_and_grad(H)
    _sync_tree(jax,grad_ref)
    dp_ref=ref_sr(vstate,grad_ref,step=0)
    _sync_tree(jax,dp_ref)

    st_shared,dp_shared,_,_=_shared_sr_update(
        vstate=vstate,H=H,nk=nk,jax=jax,
        diag_shift=diag_shift,linear_solve=shared_linear_solve,
        chunk_size=chunk_size,x0=None,
    )
    _sync_tree(jax,dp_shared)

    report=_compare_parameter_trees(dp_ref,dp_shared)
    Er=complex(np.asarray(st_ref.mean))
    Es=complex(np.asarray(st_shared.mean))
    report["energy_abs_error"]=float(abs(Er-Es))

    legacy_times=[]
    shared_times=[]
    nrep=max(1,int(repeats))
    for _ in range(nrep):
        t=time.perf_counter()
        st,gr=vstate.expect_and_grad(H)
        try:
            np.asarray(st.mean)
        except Exception:
            pass
        _sync_tree(jax,gr)
        pre=ref_sr(vstate,gr,step=0)
        _sync_tree(jax,pre)
        legacy_times.append(time.perf_counter()-t)

        t=time.perf_counter()
        st,pre,_,_=_shared_sr_update(
            vstate=vstate,H=H,nk=nk,jax=jax,
            diag_shift=diag_shift,linear_solve=shared_linear_solve,
            chunk_size=chunk_size,x0=None,
        )
        try:
            np.asarray(st.mean)
        except Exception:
            pass
        _sync_tree(jax,pre)
        shared_times.append(time.perf_counter()-t)

    report["legacy_median_s"]=float(np.median(legacy_times))
    report["shared_median_s"]=float(np.median(shared_times))
    report["speedup"]=float(
        report["legacy_median_s"]/max(1e-15,report["shared_median_s"])
    )
    print(
        "[shared SR check] "
        f"dE={report['energy_abs_error']:.3e} "
        f"update_max={report['max_abs_error']:.3e} "
        f"update_rel_l2={report['relative_l2_error']:.3e} "
        f"speedup={report['speedup']:.2f}x "
        f"({report['legacy_median_s']:.3f}s->"
        f"{report['shared_median_s']:.3f}s)",
        flush=True,
    )
    return report


def _profile_shared_sr_components(
    *,
    vstate,
    H,
    nk,
    jax,
    solver,
    diag_shift,
    chunk_size,
    repeats,
):
    """Profile the single-Jacobian SR path on one fixed MC sample pool."""
    import jax.numpy as jnp

    repeats=max(0,int(repeats))
    if repeats<=0:
        return {}

    vstate.sample()
    np.asarray(vstate.samples)

    samples=vstate.samples
    if samples.ndim>=3:
        samples=jax.lax.collapse(samples,0,samples.ndim-1)
    n_samples=int(samples.shape[0])
    params=vstate.parameters
    inv_sqrt_n=1.0/jnp.sqrt(jnp.asarray(float(n_samples)))
    linear_solve=_make_shared_sr_linear_solve(jax,solver)

    def get_local():
        le=vstate.local_estimators(H)
        np.asarray(le.data)
        return le

    def get_jac():
        jac=nk.jax.jacobian(
            vstate._apply_fun,
            params,
            samples,
            vstate.model_state,
            mode="complex",
            dense=True,
            center=True,
            chunk_size=chunk_size,
        )
        np.asarray(jac)
        return jac

    def prepare_and_solve(jac,local_e,x0=None):
        O=jac*inv_sqrt_n
        O=jax.lax.collapse(O,0,2)
        de=local_e-jnp.mean(local_e)
        dv=2.0*de*inv_sqrt_n
        dv=jnp.stack((jnp.real(dv),jnp.imag(dv)),axis=-1)
        dv=jax.lax.collapse(dv,0,2)
        sol,info=linear_solve(
            O,dv,jnp.asarray(diag_shift),x0
        )
        del info
        np.asarray(sol)
        return sol

    # Warmup.
    le=get_local()
    jac=get_jac()
    prepare_and_solve(
        jac,jnp.asarray(le.data).reshape(-1),None
    )
    st,dp,_,_=_shared_sr_update(
        vstate=vstate,H=H,nk=nk,jax=jax,
        diag_shift=diag_shift,
        linear_solve=linear_solve,
        chunk_size=chunk_size,
        x0=None,
    )
    try:
        np.asarray(st.mean)
    except Exception:
        pass
    _sync_tree(jax,dp)

    local_times=[]
    jac_times=[]
    solve_times=[]
    total_times=[]
    for _ in range(repeats):
        t=time.perf_counter()
        le=get_local()
        local_times.append(time.perf_counter()-t)

        t=time.perf_counter()
        jac=get_jac()
        jac_times.append(time.perf_counter()-t)

        local_e=jnp.asarray(le.data).reshape(-1)
        t=time.perf_counter()
        prepare_and_solve(jac,local_e,None)
        solve_times.append(time.perf_counter()-t)

        t=time.perf_counter()
        st,dp,_,_=_shared_sr_update(
            vstate=vstate,H=H,nk=nk,jax=jax,
            diag_shift=diag_shift,
            linear_solve=linear_solve,
            chunk_size=chunk_size,
            x0=None,
        )
        try:
            np.asarray(st.mean)
        except Exception:
            pass
        _sync_tree(jax,dp)
        total_times.append(time.perf_counter()-t)

    def med(xs):
        return float(np.median(np.asarray(xs,float)))

    out={
        "repeats":int(repeats),
        "local_energy_median_s":med(local_times),
        "jacobian_median_s":med(jac_times),
        "prepare_solve_median_s":med(solve_times),
        "total_median_s":med(total_times),
    }
    out["accounted_median_s"]=float(
        out["local_energy_median_s"]
        +out["jacobian_median_s"]
        +out["prepare_solve_median_s"]
    )
    print(
        "[shared SR profile] "
        f"local={out['local_energy_median_s']:.3f}s "
        f"jac={out['jacobian_median_s']:.3f}s "
        f"solve={out['prepare_solve_median_s']:.3f}s "
        f"total={out['total_median_s']:.3f}s",
        flush=True,
    )
    return out


def _profile_vmc_components(
    *,
    vstate,H_fast,jax,nk,
    qgt_ctor,solver,diag_shift,
    repeats,
    H_reference=None,
):
    """Microprofile the expensive VMC components without updating parameters.

    The timings are intentionally component-level:
      sample            : Markov-chain generation only;
      expect            : local-energy evaluation/statistics on cached samples;
      expect_and_grad   : local energy plus log-derivative/Jacobian gradient;
      sr                : QGT construction/application plus the linear solve.

    One untimed pass warms up JIT compilation.  The profiler uses its own SR
    object so that its solver-restart state cannot affect the training driver.
    """
    repeats=max(0,int(repeats))
    if repeats<=0:
        return {}

    profile_sr=None
    if qgt_ctor is not None and solver is not None:
        try:
            profile_sr=nk.optimizer.SR(
                qgt=qgt_ctor,
                solver=solver,
                diag_shift=diag_shift,
                solver_restart=True,
            )
        except TypeError:
            profile_sr=nk.optimizer.SR(
                qgt=qgt_ctor,
                solver=solver,
                diag_shift=diag_shift,
            )

    def _sync_stats(st):
        try:
            np.asarray(st.mean)
        except Exception:
            pass

    # JIT warmup. These calls do not update variational parameters.
    vstate.sample()
    np.asarray(vstate.samples)
    st=vstate.expect(H_fast)
    _sync_stats(st)
    st,grad=vstate.expect_and_grad(H_fast)
    _sync_stats(st)
    _sync_tree(jax,grad)
    if H_reference is not None:
        st_ref=vstate.expect(H_reference)
        _sync_stats(st_ref)
        st_ref,grad_ref=vstate.expect_and_grad(H_reference)
        _sync_stats(st_ref)
        _sync_tree(jax,grad_ref)
    if profile_sr is not None:
        pgrad=profile_sr(vstate,grad,step=0)
        _sync_tree(jax,pgrad)

    sample_times=[]
    expect_times=[]
    grad_times=[]
    ref_expect_times=[]
    ref_grad_times=[]
    sr_times=[]
    for _ in range(repeats):
        t=time.perf_counter()
        vstate.sample()
        np.asarray(vstate.samples)
        sample_times.append(time.perf_counter()-t)

        if H_reference is not None:
            t=time.perf_counter()
            st_ref=vstate.expect(H_reference)
            _sync_stats(st_ref)
            ref_expect_times.append(time.perf_counter()-t)

            t=time.perf_counter()
            st_ref,grad_ref=vstate.expect_and_grad(H_reference)
            _sync_stats(st_ref)
            _sync_tree(jax,grad_ref)
            ref_grad_times.append(time.perf_counter()-t)

        t=time.perf_counter()
        st=vstate.expect(H_fast)
        _sync_stats(st)
        expect_times.append(time.perf_counter()-t)

        t=time.perf_counter()
        st,grad=vstate.expect_and_grad(H_fast)
        _sync_stats(st)
        _sync_tree(jax,grad)
        grad_times.append(time.perf_counter()-t)

        if profile_sr is not None:
            t=time.perf_counter()
            pgrad=profile_sr(vstate,grad,step=0)
            _sync_tree(jax,pgrad)
            sr_times.append(time.perf_counter()-t)

    def _stats(xs):
        if not xs:
            return None
        arr=np.asarray(xs,float)
        return {
            "median_s":float(np.median(arr)),
            "min_s":float(np.min(arr)),
            "max_s":float(np.max(arr)),
        }

    out={
        "repeats":int(repeats),
        "sample":_stats(sample_times),
        "expect":_stats(expect_times),
        "expect_and_grad":_stats(grad_times),
        "reference_expect":_stats(ref_expect_times),
        "reference_expect_and_grad":_stats(ref_grad_times),
        "sr":_stats(sr_times),
    }
    if (
        out["reference_expect"] is not None
        and out["expect"] is not None
    ):
        out["expect_speedup_vs_reference"]=float(
            out["reference_expect"]["median_s"]
            / max(1e-15,out["expect"]["median_s"])
        )
    if (
        out["reference_expect_and_grad"] is not None
        and out["expect_and_grad"] is not None
    ):
        out["expect_and_grad_speedup_vs_reference"]=float(
            out["reference_expect_and_grad"]["median_s"]
            / max(1e-15,out["expect_and_grad"]["median_s"])
        )
    if out["expect"] is not None and out["expect_and_grad"] is not None:
        out["gradient_overhead_median_s"]=float(
            max(
                0.0,
                out["expect_and_grad"]["median_s"]
                - out["expect"]["median_s"],
            )
        )
    msg=(
        "[profile] "
        f"sample={out['sample']['median_s']:.3f}s "
        f"expect={out['expect']['median_s']:.3f}s "
        f"expect+grad={out['expect_and_grad']['median_s']:.3f}s "
        + (
            f"SR={out['sr']['median_s']:.3f}s"
            if out["sr"] is not None else "SR=n/a"
        )
    )
    if out["reference_expect"] is not None:
        msg+=(
            " | reference "
            f"expect={out['reference_expect']['median_s']:.3f}s "
            f"expect+grad={out['reference_expect_and_grad']['median_s']:.3f}s "
            f"| speedup={out['expect_speedup_vs_reference']:.2f}x/"
            f"{out['expect_and_grad_speedup_vs_reference']:.2f}x"
        )
    print(msg,flush=True)
    return out


def _validate_fast_raw_local_energy(
    *,
    vstate,g,V,seed_data,shell_mats,bf_mats,
    n_roots=8,
    repeats=3,
    orthonormalize_orbitals=True,
):
    """Validate the Woodbury Raw local-energy prototype against full logpsi.

    This is deliberately a post-training diagnostic only.  It compares both
    every active hopping ratio and the final local energy, so sign mistakes
    cannot hide behind cancellations among bonds.
    """
    import jax.numpy as jnp

    roots=np.asarray(vstate.samples).reshape((-1,g.Ns))
    n_roots=max(1,min(int(n_roots),len(roots)))
    roots=np.asarray(roots[:n_roots],float)

    edge_i=np.asarray([int(e.i) for e in g.edges],np.int32)
    edge_j=np.asarray([int(e.j) for e in g.edges],np.int32)
    edge_t=np.asarray([float(g.t[e.kind]) for e in g.edges],float)
    edge_v=np.asarray([1.0 if e.kind=="t" else 0.0 for e in g.edges],float)

    kernel=make_fast_raw_local_energy_kernel(
        sc_coords=seed_data["sc_coords"],
        internal_sites=seed_data["internal_sites"],
        kpoints=seed_data["kpoints"],
        occupations_per_k=seed_data["occupations_per_k"],
        shell_matrices=shell_mats,
        backflow_matrices=bf_mats,
        edge_i=edge_i,
        edge_j=edge_j,
        edge_t=edge_t,
        edge_has_interaction=edge_v,
        interaction_V=V,
        orthonormalize_orbitals=orthonormalize_orbitals,
    )

    # Warm up JIT.
    fast=kernel(vstate.parameters,jnp.asarray(roots))
    fast_E=np.asarray(fast[0])
    fast_ratio=np.asarray(fast[1])
    fast_active=np.asarray(fast[2],bool)
    fast_src=np.asarray(fast[3],int)
    fast_dst=np.asarray(fast[4],int)

    fast_times=[]
    for _ in range(max(1,int(repeats))):
        t=time.perf_counter()
        out=kernel(vstate.parameters,jnp.asarray(roots))
        vals=np.asarray(out[0])
        fast_times.append(time.perf_counter()-t)
        fast_E=vals

    root_log=np.asarray(vstate.log_value(roots),np.complex128).reshape(-1)
    xprime=[]
    root_id=[]
    edge_id=[]
    signs=[]
    direct_diag=np.zeros(n_roots,np.complex128)
    for ir,x in enumerate(roots):
        direct_diag[ir]=float(V)*sum(
            1.0
            for e in g.edges
            if e.kind=="t" and x[int(e.i)]>0.5 and x[int(e.j)]>0.5
        )
        for ie,e in enumerate(g.edges):
            i=int(e.i); j=int(e.j)
            xi=x[i]>0.5; xj=x[j]>0.5
            if xi==xj:
                continue
            src=i if xi else j
            dst=j if xi else i
            xp=x.copy()
            xp[src]=0.0
            xp[dst]=1.0
            lo=min(src,dst)+1
            hi=max(src,dst)
            n_between=int(np.sum(x[lo:hi]))
            sign=-1.0 if (n_between&1) else 1.0
            xprime.append(xp)
            root_id.append(ir)
            edge_id.append(ie)
            signs.append(sign)

    xprime=np.asarray(xprime,float)
    root_id=np.asarray(root_id,int)
    edge_id=np.asarray(edge_id,int)
    signs=np.asarray(signs,float)

    t=time.perf_counter()
    if len(xprime):
        logp=np.asarray(vstate.log_value(xprime),np.complex128).reshape(-1)
    else:
        logp=np.empty(0,np.complex128)
    direct_logpsi_s=time.perf_counter()-t

    direct_E=direct_diag.copy()
    ratio_abs_errors=[]
    ratio_rel_errors=[]
    for k in range(len(xprime)):
        ir=int(root_id[k]); ie=int(edge_id[k])
        canonical_ratio=np.exp(logp[k]-root_log[ir])
        replacement_ref=signs[k]*canonical_ratio
        direct_E[ir]+=edge_t[ie]*replacement_ref

        replacement_fast=fast_ratio[ir,ie]
        ae=abs(replacement_fast-replacement_ref)
        re=ae/max(1e-12,abs(replacement_ref))
        ratio_abs_errors.append(float(ae))
        ratio_rel_errors.append(float(re))

        if not fast_active[ir,ie]:
            raise RuntimeError(
                "Fast local-energy active-edge mask disagrees with direct hop list"
            )
        expected_src=(
            edge_i[ie] if roots[ir,edge_i[ie]]>0.5 else edge_j[ie]
        )
        expected_dst=(
            edge_j[ie] if roots[ir,edge_i[ie]]>0.5 else edge_i[ie]
        )
        if (
            int(fast_src[ir,ie])!=int(expected_src)
            or int(fast_dst[ir,ie])!=int(expected_dst)
        ):
            raise RuntimeError("Fast local-energy hop orientation mismatch")

    energy_abs=np.abs(fast_E-direct_E)
    energy_rel=energy_abs/np.maximum(1e-12,np.abs(direct_E))
    report={
        "n_roots":int(n_roots),
        "n_active_hops":int(len(xprime)),
        "ratio_max_abs_error":(
            float(max(ratio_abs_errors)) if ratio_abs_errors else 0.0
        ),
        "ratio_max_rel_error":(
            float(max(ratio_rel_errors)) if ratio_rel_errors else 0.0
        ),
        "local_energy_max_abs_error":float(np.max(energy_abs)),
        "local_energy_max_rel_error":float(np.max(energy_rel)),
        "fast_kernel_median_s":float(np.median(fast_times)),
        "direct_connected_logpsi_s":float(direct_logpsi_s),
    }
    print(
        "[fast local check] "
        f"roots={report['n_roots']} hops={report['n_active_hops']} "
        f"ratio_abs={report['ratio_max_abs_error']:.3e} "
        f"Eloc_abs={report['local_energy_max_abs_error']:.3e} "
        f"fast={report['fast_kernel_median_s']:.4f}s "
        f"direct_logpsi={report['direct_connected_logpsi_s']:.4f}s",
        flush=True,
    )
    return report


def _validate_fast_raw_sampler_sweep(
    *,
    vstate,
    g,
    seed_data,
    shell_mats,
    bf_mats,
    sweep_size,
    rng_seed,
    n_roots=8,
    repeats=3,
    n_sweeps=40,
    orthonormalize_orbitals=False,
):
    """Compare persistent Woodbury cache with full-logpsi Metropolis dynamics."""
    import jax
    import jax.numpy as jnp

    roots=np.asarray(vstate.samples).reshape((-1,g.Ns))
    n_roots=max(1,min(int(n_roots),len(roots)))
    roots=np.asarray(roots[:n_roots],np.int8)
    sweep_size=int(sweep_size)
    n_sweeps=max(1,int(n_sweeps))
    n_steps=sweep_size*n_sweeps
    clusters=np.asarray(
        [(int(e.i),int(e.j)) for e in g.edges],
        np.int32,
    )

    key=jax.random.PRNGKey(int(rng_seed)+99173)
    k1,k2=jax.random.split(key)
    up=np.asarray(
        jax.random.uniform(k1,(n_roots,n_steps),dtype=jnp.float64)
    )
    ua=np.asarray(
        jax.random.uniform(k2,(n_roots,n_steps),dtype=jnp.float64)
    )

    init_cache,advance,inverse_residual=make_fast_raw_sampler_cache_kernels(
        sc_coords=seed_data["sc_coords"],
        internal_sites=seed_data["internal_sites"],
        kpoints=seed_data["kpoints"],
        occupations_per_k=seed_data["occupations_per_k"],
        shell_matrices=shell_mats,
        backflow_matrices=bf_mats,
        clusters=clusters,
        machine_pow=2.0,
        orthonormalize_orbitals=orthonormalize_orbitals,
    )

    clusters_j=jnp.asarray(clusters)
    apply_fun=vstate._apply_fun
    model_state=vstate.model_state

    def choose_hoppable(x,u):
        a=clusters_j[:,0]
        b=clusters_j[:,1]
        mask=x[a]!=x[b]
        n_active=jnp.sum(mask,dtype=jnp.int32)
        rank=jnp.minimum(
            jnp.asarray(jnp.floor(u*n_active),jnp.int32),
            n_active-1,
        )
        cumulative=jnp.cumsum(mask.astype(jnp.int32))
        edge=jnp.argmax(cumulative>rank)
        return edge,n_active

    def direct_kernel(parameters,configurations,proposal_uniforms,accept_uniforms):
        variables={"params":parameters,**model_state}

        def one(x,u_prop,u_acc):
            x=x.astype(jnp.int8)
            logp=2.0*jnp.real(apply_fun(variables,x))
            logp0=logp

            def step(carry,inputs):
                x,logp,naccepted=carry
                up_i,ua_i=inputs
                edge,n_active=choose_hoppable(x,up_i)
                i=clusters_j[edge,0]
                j=clusters_j[edge,1]
                src=jnp.where(x[i]>0,i,j)
                dst=jnp.where(x[i]>0,j,i)
                x_prop=x.at[src].set(0).at[dst].set(1)
                logp_prop=2.0*jnp.real(apply_fun(variables,x_prop))
                _,n_active_prop=choose_hoppable(
                    x_prop,jnp.asarray(0.0,dtype=up_i.dtype)
                )
                log_corr=(
                    jnp.log(n_active.astype(jnp.float64))
                    -jnp.log(n_active_prop.astype(jnp.float64))
                )
                log_wave_ratio=logp_prop-logp
                accept=jnp.log(
                    jnp.maximum(
                        ua_i,jnp.asarray(1e-300,dtype=ua_i.dtype)
                    )
                )<(log_wave_ratio+log_corr)
                x_new=jnp.where(accept,x_prop,x)
                lp_new=jnp.where(accept,logp_prop,logp)
                na_new=naccepted+accept.astype(jnp.int32)
                return (
                    x_new,lp_new,na_new
                ),(edge,accept,log_wave_ratio,log_corr)

            carry,diag=jax.lax.scan(
                step,
                (x,logp,jnp.asarray(0,jnp.int32)),
                (u_prop,u_acc),
            )
            x,logp,naccepted=carry
            return x,logp-logp0,naccepted,diag

        return jax.vmap(one)(
            configurations,proposal_uniforms,accept_uniforms
        )

    direct_kernel=jax.jit(direct_kernel)

    def fast_kernel(parameters,configurations,proposal_uniforms,accept_uniforms):
        cache=init_cache(parameters,configurations)
        cache,logp_delta,naccepted,diag=advance(
            cache,proposal_uniforms,accept_uniforms
        )
        inv_res=inverse_residual(cache)
        return cache["x"],logp_delta,naccepted,inv_res,diag

    fast_kernel=jax.jit(fast_kernel)

    roots_j=jnp.asarray(roots)
    up_j=jnp.asarray(up)
    ua_j=jnp.asarray(ua)

    # JIT warmup.
    fast=fast_kernel(vstate.parameters,roots_j,up_j,ua_j)
    direct=direct_kernel(vstate.parameters,roots_j,up_j,ua_j)
    np.asarray(fast[0]); np.asarray(direct[0])

    fast_times=[]
    direct_times=[]
    for _ in range(max(1,int(repeats))):
        t=time.perf_counter()
        out=fast_kernel(vstate.parameters,roots_j,up_j,ua_j)
        np.asarray(out[0])
        fast_times.append(time.perf_counter()-t)
        fast=out

        t=time.perf_counter()
        out=direct_kernel(vstate.parameters,roots_j,up_j,ua_j)
        np.asarray(out[0])
        direct_times.append(time.perf_counter()-t)
        direct=out

    fast_x=np.asarray(fast[0])
    fast_lp=np.asarray(fast[1])
    fast_na=np.asarray(fast[2])
    inv_res=np.asarray(fast[3])
    fast_diag=fast[4]

    direct_x=np.asarray(direct[0])
    direct_lp=np.asarray(direct[1])
    direct_na=np.asarray(direct[2])
    direct_diag=direct[3]

    config_mismatch=int(np.sum(np.any(fast_x!=direct_x,axis=1)))
    accept_fast=np.asarray(fast_diag[1],bool)
    accept_direct=np.asarray(direct_diag[1],bool)
    accept_mismatch=int(np.sum(accept_fast!=accept_direct))
    logratio_err=float(
        np.max(
            np.abs(
                np.asarray(fast_diag[2])
                -np.asarray(direct_diag[2])
            )
        )
    )
    logp_err=float(np.max(np.abs(fast_lp-direct_lp)))
    accepted_count_err=int(np.max(np.abs(fast_na-direct_na)))
    max_inv_res=float(np.max(inv_res))

    report={
        "n_roots":int(n_roots),
        "sweep_size":int(sweep_size),
        "n_sweeps":int(n_sweeps),
        "n_proposals_per_chain":int(n_steps),
        "config_mismatch_chains":config_mismatch,
        "accept_mismatch_steps":accept_mismatch,
        "accepted_count_max_error":accepted_count_err,
        "log_wave_ratio_max_abs_error":logratio_err,
        "log_probability_delta_max_abs_error":logp_err,
        "cached_inverse_max_residual":max_inv_res,
        "fast_median_s":float(np.median(fast_times)),
        "direct_median_s":float(np.median(direct_times)),
    }
    report["speedup"]=float(
        report["direct_median_s"]
        /max(1e-15,report["fast_median_s"])
    )
    print(
        "[fast sampler check] "
        f"roots={n_roots} sweeps={n_sweeps} "
        f"proposals={n_steps} "
        f"config_mismatch={config_mismatch} "
        f"accept_mismatch={accept_mismatch} "
        f"logratio_abs={logratio_err:.3e} "
        f"inv_res={max_inv_res:.3e} "
        f"speedup={report['speedup']:.2f}x "
        f"({report['direct_median_s']:.4f}s->"
        f"{report['fast_median_s']:.4f}s)",
        flush=True,
    )
    return report


def _stage(
    *,
    stage_name,
    directory,
    g,nf,V,
    hi,H_fast,graph,current_ops,
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
    learning_rate_after=None,
    learning_rate_switch_iter=None,
    train_samples=None,
    train_samples_after=None,
    train_samples_switch_iter=None,
    train_schedule=None,
    train_sweep_size=None,
    eval_samples_override=None,
    make_maps=False,
    map_edge_chunk=12,
    measure_orders=True,
    return_vstate=False,
):
    import jax
    jax.config.update("jax_enable_x64",True)
    import netket as nk

    packed_requested=bool(getattr(args,"packed_kernel",False))
    packed_effective=False
    packed_error=None
    try:
        setattr(
            nk.config,
            "netket_experimental_packed_kernel",
            packed_requested,
        )
        packed_effective=bool(
            getattr(
                nk.config,
                "netket_experimental_packed_kernel",
            )
        )
    except Exception as exc:
        packed_error=str(exc)
        packed_effective=False
        if packed_requested:
            print(
                "[packed kernel] requested but unavailable in this NetKet "
                f"version: {exc}; using the standard padded kernel.",
                flush=True,
            )

    blocks=(
        seed_data["blocks"]
        if initial_params is None else initial_params["blocks"]
    )
    v0=None if initial_params is None else initial_params.get("v",None)
    eta0=None if initial_params is None else initial_params.get("eta",None)
    if use_backflow and eta0 is not None and bf_mats is not None:
        eta0=np.asarray(eta0,np.complex128).reshape(-1)
        nbf=int(np.asarray(bf_mats).shape[0])
        # Backward compatibility with pre-A/B backflow checkpoints:
        # old eta=[eta_t,eta_t1,eta_t2] -> [eta_tA,eta_tB,eta_t1,eta_t2].
        if eta0.size==3 and nbf==4:
            eta0=np.asarray(
                [eta0[0],eta0[0],eta0[1],eta0[2]],np.complex128
            )
            print(
                "[backflow] expanded legacy 3-channel eta checkpoint to "
                "A/B-resolved 4-channel form.",
                flush=True,
            )
        elif eta0.size!=nbf:
            raise ValueError(
                f"Backflow eta has {eta0.size} channels but model expects {nbf}"
            )
    tr0=(
        0.0 if initial_params is None
        else float(initial_params.get("tr_phase",0.0) or 0.0)
    )

    orbital_parameterization=str(
        getattr(args,"orbital_parameterization","qr")
    ).lower()
    if orbital_parameterization not in ("qr","free"):
        raise ValueError(
            f"Unknown orbital parameterization={orbital_parameterization}"
        )
    orbital_free_effective=(
        orbital_parameterization=="free"
        and maps is None
        and not bool(time_reversal_even)
    )
    orbital_orthonormalize=not orbital_free_effective
    analytic_logderiv_requested=bool(
        getattr(args,"analytic_logderiv",False)
    )
    analytic_logderiv_effective=(
        analytic_logderiv_requested
        and orbital_free_effective
        and use_backflow
    )
    analytic_logderiv_report={}
    analytic_logderiv_error=None

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
        orthonormalize_orbitals=orbital_orthonormalize,
        analytic_logderiv=analytic_logderiv_effective,
    )

    normalized_train_schedule=None
    if train_schedule is not None and int(n_iter)>0:
        normalized_train_schedule=[]
        prev_end=0
        for item in train_schedule:
            end_iter=int(item["end_iter"])
            if end_iter<=prev_end:
                raise ValueError("train_schedule end_iter values must increase")
            if end_iter>int(n_iter):
                end_iter=int(n_iter)
            if end_iter<=prev_end:
                continue
            entry={
                "end_iter":end_iter,
                "learning_rate":float(item["learning_rate"]),
                "n_samples":int(item["n_samples"]),
                "diag_shift":float(item["diag_shift"]),
            }
            if entry["n_samples"]<1:
                raise ValueError("train_schedule n_samples must be positive")
            if entry["learning_rate"]<=0:
                raise ValueError("train_schedule learning_rate must be positive")
            if entry["diag_shift"]<0:
                raise ValueError("train_schedule diag_shift must be nonnegative")
            normalized_train_schedule.append(entry)
            prev_end=end_iter
            if end_iter==int(n_iter):
                break
        if not normalized_train_schedule:
            raise ValueError("train_schedule contains no usable stages")
        if normalized_train_schedule[-1]["end_iter"]<int(n_iter):
            tail=dict(normalized_train_schedule[-1])
            tail["end_iter"]=int(n_iter)
            normalized_train_schedule.append(tail)

    if normalized_train_schedule is not None:
        stage_samples=int(normalized_train_schedule[0]["n_samples"])
        stage_samples_after=int(normalized_train_schedule[-1]["n_samples"])
        sample_switch=(
            int(normalized_train_schedule[0]["end_iter"])
            if len(normalized_train_schedule)>1 else None
        )
        use_sample_schedule=(
            any(
                x["n_samples"]!=normalized_train_schedule[0]["n_samples"]
                for x in normalized_train_schedule[1:]
            )
        )
    else:
        stage_samples=(
            int(train_samples)
            if train_samples is not None
            else int(args.n_samples)
        )
        stage_samples_after=(
            int(train_samples_after)
            if train_samples_after is not None
            else int(stage_samples)
        )
        sample_switch=(
            None if train_samples_switch_iter is None
            else int(train_samples_switch_iter)
        )
        use_sample_schedule=(
            int(n_iter)>0
            and stage_samples_after!=stage_samples
            and sample_switch is not None
            and 0<sample_switch<int(n_iter)
        )
    sweep_size=(
        int(train_sweep_size)
        if train_sweep_size is not None
        else (
            int(args.sweep_size)
            if args.sweep_size is not None
            else int(nf)
        )
    )
    sampler_kwargs={
        "graph":graph,
        "n_chains":args.n_chains,
        "sweep_size":sweep_size,
    }
    if args.sampler_chunk_size is not None:
        sampler_kwargs["chunk_size"]=int(args.sampler_chunk_size)

    fast_sampler_requested=bool(getattr(args,"fast_sampler",False))
    fast_sampler_eligible=(
        fast_sampler_requested
        and maps is None
        and not bool(time_reversal_even)
        and use_backflow
    )
    fast_sampler_effective=False
    fast_sampler_error=None
    if fast_sampler_eligible:
        try:
            sampler=_make_fast_raw_sampler(
                nk,jax,
                hilbert=hi,
                clusters=[
                    (int(e.i),int(e.j)) for e in g.edges
                ],
                seed_data=seed_data,
                shell_mats=shell_mats,
                bf_mats=bf_mats,
                n_chains=args.n_chains,
                sweep_size=sweep_size,
                chunk_size=args.sampler_chunk_size,
                orthonormalize_orbitals=orbital_orthonormalize,
            )
            fast_sampler_effective=True
        except Exception as exc:
            fast_sampler_error=str(exc)
            if bool(getattr(args,"fast_sampler_strict",False)):
                raise
            print(
                "[fast sampler] setup failed: "
                f"{exc}; using NetKet MetropolisFermionHop.",
                flush=True,
            )
            sampler=nk.sampler.MetropolisFermionHop(
                hi,**sampler_kwargs
            )
    else:
        sampler=nk.sampler.MetropolisFermionHop(hi,**sampler_kwargs)
        if fast_sampler_requested and not fast_sampler_eligible:
            reason=(
                "projection/TR stage"
                if maps is not None or bool(time_reversal_even)
                else "backflow disabled"
            )
            print(
                f"[fast sampler] not eligible for {stage_name}: {reason}; "
                "using NetKet MetropolisFermionHop.",
                flush=True,
            )

    vstate=nk.vqs.MCState(
        sampler,model,
        n_samples=stage_samples,
        n_discard_per_chain=args.n_discard,
        chunk_size=args.chunk_size,
        seed=rng_seed,
        sampler_seed=rng_seed+104729,
    )

    if analytic_logderiv_effective:
        key=(
            int(g.Ns),
            int(nf),
            tuple(
                int(x)
                for x in np.asarray(
                    seed_data["occupations_per_k"],int
                )
            ),
        )
        need_check=(
            bool(getattr(args,"analytic_logderiv_check",True))
            and key not in _ANALYTIC_LOGDERIV_VALIDATED_KEYS
        )
        if need_check:
            ref_model=build_supercell_netket_model(
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
                orthonormalize_orbitals=False,
                analytic_logderiv=False,
            )
            ref_vstate=nk.vqs.MCState(
                sampler,ref_model,
                n_samples=stage_samples,
                n_discard_per_chain=args.n_discard,
                chunk_size=args.chunk_size,
                seed=rng_seed+17,
                sampler_seed=rng_seed+104746,
            )
            try:
                analytic_logderiv_report=_validate_analytic_logderiv(
                    vstate_analytic=vstate,
                    vstate_reference=ref_vstate,
                    H_reference=H_fast,
                    jax=jax,
                    timing_repeats=max(
                        1,int(getattr(args,"profile_vmc_repeats",3))
                    ),
                )
                tol=float(
                    getattr(
                        args,
                        "analytic_logderiv_tolerance",
                        1e-8,
                    )
                )
                passed=(
                    analytic_logderiv_report["energy_abs_error"]<=tol
                    and analytic_logderiv_report["max_abs_error"]<=tol
                    and analytic_logderiv_report["relative_l2_error"]<=tol
                )
                analytic_logderiv_report["tolerance"]=tol
                analytic_logderiv_report["passed"]=bool(passed)
                min_speedup=float(
                    getattr(args,"analytic_logderiv_min_speedup",1.02)
                )
                perf_ok=(
                    analytic_logderiv_report.get(
                        "expect_and_grad_speedup",1.0
                    )>=min_speedup
                )
                analytic_logderiv_report["min_speedup"]=min_speedup
                analytic_logderiv_report["performance_passed"]=bool(perf_ok)

                if passed and perf_ok:
                    _ANALYTIC_LOGDERIV_VALIDATED_KEYS.add(key)
                elif passed and not perf_ok:
                    analytic_logderiv_error=(
                        "analytic log-derivative is correct but not faster: "
                        f"speedup={analytic_logderiv_report.get('expect_and_grad_speedup',1.0):.3f}x "
                        f"< required {min_speedup:.3f}x"
                    )
                    print(
                        "[analytic logderiv] "
                        +analytic_logderiv_error
                        +"; using original autodiff model.",
                        flush=True,
                    )
                    _fallback_samples=vstate.samples
                    ref_vstate.parameters=vstate.parameters
                    ref_vstate._samples=_fallback_samples
                    vstate=ref_vstate
                    model=ref_model
                    analytic_logderiv_effective=False
                else:
                    analytic_logderiv_error=(
                        "analytic log-derivative validation failed: "
                        f"{analytic_logderiv_report}"
                    )
                    print(
                        "[analytic logderiv] "
                        +analytic_logderiv_error
                        +"; using original autodiff model.",
                        flush=True,
                    )
                    # Preserve exactly the parameters/samples used by the
                    # failed check while falling back to the reference model.
                    _fallback_samples=vstate.samples
                    ref_vstate.parameters=vstate.parameters
                    ref_vstate._samples=_fallback_samples
                    vstate=ref_vstate
                    model=ref_model
                    analytic_logderiv_effective=False
            except Exception as exc:
                analytic_logderiv_error=str(exc)
                print(
                    "[analytic logderiv] validation raised "
                    f"{exc}; using original autodiff model.",
                    flush=True,
                )
                _fallback_samples=vstate.samples
                ref_vstate.parameters=vstate.parameters
                ref_vstate._samples=_fallback_samples
                vstate=ref_vstate
                model=ref_model
                analytic_logderiv_effective=False

    fast_local_gradient_check={}
    vmc_profile={}
    fast_local_check={}
    train_segment_timings=[]
    shared_sr_check={}
    shared_sr_profile={}
    fast_sampler_check={}

    fast_local_requested=bool(getattr(args,"fast_local_energy",False))
    fast_local_eligible=(
        fast_local_requested
        and maps is None
        and not bool(time_reversal_even)
        and use_backflow
    )
    fast_local_effective=False
    fast_local_error=None
    fast_map_kernel=None
    H_stage=H_fast
    if fast_local_eligible:
        try:
            edge_i=np.asarray([int(e.i) for e in g.edges],np.int32)
            edge_j=np.asarray([int(e.j) for e in g.edges],np.int32)
            edge_t=np.asarray([float(g.t[e.kind]) for e in g.edges],float)
            edge_v=np.asarray(
                [1.0 if e.kind=="t" else 0.0 for e in g.edges],
                float,
            )
            fast_kernel=make_fast_raw_local_energy_kernel(
                sc_coords=seed_data["sc_coords"],
                internal_sites=seed_data["internal_sites"],
                kpoints=seed_data["kpoints"],
                occupations_per_k=seed_data["occupations_per_k"],
                shell_matrices=shell_mats,
                backflow_matrices=bf_mats,
                edge_i=edge_i,
                edge_j=edge_j,
                edge_t=edge_t,
                edge_has_interaction=edge_v,
                interaction_V=V,
                return_diagnostics=False,
                orthonormalize_orbitals=orbital_orthonormalize,
            )
            if make_maps or measure_orders:
                # Same exact rank-3 machinery, but retain per-edge hopping
                # ratios needed by bond/current maps and LC observables.
                fast_map_kernel=make_fast_raw_local_energy_kernel(
                    sc_coords=seed_data["sc_coords"],
                    internal_sites=seed_data["internal_sites"],
                    kpoints=seed_data["kpoints"],
                    occupations_per_k=seed_data["occupations_per_k"],
                    shell_matrices=shell_mats,
                    backflow_matrices=bf_mats,
                    edge_i=edge_i,
                    edge_j=edge_j,
                    edge_t=edge_t,
                    edge_has_interaction=edge_v,
                    interaction_V=V,
                    return_diagnostics=True,
                    orthonormalize_orbitals=orbital_orthonormalize,
                )
            FastRawHamiltonian=_fast_raw_operator_class(nk)
            H_stage=FastRawHamiltonian(hi,fast_kernel)
            fast_local_effective=True
        except Exception as exc:
            fast_local_error=str(exc)
            H_stage=H_fast
            if bool(getattr(args,"fast_local_strict",False)):
                raise
            print(
                "[fast local] requested but unavailable: "
                f"{exc}; falling back to NetKet operator local energy.",
                flush=True,
            )
    elif fast_local_requested and not fast_local_eligible:
        reason=(
            "projection/TR stage"
            if maps is not None or bool(time_reversal_even)
            else "backflow disabled"
        )
        print(
            f"[fast local] not eligible for {stage_name}: {reason}; "
            "using NetKet operator local energy.",
            flush=True,
        )

    if fast_local_effective:
        global _FAST_LOCAL_GRADIENT_VALIDATED
        need_grad_check=(
            bool(getattr(args,"fast_local_gradient_check",True))
            and not _FAST_LOCAL_GRADIENT_VALIDATED
        )
        if need_grad_check:
            try:
                fast_local_gradient_check=_validate_fast_raw_gradient(
                    vstate=vstate,
                    H_reference=H_fast,
                    H_fast_custom=H_stage,
                    jax=jax,
                )
                grad_tol=float(
                    getattr(args,"fast_local_gradient_tolerance",1e-8)
                )
                passed=(
                    fast_local_gradient_check["energy_abs_error"]<=grad_tol
                    and fast_local_gradient_check["max_abs_error"]<=grad_tol
                    and fast_local_gradient_check["relative_l2_error"]<=grad_tol
                )
                fast_local_gradient_check["tolerance"]=grad_tol
                fast_local_gradient_check["passed"]=bool(passed)
                if not passed:
                    msg=(
                        "Fast local-energy gradient validation failed: "
                        f"{fast_local_gradient_check}"
                    )
                    if bool(getattr(args,"fast_local_strict",False)):
                        raise RuntimeError(msg)
                    print(
                        "[fast local] "+msg+"; falling back to NetKet operator.",
                        flush=True,
                    )
                    H_stage=H_fast
                    fast_local_effective=False
                    fast_map_kernel=None
                    fast_local_error=msg
                else:
                    _FAST_LOCAL_GRADIENT_VALIDATED=True
            except Exception as exc:
                if bool(getattr(args,"fast_local_strict",False)):
                    raise
                fast_local_error=f"gradient validation error: {exc}"
                H_stage=H_fast
                fast_local_effective=False
                fast_map_kernel=None
                print(
                    "[fast local] gradient validation raised "
                    f"{exc}; falling back to NetKet operator.",
                    flush=True,
                )

    setup_s=time.perf_counter()-t0

    warm=None
    warmup_s=0.0
    solver_restart_effective=None

    effective_qgt="not_used_pav_only"
    log=nk.logging.RuntimeLog()
    if int(n_iter)>0:
        tw=time.perf_counter()
        warm=vstate.expect(H_stage)
        warmup_s=time.perf_counter()-tw
        qgt_ctor,solver,effective_qgt=_qgt_and_solver(nk,jax,args)
        if str(args.qgt).lower()=="onthefly" and effective_qgt!="onthefly":
            print(
                "[QGT] requested onthefly, but this ansatz has mixed real/complex "
                "parameters (real Jastrow v, complex orbitals/backflow). "
                "Falling back to QGTJacobianPyTree(mode='complex') + iterative "
                "solver; no dense P x P QGT is formed.",
                flush=True,
            )
        import optax
        if normalized_train_schedule is not None:
            lr_values=[x["learning_rate"] for x in normalized_train_schedule]
            diag_values=[x["diag_shift"] for x in normalized_train_schedule]
            boundaries=[
                int(x["end_iter"]) for x in normalized_train_schedule[:-1]
            ]

            lr0=float(lr_values[0])
            lr_scales={}
            for ib,boundary in enumerate(boundaries):
                prev=float(lr_values[ib])
                nxt=float(lr_values[ib+1])
                lr_scales[int(boundary)]=nxt/prev
            lr_schedule=optax.piecewise_constant_schedule(
                init_value=lr0,
                boundaries_and_scales=lr_scales,
            )
            optimizer=optax.sgd(learning_rate=lr_schedule)

            diag0=float(diag_values[0])
            if diag0==0.0 and any(abs(x)>0 for x in diag_values[1:]):
                raise ValueError(
                    "A scheduled diag_shift cannot start at zero and later "
                    "become nonzero with the multiplicative Optax schedule"
                )
            if diag0==0.0:
                diag_shift_for_driver=0.0
            else:
                diag_scales={}
                for ib,boundary in enumerate(boundaries):
                    prev=float(diag_values[ib])
                    nxt=float(diag_values[ib+1])
                    if prev==0.0:
                        raise ValueError(
                            "diag_shift schedule cannot transition out of zero"
                        )
                    diag_scales[int(boundary)]=nxt/prev
                diag_shift_for_driver=optax.piecewise_constant_schedule(
                    init_value=diag0,
                    boundaries_and_scales=diag_scales,
                )

            def _desc(key):
                start=1
                parts=[]
                for item in normalized_train_schedule:
                    end=int(item["end_iter"])
                    parts.append(f"{item[key]:g}@{start}-{end}")
                    start=end+1
                return ", ".join(parts)
            lr_description=_desc("learning_rate")
            diag_description=_desc("diag_shift")
        else:
            lr0=float(learning_rate)
            lr1=(
                None if learning_rate_after is None
                else float(learning_rate_after)
            )
            lr_switch=(
                None if learning_rate_switch_iter is None
                else int(learning_rate_switch_iter)
            )
            use_lr_schedule=(
                lr1 is not None
                and lr_switch is not None
                and 0<lr_switch<int(n_iter)
                and not np.isclose(lr1,lr0)
            )
            if use_lr_schedule:
                if lr0==0.0:
                    raise ValueError("Initial learning rate must be nonzero for a two-stage schedule")
                schedule=optax.piecewise_constant_schedule(
                    init_value=lr0,
                    boundaries_and_scales={lr_switch:lr1/lr0},
                )
                optimizer=optax.sgd(learning_rate=schedule)
                lr_description=(
                    f"{lr0:g} for steps 1-{lr_switch}, "
                    f"then {lr1:g} for steps {lr_switch+1}-{int(n_iter)}"
                )
            else:
                optimizer=optax.sgd(learning_rate=lr0)
                lr_description=f"constant {lr0:g}"
            diag_shift_for_driver=float(args.diag_shift)
            diag_description=f"constant {float(args.diag_shift):g}"

        trainer_requested=str(getattr(args,"trainer","legacy")).lower()
        if trainer_requested not in ("auto","vmc_sr","shared_sr","legacy"):
            raise ValueError(f"Unknown trainer={trainer_requested}")
        trainer_effective=trainer_requested
        sr_momentum=float(getattr(args,"sr_momentum",0.8))
        if not 0.0<=sr_momentum<1.0:
            raise ValueError("--sr-momentum must satisfy 0 <= momentum < 1")

        driver=None
        vmc_sr_error=None
        shared_sr_error=None

        if trainer_requested in ("auto","vmc_sr"):
            try:
                linear_solver=getattr(
                    nk.optimizer.solver,
                    "cholesky_with_fallback",
                    nk.optimizer.solver.cholesky,
                )
                driver=nk.driver.VMC_SR(
                    H_stage,
                    optimizer,
                    variational_state=vstate,
                    diag_shift=diag_shift_for_driver,
                    momentum=(sr_momentum if sr_momentum>0 else None),
                    linear_solver=linear_solver,
                    mode="complex",
                    use_ntk=False,
                )
                trainer_effective="vmc_sr"
                effective_qgt="standard_parameter_space_sr"
            except Exception as exc:
                vmc_sr_error=exc
                if trainer_requested=="vmc_sr":
                    raise
                driver=None

        if driver is None and trainer_requested in ("auto","shared_sr"):
            try:
                key=(
                    int(g.Ns),int(nf),
                    tuple(
                        int(x) for x in np.asarray(
                            seed_data["occupations_per_k"],int
                        )
                    ),
                )
                initial_diag=(
                    float(normalized_train_schedule[0]["diag_shift"])
                    if normalized_train_schedule is not None
                    else float(args.diag_shift)
                )
                need_shared_check=(
                    bool(getattr(args,"shared_sr_check",True))
                    and key not in _SHARED_SR_VALIDATED_KEYS
                )
                if need_shared_check:
                    shared_sr_check=_validate_shared_sr(
                        vstate=vstate,
                        H=H_stage,
                        nk=nk,
                        jax=jax,
                        qgt_ctor=qgt_ctor,
                        solver=solver,
                        diag_shift=initial_diag,
                        chunk_size=args.qgt_chunk_size,
                        repeats=max(
                            1,int(getattr(args,"profile_vmc_repeats",3))
                        ),
                    )
                    tol=float(getattr(args,"shared_sr_tolerance",1e-8))
                    passed=(
                        shared_sr_check["energy_abs_error"]<=tol
                        and shared_sr_check["max_abs_error"]<=tol
                        and shared_sr_check["relative_l2_error"]<=tol
                    )
                    shared_sr_check["tolerance"]=tol
                    shared_sr_check["passed"]=bool(passed)
                    min_speedup=float(
                        getattr(args,"shared_sr_min_speedup",1.02)
                    )
                    perf_ok=shared_sr_check.get("speedup",1.0)>=min_speedup
                    shared_sr_check["min_speedup"]=min_speedup
                    shared_sr_check["performance_passed"]=bool(perf_ok)
                    if not passed:
                        raise RuntimeError(
                            "shared-SR validation failed: "
                            f"{shared_sr_check}"
                        )
                    if not perf_ok:
                        raise RuntimeError(
                            "shared-SR is correct but not faster: "
                            f"{shared_sr_check.get('speedup',1.0):.3f}x "
                            f"< {min_speedup:.3f}x"
                        )
                    _SHARED_SR_VALIDATED_KEYS.add(key)

                driver=_make_shared_sr_driver(
                    nk,jax,
                    H=H_stage,
                    optimizer=optimizer,
                    vstate=vstate,
                    diag_shift=diag_shift_for_driver,
                    solver=solver,
                    chunk_size=args.qgt_chunk_size,
                )
                trainer_effective="shared_sr"
                effective_qgt="shared_dense_jacobian_matrix_free_sr"
                solver_restart_effective=True
            except Exception as exc:
                shared_sr_error=exc
                if trainer_requested=="shared_sr" and bool(
                    getattr(args,"shared_sr_strict",False)
                ):
                    raise
                driver=None
                if trainer_requested=="shared_sr":
                    print(
                        f"[trainer fallback] shared_sr unavailable/failed: "
                        f"{exc}; using legacy QGT+iterative SR.",
                        flush=True,
                    )

        if driver is None:
            trainer_effective="legacy"
            # Keep CG restart for ordinary steps.  The orbital gauge-fix
            # callback clears SR.x0 only when the parameter gauge is actually
            # changed, so we retain restart speed without carrying an old-gauge
            # solution across a QR reparameterisation.
            solver_restart_effective=True
            try:
                sr=nk.optimizer.SR(
                    qgt=qgt_ctor,
                    solver=solver,
                    diag_shift=diag_shift_for_driver,
                    solver_restart=solver_restart_effective,
                )
            except TypeError:
                sr=nk.optimizer.SR(
                    qgt=qgt_ctor,
                    solver=solver,
                    diag_shift=diag_shift_for_driver,
                )
            driver=nk.driver.VMC(
                H_stage,optimizer,
                variational_state=vstate,
                preconditioner=sr,
            )
            if vmc_sr_error is not None:
                print(
                    f"[trainer fallback] VMC_SR unavailable/failed: {vmc_sr_error}; "
                    "using legacy QGT+iterative SR.",
                    flush=True,
                )

        profile_repeats=int(getattr(args,"profile_vmc_repeats",0))
        profile_seed=str(getattr(args,"profile_seed","normal"))
        if (
            profile_repeats>0
            and str(args._active_seed)==profile_seed
            and trainer_effective=="legacy"
        ):
            profile_diag=(
                float(normalized_train_schedule[0]["diag_shift"])
                if normalized_train_schedule is not None
                else float(args.diag_shift)
            )
            vmc_profile=_profile_vmc_components(
                vstate=vstate,H_fast=H_stage,jax=jax,nk=nk,
                qgt_ctor=(qgt_ctor if trainer_effective=="legacy" else None),
                solver=(solver if trainer_effective=="legacy" else None),
                diag_shift=profile_diag,
                repeats=profile_repeats,
                H_reference=(H_fast if fast_local_effective else None),
            )

        if (
            profile_repeats>0
            and str(args._active_seed)==profile_seed
            and trainer_effective=="shared_sr"
        ):
            profile_diag=(
                float(normalized_train_schedule[0]["diag_shift"])
                if normalized_train_schedule is not None
                else float(args.diag_shift)
            )
            shared_sr_profile=_profile_shared_sr_components(
                vstate=vstate,H=H_stage,nk=nk,jax=jax,
                solver=solver,
                diag_shift=profile_diag,
                chunk_size=args.qgt_chunk_size,
                repeats=profile_repeats,
            )

        gauge_fix_callback=(
            _make_orbital_gauge_fix_callback(
                nk,
                seed_data["occupations_per_k"],
                int(getattr(args,"orbital_gauge_fix_interval",20)),
            )
            if orbital_free_effective else None
        )

        n_images=1 if maps is None else int(len(maps))
        if normalized_train_schedule is not None:
            start=1
            sample_parts=[]
            for item in normalized_train_schedule:
                end=int(item["end_iter"])
                sample_parts.append(f"{int(item['n_samples'])}@{start}-{end}")
                start=end+1
            sample_description=", ".join(sample_parts)
        elif use_sample_schedule:
            sample_description=(
                f"{stage_samples} for steps 1-{sample_switch}, "
                f"then {stage_samples_after} for steps "
                f"{sample_switch+1}-{int(n_iter)}"
            )
        else:
            sample_description=f"constant {stage_samples}"
        print(
            f"[large {stage_name}] seed={args._active_seed} "
            f"Ns={g.Ns} Nf={nf} "
            f"params={vstate.n_parameters} images={n_images} "
            f"samples={sample_description} sweep={sweep_size} "
            f"packed_kernel={packed_requested}->{packed_effective} "
            f"fast_local={fast_local_requested}->{fast_local_effective} "
            f"fast_sampler={fast_sampler_requested}->{fast_sampler_effective} "
            f"orbitals={orbital_parameterization}->"
            f"{'free' if orbital_free_effective else 'qr'} "
            f"analytic_logderiv={analytic_logderiv_requested}->"
            f"{analytic_logderiv_effective} "
            f"trainer={trainer_requested}->{trainer_effective} "
            f"sr_momentum={(sr_momentum if trainer_effective=='vmc_sr' else 0.0):g} "
            f"lr={lr_description} diag_shift={diag_description}",
            flush=True,
        )
        topt=time.perf_counter()
        if normalized_train_schedule is not None:
            prev_end=0
            for item in normalized_train_schedule:
                end=int(item["end_iter"])
                nsamp=int(item["n_samples"])
                if int(vstate.n_samples)!=nsamp:
                    vstate.n_samples=nsamp
                seg_iter=end-prev_end
                ts=time.perf_counter()
                driver.run(
                    n_iter=seg_iter,
                    out=log,show_progress=True,
                    callback=gauge_fix_callback,
                )
                seg_s=time.perf_counter()-ts
                train_segment_timings.append({
                    "start_iter":int(prev_end+1),
                    "end_iter":int(end),
                    "n_iter":int(seg_iter),
                    "n_samples":int(nsamp),
                    "learning_rate":float(item["learning_rate"]),
                    "diag_shift":float(item["diag_shift"]),
                    "time_s":float(seg_s),
                    "time_per_iter_s":float(seg_s/max(1,seg_iter)),
                })
                prev_end=end
        elif use_sample_schedule:
            ts=time.perf_counter()
            driver.run(
                n_iter=sample_switch,
                out=log,show_progress=True,
                    callback=gauge_fix_callback,
            )
            seg_s=time.perf_counter()-ts
            train_segment_timings.append({
                "start_iter":1,
                "end_iter":int(sample_switch),
                "n_iter":int(sample_switch),
                "n_samples":int(stage_samples),
                "time_s":float(seg_s),
                "time_per_iter_s":float(seg_s/max(1,sample_switch)),
            })
            vstate.n_samples=stage_samples_after
            seg_iter=int(n_iter)-sample_switch
            ts=time.perf_counter()
            driver.run(
                n_iter=seg_iter,
                out=log,show_progress=True,
                    callback=gauge_fix_callback,
            )
            seg_s=time.perf_counter()-ts
            train_segment_timings.append({
                "start_iter":int(sample_switch+1),
                "end_iter":int(n_iter),
                "n_iter":int(seg_iter),
                "n_samples":int(stage_samples_after),
                "time_s":float(seg_s),
                "time_per_iter_s":float(seg_s/max(1,seg_iter)),
            })
        else:
            ts=time.perf_counter()
            driver.run(
                n_iter=n_iter,out=log,show_progress=True,
                callback=gauge_fix_callback,
            )
            seg_s=time.perf_counter()-ts
            train_segment_timings.append({
                "start_iter":1,
                "end_iter":int(n_iter),
                "n_iter":int(n_iter),
                "n_samples":int(stage_samples),
                "time_s":float(seg_s),
                "time_per_iter_s":float(seg_s/max(1,int(n_iter))),
            })
        opt_s=time.perf_counter()-topt
    else:
        n_images=1 if maps is None else int(len(maps))
        requested_eval=(
            int(eval_samples_override)
            if eval_samples_override is not None
            else (
                int(args.eval_samples)
                if args.eval_samples is not None
                else int(args.n_samples)
            )
        )
        print(
            f"[large {stage_name} eval-only] seed={args._active_seed} "
            f"Ns={g.Ns} Nf={nf} "
            f"params={vstate.n_parameters} images={n_images} "
            f"eval_samples={requested_eval} sweep={sweep_size} "
            f"packed_kernel={packed_requested}->{packed_effective} "
            f"fast_local={fast_local_requested}->{fast_local_effective} "
            f"fast_sampler={fast_sampler_requested}->{fast_sampler_effective} "
            f"analytic_logderiv={analytic_logderiv_requested}->"
            f"{analytic_logderiv_effective}; "
            "no SR/Jacobian and no duplicate warmup energy evaluation",
            flush=True,
        )
        opt_s=0.0

    eval_samples=(
        int(eval_samples_override)
        if eval_samples_override is not None
        else (
            int(args.eval_samples)
            if args.eval_samples is not None
            else int(args.n_samples)
        )
    )
    if int(vstate.n_samples)!=eval_samples:
        vstate.n_samples=eval_samples
    te=time.perf_counter()
    vstate.sample()
    final_stats=vstate.expect(H_stage)
    final_eval_s=time.perf_counter()-te

    if (
        bool(getattr(args,"fast_local_check",False))
        and maps is None
        and use_backflow
    ):
        fast_local_check=_validate_fast_raw_local_energy(
            vstate=vstate,g=g,V=V,seed_data=seed_data,
            shell_mats=shell_mats,bf_mats=bf_mats,
            n_roots=int(getattr(args,"fast_local_check_roots",8)),
            repeats=int(getattr(args,"fast_local_check_repeats",3)),
            orthonormalize_orbitals=orbital_orthonormalize,
        )

    if (
        bool(getattr(args,"fast_sampler_check",False))
        and maps is None
        and use_backflow
    ):
        fast_sampler_check=_validate_fast_raw_sampler_sweep(
            vstate=vstate,g=g,seed_data=seed_data,
            shell_mats=shell_mats,bf_mats=bf_mats,
            sweep_size=sweep_size,
            rng_seed=rng_seed,
            n_roots=int(getattr(args,"fast_sampler_check_roots",8)),
            repeats=int(getattr(args,"fast_sampler_check_repeats",3)),
            n_sweeps=int(getattr(args,"fast_sampler_check_sweeps",40)),
            orthonormalize_orbitals=orbital_orthonormalize,
        )

    fast_hopping_data=None
    if (
        fast_map_kernel is not None
        and (measure_orders or make_maps)
    ):
        fast_hopping_data=_mc_fast_hopping_data(
            vstate,g,fast_map_kernel,
            chunk_size=(
                None if args.chunk_size is None
                else int(args.chunk_size)
            ),
            show_progress=True,
        )

    tobs=time.perf_counter()
    if measure_orders:
        order_obs=_mc_order_observables(
            vstate,g,nf,hi,current_ops,
            fast_hopping_data=fast_hopping_data,
        )
    else:
        order_obs={}
    observable_eval_s=time.perf_counter()-tobs

    params=extract_supercell_parameters(
        vstate.parameters,
        len(seed_data["kpoints"]),
        occupations_per_k=seed_data["occupations_per_k"],
    )

    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    _save_params(directory/"final_params.npz",params,seed_data)
    atomic_json(directory/"history.json",_history(log))

    map_meta=None
    if make_maps:
        map_meta=_save_mc_maps(
            directory,vstate,g,
            title=(
                f"{g.name} | seed={args._active_seed} | "
                f"{stage_name} | irrep={point_label}"
            ),
            edge_chunk=map_edge_chunk,
            log_chunk_size=(
                None if args.chunk_size is None
                else int(args.chunk_size)
            ),
            fast_map_kernel=fast_map_kernel,
            fast_map_chunk_size=(
                None if args.chunk_size is None
                else int(args.chunk_size)
            ),
            fast_hopping_data=fast_hopping_data,
        )
    elif measure_orders:
        # Preserve cheap one-point CO information at scan points where the
        # expensive bond/current spatial map is intentionally skipped.
        map_meta=_mc_density_onepoint_meta(vstate,g)

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
        "orbital_parameterization_requested":orbital_parameterization,
        "orbital_parameterization_effective":(
            "free" if orbital_free_effective else "qr"
        ),
        "orbital_gauge_fix_interval":(
            int(getattr(args,"orbital_gauge_fix_interval",20))
            if orbital_free_effective else None
        ),
        "analytic_logderiv_requested":bool(analytic_logderiv_requested),
        "analytic_logderiv_effective":bool(analytic_logderiv_effective),
        "analytic_logderiv_check":analytic_logderiv_report,
        "analytic_logderiv_error":analytic_logderiv_error,
        "packed_kernel_requested":bool(packed_requested),
        "packed_kernel_effective":bool(packed_effective),
        "packed_kernel_error":packed_error,
        "fast_local_energy_requested":bool(fast_local_requested),
        "fast_local_energy_effective":bool(fast_local_effective),
        "fast_local_energy_error":fast_local_error,
        "fast_local_gradient_check":fast_local_gradient_check,
        "fast_sampler_requested":bool(fast_sampler_requested),
        "fast_sampler_effective":bool(fast_sampler_effective),
        "fast_sampler_error":fast_sampler_error,
        "vmc_component_profile":vmc_profile,
        "shared_sr_check":shared_sr_check,
        "shared_sr_profile":shared_sr_profile,
        "fast_sampler_check":fast_sampler_check,
        "fast_local_check":fast_local_check,
        "train_segment_timings":train_segment_timings,
        "trainer_requested":str(getattr(args,"trainer","legacy")),
        "trainer":(
            trainer_effective if int(n_iter)>0 else "not_used_eval_only"
        ),
        "sr_momentum":(
            float(sr_momentum)
            if int(n_iter)>0 and trainer_effective=="vmc_sr" else 0.0
        ),
        "qgt_requested":args.qgt,
        "qgt":effective_qgt,
        "qgt_mode":"complex",
        "qgt_holomorphic":"implicit_nonholomorphic_from_mode_complex",
        "sr_solver":args.sr_solver,
        "solver_restart_effective":solver_restart_effective,
        "cg_tol":float(args.cg_tol),
        "cg_maxiter":int(args.cg_maxiter),
        "diag_shift":float(args.diag_shift),
        "diag_shift_schedule":(
            diag_description if int(n_iter)>0 else "not_used_eval_only"
        ),
        "train_schedule":(
            normalized_train_schedule if normalized_train_schedule is not None
            else None
        ),
        "n_samples":int(stage_samples),
        "training_n_samples":int(stage_samples),
        "training_n_samples_after":int(stage_samples_after),
        "training_n_samples_switch_iter":(
            int(sample_switch)
            if use_sample_schedule and normalized_train_schedule is None
            else None
        ),
        "training_sample_schedule":(
            sample_description if int(n_iter)>0 else "not_used_eval_only"
        ),
        "evaluation_n_samples":int(eval_samples),
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
        "warmup_E":(
            None if warm is None else float(np.real(warm.mean))
        ),
        "warmup_E_error":(
            None if warm is None else stats_error(warm)
        ),
        "mc_E":float(np.real(final_stats.mean)),
        "mc_E_error":stats_error(final_stats),
        "mc_tau_corr":_scalar_stat(final_stats,"tau_corr"),
        "mc_R_hat":_scalar_stat(final_stats,"R_hat"),
        "sampler_acceptance":acceptance,
        "time_setup_s":float(setup_s),
        "time_jit_warmup_s":float(warmup_s),
        "time_opt_s":float(opt_s),
        "time_per_iteration_s":(
            float(opt_s/int(n_iter)) if int(n_iter)>0 else None
        ),
        "time_final_eval_s":float(final_eval_s),
        "time_order_observables_s":float(observable_eval_s),
        "time_fast_hopping_observables_s":(
            None if fast_hopping_data is None
            else float(fast_hopping_data.get("time_s",0.0))
        ),
        "n_iter":int(n_iter),
        "learning_rate":float(learning_rate),
        "learning_rate_after":(
            float(learning_rate_after)
            if learning_rate_after is not None else None
        ),
        "learning_rate_switch_iter":(
            int(learning_rate_switch_iter)
            if learning_rate_switch_iter is not None else None
        ),
        "learning_rate_schedule":(
            lr_description if int(n_iter)>0 else "not_used_pav_only"
        ),
        "seed_shell_gap":seed_data["shell_gap"],
        "total_supercell_momentum":[
            float(x) for x in seed_data["total_supercell_momentum"]
        ],
        "occupations_per_k":[
            int(x) for x in seed_data["occupations_per_k"]
        ],
        **order_obs,
        "orders_measured":bool(measure_orders),
        "maps_saved":bool(make_maps),
        "map_max_abs_current":(
            None if map_meta is None else map_meta["max_abs_current"]
        ),
        "map_estimator":(
            None if map_meta is None else map_meta.get("estimator")
        ),
        "time_map_estimator_s":(
            None if map_meta is None
            else map_meta.get("time_map_estimator_s")
        ),
        "time_map_export_plot_s":(
            None if map_meta is None
            else map_meta.get("time_map_export_plot_s")
        ),
        "CO_onepoint_rms":(
            None if map_meta is None else map_meta["CO_onepoint_rms"]
        ),
        "CO_onepoint_q1":(
            None if map_meta is None else map_meta["CO_onepoint_q1"]
        ),
        "CO_onepoint_q2":(
            None if map_meta is None else map_meta["CO_onepoint_q2"]
        ),
        "density_rms_modulation":(
            None if map_meta is None else map_meta["density_rms_modulation"]
        ),
        "B_star_mean":(
            None if map_meta is None else map_meta["B_star_mean"]
        ),
        **_memory_estimates(vstate.n_parameters,args.n_samples),
    }
    atomic_json(directory/"summary.json",summary)
    if return_vstate:
        return summary,params,vstate
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
    p.add_argument(
        "--spatial-symmetry",
        choices=["auto","c3v","none"],
        default="auto",
        help=(
            "Residual point-group projection. auto uses C3v only when the "
            "finite torus supports ordinary C3v; otherwise it stops after Tint. "
            "Use c3v to require it or none to skip it explicitly."
        ),
    )
    p.add_argument(
        "--c3v-irrep",
        choices=["A1","A2","E","both","all"],
        default="both",
        help=(
            "Residual C3v sector(s). both=A1+A2; all=A1+A2+E. "
            "Raw and Tint are optimized once and shared by all requested sectors."
        ),
    )
    p.add_argument(
        "--spatial-workflow",
        choices=["screen_then_vap","direct_vap","pav_only"],
        default="screen_then_vap",
        help=(
            "screen_then_vap: PAV-screen requested C3v sectors, then short-VAP "
            "only the lower sector(s). direct_vap reproduces the older behavior. "
            "pav_only performs screening without VAP."
        ),
    )
    p.add_argument(
        "--screen-keep-within-sigma",type=float,default=2.0,
        help=(
            "During PAV screening, also short-VAP sectors whose energy is within "
            "this many combined MC standard errors of the best sector."
        ),
    )

    p.add_argument("--raw-iter",type=int,default=50)
    p.add_argument("--tint-iter",type=int,default=50)
    p.add_argument(
        "--spatial-iter",type=int,default=10,
        help="Short VAP iterations after A1/A2 PAV screening.",
    )
    p.add_argument("--tr-iter",type=int,default=50)
    p.add_argument("--raw-learning-rate",type=float,default=0.03)
    p.add_argument("--tint-learning-rate",type=float,default=0.01)
    p.add_argument("--spatial-learning-rate",type=float,default=0.01)
    p.add_argument("--tr-learning-rate",type=float,default=0.01)
    p.add_argument("--restore-time-reversal",action="store_true")

    p.add_argument("--n-samples",type=int,default=2048)
    p.add_argument(
        "--spatial-samples",type=int,default=512,
        help="Training samples per Tint+C3v/TR iteration; final observables use --eval-samples.",
    )
    p.add_argument(
        "--spatial-sweep-size",type=int,default=None,
        help="Sweep size during 18-image spatial/TR training. Default=max(2,Nf//4).",
    )
    p.add_argument(
        "--eval-samples",type=int,default=None,
        help="Samples for final E/CO/LC evaluation of every stage. Default=--n-samples.",
    )
    p.add_argument("--n-discard",type=int,default=8)
    p.add_argument("--n-chains",type=int,default=32)
    p.add_argument(
        "--sweep-size",type=int,default=None,
        help="Metropolis proposals between returned samples. Default=Nf; NetKet default would be Ns.",
    )
    p.add_argument("--sampler-chunk-size",type=int,default=None)
    p.add_argument("--chunk-size",type=int,default=256)

    p.add_argument(
        "--qgt",choices=["auto","pytree","onthefly","dense"],default="auto",
        help=(
            "QGT backend. auto selects QGTJacobianPyTree(mode=complex) for "
            "the mixed real/complex ansatz. onthefly is accepted for backward "
            "compatibility but safely falls back to pytree. dense is for small "
            "profiling only."
        ),
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
    spatial_chars_by_irrep={}
    requested_irreps=[]
    spatial_symmetry_effective="none"

    if a.spatial_symmetry!="none":
        try:
            spatial_ops=internal_c3v_space_group_site_maps(g)
            spatial_maps=np.asarray([p for _,p in spatial_ops],np.int32)
            spatial_symmetry_effective="c3v"
        except RuntimeError as exc:
            if a.spatial_symmetry=="c3v":
                raise RuntimeError(
                    "This finite torus does not preserve ordinary C3v. "
                    "Choose a C3-compatible shape such as L1=L2, or use "
                    "--spatial-symmetry auto/none to stop after Tint."
                ) from exc
            print(
                "[symmetry] ordinary C3v is unavailable for "
                f"L1={a.L1}, L2={a.L2}; auto mode will run Raw -> Tint only.",
                flush=True,
            )

    if spatial_symmetry_effective=="c3v":
        if a.c3v_irrep=="both":
            requested_irreps=["A1","A2"]
        elif a.c3v_irrep=="all":
            requested_irreps=["A1","A2","E"]
        else:
            requested_irreps=[a.c3v_irrep]

        # The full residual group is Tint x C3v. At internal Gamma the
        # characters of the one-dimensional irreps are repeated across the
        # three translation cosets. For E use the central character projector.
        for ir in requested_irreps:
            if ir in ("A1","A2"):
                spatial_chars_by_irrep[ir]=internal_gamma_c3v_characters(ir)
            else:
                from symmetry_projection import c3v_irrep_characters
                spatial_chars_by_irrep[ir]=np.tile(
                    c3v_irrep_characters("E"),3
                ).astype(np.complex128)

    settings={
        **vars(a),
        "Ns":int(g.Ns),"Nf":int(nf),
        "n_supercells":int(a.L1*a.L2),
        "variational_supercell_sites":18,
        "large_system_no_ed":True,
        "sparse_correlators":True,
        "packed_orbitals":True,
        "internal_translation_images":3,
        "full_residual_spatial_images":(
            18 if spatial_symmetry_effective=="c3v" else 0
        ),
        "spatial_symmetry_requested":a.spatial_symmetry,
        "spatial_symmetry_effective":spatial_symmetry_effective,
        "requested_c3v_irreps":requested_irreps,
        "notes":[
            "No ED/full-state enumeration is performed.",
            "Mixed real/complex parameters use QGTJacobianPyTree(mode=complex)+iterative solver by default; the dense P x P QGT is not formed.",
            "Metropolis sweep_size defaults to Nf rather than Ns.",
            "Determinant evaluation remains stateless/exact; no inverse cache is used.",
            "Tint+C3v/TR training defaults to fewer samples and shorter sweeps; final observables are reevaluated with eval_samples.",
            "Default spatial workflow is A1/A2 PAV screening followed by short VAP only in the lower sector; statistically tied sectors are both retained.",
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
                hi=hi,H_fast=H_fast,graph=graph,current_ops=current_ops,
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
                hi=hi,H_fast=H_fast,graph=graph,current_ops=current_ops,
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

        if requested_irreps:
            selected_irreps=list(requested_irreps)
            pav_rows={}
            pav_params={}

            if a.spatial_workflow in ("screen_then_vap","pav_only"):
                # Cheap A1/A2(/E) PAV screening: same optimized Tint parameters,
                # projected into each residual C3v sector, with no SR/Jacobian.
                for ir in requested_irreps:
                    spatial_chars=spatial_chars_by_irrep[ir]
                    pav_dir=out/(
                        f"large_tint_c3v_pav_{a.correlator}__{ir}__{seed}"
                    )
                    pav_summary,pp=_stage(
                        stage_name="Tint+C3v-PAV",
                        directory=pav_dir,
                        g=g,nf=nf,V=a.V,
                        hi=hi,H_fast=H_fast,graph=graph,current_ops=current_ops,
                        seed_data=seed_data,
                        shell_mats=shell_mats,bf_mats=bf_mats,
                        use_backflow=use_backflow,
                        initial_params=params,
                        maps=spatial_maps,chars=spatial_chars,
                        point_label=ir,
                        time_reversal_even=False,
                        n_iter=0,
                        learning_rate=a.spatial_learning_rate,
                        args=a,
                        rng_seed=base_seed+30+17*requested_irreps.index(ir),
                        train_samples=a.spatial_samples,
                        train_sweep_size=(
                            a.spatial_sweep_size
                            if a.spatial_sweep_size is not None
                            else max(2,nf//4)
                        ),
                    )
                    rows.append(pav_summary)
                    pav_rows[ir]=pav_summary
                    pav_params[ir]=pp

                if a.spatial_workflow=="screen_then_vap":
                    best_ir=min(
                        requested_irreps,
                        key=lambda ir:float(pav_rows[ir]["mc_E"]),
                    )
                    best_E=float(pav_rows[best_ir]["mc_E"])
                    best_err=float(pav_rows[best_ir].get("mc_E_error") or 0.0)
                    selected_irreps=[]
                    for ir in requested_irreps:
                        Ei=float(pav_rows[ir]["mc_E"])
                        ei=float(pav_rows[ir].get("mc_E_error") or 0.0)
                        comb=(best_err*best_err+ei*ei)**0.5
                        if ir==best_ir or (
                            comb>0.0
                            and (Ei-best_E)<=a.screen_keep_within_sigma*comb
                        ):
                            selected_irreps.append(ir)
                    print(
                        f"[C3v screen] seed={seed}: best={best_ir}; "
                        f"short-VAP sectors={selected_irreps}",
                        flush=True,
                    )
                else:
                    selected_irreps=[]

            if a.spatial_workflow!="pav_only":
                # Direct VAP starts from Tint. Screened VAP starts from the same
                # Tint parameters; the PAV state is a projected evaluation, not a
                # different variational parameter set.
                for ir in selected_irreps:
                    spatial_chars=spatial_chars_by_irrep[ir]
                    spatial_dir=out/(
                        f"large_tint_c3v_{a.correlator}__{ir}__{seed}"
                    )
                    if a.reuse_spatial:
                        spatial_summary,spatial_params=_load_params(spatial_dir)
                    else:
                        spatial_summary,spatial_params=_stage(
                            stage_name="Tint+C3v-VAP",
                            directory=spatial_dir,
                            g=g,nf=nf,V=a.V,
                            hi=hi,H_fast=H_fast,graph=graph,current_ops=current_ops,
                            seed_data=seed_data,
                            shell_mats=shell_mats,bf_mats=bf_mats,
                            use_backflow=use_backflow,
                            initial_params=params,
                            maps=spatial_maps,chars=spatial_chars,
                            point_label=ir,
                            time_reversal_even=False,
                            n_iter=a.spatial_iter,
                            learning_rate=a.spatial_learning_rate,
                            args=a,
                            rng_seed=base_seed+3+17*requested_irreps.index(ir),
                            train_samples=a.spatial_samples,
                            train_sweep_size=(
                                a.spatial_sweep_size
                                if a.spatial_sweep_size is not None
                                else max(2,nf//4)
                            ),
                        )
                    rows.append(spatial_summary)

                    if a.restore_time_reversal:
                        tr_dir=out/(
                            f"large_tint_c3vtr_{a.correlator}__"
                            f"{ir}__{seed}"
                        )
                        tr_summary,_=_stage(
                            stage_name="Tint+C3v+TR",
                            directory=tr_dir,
                            g=g,nf=nf,V=a.V,
                            hi=hi,H_fast=H_fast,graph=graph,current_ops=current_ops,
                            seed_data=seed_data,
                            shell_mats=shell_mats,bf_mats=bf_mats,
                            use_backflow=use_backflow,
                            initial_params=spatial_params,
                            maps=spatial_maps,chars=spatial_chars,
                            point_label=ir,
                            time_reversal_even=True,
                            n_iter=a.tr_iter,
                            learning_rate=a.tr_learning_rate,
                            args=a,
                            rng_seed=base_seed+4+17*requested_irreps.index(ir),
                            train_samples=a.spatial_samples,
                            train_sweep_size=(
                                a.spatial_sweep_size
                                if a.spatial_sweep_size is not None
                                else max(2,nf//4)
                            ),
                        )
                        rows.append(tr_summary)

        _write_csv(out/"large_supercell_summary.csv",rows)

    if spatial_symmetry_effective=="none":
        print(
            "\n[large supercell] point-group projection skipped: "
            "this torus does not preserve ordinary C3v.",
            flush=True,
        )

    print("\n[large supercell] summary")
    for r in rows:
        tp=r.get("time_per_iteration_s",None)
        tp_text=("PAV-only" if tp is None else f"{float(tp):.3f}s/it")
        print(
            f"  seed={r['seed']:11s} stage={r['stage']:14s} "
            f"irrep={str(r.get('point_label','-')):4s} "
            f"E={r['mc_E']:+.8f} +/- {r['mc_E_error']:.2e} "
            f"SCO={r['S_CO']:.5g} "
            f"SLC-={r['S_LC_opposite']:.5g} "
            f"time={tp_text} "
            f"images={r['explicit_projection_images']}"
        )
    print(f"Saved: {out}")


if __name__=="__main__":
    main()
