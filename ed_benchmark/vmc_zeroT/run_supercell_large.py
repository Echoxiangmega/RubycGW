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


def _mc_order_observables(vstate, g, nf, hi, current_ops):
    """Evaluate CO and the historical LC channels on the current MCState."""
    samples=np.asarray(vstate.samples)
    out=_charge_structure_from_samples(g,samples,nf)

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
    })
    return out



def _mc_site_bond_current(vstate,g,edge_chunk=12):
    """Estimate density, <c_i^dag c_j>, bond order and current from cached MC samples.

    Off-diagonal one-body density-matrix elements are evaluated from batched
    fermion-hop wavefunction ratios instead of calling expect() once per bond.
    For rho_ij=<c_i^dag c_j>, sampled configurations with i occupied and j
    empty contribute the ratio psi(x with i->j)/psi(x), including the canonical
    fermionic parity between i and j.
    """
    samples=np.asarray(vstate.samples)
    x=np.asarray(samples,float).reshape((-1,g.Ns))
    nsamp=len(x)
    density=np.mean(x,axis=0)
    logx=np.asarray(vstate.log_value(x),np.complex128).reshape(-1)
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
        logp=np.asarray(vstate.log_value(joined),np.complex128).reshape(-1)
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


def _save_mc_maps(directory,vstate,g,title,edge_chunk=12):
    """Save site/bond/current CSVs and a single triptych for a large-system MC state."""
    from plot_saved_state_maps import (
        site_positions,edge_segments,export_site_csv,export_bond_csv,
        plot_site_bond_current_triptych,
    )
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    density,rho,bond,current=_mc_site_bond_current(
        vstate,g,edge_chunk=edge_chunk
    )

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
        "estimator":"batched fermion-hop wavefunction ratios on cached MC samples",
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
        def on_step_end(self,step,log_data,driver):
            del log_data
            if (int(step)+1)%interval==0:
                _gauge_fix_packed_orbitals(
                    driver.state,occupations_per_k
                )

        def on_run_end(self,step,driver):
            del step
            _gauge_fix_packed_orbitals(
                driver.state,occupations_per_k
            )

    return OrbitalGaugeFix()


def _validate_analytic_logderiv(
    *,
    vstate_analytic,
    vstate_reference,
    H_reference,
    jax,
):
    """Compare analytic-JVP and original-autodiff Raw energy gradients."""
    vstate_analytic.sample()
    samples=vstate_analytic.samples
    # Supported NetKet extension point: inject the same cached samples so the
    # two gradient estimates are strictly comparable.
    vstate_reference._samples=samples
    vstate_reference.parameters=vstate_analytic.parameters

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
    print(
        "[analytic logderiv check] "
        f"dE={report['energy_abs_error']:.3e} "
        f"grad_max={report['max_abs_error']:.3e} "
        f"grad_rel_l2={report['relative_l2_error']:.3e}",
        flush=True,
    )
    return report


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

    sampler=nk.sampler.MetropolisFermionHop(hi,**sampler_kwargs)
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
                if passed:
                    _ANALYTIC_LOGDERIV_VALIDATED_KEYS.add(key)
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
                    ref_vstate._samples=vstate.samples
                    ref_vstate.parameters=vstate.parameters
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
                ref_vstate._samples=vstate.samples
                ref_vstate.parameters=vstate.parameters
                vstate=ref_vstate
                model=ref_model
                analytic_logderiv_effective=False

    fast_local_requested=bool(getattr(args,"fast_local_energy",False))
    fast_local_eligible=(
        fast_local_requested
        and maps is None
        and not bool(time_reversal_even)
        and use_backflow
    )
    fast_local_effective=False
    fast_local_error=None
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
                    fast_local_error=msg
                else:
                    _FAST_LOCAL_GRADIENT_VALIDATED=True
            except Exception as exc:
                if bool(getattr(args,"fast_local_strict",False)):
                    raise
                fast_local_error=f"gradient validation error: {exc}"
                H_stage=H_fast
                fast_local_effective=False
                print(
                    "[fast local] gradient validation raised "
                    f"{exc}; falling back to NetKet operator.",
                    flush=True,
                )

    setup_s=time.perf_counter()-t0

    warm=None
    warmup_s=0.0
    vmc_profile={}
    fast_local_check={}
    fast_local_gradient_check={}
    train_segment_timings=[]

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
        solver_restart_effective=None
        if trainer_requested not in ("auto","vmc_sr","legacy"):
            raise ValueError(f"Unknown trainer={trainer_requested}")
        trainer_effective=trainer_requested
        sr_momentum=float(getattr(args,"sr_momentum",0.8))
        if not 0.0<=sr_momentum<1.0:
            raise ValueError("--sr-momentum must satisfy 0 <= momentum < 1")

        driver=None
        vmc_sr_error=None
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

        if driver is None:
            trainer_effective="legacy"
            # Periodic QR gauge fixing changes the parameter coordinates while
            # leaving the normalized physical state unchanged.  A previous CG
            # solution is therefore not a meaningful restart vector across a
            # gauge-fix event.  Disable solver restart in the QR-free gauge.
            solver_restart_effective=not orbital_free_effective
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

    tobs=time.perf_counter()
    if measure_orders:
        order_obs=_mc_order_observables(
            vstate,g,nf,hi,current_ops
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
        )

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
        "vmc_component_profile":vmc_profile,
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
