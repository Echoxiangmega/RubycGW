"""Projected rank-3 Woodbury kernels for the Ruby ring3 supercell VMC.

Install next to run_supercell_large.py and supercell_ansatz.py.

The *unprojected* Woodbury kernel E_loc^(g)(x) from supercell_ansatz
can be re-used exactly for each spatial symmetry image g because
[H,U_g] = 0.  The projected energy is

  E_loc^P(x) = (sum_g a_g(x) E_loc(g.x))/(sum_g a_g(x)),
  a_g(x) = chi_g^* sign_g(x) psi(g.x),

with an optional spinless time reversal T=K applied to BOTH numerator and
denominator. This is not a heuristic or mean-field approximation.

A persistent-cache projected Metropolis sampler is also provided. It stores
one rank-3-update inverse per (chain, spatial image), and samples the exact
projected |Psi|^2, not an incoherent mixture of image probabilities.

Requires JAX and NetKet only when constructing runtime kernels / sampler.
"""
from __future__ import annotations

from functools import partial
import numpy as np


def check_spatial_maps(g, site_maps, characters, *, tr_even=False):
    maps=np.asarray(site_maps, dtype=np.int32)
    chars=np.asarray(characters, dtype=np.complex128)
    if maps.ndim!=2 or maps.shape[1]!=g.Ns:
        raise ValueError('site_maps must have shape (n_images,Ns)')
    if chars.shape!=(len(maps),):
        raise ValueError('characters length does not match number of images')
    for i,p in enumerate(maps):
        if sorted(map(int,p))!=list(range(g.Ns)):
            raise ValueError(f'image {i} site map is not a permutation')
        if not g.check_signed_map(p):
            raise ValueError(f'image {i} does not commute with physical hopping and V')
    if tr_even and any(abs(np.imag(v))>1e-12 for v in g.t.values()):
        raise ValueError('T=K time-reversal combination needs real hopping gauge')
    return maps,chars


def _fock_signs_jax(maps, roots, nf):
    """Match supercell_ansatz.py's fock_sign(p, source=x[p]) exactly.

    roots shape (n_roots,Ns), result (n_roots,n_images).
    """
    import jax
    import jax.numpy as jnp
    maps=jnp.asarray(maps,dtype=jnp.int32)
    def single(p, x):
        source=x[p]
        occ=jnp.nonzero(source,size=nf)[0]
        mapped=p[occ]
        inv=(mapped[:,None]>mapped[None,:])
        parity=jnp.sum(jnp.triu(inv,k=1), dtype=jnp.int32)&1
        return jnp.where(parity==0,1.0,-1.0)
    return jax.vmap(lambda x:jax.vmap(single,in_axes=(0,None))(maps,x))(roots)


def _project_sum(amps, tr_phase, tr_even):
    import jax.numpy as jnp
    z=jnp.sum(amps,axis=-1)
    if tr_even:
        z=jnp.exp(1j*tr_phase)*z
        z=z+jnp.conj(z)
    return z


def make_projected_fast_local_energy_kernel(
    *, raw_model, raw_local_energy_kernel, site_maps, characters,
    nf: int, time_reversal_even: bool=False, batch_strategy: str='scan',
):
    """Exact projected local-energy kernel, re-using existing Raw Woodbury.

    raw_model is build_supercell_netket_model(..., point_site_maps=None,
    time_reversal_even=False, packed_orbitals=True, ...), with the same
    sparse/Jastrow/orbital switches as the projected model.

    Input is (params, samples[n,Ns]); output is complex E_loc[n].
    'scan' avoids materializing n_images * samples in the costly rank-3
    intermediates; 'flatten' can be faster at smaller Ns with enough memory.
    """
    import jax
    import jax.numpy as jnp
    maps=jnp.asarray(site_maps,dtype=jnp.int32)
    chars=jnp.asarray(characters,dtype=jnp.complex128)
    nimages=int(len(characters))
    nf=int(nf)
    if batch_strategy not in ('scan','flatten'):
        raise ValueError('batch_strategy must be scan or flatten')

    def per_image(pars,p,x):
        xg=x[:,p]
        logs=raw_model.apply({'params':pars},xg)
        energy=raw_local_energy_kernel(pars,xg)
        occ=jax.vmap(lambda row:jnp.nonzero(row,size=nf)[0])(xg)
        mapped=jnp.take(p,occ)
        inv=mapped[:,:,None]>mapped[:,None,:]
        sign=jnp.where((jnp.sum(jnp.triu(inv,k=1),axis=(1,2))&1)==0,1.0,-1.0)
        return logs,energy,sign

    @jax.jit
    def kernel(parameters, configurations):
        pars={k:v for k,v in parameters.items() if k!='tr_phase'}
        phi=parameters['tr_phase'] if time_reversal_even else jnp.asarray(0.0)
        x=jnp.asarray(configurations)
        if batch_strategy=='scan':
            def scan_one(_,p):
                return None,per_image(pars,p,x)
            _,out=jax.lax.scan(scan_one,None,maps)
            logs, energies, signs=out # (G,B)
        else:
            sources=jnp.take(x,maps,axis=1) # (B,G,Ns)
            flat=sources.reshape((-1,x.shape[1]))
            logs=raw_model.apply({'params':pars},flat).reshape((x.shape[0],nimages)).T
            energies=raw_local_energy_kernel(pars,flat).reshape((x.shape[0],nimages)).T
            signs=_fock_signs_jax(maps,x,nf).T
        shift=jnp.max(jnp.real(logs),axis=0,keepdims=True)
        amps=jnp.conj(chars)[:,None]*signs*jnp.exp(logs-shift)
        denominator=jnp.sum(amps,axis=0)
        numerator=jnp.sum(amps*energies,axis=0)
        if time_reversal_even:
            phase=jnp.exp(1j*phi)
            denominator=phase*denominator
            denominator=denominator+jnp.conj(denominator)
            numerator=phase*numerator
            numerator=numerator+jnp.conj(numerator)
        small=jnp.asarray(1e-300+0j,dtype=denominator.dtype)
        safe=jnp.where(jnp.abs(denominator)>1e-300,denominator,small)
        return numerator/safe
    return kernel


def make_projected_cached_sampler(
    nk, jax, *, hilbert, seed_data, shell_matrices, backflow_matrices,
    clusters, site_maps, characters, time_reversal_even=False,
    n_chains=32, sweep_size=12, chunk_size=None, orthonormalize_orbitals=True,
):
    """Persistent-cache exact projected Metropolis sampler (experimental).

    Each image keeps a slot-ordered determinant inverse and Jastrow field.
    A physical hop maps to the corresponding hop in every image. The
    antisymmetric Fock-map signs reduce the image amplitude update to the
    *common original Fock hopping parity* times each slot-ordered determinant
    ratio, which avoids O(Nf^2) sign recomputation on every proposal.

    Cache initialization uses the vetted Raw init_cache implementation.
    A caller MUST validate sampled log-density against the projected model
    and compare observable averages to the standard sampler before using it
    for production. The --check-sampler option does so automatically.
    """
    import jax.numpy as jnp
    from supercell_ansatz import make_fast_raw_sampler_cache_kernels
    maps_np=np.asarray(site_maps,np.int32)
    chars_np=np.asarray(characters,np.complex128)
    inv_maps=jnp.asarray(np.argsort(maps_np,axis=1),jnp.int32)
    maps=jnp.asarray(maps_np,jnp.int32)
    chars=jnp.asarray(chars_np,jnp.complex128)
    edges=jnp.asarray(clusters,jnp.int32)
    ns=int(maps_np.shape[1]); ng=int(maps_np.shape[0]); nf=int(np.sum(seed_data['occupations_per_k']))
    sites_range=jnp.arange(ns)
    eye3=jnp.eye(3,dtype=jnp.complex128)
    raw_init,_,_=make_fast_raw_sampler_cache_kernels(
        sc_coords=seed_data['sc_coords'],
        internal_sites=seed_data['internal_sites'],
        kpoints=seed_data['kpoints'],
        occupations_per_k=seed_data['occupations_per_k'],
        shell_matrices=shell_matrices,
        backflow_matrices=backflow_matrices,
        clusters=np.asarray(clusters,np.int32),
        orthonormalize_orbitals=orthonormalize_orbitals,
        machine_pow=2.0,
    )
    filling=float(nf/ns)

    @jax.jit
    def init_cache(parameters, configurations):
        pars={k:v for k,v in parameters.items() if k!='tr_phase'}
        x=jnp.asarray(configurations,dtype=jnp.int8)
        b=int(x.shape[0]); y=jnp.take(x,maps,axis=1)
        raw=raw_init(pars,y.reshape((b*ng,ns)))
        qx=raw['Qx']; occupied=raw['sites']
        matrices=jax.vmap(lambda q,idx:q[idx])(qx,occupied)
        detsign,logabs=jnp.linalg.slogdet(matrices)
        dn=y.reshape((b*ng,ns)).astype(jnp.float64)-filling
        J=0.5*jnp.sum(dn*raw['h'],axis=-1)
        log_weight=(logabs+J).reshape((b,ng))
        phas=detsign.reshape((b,ng))
        fsgn=_fock_signs_jax(maps,x,nf)
        shift=jnp.max(log_weight,axis=1,keepdims=True)
        amps=(
            jnp.conj(chars)[None,:]
            *fsgn*phas*jnp.exp(log_weight-shift)
        )
        return {
            'x':x,'Q':raw['Q'],'W':raw['W'],'C':raw['C'],
            'sites':raw['sites'].reshape((b,ng,nf)),
            'Qx':raw['Qx'].reshape((b,ng,ns,nf)),
            'Ainv':raw['Ainv'].reshape((b,ng,nf,nf)),
            'h':raw['h'].reshape((b,ng,ns)),
            'amps':amps,
            'tr_phase':parameters['tr_phase'] if time_reversal_even else jnp.asarray(0.0),
        }

    def one_step(cache, prop_uniforms, accept_uniforms):
        Q=cache['Q']; W=cache['W']; C=cache['C']
        phi=cache['tr_phase']
        def hop_one_chain(x, sites, Qx, Ainv, h, amps, up, ua):
            mask=x[edges[:,0]]!=x[edges[:,1]]
            nactive=jnp.sum(mask,dtype=jnp.int32)
            rank=jnp.minimum(jnp.asarray(jnp.floor(up*nactive),jnp.int32),nactive-1)
            edge=jnp.argmax(jnp.cumsum(mask.astype(jnp.int32))>rank)
            i=edges[edge,0]; j=edges[edge,1]
            src=jnp.where(x[i]>0,i,j)
            dst=jnp.where(x[i]>0,j,i)
            gsrc=inv_maps[:,src]; gdst=inv_maps[:,dst]
            parity=jnp.sum(x*((sites_range>jnp.minimum(src,dst))&(sites_range<jnp.maximum(src,dst))))
            fock_parity=jnp.where((parity&1)==0,1.,-1.)

            def image_ratio(site, qx, ainv, hh, s, d):
                slot=jnp.argmax(site==s)
                U=jnp.stack((-W[site,s],W[site,d],(jnp.arange(nf)==slot).astype(Q.dtype)),axis=1)
                delta_q=-W[:,s,None]*Q[s,:]+W[:,d,None]*Q[d,:]
                qx_new=qx+delta_q
                replacement=qx_new[d,:]-qx_new[s,:]
                VV=jnp.stack((Q[s,:],Q[d,:],replacement),axis=1)
                XX=ainv@U
                small=eye3+VV.T@XX
                ratio=jnp.linalg.det(small)
                delta_j=-hh[s]+hh[d]+0.5*(C[s,s]+C[d,d]-C[s,d]-C[d,s])
                new_inv=ainv-XX@jnp.linalg.solve(small,VV.T@ainv)
                new_site=site.at[slot].set(d)
                new_h=hh-C[:,s]+C[:,d]
                return ratio*jnp.exp(delta_j),new_site,qx_new,new_inv,new_h

            ratios, new_sites, new_Qx, new_Ainv,new_h=jax.vmap(image_ratio)(
                sites,Qx,Ainv,h,gsrc,gdst
            )
            proposed=amps*(fock_parity*ratios)
            zold=_project_sum(amps,phi,time_reversal_even)
            znew=_project_sum(proposed,phi,time_reversal_even)
            logratio=2.0*(
                jnp.log(jnp.maximum(jnp.abs(znew),1e-300))
                -jnp.log(jnp.maximum(jnp.abs(zold),1e-300))
            )
            xnew=x.at[src].set(0).at[dst].set(1)
            masknew=xnew[edges[:,0]]!=xnew[edges[:,1]]
            nactive_new=jnp.sum(masknew,dtype=jnp.int32)
            hastings=jnp.log(nactive.astype(jnp.float64))-jnp.log(nactive_new.astype(jnp.float64))
            finite=jnp.isfinite(logratio)&jnp.all(jnp.isfinite(ratios))&jnp.all(jnp.isfinite(new_Ainv))
            accept=finite & (jnp.log(jnp.maximum(ua,1e-300))<(logratio+hastings))
            scale=jnp.maximum(jnp.max(jnp.abs(proposed)),1e-300)
            normalized=proposed/scale
            return (
                jnp.where(accept,xnew,x),
                jnp.where(accept,new_sites,sites),
                jnp.where(accept,new_Qx,Qx),
                jnp.where(accept,new_Ainv,Ainv),
                jnp.where(accept,new_h,h),
                jnp.where(accept,normalized,amps),
                accept,
                jnp.where(accept,logratio,0.0),
            )

        (x,sites,qx,ainv,h,amps,accepted,logdelta)=jax.vmap(hop_one_chain)(
            cache['x'],cache['sites'],cache['Qx'],cache['Ainv'],cache['h'],
            cache['amps'],prop_uniforms,accept_uniforms,
        )
        new_cache={
            **cache,'x':x,'sites':sites,'Qx':qx,'Ainv':ainv,'h':h,'amps':amps
        }
        return new_cache,accepted,logdelta

    @jax.jit
    def advance(cache, proposal_uniforms, accept_uniforms):
        def step(carry,inputs):
            state,total,totalaccept=carry
            up,ua=inputs
            state,accept,logdelta=one_step(state,up,ua)
            return (state,total+logdelta,totalaccept+accept.astype(jnp.int32)),None
        (state,total,naccepted),_=jax.lax.scan(
            step,
            (cache,jnp.zeros(cache['x'].shape[0]),jnp.zeros(cache['x'].shape[0],dtype=jnp.int32)),
            (proposal_uniforms.T,accept_uniforms.T)
        )
        return state,total,naccepted

    class FastProjectedMetropolisSampler(nk.sampler.MetropolisSampler):
        _init_fun: object=nk.utils.struct.field(pytree_node=False,serialize=False,default=None)
        _advance_fun: object=nk.utils.struct.field(pytree_node=False,serialize=False,default=None)

        def __init__(self):
            rule=nk.sampler.rules.FermionHopRule(
                hilbert,clusters=[tuple(map(int,e)) for e in np.asarray(clusters)]
            )
            super().__init__(
                hilbert,rule,n_chains=int(n_chains),sweep_size=int(sweep_size),
                chunk_size=(None if chunk_size is None else int(chunk_size)),
                machine_pow=2,
            )
            self._init_fun=init_cache
            self._advance_fun=advance

        def _reset(self,machine,parameters,state):
            state=super()._reset(machine,parameters,state)
            pars=parameters['params'] if 'params' in parameters else parameters
            cache=self._init_fun(pars,state.σ)
            # Catch malformed cache before any sampling. JAX assertions outside
            # jit are possible here, but runtime NaN checks during steps also
            # reject singular proposals.
            return state.replace(rule_state=cache)

        def _sample_next(self,machine,parameters,state):
            del machine,parameters
            next_rng,key_prop,key_acc=jax.random.split(state.rng,3)
            shape=(self.n_batches,self.sweep_size)
            up=jax.random.uniform(key_prop,shape=shape,dtype=state.log_prob.dtype)
            ua=jax.random.uniform(key_acc,shape=shape,dtype=state.log_prob.dtype)
            new_cache,dlogp,naccepted=self._advance_fun(state.rule_state,up,ua)
            state=state.replace(
                σ=new_cache['x'],
                log_prob=state.log_prob+dlogp,
                rng=next_rng,
                rule_state=new_cache,
                n_accepted_proc=state.n_accepted_proc+naccepted,
                n_steps_proc=state.n_steps_proc+jnp.asarray(
                    self.n_batches*self.sweep_size,dtype=state.n_steps_proc.dtype
                ),
            )
            return state,(state.σ,state.log_prob)

    return FastProjectedMetropolisSampler()