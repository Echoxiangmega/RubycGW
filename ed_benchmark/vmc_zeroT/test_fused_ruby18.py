"""18-site Ruby ring3 regression for fused VAP energy and LC estimators.

Uses real native_ed.geometry, 36 hopping edges, six triangular loops and
all 18 ordinary internal-translation x C3v site maps.
No NetKet is required. Run:
    python ed_benchmark/vmc_zeroT/test_fused_ruby18.py
"""
from pathlib import Path
import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'native_ed'))
import geometry
from projected_fused_kernels import make_fused_projected_kernel


def check():
    g=geometry.RubyPBC('ring3',t=1.,t1=.5,t2=-.5)
    ns=g.Ns; nf=6
    ident=np.arange(ns,dtype=int)
    translation=g.translate((1,0))
    point=g.point_maps()
    rotation=point['C3']
    mirrors=point['mirrors']
    if rotation is None or not mirrors:
        raise RuntimeError('The ring3 geometry does not provide C3v')
    mirror=next(iter(mirrors.values()))

    def compose(a,b):
        return b[a]

    group=[ident]
    while True:
        new=[]
        for m in group:
            for gen in (translation,rotation,mirror):
                p=compose(m,gen)
                if not any(np.array_equal(p,k) for k in group+new):
                    new.append(p)
        if not new:
            break
        group.extend(new)
        if len(group)>18:
            raise AssertionError('More than 18 maps generated')
    maps=np.asarray(group,dtype=int)
    assert len(maps)==18
    for p in maps:
        assert g.check_signed_map(p)

    rng=np.random.default_rng(4235)
    m0=(rng.normal(size=(ns,nf))+1j*rng.normal(size=(ns,nf)))*0.23
    backflow=np.zeros((4,ns,ns))
    shells=np.zeros((3,ns,ns))
    edges=np.asarray([(e.i,e.j) for e in g.edges],int)
    tvals=np.asarray([g.t[e.kind] for e in g.edges],float)
    vmask=np.asarray([e.kind=='t' for e in g.edges],float)
    for e in g.edges:
        k=(0 if e.kind=='t' and e.i%6<3 else
           1 if e.kind=='t' else 2 if e.kind=='t1' else 3)
        backflow[k,e.i,e.j]=backflow[k,e.j,e.i]=1
        c=(0 if e.kind=='t' else 1 if e.kind=='t1' else 2)
        shells[c,e.i,e.j]=shells[c,e.j,e.i]=1
    eta=np.asarray([.04+.02j,.05-.015j,.03+.01j,.02-.01j])
    v=np.asarray([.15,-.05,.04])
    W=np.einsum('k,kij->ij',eta,backflow)
    C=np.einsum('k,kij->ij',v,shells)
    loops=[loop for loop in g.cycles(lengths=(3,)) if len(loop)==3]
    assert len(loops)==6
    w=np.zeros((2,len(edges)),float)
    for loop in loops:
        sign_ab=(1 if loop[0]%6<3 else -1)
        for i,j in zip(loop,loop[1:]+loop[:1]):
            for ie,(a,b) in enumerate(edges):
                if (a,b)==(i,j):
                    sign=1
                    break
                if (a,b)==(j,i):
                    sign=-1
                    break
            else:
                raise AssertionError('Missing triangle edge')
            w[:,ie]+=np.asarray([1,sign_ab])*sign*tvals[ie]/(len(loops)*3)

    seed={'sc_coords':np.zeros((ns,2),int),'internal_sites':np.arange(ns),
          'kpoints':np.array([[0.,0.]]),'occupations_per_k':np.array([nf])}
    params={'M_packed':jnp.asarray(m0),'eta':jnp.asarray(eta),
            'v':jnp.asarray(v),'tr_phase':jnp.asarray(.25)}

    def fock_sign(p,source):
        occ=np.flatnonzero(source)
        m=p[occ]
        return (-1.)**sum(m[i]>m[j] for i in range(len(m))
                          for j in range(i+1,len(m)))

    def raw(x):
        dn=x-nf/ns
        qx=m0+W@(dn[:,None]*m0)
        return np.linalg.det(qx[np.flatnonzero(x)])*np.exp(.5*dn@C@dn)

    def projected(x,tr):
        z=sum(fock_sign(p,x[p])*raw(x[p]) for p in maps)
        return np.exp(.25j)*z+np.conj(np.exp(.25j)*z) if tr else z

    for tr in (False,True):
        kernel=make_fused_projected_kernel(
            seed_data=seed,shell_matrices=shells,backflow_matrices=backflow,
            edge_i=edges[:,0],edge_j=edges[:,1],edge_t=tvals,
            edge_has_interaction=vmask,interaction_V=1.6,
            site_maps=maps,characters=np.ones(len(maps)),
            w_same=w[0],w_opposite=w[1],
            time_reversal_even=tr,orthonormalize_orbitals=False)
        roots=[]
        for attempt in range(60):
            x=np.zeros(ns)
            x[rng.choice(ns,nf,replace=False)]=1
            if abs(projected(x,tr))>1e-10:
                roots.append(x)
            if len(roots)==3:
                break
        assert len(roots)==3
        fast=np.column_stack([np.asarray(z) for z in
                              kernel(params,jnp.asarray(roots))])
        direct=[]
        for x in roots:
            z=projected(x,tr)
            E=1.6*sum(x[i]*x[j] for (i,j),iv in zip(edges,vmask) if iv)
            js=0j;jo=0j
            for ie,(i,j) in enumerate(edges):
                if x[i]==x[j]:
                    continue
                y=x.copy()
                y[i],y[j]=y[j],y[i]
                parity=(-1.)**int(np.sum(x[min(i,j)+1:max(i,j)]))
                rr=parity*projected(y,tr)/z
                E+=tvals[ie]*rr
                direction=1 if x[i] else -1
                js+=1j*w[0,ie]*direction*rr
                jo+=1j*w[1,ie]*direction*rr
            direct.append((E,js,jo))
        error=float(np.max(np.abs(fast-np.asarray(direct))))
        print('Ruby18', 'TR' if tr else 'spatial','maximum error',error)
        assert error<1e-8
    print('PASS: full ring3 hopping/LC geometry')


if __name__=='__main__':
    check()
