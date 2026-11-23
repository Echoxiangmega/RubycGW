"""JAX/NumPy regression for fused projected energy and current estimators.

Run from ed_benchmark/vmc_zeroT:
    python test_fused_projected_kernels.py

Does not require NetKet. It independently evaluates each hopping-related
projected many-body amplitude, including fermionic signs and time reversal.
"""
from itertools import combinations
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from projected_fused_kernels import make_fused_projected_kernel


def test():
    ns=6
    nf=3
    rng=np.random.default_rng(3811)
    q=(rng.normal(size=(ns,nf))+1j*rng.normal(size=(ns,nf)))*0.33
    edge=np.asarray([(i,(i+1)%ns) for i in range(ns)],int)
    b=np.zeros((ns,ns),float)
    c=np.zeros((ns,ns),float)
    for i,j in edge:
        b[i,j]=b[j,i]=1
        c[i,j]=c[j,i]=1
    maps=np.asarray([[(i+k)%ns for i in range(ns)] for k in range(ns)] +
                    [[(-i+k)%ns for i in range(ns)] for k in range(ns)],int)
    sw=np.array([1,1,-1,1,-1,1],float)/ns
    ow=np.array([-1,1,1,-1,1,-1],float)/ns
    amp=0.07+0.03j
    vv=0.28
    pars={'M_packed':jnp.asarray(q),'eta':jnp.asarray([amp]),
          'v':jnp.asarray([vv]),'tr_phase':jnp.asarray(0.23)}
    sd={'sc_coords':np.zeros((ns,2),int),'internal_sites':np.arange(ns),
        'kpoints':np.array([[0.,0.]]),'occupations_per_k':np.array([nf])}

    def fsign(p,x):
        occupied=np.flatnonzero(x[p])
        mapped=p[occupied]
        inversions=sum(mapped[i]>mapped[j] for i in range(len(mapped))
                       for j in range(i+1,len(mapped)))
        return (-1.)**inversions

    def raw(x):
        dn=x-nf/ns
        qx=q+amp*b@(dn[:,None]*q)
        return np.linalg.det(qx[np.flatnonzero(x)])*np.exp(0.5*vv*dn@c@dn)

    def projected(x,chars,tr):
        z=sum(np.conj(ch)*fsign(p,x)*raw(x[p])
              for p,ch in zip(maps,chars))
        return np.exp(0.23j)*z+np.conj(np.exp(0.23j)*z) if tr else z

    def direct(x,chars,tr):
        z=projected(x,chars,tr)
        energy=0.6*sum(x[i]*x[j] for i,j in edge)
        js=0j;jo=0j
        for ie,(i,j) in enumerate(edge):
            if x[i]==x[j]:continue
            y=x.copy();y[i],y[j]=y[j],y[i]
            f=(-1.)**sum(x[min(i,j)+1:max(i,j)])
            ratio=f*projected(y,chars,tr)/z
            energy+=0.13*ratio
            direction=1 if x[i] else -1
            js+=1j*sw[ie]*direction*ratio
            jo+=1j*ow[ie]*direction*ratio
        return (energy,js,jo)

    states=np.asarray([np.eye(ns,dtype=int)[list(occ)].sum(axis=0)
                       for occ in combinations(range(ns),nf)],float)
    worst=0.
    for label,chars in [
        ('A1',np.ones(12,complex)),
        ('A2',np.r_[np.ones(6),-np.ones(6)].astype(complex)),
        ('E',np.r_[2,1,-1,-2,-1,1,0,0,0,0,0,0].astype(complex)),
    ]:
        for tr in (False,True):
            roots=np.asarray([x for x in states if abs(projected(x,chars,tr))>1e-9])
            if not len(roots):continue
            ker=make_fused_projected_kernel(
                seed_data=sd,shell_matrices=np.asarray([c]),
                backflow_matrices=np.asarray([b]),edge_i=edge[:,0],
                edge_j=edge[:,1],edge_t=np.full(len(edge),0.13),
                edge_has_interaction=np.ones(len(edge)),interaction_V=0.6,
                site_maps=maps,characters=chars,w_same=sw,w_opposite=ow,
                time_reversal_even=tr,orthonormalize_orbitals=False,
            )
            observed=np.column_stack([np.asarray(z) for z in ker(pars,jnp.asarray(roots))])
            expected=np.asarray([direct(x,chars,tr) for x in roots])
            error=float(np.max(np.abs(observed-expected)))
            worst=max(worst,error)
            print(label,'TR' if tr else 'spatial','max abs error',error)
            assert error<1e-10, (label,tr,error)
    print('PASS: all direct hopping, current and TR checks; worst =',worst)


if __name__=='__main__':
    test()
