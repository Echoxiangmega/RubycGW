"""Pure JAX/NumPy algebra tests (NetKet not needed)."""
from __future__ import annotations
import sys
from types import SimpleNamespace
from itertools import combinations
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from projected_fast_kernels import make_projected_fast_local_energy_kernel,make_projected_cached_sampler

NS=6; NF=3
rng=np.random.default_rng(3434)
Q=(rng.normal(size=(NS,NF))+1j*rng.normal(size=(NS,NF)))/2.5
T=0.13
V=0.5
ES=np.asarray([(i,(i+1)%NS) for i in range(NS)],np.int32)
W=np.zeros((NS,NS),complex)
C=np.zeros((NS,NS),float)
for i,j in ES:
    W[i,j]=W[j,i]=0.07+0.03j
    C[i,j]=C[j,i]=0.28
maps=np.asarray([[(i+k)%NS for i in range(NS)] for k in range(NS)],int)
# Full spatial group rotation plus reflected images
maps=np.concatenate((maps,np.asarray([[(-i+k)%NS for i in range(NS)] for k in range(NS)],int)))
chars=np.ones(len(maps),complex)


def phi_raw(x):
    x=np.asarray(x,int)
    dn=x-NF/NS
    Qx=Q+np.einsum('ij,j,ja->ia',W,dn,Q)
    m=Qx[np.flatnonzero(x)]
    return np.linalg.det(m)*np.exp(0.5*dn@C@dn)


def fock_sign(p,src):
    occ=np.flatnonzero(src)
    mapped=np.asarray(p)[occ]
    inv=sum(mapped[i]>mapped[j] for i in range(len(occ)) for j in range(i+1,len(occ)))
    return -1 if inv%2 else 1


def phi_proj(x,tr=False,phase=0.23):
    a=sum(np.conj(c)*fock_sign(p,np.asarray(x)[p])*phi_raw(np.asarray(x)[p]) for p,c in zip(maps,chars))
    if tr:a=np.exp(1j*phase)*a+np.conj(np.exp(1j*phase)*a)
    return a


def eval_local(func,x):
    x=np.asarray(x,int)
    z=func(x)
    diagonal=sum(V*x[i]*x[j] for i,j in ES)
    for i,j in ES:
        if x[i]==x[j]:continue
        xx=x.copy(); xx[i],xx[j]=xx[j],xx[i]
        f=(-1)**sum(x[min(i,j)+1:max(i,j)])
        diagonal+=T*f*func(xx)/z
    return diagonal

class RawModel:
    def apply(self,d,x):
        xx=jnp.asarray(x)
        QQ=jnp.asarray(d['params']['M_packed'])
        WW=jnp.asarray(d['params']['eta'][0])*jnp.asarray(W/(0.07+0.03j))
        CC=jnp.asarray(d['params']['v'][0])*jnp.asarray(C/0.28)
        def one(row):
            dn=row-NF/NS
            M=QQ+jnp.einsum('ij,j,ja->ia',WW,dn,QQ)
            ids=jnp.nonzero(row,size=NF)[0]
            return jnp.log(jnp.linalg.det(M[ids]))+0.5*(dn@CC@dn)
        return jax.vmap(one)(xx)

params={'M_packed':jnp.asarray(Q),'eta':jnp.asarray([0.07+0.03j]),'v':jnp.asarray([0.28]),'tr_phase':jnp.asarray(0.23)}
model=RawModel()

def raw_direct_kernel(pars,x):
    def one(row):
        den=jnp.exp(model.apply({'params':pars},row[None,:])[0])
        diag=0.
        for i,j in ES:
            diag+=V*row[i]*row[j]
        for i,j in ES:
            r2=row.at[i].set(row[j]).at[j].set(row[i])
            f=(-1)**sum(row[min(i,j)+1:max(i,j)])
            ratio=jnp.exp(model.apply({'params':pars},r2[None,:])[0])/den
            diag+=jnp.where(row[i]!=row[j],T*f*ratio,0j)
        return diag
    return jax.vmap(one)(x)


def test_projected_energy():
    all_x=np.asarray([np.eye(NS,dtype=int)[list(c)].sum(axis=0) for c in combinations(range(NS),NF)],float)
    global chars
    for char_label in ('A1','A2','E'):
        chars=(np.tile(np.asarray([2,-1,-1,0,0,0],float),2) if char_label=='E' else np.concatenate((np.ones(6),(-np.ones(6) if char_label=='A2' else np.ones(6)))).astype(complex))
        for tr in [False,True]:
            for strategy in ['scan','flatten']:
                kernel=make_projected_fast_local_energy_kernel(
                    raw_model=model,raw_local_energy_kernel=raw_direct_kernel,
                    site_maps=maps,characters=chars,nf=NF,
                    time_reversal_even=tr,batch_strategy=strategy,
                )
                valid=np.asarray([x for x in all_x if abs(phi_proj(x,tr=tr))>1e-7])
                if not len(valid):
                    print('skipped symmetry sector (zero norm):',char_label,tr);continue
                fast=np.asarray(kernel(params,valid))
                exact=np.asarray([eval_local(lambda z:phi_proj(z,tr=tr),x) for x in valid])
                assert np.max(np.abs(fast-exact))<1e-8,(tr,char_label,strategy,np.max(np.abs(fast-exact)))
                print('projected energy',char_label,'TR' if tr else 'spatial',strategy,'max error:',np.max(np.abs(fast-exact)))
    chars=np.ones(len(maps),complex)


def _fake_raw_init(*args,**kwargs):
    def initial(pars,config):
        qq=jnp.asarray(pars['M_packed'])
        ww=jnp.asarray(W/(0.07+0.03j))*pars['eta'][0]
        cc=jnp.asarray(C/0.28)*pars['v'][0]
        def one(x):
            dn=x-NF/NS
            qx=qq+jnp.einsum('ij,j,ja->ia',ww,dn,qq)
            ss=jnp.nonzero(x,size=NF)[0]
            m=qx[ss]
            return ss,qx,jnp.linalg.inv(m),cc@dn
        sites,qx,ainv,h=jax.vmap(one)(jnp.asarray(config))
        return {'Q':qq,'W':ww,'C':cc,'sites':sites,'Qx':qx,'Ainv':ainv,'h':h}
    return initial,None,None

class FakeStruct:
    @staticmethod
    def field(**kwargs):return None

class FakeSampler:
    def __init__(self,*args,**kwargs):
        self.n_batches=kwargs.get('n_chains',4);self.sweep_size=kwargs.get('sweep_size',3)

fake_nk=SimpleNamespace(sampler=SimpleNamespace(MetropolisSampler=FakeSampler,rules=SimpleNamespace(FermionHopRule=lambda *a,**kw:None)),utils=SimpleNamespace(struct=FakeStruct))
sys.modules['supercell_ansatz']=SimpleNamespace(make_fast_raw_sampler_cache_kernels=_fake_raw_init)
seed_data={'sc_coords':np.zeros((NS,2),int),'internal_sites':np.arange(NS),'kpoints':np.asarray([[0,0.]]),'occupations_per_k':np.asarray([NF])}

def test_sampler():
    x=np.asarray([np.eye(NS,dtype=int)[list(c)].sum(axis=0) for c in [c for c in combinations(range(NS),NF) if abs(phi_proj(np.eye(NS,dtype=int)[list(c)].sum(axis=0)))>1e-7][:4]],float)
    global chars
    for char_label in ('A1','A2','E'):
        chars=(np.tile(np.asarray([2,-1,-1,0,0,0],float),2) if char_label=='E' else np.concatenate((np.ones(6),(-np.ones(6) if char_label=='A2' else np.ones(6)))).astype(complex))
        x=np.asarray([np.eye(NS,dtype=int)[list(c)].sum(axis=0) for c in [c for c in combinations(range(NS),NF) if abs(phi_proj(np.eye(NS,dtype=int)[list(c)].sum(axis=0)))>1e-7][:4]],float)
        for tr in [False,True]:
            sampler=make_projected_cached_sampler(fake_nk,jax,hilbert=None,seed_data=seed_data,
                shell_matrices=np.asarray([C/0.28]),backflow_matrices=np.asarray([W/(0.07+0.03j)]),
                clusters=ES,site_maps=maps,characters=chars,time_reversal_even=tr,
                n_chains=4,sweep_size=3,orthonormalize_orbitals=False)
            init=sampler._init_fun(params,jnp.asarray(x))
            uu=rng.random((4,10));aa=rng.random((4,10))
            state,delta,naccepted=sampler._advance_fun(init,jnp.asarray(uu),jnp.asarray(aa))
            fin=np.asarray(state['x'])
            exact=np.asarray([2*np.log(abs(phi_proj(y,tr=tr))/abs(phi_proj(z,tr=tr))) for z,y in zip(x,fin)])
            err=np.max(abs(exact-np.asarray(delta)))
            assert err<1e-8,(tr,char_label,err,exact,np.asarray(delta),fin)
            print('projected sampler',char_label,'TR' if tr else 'spatial','max error:',err,'accepted:',np.asarray(naccepted))

if __name__=='__main__':
    test_projected_energy()
    test_sampler()
    print('ALL TESTS PASSED')
