#!/usr/bin/env python3
"""Native zero-T Matsubara Green matrix using Nf+1 and Nf-1 resolvents.

This is NOT finite-T Green: beta_reference specifies a Matsubara-grid spacing.
The chosen chemical potential is within the finite-cluster addition/removal gap
when such a gap exists. All orbital indices are absolute site indices.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from core import FixedHamiltonian,lowest,make_states
from geometry import RubyPBC
from observables import apply_ops


def krylov_cross(H,source,bra,zs,eta,sigma,m=80,tol=1e-11):
    """Return <bra_a|[z + eta + sigma*H]^-1|source> for all z."""
    norm=np.linalg.norm(source)
    result=np.zeros((len(zs),bra.shape[1]),complex)
    if norm<tol:return result
    q=source/norm;Q=[]
    alphas=[];betas=[]
    for k in range(min(m,H.shape[0])):
        v=H@q
        # Fully reorthogonalized Arnoldi/Lanczos; Hermitian tridiagonal verified.
        a=float(np.vdot(q,v).real)
        v-=a*q
        Q.append(q)
        for old in Q:
            v-=old*np.vdot(old,v)
        b=float(np.linalg.norm(v))
        alphas.append(a)
        if b<tol or k+1==m:break
        betas.append(b);q=v/b
    M=np.diag(alphas).astype(complex)
    if betas:M+=np.diag(betas,1)+np.diag(betas,-1)
    overlaps=bra.conj().T@np.column_stack(Q)
    e1=np.zeros(len(alphas),complex);e1[0]=norm
    for ix,z in enumerate(zs):
        result[ix]=overlaps@np.linalg.solve((z+eta)*np.eye(len(alphas))+sigma*M,e1)
    return result


def compute(path,orbitals=None,nfreq=32,beta_reference=100.,krylov=80,mu=None,method='sparse'):
    path=Path(path)
    cfg=json.loads((path/'geometry.json').read_text())
    g=RubyPBC(cfg['name'],**cfg['t'])
    nf=int(cfg['Nf']);V=float(cfg['V'])
    if not 0<nf<g.Ns:raise ValueError('Green requires 0<Nf<Ns')
    d=np.load(path/'ground_state.npz')
    psi=np.asarray(d['psi']);st=np.asarray(d['basis_states'])
    e0=float(d['e0']);ns=g.Ns
    if not np.array_equal(st,make_states(ns,nf)):raise AssertionError('Unexpected ground basis ordering')
    orbs=list(range(6)) if orbitals is None else list(orbitals)
    if not all(0<=i<ns for i in orbs):raise ValueError('orbital indices out of range')
    Hp=FixedHamiltonian(g,nf+1,V,method=method)
    Hm=FixedHamiltonian(g,nf-1,V,method=method)
    ep=lowest(Hp.operator,1)[0][0];em=lowest(Hm.operator,1)[0][0]
    mu_plus=float(ep-e0);mu_minus=float(e0-em)
    chemical_potential=(mu_plus+mu_minus)/2 if mu is None else float(mu)
    plus=np.column_stack([apply_ops(psi,st,Hp.states,ns,[('+',i)]) for i in orbs])
    minus=np.column_stack([apply_ops(psi,st,Hm.states,ns,[('-',i)]) for i in orbs])
    omega=np.pi*(2*np.arange(nfreq)+1)/beta_reference
    z=1j*omega;na=len(orbs)
    G=np.zeros((nfreq,na,na),complex)
    for j in range(na):
        # <c_i psi| (i w + E0 + mu - H_N+1)^-1 |c_j^dag psi>
        G[:,:,j]+=krylov_cross(Hp.operator,plus[:,j],plus,z,e0+chemical_potential,-1,m=krylov)
    for i in range(na):
        # <c_j psi| (i w - E0 + mu + H_N-1)^-1 |c_i psi>
        G[:,i,:]+=krylov_cross(Hm.operator,minus[:,i],minus,z,-e0+chemical_potential,1,m=krylov)
    outfile=path/'green_iw.npz'
    np.savez_compressed(outfile,G_iw=G,omega=omega,site_indices=orbs,
                        mu=chemical_potential,mu_plus=mu_plus,mu_minus=mu_minus,
                        E0=e0,Eplus=ep,Eminus=em,
                        beta_reference=beta_reference,krylov=krylov,
                        definition='G_ij(z)=<c_i (z+E0+mu-Hp)^-1 c_j^dag>+<c_j^dag (z-E0+mu+Hm)^-1 c_i>')
    print('Wrote',outfile,'shape',G.shape,'mu',chemical_potential,'charge gap',mu_plus-mu_minus)
    return G


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--orbitals',type=int,nargs='*',default=None)
    p.add_argument('--nfreq',type=int,default=32)
    p.add_argument('--beta-reference',type=float,default=100.)
    p.add_argument('--krylov',type=int,default=80)
    p.add_argument('--mu',type=float,default=None)
    p.add_argument('--method',choices=['sparse','matrixfree'],default='sparse')
    return p
if __name__=='__main__':
    a=parser().parse_args()
    compute(a.input,a.orbitals,a.nfreq,a.beta_reference,a.krylov,a.mu,a.method)