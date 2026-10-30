#!/usr/bin/env python3
"""Zero-temperature Matsubara Green matrix and one-body response spectra.

Reads a fixed-N QuSpin ground state saved by run_zeroT.py --save-psi.
The chemical potential of a *finite* cluster is chosen inside the charge gap;
all quantities retain their finite-cluster definition. No analytic-continuation
claim is made. Requires QuSpin. The associated chi(0) routines are in
run_zeroT.py and store Lehmann/Krylov poles and weights.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal
from geometry import RubyPBC
from observables import apply_ops
from run_zeroT import create_hamiltonian, get_quspin, lowest_eigenpairs


def lanczos_matrix_element(H,source,braz,denominators,maxiter=80,tol=1e-11):
    """For each complex z, calculate <bra_i|(z +/- H)^-1|source>.
    For the addition part, denominator z + E0 + mu - H.
    For the removal part, denominator z - E0 + mu + H.
    denominators: list of (z, eta, sigma), denominator = z+eta+sigma*H.
    """
    norm=np.linalg.norm(source)
    result=np.zeros((len(denominators),braz.shape[1]),complex)
    if norm<tol:return result
    q=source/norm
    last=np.zeros_like(q);beta=0.
    diag=[];offs=[];vecs=[]
    for i in range(maxiter):
        vecs.append(q)
        v=H.dot(q)-beta*last
        a=float(np.vdot(q,v).real)
        v-=a*q
        # Reorthogonalize to control ghost eigenvalues in long Krylov runs.
        for old in vecs:v-=old*np.vdot(old,v)
        b=float(np.linalg.norm(v))
        diag.append(a)
        if b<tol or i==maxiter-1:break
        offs.append(b);last,q=q,v/b;beta=b
    T=np.diag(diag)
    if offs:
        T+=np.diag(offs,1)+np.diag(offs,-1)
    projections=braz.conj().T@np.column_stack(vecs)
    unit=np.zeros(len(diag),complex);unit[0]=norm
    for iw,(z,eta,sigma) in enumerate(denominators):
        y=np.linalg.solve((z+eta)*np.eye(len(diag))+sigma*T,unit)
        result[iw,:]=projections@y
    return result


def run(a):
    path=Path(a.input)
    model=json.loads((path/'geometry.json').read_text())
    g=RubyPBC(model['name'],**model['t'])
    V=float(model['V']);Nf=int(model['Nf'])
    dat=np.load(path/'ground_state.npz')
    psi=dat['psi']; states=dat['basis_states']; E0=float(dat['e0'])
    bs,_=get_quspin()
    basis_plus=bs(g.Ns,Nf=Nf+1)
    basis_minus=bs(g.Ns,Nf=Nf-1)
    Hplus=create_hamiltonian(g,V,basis_plus)
    Hminus=create_hamiltonian(g,V,basis_minus)
    Eplus=float(lowest_eigenpairs(Hplus,1)[0][0]);Eminus=float(lowest_eigenpairs(Hminus,1)[0][0])
    mu_add=Eplus-E0;mu_remove=E0-Eminus
    mu=(mu_add+mu_remove)/2 if a.mu is None else float(a.mu)
    if not (mu_remove <= mu <= mu_add):
        print('WARNING: mu is outside the finite-size addition/removal gap.')
    ns=g.Ns;orbs=list(range(6)) if not a.orbitals else a.orbitals
    fplus=np.column_stack([apply_ops(psi,states,basis_plus.states,ns,[('+',i)]) for i in orbs])
    fminus=np.column_stack([apply_ops(psi,states,basis_minus.states,ns,[('-',i)]) for i in orbs])
    omega=np.pi*(2*np.arange(a.nfreq)+1)/a.beta_reference
    z=1j*omega
    denplus=[(x,E0+mu,-1) for x in z]
    denminus=[(x,-E0+mu,+1) for x in z]
    G=np.zeros((len(z),len(orbs),len(orbs)),complex)
    for j in range(len(orbs)):
        G[:,:,j]+=lanczos_matrix_element(Hplus,fplus[:,j],fplus,denplus,maxiter=a.krylov)
    for i in range(len(orbs)):
        # hole part: G_ij = <c_j psi| resolvent |c_i psi>
        G[:,i,:]+=lanczos_matrix_element(Hminus,fminus[:,i],fminus,denminus,maxiter=a.krylov)
    np.savez_compressed(path/'green_iw.npz',G_iw=G,omega=omega,orbital_indices=orbs,
                        mu=mu,mu_add=mu_add,mu_remove=mu_remove,
                        E0=E0, Eplus=Eplus,Eminus=Eminus,
                        sign_convention='G(z)=plus resolvent + hole resolvent; z=i omega',
                        krylov_steps=a.krylov,beta_reference=a.beta_reference)
    print('G(iw) stored in',path/'green_iw.npz','matrix',G.shape,
          'chemical potential',mu,'finite gap',mu_add-mu_remove)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Directory from run_zeroT --save-psi')
    p.add_argument('--orbitals',type=int,nargs='*',default=None)
    p.add_argument('--mu',type=float,default=None)
    p.add_argument('--nfreq',type=int,default=20)
    p.add_argument('--beta-reference',type=float,default=100.)
    p.add_argument('--krylov',type=int,default=80)
    return p
if __name__=='__main__':run(parser().parse_args())