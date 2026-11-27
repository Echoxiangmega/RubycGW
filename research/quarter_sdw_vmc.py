#!/usr/bin/env python3
"""Exact 24-site evaluation of structured pseudospin SDW Slater-Jastrow trials.

Run from the workspace root, after the existing 2x2 ED/VMC benchmark:
  OPENBLAS_NUM_THREADS=1 python repo/research/quarter_sdw_vmc.py --V 1.5

The physical model is spinless.  The two chiral triangle E orbitals play the
role of a pseudospin.  Four primitive cells of the 2x2 torus give the three
commensurate M sign patterns (-1)^x, (-1)^y, and (-1)^(x+y).  The latter three
coupled to Sx,Sy,Sz give a tetrahedral texture when mx=my=mz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO=Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT=REPO.parent
sys.path[:0]=[str(REPO/'ed_benchmark'/'native_ed'),
              str(REPO/'ed_benchmark'/'vmc_zeroT'),
              str(REPO/'research')]
from core import FixedHamiltonian  # noqa: E402
from geometry import RubyPBC  # noqa: E402
from vmc_ansatz import jastrow_channels  # noqa: E402
from ruby_did_vmc import hopping_matrix  # noqa: E402
from run_projection_comparison import build_ed_sectors  # noqa: E402
from symmetry_projection import project_to_characters  # noqa: E402

OMEGA=np.exp(2j*np.pi/3)
W=np.array([[1,1],[OMEGA,OMEGA.conjugate()],
            [OMEGA**2,OMEGA]],complex)/np.sqrt(3)
SX=np.array([[0,1],[1,0]],complex)
SY=np.array([[0,-1j],[1j,0]],complex)
SZ=np.diag([1.,-1.]).astype(complex)
PAULI=(SX,SY,SZ)
PATTERNS=('stripe_xy','stripe_z','double_xy','tetra')


def local_fields(g, mxy, mz, pattern, sublattice_sign):
    """Hermitian 24x24 auxiliary field; B pseudospin is aligned by sigma_y."""
    field=np.zeros((g.Ns,g.Ns),complex)
    for ci,(x,y) in enumerate(g.reps):
        eta=np.asarray([(-1)**x,(-1)**y,(-1)**(x+y)])
        coeff=np.array([mxy*eta[0],mxy*eta[1],mz*eta[2]],float)
        if pattern=='stripe_xy':coeff[1:]=0.
        if pattern=='stripe_z':coeff[:2]=0.
        if pattern=='double_xy':coeff[2]=0.
        for sub in (0,1):
            rotate=np.eye(2) if sub==0 else SY
            sign=1. if sub==0 else float(sublattice_sign)
            local=sum((sign*c)*(W@rotate@p@rotate@W.conj().T)
                      for c,p in zip(coeff,PAULI))
            sl=slice(6*ci+3*sub,6*ci+3*sub+3)
            field[sl,sl]+=local
    if np.linalg.norm(field-field.conj().T)>1e-11:
        raise RuntimeError('SDW field is not Hermitian')
    return field


class ExactTrial:
    def __init__(self,V):
        self.g=RubyPBC('2x2',t=1.,t1=-.5,t2=.5)
        self.H=FixedHamiltonian(self.g,6,V,method='sparse')
        bits=((self.H.states[:,None] >> (self.g.Ns-1-
                   np.arange(self.g.Ns,dtype=np.uint64))) & 1).astype(bool)
        self.occ=np.where(bits)[1].reshape(-1,6)
        labels,_,mats,_=jastrow_channels(self.g,mode='bond_aware',n_shells=4)
        assert len(labels)==7
        dn=bits.astype(float)-.25
        self.features=np.empty((len(self.occ),len(mats)),float)
        for k,M in enumerate(mats):
            i,j=np.where(np.triu(M,1)!=0)
            self.features[:,k]=(dn[:,i]*dn[:,j]*M[i,j]).sum(axis=1)
            self.features[:,k]+=.5*(dn**2*M.diagonal()).sum(axis=1)
        self.h0=hopping_matrix(self.g,0.)

    def slater(self,mxy,mz,pattern,sign):
        h=self.h0+local_fields(self.g,mxy,mz,pattern,sign)
        e,Q=np.linalg.eigh(h)
        amp=np.linalg.det(Q[:,:6][self.occ])
        return amp,float(e[6]-e[5])

    def jastrow_energy(self,v,amp,return_state=False):
        lj=self.features@v
        psi=amp*np.exp(lj-lj.max())
        norm=np.vdot(psi,psi).real
        if norm<1e-99:
            raise ValueError('Vanishing SDW trial-state norm')
        psi/=np.sqrt(norm)
        Hp=self.H.operator@psi
        energy=float(np.vdot(psi,Hp).real)
        if return_state:return energy,psi
        grad=2*np.real((np.conj(psi)*(Hp-energy*psi))@self.features)
        return energy,grad

    def fit(self,mxy,mz,pattern,sign,starts):
        amp,gap=self.slater(mxy,mz,pattern,sign)
        fits=[minimize(lambda v:self.jastrow_energy(v,amp),v0,jac=True,
                       method='L-BFGS-B',bounds=[(-2.,2.)]*7,
                       options={'maxiter':75,'ftol':1e-11,'gtol':1e-7})
              for v0 in starts]
        best=min(fits,key=lambda x:x.fun)
        return {'E':float(best.fun),'mxy':float(mxy),'mz':float(mz),
                'pattern':pattern,'B_sign':int(sign),'gap_aux':gap,
                'v':best.x.tolist(),'optimizer_success':bool(best.success)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--V',type=float,required=True)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA_ROOT,
                   help='Parent directory of existing vmc_2x2_V*_refined benchmarks')
    a=p.parse_args()
    tag=str(a.V).replace('.','p')
    trial=ExactTrial(a.V)
    prev=a.data_root/f'vmc_2x2_V{tag}_refined'
    normal=np.load(prev/'slater_jastrow__normal'/'final_state.npz')
    ed=np.load(prev/'ed'/'final_state.npz')
    assert np.array_equal(trial.H.states,normal['native_states'])
    assert np.array_equal(trial.H.states,ed['native_states'])
    start=np.asarray(normal['variational_v'],float)
    rows=[]
    grid={'stripe_xy':[(x,0.) for x in (.08,.2,.45,.8)],
          'stripe_z':[(0.,z) for z in (.08,.2,.45,.8)],
          'double_xy':[(x,0.) for x in (.08,.2,.45,.8)],
          'tetra':[(x,z) for x in (.08,.2,.45,.8)
                   for z in (.08,.2,.45,.8)]}
    for pattern in PATTERNS:
        for sign in (1,-1):
            for mx,mz in grid[pattern]:
                row=trial.fit(mx,mz,pattern,sign,[start])
                rows.append(row)
                print(a.V,pattern,sign,mx,mz,row['E'],flush=True)
    best_per={}
    for pattern in PATTERNS:
        for sign in (1,-1):
            pair=(pattern,sign)
            best_per[pair]=min((r for r in rows if
                                (r['pattern'],r['B_sign'])==pair),
                               key=lambda r:r['E'])
    # Local refinement keeps the original branch and exactly reoptimizes J.
    for (pattern,sign),coarse in best_per.items():
        x0=[coarse['mxy']] if pattern in ('stripe_xy','double_xy') else (
             [coarse['mz']] if pattern=='stripe_z' else
             [coarse['mxy'],coarse['mz']])
        def objective(x):
            mx=0. if pattern=='stripe_z' else float(x[0])
            mz=float(x[0]) if pattern=='stripe_z' else (
                 0. if pattern!='tetra' else float(x[1]))
            if mx<.002 or mx>1.4:
                if pattern!='stripe_z':return 1e4
            if mz<.002 or mz>1.4:
                if pattern in ('stripe_z','tetra'):return 1e4
            row=trial.fit(mx,mz,pattern,sign,[np.asarray(coarse['v'])])
            rows.append(row)
            return row['E']
        minimize(objective,x0,method='Nelder-Mead',
                 options={'maxiter':36,'xatol':.004,'fatol':1e-6})
    group,_,sectors=build_ed_sectors(trial.g,trial.H,
        np.asarray(ed['psi_ground_manifold']),1e-8,
        validate_commutator=False)
    result={'V':a.V,'ed_energy':float(ed['ground_energy']),'candidates':[],
            'all_grid_and_refined':sorted(rows,key=lambda r:r['E'])}
    out=a.output or a.data_root/f'sdw_V{tag}'
    out.mkdir(exist_ok=True)
    for pattern in PATTERNS:
        for sign in (1,-1):
            best=min((r for r in rows if
                (r['pattern'],r['B_sign'])==(pattern,sign)),key=lambda r:r['E'])
            amp,_=trial.slater(best['mxy'],best['mz'],pattern,sign)
            E,psi=trial.jastrow_energy(np.asarray(best['v']),amp,True)
            best={**best,'E':E,'fidelity_ground':float(np.linalg.norm(
                ed['psi_ground_manifold'].conj().T@psi)**2)}
            projections=[]
            for sec in sectors:
                proj,weight=project_to_characters(group,sec['characters'],psi)
                if proj is not None:
                    ep=float(np.vdot(proj,trial.H.operator@proj).real)
                    projections.append({'weight':weight,'E':ep,
                       'fidelity_ground':float(np.linalg.norm(
                         ed['psi_ground_manifold'].conj().T@proj)**2),
                       'c3':sec['quantum_numbers']['c3']['label']})
            best['PAV']=projections
            result['candidates'].append(best)
            if pattern=='tetra':
                np.savez_compressed(out/f'final_state_tetra_B{sign:+d}.npz',
                    psi_native=psi,native_states=trial.H.states,
                    variational_v=np.asarray(best['v']),
                    variational_mxy=best['mxy'],variational_mz=best['mz'],
                    E_exact=E)
            print('BEST',json.dumps(best),flush=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print('SAVED',out/'summary.json',flush=True)


if __name__=='__main__':main()
