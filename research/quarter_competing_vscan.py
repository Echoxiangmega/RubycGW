#!/usr/bin/env python3
"""Quarter-filled ring3 V scan: M-point pseudospin SDW and Pfaffian pairing.

This is a CPU reference branch driver for the existing run_supercell_vscan.py
output tree. Its Slater/Pfaffian states share the seven bond-aware density
Jastrow channels, but do not include the fast Slater-only backflow correlator.
Run one seed (sdw, pf_s, pf_dplus, pf_dminus) per process. The usual collector
can combine its branch_scan.csv with the pre-existing normal/CO/LC branches.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'ed_benchmark'/'native_ed'),
              str(REPO/'ed_benchmark'/'vmc_zeroT'),str(REPO/'research')]
from geometry import make_ring3_supercell  # noqa: E402
from vmc_ansatz import jastrow_channels  # noqa: E402
from ruby_did_vmc import (bcs_pair_function,hopping_matrix,  # noqa: E402
                          pairing_source,orbital_pair_matrix,form_factors)
from quarter_sdw_vmc import local_fields  # noqa: E402

PF_FORMS={'pf_s':'s','pf_dplus':'d+','pf_dminus':'d-'}
SEEDS=('sdw',*PF_FORMS)


def exact_filling(l1,l2,filling):
    ns=18*l1*l2
    fraction=Fraction(filling)
    particles=ns*fraction
    if particles.denominator!=1:
        raise ValueError(f'Ns={ns}, filling={fraction}: Nf={particles} is not an integer')
    nf=int(particles)
    if not 0<nf<ns:
        raise ValueError('Require 0 < Nf < Ns')
    if l1%2 or l2%2:
        raise ValueError('Three M points require both ring3 repeat counts L1,L2 even')
    if nf%2:
        raise ValueError('This paired ansatz requires an even particle number')
    return ns,nf


def vtag(v):
    sign='m' if v<0 else 'p'
    digits=f'{abs(float(v)):.8f}'.rstrip('0').rstrip('.')
    return 'V_'+sign+digits.replace('.','p')


def v_values(a):
    if a.V_values:
        vals=sorted(set(float(x) for x in a.V_values))
    else:
        if a.V_min is None or a.V_max is None or a.V_step is None:
            raise ValueError('Specify --V-values or --V-min/--V-max/--V-step')
        if a.V_step<=0 or a.V_max<a.V_min:
            raise ValueError('Require V-step>0 and V-max>=V-min')
        n=int(math.floor((a.V_max-a.V_min)/a.V_step+1e-10))
        vals=[a.V_min+i*a.V_step for i in range(n+1)]
        if vals[-1]<a.V_max-1e-9:vals.append(a.V_max)
    return vals if a.direction=='up' else vals[::-1]


@lru_cache(maxsize=8)
def sdw_sources(g,sign):
    sx=local_fields(g,1.,0.,'stripe_xy',sign)
    sy=local_fields(g,1.,0.,'double_xy',sign)-sx
    sz=local_fields(g,0.,1.,'stripe_z',sign)
    return sx,sy,sz


@lru_cache(maxsize=128)
def sdw_orbitals(g,nf,mx,my,mz,sign):
    sx,sy,sz=sdw_sources(g,sign)
    h=hopping_matrix(g,0.)+mx*sx+my*sy+mz*sz
    return np.asarray(np.linalg.eigh(h)[1][:,:nf],complex)


@lru_cache(maxsize=128)
def pair_orbitals(g,mu,delta,form):
    h=hopping_matrix(g,mu)
    D=pairing_source(g,delta,form)
    return bcs_pair_function(h,D)


def pair_bonds_for_cell(g,ci):
    """Three E-singlet A-B pair bonds from one primitive cell."""
    M=orbital_pair_matrix()
    R=g.reps[ci]
    bonds=[]
    for dx,dy in ((0,0),(0,1),(-1,0)):
        cj,_=g.reduce((R[0]+dx,R[1]+dy))
        bonds.append([(6*ci+i,6*cj+3+j,M[i,j])
                      for i in range(3) for j in range(3) if abs(M[i,j])>1e-12])
    return bonds


def far_pair_cells(g):
    """A fixed, maximally separated A-triangle pair; distance is minimum image."""
    centers=np.asarray([g.positions[6*i:6*i+3].mean(axis=0)
                        for i in range(g.num_cells)])
    shifts=[g.B@np.asarray([x,y]) for x in (-1,0,1) for y in (-1,0,1)]
    distances=[min(np.linalg.norm(centers[j]-centers[0]+s) for s in shifts)
               for j in range(g.num_cells)]
    j=int(np.argmax(distances))
    return 0,j,float(distances[j])


def _remove(occ,site):
    k=int(np.searchsorted(occ,site))
    if k==len(occ) or occ[k]!=site:return None,0
    return np.delete(occ,k),(-1 if k%2 else 1)


def _add(occ,site):
    k=int(np.searchsorted(occ,site))
    if k<len(occ) and occ[k]==site:return None,0
    return np.insert(occ,k,site),(-1 if k%2 else 1)


def far_pair_estimator(trial,occ,logpsi,bonds_a,bonds_b):
    """C_uv=<P_(R,u)^dag P_(R',v)> for three bond directions.

    A fixed-N state has <P>=0. This equal-time, separated pair correlator
    measures coherence without treating the BdG seed delta as an observable.
    """
    C=np.zeros((3,3),complex)
    for v,bond_b in enumerate(bonds_b):
        for i,j,cb in bond_b:
            state,s1=_remove(occ,j)
            if state is None:continue
            state,s2=_remove(state,i)
            if state is None:continue
            for u,bond_a in enumerate(bonds_a):
                for k,l,ca in bond_a:
                    created,s3=_add(state,k)
                    if created is None:continue
                    created,s4=_add(created,l)
                    if created is None:continue
                    # O|occ> produces |created>. The local estimator of
                    # <psi|O|psi> uses the *conjugate* amplitude ratio.
                    z=trial.logpsi(created)
                    if np.isfinite(z.real):
                        C[u,v]+=ca*cb*s1*s2*s3*s4*np.exp(np.conj(z-logpsi))
    return C


class Trial:
    def __init__(self,g,nf,seed,params,jastrow):
        self.g,self.nf,self.seed,self.params=g,nf,seed,dict(params)
        self.mats=np.asarray(jastrow,float)
        self.v=np.asarray(params['v'],float)
        self.fill=nf/g.Ns
        self.rows=self.mats.sum(axis=2)
        self.constant=.5*self.fill**2*self.mats.sum(axis=(1,2))
        if seed=='sdw':
            sign=int(params['B_sign'])
            self.m_sources=sdw_sources(g,sign)
            self.Q=sdw_orbitals(g,nf,params['mx'],params['my'],params['mz'],sign)
            self.F=self.blocked=None
        else:
            self.F,self.blocked=pair_orbitals(
                g,params['mu'],params['delta'],PF_FORMS[seed])
            if self.blocked.shape[1]>nf or (nf+self.blocked.shape[1])%2:
                raise ValueError('Incompatible blocked BdG orbitals and particle parity')
            self.Q=None
            self.m_sources=None

    def features(self,occ):
        # Exact 0.5*(n-f)^T M (n-f), with M symmetric.
        return (.5*self.mats[:,occ[:,None],occ].sum(axis=(1,2))
                -self.fill*self.rows[:,occ].sum(axis=1)+self.constant)

    @staticmethod
    def logpf(matrix):
        a=np.array(matrix,complex,copy=True)
        n=len(a)
        if n%2:return -np.inf+0j
        ans=0j
        for k in range(0,n,2):
            p=k+1+int(np.argmax(np.abs(a[k,k+1:])))
            if abs(a[k,p])<1e-14:return -np.inf+0j
            if p!=k+1:
                a[[k+1,p],:]=a[[p,k+1],:]
                a[:,[k+1,p]]=a[:,[p,k+1]]
                ans+=1j*np.pi
            pivot=a[k,k+1]
            ans+=np.log(pivot)
            if k+2<n:
                u=a[k,k+2:].copy();v=a[k+1,k+2:].copy()
                a[k+2:,k+2:]+=(np.outer(v,u)-np.outer(u,v))/pivot
        return ans

    def logpsi(self,occ):
        occ=np.asarray(occ,int)
        if self.Q is not None:
            sign,logabs=np.linalg.slogdet(self.Q[occ,:])
            base=logabs+1j*np.angle(sign) if sign else -np.inf+0j
        else:
            if self.blocked.shape[1]:
                z=np.block([[self.F[np.ix_(occ,occ)],self.blocked[occ]],
                            [-self.blocked[occ].T,np.zeros((self.blocked.shape[1],
                                                             self.blocked.shape[1]))]])
            else:z=self.F[np.ix_(occ,occ)]
            base=self.logpf(z)
        return base+float(self.v@self.features(occ))

    def hop_cache(self,occ):
        """One inverse for all connected hops out of this configuration."""
        if self.Q is not None:a=self.Q[occ,:]
        elif self.blocked.shape[1]:
            a=np.block([[self.F[np.ix_(occ,occ)],self.blocked[occ]],
                        [-self.blocked[occ].T,np.zeros((self.blocked.shape[1],
                                                         self.blocked.shape[1]))]])
        else:a=self.F[np.ix_(occ,occ)]
        try:inverse=np.linalg.inv(a)
        except np.linalg.LinAlgError:inverse=None
        return inverse,self.features(occ)

    def hop_ratio(self,occ,a,b,cache=None):
        """Canonical amplitude ratio after moving an occupied a to empty b.

        For a Pfaffian, replacing occupied row/column k gives
        Pf(A')/Pf(A)=sum_j A'_{kj}(A^{-1})_{jk}. The final sign reorders the
        replaced orbital to ascending physical-site order.
        """
        k=int(np.searchsorted(occ,a))
        target=np.sort(np.r_[occ[occ!=a],b])
        parity=(-1 if np.count_nonzero((occ>min(a,b)) &
                                       (occ<max(a,b)))%2 else 1)
        if cache is None:cache=self.hop_cache(occ)
        inverse,old_features=cache
        if inverse is None:
            ratio=np.exp(self.logpsi(target)-self.logpsi(occ))
            return ratio,parity,target
        if self.Q is not None:
            basic=self.Q[b,:]@inverse[:,k]
        else:
            v=np.r_[self.F[b,occ],self.blocked[b]]
            v[k]=0.
            basic=v@inverse[:,k]
        exponent=float(self.v@(self.features(target)-old_features))
        ratio=parity*basic*np.exp(exponent)
        if not (np.isfinite(ratio.real) and np.isfinite(ratio.imag)):
            ratio=np.exp(self.logpsi(target)-self.logpsi(occ))
        if not (np.isfinite(ratio.real) and np.isfinite(ratio.imag)):
            raise FloatingPointError('Non-finite one-particle move ratio')
        return ratio,parity,target

    def local_energy(self,occ,logpsi,V):
        occupied=np.zeros(self.g.Ns,bool);occupied[occ]=True
        tri=np.bincount(occ//3,minlength=self.g.Ns//3)
        energy=complex(V*np.sum(tri*(tri-1)//2))
        cache=self.hop_cache(occ)
        for e in self.g.edges:
            i,j=int(e.i),int(e.j)
            if occupied[i]==occupied[j]:continue
            a,b=(i,j) if occupied[i] else (j,i)
            ratio,sign,_=self.hop_ratio(occ,a,b,cache)
            energy+=self.g.t[e.kind]*sign*ratio
        return energy

    def m_point_onebody(self,occ,logpsi):
        """Local estimators of the three *measured* M-point order sources."""
        if self.m_sources is None:return None
        filled=np.zeros(self.g.Ns,bool);filled[occ]=True
        cache=self.hop_cache(occ)
        values=[]
        for source in self.m_sources:
            value=complex(np.sum(source[occ,occ]))
            for a in occ:
                for b in np.flatnonzero(np.abs(source[:,a])>1e-12):
                    if filled[b] or b==a:continue
                    ratio,sign,_=self.hop_ratio(occ,a,b,cache)
                    value+=source[b,a]*sign*ratio
            values.append(value)
        return values


def sample(trial,V,*,warmup,samples,interval,rng,measure=True,orders=False,
           pair_samples=0):
    g=trial.g;nf=trial.nf
    for _ in range(2000):
        occ=np.sort(rng.choice(g.Ns,nf,replace=False))
        logpsi=trial.logpsi(occ)
        if np.isfinite(logpsi.real):break
    else:raise RuntimeError('Could not start chain in trial-state support')
    cache=trial.hop_cache(occ)
    pair_info=None
    if pair_samples:
        ci,cj,dist=far_pair_cells(g)
        pair_info=(pair_bonds_for_cell(g,ci),pair_bonds_for_cell(g,cj),ci,cj,dist)
    pair_values=[]
    pair_stride=max(1,samples//pair_samples) if pair_samples else 0
    values=[];features=[];measured_orders=[];accepted=attempted=0
    for sweep in range(warmup+samples*interval):
        for _ in range(nf):
            filled=np.zeros(g.Ns,bool);filled[occ]=True
            a=int(rng.choice(occ));b=int(rng.choice(np.flatnonzero(~filled)))
            ratio,_,target=trial.hop_ratio(occ,a,b,cache);attempted+=1
            if abs(ratio)>0 and np.log(max(rng.random(),1e-300))<min(
                    0.,2*np.log(abs(ratio))):
                occ=target
                logpsi+=np.log(ratio)
                cache=trial.hop_cache(occ)
                accepted+=1
        if sweep>=warmup and (sweep-warmup)%interval==0:
            isample=(sweep-warmup)//interval
            if measure:values.append(trial.local_energy(occ,logpsi,V))
            if orders and trial.m_sources is not None:
                measured_orders.append(trial.m_point_onebody(occ,logpsi))
            if pair_info is not None and isample%pair_stride==0 and len(pair_values)<pair_samples:
                pair_values.append(far_pair_estimator(trial,occ,logpsi,
                                      pair_info[0],pair_info[1]))
            features.append(trial.features(occ))
    feats=np.asarray(features,float)
    out={'acceptance':accepted/attempted,'samples':samples,
         'features':feats}
    if measure:
        y=np.asarray(values,complex)
        if not np.all(np.isfinite(y)):
            raise FloatingPointError('Non-finite local energy in MC evaluation')
        blocks=np.array_split(y.real,max(2,min(20,len(y)//8)))
        bm=np.array([x.mean() for x in blocks])
        out.update(E=float(y.real.mean()),
                   mc_E_error=float(bm.std(ddof=1)/np.sqrt(len(bm))),
                   imag_E_check=float(abs(y.imag.mean())),
                   local_energies=y)
    if measured_orders:
        o=np.asarray(measured_orders,complex)
        ntri=trial.g.Ns//3
        m=o.real.mean(axis=0)/ntri
        parts=np.array_split(o.real,max(2,min(20,len(o)//8)))
        block_means=np.asarray([b.mean(axis=0)/ntri for b in parts])
        err=block_means.std(axis=0,ddof=1)/np.sqrt(len(parts))
        out['M_pseudospin_x']=float(m[0])
        out['M_pseudospin_y']=float(m[1])
        out['M_pseudospin_z']=float(m[2])
        out['M_pseudospin_x_error']=float(err[0])
        out['M_pseudospin_y_error']=float(err[1])
        out['M_pseudospin_z_error']=float(err[2])
        out['M_pseudospin_xy_amplitude']=float(np.hypot(m[0],m[1]))
        out['M_pseudospin_z_amplitude']=float(abs(m[2]))
        out['M_order_imag_check']=float(np.max(abs(o.imag.mean(axis=0)))/ntri)
    if pair_values:
        matrices=np.asarray(pair_values)
        for form,key in (('s','s'),('d+','dplus'),('d-','dminus')):
            weights=form_factors(form)
            # Delta_R=sum_u f_u P_Ru/sqrt(3); C_uv=<P_Ru^dag P_R'v>.
            z=np.einsum('u,nuv,v->n',weights.conj(),matrices,weights)/3.
            parts=np.array_split(z,max(2,min(12,len(z)//8)))
            means=np.asarray([x.mean() for x in parts])
            out[f'pair_far_{key}_real']=float(z.real.mean())
            out[f'pair_far_{key}_imag']=float(z.imag.mean())
            out[f'pair_far_{key}_abs']=float(abs(z.mean()))
            out[f'pair_far_{key}_error']=float(np.sqrt(
                means.real.var(ddof=1)+means.imag.var(ddof=1))/np.sqrt(len(means)))
        out['pair_far_samples']=len(matrices)
        out['pair_far_cells']=[int(pair_info[2]),int(pair_info[3])]
        out['pair_far_distance']=float(pair_info[4])
    return out


def initialize(seed,amplitude):
    v=[0.]*7
    if seed=='sdw':
        a=amplitude
        return [dict(mx=x,my=y,mz=z,B_sign=s,v=v)
                for s in (+1,-1) for x,y,z in
                ((a,0.,0.),(0.,a,0.),(0.,0.,a),(a,a,0.),
                 (a,a,a),(a,a,-a))]
    return [dict(delta=d,mu=mu,v=v)
            for d in (max(.01,amplitude/2),amplitude,2*amplitude)
            for mu in (-1.3,-.9)]


def optimize(g,nf,mats,seed,V,initial,options,rng):
    """Bounded stochastic coordinate search and an SR update of all 7 Jastrows."""
    params=[dict(x) for x in initial]
    if seed=='sdw':
        params+=initialize(seed,options.seed_amplitude)
    else:
        params+=initialize(seed,options.seed_amplitude)
    scores=[]
    for index,p in enumerate(params):
        try:
            trial=Trial(g,nf,seed,p,mats)
            s=sample(trial,V,warmup=options.burn,
                     samples=options.train_samples,interval=options.interval,
                     rng=np.random.default_rng(options.rng_seed+index+100000*int(V*100)))
            scores.append((s['E'],p))
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError):continue
    if not scores:raise RuntimeError('All initial candidates failed')
    best=min(scores,key=lambda x:x[0])[1]
    for step in range(options.refine_steps):
        trial=Trial(g,nf,seed,best,mats)
        s=sample(trial,V,warmup=options.burn,
                 samples=options.train_samples,interval=options.interval,rng=rng)
        feats=s['features'];energy=s['local_energies'];f=feats-feats.mean(axis=0)
        cov=(f.T@f)/len(f)
        force=2*np.real((f.T@energy)/len(f))
        shift=max(.1,float(np.trace(cov)/len(cov))*.02)
        v=np.clip(np.asarray(best['v'])-.05*np.linalg.solve(
            cov+shift*np.eye(len(cov)),force),-1.5,1.5)
        proposed={**best,'v':v.tolist()}
        # Sweep both structural directions rather than freezing the seed
        # orientation. The winning component may rotate from xy to z.
        if seed=='sdw':
            key=('mx','my','mz')[step%3]
            increment=options.seed_amplitude/(1+step//3)
            candidates=[proposed,{**proposed,key:best[key]+increment},
                        {**proposed,key:best[key]-increment}]
            if step%3==0:candidates.append({**proposed,'B_sign':-best['B_sign']})
        else:
            key=('delta','mu')[step%2]
            inc=(.05 if key=='delta' else .2)/(1+step//2)
            candidates=[proposed,{**proposed,key:best[key]+inc},
                        {**proposed,key:max(.001,best[key]-inc)
                         if key=='delta' else best[key]-inc}]
        results=[]
        for index,p in enumerate(candidates):
            try:
                t=Trial(g,nf,seed,p,mats)
                # Common RNG stream across candidates helps make the local
                # comparison reproducible; errors still require remeasurement.
                s0=sample(t,V,warmup=options.burn,
                          samples=options.train_samples,interval=options.interval,
                          rng=np.random.default_rng(options.rng_seed+1000*step+index))
                results.append((s0['E'],p))
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError):continue
        if results:best=min(results,key=lambda x:x[0])[1]
    final=Trial(g,nf,seed,best,mats)
    result=sample(final,V,warmup=options.eval_burn,
                  samples=options.eval_samples,interval=options.interval,
                  rng=np.random.default_rng(options.rng_seed+97333+int(V*1000)),
                  orders=(seed=='sdw'),pair_samples=options.pair_samples)
    row={k:v for k,v in result.items() if k not in ('features','local_energies')}
    row['E']=row['mc_E']=row.pop('E')
    row.update(best)
    return row


def write_rows(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--L1',type=int,required=True);p.add_argument('--L2',type=int,required=True)
    p.add_argument('--filling',default='1/4');p.add_argument('--seed',choices=SEEDS,required=True)
    p.add_argument('--t',type=float,default=1.);p.add_argument('--t1',type=float,default=-.5)
    p.add_argument('--t2',type=float,default=.5)
    p.add_argument('--V-values',nargs='*',type=float)
    p.add_argument('--V-min',type=float);p.add_argument('--V-max',type=float)
    p.add_argument('--V-step',type=float)
    p.add_argument('--direction',choices=('up','down'),default='up')
    p.add_argument('--mode',choices=('continuation','independent'),default='continuation')
    p.add_argument('--seed-amplitude',type=float,default=.08)
    p.add_argument('--burn',type=int,default=30);p.add_argument('--train-samples',type=int,default=512)
    p.add_argument('--eval-burn',type=int,default=120);p.add_argument('--eval-samples',type=int,default=4096)
    p.add_argument('--interval',type=int,default=2);p.add_argument('--refine-steps',type=int,default=9)
    p.add_argument('--pair-samples',type=int,default=64,
                   help='Sparse final-chain samples of separated s/d pair correlations')
    p.add_argument('--rng-seed',type=int,default=2026)
    p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--output',required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if a.L1<=0 or a.L2<=0:raise ValueError('L1,L2 must be positive')
    if not np.isclose(a.t1,-a.t2,atol=1e-12):
        raise ValueError('These projected singlet Pf seeds require t1=-t2')
    ns,nf=exact_filling(a.L1,a.L2,a.filling)
    values=v_values(a)
    if (a.train_samples<16 or a.eval_samples<16 or a.interval<1 or
        a.refine_steps<0 or a.pair_samples<0 or a.pair_samples>a.eval_samples):
        raise ValueError('Require >=16 samples, interval>=1, and refine-steps>=0')
    root=Path(a.output).expanduser().resolve()
    branch=root/f'L{a.L1}x{a.L2}'/f'seed_{a.seed}'/f'{a.mode}_{a.direction}'
    print(f'[quarter] {a.seed}: Ns={ns}, Nf={nf}, V={values}; branch={branch}',flush=True)
    if a.dry_run:return
    g=make_ring3_supercell(a.L1,a.L2,t=a.t,t1=a.t1,t2=a.t2)
    labels,_,mats,_=jastrow_channels(g,mode='bond_aware',n_shells=4)
    if len(labels)!=7:raise RuntimeError(f'Unexpected Jastrow schema: {labels}')
    branch.mkdir(parents=True,exist_ok=True)
    settings={**vars(a),'Ns':ns,'Nf':nf,
       'ansatz_schema_version':2,
       'wavefunction':'Slater or bordered Pfaffian times 7-channel density Jastrow',
       'backflow_enabled':False}
    settings_path=branch/'scan_settings.json'
    if settings_path.exists():
        old=json.loads(settings_path.read_text())
        for k in ('L1','L2','seed','filling','Nf','t','t1','t2',
                  'ansatz_schema_version',
                  'seed_amplitude','rng_seed','burn','eval_burn','interval',
                  'train_samples','eval_samples','refine_steps','pair_samples'):
            if old.get(k)!=settings[k]:
                raise ValueError(f'Incompatible saved branch setting {k}: '
                                 f'{old.get(k)!r} vs {settings[k]!r}')
    settings_path.write_text(json.dumps(settings,indent=2))
    rows=[];previous=None;rng=np.random.default_rng(a.rng_seed)
    for index,V in enumerate(values):
        point=branch/vtag(V);point.mkdir(exist_ok=True)
        cached=point/'competing_summary.json'
        if a.resume and cached.exists():
            row=json.loads(cached.read_text())
        else:
            starts=[] if previous is None or a.mode=='independent' else [previous]
            row=optimize(g,nf,mats,a.seed,V,starts,a,rng)
            if a.seed=='sdw':
                xy=row['M_pseudospin_xy_amplitude']
                z=row['M_pseudospin_z_amplitude']
                errxy=np.hypot(row['M_pseudospin_x_error'],
                               row['M_pseudospin_y_error'])
                errz=row['M_pseudospin_z_error']
                if abs(xy-z)<2*np.hypot(errxy,errz):
                    label='SDW_M_orientation_uncertain'
                else:label='SDW_M_LC' if z>xy else 'SDW_M_CO'
            else:
                label=('Pf_s_candidate' if a.seed=='pf_s'
                       else 'Pf_d_chiral_candidate')
            row.update(seed=a.seed,branch_seed=a.seed,scan_V=V,scan_index=index,
                scan_mode=a.mode,scan_direction=a.direction,L1=a.L1,L2=a.L2,
                Ns=ns,Nf=nf,filling=nf/ns,scan_Ns=ns,scan_Nf=nf,
                scan_filling=nf/ns,method='Raw-Pfaffian-Jastrow' if a.seed in PF_FORMS
                else 'Raw-SDW-Slater-Jastrow',backflow_enabled=False,
                correlator='bond_aware_J7',phase_label=label)
            cached.write_text(json.dumps(row,indent=2))
        previous={k:row[k] for k in (('mx','my','mz','B_sign','v')
                  if a.seed=='sdw' else ('delta','mu','v'))}
        rows.append(row);write_rows(branch/'branch_scan.csv',rows)
        print(f'[quarter] V={V:g} E={row["mc_E"]:.8f} +/- {row["mc_E_error"]:.5f}'
              f' parameters={previous}',flush=True)


if __name__=='__main__':main()
