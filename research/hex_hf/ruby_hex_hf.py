#!/usr/bin/env python3
"""Symmetry-reduced spinless Ruby lattice Hartree-Fock, no superconductivity.

Geometry is reconstructed exactly (in triangle-edge units) from the uploaded
'ring3_pbc_verified.svg' six-triangle ring: 18 sites, 36 ED-coloured bonds,
6 crossing the hexagonal periodic boundary. The alternative hex2x2 period is
built from the SAME reconstructed infinite Ruby lattice (24 sites, 48 bonds).

The reference 'ruby_hfb.py' standalone template differs from this visual
geometry in some cell-offset conventions. Compare against native ED geometry
before asserting Hamiltonian identity; `--check-ed` helps when available.

C6 rotates t1 <-> t2, hence enforcing C6 requires t1=t2 and Vt1=Vt2.
C3 leaves t1 and t2 separately invariant. C6 also exchanges the two triangle
orientations: to allow alternating A/B currents, C3 is necessary.

Definitions:
  chi_b = <c_(j+R)^dagger c_i> for directed bond b: i -> j+R,
  h_ij(k) = [t_b - V_b chi_b] exp(+ik.R),
  J_(i->j) = 2 Im(t_b * chi_b) (hbar=1) for real hoppings,
  E = sum_b [2 Re(t_b chi_b*) + V_b(n_i n_j-|chi_b|^2)].
All k vectors use supercell reciprocal fractional coordinates.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import csv
import itertools
import json
from pathlib import Path
import math
import warnings
import numpy as np
from scipy.special import expit, xlogy
from scipy.optimize import brentq

SQ3 = math.sqrt(3.0)
# A triangle of side one, with origin at the central hexagon of the ring.
ORBITAL = np.array([
    [0.,-1.],[-.5,-1.-SQ3/2],[+.5,-1.-SQ3/2],
    [0.,+1.],[-.5,+1.+SQ3/2],[+.5,+1.+SQ3/2],
],float)
A = np.array([[-(1+SQ3)/2,(1+SQ3)/2],[(3+SQ3)/2,(3+SQ3)/2]],float)
ROT_CELL = np.array([[1,1],[-1,0]],int)  # CCW +60 degree on primitive coordinates
ROT_ORB = [4,3,5,2,1,0]
ROT_ORB_SHIFT = np.array([[-1,0],[-1,0],[-1,0],[1,0],[1,0],[1,0]],int)
# Unique 12 directed primitive bonds reconstructed from the ring3 SVG.
# These should NOT silently be replaced by the earlier 'physical-label' table.
PRIMITIVE_BONDS = [
    (0,1,(0,0),'t'), (0,2,(0,0),'t'), (2,1,(0,0),'t'),
    (3,4,(0,0),'t'), (3,5,(0,0),'t'), (4,5,(0,0),'t'),
    (1,4,(-1,-1),'t1'), (5,0,(0,1),'t1'), (2,3,(-1,0),'t1'),
    (3,1,(0,1),'t2'), (2,5,(-1,-1),'t2'), (0,4,(-1,0),'t2')]
TRIANGLE_PATHS = [(0,1,2),(3,5,4)]  # geometric, counterclockwise, A/B
PERIODS = {
    'ring3': np.array([[-1,2],[-1,-1]],int), # det=3; hex Wigner-Seitz boundary
    'hex2x2': np.array([[2,0],[0,2]],int),  # det=4; 3 M vectors, 24-site model
}
REPS = {
    'ring3': [(0,0),(1,0),(0,1)],
    'hex2x2': [(0,0),(1,0),(0,1),(1,1)],
}

@dataclass(frozen=True)
class Bond:
    i: int
    j: int
    wind: tuple[int,int]
    kind: str
    t: float
    V: float
    prim_delta: tuple[int,int]
    cell: int


def _exact_quotient(M, delta):
    """Return exact integer M^{-1} delta, or None if outside sublattice."""
    inv = np.linalg.solve(M,np.asarray(delta,float))
    v = np.rint(inv).astype(int)
    return v if np.max(abs(inv-v))<1e-9 else None


def _site_orbits(permutation):
    visited=set();out=[]
    for i in range(len(permutation)):
        if i in visited:continue
        orbit=[];j=i
        while j not in visited:
            visited.add(j);orbit.append(j);j=permutation[j]
        if j!=i:raise AssertionError('site permutation is not cyclic by orbits')
        out.append(orbit)
    return out


class RubyHexHF:
    def __init__(self, period='ring3',symmetry='none',nk=(6,6),
                 t=1.,t1=.5,t2=.5,Vtri=0.,Vt1=0.,Vt2=0.):
        if period not in PERIODS:raise ValueError('period must be ring3 or hex2x2')
        if symmetry not in ['c6','c3','none']:raise ValueError('symmetry must be c6, c3, or none')
        if min(nk)<1:raise ValueError('Invalid k mesh')
        if symmetry!='none' and nk[0]!=nk[1]:
            raise ValueError('C3/C6 needs a symmetry-compatible square k mesh: --nk N N')
        if symmetry=='c6' and (abs(t1-t2)>1e-12 or abs(Vt1-Vt2)>1e-12):
            raise ValueError('C6 maps t1 <-> t2. Set t1=t2 and Vt1=Vt2, '
                             'or choose --symmetry c3.')
        self.period=period;self.symmetry=symmetry;self.M=PERIODS[period].copy()
        self.reps=np.array(REPS[period],dtype=int);self.ncell=len(self.reps)
        self.N=6*self.ncell;self.kmesh=tuple(nk);self.Nk=int(nk[0]*nk[1])
        self.nk=int(nk[0]);self.mk=int(nk[1]);self.K=np.array(
            [(x/self.nk,y/self.mk) for x in range(self.nk) for y in range(self.mk)],float)
        self.tmap={'t':float(t),'t1':float(t1),'t2':float(t2)}
        self.Vmap={'t':float(Vtri),'t1':float(Vt1),'t2':float(Vt2)}
        self.positions=np.zeros((self.N,2),float)
        self.image_shift=np.zeros((self.N,2),dtype=int)
        S=A@self.M
        for c,r in enumerate(self.reps):
            for orb in range(6):
                i=6*c+orb
                if period=='ring3':
                    # Matches the chosen 18 sites in the uploaded SVG exactly.
                    if c in (1,2) and orb>=3:self.image_shift[i]=[1,0]
                else:
                    # Choose a Wigner-Seitz representative about the central hexagon.
                    opts=[]
                    for p in itertools.product(range(-2,3),repeat=2):
                        pos=ORBITAL[orb]+A@r+S@p
                        opts.append((float(np.dot(pos,pos)),abs(p[0])+abs(p[1]),p,pos))
                    _,_,u,pos=min(opts,key=lambda q:(round(q[0],10),q[1],q[2]))
                    self.image_shift[i]=u
                self.positions[i]=ORBITAL[orb]+A@r+S@self.image_shift[i]
        self.S=S
        # Fold arbitrary primitive positions into the finite Bravais quotient.
        self.bonds=[]
        for c,r in enumerate(self.reps):
            for i,j,delta,kind in PRIMITIVE_BONDS:
                abs_dest=r+np.asarray(delta,int)
                matches=[]
                for cp,rp in enumerate(self.reps):
                    shift=_exact_quotient(self.M,abs_dest-rp)
                    if shift is not None:matches.append((cp,shift))
                if len(matches)!=1:raise AssertionError(f'Bad supercell folding: {r} {delta}')
                cp,shift=matches[0]
                ib=6*c+i; jb=6*cp+j
                wind=shift+self.image_shift[ib]-self.image_shift[jb]
                self.bonds.append(Bond(ib,jb,tuple(map(int,wind)),kind,
                                      self.tmap[kind],self.Vmap[kind],tuple(delta),c))
        self.B=len(self.bonds)
        self.ii=np.array([b.i for b in self.bonds],int)
        self.jj=np.array([b.j for b in self.bonds],int)
        self.winds=np.array([b.wind for b in self.bonds],int)
        self.tb=np.array([b.t for b in self.bonds],float)
        self.Vb=np.array([b.V for b in self.bonds],float)
        self.ph=np.exp(2j*np.pi*self.K@self.winds.T) # shape Nk,B
        self.H0=self._h0()
        self.triangles=self._triangles()
        self._build_symmetry()
        self._validate()

    def _h0(self):
        H=np.zeros((self.Nk,self.N,self.N),complex)
        for b in range(self.B):
            i=self.ii[b];j=self.jj[b];z=self.tb[b]*self.ph[:,b]
            H[:,i,j]+=z; H[:,j,i]+=z.conj()
        return H

    def _triangles(self):
        lookup={}
        for b,link in enumerate(self.bonds):
            key=(link.i,link.j,link.wind)
            lookup[key]=(b,1)
            key_rev=(link.j,link.i,tuple(-x for x in link.wind))
            lookup[key_rev]=(b,-1)
        triangles=[]
        for c in range(self.ncell):
            for which,path in enumerate(TRIANGLE_PATHS):
                vertices=[6*c+p for p in path]
                orient=[]
                for i,j in zip(vertices,vertices[1:]+vertices[:1]):
                    expected=tuple(map(int,self.image_shift[i]-self.image_shift[j]))
                    match=lookup.get((i,j,expected))
                    if match is None:raise AssertionError(f'Broken CCW triangular path {i}->{j}')
                    orient.append(match)
                triangles.append((c,which,orient))
        return triangles

    def _site_rotate_once(self,i):
        c,a=divmod(i,6)
        new_a=ROT_ORB[a]
        absolute=ROT_CELL@(self.reps[c]+self.M@self.image_shift[i])+ROT_ORB_SHIFT[a]
        matches=[]
        for cp,rp in enumerate(self.reps):
            shift=_exact_quotient(self.M,absolute-rp)
            if shift is not None:
                ip=6*cp+new_a;wind=shift-self.image_shift[ip]
                matches.append((ip,wind))
        if len(matches)!=1:raise AssertionError('C6 site action failed')
        ip,w=matches[0]
        # Real-space accuracy check in geometric coordinates.
        Rot=np.array([[.5,-SQ3/2],[SQ3/2,.5]])
        assert np.linalg.norm(self.positions[ip]+self.S@w-Rot@self.positions[i])<1e-8
        return ip,w

    def _rotate_map(self,power):
        # Effect of one +60 degree rotation in supercell coordinates.
        D=np.rint(np.linalg.solve(self.M,ROT_CELL@self.M)).astype(int)
        if not np.array_equal(self.M@D,ROT_CELL@self.M):
            raise AssertionError('Supercell is not C6 invariant')
        powerD=np.linalg.matrix_power(D,power)
        perm=np.zeros(self.N,dtype=int);sw=np.zeros((self.N,2),dtype=int)
        for i in range(self.N):
            cur=i;wind=np.zeros(2,int)
            for _ in range(power):
                nxt,u=self._site_rotate_once(cur)
                wind=u+D@wind
                cur=nxt
            perm[i]=cur;sw[i]=wind
        lookup={}
        for b,link in enumerate(self.bonds):
            a=(link.i,link.j,link.wind)
            back=(link.j,link.i,tuple(-x for x in link.wind))
            if a in lookup or back in lookup:raise AssertionError('Duplicate PBC bonds')
            lookup[a]=(b,1);lookup[back]=(b,-1)
        bperm=[];orient=[]
        for link in self.bonds:
            i,j=link.i,link.j
            wind_new=sw[j]+powerD@np.asarray(link.wind)-sw[i]
            mapped=lookup.get((int(perm[i]),int(perm[j]),tuple(map(int,wind_new))))
            if mapped is None:raise AssertionError(f'Failed to rotate bond {link}')
            bperm.append(mapped[0]);orient.append(mapped[1])
        return perm,np.array(bperm,int),np.array(orient,int)

    def _build_symmetry(self):
        self.rotation_c6=self._rotate_map(1)
        perm,bperm,orient=(np.arange(self.N),np.arange(self.B),np.ones(self.B,int))
        if self.symmetry=='c6':perm,bperm,orient=self.rotation_c6
        elif self.symmetry=='c3':perm,bperm,orient=self._rotate_map(2)
        self.site_orbits=_site_orbits(perm)
        self.bond_orbits=[]
        visited=set()
        for b in range(self.B):
            if b in visited:continue
            orbit=[];cur=b;sgn=1
            while cur not in visited:
                visited.add(cur);orbit.append((cur,sgn));sgn*=orient[cur];cur=bperm[cur]
            if cur!=b:raise AssertionError('Invalid bond permutation')
            if sgn==-1:raise AssertionError('Bond self-orbit reverses orientation; '
                                            'would force a real channel')
            self.bond_orbits.append(orbit)
        self.G=len(self.bond_orbits)
        self.bclass=np.empty(self.B,dtype=int)
        self.bsign=np.ones(self.B,int)
        for q,orb in enumerate(self.bond_orbits):
            for b,s in orb:self.bclass[b]=q;self.bsign[b]=s
        self.sclass=np.empty(self.N,dtype=int)
        for q,orb in enumerate(self.site_orbits):self.sclass[orb]=q
        # Orientation-aware flow matrix for the imaginary bond amplitudes.
        self.divergence_matrix=np.zeros((self.N,self.G),float)
        for q,orb in enumerate(self.bond_orbits):
            for b,s in orb:
                self.divergence_matrix[self.ii[b],q]+=s*self.tb[b]
                self.divergence_matrix[self.jj[b],q]-=s*self.tb[b]
        sv=np.linalg.svd(self.divergence_matrix,full_matrices=True)
        tol=1e-10*max(1.,sv[1][0] if sv[1].size else 0)
        self.current_rank=int(np.count_nonzero(sv[1]>tol))
        self.current_null=sv[2][self.current_rank:].T  # columns are Kirchhoff-safe seed directions
        self._class_representatives()

    def _class_representatives(self):
        info=[]
        for ci,orb in enumerate(self.bond_orbits):
            kinds=sorted(set(self.bonds[b].kind for b,_ in orb))
            # Minimum midpoint radius over translations gives a natural visible classification.
            mids=[]
            for b,_ in orb:
                e=self.bonds[b]
                p=.5*(self.positions[e.i]+self.positions[e.j]+self.S@np.asarray(e.wind))
                mins=min(np.linalg.norm(p+self.S@np.asarray(w)) for w in itertools.product(range(-2,3),repeat=2))
                mids.append(mins)
            info.append({'class':ci+1,'kind':'+'.join(kinds),'count':len(orb),
                         'min_mid_radius':float(np.mean(mids)),
                         'bonds':[int(b) for b,_ in orb],
                         'orientation_signs':[int(s) for _,s in orb]})
        self.class_info=info

    def _validate(self):
        if self.B!=12*self.ncell:raise AssertionError('Unexpected bond count')
        if np.max(abs(self.H0-self.H0.conj().transpose(0,2,1)))>1e-11:
            raise AssertionError('non-Hermitian bare Hamiltonian')
        # All physical links of this ED graph have the same geometric length.
        for b in self.bonds:
            v=self.positions[b.j]+self.S@np.asarray(b.wind)-self.positions[b.i]
            if abs(np.linalg.norm(v)-1)>1e-10:raise AssertionError(f'Broken geometric edge {b}')
        # Ensure each site has exactly four neighbors, as in native spinless ED.
        deg=np.bincount(np.r_[self.ii,self.jj],minlength=self.N)
        if not np.all(deg==4):raise AssertionError(f'Coordination wrong: {deg}')
        if self.period=='ring3':
            assert self.N==18 and self.B==36
            assert np.count_nonzero(np.any(self.winds,axis=1))==6
        if self.symmetry=='c6':
            assert self.G<=self.B

    def project(self,n,chi):
        """Project onto the exact site and oriented-bond point-group fixed subspace."""
        n=np.asarray(n,float);chi=np.asarray(chi,complex)
        outn=n.copy();outc=chi.copy()
        for orb in self.site_orbits:outn[orb]=np.mean(n[orb])
        for orb in self.bond_orbits:
            z=np.mean([chi[b] if s==1 else np.conjugate(chi[b]) for b,s in orb])
            for b,s in orb:outc[b]=z if s==1 else np.conjugate(z)
        return outn,outc

    def channel_parameters(self,chi):
        return np.array([np.mean([chi[b] if s==1 else chi[b].conjugate() for b,s in orb])
                         for orb in self.bond_orbits],complex)

    def hmatrix(self,n,chi,mu):
        H=self.H0.copy()
        v=np.zeros(self.N,float)
        np.add.at(v,self.ii,self.Vb*n[self.jj]);np.add.at(v,self.jj,self.Vb*n[self.ii])
        for b in range(self.B):
            i,j=self.ii[b],self.jj[b]
            z=(-self.Vb[b]*chi[b])*self.ph[:,b]
            H[:,i,j]+=z;H[:,j,i]+=z.conjugate()
        H[:,np.arange(self.N),np.arange(self.N)]+=v[None,:]-mu
        return H

    def _observables(self,e0,u,mu,T):
        """Construct normal density matrix from a single eigendecomposition.

        Spectrum e0 belongs to the mu=0 Hartree-Fock matrix. Chemical
        potential shifts eigenvalues only, leaving all eigenvectors unchanged.
        """
        f=(e0<mu).astype(float) if T<=0 else expit((mu-e0)/T)
        rho_k=(u*f[:,None,:])@u.conjugate().transpose(0,2,1)
        nn=np.real(np.mean(np.diagonal(rho_k,axis1=1,axis2=2),axis=0))
        cb=np.mean(rho_k[:,self.ii,self.jj]/self.ph,axis=0)
        entropy=0. if T<=0 else -float(np.mean(np.sum(xlogy(f,f)+xlogy(1-f,1-f),axis=1)))
        ekin=2*float(np.sum(np.real(self.tb*cb.conjugate())))
        eH=float(np.sum(self.Vb*nn[self.ii]*nn[self.jj]))
        eF=-float(np.sum(self.Vb*abs(cb)**2))
        E=ekin+eH+eF
        return {'n':nn,'chi':cb,'Npart':float(np.sum(nn)),'mu':float(mu),
                'E':E,'F':E-T*entropy,'Omega':E-T*entropy-mu*np.sum(nn),
                'Ekin':ekin,'EH':eH,'EFock':eF,'S':entropy}

    def evaluate(self,n,chi,mu,T):
        e,u=np.linalg.eigh(self.hmatrix(n,chi,0.))
        return self._observables(e,u,mu,T)

    def find_mu_from_evals(self,e0,target,T):
        if T<=0:raise ValueError('Fixed filling requires --temperature > 0')
        if not 0<target<self.N:raise ValueError('Target particle count outside (0,N)')
        low=float(np.min(e0))-max(1.,60*T)
        high=float(np.max(e0))+max(1.,60*T)
        def f(mu):return float(np.mean(np.sum(expit((mu-e0)/T),axis=1)))-target
        return float(brentq(f,low,high,xtol=1e-12,rtol=1e-13))

    def find_mu(self,n,chi,target,T,mu=0.):
        e0=np.linalg.eigvalsh(self.hmatrix(n,chi,0.))
        return self.find_mu_from_evals(e0,target,T)

    def _real_star_formfactor(self):
        """Real SD/ISD seed field: geometric star for ring3, triple-M for 2x2.

        These are Ruby *analogs* of kagome SD/ISD, not their kagome order
        parameters; both are fully symmetry-projected after initialization.
        """
        f=np.zeros(self.B,float);site=np.zeros(self.N,float)
        if self.period=='ring3':
            # C6 classes: B1+B2 triangle spokes, B6 central hexagon.
            # Complement: B3 outer triangle bases and B4+B5 connectors.
            _,rotate,_=self.rotation_c6
            visited=set();orbits=[]
            for ib in range(self.B):
                if ib in visited:continue
                q=[];j=ib
                while j not in visited:
                    q.append(j);visited.add(j);j=rotate[j]
                orbits.append(q)
            star=set()
            for q in orbits:
                rr=[]
                for ib in q:
                    b=self.bonds[ib]
                    mid=.5*(self.positions[b.i]+self.positions[b.j]+self.S@np.array(b.wind))
                    rr.append(min(np.linalg.norm(mid+self.S@np.array(w))
                                  for w in itertools.product(range(-2,3),repeat=2)))
                dist=float(np.mean(rr));kind=self.bonds[q[0]].kind
                if (kind=='t' and dist<1.7) or (kind!='t' and dist<1.05):star.update(q)
            for ib in range(self.B):f[ib]=1 if ib in star else -1
            # Circular site-density contrast is generated automatically by
            # self-consistency. Do not impose spurious A/B density splitting.
        else:
            # 3 symmetry-related M points of the 2x2 enlarged Bravais period.
            # Real-space triple-M harmonic is exactly C6 invariant.
            G=2*np.pi*np.linalg.inv(A).T
            wavevectors=[G@np.array([.5,0.]),G@np.array([0.,.5]),G@np.array([.5,.5])]
            for ib,b in enumerate(self.bonds):
                x=A@self.reps[b.cell]+.5*(ORBITAL[b.i%6]+ORBITAL[b.j%6]+A@np.array(b.prim_delta))
                f[ib]=sum(np.cos(Q@x) for Q in wavevectors)
            for i in range(self.N):
                x=ORBITAL[i%6]+A@self.reps[i//6]
                site[i]=sum(np.cos(Q@x) for Q in wavevectors)
        f-=np.mean(f);site-=np.mean(site)
        if np.max(abs(f))>1e-12:f/=np.max(abs(f))
        if np.max(abs(site))>1e-12:site/=np.max(abs(site))
        return f,site

    def _inplane_co_formfactor(self,name):
        """Return a physical triangle-pseudospin x/y CO density pattern.

        The even/odd labels refer to the A/B triangle combination in the
        physical pseudospin frame.  These patterns break C3 and must not be
        projected into a C3/C6-invariant subspace.
        """
        if name not in ('x_even','x_odd','y_even','y_odd'):
            raise ValueError(f'Unknown in-plane CO seed {name}')
        comp,parity=name.split('_')
        a=np.zeros(6,float);b=np.zeros(6,float)
        a[:3]=[2.,-1.,-1.] if comp=='x' else [0.,-SQ3,SQ3]
        # B's physical pseudospin frame reverses the algebraic y component.
        b[3:]=[2.,-1.,-1.] if comp=='x' else [0.,SQ3,-SQ3]
        pattern=a+b if parity=='even' else a-b
        pattern/=np.max(np.abs(pattern))
        return np.tile(pattern,self.ncell)

    def seed(self,name,fill=1/3,amp=.05,random_seed=0):
        rng=np.random.default_rng(random_seed)
        n=np.full(self.N,fill,float)
        chi=np.full(self.B,.1+0j,complex)
        if name in ('x_even','x_odd','y_even','y_odd'):
            if self.symmetry!='none':
                raise ValueError(f'{name} breaks C3; use --symmetry none to retain this seed')
            n+=amp*self._inplane_co_formfactor(name)
        elif name=='cdw':
            n+=amp*rng.normal(size=self.N)
        elif name in ('sd','isd'):
            form,dens=self._real_star_formfactor();sign=1 if name=='sd' else -1
            chi+=amp*sign*form
            n+=.4*amp*sign*dens
        elif name=='real_bond':
            chi+=amp*rng.normal(size=self.B)
        elif name in ('lc_same','lc_opposite'):
            signs=[1,1] if name=='lc_same' else [1,-1]
            for c,q,links in self.triangles:
                for b,s in links:
                    chi[b]+=1j*amp*s*signs[q]*np.sign(self.tb[b])
        elif name.startswith('lc_mode'):
            idx=int(name[len('lc_mode'):])
            if idx>=self.current_null.shape[1]:raise ValueError('Invalid current mode')
            coeff=self.current_null[:,idx]
            chi+=1j*amp*(self.bsign*coeff[self.bclass])
        elif name=='random':
            n+=amp*rng.normal(size=self.N)
            chi+=amp*rng.normal(size=self.B)+1j*amp*rng.normal(size=self.B)
        elif name!='normal':raise ValueError(f'Unknown seed {name}')
        n+=fill-n.mean()
        n=np.clip(n,1e-6,1-1e-6)
        return self.project(n,chi)

    def solve(self,name='normal',target=None,mu=0.,T=.03,amp=.05,
              mix=.35,max_iter=350,tol=1e-7,seed=1000,initial=None,verbose=False):
        if target is not None and not 0<target<self.N:raise ValueError('Target outside (0,N)')
        if mix<=0 or mix>1:raise ValueError('mix must be (0,1]')
        if T<0:raise ValueError('Temperature must be >=0')
        n,chi=self.seed(name,target/self.N if target is not None else .5,amp,seed) if initial is None else (
            np.asarray(initial['n'],float).copy(),np.asarray(initial['chi'],complex).copy())
        n,chi=self.project(n,chi)
        mu0=float(mu)
        for it in range(1,max_iter+1):
            e0,u=np.linalg.eigh(self.hmatrix(n,chi,0.))
            if target is not None:mu0=self.find_mu_from_evals(e0,target,T)
            r=self._observables(e0,u,mu0,T)
            ni,ci=self.project(r['n'],r['chi'])
            error=max(float(np.max(abs(ni-n))),float(np.max(abs(ci-chi))))
            if verbose and (it==1 or it%20==0 or error<tol):
                print(f'  {name:14s} it={it:4d} res={error:.3e} F={r["F"]:.9g} mu={mu0:.7g}',flush=True)
            if error<tol:break
            n=(1-mix)*n+mix*ni
            chi=(1-mix)*chi+mix*ci
        e0,u=np.linalg.eigh(self.hmatrix(n,chi,0.))
        if target is not None:mu0=self.find_mu_from_evals(e0,target,T)
        r=self._observables(e0,u,mu0,T)
        ni,ci=self.project(r['n'],r['chi'])
        residual=max(float(np.max(abs(ni-n))),float(np.max(abs(ci-chi))))
        # Symmetry covariance should hold automatically in the raw output.
        symerr=max(float(np.max(abs(r['n']-ni))),float(np.max(abs(r['chi']-ci))))
        r.update({'residual':residual,'symmetry_residual':symerr,'converged':bool(residual<tol and symerr<tol),
                  'iterations':it,'seed':name,'T':T,'symmetry':self.symmetry,'period':self.period})
        r.update(self.diagnostics(r))
        return r

    def diagnostics(self,r):
        J=2*np.imag(self.tb*r['chi'])
        div=np.zeros(self.N,float)
        np.add.at(div,self.ii,J);np.add.at(div,self.jj,-J)
        curr=np.zeros((self.ncell,2))
        for c,q,edges in self.triangles:
            curr[c,q]=np.mean([sgn*J[b] for b,sgn in edges])
        chi_sym=self.channel_parameters(r['chi'])
        # primitive-cell Fourier decomposition of charge for nonzero commensurate Q
        modes=[]
        mrange=range(0,4)
        for k1,k2 in itertools.product(mrange,repeat=2):
            if (k1,k2)==(0,0):continue
            q=np.mod(np.linalg.solve(self.M.T,np.array([k1,k2])),1)
            if np.linalg.norm(q)<1e-9 or any(np.linalg.norm(np.mod(q-qp+.5,1)-.5)<1e-9 for qp,_ in modes):continue
            phases=np.exp(-2j*np.pi*self.reps@q)
            D=np.asarray(r['n']).reshape(self.ncell,6)
            value=float(np.linalg.norm(phases@D/self.ncell))
            modes.append((q,value))
        strongest=max(modes,key=lambda z:z[1]) if modes else (np.zeros(2),0.)
        return dict(triangle_current=curr,bond_current=J,
                    J_A=float(np.mean(curr[:,0])),J_B=float(np.mean(curr[:,1])),
                    J_plus=float(np.mean((curr[:,0]+curr[:,1])/2)),
                    J_minus=float(np.mean((curr[:,0]-curr[:,1])/2)),
                    J_tri_rms=float(np.sqrt(np.mean(curr**2))),
                    max_divergence=float(np.max(abs(div))),
                    charge_std=float(np.std(r['n'])),
                    charge_mod_q=np.asarray(strongest[0]),
                    charge_mod_max=float(strongest[1]),
                    real_bond_std=float(np.std(np.real(r['chi']))),
                    current_rms=float(np.sqrt(np.mean(J**2))),
                    imag_channel=np.imag(chi_sym),
                    real_channel=np.real(chi_sym))

    def report_geometry(self):
        print(f'period={self.period}, sites={self.N}, independent directed bonds={self.B}, '
              f'PBC-crossing bonds={np.count_nonzero(np.any(self.winds,axis=1))}')
        print(f'symmetry={self.symmetry}: density classes={len(self.site_orbits)}, '
              f'bond classes={self.G}, current continuity rank={self.current_rank}, '
              f'divergence-free imaginary channels={self.current_null.shape[1]}')
        print('site orbits:',self.site_orbits)
        for info,orb in zip(self.class_info,self.bond_orbits):
            representative=self.bonds[orb[0][0]]
            print(f'  B{info["class"]}: {info["count"]:2} edges; kind={info["kind"]:5} '
                  f'example=({representative.i},{representative.j},{representative.wind}), '
                  f'avg radius={info["min_mid_radius"]:.4f}')


def _jsonable(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,tuple):return list(v)
    return v


def save_solution(m,r,directory):
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    summary={k:_jsonable(v) for k,v in r.items() if k not in ('n','chi','bond_current','triangle_current')}
    (path/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    np.savez_compressed(path/'state.npz',n=r['n'],chi=r['chi'],bond_current=r['bond_current'],
                        triangle_current=r['triangle_current'],pos=m.positions,ii=m.ii,jj=m.jj,
                        wind=m.winds,kind=np.array([b.kind for b in m.bonds]),bclass=m.bclass,
                        bsign=m.bsign,M=m.M,A=A,chi_class=m.channel_parameters(r['chi']))
    with (path/'sites.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['i','primitive_cell','orbital','density','x','y','symmetry_class'])
        for i,n in enumerate(r['n']):w.writerow([i,i//6,i%6,float(n),*m.positions[i],int(m.sclass[i])+1])
    with (path/'bonds.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['bond','i','j','wind1','wind2','kind','group','orient',
                                     't','V','chi_real','chi_imag','J_particle'])
        for ib,b in enumerate(m.bonds):w.writerow([ib,b.i,b.j,*b.wind,b.kind,int(m.bclass[ib])+1,
                          int(m.bsign[ib]),b.t,b.V,r['chi'][ib].real,r['chi'][ib].imag,r['bond_current'][ib]])


def draw_classes(m,outprefix='ruby_hex_bond_classes'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon
    from matplotlib import cm
    from matplotlib.colors import to_hex
    prefix=Path(outprefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    cmap=plt.get_cmap('tab20')
    # A stable full set of named groups and explicit orientation arrows.
    colors=[to_hex(cmap(i%20)) for i in [0,2,4,6,8,10,12,14,16,18]+list(range(20))]
    pts=m.positions
    centroid=np.zeros(2)
    def draw_panel(ax,highlight=None,annotate=False):
        minsize=np.min(np.linalg.norm(m.S,axis=0))
        # Wigner-Seitz hex boundary (Voronoi polygon of superlattice).
        shifts=[m.S@np.array([a,b]) for a,b in itertools.product(range(-2,3),repeat=2) if (a,b)!=(0,0)]
        from scipy.spatial import HalfspaceIntersection,ConvexHull
        hs=np.array([[v[0],v[1],-np.dot(v,v)/2] for v in shifts],float)
        poly=HalfspaceIntersection(hs, np.array([0.,0.])).intersections
        poly=poly[ConvexHull(poly).vertices]
        ax.add_patch(Polygon(poly,closed=True,facecolor='#f7f9fc',edgecolor='#37465A',
                             linewidth=1.4,linestyle=(0,(4,3)),zorder=0))
        # Draw bonds with segments crossing the boundary visible in faded dashed style.
        for ib,b in enumerate(m.bonds):
            col='#B7BEC6' if highlight is not None and m.bclass[ib]!=highlight else colors[m.bclass[ib]%len(colors)]
            alpha=.15 if highlight is not None and m.bclass[ib]!=highlight else .95
            p=pts[b.i];q=pts[b.j]+m.S@np.asarray(b.wind)
            ax.plot([p[0],q[0]],[p[1],q[1]],color=col,lw=2.1 if highlight is None else 2.5,
                    ls='--' if any(b.wind) else '-',alpha=alpha,zorder=1)
            if highlight is not None and m.bclass[ib]==highlight:
                sign=m.bsign[ib];start,end=(p,q) if sign==1 else (q,p)
                mid=.5*(start+end);vec=(end-start)*.12
                ax.annotate('',xy=mid+vec,xytext=mid-vec,arrowprops=dict(arrowstyle='-|>',
                            color=col,lw=1.2,mutation_scale=10),zorder=4)
        ax.scatter(pts[:,0],pts[:,1],s=42,facecolor='white',edgecolor='#1e344b',linewidths=.9,zorder=3)
        if annotate:
            for i,(x,y) in enumerate(pts):
                ax.text(x,y,f'{i//6}:{i%6}',fontsize=6.6,ha='center',va='center',color='#152c47',zorder=5)
        allp=np.r_[pts,poly]
        lim=max(abs(allp[:,0]).max(),abs(allp[:,1]).max())*1.12
        ax.set(xlim=(-lim,lim),ylim=(-lim,lim));ax.set_aspect('equal');ax.axis('off')
        return poly
    ncols=3;nrows=math.ceil(m.G/ncols)
    fig,axs=plt.subplots(nrows,ncols,figsize=(13,4.2*nrows),constrained_layout=True)
    axs=np.atleast_1d(axs).ravel()
    for g in range(m.G):
        draw_panel(axs[g],highlight=g)
        info=m.class_info[g]
        axs[g].set_title(f'B{g+1}  |  {info["count"]} links  |  {info["kind"]}',
                         color=colors[g%len(colors)],fontsize=14,weight='bold')
    for ax in axs[m.G:]:ax.axis('off')
    fig.suptitle(f'Ruby {m.period} | {m.symmetry.upper()} bond orbits (colored = one independent complex $\\chi$)',fontsize=16)
    fig.savefig(str(prefix)+'_detail.png',dpi=210,bbox_inches='tight')
    fig.savefig(str(prefix)+'_detail.svg',bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,8.1))
    draw_panel(ax,annotate=False)
    ax.set_title(f'{m.period} | {m.symmetry.upper()} symmetry | {m.N} sites, {m.B} bonds, {m.G} bond classes',fontsize=12)
    from matplotlib.lines import Line2D
    hs=[Line2D([0],[0],color=colors[i%len(colors)],linewidth=4,
               label=f'B{i+1}: {q["kind"]}, {q["count"]} bonds') for i,q in enumerate(m.class_info)]
    fig.legend(handles=hs,loc='lower center',ncol=min(3,m.G),fontsize=9,frameon=False)
    fig.subplots_adjust(bottom=.19)
    fig.savefig(str(prefix)+'_overview.png',dpi=230,bbox_inches='tight')
    fig.savefig(str(prefix)+'_overview.svg',bbox_inches='tight')
    plt.close(fig)
    with open(str(prefix)+'_mapping.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['bond','from_site','to_site','wind1','wind2','kind','class',
                                      'orientation_conjugate'])
        for bi,b in enumerate(m.bonds):w.writerow([bi,b.i,b.j,*b.wind,b.kind,m.bclass[bi]+1,
                                                  int(m.bsign[bi]==-1)])
    (Path(str(prefix)+'_metadata.json')).write_text(json.dumps({
          'period':m.period,'symmetry':m.symmetry,'M':m.M.tolist(),'site_orbits':[[int(i) for i in o] for o in m.site_orbits],
          'bond_orbits':m.class_info,'current_continuity_rank':m.current_rank,
          'independent_imag_current_modes':int(m.current_null.shape[1])},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--period',choices=PERIODS.keys(),default='ring3')
    p.add_argument('--symmetry',choices=['c6','c3','none'],default='none')
    p.add_argument('--nk',nargs=2,type=int,default=[6,6])
    p.add_argument('--t',type=float,default=1.)
    p.add_argument('--t1',type=float,default=.5)
    p.add_argument('--t2',type=float,default=.5)
    p.add_argument('--V-tri',type=float,default=1.)
    p.add_argument('--V-t1',type=float,default=0.)
    p.add_argument('--V-t2',type=float,default=0.)
    p.add_argument('--scan-V-tri',type=float,nargs='*')
    p.add_argument('--scan-V-t1',type=float,nargs='*')
    p.add_argument('--scan-V-t2',type=float,nargs='*')
    p.add_argument('--scan-V-connector',type=float,nargs='*',default=None,
                   help='Linked scan Vt1=Vt2, useful in the C6 sector')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--filling',type=float,default=None,help='Particles PER six-site primitive unit cell (0..6)')
    mode.add_argument('--mu',type=float,default=None)
    p.add_argument('--temperature',type=float,default=.04)
    p.add_argument('--max-iter',type=int,default=300)
    p.add_argument('--tol',type=float,default=1e-7)
    p.add_argument('--mix',type=float,default=.3)
    p.add_argument('--seed-amplitude',type=float,default=.06)
    p.add_argument('--seeds',nargs='+',default=['normal','x_even','x_odd','y_even','y_odd',
                  'cdw','sd','isd','real_bond','lc_same','lc_opposite','random'])
    p.add_argument('--random-starts',type=int,default=1)
    p.add_argument('--no-auto-lc-modes',action='store_true',
                   help='Skip independent Kirchhoff-compatible LC channel seeds')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--only-geometry',action='store_true')
    p.add_argument('--draw-bonds',action='store_true')
    p.add_argument('--no-continuation',action='store_true')
    p.add_argument('--save-candidates',action='store_true',
                   help='Also save complete converged metastable states, not just the winner')
    p.add_argument('--verbose',action='store_true')
    p.add_argument('--output',default='ruby_hex_hf_results')
    a=p.parse_args()
    if a.scan_V_connector is not None:
        if a.scan_V_t1 is not None or a.scan_V_t2 is not None:
            raise ValueError('Use either --scan-V-connector OR separate --scan-V-t1/t2')
        combos=[(vt,v,v) for vt,v in itertools.product(
            a.scan_V_tri if a.scan_V_tri is not None else [a.V_tri],a.scan_V_connector)]
    else:
        combos=list(itertools.product(a.scan_V_tri if a.scan_V_tri is not None else [a.V_tri],
             a.scan_V_t1 if a.scan_V_t1 is not None else [a.V_t1],
             a.scan_V_t2 if a.scan_V_t2 is not None else [a.V_t2]))
    if not combos:raise ValueError('Scan list is empty')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    previous=None;rows=[]
    for ip,(vt,v1,v2) in enumerate(combos):
        m=RubyHexHF(a.period,a.symmetry,tuple(a.nk),a.t,a.t1,a.t2,vt,v1,v2)
        if ip==0:
            m.report_geometry()
            if a.draw_bonds or a.only_geometry:draw_classes(m,str(root/'bonds'))
        if a.only_geometry:return
        target=None if a.mu is not None else (2. if a.filling is None else a.filling)*m.ncell
        mu=0. if a.mu is None else a.mu
        print(f'point {ip+1}/{len(combos)} Vtri={vt:g} Vt1={v1:g} Vt2={v2:g} '
              f'Ntarget={target} symmetry={m.symmetry}',flush=True)
        seeds=list(a.seeds)+['random']*a.random_starts
        if not a.no_auto_lc_modes:
            seeds += [f'lc_mode{j}' for j in range(m.current_null.shape[1])]
        if previous is not None and not a.no_continuation:seeds=['continuation']+seeds
        trials=[]
        for i,name in enumerate(seeds):
            if name.startswith('lc_mode') and int(name[7:])>=m.current_null.shape[1]:
                print(f'  skip {name}: no such divergence-free LC channel');continue
            r=m.solve(name if name!='continuation' else 'normal',target=target,mu=mu,T=a.temperature,
                      amp=a.seed_amplitude,max_iter=a.max_iter,mix=a.mix,tol=a.tol,
                      seed=a.seed+i,initial=previous if name=='continuation' else None,
                      verbose=a.verbose)
            r['seed']=name;trials.append(r)
            if a.save_candidates and r['converged']:
                save_solution(m,r,root/f'point_{ip:04d}'/'candidate_states'/f'{i:02d}_{name}')
            print(f'  {name:14s} conv={r["converged"]!s:5s} res={r["residual"]:.2e} '
                  f'F={r["F"]:+.9f} LC={r["J_tri_rms"]:.3e} '
                  f'CDW={r["charge_mod_max"]:.3e} div={r["max_divergence"]:.2e}',flush=True)
        point=root/f'point_{ip:04d}';point.mkdir(exist_ok=True)
        with (point/'candidates.csv').open('w',newline='') as f:
            cols=['seed','converged','iterations','residual','symmetry_residual','F','Omega',
                  'E','Ekin','EH','EFock','Npart','mu','charge_std','charge_mod_max',
                  'J_A','J_B','J_plus','J_minus','J_tri_rms','current_rms','max_divergence']
            w=csv.DictWriter(f,cols);w.writeheader()
            for r in trials:w.writerow({c:r.get(c) for c in cols})
        good=[r for r in trials if r['converged']]
        row={'point':ip,'V_tri':vt,'V_t1':v1,'V_t2':v2,'period':a.period,'symmetry':a.symmetry,
             'valid':bool(good),'Nk1':a.nk[0],'Nk2':a.nk[1],
             'temperature':a.temperature,'filling_per_primitive':None if target is None else target/m.ncell,
             'mu_fixed':a.mu,'t':a.t,'t1':a.t1,'t2':a.t2}
        if good:
            best=min(good,key=lambda r:r['Omega'] if target is None else r['F'])
            save_solution(m,best,point/'best')
            previous=best
            row.update({k:_jsonable(best[k]) for k in ('seed','F','Omega','E','mu','Npart',
                         'J_A','J_B','J_plus','J_minus','J_tri_rms','current_rms',
                         'max_divergence','charge_std','charge_mod_max','real_bond_std')})
        else:
            print('WARNING: no converged candidate; phase undefined',flush=True)
            previous=None
        rows.append(row)
        (root/'scan.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        with (root/'scan.csv').open('w',newline='') as f:
            cols=list(dict.fromkeys(k for row2 in rows for k in row2))
            w=csv.DictWriter(f,cols);w.writeheader();w.writerows(rows)
    print('Results:',root/'scan.csv',flush=True)


if __name__=='__main__':main()
