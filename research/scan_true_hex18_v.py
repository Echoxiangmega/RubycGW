#!/usr/bin/env python3
"""Spinless Ruby exact diagonalization on the genuine periodic 18-site hexagon.

This is an independent *finite-torus ED* V scan, not a Hartree-Fock calculation,
not 18 independent sites with open boundaries, and not the cluster-ED+GW method.
The true_hex geometry reproduces the reconstructed ring3_pbc_verified.svg
geometry used in ruby_hex_hf.py, including 6 PBC-wrapped intertriangle bonds.

The existing RubycGW ``hex18`` uses EXACTLY THE SAME torus as ``true_hex``
after a unimodular primitive-coordinate transformation and an orbital-origin
shift on the B triangle. See verify_hex18_equivalence.py for the exact
18-site permutation proof.  ``skew18`` here is a different period in the
DRAWING coordinates, not the legacy repository ``hex18``!

Positive hopping: H=sum_b t_b(c_i^dag c_j + h.c.)+V_b*n_i*n_j.
Default V scan acts on the 18 triangle bonds (6 per primitive cell), as in
native Ruby ED; V-t1 and V-t2 can be set separately.

Outputs: geometry.json, bonds.csv, torus.png/svg, scan.csv/json,
optional ground states and a concise noninteracting band check.

Dependencies: numpy, scipy, matplotlib (only for plotting).
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

SQ3=math.sqrt(3)
A=np.array([[-(1+SQ3)/2,(1+SQ3)/2],[(3+SQ3)/2,(3+SQ3)/2]],float)
ORBITAL=np.array([
 [0.,-1.],[-.5,-1.-SQ3/2],[+.5,-1.-SQ3/2],
 [0.,+1.],[-.5,+1.+SQ3/2],[+.5,+1.+SQ3/2]],float)
# Each entry represents ONE independent hopping bond in the infinite lattice.
# t: 6 intra-triangle bonds, t1/t2: 3 connecting bonds each per 6-site cell.
PRIMITIVE_BONDS=[
 (0,1,(0,0),'t'),(0,2,(0,0),'t'),(2,1,(0,0),'t'),
 (3,4,(0,0),'t'),(3,5,(0,0),'t'),(4,5,(0,0),'t'),
 (1,4,(-1,-1),'t1'),(5,0,(0,1),'t1'),(2,3,(-1,0),'t1'),
 (3,1,(0,1),'t2'),(2,5,(-1,-1),'t2'),(0,4,(-1,0),'t2')]
TRIANGLE_PATHS=[(0,1,2),(3,5,4)] # two physically CCW triangles in each cell
PERIODS={
 'true_hex':np.array([[-1,2],[-1,-1]],int),
 'skew18':np.array([[1,1],[-1,2]],int)}
REPS={
 'true_hex':np.array([[0,0],[1,0],[0,1]],int),
 'skew18':np.array([[0,0],[1,0],[2,0]],int)}
KINDS=['t','t1','t2']

@dataclass(frozen=True)
class Bond:
    i:int
    j:int
    kind:str
    wind:tuple[int,int]
    delta:tuple[int,int]
    cell:int


def quotient(M,x):
    v=np.linalg.solve(M,np.asarray(x,float))
    vr=np.rint(v).astype(int)
    return tuple(map(int,vr)) if np.allclose(v,vr,atol=1e-11) else None


def momentum_set(M):
    q=[]
    for a,b in itertools.product(range(3),repeat=2):
        p=np.mod(np.linalg.solve(M.T,[a,b]),1)
        if not any(np.allclose(np.mod(p-v+.5,1)-.5,0,atol=1e-10) for v in q):q.append(p)
    return np.asarray(sorted(q,key=lambda v:(round(v[0],8),round(v[1],8))),float)


class RubyTorus:
    def __init__(self,torus='true_hex',t=1.,t1=.5,t2=.5):
        if torus not in PERIODS:raise ValueError('Unknown torus')
        self.name=torus
        self.M=PERIODS[torus].copy()
        self.reps=REPS[torus].copy()
        self.ncell=len(self.reps);self.N=6*self.ncell
        self.tmap={'t':float(t),'t1':float(t1),'t2':float(t2)}
        self.shift=np.zeros((self.N,2),dtype=int)
        # These gauge choices create the same displayed physical six-triangle
        # ring used in the earlier Hartree-Fock diagrams. They change only
        # the chosen images of the stored sites, never the ED graph.
        if torus=='true_hex':
            for c in (1,2):self.shift[6*c+3:6*c+6]=[1,0]
        self.S=A@self.M
        self.pos=np.empty((self.N,2),float)
        for c,r in enumerate(self.reps):
            for a in range(6):self.pos[6*c+a]=ORBITAL[a]+A@(r+self.M@self.shift[6*c+a])
        self.bonds=[]
        for c,r in enumerate(self.reps):
            for i,j,d,k in PRIMITIVE_BONDS:
                dest=r+np.asarray(d,int)
                matches=[]
                for cp,rp in enumerate(self.reps):
                    w=quotient(self.M,dest-rp)
                    if w is not None:matches.append((cp,np.asarray(w,dtype=int)))
                if len(matches)!=1:raise AssertionError(('Nonunique quotient',r,i,j,d,matches))
                cp,w=matches[0];ii=6*c+i;jj=6*cp+j
                wind=w+self.shift[ii]-self.shift[jj]
                self.bonds.append(Bond(ii,jj,k,tuple(map(int,wind)),tuple(d),c))
        self.q=momentum_set(self.M)
        self._check()

    def _check(self):
        assert self.N==18 and self.ncell==3 and len(self.bonds)==36
        assert len(self.q)==3
        counts={k:sum(b.kind==k for b in self.bonds) for k in KINDS}
        assert counts=={'t':18,'t1':9,'t2':9},counts
        seen=set()
        for b in self.bonds:
            key=(b.i,b.j,b.wind);rk=(b.j,b.i,tuple(-v for v in b.wind))
            if key in seen or rk in seen:raise AssertionError(('Duplicate PBC link',b))
            seen.add(key)
            displacement=self.pos[b.j]+self.S@b.wind-self.pos[b.i]
            if not np.isclose(np.linalg.norm(displacement),1,atol=1e-10):
                raise AssertionError(('Not a physical NN link',b,displacement))
        if self.name=='true_hex':
            assert sum(any(b.wind) for b in self.bonds)==6
            expected={(1,4,(1,0),'t1'),(2,5,(1,0),'t2'),
                      (17,6,(0,-1),'t1'),(14,9,(-1,-1),'t1'),
                      (15,7,(0,-1),'t2'),(12,10,(-1,-1),'t2')}
            got={(b.i,b.j,b.wind,b.kind) for b in self.bonds if any(b.wind)}
            assert got==expected,('SVG/PBC link mismatch',got,expected)
            # The true hexagonal torus preserves the 60-degree lattice rotation.
            rot=np.array([[1,1],[-1,0]],int)
            C=np.linalg.solve(self.M,rot@self.M)
            assert np.allclose(C,np.rint(C),atol=1e-12)
            assert np.allclose(np.sort(self.q,axis=0),np.sort(np.array([[0,0],[1/3,2/3],[2/3,1/3]]),axis=0))
        else:
            rot=np.array([[1,1],[-1,0]],int)
            C=np.linalg.solve(self.M,rot@self.M)
            assert not np.allclose(C,np.rint(C),atol=1e-12)

    def one_body(self):
        H=np.zeros((self.N,self.N))
        for b in self.bonds:
            t=self.tmap[b.kind];H[b.i,b.j]+=t;H[b.j,b.i]+=t
        return H

    def folded_bands(self):
        spectrum=[]
        for q in self.q:
            H=np.zeros((6,6),complex)
            for i,j,d,k in PRIMITIVE_BONDS:
                z=self.tmap[k]*np.exp(2j*np.pi*np.dot(q,d))
                H[i,j]+=z;H[j,i]+=z.conjugate()
            spectrum.extend(np.linalg.eigvalsh(H))
        return np.sort(spectrum)

    def triangles(self):
        out=[]
        lookup={(b.i,b.j): (k,+1) for k,b in enumerate(self.bonds)}
        lookup.update({(b.j,b.i):(k,-1) for k,b in enumerate(self.bonds)})
        for c in range(3):
            for ab,path in enumerate(TRIANGLE_PATHS):
                v=[6*c+a for a in path];l=[]
                for i,j in zip(v,v[1:]+v[:1]):
                    if (i,j) not in lookup:raise AssertionError(('Missing triangle edge',i,j))
                    ib,sgn=lookup[i,j]
                    if self.bonds[ib].kind!='t':raise AssertionError('Triangle uses connector')
                    l.append((ib,sgn))
                out.append((c,ab,l))
        return out

    def write_geometry(self,outdir):
        outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
        rows=[]
        for ib,b in enumerate(self.bonds):
            rows.append(dict(bond=ib,i=b.i,j=b.j,kind=b.kind,wind1=b.wind[0],wind2=b.wind[1],
                             primitive_delta1=b.delta[0],primitive_delta2=b.delta[1],cell=b.cell,
                             hopping=self.tmap[b.kind],wrapped=int(any(b.wind))))
        with (outdir/'bonds.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        data=dict(torus=self.name,period_matrix=self.M.tolist(),representatives=self.reps.tolist(),
                  S_cartesian=self.S.tolist(),positions=self.pos.tolist(),image_shifts=self.shift.tolist(),
                  q_primitive_fractional=self.q.tolist(),Nsites=18,Ncells=3,Nhopping_bonds=36,
                  Ntriangle_interactions=18,Nwrapped_display=sum(any(b.wind) for b in self.bonds),
                  hopping_sign='positive +t',primitive_bonds=PRIMITIVE_BONDS)
        (outdir/'geometry.json').write_text(json.dumps(data,indent=2))

    def plot(self,outdir):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from scipy.spatial import HalfspaceIntersection,ConvexHull
        p=Path(outdir);p.mkdir(parents=True,exist_ok=True)
        fig,ax=plt.subplots(figsize=(8.5,7.2))
        if self.name=='true_hex':
            shift=[self.S@np.array([u,v]) for u in range(-2,3) for v in range(-2,3) if (u,v)!=(0,0)]
            hs=np.array([[v[0],v[1],-np.dot(v,v)/2] for v in shift])
            poly=HalfspaceIntersection(hs,np.array([0.,0.])).intersections
            poly=poly[ConvexHull(poly).vertices]
            ax.add_patch(Polygon(poly,fill=False,edgecolor='#415A77',lw=2,ls='--'))
        colors={'t':'#C84D59','t1':'#327CC4','t2':'#21927D'}
        for b in self.bonds:
            a=self.pos[b.i];z=self.pos[b.j]+self.S@b.wind
            ax.plot([a[0],z[0]],[a[1],z[1]],color=colors[b.kind],lw=2.4,
                    ls='--' if any(b.wind) else '-',alpha=.9)
            if any(b.wind):ax.scatter(z[0],z[1],facecolor='white',edgecolor=colors[b.kind],s=25,zorder=3)
        ax.scatter(self.pos[:,0],self.pos[:,1],s=86,c='white',edgecolors='#20334B',zorder=5)
        for i,(x,y) in enumerate(self.pos):ax.text(x,y,str(i),ha='center',va='center',fontsize=8,zorder=6)
        ax.set_aspect('equal');ax.axis('off')
        ax.set_title(f'Ruby {self.name}: 18-site periodic torus (t red, t1 blue, t2 green)')
        from matplotlib.lines import Line2D
        handles=[Line2D([],[],color=colors[k],lw=2.5,label=k) for k in KINDS]
        handles.append(Line2D([],[],color='gray',lw=1.5,ls='--',label='PBC-crossing bond'))
        ax.legend(handles=handles,loc='lower left',fontsize=9)
        for ext in ('png','svg'):fig.savefig(p/f'torus.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)


class FixedNED:
    def __init__(self,g,Nf):
        self.g=g;self.Nf=Nf
        if not(0<=Nf<=g.N):raise ValueError('Nf out of range')
        # Site i is bit i, not the reversed bit convention in one native ED
        # implementation; spectra and physical observables are basis-independent.
        self.states=np.array(sorted(sum(1<<i for i in comb)
                 for comb in itertools.combinations(range(g.N),Nf)),dtype=np.uint32)
        self.dim=len(self.states)
        self.occ=((self.states[:,None]>>np.arange(g.N,dtype=np.uint32))&1).astype(np.float64)
        self.diag={k:np.zeros(self.dim,float) for k in KINDS}
        for b in g.bonds:self.diag[b.kind]+=self.occ[:,b.i]*self.occ[:,b.j]
        self._hop_info=[]
        data=[];rows=[];cols=[]
        for ib,b in enumerate(g.bonds):
            i,j=b.i,b.j
            assert i!=j
            mask=np.uint32((1<<i)|(1<<j))
            between_mask=np.uint32(sum(1<<p for p in range(min(i,j)+1,max(i,j))))
            valid=self.occ[:,i]!=self.occ[:,j]
            col=np.flatnonzero(valid)
            dest=self.states[col]^mask
            row=np.searchsorted(self.states,dest)
            if not np.array_equal(self.states[row],dest):raise AssertionError('Destination state absent')
            parity=np.array([int(int(s & between_mask).bit_count())&1 for s in self.states[col]])
            sign=(1-2*parity).astype(float)
            t=g.tmap[b.kind]
            rows.append(row);cols.append(col);data.append(t*sign)
            self._hop_info.append((row,col,sign))
        rows=np.concatenate(rows);cols=np.concatenate(cols);data=np.concatenate(data)
        self.Hhop=coo_matrix((data,(rows,cols)),shape=(self.dim,self.dim)).tocsr()
        self.Hhop.sum_duplicates()
        defect=self.Hhop-self.Hhop.T
        if defect.nnz and max(abs(defect.data))>1e-12:raise AssertionError('ED hopping not Hermitian')
        self.current_vec={}
        for name,signB in [('same',+1),('opposite',-1)]:
            weights=np.zeros(len(g.bonds),float)
            for c,ab,edges in g.triangles():
                s=1 if ab==0 else signB
                for ib,sgn in edges:weights[ib]+=s*sgn/(3*math.sqrt(6))
            self.current_vec[name]=weights

    def hamiltonian(self,Vtri,Vt1,Vt2):
        d=Vtri*self.diag['t']+Vt1*self.diag['t1']+Vt2*self.diag['t2']
        return self.Hhop+diags(d,format='csr')

    def current_action(self,name,psi):
        """Sum of physical CCW triangle currents / sqrt(6) acting on psi."""
        out=np.zeros(self.dim,complex)
        weights=self.current_vec[name]
        for ib,b in enumerate(self.g.bonds):
            w=weights[ib]
            if abs(w)<1e-16:continue
            row,col,sign=self._hop_info[ib]
            # J_(i->j) = +i*t*(ci^dag*cj - cj^dag*ci) for +t.
            # For oriented link b.i -> b.j the two occupations give +/-.
            occ_j=self.occ[col,b.j]
            factor=(2*occ_j-1)
            out[row]+=1j*w*self.g.tmap[b.kind]*sign*factor*psi[col]
        return out

    def free_ground_manifold(self,energy_tolerance=1e-9,max_states=64):
        """Construct the COMPLETE noninteracting Fermi-shell multiplet.

        Lanczos on a degenerate free-fermion problem can miss eigenvectors;
        making a ground-state mixture from those incomplete vectors gives
        false density modulation and LC correlations. Slater determinants
        of occupied one-body orbitals avoid that failure exactly at V=0.
        """
        eps,U=np.linalg.eigh(self.g.one_body())
        occupancies=[];Emin=float(np.sum(eps[:self.Nf]))
        Eexc=float('inf')
        for occ in itertools.combinations(range(self.g.N),self.Nf):
            E=float(sum(eps[a] for a in occ))
            if E<Emin+energy_tolerance:occupancies.append(occ)
            elif E < Eexc:Eexc=E
        if not occupancies:raise AssertionError('No free ground configurations')
        if len(occupancies)>max_states:
            return None,Emin,Eexc,len(occupancies)
        occupied_sites=np.column_stack([((self.states>>i)&1).astype(bool) for i in range(self.g.N)])
        # shape (Fock states, occupied sites per state), needed for Slater minors
        rows=np.stack([np.flatnonzero(occupied_sites[k]) for k in range(self.dim)])
        PS=np.empty((self.dim,len(occupancies)),complex)
        for k,occ in enumerate(occupancies):
            columns=U[:,occ]
            PS[:,k]=np.linalg.det(columns[rows])
        Gram=PS.conj().T@PS
        if np.max(abs(Gram-np.eye(len(occupancies))))>1e-8:
            raise AssertionError('Free Slater basis nonorthonormal')
        return PS,Emin,Eexc,len(occupancies)

    def charge_structure(self,psivectors):
        qlist=self.g.q
        avg_n=np.zeros(self.g.N,float)
        nn=np.zeros((self.g.N,self.g.N),float)
        for psi in psivectors.T:
            prob=abs(psi)**2
            avg_n+=prob@self.occ/psivectors.shape[1]
            nn+=(self.occ*prob[:,None]).T@self.occ/psivectors.shape[1]
        C=nn-np.outer(avg_n,avg_n)
        out={}
        for q in qlist:
            U=np.zeros((self.g.N,6),complex)
            for c,r in enumerate(self.g.reps):
                phase=np.exp(-2j*np.pi*np.dot(q,r))/math.sqrt(self.g.ncell)
                for a in range(6):U[c*6+a,a]=phase
            S=U.conj().T@C@U
            S=(S+S.conj().T)/2
            eig=np.linalg.eigvalsh(S)
            key=','.join(f'{x:.6g}' for x in q)
            out[key]={'max_eigenvalue':float(eig[-1]),'trace':float(np.trace(S).real),
                      'matrix_real':S.real.tolist(),'matrix_imag':S.imag.tolist()}
        return avg_n,out

    def solve(self,Vtri,Vt1,Vt2,k=4,tol=1e-10,maxiter=None,v0=None):
        H=self.hamiltonian(Vtri,Vt1,Vt2)
        if self.dim<=k+1:
            w,U=np.linalg.eigh(H.toarray())
            w=w[:k];U=U[:,:k]
        else:
            try:w,U=eigsh(H,k=min(k,self.dim-2),which='SA',v0=v0,tol=tol,maxiter=maxiter)
            except ArpackNoConvergence as ex:
                if ex.eigenvalues is None or len(ex.eigenvalues)<2:raise
                w,U=ex.eigenvalues,ex.eigenvectors
            idx=np.argsort(w);w=w[idx];U=U[:,idx]
        return w,U

    def diagnostics(self,w,U,deg_tol=1e-7):
        ndeg=int(np.count_nonzero(w-w[0]<deg_tol))
        P=U[:,:ndeg]
        density,S=self.charge_structure(P)
        cl=[]
        for name in ['same','opposite']:
            val=0.;mean=0.
            for p in P.T:
                a=self.current_action(name,p)
                value=np.vdot(p,a)
                if abs(value.imag)>1e-7:raise AssertionError('Non-Hermitian current')
                val+=float(np.vdot(a,a).real)/ndeg
                mean+=float(value.real)/ndeg
            cl.append((val-mean**2,mean))
        # Preserve q=0 and nonzero channel information without guessing an order.
        gamma=min(S,key=lambda x:sum(float(q)**2 for q in x.split(',')))
        other=[S[key]['max_eigenvalue'] for key in S if key!=gamma]
        return dict(ground_multiplet=ndeg,charge_mean=np.mean(density),
                    charge_density_std=float(np.std(density)),
                    charge_Sq_nonzero_max=float(max(other,default=0.)),
                    charge_Sq_all=S,
                    lc_same_equal_time=float(cl[0][0]),lc_opposite_equal_time=float(cl[1][0]),
                    lc_same_mean=float(cl[0][1]),lc_opposite_mean=float(cl[1][1]),
                    site_density=density.tolist())


def noninteracting_test(g,solver):
    single=np.linalg.eigvalsh(g.one_body())
    unfolded=g.folded_bands()
    err=float(np.max(abs(single-unfolded)))
    if err>1e-9:raise AssertionError(('Single-particle band folding failed',err))
    E_free=float(np.sum(single[:solver.Nf]))
    return dict(folded_spectrum_error=err,free_ED_energy_exact=E_free,
                q_primitive_fractional=g.q.tolist(),single_particle_spectrum=single.tolist())


def self_test():
    for name in PERIODS:
        g=RubyTorus(name,t=1.,t1=.43,t2=-.27)
        ed=FixedNED(g,2)
        test=noninteracting_test(g,ed)
        e,_=ed.solve(0.,0.,0.,k=3)
        assert abs(e[0]-test['free_ED_energy_exact'])<2e-9,(name,e[0],test)
        psi=np.random.default_rng(31).normal(size=ed.dim)+1j*np.random.default_rng(32).normal(size=ed.dim)
        psi=psi/np.linalg.norm(psi)
        for mode in ('same','opposite'):
            # Hermiticity tested by comparing two independent expectation orders.
            a=ed.current_action(mode,psi)
            phi=np.random.default_rng(15).normal(size=ed.dim)+1j*np.random.default_rng(16).normal(size=ed.dim)
            lhs=np.vdot(phi,a);rhs=np.vdot(ed.current_action(mode,phi),psi)
            assert abs(lhs-rhs)<1e-10,(mode,lhs,rhs)
        print('PASS',name,'Nsites',g.N,'edges',len(g.bonds),'wrapped',sum(any(b.wind) for b in g.bonds),
              'q',g.q.tolist(),'folding error',test['folded_spectrum_error'])
    a=RubyTorus('true_hex',t1=.43,t2=-.27)
    b=RubyTorus('skew18',t1=.43,t2=-.27)
    mismatch=max(abs(np.linalg.eigvalsh(a.one_body())-np.linalg.eigvalsh(b.one_body())))
    assert mismatch>1e-5,mismatch
    print('PASS: true hex and skew have DIFFERENT one-body spectra, max distance',mismatch)
    g=RubyTorus('true_hex',t=1.,t1=.5,t2=.5)
    ed=FixedNED(g,6)
    P,Eno,Eexc,d=ed.free_ground_manifold()
    assert d==6 and P is not None
    neutral=ed.diagnostics(np.repeat(Eno,d),P)
    assert neutral['charge_density_std']<1e-12, 'Must average complete free ground multiplet'
    print('PASS: complete 6-fold free multiplet has no artificial CDW modulation')
    for V,expected in [(0.,-9.162277660168),(1.,-7.993849322158)]:
        val,_=ed.solve(V,0,0,k=8)
        assert abs(val[0]-expected)<2e-9,(V,val[0],expected)
        print(f'PASS: 18-site known native ED reference V={V:g} E0={val[0]:+.12f}')
    print('SELF-TEST OK')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--torus',choices=PERIODS,default='true_hex',help='true_hex is identical to the repository hex18 after gauge relabeling; skew18 is an alternative in drawing coordinates')
    p.add_argument('--t',type=float,default=1.)
    p.add_argument('--t1',type=float,default=.5)
    p.add_argument('--t2',type=float,default=.5)
    p.add_argument('--Nf',type=int,default=6,help='Integer spinless fermion count on the 18-site torus')
    p.add_argument('--V',type=float,nargs='+',default=None,help='Explicit triangle-bond interaction list')
    p.add_argument('--Vmin',type=float,default=0.)
    p.add_argument('--Vmax',type=float,default=2.)
    p.add_argument('--Vstep',type=float,default=.25)
    p.add_argument('--V-t1',type=float,default=0.,dest='Vt1')
    p.add_argument('--V-t2',type=float,default=0.,dest='Vt2')
    p.add_argument('--n-eig',type=int,default=8)
    p.add_argument('--tol',type=float,default=1e-9)
    p.add_argument('--maxiter',type=int,default=None)
    p.add_argument('--deg-tol',type=float,default=1e-7)
    p.add_argument('--max-free-degenerate',type=int,default=64,help='Maximum analytic free multiplet dimension')
    p.add_argument('--skip-correlations',action='store_true')
    p.add_argument('--save-ground-state',action='store_true')
    p.add_argument('--no-draw',action='store_true')
    p.add_argument('--geometry-only',action='store_true')
    p.add_argument('--output',default='true_hex18_ed_scan')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    g=RubyTorus(a.torus,a.t,a.t1,a.t2)
    g.write_geometry(out)
    if not a.no_draw:g.plot(out)
    if a.geometry_only:
        print('Geometry written to',out);return
    ed=FixedNED(g,a.Nf)
    print(f'ED torus={a.torus}, Nsites={g.N}, Nf={a.Nf}, Hilbert dim={ed.dim}, '
          f'hopping bonds={len(g.bonds)}, wrapped={sum(any(b.wind) for b in g.bonds)}, '
          f'triangle V bonds={sum(b.kind=="t" for b in g.bonds)}',flush=True)
    test=noninteracting_test(g,ed)
    (out/'band_check.json').write_text(json.dumps(test,indent=2))
    if a.V is not None:
        vals=sorted(set(map(float,a.V)))
    else:
        if a.Vstep<=0:raise ValueError('Vstep must be positive')
        if a.Vmax<a.Vmin:raise ValueError('Vmax must be >= Vmin')
        vals=list(np.arange(a.Vmin,a.Vmax+0.5*a.Vstep,a.Vstep))
    rows=[];v0=None
    for i,V in enumerate(vals):
        ti=time.perf_counter()
        eig,U=ed.solve(float(V),a.Vt1,a.Vt2,k=a.n_eig,tol=a.tol,
                       maxiter=a.maxiter,v0=v0)
        v0=U[:,0].copy()
        record=dict(point=i,torus=a.torus,Nsites=g.N,Ncells=g.ncell,Nf=a.Nf,
                    filling_per_primitive=a.Nf/3,dimension=ed.dim,
                    t=a.t,t1=a.t1,t2=a.t2,V_tri=float(V),V_t1=a.Vt1,V_t2=a.Vt2,
                    E0=float(eig[0]),E0_per_primitive=float(eig[0]/3),
                    gap=float(eig[1]-eig[0]) if len(eig)>1 else None,
                    E_levels=list(map(float,eig)),runtime_seconds=time.perf_counter()-ti)
        correl_P=U[:,:max(1,int(np.count_nonzero(eig-eig[0]<a.deg_tol)))]
        record['multiplet_method']='Lanczos (finite-size ED)'
        ndegest=int(np.count_nonzero(eig-eig[0]<a.deg_tol))
        record['gap_above_ground_manifold']=(float(eig[ndegest]-eig[0])
                   if ndegest<len(eig) else None)
        record['possible_multiplet_truncation']=(ndegest==len(eig))
        # At V=0, exactly degenerate Slater determinants MUST be included in
        # the complete uniform ground mixture; scalar Lanczos is unreliable.
        if abs(V)<1e-14 and abs(a.Vt1)<1e-14 and abs(a.Vt2)<1e-14:
            P,Efree,Enext,ndeg=ed.free_ground_manifold(a.deg_tol,a.max_free_degenerate)
            record['free_manifold_dimension']=ndeg
            if abs(Efree-eig[0])>2e-7:raise AssertionError('ED free energy disagrees with bands')
            if P is not None:
                correl_P=P
                record['multiplet_method']='complete analytic Slater multiplet'
                record['gap']=float(0. if ndeg>1 else Enext-Efree)
                record['gap_above_ground_manifold']=float(Enext-Efree)
                record['possible_multiplet_truncation']=False
            else:
                record['multiplet_method']='INCOMPLETE Lanczos: free multiplet exceeds cap'
                record['possible_multiplet_truncation']=True
        if not a.skip_correlations:
            # Reconstruct a matching list of eigenvalues only for degeneracy
            # counting; the *vectors* always determine the observable.
            eobs=np.repeat(eig[0],correl_P.shape[1]) if 'analytic' in record['multiplet_method'] else eig
            details=ed.diagnostics(eobs,correl_P,a.deg_tol)
            record.update({k:v for k,v in details.items() if k not in ('charge_Sq_all','site_density')})
            (out/f'point_{i:04d}.json').write_text(json.dumps(dict(record,**details),indent=2))
        if a.save_ground_state:
            np.savez_compressed(out/f'ground_{i:04d}.npz',states=ed.states,psi=correl_P,
                                energy=eig,V_tri=V,multiplet_method=record['multiplet_method'])
        rows.append(record)
        print(f'[{i+1}/{len(vals)}] Vtri={V:g} E0={eig[0]:+.10f} '
              f'gap={record["gap"]:.4g} elapsed={record["runtime_seconds"]:.2f}s',flush=True)
        (out/'scan.json').write_text(json.dumps(rows,indent=2))
        with (out/'scan.csv').open('w',newline='') as f:
            cols=[k for k in rows[0] if k!='E_levels']
            writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print('Saved:',out/'scan.csv',flush=True)

if __name__=='__main__':main()