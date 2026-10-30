"""Check ordinary/twisted Ruby point symmetries, independently of QuSpin.

Tests all 12 representative hopping and every density interaction on the torus,
including fermion signs and twisted C6 projective closure for odd/even N.
"""
import itertools
from math import comb
import numpy as np
from scipy.sparse import coo_matrix
from geometry import RubyPBC
from run_zeroT import sectors, point_preserves_momentum, translation_generators


def manybody_action(p, signs, Ns, Nf):
    """Unitary U c^dag_i U^-1=signs[i] c^dag_{p[i]} in fixed Nf space."""
    occs=list(itertools.combinations(range(Ns),Nf))
    which={s:i for i,s in enumerate(occs)}
    columns=[]; data=[]
    for occ in occs:
        dest=[int(p[i]) for i in occ]
        parity=(-1)**sum(dest[x]>dest[y] for x in range(len(dest)) for y in range(x+1,len(dest)))
        phase=int(np.prod(signs[list(occ)])) if occ else 1
        columns.append(which[tuple(sorted(dest))]); data.append(parity*phase)
    return coo_matrix((data,(columns,np.arange(len(occs)))),shape=(len(occs),len(occs))).tocsr()


def h_fixed_Nf(g,Nf,V):
    """Independent sparse many-body Hamiltonian, with standard fermion signs."""
    Ns=g.Ns; occs=list(itertools.combinations(range(Ns),Nf))
    ind={o:i for i,o in enumerate(occs)}; rows=[];cols=[]; vals=[]
    for col,oc in enumerate(occs):
        st=set(oc); diag=0.
        for e in g.edges:
            if e.kind=='t' and e.i in st and e.j in st:diag+=V
            if (e.i in st)==(e.j in st):continue
            src,dst=(e.i,e.j) if e.i in st else (e.j,e.i)
            nextocc=tuple(sorted(st-{src}|{dst}))
            sign=(-1)**sum(min(src,dst)<i<max(src,dst) for i in st)
            rows.append(ind[nextocc]);cols.append(col);vals.append(sign*g.t[e.kind])
        rows.append(col);cols.append(col);vals.append(diag)
    return coo_matrix((vals,(rows,cols)),shape=(len(occs),len(occs))).tocsr()


def verify_unitary_commutes(g, p, sign, nf, V=.7):
    U=manybody_action(p,sign,g.Ns,nf)
    H=h_fixed_Nf(g,nf,V)
    error=U@H-H@U
    assert error.nnz==0 or np.max(np.abs(error.data))<1e-10
    assert (U.T@U).diagonal().min()==1
    return U


def test_symmetry_classes():
    for t1,t2 in ((.5,.5),(.5,-.5),(.6,.2)):
        for name in ('2x2','3x1','ring3'):
            g=RubyPBC(name,t1=t1,t2=t2)
            point=g.point_maps();signed=g.gauge_point_maps()
            same=abs(t1-t2)<1e-10
            opposite=abs(t1+t2)<1e-10
            if name!='3x1':
                assert point['C3'] is not None
                assert point['C6'] is not None if same else point['C6'] is None
                assert sorted(map(int,point['mirrors']))==([0,30,60,90,120,150] if same else [0,60,120])
                assert (signed['C6'] is not None)==opposite
            else:
                assert point['C3'] is None
                assert point['C6'] is None
                assert (point['C2'] is not None)==same
                assert '120' in point['mirrors']
                assert (signed['C2'] is not None)==opposite
            # Matrix-level symmetry with FULL fermionic exchange signs.
            for p in (point['C3'],point['C2'],point['C6'],point['mirror']):
                if p is not None:verify_unitary_commutes(g,p,np.ones(g.Ns,dtype=int),2)
            if signed['C6'] is not None:
                p,s= signed['C6']
                U1=verify_unitary_commutes(g,p,s,1)
                U2=verify_unitary_commutes(g,p,s,2)
                assert (U1**6+__import__('scipy').sparse.eye(g.Ns,format='csr')).nnz==0
                assert (U2**6-__import__('scipy').sparse.eye(comb(g.Ns,2),format='csr')).nnz==0
            if signed['C2'] is not None:
                p,s=signed['C2'];verify_unitary_commutes(g,p,s,2)
            # No equal strength assumption needed for generic C3 + mirror.
            allsectors=list(sectors(g,True))
            assert len(allsectors)>0
            assert len(set(str(x) for x in allsectors))==len(allsectors)
            print(f'PASS {name:5s} t1={t1:+.2f} t2={t2:+.2f} '
                  f'plain C6/C3/C2={tuple(point[k] is not None for k in ("C6","C3","C2"))}, '
                  f'mirrors={list(point["mirrors"])}, signed C6={signed["C6"] is not None}, '
                  f'blocks={len(allsectors)}')


if __name__=='__main__':
    test_symmetry_classes()
    print('All ordinary/twisted symmetry and fermionic-sign checks passed.')


def test_twisted_gauge_as_plain_permutation():
    from geometry import twisted_c6_gauge
    from run_zeroT import undo_one_body_gauge
    from scipy.linalg import eigh
    for name in ('2x2','ring3'):
        g=RubyPBC(name,t1=.5,t2=-.5)
        p,z=twisted_c6_gauge(g)
        assert len(p)==g.Ns
        H=np.zeros((g.Ns,g.Ns),complex)
        for e in g.edges:
            H[e.i,e.j]+=g.t[e.kind]
            H[e.j,e.i]+=g.t[e.kind]
        Hprime=z[:,None]*H*z.conjugate()[None,:]
        assert np.allclose(Hprime,Hprime.conj().T)
        assert np.max(np.abs(Hprime-Hprime[np.ix_(p,p)]))<1e-10
        assert np.max(np.abs(np.linalg.eigvalsh(H)-np.linalg.eigvalsh(Hprime)))<1e-10
        # Check Fock-space transformation with nontrivial fermionic signs.
        Hn=h_fixed_Nf(g,2,.7).toarray()
        occs=list(itertools.combinations(range(g.Ns),2))
        diag=np.array([np.prod(z[list(o)]) for o in occs])
        Hnprime=diag[:,None]*Hn*diag.conjugate()[None,:]
        Uplain=manybody_action(p,np.ones(g.Ns,dtype=int),g.Ns,2).toarray()
        assert np.max(np.abs(Hnprime@Uplain-Uplain@Hnprime))<1e-10
        assert np.max(np.abs(np.linalg.eigvalsh(Hn)-np.linalg.eigvalsh(Hnprime)))<1e-10
        ev,Q=np.linalg.eigh(Hnprime)
        psi_old=diag.conjugate()*Q[:,0]
        assert np.linalg.norm(Hn@psi_old-ev[0]*psi_old)<1.e-9
        # Same check against original QuSpin ordering convention in wavefunction conversion.
        states=np.array([sum(1<<(g.Ns-1-i) for i in occ) for occ in occs],dtype=np.uint64)
        psi_converted=undo_one_body_gauge(Q[:,0],states,z,g.Ns)
        assert np.allclose(psi_converted,psi_old)
    return True


if __name__=='__main__':
    test_twisted_gauge_as_plain_permutation()
    print('Gauge-rotated C6 also passes full fixed-N matrix and wavefunction checks.')