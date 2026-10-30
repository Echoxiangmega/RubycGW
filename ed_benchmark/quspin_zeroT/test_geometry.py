"""Numerical tests run WITHOUT QuSpin; full ED requires QuSpin separately."""
from math import comb
import numpy as np
from geometry import RubyPBC
from observables import apply_ops,build_observables
from scipy.sparse import coo_matrix


def test_lattice_and_loops():
    for label,nc in [('2x2',4),('3x1',3),('ring3',3)]:
        g=RubyPBC(label)
        assert len(g.edges)==12*nc
        loops=g.cycles()
        assert [sum(len(c)==j for c in loops) for j in [3,4,6]]==[2*nc,3*nc,nc]
        assert len(loops)==g.Ns # Euler characteristic of torus V-E+F=0
        assert len(g.permitted_momenta())==nc
        assert g.point_maps()['mirror'] is not None
        assert g.point_maps()['C2'] is not None
        if label!='3x1':assert g.point_maps()['C6'] is not None
        else:assert g.point_maps()['C6'] is None
        for p in g.point_maps().values():
            if p is None:continue
            assert sorted(p.tolist())==list(range(g.Ns))


def test_no_false_rotation_if_t1_ne_t2():
    for label in ['2x2','ring3']:
        g=RubyPBC(label,t1=.5,t2=.3)
        assert g.point_maps()['C6'] is None


def test_fermion_operator_anticommutation():
    N=6
    for nf in (1,2,3):
        st=np.array([b for b in range(1<<N) if b.bit_count()==nf][::-1],dtype=np.uint64)
        v=np.arange(len(st),dtype=np.complex128)+1j*np.arange(len(st))[::-1]
        for a in range(N):
            for b in range(N):
                x=apply_ops(v,st,st,N,[('+',a),('-',b)])
                y=apply_ops(v,st,st,N,[('-',b),('+',a)])
                assert np.allclose(x+y,v if a==b else 0)


def test_one_particle_spectrum_matches_folded_bands():
    for label in ['2x2','3x1','ring3']:
        g=RubyPBC(label)
        h=np.zeros((g.Ns,g.Ns),complex)
        for e in g.edges:
            h[e.i,e.j]+=g.t[e.kind]
            h[e.j,e.i]+=g.t[e.kind]
        bands=[]
        for k in g.permitted_momenta():
            hk=np.zeros((6,6),complex)
            for a,b,dx,dy,kind in __import__('geometry').BONDS:
                z=g.t[kind]*np.exp(2j*np.pi*np.dot(k,[dx,dy]))
                hk[a,b]+=z;hk[b,a]+=z.conjugate()
            bands.extend(np.linalg.eigvalsh(hk))
        assert np.max(np.abs(np.sort(np.linalg.eigvalsh(h))-np.sort(bands)))<1e-11


def test_equal_time_single_particle_no_flux():
    g=RubyPBC('ring3');n=g.Ns
    st=np.array([1<<(n-1-i) for i in range(n)],dtype=np.uint64)
    st.sort();st=st[::-1]
    # all-one single-particle wavefunction: rho is real -> no imaginary flux
    psi=np.ones(n,complex)/np.sqrt(n)
    data=build_observables(g,psi,st,pairs=False)
    assert abs(data['density'].sum()-1)<1e-12
    assert np.allclose(data['current_mean'],0)
    assert np.allclose(data['rho'],data['rho'].conj().T)


if __name__=='__main__':
    test_lattice_and_loops()
    test_no_false_rotation_if_t1_ne_t2()
    test_fermion_operator_anticommutation()
    test_one_particle_spectrum_matches_folded_bands()
    test_equal_time_single_particle_no_flux()
    print('All geometry and fermion-sign tests passed (no QuSpin needed).')