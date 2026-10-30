"""Regression tests without QuSpin. Run: python test_native.py"""
import json
import tempfile
from pathlib import Path
import numpy as np
from geometry import RubyPBC,twisted_c6_gauge
from core import FixedHamiltonian,make_states,lowest,full_hermiticity_error
from symmetry import symmetry_projectors,gauge_back,sector_labels,projected_basis
from observables import apply_ops,build_observables
from run_zeroT import main,parser
from response_iw import compute


def test_geometry():
    for name,nc in [('ring3',3),('3x1',3),('2x2',4)]:
        g=RubyPBC(name)
        assert g.Ns==6*nc
        assert len(g.edges)==12*nc
        assert [sum(len(c)==j for c in g.cycles()) for j in [3,4,6]]==[2*nc,3*nc,nc]
        assert g.point_maps()['mirror'] is not None


def test_fermion_hopping():
    g=RubyPBC('ring3');H=FixedHamiltonian(g,3,.7)
    assert full_hermiticity_error(H.operator)<1e-13
    for ie in (0,4,18,21):
        # explicitly compare constructed sparse to one-body second-quantized action
        e=g.edges[ie]
        from core import edge_triplets
        row,col,sign=edge_triplets(g,H.states,ie)
        rng=np.random.default_rng(13)
        psi=rng.normal(size=H.n)+1j*rng.normal(size=H.n)
        jright=apply_ops(psi,H.states,H.states,g.Ns,[('+',e.i),('-',e.j)])
        jleft=apply_ops(psi,H.states,H.states,g.Ns,[('+',e.j),('-',e.i)])
        target=np.zeros(H.n,complex)
        for r,c,s in zip(row,col,sign): target[r]+=s*psi[c]
        assert np.max(abs(target-jright-jleft))<1e-12
    free=FixedHamiltonian(g,6,0.)
    evals=lowest(free.operator,1)[0]
    assert abs(evals[0]+9.16227766016838)<1e-9


def test_all_symmetry_dimensions():
    for name in ('ring3','3x1','2x2'):
        for t1,t2 in ((.5,.5),(.6,.2),(.5,-.5)):
            g=RubyPBC(name,t1=t1,t2=t2)
            for kind in ('translation','point'):
                qs=list(symmetry_projectors(g,2,kind))
                assert sum(Q.shape[1] for label,Q,states,z in qs)==len(make_states(g.Ns,2))


def test_symmetry_energies():
    g=RubyPBC('ring3')
    energies=[]
    for sym in ('none','translation','point'):
        from run_zeroT import solve_all
        H,records,found,st,z=solve_all(g,3,1.,symmetry=sym,low=2)
        energies.append(found[0][0])
    assert np.max(energies)-np.min(energies)<1e-8
    g=RubyPBC('ring3',t1=.5,t2=-.5)
    from run_zeroT import solve_all
    E=[]
    for untwisted in (False,True):
        H,rec,found,st,z=solve_all(g,3,.75,symmetry='point',no_twisted=untwisted)
        E.append(found[0][0])
        assert any(r['labels']['signed_rotation'] for r in rec) == (not untwisted)
        ps=np.asarray(found[0][2]@found[0][3]).ravel()
        raw=FixedHamiltonian(g,3,.75,states=st)
        assert np.linalg.norm(raw.operator@ps-E[-1]*ps)<1e-7
    assert max(E)-min(E)<1e-8


def test_observables_and_green():
    with tempfile.TemporaryDirectory() as tmp:
        flags=parser().parse_args(['--cluster','3x1','--nf','6','--V','0',
                                  '--symmetry','point','--save-psi','--out',tmp])
        r=main(flags)
        assert abs(r['E0']+9.614627004195)<1e-8
        # NpzFile holds an OS file handle. Windows forbids removal of open
        # files, so close it before TemporaryDirectory cleans up.
        with np.load(Path(tmp)/'equal_time.npz') as data:
            assert abs(data['density'].sum()-6.)<1e-8
            assert np.max(abs(data['rho']-data['rho'].conj().T))<1e-10
            assert abs(data['current_mean']).max()<1e-8  # unique TR-symmetric ground state
        G=compute(tmp,orbitals=[0,1,2],nfreq=4,krylov=60)
        g=RubyPBC('3x1')
        h=np.zeros((g.Ns,g.Ns))
        for e in g.edges:
            h[e.i,e.j]+=g.t[e.kind];h[e.j,e.i]+=g.t[e.kind]
        with np.load(Path(tmp)/'green_iw.npz') as f:
            for omega,Gz in zip(f['omega'],G):
                exact=np.linalg.inv((1j*omega+f['mu'])*np.eye(g.Ns)-h)[:3,:3]
                assert np.max(abs(Gz-exact))<1e-9


if __name__=='__main__':
    for f in (test_geometry,test_fermion_hopping,test_all_symmetry_dimensions,
              test_symmetry_energies,test_observables_and_green):
        print('Testing',f.__name__,flush=True)
        f()
    print('All native ED regression checks passed')