"""Tests for the independent zero-T V-scan susceptibility workflow."""
import json
import tempfile
from pathlib import Path
import numpy as np
from geometry import RubyPBC
from core import FixedHamiltonian
from scan_susceptibility import (VScanSolver,channel_specs,one_channel,source_action,
    spectral_chi,charge_real_sources,pseudospin_e_charge_modes,pairing_response,parser,run)



def test_hf_physical_pseudospin_charge_modes():
    """Six-site HF density patterns, B physical tau_y sign, orthogonal sectors."""
    omega=np.exp(2j*np.pi/3)
    for name in ('ring3','3x1','2x2'):
        g=RubyPBC(name)
        modes=pseudospin_e_charge_modes(g)
        names=['CO_E_even_x','CO_E_even_y','CO_E_odd_x','CO_E_odd_y']
        W=np.array([modes[n] for n in names])
        np.testing.assert_allclose(W@W.T,np.eye(4),atol=1e-12)
        scale=np.sqrt(12*g.num_cells)
        np.testing.assert_allclose(W[0,:6]*scale,[2,-1,-1,2,-1,-1])
        np.testing.assert_allclose(W[1,:6]*scale,[0,-np.sqrt(3),np.sqrt(3),
                                                 0,np.sqrt(3),-np.sqrt(3)])
        np.testing.assert_allclose(W[2,:6]*scale,[2,-1,-1,-2,1,1])
        np.testing.assert_allclose(W[3,:6]*scale,[0,-np.sqrt(3),np.sqrt(3),
                                                 0,-np.sqrt(3),np.sqrt(3)])
        # The local doublet CO along x is a one-rich/two-poor density pattern.
        nA=np.array([2/3,1/6,1/6])
        assert abs(2*nA[0]-nA[1]-nA[2]-1.)<1e-12
        assert abs(np.sqrt(3)*(nA[2]-nA[1]))<1e-12
        nB=np.array([1/3,1/3-1/(2*np.sqrt(3)),1/3+1/(2*np.sqrt(3))])
        # For physical B tau_y this yields negative unity, not positive.
        assert abs(np.sqrt(3)*(nB[1]-nB[2])+1.)<1e-12
        qa=nA[0]+omega**2*nA[1]+omega*nA[2]
        np.testing.assert_allclose(qa,(1.+0j)/2,atol=1e-12)


def test_against_dense_lehmann():
    g=RubyPBC('ring3',t1=.6,t2=.2)
    solver=VScanSolver(g,2,symmetry='point',low=4)
    for V in (0.,.75,-.6):
        H,E0,psis,records,incomplete,resid=solver.solve(V,1e-8)
        assert not incomplete
        assert resid<1e-7
        assert psis.shape[1]==1
        w,U=np.linalg.eigh(H.toarray())
        assert abs(w[0]-E0)<1e-8
        channels,lead=channel_specs(g,solver.occ,psis)
        for ch in channels:
            actual=one_channel(ch,H,E0,psis,solver.states,solver.occ,g,
                               depth=H.shape[0],precision=1e-8)
            expected=0.
            for src in source_action(ch,psis[:,0],solver.states,g.Ns,solver.occ,g):
                src=src-np.vdot(psis[:,0],src)*psis[:,0]
                z=U.conj().T@src
                expected+=2*np.sum(abs(z[1:])**2/(w[1:]-w[0]))
            assert actual['status']=='ok',(ch['name'],actual)
            assert abs(actual['chi_static']-expected)<3e-7,(ch['name'],V,actual['chi_static'],expected)


def test_degenerate_response_flags():
    g=RubyPBC('ring3')
    solver=VScanSolver(g,6,'point',low=5)
    H,e0,psis,_,partial,res=solver.solve(0,1e-7)
    assert psis.shape[1]>=2
    channels,_=channel_specs(g,solver.occ,psis)
    results={ch['name']:one_channel(ch,H,e0,psis,solver.states,solver.occ,g,
        depth=25,precision=1e-7,ground_incomplete=partial) for ch in channels[:4]}
    for v in results.values():
        assert np.isfinite(v['chi_regular'])
        if v['status']!='ok':assert np.isnan(v['chi_static'])



def test_pairing_against_sparse_resolvent():
    # Small Nf gives Nf+2=4 (3060 basis states): direct sparse linear solve
    # is an independent check of particle-number-changing Lanczos poles.
    from scipy.sparse.linalg import spsolve
    from core import make_states
    from observables import apply_ops
    g=RubyPBC('ring3',t1=.6,t2=.2)
    nf=2;V=.7
    solver=VScanSolver(g,nf,'point',low=4)
    H,e0,psis,rec,partial,res=solver.solve(V,1e-8)
    rsp,sp,meta=pairing_response(g,nf,V,e0,psis,solver.states,
                                depth=100,pole_tol=1e-9)
    assert rsp['status']=='ok',rsp
    mode=np.array([x[0]+1j*x[1] for x in meta['mode_weights']])
    HM=FixedHamiltonian(g,nf-2,V).operator
    HP=FixedHamiltonian(g,nf+2,V).operator
    minus=make_states(g.Ns,nf-2);plus=make_states(g.Ns,nf+2)
    exact=0.
    for psi in psis.T:
        fm=np.zeros(len(minus),complex);fp=np.zeros(len(plus),complex)
        for z,(i,j) in zip(mode,meta['mode_bonds']):
            fm+=z*apply_ops(psi,solver.states,minus,g.Ns,[('-',i),('-',j)])
            fp+=np.conj(z)*apply_ops(psi,solver.states,plus,g.Ns,[('+',j),('+',i)])
        em=e0-2*meta['mu'];ep=e0+2*meta['mu']
        exact+=(2*np.vdot(fm,spsolve(HM-em*__import__('scipy').sparse.eye(HM.shape[0]),fm)).real
                +2*np.vdot(fp,spsolve(HP-ep*__import__('scipy').sparse.eye(HP.shape[0]),fp)).real)/psis.shape[1]
    assert abs(exact-rsp['chi_static'])/(1+abs(exact))<1e-8,(exact,rsp)


def test_scan_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        a=parser().parse_args(['--cluster','ring3','--nf','2','--V','0','0.75',
             '--t1','0.6','--t2','0.2','--depth','20','--no-extra-loops',
             '--output',tmp])
        run(a)
        with (Path(tmp)/'susceptibility.csv').open(encoding='utf-8') as f:
            text=f.read()
        assert 'LC_same_chi' in text and 'CO_E_even_chi' in text
        assert 'CO_E_odd_chi' in text and 'CO_E_odd_y_chi' in text
        assert (Path(tmp)/'chi_overview.png').exists()
        run(a)  # reuses cached points; must not change output schema
        with (Path(tmp)/'susceptibility.csv').open(encoding='utf-8') as f:
            assert f.read()==text


def test_n_grid_validation():
    with tempfile.TemporaryDirectory() as tmp:
        a=parser().parse_args(['--cluster','ring3','--n','0.5','--V','0',
                               '--output',tmp])
        try:run(a)
        except ValueError as e:assert 'must be an integer' in str(e)
        else:raise AssertionError('Noninteger total particle number accepted')


if __name__=='__main__':
    for test in (test_hf_physical_pseudospin_charge_modes,
                 test_against_dense_lehmann,test_degenerate_response_flags,
                 test_pairing_against_sparse_resolvent,test_scan_and_resume,test_n_grid_validation):
        print('Testing',test.__name__,flush=True)
        test()
    print('All susceptibility scan checks passed')