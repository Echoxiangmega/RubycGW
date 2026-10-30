"""Tests for the independent zero-T V-scan susceptibility workflow."""
import json
import tempfile
from pathlib import Path
import numpy as np
from geometry import RubyPBC
from core import FixedHamiltonian
from scan_susceptibility import (VScanSolver,channel_specs,one_channel,source_action,
    spectral_chi,charge_real_sources,pairing_response,parser,run,
    E_CHANNEL_NAMES,e_charge_response_matrix,first_active_gap,level_gap_diagnostics)


def test_cli_entrypoint_runs():
    """Regression: an importable module still needs a runnable CLI entrypoint."""
    import subprocess
    import sys
    cmd=[sys.executable,str(Path(__file__).with_name('scan_susceptibility.py')),'--help']
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,(result.stdout,result.stderr)
    assert all(option in result.stdout for option in ('--cluster','--n','--depth','--Vmin'))


def test_full_e_response_against_dense_lehmann():
    """Check all 16 susceptibility matrix elements, including off-diagonals."""
    from scan_susceptibility import pseudospin_e_charge_modes
    g=RubyPBC('ring3',t1=.6,t2=.2)
    solver=VScanSolver(g,2,'point',low=4)
    H,e0,psis,records,partial,res=solver.solve(.7,1e-9)
    assert not partial and psis.shape[1]==1
    es,U=np.linalg.eigh(H.toarray())
    channels,_=channel_specs(g,solver.occ,psis,include_extra=False)
    lookup={ch['name']:ch for ch in channels}
    capture={};states={};diags={}
    for name in E_CHANNEL_NAMES:
        rr=one_channel(lookup[name],H,e0,psis,solver.states,solver.occ,g,
                       depth=H.shape[0],precision=1e-10,capture=True)
        capture[name]=rr['captured'];states[name]=rr['status']
        diags[name]=rr['chi_regular']
    result=e_charge_response_matrix(capture,psis,states)
    source=np.column_stack([source_action(lookup[name],psis[:,0],solver.states,
                             g.Ns,solver.occ,g)[0] for name in E_CHANNEL_NAMES])
    source-=psis@(psis.conj().T@source)
    amp=U.conj().T@source
    exact=2*np.real(amp[1:].conj().T@((1/(es[1:]-es[0]))[:,None]*amp[1:]))
    np.testing.assert_allclose(result['matrix_regular'],exact,rtol=5e-9,atol=1e-9)
    np.testing.assert_allclose(np.diag(result['matrix_regular']),
                              [diags[x] for x in E_CHANNEL_NAMES],atol=1e-9)
    assert result['status']=='ok'
    assert result['leading_rank']>=1
    assert abs(result['leading_even_weight']+result['leading_odd_weight']-1)<1e-11
    assert np.allclose(result['leading_projector']@result['leading_projector'],
                       result['leading_projector'],atol=1e-10)
    assert abs(result['matrix_regular'][0,3])>1e-3  # genuine E mixing!


def test_gap_and_zero_mode_diagnostics():
    from scan_susceptibility import pseudospin_e_charge_modes
    g=RubyPBC('ring3',t1=.6,t2=.2)
    solver=VScanSolver(g,2,'point',low=4)
    H,e0,psis,records,partial,res=solver.solve(.7,1e-9)
    out=level_gap_diagnostics(records,e0,1e-8)
    assert out['gap_E1']>=0
    assert out['gap_above_manifold']>0
    channels,_=channel_specs(g,solver.occ,psis,include_extra=False)
    ch=next(x for x in channels if x['name']=='CO_E_odd_x')
    response=one_channel(ch,H,e0,psis,solver.states,solver.occ,g,
                         80,1e-10,capture=True)
    d,w=first_active_gap(response['spectra'],1e-10,1e-3)
    assert np.isfinite(d) and d>0 and w>0
    # No ground-state overlap in a nondegenerate eigenstate after subtraction.
    src=response['captured'][0][0][0]
    assert abs(np.vdot(psis[:,0],src))<1e-9


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
        assert 'LC_same_chi' in text and 'CO_leading_chi' in text
        assert 'CO_E_matrix_chi' in text and 'gap_E1' in text
        assert (Path(tmp)/'chi_overview.png').exists()
        assert (Path(tmp)/'gap_levels.png').exists()
        assert (Path(tmp)/'gap_active.png').exists()
        assert (Path(tmp)/'ground_degeneracy.png').exists()
        assert (Path(tmp)/'chi_E_matrix_eigenvalues.png').exists()
        assert (Path(tmp)/'chi_lc_regular.png').exists()
        assert (Path(tmp)/'lc_ground_Jmax.png').exists()
        assert (Path(tmp)/'charge_mode_identity.png').exists()
        from plot_e_matrix import main as plot_e_main
        import sys
        original=sys.argv
        try:
            sys.argv=['plot_e_matrix.py','--input',tmp,'--V','0.75']
            plot_e_main()
        finally:
            sys.argv=original
        assert list(Path(tmp).glob('V_*/chi_E_matrix.png'))
        with np.load(next(Path(tmp).glob('V_*/lc_ground_projected.npz'))) as z:
            assert z['LC_same_matrix'].shape==(1,1)
            assert z['LC_opposite_eigenvalues'].shape==(1,)
        with np.load(next(Path(tmp).glob('V_*/co_e_matrix.npz'))) as z:
            assert z['chi_regular'].shape==(4,4)
            assert z['leading_projector'].shape==(4,4)
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


def test_projected_current_manifold_against_zero_mode():
    """The full ground-matrix Frobenius norm equals the ED zero-mode W0."""
    from scan_susceptibility import projected_current_manifold,lc_triangle_channels
    g=RubyPBC('ring3',t1=.6,t2=.2)
    sol=VScanSolver(g,2,symmetry='point',low=4)
    H,E0,psis,records,incomplete,resid=sol.solve(.7,1e-9)
    current=projected_current_manifold(g,psis,sol.states)
    for name,ch in lc_triangle_channels(g).items():
        resp=one_channel(ch,H,E0,psis,sol.states,sol.occ,g,depth=H.shape[0],
                         precision=1e-9)
        np.testing.assert_allclose(current[name]['zero_weight'],resp['zero_mode_weight'],atol=1e-10)
        eig=np.linalg.eigvalsh(current[name]['matrix'])
        np.testing.assert_allclose(current[name]['jmax_abs'],max(abs(eig)),atol=1e-10)
    # Test a nontrivial two-state subspace; its current matrix need not be diagonal.
    st=sol.states
    vec=np.random.default_rng(1).normal(size=(len(st),2))+1j*np.random.default_rng(2).normal(size=(len(st),2))
    vec,_=np.linalg.qr(vec)
    cur=projected_current_manifold(g,vec,st)
    for name,ch in lc_triangle_channels(g).items():
        resp=one_channel(ch,H,E0,vec,st,sol.occ,g,depth=7,precision=1e-9)
        np.testing.assert_allclose(cur[name]['zero_weight'],resp['zero_mode_weight'],atol=1e-10)
        assert np.linalg.norm(cur[name]['matrix']-cur[name]['matrix'].conj().T)<1e-12


if __name__=='__main__':
    for test in (test_cli_entrypoint_runs,
                 test_projected_current_manifold_against_zero_mode,
                 test_full_e_response_against_dense_lehmann,
                 test_gap_and_zero_mode_diagnostics,
                 test_against_dense_lehmann,test_degenerate_response_flags,
                 test_pairing_against_sparse_resolvent,test_scan_and_resume,test_n_grid_validation):
        print('Testing',test.__name__,flush=True)
        test()
    print('All susceptibility scan checks passed')

