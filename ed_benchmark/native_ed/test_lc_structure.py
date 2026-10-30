"""Native LC all-q structure-factor tests, including exact Fock signs and degeneracy.

These tests run from ed_benchmark/native_ed where geometry.py and core.py
exist. They do not use QuSpin and do not require a ground-state ED run.
"""
import numpy as np
from geometry import RubyPBC
from core import make_states
from observables import op_sum,current_terms
from lc_structure import (triangle_index,one_triangle_action,lc_correlations,
                          momentum_structure,ground_jmax,pairwise_nonlocal,
                          summarize_shells)


def test_triangle_actions_vs_existing_observables():
    g=RubyPBC('ring3',t1=-.6,t2=.6)
    states=make_states(g.Ns,2)
    rng=np.random.default_rng(1223)
    psi=rng.normal(size=len(states))+1j*rng.normal(size=len(states))
    psi/=np.linalg.norm(psi)
    loops,_,_=triangle_index(g)
    assert len(loops)==6
    for loop in loops:
        actual=one_triangle_action(g,states,psi,loop)
        expected=op_sum(psi,states,g.Ns,current_terms(g,list(loop)))
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        assert abs(np.vdot(psi,actual).imag)<1e-12


def test_symmetry_basis_rotation_and_fourier_sum_rules():
    for name in ('ring3','3x1','2x2'):
        g=RubyPBC(name,t1=.6,t2=.2)
        states=make_states(g.Ns,2)
        rng=np.random.default_rng(1400+g.Ns)
        psis=rng.normal(size=(len(states),2))+1j*rng.normal(size=(len(states),2))
        psis,_=np.linalg.qr(psis)
        res=lc_correlations(g,states,psis)
        S=momentum_structure(g,res)
        assert S['S'].shape==(g.num_cells,2,2)
        assert res['C'].shape==(2*g.num_cells,2*g.num_cells)
        np.testing.assert_allclose(res['C_zero']+res['C_finite'],res['C'],atol=1e-10)
        for x in S['S']:
            assert min(np.linalg.eigvalsh(x))>-1e-10
        assert abs(sum(np.trace(z) for z in S['S'])-np.trace(res['C']))<1e-10
        U=np.array([[1,1j],[1j,1]],complex)/np.sqrt(2)
        changed=lc_correlations(g,states,psis@U)
        np.testing.assert_allclose(changed['C'],res['C'],atol=1e-11)
        np.testing.assert_allclose(changed['C_zero'],res['C_zero'],atol=1e-11)
        np.testing.assert_allclose(momentum_structure(g,changed)['S'],S['S'],atol=1e-11)
        J=ground_jmax(g,res)
        for mode in ('same','opposite'):
            if mode=='same':other=S['S_same_gamma'];zer=S['S_zero_same_gamma']
            else:other=S['S_opposite_gamma'];zer=S['S_zero_opposite_gamma']
            assert other>=-1e-10
            np.testing.assert_allclose(J[mode]['zero_weight'],zer,atol=1e-10)
        pairs=pairwise_nonlocal(g,res)
        assert len(pairs)==g.num_cells*(2*g.num_cells-1)
        groups=summarize_shells(pairs)
        assert sum(k['pairs'] for k in groups)==len(pairs)


def test_fixed_one_state_zero_modes_and_positive_nonlocal_not_assumed():
    g=RubyPBC('ring3')
    states=make_states(g.Ns,2)
    psi=np.ones(len(states),complex)/np.sqrt(len(states))
    res=lc_correlations(g,states,psi[:,None]);sq=momentum_structure(g,res)
    assert np.max(abs(res['C_zero']))<1e-12
    assert np.max(abs(sq['S_zero']))<1e-12
    assert np.min(np.linalg.eigvalsh(sq['S'][0]))>=-1e-12


if __name__=='__main__':
    for f in (test_triangle_actions_vs_existing_observables,
              test_symmetry_basis_rotation_and_fourier_sum_rules,
              test_fixed_one_state_zero_modes_and_positive_nonlocal_not_assumed):
        print('Testing',f.__name__,flush=True)
        f()
    print('All LC structure-factor tests passed',flush=True)