"""Independent regression tests for ED charge-density and site-variance plots."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

from geometry import RubyPBC
from core import make_states
from scan_susceptibility import VScanSolver,pseudospin_e_charge_modes
from plot_charge_pattern import (projected_site_densities,density_for_coeff,
          spatial_measure,optimize_quartic,optimized_charge_state,export_one)


def test_on_site_variance_not_CO():
    g=RubyPBC('ring3')
    states=make_states(g.Ns,2)
    # An exactly uniform real pure Fock superposition: local fluctuations
    # are nonzero, but all *expectation* densities have no spatial contrast.
    psi=np.ones(len(states),complex)/np.sqrt(len(states))
    M,occ=projected_site_densities(states,psi[:,None],g.Ns)
    n=density_for_coeff(M,np.array([1.]))
    assert np.allclose(n,2/g.Ns)
    assert abs(spatial_measure(n,'triangle'))<1e-15
    assert np.all(n*(1-n)>0)
    # For spinless n_i^2=n_i, so local variance = n_i(1-n_i).
    prob=abs(psi)**2
    assert np.max(abs(prob@occ**2-(prob@occ)**2-n*(1-n)))<1e-12


def test_ground_manifold_basis_and_charge_optimum():
    g=RubyPBC('ring3')
    states=make_states(g.Ns,2)
    # Two orthogonal occupancy basis vectors, intentionally different
    # triangle-internal charge distributions. Not claimed to be eigenstates.
    psis=np.zeros((len(states),2),complex)
    for i,occup in enumerate([(0,3),(1,4)]):
        bits=sum(1<<(g.Ns-1-j) for j in occup)
        k=int(np.searchsorted(states,bits));assert int(states[k])==bits
        psis[k,i]=1
    D=optimized_charge_state(g,states,psis,restarts=5,check_h=False)
    assert D['ground_dim']==2
    assert D['selected_tri_spatial_var']>D['ensemble_tri_spatial_var']
    assert D['selected_tri_spatial_var']>0.07
    assert np.allclose(np.sum(D['n_selected']),2)
    ds=optimized_charge_state(g,states,psis,restarts=0,check_h=False,
                              source='CO_E_odd_x',polarity='positive')
    assert ds['selection_source']=='CO_E_odd_x'
    assert ds['ground_dim']==2
    assert np.allclose(D['mean_onsite_var_selected'],0,atol=1e-7)
    for p in D['proposals']:
        assert np.isclose(np.linalg.norm(p['vector']),1)
    U=np.array([[1,1j],[1j,1]],complex)/np.sqrt(2)
    D2=optimized_charge_state(g,states,psis@U,restarts=5,check_h=False)
    assert abs(D2['selected_tri_spatial_var']-D['selected_tri_spatial_var'])<1e-7


def test_HF_E_projection_and_fock_occupation():
    g=RubyPBC('3x1',t1=.6,t2=.2)
    solver=VScanSolver(g,2,'point',low=4)
    H,e0,psis,rec,partial,res=solver.solve(.7,1e-9)
    D=optimized_charge_state(g,solver.states,psis,e0,.7,restarts=2)
    assert D['ground_residual']<1e-7
    assert np.isclose(D['n_ensemble'].sum(),2)
    # Every fixed HF pseudospin charge source has an exact projected O matrix.
    for name,w in pseudospin_e_charge_modes(g).items():
        O=np.einsum('i,iab->ab',w,D['m'])
        eig=np.linalg.eigvalsh(O)
        proposal=[p for p in D['proposals'] if p['name']==name]
        assert len(proposal)==2
        np.testing.assert_allclose(sorted(p['O_value'] for p in proposal),[eig[0],eig[-1]],atol=1e-9)
    with tempfile.TemporaryDirectory() as tmp:
        f=Path(tmp)/'point'
        export_one(g,solver.states,psis,e0,.7,f,restarts=2)
        for name in ('site_density.png','density_contrast.svg','charge_covariance.png',
                     'onsite_quantum_variance.png','triangle_charge.png',
                     'charge_summary.json','site_density.csv','triangle_charge.csv','charge_pattern.npz'):
            assert (f/name).exists(),name
        assert (f/'projected_charge_sources.csv').exists()
        with np.load(f/'charge_pattern.npz') as z:
            assert z['charge_covariance_ensemble'].shape==(g.Ns,g.Ns)
            assert np.allclose(z['quantum_var_ensemble'],z['n_ensemble']*(1-z['n_ensemble']))
        saved=json.loads((f/'charge_summary.json').read_text())
        assert saved['ground_dim']==psis.shape[1]


def test_cli_entrypoint():
    p=Path(__file__).with_name('plot_charge_pattern.py')
    x=subprocess.run([sys.executable,str(p),'--help'],capture_output=True,text=True,timeout=30)
    assert x.returncode==0,x.stderr
    assert all(k in x.stdout for k in ('--input','--V','--kind','--restarts','--source'))


if __name__=='__main__':
    for f in (test_on_site_variance_not_CO,test_ground_manifold_basis_and_charge_optimum,
              test_HF_E_projection_and_fock_occupation,test_cli_entrypoint):
        print('Testing',f.__name__,flush=True)
        f()
    print('All charge-pattern tests passed')