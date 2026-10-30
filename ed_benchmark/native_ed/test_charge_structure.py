"""Regression: complete 6x6 momentum-resolved ED connected charge covariance."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from geometry import RubyPBC
from scan_susceptibility import VScanSolver,leading_charge_mode
from plot_charge_pattern import projected_site_densities
from charge_structure import (fourier_orbital_matrix,ground_ensemble_charge,
    structure_factor_all,triangle_charge_content,project_polarized_density,export_structure_one,plot_scan_summary)
from plot_charge_structure import main as command_main


def test_all_momenta_vs_independent_covariance():
    rng=np.random.default_rng(44)
    for cluster in ('2x2','3x1','ring3'):
        g=RubyPBC(cluster,t1=.6,t2=.2)
        solver=VScanSolver(g,2,'point',low=3)
        H,e0,psis,rec,incomplete,resid=solver.solve(.8,1e-8)
        assert not incomplete and resid<1e-7
        density,C=ground_ensemble_charge(solver.states,psis,g.Ns)
        data=structure_factor_all(g,C)
        assert data['S_q'].shape==(g.num_cells,6,6)
        assert data['eigenvectors'].shape==(g.num_cells,6,6)
        assert abs(np.sum(density)-2)<1e-10
        projectors=[]
        for k,q in enumerate(data['momenta']):
            F=fourier_orbital_matrix(g,q)
            projectors.append(F@F.conj().T)
            np.testing.assert_allclose(F.conj().T@F,np.eye(6),atol=1e-12)
            np.testing.assert_allclose(data['S_q'][k],F.conj().T@C@F,atol=1e-11)
            np.testing.assert_allclose(data['S_q'][k],data['S_q'][k].conj().T,atol=1e-12)
            np.testing.assert_allclose(np.linalg.eigvalsh(data['S_q'][k]),data['eigenvalues'][k],atol=1e-12)
            assert data['eigenvalues'][k,0]>-1e-9
        # Density covariance is real: S(-q)=conj(S(q)), so K/K' have
        # identical spectra and conjugate charge patterns.
        for i,q in enumerate(data['momenta']):
            partner=np.mod(-q,1.)
            j=next(t for t,p in enumerate(data['momenta'])
                   if np.allclose(np.mod(p-partner+.5,1.)-.5,0.,atol=1e-9))
            np.testing.assert_allclose(data['S_q'][j],data['S_q'][i].conj(),atol=1e-9)
            np.testing.assert_allclose(data['eigenvalues'][j],data['eigenvalues'][i],atol=1e-9)
        np.testing.assert_allclose(np.sum(projectors,axis=0),np.eye(g.Ns),atol=1e-11)
        np.testing.assert_allclose(np.sum(data['eigenvalues']),np.trace(C),atol=1e-10)
        top=leading_charge_mode(solver.occ,psis,g)
        np.testing.assert_allclose(max(data['eigenvalues'][:,-1]),top['structure_factor'],atol=1e-10)
        # Fixed N: at q=0 uniform total density is a null vector.
        ki=next(k for k,q in enumerate(data['momenta']) if np.allclose(q,[0,0]))
        ones=np.ones(6)/np.sqrt(6)
        np.testing.assert_allclose(data['S_q'][ki]@ones,0,atol=1e-10)
        # Rotate the exact ground basis: ensemble C must be unchanged.
        if psis.shape[1]>1:
            Z=rng.normal(size=(psis.shape[1],psis.shape[1]))+1j*rng.normal(size=(psis.shape[1],psis.shape[1]))
            U=np.linalg.qr(Z)[0]
            d2,C2=ground_ensemble_charge(solver.states,psis@U,g.Ns)
            np.testing.assert_allclose(density,d2,atol=1e-11)
            np.testing.assert_allclose(C,C2,atol=1e-11)


def test_fourier_phase_and_projected_density():
    g=RubyPBC('ring3')
    states=np.array([1<<(g.Ns-1-i) for i in range(g.Ns)],dtype=np.uint64)
    states.sort()
    # two exactly orthonormal occupation Fock states, interpreted as a
    # synthetic degenerate manifold; avoid claiming these are H eigenstates.
    psi=np.zeros((len(states),2),complex)
    for a,i in enumerate((0,6)):
        idx=np.where(states==1<<(g.Ns-1-i))[0][0]
        psi[idx,a]=1
    n,C=ground_ensemble_charge(states,psi,g.Ns)
    M,_=projected_site_densities(states,psi,g.Ns)
    results=structure_factor_all(g,C)
    k=int(np.argmax(results['eigenvalues'][:,-1]))
    dense,meta=project_polarized_density(M,results['leading_complex_mode'][k],n)
    assert meta['available'] and meta['ground_dim']==2
    assert meta['contrast']>1e-4
    assert abs(dense.sum()-1)<1e-12
    unique,meta_unique=project_polarized_density(M[:,:1,:1],results['leading_complex_mode'][k],n)
    assert meta_unique['contrast']==0 and np.allclose(unique,n)
    # For K/K', direct density Fourier action is properly complex.
    ki=next(i for i,q in enumerate(results['momenta']) if not np.allclose(q,[0,0]))
    F=fourier_orbital_matrix(g,results['momenta'][ki])
    assert np.max(np.abs(F.imag))>0


def test_triangle_irrep_character():
    for mode,expected in [(np.array([2,-1,-1,0,0,0]),1.),
                          (np.array([0,0,0,0,1,-1]),1.),
                          (np.array([1,1,1,0,0,0]),0.)]:
        x=triangle_charge_content(mode/np.linalg.norm(mode))
        assert abs(x['E_internal_total']-expected)<1e-12
        assert abs(x['E_internal_total']+x['triangle_total']-1.)<1e-12


def test_cache_and_cli_no_resolve():
    with tempfile.TemporaryDirectory() as tmp:
        scan=Path(tmp)/'scan';scan.mkdir()
        g=RubyPBC('ring3',t1=.6,t2=.2)
        nf=2;V=0.7
        solver=VScanSolver(g,nf,'point',low=3)
        H,e0,psis,rec,partial,res=solver.solve(V,1e-8)
        n,C=ground_ensemble_charge(solver.states,psis,g.Ns)
        M,_=projected_site_densities(solver.states,psis,g.Ns)
        info={'cluster':'ring3','nf':nf,'t':1.,'t1':.6,'t2':.2,'symmetry':'point','low':3}
        (scan/'scan_metadata.json').write_text(json.dumps(info))
        point=scan/f'V_{V:+.9f}'
        point.mkdir()
        dat={'V':V,'parameters':info,'row':{'CO_leading_equal_time_lambda':float(max(structure_factor_all(g,C)['eigenvalues'][:,-1]))}}
        (point/'responses.json').write_text(json.dumps(dat))
        cache=scan/'charge_patterns'/f'V_{V:+.9f}'
        cache.mkdir(parents=True)
        np.savez_compressed(cache/'charge_pattern.npz',charge_covariance_ensemble=C,n_ensemble=n,
                            projected_site_density_matrices=M)
        rows=command_main(['--input',str(scan),'--V',str(V),'--cache-only'])
        assert len(rows)==1 and 'charge_pattern.npz' in rows[0]['source']
        output=scan/'charge_structure'
        root=output/f'V_{V:+.9f}'
        for file in ('charge_structure.npz','charge_structure_modes.csv','charge_structure_summary.json',
                     'ensemble_density.png','selected_density.png'):
            assert (root/file).exists(),file
        assert len(list(root.glob('q_*/S_matrix.png')))==g.num_cells
        assert (output/'charge_Sq_vs_V.png').exists()
        assert (output/'charge_Sq_vs_V.csv').exists()
        assert (root/'charge_orbital_character.csv').exists()
        with np.load(root/'charge_structure.npz') as saved:
            np.testing.assert_allclose(saved['S_q'],structure_factor_all(g,C)['S_q'])
        help_out=subprocess.run([sys.executable,str(Path(__file__).with_name('plot_charge_structure.py')),'--help'],capture_output=True,text=True,timeout=30)
        assert help_out.returncode==0,help_out.stderr
        assert '--cache-only' in help_out.stdout


if __name__=='__main__':
    for f in (test_all_momenta_vs_independent_covariance,test_fourier_phase_and_projected_density,
              test_triangle_irrep_character,
              test_cache_and_cli_no_resolve):
        print('Testing',f.__name__,flush=True)
        f()
    print('All momentum-charge tests passed')