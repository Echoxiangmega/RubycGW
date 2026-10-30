"""No-QuSpin regression tests of real-space Ruby ED bond-current diagrams."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

from core import make_states
from geometry import RubyPBC
from observables import op_sum, current_terms
from scan_susceptibility import VScanSolver,lc_triangle_channels,source_action
from plot_current_pattern import (selected_site_positions,directed_bond_currents,
             plaquette_data,optimize_state,export_one,load_scan_metadata,
             load_or_compute_ground)


def test_geometry_all_supercells():
    for name in ('ring3','3x1','2x2'):
        for t1,t2 in ((.5,.5),(.5,-.5)):
            g=RubyPBC(name,t1=t1,t2=t2)
            xy,ends,bound,cross,winds,cells=selected_site_positions(g)
            assert len(xy)==g.Ns
            assert len(ends)==12*g.num_cells
            assert sum(cross)>0
            assert np.allclose(np.linalg.norm(xy[[e.i for e in g.edges]]-ends,axis=1),1.)
            assert len(bound)==(6 if name=='ring3' else 4)
            assert np.allclose((np.linalg.solve(g.B,(ends-xy[[e.j for e in g.edges]]).T)).T,winds)


def test_current_vertex_and_continuity():
    rng=np.random.default_rng(184)
    for name in ('ring3','3x1','2x2'):
        g=RubyPBC(name,t1=-.6,t2=.6)
        states=make_states(g.Ns,2)
        psi=rng.normal(size=len(states))+1j*rng.normal(size=len(states))
        psi/=np.linalg.norm(psi)
        rho,J,div=directed_bond_currents(g,states,psi)
        assert abs(div.sum())<1e-12
        # One-particle hopping bond current is exactly the current_terms vertex.
        for ie in (0,7,21):
            e=g.edges[ie]
            vals=op_sum(psi,states,g.Ns,[(1j*g.t[e.kind],[('+',e.i),('-',e.j)]),
                (-1j*g.t[e.kind],[('+',e.j),('-',e.i)])])
            assert abs(np.vdot(psi,vals).real-J[ie])<1e-11
        assert np.max(abs(J+directed_bond_currents(g,states,psi.conjugate())[1]))<1e-11
        # Every triangle loop constructed from the bond table reproduces the
        # direct O|psi> expectation, including physical B chirality convention.
        loops=plaquette_data(g,rho,J,selected_site_positions(g)[0])
        assert len(loops)==g.Ns
        for lp in loops:
            ex=np.vdot(psi,op_sum(psi,states,g.Ns,current_terms(g,lp['sites'])))
            assert abs(ex.real-lp['I_ccw'])<1e-11


def test_ground_current_pattern():
    g=RubyPBC('ring3',t1=.5,t2=.5)
    sol=VScanSolver(g,2,symmetry='point',low=4)
    H,e0,psis,records,partial,res=sol.solve(.7,1e-8)
    assert not partial
    d=optimize_state(g,sol.states,psis,channel='opposite',E0=e0,V=.7)
    assert abs(d['current_from_bonds']-d['J_selected'])<1e-10
    assert d['max_divergence']<1e-7
    assert d['J_TR_error']<1e-10
    with tempfile.TemporaryDirectory() as tmp:
        rec=export_one(g,sol.states,psis,e0,.7,Path(tmp)/'pattern')
        assert (Path(tmp)/'pattern'/'bond_current.png').exists()
        assert (Path(tmp)/'pattern'/'triangle_current.svg').exists()
        assert (Path(tmp)/'pattern'/'plaquette_current.csv').exists()
        with np.load(Path(tmp)/'pattern'/'current_pattern.npz') as z:
            assert len(z['edge_J'])==36
            assert np.max(np.abs(z['divergence']))<1e-7
        assert abs(rec['Jmax_signed']-d['J_selected'])<1e-10


def test_cli():
    x=subprocess.run([sys.executable,str(Path(__file__).with_name('plot_current_pattern.py')),'--help'],
                    capture_output=True,text=True,timeout=30)
    assert x.returncode==0,x.stderr
    assert '--input' in x.stdout and '--V' in x.stdout and '--channel' in x.stdout


if __name__=='__main__':
    for fn in (test_geometry_all_supercells,test_current_vertex_and_continuity,
               test_ground_current_pattern,test_cli):
        print('Testing',fn.__name__,flush=True)
        fn()
    print('All current pattern tests passed',flush=True)