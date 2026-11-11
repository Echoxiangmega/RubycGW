"""Fast unit tests for identical finite-torus benchmark geometry and observables."""
import numpy as np
import pytest

from rubycgw.model import RubyParameters, build_h0
from rubycgw.grids import MatsubaraGrid
from rubycgw.pseudospin import primitive_pseudospin_vertex
from rubycgw.three_torus_benchmark import (
    build_torus_geometry, k_projector, project_k_green,
    hermitian_source_components, green_k_tau, max_required_ed_dimension,
)


@pytest.mark.parametrize("name,ncell,momenta",[
    ("3x1",3,[(0,0),(1/3,0),(2/3,0)]),
    ("2x2",4,[(0,0),(.5,0),(0,.5),(.5,.5)]),
    ("hex18",3,[(0,0),(1/3,1/3),(2/3,2/3)]),
])
def test_torus_geometry_projectors_orthogonal_and_spectral(name,ncell,momenta):
    params=RubyParameters(ti=1.,t1=-.5,t2=.5,V=.3)
    g=build_torus_geometry(name,params)
    assert g.nsites==ncell*6
    assert g.h0.shape==(1,1,6*ncell,6*ncell)
    assert g.vq.shape==(1,1,6*ncell,6*ncell)
    assert np.max(abs(g.h0-g.h0.swapaxes(-1,-2).conj()))<1e-12
    assert np.max(abs(g.vq-g.vq.swapaxes(-1,-2).conj()))<1e-12
    P=[k_projector(g,q) for q in momenta]
    allP=np.column_stack(P)
    np.testing.assert_allclose(allP.conj().T@allP,np.eye(6*ncell),atol=2e-12)
    for q,U in zip(momenta,P):
        H=U.conj().T@g.h0[0,0]@U
        h_expected=build_h0(np.array([q]),params)[0]
        np.testing.assert_allclose(H,h_expected,atol=2e-12)
    vals=np.sort(np.linalg.eigvalsh(g.h0[0,0]))
    valsk=np.sort(np.concatenate([
        np.linalg.eigvalsh(U.conj().T@g.h0[0,0]@U) for U in P
    ]))
    np.testing.assert_allclose(vals,valsk,atol=2e-12)


def test_invalid_torus_momenta_are_rejected():
    p=RubyParameters()
    for geo,invalid in [("3x1",(.5,0)),("2x2",(1/3,0)),("hex18",(1/3,0))]:
        g=build_torus_geometry(geo,p)
        with pytest.raises(ValueError,match="commensurate"):
            g.validate_momentum(invalid)


@pytest.mark.parametrize("name,Q",[
    ("3x1",(1/3,0)),("2x2",(.5,0)),("hex18",(1/3,1/3))
])
def test_source_components_reconstruct_complex_q(name,Q):
    g=build_torus_geometry(name,RubyParameters())
    comps=hermitian_source_components(g,"z_same",Q)
    assert all(np.max(abs(x-x.conj().T))<1e-12 for x in comps)
    K=primitive_pseudospin_vertex("z_same")
    reference=np.zeros((g.nsites,g.nsites),dtype=complex)
    for i,R in enumerate(g.cells):
        reference[6*i:6*(i+1),6*i:6*(i+1)]=(
            np.exp(-2j*np.pi*np.dot(Q,R))*K/np.sqrt(g.ncell)
        )
    reconstructed=comps[0] if len(comps)==1 else comps[0]-1j*comps[1]
    np.testing.assert_allclose(reconstructed,reference,atol=1e-12)


def test_finite_fermionic_green_tau_reference_exact():
    p=RubyParameters(ti=1.,t1=-.5,t2=.5,V=0.)
    g=build_torus_geometry("3x1",p)
    q=(1/3,0)
    U=k_projector(g,q)
    H=U.conj().T@g.h0[0,0]@U
    T=.125; mu=.11
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=10,nOmega=2,T=T)
    G=np.asarray([np.linalg.inv((1j*w+mu)*np.eye(6)-H) for w in grid.omega])
    tau=np.linspace(0,1/T,17)
    gt=green_k_tau(G,grid.omega,T,mu,H,tau)
    evals=np.linalg.eigvalsh(H)
    from scipy.special import expit
    n=float(np.sum(expit(-(evals-mu)/T)))
    np.testing.assert_allclose(np.trace(gt[0]),-(6-n),atol=3e-13)
    np.testing.assert_allclose(np.trace(gt[-1]),-n,atol=3e-13)
    assert np.max(abs(gt[:,0,0].imag))<1e-13


def test_noninteracting_GK_projection_and_ed_sector_cap():
    g=build_torus_geometry("2x2",RubyParameters(V=0))
    U=k_projector(g,(.5,.5))
    H=g.h0[0,0]
    omega=(2*np.arange(-4,4)+1)*np.pi*.2
    full=np.asarray([np.linalg.inv((1j*w+.05)*np.eye(24)-H) for w in omega])
    projected=project_k_green(full,g,(.5,.5))
    direct=np.asarray([np.linalg.inv((1j*w+.05)*np.eye(6)-U.conj().T@H@U) for w in omega])
    np.testing.assert_allclose(projected,direct,atol=2e-12)
    assert max_required_ed_dimension(24,8,1)>=735471


def test_three_torus_cli_end_to_end_small_free_case(tmp_path):
    """Small noninteracting full-pipeline smoke; no physics claim at tiny grids."""
    import subprocess
    import sys
    from pathlib import Path
    script=Path(__file__).resolve().parents[1]/"research"/"benchmark_three_tori.py"
    out=tmp_path/"smoke"
    cmd=[
        sys.executable,str(script),
        "--geometry","3x1","--V","0","--filling",str(2./3.),
        "--K","0","0","--Q","0","0",
        "--T",".4","--nw","3","--nomega","1",
        "--methods","ed","gw","cgw","postgw","cluster",
        "--channels","x_even",
        "--gw-max-iter","8","--gw-tol","1e-6",
        "--vertex-max-iter","12","--vertex-tol","1e-5",
        "--post-mmax","0","--nbath","1",
        "--cluster-max-iter","3","--cluster-tol","1e-4",
        "--ed-samples","1","--ed-tau-points","3",
        "--ed-thermal-eigs","2","--ed-krylov-steps","12",
        "--ed-sector-padding","1","--plot-tau-points","9",
        "--skip-cluster-chi",
        "--out",str(out),
    ]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    assert result.returncode==0, result.stdout+"\n"+result.stderr
    assert (out/"summary.json").is_file()
    assert (out/"comparison.csv").is_file()
    assert (out/"raw_arrays.npz").is_file()
    assert (out/"green_tau.png").is_file()
    import json
    summary=json.loads((out/"summary.json").read_text())
    assert summary["geometry"]=="3x1"
    assert summary["N"]==2
    assert "GW" in summary["timings"]
    assert "ED_Green" in summary["timings"]
