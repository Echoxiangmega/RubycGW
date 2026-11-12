"""Ensure Fierz-matched cluster subtraction and the three-way Hex18 benchmark."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rubycgw.cluster_ed_gw import (
    cluster_gw_self_energy,cluster_interaction_matrix,
    ruby_cluster_interactions
)
from rubycgw.fierz_cluster_embedding import (
    build_embedding_channels,cluster_gw_map,lattice_gw_map
)
from rubycgw.fierz_channel_gw import (
    build_channel_definition,channel_static_self_energy,
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.supercell_gw_fast import _solve_mu_matrix_fast
from rubycgw.supercell_gw_split import one_body_density_matrix_tail
from rubycgw.three_torus_benchmark import build_torus_geometry,noninteracting_cluster_seed


def _pieces(V=.12):
    params=RubyParameters(ti=1,t1=-.5,t2=.5,V=V)
    geom=build_torus_geometry("hex18",params)
    ints=ruby_cluster_interactions(params)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=3,nOmega=1,T=.4)
    Gstate=noninteracting_cluster_seed(geom,grid,2.)
    rho=one_body_density_matrix_tail(Gstate.G,grid,geom.h0,Gstate.mu,Gstate.Sigma_H)
    return params,geom,ints,grid,Gstate,rho


@pytest.mark.parametrize("mode",("density","pseudospin","bj"))
def test_matched_fierz_cluster_HF_exact_bare_V_and_local_GW(mode):
    params,geom,ints,grid,state,rho=_pieces()
    defs=build_embedding_channels(ints,geom.ncell,mode,geom.vq)
    sh,sigma,P,W=lattice_gw_map(
        state.G,rho,geom.h0,geom.vq,grid,defs,state.mu,
        static_reference=state.Sigma_H
    )
    Gc=np.asarray(state.G[:,0,0,:6,:6])
    rhoc=rho[0,0,:6,:6]
    sigma_c,pc,wc=cluster_gw_map(
        Gc,rhoc,ints,grid,defs
    )
    assert np.all(np.isfinite(sh))
    assert np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(sigma_c))
    assert sigma_c.shape==(grid.nf,6,6)
    if mode=="density":
        ref,_,_=cluster_gw_self_energy(
            Gc,rhoc,cluster_interaction_matrix(ints,6),grid
        )
        np.testing.assert_allclose(sigma_c,ref,atol=2e-12,rtol=2e-12)
        assert W.shape==(grid.nb,1,1,geom.nsites,geom.nsites)
        assert sigma.shape==state.G.shape
    else:
        assert P.shape==(grid.nb,len(defs.lattice.labels),len(defs.lattice.labels))
        assert W.shape==P.shape
        np.testing.assert_allclose(sh,channel_static_self_energy(
            rho[0,0],defs.lattice
        )[0],atol=1e-12,rtol=0)


def test_pseudospin_and_bj_HF_equal_physical_density_HF():
    _,geom,ints,grid,state,rho=_pieces(.27)
    totals={}
    for mode in ("density","pseudospin","bj"):
        definition=build_embedding_channels(ints,geom.ncell,mode,geom.vq)
        if mode=="density":
            channel=build_channel_definition(
                tuple((6*c+i,6*c+j)
                      for c in range(geom.ncell)
                      for i,j,_ in ints),
                geom.nsites,.27,"density"
            )
        else:
            channel=definition.lattice
        totals[mode]=channel_static_self_energy(rho[0,0],channel)[0]
    np.testing.assert_allclose(
        totals["density"],totals["bj"],atol=3e-12,rtol=0
    )
    np.testing.assert_allclose(
        totals["density"],totals["pseudospin"],atol=3e-12,rtol=0
    )


@pytest.mark.parametrize("bad",["unmatched","unsupported"])
def test_no_silent_unmatched_Fierz_interactions(bad):
    _,geom,ints,grid,state,rho=_pieces()
    if bad=="unmatched":
        Vq=geom.vq.copy()
        Vq[0,0,0,7]+=0.123
        with pytest.raises(ValueError,match="unrepresented"):
            build_embedding_channels(ints,geom.ncell,"bj",Vq)
    else:
        with pytest.raises(ValueError,match="density, pseudospin or bj"):
            build_embedding_channels(ints,geom.ncell,"some-mode",geom.vq)


def test_fierz_compare_cli_three_modes_green_and_chi_at_V0(tmp_path):
    """Actual ED + three zero-field embeddings + 3 finite-source chi values."""
    out=tmp_path/"fierz"
    script=Path(__file__).resolve().parents[1]/"research"/"compare_fierz_cluster_ed_gw.py"
    cmd=[
        sys.executable,str(script),
        "--V","0","--n",str(2/3),"--T",".4",
        "--K","0","0","--Q","0","0",
        "--nw","3","--nomega","1","--nbath","1",
        "--bath-fit-nfreq","2","--bath-max-nfev","35",
        "--channels","x_even",
        "--ed-samples","1","--ed-tau-points","3",
        "--ed-sector-padding","1","--ed-thermal-eigs","1",
        "--ed-krylov-steps","12","--tau-points","9",
        "--uniform-max-iter","5","--uniform-tol","1e-4",
        "--baseline-max-iter","6","--baseline-tol","1e-4",
        "--source-max-iter","6","--source-tol","1e-4",
        "--source-step",".01","--out",str(out)
    ]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
    assert r.returncode==0,r.stdout+"\n"+r.stderr
    with (out/"green_comparison.csv").open(encoding="utf-8",newline="") as f:
        gs=list(csv.DictReader(f))
    with (out/"chi_comparison.csv").open(encoding="utf-8",newline="") as f:
        cs=list(csv.DictReader(f))
    assert len(gs)==4
    assert len(cs)==4
    assert gs[0]["method"]=="ED"
    for mode in ("density","pseudospin","bj"):
        green=next(x for x in gs if x["method"]=="cluster-"+mode)
        chi=next(x for x in cs if x["method"]=="cluster-"+mode)
        assert green["status"]=="converged",(mode,r.stdout)
        assert chi["status"]=="ok",(mode,r.stdout)
        assert float(green["relative_Frobenius_K"])<1e-2
    chis=[float(x["chi"]) for x in cs if x["method"].startswith("cluster-")]
    np.testing.assert_allclose(chis,chis[0],atol=1e-4,rtol=1e-3)
    assert (out/"green_curves.npz").is_file()
    d=json.loads((out/"diagnostics.json").read_text())
    assert all(d["methods"][mode]["gw_mode"]==mode
               for mode in ("density","pseudospin","bj"))


def test_fierz_compare_cli_nonzero_V_distinct_dynamic_skeleton(tmp_path):
    """Weak V checks genuine nonzero GW and channel-specific cluster subtraction."""
    params,geom,ints,grid,state,rho=_pieces(.12)
    local=np.asarray(state.G[:,0,0,:6,:6])
    rhoc=np.asarray(rho[0,0,:6,:6])
    vals=[]
    for mode in ("density","pseudospin","bj"):
        definition=build_embedding_channels(ints,geom.ncell,mode,geom.vq)
        sigma,_,_=cluster_gw_map(local,rhoc,ints,grid,definition)
        vals.append(sigma)
    # Same HF, different resummed dynamical diagrams at O(V^2)
    assert np.linalg.norm(vals[0]-vals[1])>1.e-9
    assert np.linalg.norm(vals[0]-vals[2])>1.e-9



def test_weak_interaction_three_mode_cluster_fixed_points(tmp_path):
    """Nonzero V exercises each Fierz lattice/local GW subtraction in Dyson."""
    output=tmp_path/"weak"
    script=Path(__file__).resolve().parents[1]/"research"/"compare_fierz_cluster_ed_gw.py"
    cmd=[
        sys.executable,str(script),"--V",".015",
        "--n",str(2/3),"--T",".4",
        "--K","0","0","--Q","0","0",
        "--nw","3","--nomega","1","--nbath","1",
        "--bath-fit-nfreq","2","--bath-max-nfev","70",
        "--ed-sector-padding","1","--ed-thermal-eigs","1",
        "--ed-krylov-steps","10","--tau-points","9",
        "--uniform-max-iter","35","--uniform-tol","1e-5",
        "--baseline-max-iter","35","--baseline-tol","1e-5",
        "--skip-chi","--out",str(output)
    ]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
    assert p.returncode==0,p.stdout+"\n"+p.stderr
    with (output/"green_comparison.csv").open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==4
    for mode in ("density","bj","pseudospin"):
        x=next(v for v in rows if v["method"]=="cluster-"+mode)
        assert x["status"]=="converged",(mode,p.stdout)
        assert np.isfinite(float(x["relative_Frobenius_K"]))
    diag=json.loads((output/"diagnostics.json").read_text())
    assert all(diag["methods"][mode]["gw_mode"]==mode for mode in (
        "density","bj","pseudospin"
    ))
