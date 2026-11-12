"""Nonconverged *preliminary* GW is a permitted cluster embedding warm start.

The preliminary GW is not reported as a converged physical calculation.
Cluster ED+GW must independently converge its own embedding fixed point.
"""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rubycgw.fierz_channel_gw import solve_channel_gw_same_torus
from rubycgw.gw import GWOptions
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters, build_h0, build_interaction
from rubycgw.supercell_gw import dyson_from_sigma_matrix
from rubycgw.supercell_gw_fast import solve_matrix_gw_fast
from rubycgw.three_torus_benchmark import (
    build_torus_geometry, cluster_seed_from_density_gw,
)
from rubycgw.three_torus_interaction_channels import build_torus_interaction_channel


def _nonconverged_density_iterate():
    p=RubyParameters(ti=1.,t1=-.5,t2=.5,V=.2)
    geom=build_torus_geometry("3x1",p)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=3,nOmega=1,T=.4)
    definition=build_torus_interaction_channel(geom,p.V,"density")
    opts=GWOptions(target_filling=2.0,max_iter=1,tol=1e-14,
                   mixing=.2,mixing_method="pulay",verbose=False)
    trial=solve_channel_gw_same_torus(geom.h0,definition,grid,opts)
    assert not trial.converged
    return geom,grid,trial


def test_nonconverged_channel_gw_seed_gets_consistent_dyson_green():
    geom,grid,trial=_nonconverged_density_iterate()
    seed=cluster_seed_from_density_gw(trial,geom,grid,target_particles=2.0)
    assert not seed.converged
    assert seed.seed_dyson_rebuilt
    assert seed.seed_origin=="density-GW-preiterate"
    assert abs(seed.final_error-trial.final_error)<1e-14
    # The reported output Sigma can differ from the mixed Sigma behind
    # the last GW iterate. One Dyson pass restores mutual consistency.
    reference=dyson_from_sigma_matrix(
        geom.h0,grid,seed.mu,seed.Sigma_H,seed.Sigma_GW
    )
    np.testing.assert_allclose(seed.G,reference,rtol=0,atol=3e-12)
    assert abs(np.sum(seed.density)-2.0)<3e-6


def test_channel_seed_fails_cleanly_on_nonfinite_self_energy():
    geom,grid,trial=_nonconverged_density_iterate()
    trial.Sigma_c[0,0,0,0,0]=np.nan
    with pytest.raises(ValueError,match="nonfinite"):
        cluster_seed_from_density_gw(trial,geom,grid,2.0)


def test_generic_cluster_ed_gw_can_bypass_nonconverged_seed_gate(monkeypatch):
    ced=importlib.import_module("rubycgw.cluster_ed_gw")
    p=RubyParameters(ti=1.,t1=-.5,t2=.5,V=.2)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=3,nOmega=1,T=.4)
    h0=build_h0(grid.kmesh(),p)
    v=build_interaction(grid.qmesh(),p)
    opts=GWOptions(target_filling=2.,max_iter=1,tol=1e-14,
                   verbose=False,momentum_backend="direct")
    bg=solve_matrix_gw_fast(h0,v,grid,opts=opts)
    assert not bg.converged
    eopts=ced.ClusterEDGWOptions(max_iter=1,nbath=1,verbose=False)
    with pytest.raises(RuntimeError,match="initial lattice GW background is not converged"):
        ced.solve_cluster_ed_gw(
            h0,v,p,grid,gw_opts=opts,embed_opts=eopts,
            background=bg,require_converged_background=True
        )

    def entered_bath(*args,**kwargs):
        raise RuntimeError("entered cluster ED bath stage")

    monkeypatch.setattr(ced,"fit_finite_bath",entered_bath)
    with pytest.raises(RuntimeError,match="entered cluster ED bath stage"):
        ced.solve_cluster_ed_gw(
            h0,v,p,grid,gw_opts=opts,embed_opts=eopts,
            background=bg,require_converged_background=False
        )


def test_strict_cluster_seed_option_skips_when_gw_maxiter_hit(tmp_path):
    script=Path(__file__).resolve().parents[1]/"research"/"benchmark_three_tori.py"
    out=tmp_path/"strict_cluster_seed"
    cmd=[
        sys.executable,str(script),"--geometry","3x1",
        "--V",".2","--filling",str(2/3),
        "--T",".4","--nw","3","--nomega","1",
        "--methods","gw","cluster","--gw-channels","density",
        "--gw-max-iter","1","--gw-tol","1e-14",
        "--cluster-require-converged-gw",
        "--out",str(out)
    ]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    assert r.returncode==0,r.stdout+"\n"+r.stderr
    info=json.loads((out/"summary.json").read_text())
    assert info["method_status"]["GW[density]"]["status"]=="unconverged"
    assert info["method_status"]["cluster-ED+GW"]["status"]=="skipped"
    assert "cluster-ED+GW" not in info["timings"]
