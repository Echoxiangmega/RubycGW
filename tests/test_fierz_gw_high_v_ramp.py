"""Regression tests: warm-start V ramp and persistent ED restart at strong V."""
from dataclasses import replace
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from rubycgw.model import RubyParameters
from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.fierz_channel_gw import (
    build_channel_definition, solve_channel_gw_same_torus,
)
from rubycgw.fierz_channel_continuation import solve_channel_gw_v_ramp
from rubycgw.three_torus_benchmark import build_torus_geometry
from rubycgw.three_torus_interaction_channels import triangle_v_pairs


def test_fierz_gw_warm_start_and_adaptive_ramp_match_direct_at_small_V():
    g=build_torus_geometry("3x1",RubyParameters(ti=1,t1=-.5,t2=.5,V=.12))
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=3,nOmega=1,T=.4)
    pairs=triangle_v_pairs(g.ncell)
    definition=build_channel_definition(pairs,g.nsites,.12,"density")
    opts=GWOptions(target_filling=2,max_iter=70,tol=2e-8,mixing=.25,
                   mixing_method="pulay",verbose=False)
    direct=solve_channel_gw_same_torus(g.h0,definition,grid,opts=opts)
    assert direct.converged
    warm=solve_channel_gw_same_torus(
        g.h0,definition,grid,opts=opts,initial=direct
    )
    assert warm.converged
    ramp=solve_channel_gw_v_ramp(
        g.h0,pairs,grid,.12,"density",opts,
        initial_steps=3,min_step=.005,max_stages=30
    )
    assert ramp.reached_target
    assert ramp.last_converged_V==.12
    assert len(ramp.attempted)>=3
    assert all(x.converged for x in ramp.attempted)
    np.testing.assert_allclose(ramp.state.G,direct.G,atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(warm.G,direct.G,atol=2e-8,rtol=2e-8)
    assert ramp.state.final_error<opts.tol


def test_fierz_warm_start_rejects_incompatible_mode():
    g=build_torus_geometry("3x1",RubyParameters(V=0.))
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=3,nOmega=1,T=.4)
    pairs=triangle_v_pairs(g.ncell)
    opts=GWOptions(target_filling=2,max_iter=4,tol=1e-7,verbose=False)
    d=build_channel_definition(pairs,g.nsites,0.,"density")
    bg=solve_channel_gw_same_torus(g.h0,d,grid,opts=opts)
    b=build_channel_definition(pairs,g.nsites,0.,"bj")
    import pytest
    with pytest.raises(ValueError,match="warm start mode"):
        solve_channel_gw_same_torus(g.h0,b,grid,opts=opts,initial=bg)


def _cli(out, *, steps=2, iterations=60, tol="1e-6", refresh=False):
    script=Path(__file__).resolve().parents[1]/"research"/"benchmark_three_tori.py"
    cmd=[
        sys.executable,str(script),
        "--geometry","3x1","--V",".1","--filling",str(2/3),
        "--K","0","0","--Q","0","0",
        "--T",".4","--nw","3","--nomega","1",
        "--methods","ed","gw","--gw-channels","density",
        "--channels","z_same",
        "--ed-eigs","2","--ed-samples","1","--ed-tau-points","3",
        "--ed-sector-padding","1","--ed-thermal-eigs","2",
        "--ed-krylov-steps","12","--plot-tau-points","9",
        "--gw-max-iter",str(iterations),"--gw-tol",str(tol),
        "--gw-mixing",".15","--gw-ramp-steps",str(steps),
        "--gw-ramp-min-step",".012",
        "--out",str(out),
    ]
    if refresh:
        cmd.append("--refresh-ed-cache")
    return subprocess.run(cmd,capture_output=True,text=True,timeout=180)


def test_ed_cache_survives_failed_gw_and_is_reused(tmp_path):
    out=tmp_path/"reuse"
    # Deliberately force failure after ED by permitting only one GW iteration
    # while demanding an unattainably tight residual at nonzero V.
    failed=_cli(out,steps=2,iterations=1,tol="1e-14")
    assert failed.returncode!=0,failed.stdout+"\n"+failed.stderr
    assert (out/"ed_reference.npz").exists()
    assert (out/"gw_failure_density.json").exists()
    details=json.loads((out/"gw_failure_density.json").read_text())
    assert not details["ramp"]["reached_target"]
    assert details["last_converged_V"]<.1

    recovered=_cli(out,steps=2,iterations=60,tol="1e-6")
    assert recovered.returncode==0,recovered.stdout+"\n"+recovered.stderr
    assert "matching ED cache" in recovered.stdout
    summary=json.loads((out/"summary.json").read_text())
    assert summary["diagnostics"]["ED"]["cache_hit"]
    assert "GW[density]" in summary["timings"]
    assert summary["diagnostics"]["GW[density]_ramp"]["reached_target"]
    assert (out/"green_tau.png").exists()
