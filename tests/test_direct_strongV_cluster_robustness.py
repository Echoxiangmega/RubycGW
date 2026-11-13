"""Focused tests for strong-V Fierz embedding solver safeguards.

These tests use cheap noninteracting impurities to verify plumbing; they
do not claim that the genuinely strongly interacting V=5 fixed point exists.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rubycgw.cluster_convergence import (
    CoupledSelfEnergyMixer,screening_diagnostic,fit_bath_with_restarts,
)
from rubycgw.cluster_ed_gw import BathParameters,fit_finite_bath,ruby_cluster_interactions
from rubycgw.cluster_ed_gw_multicell import (
    MultiCellEDGWOptions,solve_multicell_cluster_ed_gw,
    restart_from_uniform_cluster,
)
from rubycgw.fierz_cluster_embedding import build_embedding_channels
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.three_torus_benchmark import (
    build_torus_geometry,noninteracting_cluster_seed,
    uniform_primitive_cluster_ed_gw,
)
from research.compare_fierz_cluster_ed_gw import (
    _checkpoint_signature,_checkpoint_path,
    _save_checkpoint,_load_checkpoint,
)


def test_safeguarded_pulay_bounds_step_and_retains_finite_blocks():
    mixer=CoupledSelfEnergyMixer(
        method="pulay",alpha=.3,history=4,start=2,
        step_cap=1.25,reset_growth=2.
    )
    h=np.zeros((2,2),complex)
    e=np.zeros((4,1,1,4,4),complex)
    imp=np.zeros((4,2,2),complex)
    for iteration in range(1,8):
        out_h=h+.2*np.eye(2)
        out_e=e+.08
        out_imp=imp-.17
        hn,en,inn=mixer.propose(
            h,e,imp,out_h,out_e,out_imp,iteration
        )
        assert np.all(np.isfinite(hn))
        assert np.all(np.isfinite(en))
        assert np.all(np.isfinite(inn))
        assert max(np.max(abs(hn-h)),np.max(abs(en-e)),np.max(abs(inn-imp)))<=.25
        h,e,imp=hn,en,inn
    assert len(mixer.history)<=4


def test_screening_diagnostics_all_fierz_modes_at_zero_interaction():
    params=RubyParameters(V=0.)
    geom=build_torus_geometry("hex18",params)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=3,nOmega=1,T=.4)
    ints=ruby_cluster_interactions(params)
    for mode in ("density","pseudospin","bj"):
        channel=build_embedding_channels(ints,geom.ncell,mode,geom.vq)
        n=(geom.nsites if mode=="density"
           else len(channel.lattice.labels))
        shape=((grid.nb,1,1,n,n) if mode=="density"
               else (grid.nb,n,n))
        info=screening_diagnostic(np.zeros(shape,complex),geom.vq,channel,grid)
        np.testing.assert_allclose(info["min_singular_value"],1.0,atol=1e-12)


def test_finite_bath_exposes_optimizer_status_and_retry_count():
    w=(2*np.arange(-3,3)+1)*np.pi*.4
    target=np.zeros((len(w),3,3),complex)
    fitted,count=fit_bath_with_restarts(
        fit_finite_bath,target,w,.1,initial=None,
        nbath=1,nfit=2,max_nfev=40,
        restarts=1,retry_error=0.,seed=123,
        energy_window=4.,coupling_bound=4.,
        metric="delta",one_body=np.eye(3),
    )
    assert count in (1,2)
    assert isinstance(fitted.success,bool)
    assert isinstance(fitted.active_bounds,bool)
    assert isinstance(fitted.fit_message,str)
    assert np.isfinite(fitted.fit_error)


def _zero_case():
    params=RubyParameters(ti=1.,t1=-.5,t2=.5,V=0.)
    geom=build_torus_geometry("hex18",params)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=3,nOmega=1,T=.4)
    target=2.
    return params,geom,grid,target


def test_uniform_complex_bath_seed_checkpoint_resume_and_multicell_pulay(tmp_path):
    _,geom,grid,target=_zero_case()
    bg=noninteracting_cluster_seed(geom,grid,target)
    args=SimpleNamespace(
        V=0.,filling=target/3,T=.4,
        t=1.,t1=-.5,t2=.5,nw=3,nomega=1,nbath=1,uniform_bath="complex"
    )
    signature=_checkpoint_signature(args,"density","uniform")
    path=_checkpoint_path(tmp_path,"density","uniform")
    def write(it,state):
        _save_checkpoint(path,signature,state,it,stage="uniform")
    result=uniform_primitive_cluster_ed_gw(
        geom,grid,bg,nbath=1,max_iter=12,tol=1e-4,
        bath_nfit=2,bath_max_nfev=50,
        target_particles=target,gw_mode="density",
        mixing_method="pulay",bath_complex=True,
        initial_channel="z_same",initial_amplitude=0.002,
        screening_monitor=True,checkpoint_callback=write,
        checkpoint_every=2
    )
    assert result.converged,result.residual_history
    assert result.bath_mode=="complex"
    assert path.is_file()
    assert result.component_residual_history.shape[1]==3
    assert len(result.screening_history)==result.iterations
    assert result.screening_history[0]["min_singular_value"]==pytest.approx(1.)
    restarted=_load_checkpoint(
        path,signature,stage="uniform",grid=grid,
        nsites=geom.nsites,nbath=1
    )
    again=uniform_primitive_cluster_ed_gw(
        geom,grid,bg,nbath=1,max_iter=5,tol=1e-4,
        bath_nfit=2,bath_max_nfev=50,
        target_particles=target,gw_mode="density",
        mixing_method="pulay",bath_complex=True,
        restart=restarted,screening_monitor=True
    )
    assert again.converged
    bad=dict(signature,V=5.)
    with pytest.raises(ValueError,match="do not match"):
        _load_checkpoint(path,bad,stage="uniform",
                         grid=grid,nsites=geom.nsites,nbath=1)

    ints=ruby_cluster_interactions(RubyParameters(V=0.))
    multi=solve_multicell_cluster_ed_gw(
        geom.h0,geom.vq,ints,grid,target_particles=target,
        opts=MultiCellEDGWOptions(
            max_iter=8,tol=1e-4,mixing=.3,
            mixing_method="pulay",nbath=1,
            bath_fit_nfreq=2,bath_fit_max_nfev=50,
            complex_bath=True,screening_monitor=True,verbose=False
        ),
        restart=restart_from_uniform_cluster(again,geom.ncell)
    )
    assert multi.converged,multi.residual_history
    assert len(multi.screening_history)==multi.iterations
    assert multi.component_residual_history.shape[1]==3
    assert isinstance(multi.bath_diagnostics,list)
