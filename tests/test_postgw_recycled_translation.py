"""Physics regressions for shared/cached post-GW covariant vertex solves.

Check equivalence against the legacy per-source GMRES full response, including
nonzero interaction and the unchanged --post-mmax 0 frequency fallback.
"""
import numpy as np
import pytest
from types import SimpleNamespace

from rubycgw.grids import MatsubaraGrid
from rubycgw.gw import GWOptions
from rubycgw.model import RubyParameters
from rubycgw.fierz_channel_gw import solve_channel_gw_same_torus
from rubycgw.fierz_channel_gamma import compute_covariant_channel_susceptibility
from rubycgw.supercell_cgw import SupercellVertexOptions
from rubycgw.three_torus_benchmark import build_torus_geometry
from rubycgw.three_torus_interaction_channels import (
    build_torus_interaction_channel, one_shot_post_interaction_channel,
)


def _background(geometry="3x1", mode="density", V=0.08, nw=3):
    geom=build_torus_geometry(
        geometry, RubyParameters(ti=1., t1=-.5, t2=.5, V=V)
    )
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=nw,nOmega=1,T=.4)
    definition=build_torus_interaction_channel(geom,V,mode)
    opts=GWOptions(
        target_filling=2., max_iter=60, tol=1e-9,
        mixing=.28,mixing_method="pulay",verbose=False
    )
    state=solve_channel_gw_same_torus(geom.h0,definition,grid,opts=opts)
    assert state.converged, state.final_error
    vopts=SupercellVertexOptions(
        max_iter=40,tol=2e-8,gmres_restart=8,verbose=False
    )
    return geom,grid,definition,state,vopts


def test_translation_reduced_post_mzero_matches_all_source_gmres_and_fallback():
    geom,grid,d,bg,opts=_background()
    full,_=compute_covariant_channel_susceptibility(
        bg,d,geom.h0,grid,vertex_opts=opts,m_max=0,
        optimize_translation=False,recycle_krylov=False,
    )
    fast,_=compute_covariant_channel_susceptibility(
        bg,d,geom.h0,grid,vertex_opts=opts,m_max=0,
        geometry=geom,optimize_translation=True,recycle_krylov=True,
    )
    assert fast.translation_reduced
    assert full.source_vertices_solved==len(d.labels)
    assert fast.source_vertices_solved==len(d.labels)//geom.ncell
    izero=int(np.flatnonzero(grid.m_values==0)[0])
    np.testing.assert_allclose(
        fast.chi_completed[izero],full.chi_completed[izero],
        atol=8e-7,rtol=5e-6,
    )
    result=one_shot_post_interaction_channel(
        bg,d,geom.h0,grid,2.,opts,
        geometry=geom,m_max=0,optimize_translation=True,
        recycle_krylov=True,
    )
    assert result.translation_reduced
    assert result.source_vertices_solved==len(d.labels)//geom.ncell
    np.testing.assert_allclose(
        result.W[np.asarray(grid.m_values)!=0],
        bg.W[np.asarray(grid.m_values)!=0],
        atol=1e-12,
    )
    assert np.all(np.isfinite(result.G))


def test_full_dynamic_reduced_vs_legacy_density_on_skew_hex18():
    geom,grid,d,bg,opts=_background(geometry="hex18",V=0.06,nw=3)
    full,_=compute_covariant_channel_susceptibility(
        bg,d,geom.h0,grid,vertex_opts=opts,m_max=None,
        optimize_translation=False,recycle_krylov=False,
    )
    fast,_=compute_covariant_channel_susceptibility(
        bg,d,geom.h0,grid,vertex_opts=opts,m_max=None,
        geometry=geom,optimize_translation=True,recycle_krylov=True,
    )
    assert fast.translation_reduced
    assert fast.source_vertices_solved == 2*len(d.labels)//geom.ncell
    np.testing.assert_allclose(
        fast.chi_completed,full.chi_completed,atol=1e-6,rtol=8e-6
    )


@pytest.mark.parametrize("mode,nsource",[
    ("density",6),("pseudospin",10),("bj",12)
])
def test_all_three_fierz_channel_internal_vertices_reduce(mode,nsource):
    geom,grid,d,bg,opts=_background(mode=mode,V=0.,nw=2)
    fast,_=compute_covariant_channel_susceptibility(
        bg,d,geom.h0,grid,vertex_opts=opts,m_max=0,
        geometry=geom,optimize_translation=True,recycle_krylov=False,
    )
    assert fast.translation_reduced
    assert fast.source_vertices_solved==nsource
    assert np.max(np.abs(fast.chi_completed[np.asarray(grid.m_values)==0]))<1e5


def test_nonuniform_input_disables_translation_reduction():
    geom,grid,d,bg,opts=_background(V=0.)
    G=np.asarray(bg.G,complex).copy()
    G[:,0,0,0,0]+=0.012
    wrong=SimpleNamespace(**vars(bg))
    wrong.G=G
    res,_=compute_covariant_channel_susceptibility(
        wrong,d,geom.h0,grid,vertex_opts=opts,m_max=0,
        geometry=geom,optimize_translation=True,recycle_krylov=False,
    )
    assert not res.translation_reduced
    assert res.source_vertices_solved==len(d.labels)


def test_period_two_torus_translation_reduction():
    # Index-four 2x2 torus must reduce one orbital-density source per cell
    # and preserve both independent period-two translations.
    geom,grid,d,bg,opts=_background(geometry="2x2",V=0.,nw=2)
    reduced,_=compute_covariant_channel_susceptibility(
        bg,d,geom.h0,grid,vertex_opts=opts,m_max=0,
        geometry=geom,optimize_translation=True,recycle_krylov=False,
    )
    assert reduced.translation_reduced
    assert reduced.source_vertices_solved==6
    iz=int(np.flatnonzero(grid.m_values==0)[0])
    np.testing.assert_allclose(
        reduced.chi_completed[iz],reduced.chi_completed[iz].conj().T,
        atol=5e-9,
    )
