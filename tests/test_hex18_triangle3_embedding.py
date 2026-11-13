"""Triangle3 (A and B separate) cluster ED+GW on full periodic hex18 Ruby."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rubycgw.cluster_ed_gw import (
    cluster_interaction_matrix,ruby_cluster_interactions,
)
from rubycgw.cluster_ed_gw_multicell import (
    MultiCellEDGWOptions,solve_multicell_cluster_ed_gw,
)
from rubycgw.cluster_ed_gw_multicell_free_energy import evaluate_multicell_free_energy
from rubycgw.fierz_cluster_embedding import (
    build_embedding_channels,triangle_cluster_interactions,
    triangle_embedding_channel,cluster_gw_map
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.supercell_gw_split import one_body_density_matrix_tail
from rubycgw.three_torus_benchmark import (
    build_torus_geometry,project_k_green
)
from research.scan_hex18_cluster_order_vn import (
    parse_args,_fresh_bootstrap,_load_state,
)


def test_triangle_partition_has_six_disjoint_interacting_blocks_and_full_hopping():
    params=RubyParameters(ti=1.,t1=-.5,t2=.5,V=.4)
    geom=build_torus_geometry("hex18",params)
    interactions=ruby_cluster_interactions(params)
    assert len(interactions)==6
    for side in (0,1):
        terms=triangle_cluster_interactions(interactions,side)
        assert len(terms)==3
        assert {tuple(sorted((i,j))) for i,j,v in terms}=={
            (0,1),(0,2),(1,2)
        }
        np.testing.assert_allclose(
            cluster_interaction_matrix(terms,3),
            np.full((3,3),.4)-np.eye(3)*.4
        )
    H=geom.h0[0,0]
    # No t1/t2 between different triangles inside any 3-site block.
    # Those bonds MUST remain in the *full* lattice Hamiltonian.
    assert np.any(np.abs(H[:3,3:6])>0)
    assert np.any(np.abs(H[:3,6:])>0)
    for c in range(6):
        block=H[3*c:3*c+3,3*c:3*c+3]
        assert block.shape==(3,3)
        assert np.max(abs(block))==pytest.approx(1.)


@pytest.mark.parametrize("mode",["density","pseudospin","bj"])
def test_local_triangle_fierz_double_counting_uses_three_bonds_and_three_orbitals(mode):
    params=RubyParameters(ti=1.,t1=-.5,t2=.5,V=.1)
    geom=build_torus_geometry("hex18",params)
    terms=ruby_cluster_interactions(params)
    definition=build_embedding_channels(
        terms,geom.ncell,mode,geom.vq
    )
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=3,nOmega=1,T=.4)
    for side in (0,1):
        loc=triangle_embedding_channel(definition,side)
        loc_interactions=triangle_cluster_interactions(terms,side)
        assert np.shape(loc.cluster.vertices)[-2:]==(3,3) if mode!="density" else loc.cluster is None
        if mode!="density":
            assert len(loc.cluster.labels)==(5 if mode=="pseudospin" else 6)
            assert loc.cluster.interaction_pairs==((0,1),(0,2),(1,2))
        # A valid causal noninteracting 3x3 local G and site occupations.
        bare=np.diag(np.asarray([-.3,0.,.3]))
        g=np.linalg.inv(
            1j*grid.omega[:,None,None]*np.eye(3)[None]-bare[None]
        )
        rho=np.eye(3)*.3
        sigma,_,_=cluster_gw_map(
            g,rho,loc_interactions,grid,loc,
            Vc=cluster_interaction_matrix(loc_interactions,3)
        )
        assert sigma.shape==(grid.nf,3,3)
        assert np.all(np.isfinite(sigma))


def test_triangle3_solver_fixed_total_filling_and_free_limit():
    params=RubyParameters(ti=1.,t1=-.5,t2=.5,V=0.)
    geom=build_torus_geometry("hex18",params)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=3,nOmega=1,T=.4)
    args=parse_args([
        "--V-scan","0","--fillings","2.2",
        "--cluster-partition","triangle3",
        "--nw","3","--nomega","1","--no-ed",
        "--out","results/test_tri3_unit"
    ])
    assert args.nbath==3
    assert args.cluster_partition=="triangle3"
    init,origin=_fresh_bootstrap(geom,params,grid,6.6,"density",args)
    assert origin=="free"
    assert init.Sigma_imp.shape==(6,grid.nf,3,3)
    assert init.bath_energies.shape==(6,3)
    assert init.bath_couplings.shape==(6,3,3)
    result=solve_multicell_cluster_ed_gw(
        geom.h0,geom.vq,ruby_cluster_interactions(params),grid,
        target_particles=6.6,
        opts=MultiCellEDGWOptions(
            cluster_size=3,nbath=3,max_iter=15,tol=1e-4,
            bath_fit_nfreq=2,bath_fit_max_nfev=50,
            bath_fit_restarts=0,mixing=.3,
            mixing_method="pulay",complex_bath=True,verbose=False,
        ),
        restart=init
    )
    assert result.converged,result.final_error
    assert result.cluster_size==3
    assert result.G.shape==(grid.nf,1,1,18,18)
    assert result.G_cluster.shape==(6,grid.nf,3,3)
    assert result.G_impurity.shape==(6,grid.nf,3,3)
    assert result.bath_couplings.shape==(6,3,3)
    rho=one_body_density_matrix_tail(
        result.G,grid,geom.h0,result.mu,result.Sigma_H
    )[0,0]
    np.testing.assert_allclose(np.trace(rho).real,6.6,rtol=0.,atol=1e-6)
    occup=np.diag(rho).real.reshape(3,2,3).sum(axis=-1)
    np.testing.assert_allclose(np.sum(occup),6.6,atol=1e-6)
    Gk=project_k_green(result.G,geom,(1/3,1/3))
    assert Gk.shape==(grid.nf,6,6)
    free=evaluate_multicell_free_energy(
        result,geom.h0,geom.vq,ruby_cluster_interactions(params),
        grid,target_particles=6.6
    )
    assert np.isfinite(free.helmholtz_free_energy_per_primitive_cell)
    assert len(free.impurity_grand_potential)==6


def test_triangle3_v_n_warm_start_ED_G_and_resume(tmp_path):
    """An ordered 2D triangle scan calls full ED Green once per point, no chi."""
    script=Path(__file__).resolve().parents[1]/"research"/"scan_hex18_triangle_ed_gw_vn.py"
    cmd=[
        sys.executable,str(script),
        "--V-scan","0",".001",
        "--fillings",str(2/3),".7",
        "--T",".4","--K","0","0",
        "--modes","density","--candidates","symmetric",
        "--nw","3","--nomega","1",
        "--bath-fit-nfreq","2","--bath-max-nfev","45",
        "--bath-fit-restarts","0",
        "--bootstrap","free","--max-iter","15","--tol","1e-4",
        "--mixing",".3",
        "--ed-sector-padding","1","--ed-thermal-eigs","1",
        "--ed-krylov-steps","12",
        "--no-free-energy","--out",str(tmp_path)
    ]
    first=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
    assert first.returncode==0,first.stdout+"\n"+first.stderr
    with (tmp_path/"branches.csv").open(
        newline="",encoding="utf-8"
    ) as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==4
    assert all(r["cluster_partition"]=="triangle3" for r in rows)
    assert all(r["status"]=="converged" for r in rows),first.stdout
    assert all(r["ed_status"]=="ok" for r in rows),first.stdout
    assert all(np.isfinite(float(r["g_error"])) for r in rows)
    assert all(abs(float(r["filling_error"]))<1e-6 for r in rows)
    assert any(r["V"]=="0.001" and r["warm_from_V"]=="0.0"
               for r in rows)
    assert any(r["V"]=="0.0" and r["filling"]=="0.7"
               and r["warm_from_n"]==str(2/3) for r in rows)
    folder=tmp_path/"n_0p6666666666666666"/"V_0"
    with np.load(folder/"density__symmetric.npz",allow_pickle=False) as z:
        assert z["Sigma_imp"].shape==(6,6,3,3)
        assert z["bath_couplings"].shape==(6,3,3)
        assert z["Gk"].shape==(6,6,6)
    with np.load(folder/"density__symmetric_orders.npz") as z:
        assert z["triangle_occupations"].shape==(3,2)
        assert z["density"].shape==(18,)
    with np.load(folder/"ed_Gk.npz") as z:
        assert z["Gk"].shape==(6,6,6)

    again=subprocess.run(cmd+["--resume"],
                         capture_output=True,text=True,timeout=120)
    assert again.returncode==0,again.stdout+"\n"+again.stderr
    assert again.stdout.count("[hex18-order] restored")==4
