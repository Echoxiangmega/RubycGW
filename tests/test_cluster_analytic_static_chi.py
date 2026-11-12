"""Analytic cluster static response versus exact V=0 thermodynamics.

The V=0 limit is a strict test of the source signs, tail completion,
fixed-filling chemical-potential derivative and full 18-site projection.
The ED and finite-source integration regression checks the three-way
command from a cold start using a small bath.
"""
from __future__ import annotations
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rubycgw.cluster_ed_gw_linear_response import (
    StaticClusterEDGWLinearizer,_polarization_tangent
)
from rubycgw.grids import MatsubaraGrid
from rubycgw.model import RubyParameters
from rubycgw.supercell_gw import compute_polarization_matrix
from rubycgw.supercell_gw_fast import _solve_mu_matrix_fast
from rubycgw.supercell_gw_split import one_body_density_matrix_tail
from rubycgw.three_torus_benchmark import (
    build_torus_geometry,hermitian_source_components,
    noninteracting_cluster_seed
)


def test_analytic_zero_V_cluster_chi_matches_noninteracting_finite_source():
    params=RubyParameters(ti=1,t1=-.5,t2=.5,V=0.)
    geom=build_torus_geometry("hex18",params)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=4,nOmega=2,T=.35)
    target=2.
    bg=noninteracting_cluster_seed(geom,grid,target)
    ref=SimpleNamespace(
        converged=True,G=bg.G,mu=bg.mu,
        Sigma_H=np.zeros((18,18),complex),
        Sigma_emb=np.zeros_like(bg.G),
        Sigma_ED_cluster=np.zeros((3,grid.nf,6,6),complex),
        W=np.zeros((grid.nb,1,1,18,18),complex),
        bath_energies=np.ones((3,1),float)*.2,
        bath_couplings=np.zeros((3,6,1),complex),
        bath_fit_error=np.zeros(3),
        impurity_static_shift=np.zeros((3,6,6),complex),
    )
    kernel=StaticClusterEDGWLinearizer(
        geom,params,grid,ref,bath_nfit=2
    )
    for channel in ("x_even","z_same"):
        analytic=0.
        finite=0.
        for K in hermitian_source_components(geom,channel,(1/3,1/3)):
            analytic+=kernel.response(K,rtol=1e-8,maxiter=3).chi
            measured=[]
            for sign in (1.,-1.):
                Hsource=np.asarray(geom.h0,complex).copy()
                Hsource[0,0]-=sign*1e-4*K
                mu,G,_,_=_solve_mu_matrix_fast(
                    Hsource,np.zeros((18,18),complex),
                    np.zeros_like(bg.G),grid,target,bg.mu,1e-12,100
                )
                rho=one_body_density_matrix_tail(
                    G,grid,Hsource,mu,np.zeros((18,18),complex)
                )
                measured.append(
                    float(np.einsum("ab,ba->",K,rho[0,0]).real)
                )
            finite+=(measured[0]-measured[1])/(2e-4)
        np.testing.assert_allclose(analytic,finite,rtol=5e-6,atol=1e-7)


def test_analytic_polarization_derivative_equals_actual_GW_bubble_difference():
    rng=np.random.default_rng(155)
    grid=MatsubaraGrid(nk1=1,nk2=1,nw=3,nOmega=1,T=.2)
    G=rng.normal(size=(grid.nf,1,1,4,4))+1j*rng.normal(size=(
        grid.nf,1,1,4,4
    ))
    dG=rng.normal(size=G.shape)+1j*rng.normal(size=G.shape)
    pred=_polarization_tangent(G,dG,grid)
    d=1e-6
    actual=(compute_polarization_matrix(G+d*dG,grid)
            -compute_polarization_matrix(G-d*dG,grid))/(2*d)
    np.testing.assert_allclose(pred,actual,atol=2e-8,rtol=1e-7)


def test_hex18_three_methods_cli_noninteracting(tmp_path):
    out=tmp_path/"threeways"
    script=Path(__file__).resolve().parents[1]/"research"/"compare_ed_cluster_chi_analytic.py"
    cmd=[
        sys.executable,str(script),
        "--V","0","--n",str(2/3),
        "--T",".4","--nw","3","--nomega","1","--nbath","1",
        "--bath-fit-nfreq","2","--bath-max-nfev","40",
        "--channels","x_even","z_same",
        "--ed-samples","1","--ed-tau-points","3",
        "--cluster-max-iter","5","--cluster-tol","1e-4",
        "--baseline-max-iter","5","--baseline-tol","1e-4",
        "--source-max-iter","6","--source-tol","1e-4",
        "--analytic-tol","1e-7","--analytic-maxiter","3",
        "--out",str(out),
    ]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+"\n"+result.stderr
    with (out/"chi_comparison.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==6
    for channel in ("x_even","z_same"):
        finite=next(r for r in rows if r["method"]=="cluster_finite_source"
                    and r["channel"]==channel)
        analytic=next(r for r in rows if r["method"]=="cluster_analytic"
                      and r["channel"]==channel)
        ED=next(r for r in rows if r["method"]=="ED"
                and r["channel"]==channel)
        assert ED["status"]=="ok"
        assert finite["status"]=="ok",result.stdout
        assert analytic["status"]=="ok",result.stdout
        np.testing.assert_allclose(
            float(analytic["chi"]),float(finite["chi"]),
            rtol=2e-3,atol=1e-4
        )
    diag=json.loads((out/"diagnostics.json").read_text())
    assert diag["stage"]["zero_field_complex_bath"]["status"]=="ok"


def test_three_method_cli_weakly_interacting_cluster_finite_source_agreement(tmp_path):
    """V>0 exercises nonzero GW, double-counting, bath Hessian and ED response."""
    out=tmp_path/"weak"
    script=Path(__file__).resolve().parents[1]/"research"/"compare_ed_cluster_chi_analytic.py"
    cmd=[
        sys.executable,str(script),"--V",".015","--n",str(2/3),
        "--T",".4","--nw","3","--nomega","1","--nbath","1",
        "--bath-fit-nfreq","2","--bath-max-nfev","120",
        "--channels","z_same","--ed-samples","1","--ed-tau-points","3",
        "--cluster-max-iter","35","--cluster-tol","1e-6",
        "--baseline-max-iter","35","--baseline-tol","1e-6",
        "--source-max-iter","40","--source-tol","1e-6",
        "--source-step",".002",
        "--analytic-tol","3e-5","--analytic-maxiter","7",
        "--analytic-restart","8",
        "--out",str(out),
    ]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
    assert r.returncode==0,r.stdout+"\\n"+r.stderr
    with (out/"chi_comparison.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    a=next(row for row in rows if row["method"]=="cluster_analytic")
    b=next(row for row in rows if row["method"]=="cluster_finite_source")
    assert a["status"]=="ok",(r.stdout,r.stderr)
    assert b["status"]=="ok",(r.stdout,r.stderr)
    np.testing.assert_allclose(
        float(a["chi"]),float(b["chi"]),rtol=4e-2,atol=4e-3
    )
    diag=json.loads((out/"diagnostics.json").read_text())
    assert diag["stage"]["zero_field_complex_bath"]["status"]=="ok"


def test_analytic_full_GW_split_tangent_matches_finite_difference():
    from rubycgw.cluster_ed_gw_linear_response import (
        _gw_tangent,_density_tangent_cache
    )
    from rubycgw.supercell_gw import (
        compute_screened_interaction_matrix,dyson_from_sigma_matrix
    )
    from rubycgw.supercell_gw_split import compute_sigma_gw_split_matrix
    p=RubyParameters(ti=1,t1=-.5,t2=.5,V=.13)
    geom=build_torus_geometry("hex18",p)
    g=MatsubaraGrid(nk1=1,nk2=1,nw=3,nOmega=1,T=.4)
    h=np.asarray(geom.h0,complex)
    Vq=np.asarray(geom.vq,complex)
    sh=np.zeros((18,18),complex)
    sg=np.zeros((g.nf,1,1,18,18),complex)
    mu=.22
    G=dyson_from_sigma_matrix(h,g,mu,sh,sg)
    source=np.diag(np.linspace(-.3,.25,18)).astype(complex)
    dG=np.matmul(np.matmul(G,source),G)
    cache=_density_tangent_cache(h,mu,sh,G,g)
    drho=cache["rho_tangent"](dG[:,0,0],np.zeros((18,18)))
    P=compute_polarization_matrix(G,g)
    W=compute_screened_interaction_matrix(P,Vq)
    predicted,_,_=_gw_tangent(G,dG,drho,drho,W,Vq,g)
    eps=2e-5
    values=[]
    for sign in (-1.,1.):
        gs=G+sign*eps*dG
        ps=compute_polarization_matrix(gs,g)
        ws=compute_screened_interaction_matrix(ps,Vq)
        values.append(compute_sigma_gw_split_matrix(
            gs,ws,Vq,g,h,mu,sh,backend="fft"
        ))
    numeric=(values[1]-values[0])/(2*eps)
    np.testing.assert_allclose(predicted,numeric,rtol=2e-5,atol=2e-7)
