"""Ordered hex18 self-energy seed, phase diagnostic, and scan integration."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rubycgw.hex18_order import (
    FAMILIES,DEFAULT_FAMILIES,QK,seed_matrix,order_observables,
)
from rubycgw.model import RubyParameters
from rubycgw.three_torus_benchmark import build_torus_geometry
from research.scan_hex18_cluster_order_vn import (
    _grid_explicit_or_range,_nearest_warm,_variants,_classify_winner,
)


def test_scan_grids_and_mode_specific_warm_priority():
    assert _grid_explicit_or_range(None,0,1,.4,"V")==(
        0.,.4,.8,1.
    )
    assert _grid_explicit_or_range([2,2.1,2.2],None,None,None,"n")==(
        2.,2.1,2.2
    )
    with pytest.raises(ValueError):
        _grid_explicit_or_range([2,2,3],None,None,None,"n")
    cache={
        (0,0,"density","symmetric"):"old",
        (0,1,"density","symmetric"):"previous-fill",
        (1,0,"density","symmetric"):"previous-V",
        (1,0,"bj","symmetric"):"different-mode",
    }
    x,src=_nearest_warm(cache,1,1,"density","symmetric")
    assert x=="previous-V" and src==(1,0)
    x,src=_nearest_warm(cache,1,1,"bj","symmetric")
    assert x=="different-mode" and src==(1,0)
    x,src=_nearest_warm(cache,1,1,"pseudospin","symmetric")
    assert x is None and src is None
    assert _variants(("symmetric","lc_qK_same_cos"),(.02,.05))==[
        ("symmetric","symmetric",0.),
        ("lc_qK_same_cos_h0p02","lc_qK_same_cos",.02),
        ("lc_qK_same_cos_h0p05","lc_qK_same_cos",.05),
    ]


def test_hex18_candidate_seeds_hermitian_finiteq_and_order_operator():
    geom=build_torus_geometry(
        "hex18",RubyParameters(ti=1.,t1=-.5,t2=.5,V=.2)
    )
    pars=RubyParameters(ti=1.,t1=-.5,t2=.5,V=.2)
    assert geom.nsites==18 and geom.ncell==3
    for name in FAMILIES:
        K=seed_matrix(geom,pars,name)
        assert K.shape==(18,18) and np.all(np.isfinite(K))
        np.testing.assert_allclose(K,K.conj().T,atol=1e-12)
        if name=="symmetric":
            assert np.max(abs(K))==0.
        else:
            assert np.max(abs(K))==pytest.approx(1.)
    same=seed_matrix(geom,pars,"lc_qK_same_cos")
    opp=seed_matrix(geom,pars,"lc_qK_opposite_cos")
    assert not np.allclose(same,opp)
    assert not np.allclose(
        same[0:6,0:6],same[6:12,6:12]
    )
    rho=np.eye(18,dtype=complex)*.3
    obs=order_observables(rho,geom,pars)
    assert obs["phase_family"]=="symmetric"
    assert len(obs["density"])==18
    assert len(obs["bond_rows"])==36
    rho=rho+.015*same
    obs=order_observables(rho,geom,pars,threshold=1e-3)
    assert obs["phase_family"]=="LC"
    assert obs["lc_amplitude"]>0.001
    assert abs(obs["amplitudes"]["z_same_qK"])>0.001
    assert obs["co_amplitude"]<1e-10


def test_winner_ignores_uncertain_free_energy_and_same_endpoints():
    dummy=dict(mode="density",phase_family="CO",status="converged",
               free_energy_reliable=True,F_per_pc=-2.,
               order_signature=[0.,1.,2.],candidate="co")
    other=dict(dummy,candidate="co-other",F_per_pc=-2.+1e-7)
    answer=_classify_winner([dummy,other],"density",1e-5)
    assert answer["phase"]=="CO"
    assert answer["n_reliable"]==1
    assert answer["winner"]=="co"
    ignore=dict(dummy,free_energy_reliable=False)
    assert _classify_winner([ignore],"density",1e-5)["phase"]=="unresolved"


def test_hex18_order_cli_2d_warm_without_chi(tmp_path):
    """Two fillings x two V with actual 18-site multi-impurity solves."""
    script=(Path(__file__).resolve().parents[1]/
            "research"/"scan_hex18_cluster_order_vn.py")
    cmd=[
        sys.executable,str(script),
        "--V-scan","0",".001",
        "--fillings",str(2/3),".7",
        "--T",".4","--K","0","0",
        "--modes","density","--candidates",
        "symmetric","lc_qK_same_cos",
        "--seed-amplitudes",".01",
        "--nbath","1","--nw","3","--nomega","1",
        "--bath-fit-nfreq","2","--bath-max-nfev","40",
        "--bath-fit-restarts","0",
        "--bootstrap","free","--max-iter","35",
        "--tol","1e-4","--mixing",".3",
        "--no-ed","--no-free-energy","--tau-points","9",
        "--out",str(tmp_path),
    ]
    result=subprocess.run(cmd,text=True,capture_output=True,timeout=300)
    assert result.returncode==0,result.stdout+"\n"+result.stderr
    with (tmp_path/"branches.csv").open(
        newline="",encoding="utf-8"
    ) as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==8,result.stdout
    assert any(
        x["V"]=="0.001" and x["candidate"]=="symmetric"
        and x["status"]=="converged"
        and x["warm_from_V"]=="0.0"
        for x in rows
    ),result.stdout
    assert any(
        x["filling"]=="0.7" and x["V"]=="0.0"
        and x["candidate"]=="symmetric" and
        x["status"]=="converged" and
        x["warm_from_n"]==str(2/3)
        for x in rows
    ),result.stdout
    assert (tmp_path/"phase_summary.csv").is_file()
    assert (tmp_path/"n_0p6666666666666666"/"V_0"/
            "density__symmetric.npz").exists()
    manifest=json.loads((tmp_path/"scan_manifest.json").read_text())
    assert len(manifest["points"])==4
    rerun=subprocess.run(cmd+["--resume"],text=True,
                         capture_output=True,timeout=120)
    assert rerun.returncode==0,rerun.stdout+"\n"+rerun.stderr
    with (tmp_path/"branches.csv").open(
        newline="",encoding="utf-8"
    ) as handle:
        resumed=list(csv.DictReader(handle))
    assert len(resumed)==8


def test_hex18_order_cli_ed_green_compared_once_per_point(tmp_path):
    """At zero V, ED reference is computed and shared by the candidate solve."""
    script=(Path(__file__).resolve().parents[1]/
            "research"/"scan_hex18_cluster_order_vn.py")
    cmd=[
        sys.executable,str(script),
        "--V-scan","0","--fillings",str(2/3),
        "--T",".4","--K","0","0",
        "--modes","density","--candidates","symmetric",
        "--nbath","1","--nw","3","--nomega","1",
        "--bath-fit-nfreq","2","--bath-max-nfev","40",
        "--bath-fit-restarts","0",
        "--bootstrap","free","--max-iter","8","--tol","1e-4",
        "--ed-sector-padding","1","--ed-thermal-eigs","1",
        "--ed-krylov-steps","12",
        "--no-free-energy","--out",str(tmp_path)
    ]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
    assert r.returncode==0,r.stdout+"\n"+r.stderr
    folder=tmp_path/"n_0p6666666666666666"/"V_0"
    assert (folder/"ed_Gk.npz").exists(),r.stdout
    with (folder/"ed_Gk.npz").open("rb") as handle:
        with np.load(handle,allow_pickle=False) as ed:
            assert ed["Gk"].shape==(6,6,6)
    with (tmp_path/"branches.csv").open(
        newline="",encoding="utf-8"
    ) as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==1
    assert rows[0]["status"]=="converged",r.stdout
    assert rows[0]["ed_status"]=="ok",r.stdout
    assert np.isfinite(float(rows[0]["g_error"]))
    assert float(rows[0]["g_error"])>=0
