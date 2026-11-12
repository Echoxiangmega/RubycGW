"""Real CLI and physics checks of cross-V GW/cGW continuation on Ruby tori."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from rubycgw.model import RubyParameters,build_h0
from rubycgw.three_torus_benchmark import build_torus_geometry
from rubycgw.cluster_ed_gw import build_intracell_h0


def _script(name):
    return Path(__file__).resolve().parents[1]/"research"/name


def test_sorted_v_scan_warm_starts_each_GW_and_cGW_and_saves_tables(tmp_path):
    out=tmp_path/"scan"
    command=[
        sys.executable,str(_script("scan_three_tori_v.py")),
        "--V-values","0.1","0","0.05",
        "--geometry","3x1","--filling",str(2./3.),
        "--K","0","0","--Q","0","0",
        "--T",".4","--nw","3","--nomega","1",
        "--methods","gw","cgw","--gw-channels","density",
        "--channels","z_same",
        "--gw-max-iter","60","--gw-tol","1e-6",
        "--vertex-max-iter","32","--vertex-tol","1e-5",
        "--plot-tau-points","9","--out",str(out)
    ]
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+"\n"+result.stderr
    manifest=json.loads((out/"scan_manifest.json").read_text())
    assert manifest["V_sorted"]==[0.,.05,.1]
    assert len(manifest["points"])==3
    assert all(x["status"]=="complete" for x in manifest["points"])
    reports=[]
    for i,V in enumerate(manifest["V_sorted"]):
        vstr=f"{V:.12g}".replace(".","p")
        directory=out/f"V_{i:03d}_{vstr}"
        assert (directory/"comparison.csv").exists()
        info=json.loads((directory/"summary.json").read_text())
        assert info["diagnostics"]["GW[density]"]["converged"]
        assert info["method_status"]["cGW[density]"]["status"]=="converged"
        reports.append(info)
    assert reports[0]["diagnostics"]["GW[density]"]["warm_started"] is False
    assert reports[1]["diagnostics"]["GW[density]"]["warm_started_from_V"]==0.
    assert reports[2]["diagnostics"]["GW[density]"]["warm_started_from_V"]==.05
    assert reports[0]["diagnostics"]["cGW[density]"]["warm_started"] is False
    assert reports[1]["diagnostics"]["cGW[density]"]["warm_started_from_V"]==0.
    assert reports[2]["diagnostics"]["cGW[density]"]["warm_started_from_V"]==.05
    with (out/"scan_results.csv").open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    assert len([x for x in rows if x["method"]=="GW[density]"])==3
    with (out/"scan_chi.csv").open(encoding="utf-8",newline="") as f:
        ch=list(csv.DictReader(f))
    assert len([x for x in ch if x["method"]=="cGW[density]" and
               x["channel"]=="z_same"])==3

    # A long scan can resume without rerunning the already completed costly
    # ED/GW/cGW points. The certified Gamma/GW seeds are saved without pickle.
    last=out/"V_002_0p1"/"summary.json"
    before=last.stat().st_mtime_ns
    resumed=subprocess.run(
        command+["--resume"],capture_output=True,text=True,timeout=180
    )
    assert resumed.returncode==0,resumed.stdout+"\\n"+resumed.stderr
    assert "resumed V=" in resumed.stdout
    assert last.stat().st_mtime_ns==before
    assert (out/"V_002_0p1"/"warm_seeds.npz").is_file()

    # Same model and fixed point: warmed V=.1 must agree with direct cold solve.
    cold=tmp_path/"cold"
    cmd=[
        sys.executable,str(_script("benchmark_three_tori.py")),
        "--geometry","3x1","--V",".1","--filling",str(2./3.),
        "--K","0","0","--Q","0","0",
        "--T",".4","--nw","3","--nomega","1",
        "--methods","gw","cgw","--gw-channels","density",
        "--channels","z_same","--gw-max-iter","60",
        "--gw-tol","1e-6","--vertex-max-iter","32",
        "--vertex-tol","1e-5","--plot-tau-points","9",
        "--out",str(cold)
    ]
    check=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    assert check.returncode==0,check.stdout+"\n"+check.stderr
    A=np.load(out/"V_002_0p1"/"raw_arrays.npz")
    B=np.load(cold/"raw_arrays.npz")
    np.testing.assert_allclose(A["Giw_full_density"],B["Giw_full_density"],
                               atol=3e-5,rtol=3e-5)
    final=json.loads((out/"V_002_0p1"/"summary.json").read_text())
    direct=json.loads((cold/"summary.json").read_text())
    assert abs(final["computed_chi"]["cGW[density]"]["z_same"]-
               direct["computed_chi"]["cGW[density]"]["z_same"])<2e-4


def test_intercell_hopping_enters_lattice_G_and_cluster_Weiss():
    params=RubyParameters(ti=1,t1=-.5,t2=.5,V=0)
    geom=build_torus_geometry("3x1",params)
    h=geom.h0[0,0]
    off=h[:6,6:]
    assert np.max(abs(off))>1e-4
    # Primitive-cell ED hloc contains only the same 6-site block. Coupling
    # to the other cells is included in Gc and its dynamical bath Δ.
    hloc=h[:6,:6]
    iw=1j*1.3
    mu=.05
    G=np.linalg.inv((iw+mu)*np.eye(18)-h)
    Gc=G[:6,:6]
    Delta=(iw+mu)*np.eye(6)-hloc-np.linalg.inv(Gc)
    assert np.linalg.norm(Delta)>1e-4
    np.testing.assert_allclose(
        np.linalg.inv((iw+mu)*np.eye(6)-hloc-Delta),
        Gc,atol=5e-12
    )


def test_scan_rejects_redundant_zero_to_target_ramp(tmp_path):
    out=tmp_path/"bad"
    cmd=[
        sys.executable,str(_script("scan_three_tori_v.py")),
        "--V-values","0",".1","--geometry","3x1",
        "--filling","1","--gw-ramp-steps","10","--out",str(out)
    ]
    out=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    assert out.returncode!=0
    assert "gw-ramp-steps" in out.stderr


def test_scan_failed_V_keeps_last_certified_GW_seed(tmp_path):
    out=tmp_path/"nonconv"
    cmd=[
        sys.executable,str(_script("scan_three_tori_v.py")),
        "--V-values",".1",".2","0",
        "--geometry","3x1","--filling",str(2./3.),
        "--K","0","0","--Q","0","0",
        "--T",".4","--nw","3","--nomega","1",
        "--methods","gw","--gw-channels","density",
        "--gw-max-iter","1","--gw-tol","1e-14",
        "--channels","z_same","--plot-tau-points","9",
        "--out",str(out)
    ]
    res=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    assert res.returncode==0,res.stdout+"\\n"+res.stderr
    data=json.loads((out/"scan_manifest.json").read_text())
    assert data["V_sorted"]==[0.,.1,.2]
    for item in data["points"]:
        assert item["status"]=="complete"
        assert item["GW_last_converged"]["density"]==0.
    assert json.loads(
        (out/"V_002_0p2"/"summary.json").read_text()
    )["diagnostics"]["GW[density]"]["warm_started_from_V"]==0.
