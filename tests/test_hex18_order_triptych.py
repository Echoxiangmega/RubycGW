"""Real-space triptych plots of native cluster ED+GW hex18 ordered states."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from research.plot_hex18_cluster_order import (
    native_to_picture,picture_bonds,plot_order_triptych,
    _collect_scan,_load_orders,
)
from research.scan_true_hex18_v import RubyTorus
from research.scan_hex18_cluster_order_vn import _plot_branch
from rubycgw.hex18_order import order_observables
from rubycgw.model import RubyParameters
from rubycgw.three_torus_benchmark import build_torus_geometry


def _synthetic_order_file(folder):
    folder.mkdir(parents=True,exist_ok=True)
    params=RubyParameters(ti=1.,t1=-.5,t2=.5,V=.3)
    geom=build_torus_geometry("hex18",params)
    rho=np.eye(18,dtype=complex)*.35
    rho[0,1]=.10+.015j
    rho[1,0]=rho[0,1].conjugate()
    rho[12,13]=.07-.02j
    rho[13,12]=rho[12,13].conjugate()
    obs=order_observables(rho,geom,params)
    path=folder/"density__symmetric_orders.npz"
    np.savez_compressed(
        path,rho=rho,
        density=obs["density"],
        bond_i=np.asarray([x["i"] for x in obs["bond_rows"]]),
        bond_j=np.asarray([x["j"] for x in obs["bond_rows"]]),
        bond_real=np.asarray([x["real_bond"] for x in obs["bond_rows"]]),
        bond_current=np.asarray([x["current"] for x in obs["bond_rows"]]),
    )
    path.with_name("density__symmetric.json").write_text(
        json.dumps(dict(
            mode="density",candidate="symmetric",
            status="converged",phase_family=obs["phase_family"],
            V=.3,filling=2.2,
        )),encoding="utf-8"
    )
    return path


def test_hex18_true_hex_geometry_is_exact_permutation_of_native_bonds(tmp_path):
    path=_synthetic_order_file(tmp_path)
    _,records,_=_load_orders(path)
    torus=RubyTorus("true_hex",t=1.,t1=-.5,t2=.5)
    site_map=native_to_picture(torus)
    assert site_map.shape==(18,)
    np.testing.assert_array_equal(np.sort(site_map),np.arange(18))
    edges=picture_bonds(torus,site_map,records)
    assert len(edges)==36
    assert {x["kind"] for x in edges}=={"t","t1","t2"}
    assert sum(x["wrapped"] for x in edges)==6
    for edge in edges:
        np.testing.assert_allclose(
            np.linalg.norm(edge["end"]-edge["start"]),1.,
            atol=1e-10
        )
        assert np.isfinite(edge["current"])
    np.testing.assert_allclose(
        sorted([abs(x["current"]) for x in edges]),
        sorted([abs(x["current"]) for x in records.values()]),
        atol=1e-12
    )


def test_plot_triptych_png_svg_and_cli(tmp_path):
    path=_synthetic_order_file(tmp_path)
    images=plot_order_triptych(
        path,formats=("png","svg"),annotate=True,bond_reference="kind-mean"
    )
    assert len(images)==2
    assert {p.suffix for p in images}=={".png",".svg"}
    assert all(p.stat().st_size>1000 for p in images)
    script=Path(__file__).resolve().parents[1]/"research"/"plot_hex18_cluster_order.py"
    out=tmp_path/"order_panels.png"
    proc=subprocess.run(
        [sys.executable,str(script),"--orders",str(path),
         "--output",str(out),"--bond-reference","zero"],
        text=True,capture_output=True,timeout=90
    )
    assert proc.returncode==0,proc.stdout+"\n"+proc.stderr
    assert out.is_file() and out.stat().st_size>1000


def test_plot_scan_root_winner_vs_all_and_auto_scan_hook(tmp_path):
    folder=tmp_path/"n_2p2"/"V_0p3"
    path=_synthetic_order_file(folder)
    rows=[
        "V,filling,mode,phase,winner",
        "0.3,2.2,density,LC,symmetric",
    ]
    (tmp_path/"phase_summary.csv").write_text(
        "\n".join(rows)+"\n",encoding="utf-8"
    )
    assert _collect_scan(tmp_path,selection="converged")==[path]
    assert _collect_scan(tmp_path,selection="winners")==[path]
    assert _collect_scan(tmp_path,selection="winners",Vs=[.4])==[]
    args=SimpleNamespace(plot_branches=True,plot_formats=("png",),
                         plot_annotate=False,verbose=False)
    _plot_branch(args,path)
    expected=path.with_name("density__symmetric_orders_realspace.png")
    assert expected.exists()
    args.plot_branches=False
    expected.unlink()
    _plot_branch(args,path)
    assert not expected.exists()


def test_triptych_rejects_invalid_bond_arrays(tmp_path):
    path=tmp_path/"bad_orders.npz"
    np.savez(path,density=np.zeros(18),bond_i=np.zeros(36),
             bond_j=np.zeros(36),bond_real=np.zeros(36),
             bond_current=np.zeros(36))
    with pytest.raises(ValueError,match="duplicate physical bond"):
        _load_orders(path)
