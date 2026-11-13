"""Mode-specific certified warm starts in multi-V Fierz cluster ED+GW scans."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from research.compare_fierz_cluster_ed_gw import (
    _requested_scan_values,_scan_point_dir
)


def test_v_scan_selection_and_grid():
    defaults=dict(V=None,V_scan=None,V_start=None,V_stop=None,V_step=None)
    def args(**changes):
        return Namespace(**dict(defaults,**changes))
    assert _requested_scan_values(args(V_scan=[0,0.5,1.0]))==(0.,.5,1.)
    assert _requested_scan_values(args(V_start=0,V_stop=1,V_step=.3))==(
        0.,.3,.6,.9,1.
    )
    assert _requested_scan_values(args(V_start=1,V_stop=0,V_step=-.5))==(
        1.,.5,0.
    )
    assert _requested_scan_values(args(V=.2)) is None
    with pytest.raises(ValueError,match="strictly"):
        _requested_scan_values(args(V_scan=[0,.2,.1]))
    with pytest.raises(ValueError,match="omit"):
        _requested_scan_values(args(V=1,V_scan=[.5,1.]))
    with pytest.raises(ValueError,match="nonzero"):
        _requested_scan_values(args(V_start=0,V_stop=1,V_step=0))
    assert _scan_point_dir("/tmp/results",.125).name=="V_0p125"
    assert _scan_point_dir("/tmp/results",-2.5).name=="V_m2p5"


def test_two_V_cli_comparison_warm_starts_each_mode_and_resume(tmp_path):
    """The second V reuses a certified complex-bath reference, not a cold guess.

    Tiny bath and weak V only validate the workflow, not strong-V physics.
    """
    script=Path(__file__).resolve().parents[1]/"research"/"compare_fierz_cluster_ed_gw.py"
    cmd=[
        sys.executable,str(script),
        "--V-scan","0",".001",
        "--n",str(2/3),"--T",".4",
        "--K","0","0","--Q","0","0",
        "--modes","density",
        "--channels","x_even",
        "--nw","3","--nomega","1","--nbath","1",
        "--bath-fit-nfreq","2","--bath-max-nfev","40",
        "--bath-fit-restarts","0",
        "--uniform-bath","complex",
        "--uniform-max-iter","12","--uniform-tol","1e-4",
        "--baseline-max-iter","18","--baseline-tol","1e-4",
        "--source-max-iter","12","--source-tol","1e-4",
        "--source-step",".005",
        "--ed-samples","1","--ed-tau-points","3",
        "--ed-sector-padding","1","--ed-thermal-eigs","1",
        "--ed-krylov-steps","10","--tau-points","9",
        "--checkpoint-every","2","--out",str(tmp_path),
    ]
    run=subprocess.run(cmd,text=True,capture_output=True,timeout=300)
    assert run.returncode==0,run.stdout+"\n"+run.stderr
    root=json.loads((tmp_path/"scan_manifest.json").read_text())
    assert root["status"]=="complete",run.stdout
    assert [x["V"] for x in root["points"]]==[0.,.001]
    assert all(x["status"]=="ok" for x in root["points"])
    first=tmp_path/"V_0"
    second=tmp_path/"V_0p001"
    assert (first/"baseline_cluster_density_checkpoint.npz").is_file()
    assert (second/"baseline_cluster_density_checkpoint.npz").is_file()
    prev_diag=json.loads((first/"diagnostics.json").read_text())
    curr_diag=json.loads((second/"diagnostics.json").read_text())
    assert prev_diag["stages"]["density_uniform"]["status"]=="ok"
    assert "density_uniform" not in curr_diag["stages"],run.stdout
    assert "density_initial" not in curr_diag["stages"]
    warm=curr_diag["methods"]["density"]["warm_start"]
    assert warm["used"] is True
    assert warm["source_V"]==0.
    assert warm["destination_V"]==.001
    assert warm["new_V_screening_recomputed"]
    assert curr_diag["stages"]["density_zero_field"]["status"]=="ok"

    with (tmp_path/"scan_green_comparison.csv").open(
        encoding="utf-8",newline=""
    ) as stream:
        greens=list(csv.DictReader(stream))
    with (tmp_path/"scan_chi_comparison.csv").open(
        encoding="utf-8",newline=""
    ) as stream:
        chis=list(csv.DictReader(stream))
    assert len(greens)==4
    assert len(chis)==4
    next_cluster=next(x for x in greens if x["V"]=="0.001"
                      and x["method"]=="cluster-density")
    assert next_cluster["status"]=="converged"
    assert float(next_cluster["warm_start_from_V"])==0.
    assert any(x["V"]=="0.001" and x["method"]=="ED" and
               x["status"]=="ok" for x in chis)

    resume=subprocess.run(cmd+["--resume-scan"],
                          text=True,capture_output=True,timeout=120)
    assert resume.returncode==0,resume.stdout+"\n"+resume.stderr
    assert "resumed 2/2 complete points" in resume.stdout
    rerun=json.loads((tmp_path/"scan_manifest.json").read_text())
    assert rerun["status"]=="complete"
    assert len(rerun["points"])==2
