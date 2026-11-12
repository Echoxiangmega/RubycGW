"""Fractional mean filling ED and warm cluster source response regressions."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rubycgw.model import RubyParameters
from rubycgw.grand_canonical_ed import (
    selected_charge_sectors,grand_canonical_ed_response,
)
from rubycgw.three_torus_benchmark import build_torus_geometry
from rubycgw.cluster_ed_gw_multicell import restart_from_uniform_cluster


def test_fractional_sector_selection_and_partition_mean():
    assert selected_charge_sectors(18,4.5,0)==[4,5]
    assert selected_charge_sectors(18,4.5,1)==[3,4,5,6]
    assert selected_charge_sectors(18,5.0,1)==[4,5,6]
    with pytest.raises(ValueError,match="requires >=2"):
        selected_charge_sectors(18,5.0,0)


def test_grand_canonical_charge_susceptibility_includes_number_fluctuations():
    params=RubyParameters(ti=1,t1=-.5,t2=.5,V=.12)
    geom=build_torus_geometry("3x1",params)
    T=.5
    response=grand_canonical_ed_response(
        geom,params,V=.12,T=T,target_particles=1.5,
        Q=(0.,0.),channels=("charge_total",),
        n_trace_vectors=1,tau_points=3,seed=46,padding=0,
        use_momentum=True,trace_mode="exact",
    )
    assert response.ensemble=="grand"
    assert response.sectors.tolist()==[1,2]
    assert abs(response.average_particles-1.5)<2e-9
    np.testing.assert_allclose(response.sector_probabilities,[.5,.5],atol=2e-9)
    # In the 3-cell torus, O_total(0) = N/sqrt(3);
    # for a distribution N=1,2 with probabilities 1/2:
    # chi_NN(0) = beta Var(N)/3 = (1/T)*0.25/3.
    exact=.25/(T*geom.ncell)
    np.testing.assert_allclose(
        response.susceptibility_matrix[0,0].real,exact,atol=4e-10
    )
    np.testing.assert_allclose(
        response.correlation_tau_matrix[:,0,0].real,
        exact*T,atol=4e-10,
    )


def test_uniform_cluster_restart_replicates_impurity_and_bath_independently():
    ncell,nw,nbath=3,6,2
    bath=SimpleNamespace(
        energies=np.array([-.3,.2]),
        couplings=np.ones((6,nbath),complex)*(.2+.03j)
    )
    imp=np.tile(np.eye(6,dtype=complex),(nw,1,1))
    unif=SimpleNamespace(
        converged=True,G=np.zeros((nw,1,1,18,18),complex),
        Sigma_H=np.eye(18,dtype=complex),
        Sigma_emb=np.zeros((nw,1,1,18,18),complex),
        Sigma_ED_cluster=imp,mu=.11,bath=bath
    )
    out=restart_from_uniform_cluster(unif,ncell)
    assert out.Sigma_imp.shape==(ncell,nw,6,6)
    assert out.bath_energies.shape==(ncell,nbath)
    assert out.bath_couplings.shape==(ncell,6,nbath)
    assert np.max(abs(out.Sigma_imp[2]-imp))==0
    assert np.max(abs(out.bath_couplings[2]-bath.couplings))==0
    # Restart copies, not views: distinct source runs cannot leak into
    # each other's initial bath/self-energy.
    out.Sigma_imp[0,0,0,0]=3
    assert unif.Sigma_ED_cluster[0,0,0]==1
    out.bath_couplings[0,0,0]=10
    assert unif.bath.couplings[0,0]==(.2+.03j)


def test_fractional_filling_cli_auto_grand_and_gw_only_noninteger(tmp_path):
    script=Path(__file__).resolve().parents[1]/"research"/"benchmark_three_tori.py"
    out=tmp_path/"fractional"
    cmd=[
        sys.executable,str(script),"--geometry","3x1",
        "--V","0","--filling",".5",
        "--K","0","0","--Q","0","0",
        "--T",".5","--nw","3","--nomega","1",
        "--methods","ed","gw","--gw-channels","density",
        "--channels","charge_total",
        "--ed-ensemble","auto","--ed-charge-sector-padding","0",
        "--ed-symmetry","momentum","--ed-green-symmetry","momentum",
        "--ed-samples","1","--ed-tau-points","3",
        "--ed-sector-padding","1","--ed-thermal-eigs","1",
        "--ed-krylov-steps","10","--ed-eigs","2",
        "--gw-max-iter","8","--plot-tau-points","9",
        "--out",str(out)
    ]
    run=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    assert run.returncode==0,run.stdout+"\n"+run.stderr
    info=json.loads((out/"summary.json").read_text())
    assert info["ed_ensemble"]=="grand"
    assert info["N"] is None
    assert info["diagnostics"]["ED"]["ensemble"]=="grand"
    assert abs(info["diagnostics"]["ED"]["target_mean_particles"]-1.5)<1e-12
    assert "GW[density]" in info["G_methods"]
    assert "ED" in info["G_methods"]
    with open(out/"comparison.csv",encoding="utf-8") as f:
        lines=f.read()
    assert "grand_canonical_mu" in lines
    assert "not-defined" in lines
    # No single-N E0 and no erroneous zero-temperature error comparison.
    assert "energy_estimate_vs_ED_E0_diagnostic" not in lines


def test_cluster_chi_warmstart_and_coldstart_agree_at_free_limit(tmp_path):
    """Run real multi-impurity ±source solvers in both modes, compare χ.

    A warm start is an optimization of the *same* nonlinear fixed-point
    problem; it must not change the physical source derivative.
    """
    script=Path(__file__).resolve().parents[1]/"research"/"benchmark_three_tori.py"
    def one_run(out,cold):
        cmd=[
            sys.executable,str(script),
            "--geometry","3x1","--V","0","--filling",str(2/3),
            "--K","0","0","--Q","0","0",
            "--T",".4","--nw","3","--nomega","1",
            "--methods","gw","cluster","--gw-channels","density",
            "--channels","x_even",
            "--gw-max-iter","8","--gw-tol","1e-6",
            "--nbath","1","--cluster-max-iter","4",
            "--cluster-tol","1e-4","--source-step",".01",
            "--cluster-bath-fit-nfreq","2",
            "--cluster-bath-max-nfev","35",
            "--plot-tau-points","9","--out",str(out)
        ]
        if cold:
            cmd.append("--cluster-chi-coldstart")
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stdout+"\\n"+result.stderr
        summary=json.loads((out/"summary.json").read_text())
        diag=summary["diagnostics"]["cluster_source_chi"]["x_even"]
        assert diag["status"]=="converged",diag
        assert summary["diagnostics"]["cluster_source_chi"]["warmstart_used"]!=cold
        return float(summary["computed_chi"]["cluster-ED+GW"]["x_even"])
    warm=one_run(tmp_path/"warm",False)
    cold=one_run(tmp_path/"cold",True)
    np.testing.assert_allclose(warm,cold,atol=2e-4,rtol=1e-3)


def test_cluster_only_free_seed_skips_preliminary_gw(tmp_path):
    """Optional cheap seed must reach cluster outer loop without GW SCF."""
    script=Path(__file__).resolve().parents[1]/"research"/"benchmark_three_tori.py"
    out=tmp_path/"free_cluster"
    cmd=[
        sys.executable,str(script),"--geometry","3x1",
        "--V","0","--filling",str(2/3),
        "--K","0","0","--Q","0","0",
        "--T",".4","--nw","3","--nomega","1",
        "--methods","cluster","--cluster-initial-state","free",
        "--gw-channels","density","--channels","x_even",
        "--nbath","1","--cluster-max-iter","3",
        "--cluster-tol","1e-4","--cluster-bath-fit-nfreq","2",
        "--cluster-bath-max-nfev","30",
        "--plot-tau-points","9","--skip-cluster-chi",
        "--out",str(out),
    ]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=150)
    assert result.returncode==0,result.stdout+"\\n"+result.stderr
    data=json.loads((out/"summary.json").read_text())
    assert "GW[density]" not in data["timings"]
    assert data["diagnostics"]["cluster_seed"]["source_method"]=="noninteracting"
    assert data["method_status"]["cluster-ED+GW"]["status"]=="converged"
    assert "cluster-ED+GW" in data["G_methods"]
    assert np.isfinite(data["diagnostics"]["cluster_seed"]["seed_density_error"])
