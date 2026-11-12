"""Regression: immutable Fock templates and parallel cluster χ physics."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rubycgw.ed_cluster import fixed_popcount_basis
from rubycgw.impurity_ed import FiniteBathImpurityED, _cached_fock_geometry


def _direct_fixed_n_hamiltonian(h,terms,N):
    """Independent reference: direct second-quantized creation/annihilation."""
    norb=h.shape[0]
    basis=fixed_popcount_basis(norb,N)
    ids={int(q):i for i,q in enumerate(basis)}
    H=np.zeros((len(basis),len(basis)),dtype=complex)
    for col,state in enumerate(basis):
        state=int(state)
        for i in range(norb):
            if (state>>i)&1:
                H[col,col]+=h[i,i]
        for i,j,V in terms:
            H[col,col]+=V*int(bool((state>>i)&1 and (state>>j)&1))
        for i in range(norb):
            for j in range(norb):
                if i==j or not ((state>>j)&1) or ((state>>i)&1):
                    continue
                parity=(state&((1<<j)-1)).bit_count()
                after=state^(1<<j)
                parity+=(after&((1<<i)-1)).bit_count()
                sgn=-1 if parity%2 else 1
                dest=after|(1<<i)
                H[ids[dest],col]+=h[i,j]*sgn
    return H


@pytest.mark.parametrize("norb,N",[(7,2),(8,4),(12,3)])
def test_cached_fock_geometry_reproduces_dense_impurity_hamiltonian(norb,N):
    rng=np.random.default_rng(norb*71+N)
    A=rng.normal(size=(norb,norb))+1j*rng.normal(size=(norb,norb))
    h=(A+A.conj().T)*.12
    h[np.abs(h)<.12]=0
    np.fill_diagonal(h,np.arange(norb)*.1)
    terms=((0,1,.35),(1,2,-.13))
    imp=FiniteBathImpurityED(h,terms,correlated_orbitals=(0,1,2))
    cached=imp.dense_hamiltonian(N)
    ref=_direct_fixed_n_hamiltonian(h,terms,N)
    np.testing.assert_allclose(cached,ref,atol=3e-13,rtol=0)
    # A different bath h produces the same topology but different H.
    newer=FiniteBathImpurityED(h+.02*np.eye(norb),terms)
    np.testing.assert_allclose(
        newer.dense_hamiltonian(N)-cached,.02*N*np.eye(len(cached)),
        atol=3e-13,rtol=0
    )
    b,lookup,occ,hops=_cached_fock_geometry(norb,N)
    assert not b.flags.writeable
    assert not lookup.flags.writeable
    assert not occ.flags.writeable
    assert (0,1) in hops
    assert _cached_fock_geometry.cache_info().hits>=1


def _run_chi(out,workers):
    script=Path(__file__).resolve().parents[1]/"research"/"benchmark_three_tori.py"
    cmd=[
        sys.executable,str(script),
        "--geometry","3x1","--V","0","--filling",str(2./3.),
        "--K","0","0","--Q","0","0",
        "--T",".4","--nw","3","--nomega","1",
        "--methods","gw","cluster","--gw-channels","density",
        "--channels","x_even","z_same",
        "--gw-max-iter","8","--gw-tol","1e-6",
        "--nbath","1","--cluster-max-iter","4","--cluster-tol","1e-4",
        "--cluster-bath-fit-nfreq","2","--cluster-bath-max-nfev","35",
        "--source-step",".01","--cluster-chi-workers",str(workers),
        "--plot-tau-points","9","--out",str(out),
    ]
    res=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    assert res.returncode==0,res.stdout+"\n"+res.stderr
    info=json.loads((out/"summary.json").read_text())
    assert info["method_status"]["cluster-ED+GW"]["status"]=="converged"
    for ch in ("x_even","z_same"):
        assert info["diagnostics"]["cluster_source_chi"][ch]["status"]=="converged"
    assert info["diagnostics"]["cluster_source_chi"]["_execution"]["workers"]==workers
    assert "phase_seconds" in info["diagnostics"]["cluster_source_chi"]["x_even"]
    return info


def test_windows_safe_parallel_source_chi_matches_serial(tmp_path):
    serial=_run_chi(tmp_path/"serial",1)
    parallel=_run_chi(tmp_path/"parallel",2)
    for ch in ("x_even","z_same"):
        np.testing.assert_allclose(
            parallel["computed_chi"]["cluster-ED+GW"][ch],
            serial["computed_chi"]["cluster-ED+GW"][ch],
            rtol=1e-8,atol=3e-8
        )
    assert "timing_seconds" in serial["diagnostics"]["cluster-ED+GW"]
