"""Exact ED order diagnostics regression and normalization checks."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ed_benchmark"/"vmc_zeroT"))
sys.path.insert(0,str(ROOT/"ed_benchmark"/"native_ed"))
from order_diagnostics import _ground_fields, _numerical_radius
from run_benchmark import solve_ed
from geometry import make_ring3_supercell
import observables
from benchmark_observables import benchmark_observables


def test_projected_operator_weights():
    A=np.array([[0,1],[0,0]],complex)
    assert abs(_numerical_radius(A)-.5)<1e-10
    f=_ground_fields("test",A,1.)
    assert abs(f["test_ground_manifold_weight"]-.5)<1e-12
    assert abs(f["test_excited_contribution"]-.5)<1e-12
    assert _ground_fields("zero",np.zeros((2,2)),0.)["zero_ground_manifold_weight"] is None


def test_ruby_reported_nf4():
    g=make_ring3_supercell(1,1,t=1.,t1=.5,t2=-.5)
    H,_,gs,row=solve_ed(g,4,8.6,k=12,degeneracy_tol=1e-8,backend="full")
    assert not row["ground_manifold_truncated"]
    assert row["ground_degeneracy"]==2
    assert abs(row["CO_qmax"]-.3055408020930988)<1e-7
    assert abs(row["S_LC_same"]-.03466590624327616)<1e-7
    assert abs(row["S_LC_opposite"]-.02145900576844498)<1e-7
    assert np.isclose(row["S_CO_over_Ntriangle"],row["CO_qmax"]/6)
    for name in ("CO","CO_E","CO_N"):
        assert np.isclose(row[f"S_{name}_over_Ntriangle"],row[f"S_{name}_local"]+row[f"S_{name}_intertriangle"])
        assert row[f"{name}_ground_manifold_contribution"]>=0
        assert np.isclose(row[f"S_{name}_over_Ntriangle"],row[f"{name}_ground_manifold_contribution"]+row[f"{name}_excited_contribution"])
    for name in ("LC_same","LC_opposite"):
        assert np.isclose(row[f"S_{name}"],row["S_LC_local"]+row[f"S_{name}_intertriangle"])
        assert np.isclose(row[f"S_{name}"],row[f"{name}_ground_manifold_contribution"]+row[f"{name}_excited_contribution"])
    assert abs(row["CO_leading_E_fraction"]+row["CO_leading_N_fraction"]-1)<1e-10
    assert np.isclose(row["triangle_number_mean"],2/3)
    # Invariance under arbitrary rotation of the complete degenerate basis.
    U=np.array([[1,1j],[1j,1]])/np.sqrt(2)
    other=benchmark_observables(g,H.states,gs@U,observables,ground_manifold=True).as_dict()
    for name in ("LC_same","LC_opposite"):
        for suffix in ("ground_manifold_contribution","max_ground_polarization"):
            key=f"{name}_{suffix}"
            assert np.isclose(row[key],other[key],atol=1e-8)
