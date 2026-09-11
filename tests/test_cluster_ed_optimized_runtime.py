from rubycgw.bath_fit_optimized import install_optimized_bath_fit
from rubycgw.bath_fit_complex_optimized import install_complex_bath_fit
from rubycgw.impurity_ed_lanczos import install_impurity_solver
import benchmark_cluster_ed_weak_chi_opt as chi_opt


def test_optimized_runtime_entrypoints_are_callable():
    assert callable(install_optimized_bath_fit)
    assert callable(install_complex_bath_fit)
    assert callable(install_impurity_solver)
    assert callable(chi_opt.main)
