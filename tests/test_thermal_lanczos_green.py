import numpy as np

from rubycgw.model import RubyParameters
from rubycgw.small_cluster_exact import ExactSmallRubyThermal
from rubycgw.thermal_lanczos_green import ThermalLanczosOptions, thermal_lanczos_green


def test_thermal_lanczos_green_matches_full_ed_on_one_cell():
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.4)
    T = 0.08
    target = 2.0
    omega = (2 * np.arange(-4, 5) + 1) * np.pi * T

    exact = ExactSmallRubyThermal(1, 1, params)
    exact.diagonalize(params.V)
    mu_exact = exact.solve_mu(target, T)
    G_exact, _ = exact.green_iomega(1j * omega, mu_exact, T)

    opts = ThermalLanczosOptions(
        sector_padding=1,
        thermal_eigs=20,
        thermal_discard_weight_tol=1e-12,
        krylov_dim=64,
        krylov_tol=1e-14,
        eig_tol=1e-12,
        verbose=False,
    )
    approx = thermal_lanczos_green(
        1,
        1,
        params,
        V=params.V,
        T=T,
        target_particles=target,
        omega=omega,
        opts=opts,
    )

    rel = np.linalg.norm(approx.G - G_exact) / np.linalg.norm(G_exact)
    assert abs(approx.average_particles - target) < 1e-9
    assert abs(approx.mu - mu_exact) < 2e-4
    assert rel < 2e-4


def test_thermal_lanczos_result_has_small_sector_edge_weight_at_low_T():
    params = RubyParameters(ti=0.4, t1=0.2, t2=0.2, V=0.4)
    T = 0.08
    omega = (2 * np.arange(-2, 3) + 1) * np.pi * T
    opts = ThermalLanczosOptions(
        sector_padding=1,
        thermal_eigs=12,
        thermal_discard_weight_tol=1e-10,
        krylov_dim=48,
        verbose=False,
    )
    result = thermal_lanczos_green(
        1,
        1,
        params,
        V=params.V,
        T=T,
        target_particles=2.0,
        omega=omega,
        opts=opts,
    )
    edge = result.sector_probabilities[0] + result.sector_probabilities[-1]
    assert edge < 0.1
    assert result.kept_thermal_weight > 1.0 - 2e-10
