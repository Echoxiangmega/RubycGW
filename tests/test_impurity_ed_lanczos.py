import numpy as np

from rubycgw.impurity_ed import FiniteBathImpurityED
from rubycgw.impurity_ed_lanczos import FiniteBathImpurityLanczosED


def _compare_dense_lanczos(h, interactions, corr, mu, T, omega, atol=2e-8):
    dense = FiniteBathImpurityED(h, interactions, correlated_orbitals=corr)
    dense.diagonalize()
    gd, _ = dense.green_iomega(1j * omega, mu, T, discard_weight_tol=0.0)

    lanc = FiniteBathImpurityLanczosED(
        h,
        interactions,
        correlated_orbitals=corr,
        thermal_state_tol=1e-13,
        thermal_initial_states=8,
        thermal_max_states=64,
        dense_sector_threshold=2,
        krylov_steps=24,
        krylov_tol=1e-13,
    )
    lanc.diagonalize()
    gl, sel = lanc.green_iomega(1j * omega, mu, T, discard_weight_tol=1e-13)
    assert np.max(np.abs(gl - gd)) < atol
    assert abs(sel.average_particles - dense.thermal_selection(mu, T, discard_weight_tol=0.0).average_particles) < 2e-8


def test_lanczos_green_matches_dense_noninteracting():
    h = np.array(
        [
            [0.10, 0.20, 0.35, 0.00],
            [0.20, -0.15, 0.00, -0.25],
            [0.35, 0.00, 0.70, 0.10],
            [0.00, -0.25, 0.10, -0.55],
        ],
        dtype=float,
    )
    T = 0.17
    mu = 0.08
    omega = (2 * np.arange(-4, 4) + 1) * np.pi * T
    _compare_dense_lanczos(h, (), (0, 1), mu, T, omega, atol=2e-9)


def test_lanczos_green_matches_dense_interacting():
    h = np.array(
        [
            [0.0, 0.22, 0.0, 0.18, 0.0, 0.0],
            [0.22, 0.05, 0.17, 0.0, 0.11, 0.0],
            [0.0, 0.17, -0.08, 0.0, 0.0, 0.14],
            [0.18, 0.0, 0.0, 0.45, 0.0, 0.0],
            [0.0, 0.11, 0.0, 0.0, -0.35, 0.0],
            [0.0, 0.0, 0.14, 0.0, 0.0, 0.65],
        ],
        dtype=float,
    )
    interactions = ((0, 1, 0.8), (1, 2, 0.8), (0, 2, 0.8))
    T = 0.14
    mu = 0.12
    omega = (2 * np.arange(-3, 3) + 1) * np.pi * T
    _compare_dense_lanczos(h, interactions, (0, 1, 2), mu, T, omega, atol=2e-7)
