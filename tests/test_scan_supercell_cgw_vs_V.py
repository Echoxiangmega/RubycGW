import numpy as np
import pytest

from scan_supercell_cgw_vs_V import (
    _bridge_midpoint,
    _curvature_fields,
    _normal_breaking_scale,
    _prepare_v_values,
    _zero_crossings,
)


def test_prepare_v_values_builds_inclusive_linspace():
    values = _prepare_v_values(0.5, 2.0, 4)
    assert np.allclose(values, [0.5, 1.0, 1.5, 2.0])


def test_prepare_v_values_single_point_requires_equal_endpoints():
    assert _prepare_v_values(1.2, 1.2, 1) == [1.2]
    with pytest.raises(ValueError):
        _prepare_v_values(1.0, 2.0, 1)


def test_prepare_v_values_rejects_descending_range():
    with pytest.raises(ValueError):
        _prepare_v_values(2.0, 1.0, 5)


def test_bridge_midpoint_respects_min_step_and_depth():
    assert abs(_bridge_midpoint(1.0, 1.2, 0.01, 0, 6) - 1.1) < 1e-14
    assert _bridge_midpoint(1.0, 1.008, 0.005, 0, 6) is None
    assert _bridge_midpoint(1.0, 1.2, 0.01, 6, 6) is None
    assert _bridge_midpoint(None, 1.2, 0.01, 0, 6) is None


def test_zero_crossings_returns_linear_interpolation():
    V = np.array([0.5, 1.0, 1.5])
    y = np.array([1.0, 0.5, -0.5])
    crossings = _zero_crossings(V, y)
    assert len(crossings) == 1
    left, right, estimate = crossings[0]
    assert left == 1.0
    assert right == 1.5
    assert abs(estimate - 1.25) < 1e-14


def test_curvature_fields_use_uniform_relaxed_and_Q_block():
    Rh = np.diag([1.1, 3.0, 4.0, 2.2, 5.0, 6.0])
    analysis = {
        "R_uniform_relaxed": np.array([[1.0, 0.2], [0.2, 2.0]]),
        "R_uniform_constrained": np.array([[1.1, 0.3], [0.3, 2.2]]),
        "R_harmonic": Rh,
        "R_eigenvalues": np.array([0.8, 1.0, 2.0, 3.0, 4.0, 5.0]),
        "soft_weight_opposite": 0.75,
        "soft_weight_same": 0.25,
        "soft_weight_q0": 0.4,
        "soft_weight_Q": 0.6,
        "chi_imag_max": 1e-13,
    }
    out = _curvature_fields(analysis)
    assert out["r_plus"] == 1.0
    assert out["r_minus"] == 2.0
    assert out["r_plusminus"] == 0.2
    assert out["r_plus_constrained"] == 1.1
    assert out["r_minus_constrained"] == 2.2
    assert out["r_Q_min"] == 3.0
    assert out["r_soft_full"] == 0.8


def test_normal_breaking_scale_checks_charge_and_both_currents():
    diag = {
        "Delta_Q": 1e-8,
        "Delta_intra": 2e-8,
        "Delta_AB": -3e-8,
        "m_plus_pc_abs": 4e-8,
        "m_minus_pc_abs": 5e-8,
    }
    assert _normal_breaking_scale(diag) == 5e-8
