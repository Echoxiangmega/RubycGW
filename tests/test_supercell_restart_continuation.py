from types import SimpleNamespace

import numpy as np

from run_supercell_gw import _densify_strong_v_schedule, _predictor_seed


class _Seed:
    def __init__(self, x: float):
        self.Sigma_H = np.full((2, 2), x, dtype=complex)
        self.Sigma_GW = np.full((1, 1, 1, 2, 2), 2.0 * x, dtype=complex)
        self.mu = 3.0 * x


def test_restart_1_to_1p2_is_densified_to_0p1_steps():
    values = _densify_strong_v_schedule(
        1.0, [1.2], onset=1.0, max_step=0.10
    )
    assert np.allclose(values, [1.1, 1.2])


def test_explicit_small_final_segment_is_preserved():
    values = _densify_strong_v_schedule(
        1.0, [1.25], onset=1.0, max_step=0.10
    )
    assert np.allclose(values, [1.0833333333333333, 1.1666666666666667, 1.25])
    assert max(np.diff([1.0] + values)) <= 0.10 + 1e-12


def test_two_restart_states_enable_first_point_secant_prediction():
    args = SimpleNamespace(
        no_v_predictor=False,
        predictor_max_ratio=2.0,
        predictor_damping=0.8,
        predictor_order_threshold=1e-4,
    )
    s09 = _Seed(0.9)
    s10 = _Seed(1.0)
    phi = 0.5 + 0.0j
    history = [(0.9, s09, phi), (1.0, s10, phi)]

    pred, label = _predictor_seed(1.1, history, args)
    assert pred is not None
    # ratio=(1.1-1.0)/(1.0-0.9)=1, damping=0.8
    expected = 1.0 + 0.8 * (1.0 - 0.9)
    assert np.allclose(pred.Sigma_H, expected)
    assert np.allclose(pred.Sigma_GW, 2.0 * expected)
    assert np.isclose(pred.mu, 3.0 * expected)
    assert "secant V=0.9,1" in label
