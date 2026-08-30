from types import SimpleNamespace

import numpy as np

from rubycgw.checkpoint import GWCheckpointSeed
from rubycgw.supercell_gw_anderson import _anderson_type2_step
from run_supercell_gw import _predictor_seed


def test_type2_anderson_solves_scalar_linear_fixed_point():
    # F(x)=0.9*x+1 => R(x)=1-0.1*x, fixed point x=10.
    x0 = np.array([[0.0 + 0.0j]])
    r0 = np.array([[1.0 + 0.0j]])
    x1 = np.array([[0.2 + 0.0j]])
    r1 = np.array([[0.98 + 0.0j]])

    # Use identical one-element Hartree/GW blocks so the block metric is simple.
    history = [
        (x0, x0.copy(), r0, r0.copy()),
        (x1, x1.copy(), r1, r1.copy()),
    ]
    hnext, gnext, accepted = _anderson_type2_step(
        x1,
        x1.copy(),
        r1,
        r1.copy(),
        history,
        beta=0.2,
        regularization=0.0,
        sh=1.0,
        sg=1.0,
        step_cap=100.0,
    )
    assert accepted
    assert np.allclose(hnext, 10.0, atol=1e-10)
    assert np.allclose(gnext, 10.0, atol=1e-10)


def test_v_secant_predictor_uses_same_broken_branch():
    g1 = GWCheckpointSeed(
        Sigma_H=np.array([[1.0 + 0.0j]]),
        Sigma_GW=np.array([[[[[2.0 + 0.0j]]]]]),
        mu=3.0,
    )
    g2 = GWCheckpointSeed(
        Sigma_H=np.array([[2.0 + 0.0j]]),
        Sigma_GW=np.array([[[[[4.0 + 0.0j]]]]]),
        mu=4.0,
    )
    args = SimpleNamespace(
        no_v_predictor=False,
        predictor_max_ratio=2.0,
        predictor_order_threshold=1e-4,
        predictor_damping=0.8,
    )
    seed, label = _predictor_seed(
        0.90,
        [(0.80, g1, 0.50 + 0.0j), (0.85, g2, 0.52 + 0.0j)],
        args,
    )
    assert seed is not None
    # ratio=1 and damping=0.8.
    assert np.allclose(seed.Sigma_H, 2.8)
    assert np.allclose(seed.Sigma_GW, 5.6)
    assert abs(seed.mu - 4.8) < 1e-12
    assert "0.8,0.85" in label


def test_v_secant_predictor_skips_across_normal_to_broken_transition():
    g = GWCheckpointSeed(
        Sigma_H=np.array([[1.0 + 0.0j]]),
        Sigma_GW=np.array([[[[[1.0 + 0.0j]]]]]),
        mu=0.0,
    )
    args = SimpleNamespace(
        no_v_predictor=False,
        predictor_max_ratio=2.0,
        predictor_order_threshold=1e-4,
        predictor_damping=0.8,
    )
    seed, _ = _predictor_seed(
        0.80,
        [(0.75, g, 0.0j), (0.78, g, 0.48 + 0.0j)],
        args,
    )
    assert seed is None
