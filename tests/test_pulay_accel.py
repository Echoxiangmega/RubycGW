import numpy as np

from rubycgw.pulay_accel import scale_invariant_pulay_coefficients


def _history(scale=1.0):
    # Three non-collinear synthetic residuals.  Output arrays are irrelevant to
    # the coefficient calculation but kept in the production tuple layout.
    zero = np.zeros((2,), dtype=complex)
    rh = [
        np.array([1.0, 0.2]),
        np.array([0.55, -0.10]),
        np.array([0.18, 0.07]),
    ]
    rg = [
        np.array([0.4, -0.3, 0.1]),
        np.array([0.15, -0.22, 0.05]),
        np.array([0.04, -0.08, 0.03]),
    ]
    return [
        (zero.copy(), zero.copy(), scale * a, scale * b)
        for a, b in zip(rh, rg)
    ]


def test_pulay_coefficients_are_residual_scale_invariant():
    c1 = scale_invariant_pulay_coefficients(_history(1.0), 1e-7)
    c2 = scale_invariant_pulay_coefficients(_history(1e-6), 1e-7)
    assert np.allclose(c1, c2, rtol=1e-11, atol=1e-12)
    assert np.isclose(np.sum(c1), 1.0, rtol=0.0, atol=1e-12)


def test_pulay_tiny_residuals_remain_finite():
    c = scale_invariant_pulay_coefficients(_history(1e-14), 1e-7)
    assert np.all(np.isfinite(c))
    assert np.isclose(np.sum(c), 1.0, rtol=0.0, atol=1e-12)
