import numpy as np

from rubycgw.fierz_mixed import WeightedFierzGWResidual, weights_from_lambda
from rubycgw.fierz_pms import PMSLambdaCodec, WeightedFierzPMSResidual
from rubycgw.grids import MatsubaraGrid


def test_pms_lambda_codec_roundtrip():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=2, T=0.2)
    base = WeightedFierzGWResidual(
        np.zeros((1, 1, 2, 2), dtype=complex),
        [(0, 1)],
        weights_from_lambda(0.5),
        grid,
        1.0,
    ).codec
    codec = PMSLambdaCodec(base, margin=1e-8)
    x = np.linspace(-0.2, 0.3, base.size)
    for lam in (0.05, 0.37, 0.5, 0.91):
        y = codec.encode(x, lam)
        x2, _, lam2 = codec.decode(y)
        np.testing.assert_allclose(x2, x, rtol=0, atol=0)
        np.testing.assert_allclose(lam2, lam, rtol=0, atol=2e-15)
        assert y.size == codec.size == base.size + 1


def test_pms_free_energy_derivative_vanishes_at_zero_interaction():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=5, nOmega=3, T=0.16)
    h0 = np.array([[[[0.12, -0.25], [-0.25, -0.07]]]], dtype=complex)
    problem = WeightedFierzPMSResidual(
        h0,
        [(0, 1)],
        grid,
        target=1.0,
        primitive_cells=1,
        lambda_fd_h=2e-3,
    )
    sigma_static = np.zeros((2, 2), dtype=complex)
    sigma_c = np.zeros((grid.nf, 1, 1, 2, 2), dtype=complex)
    x = problem.base_codec.encode(sigma_static, sigma_c, 0.01)
    y = problem.encode_base_state(x, 0.37)
    ev = problem.evaluate(y, 0.0)

    assert ev.residual.shape == y.shape
    assert np.all(np.isfinite(ev.residual))
    assert abs(ev.dF_dlambda_per_cell) < 1e-9
    assert abs(ev.d2F_dlambda2_per_cell) < 1e-6
    assert abs(ev.pms_scaled_residual) < 1e-5


def test_pms_finite_interaction_reports_finite_slope_and_curvature():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=4, nOmega=2, T=0.18)
    h0 = np.array([[[[0.05, -0.18], [-0.18, 0.02]]]], dtype=complex)
    problem = WeightedFierzPMSResidual(
        h0,
        [(0, 1)],
        grid,
        target=1.0,
        primitive_cells=1,
        lambda_fd_h=1e-3,
    )
    sigma_static = np.array([[0.06, -0.01], [-0.01, 0.04]], dtype=complex)
    sigma_c = np.zeros((grid.nf, 1, 1, 2, 2), dtype=complex)
    x = problem.base_codec.encode(sigma_static, sigma_c, 0.03)
    y = problem.encode_base_state(x, 0.55)
    ev = problem.evaluate(y, 0.4)

    assert ev.residual.shape == y.shape
    assert np.isfinite(ev.dF_dlambda_per_cell)
    assert np.isfinite(ev.d2F_dlambda2_per_cell)
    assert np.isfinite(ev.free_energy.free_energy_per_primitive_cell)
    assert 0.0 < ev.lambda_value < 1.0
    assert ev.fd_h_used > 0.0
