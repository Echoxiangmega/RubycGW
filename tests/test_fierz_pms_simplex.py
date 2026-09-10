import numpy as np

from rubycgw.fierz_pms import WeightedFierzPMSResidual
from rubycgw.grids import MatsubaraGrid


def _toy_problem(V=0.4):
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
    return problem, y, V


def test_full_simplex_gradient_reconstructs_symmetric_line_derivative():
    problem, y, V = _toy_problem()
    ev = problem.evaluate(y, V)
    grad = problem.simplex_gradient(y, V, evaluation=ev)

    assert np.isfinite(grad.dF_dlambda_B_per_cell)
    assert np.isfinite(grad.dF_dlambda_J_per_cell)
    assert np.isfinite(grad.gradient_norm_per_cell)
    np.testing.assert_allclose(
        grad.dF_dlambda_reconstructed_per_cell,
        ev.dF_dlambda_per_cell,
        rtol=2e-6,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        grad.parallel_per_cell,
        np.sqrt(2.0) * ev.dF_dlambda_per_cell,
        rtol=2e-6,
        atol=3e-9,
    )
    np.testing.assert_allclose(
        grad.gradient_norm_per_cell,
        np.hypot(grad.parallel_per_cell, grad.perpendicular_per_cell),
        rtol=0,
        atol=2e-12,
    )


def test_full_simplex_gradient_vanishes_when_V_zero():
    problem, y, _ = _toy_problem(V=0.0)
    ev = problem.evaluate(y, 0.0)
    grad = problem.simplex_gradient(y, 0.0, evaluation=ev)

    assert grad.fd_h_used > 0.0
    assert grad.gradient_norm_per_cell < 1e-8
    assert abs(grad.perpendicular_per_cell) < 1e-8
    assert abs(grad.line_consistency_error_per_cell) < 1e-8
