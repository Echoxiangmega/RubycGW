import numpy as np

from rubycgw.fierz_pms_full import FullSimplexFierzPMSResidual
from rubycgw.grids import MatsubaraGrid


def test_full_simplex_pms_residual_shape_and_zero_interaction_gradients():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=4, nOmega=2, T=0.18)
    h0 = np.array([[[[0.05, -0.18], [-0.18, 0.02]]]], dtype=complex)
    problem = FullSimplexFierzPMSResidual(
        h0,
        [(0, 1)],
        grid,
        target=1.0,
        V=0.0,
        primitive_cells=1,
        fd_h=1e-3,
    )
    sigma_static = np.zeros((2, 2), dtype=complex)
    sigma_c = np.zeros((grid.nf, 1, 1, 2, 2), dtype=complex)
    x = problem.base_codec.encode(sigma_static, sigma_c, 0.01)
    z = problem.encode(x, s=0.4, a=-0.08)
    ev = problem.evaluate(z)

    assert ev.residual.shape == z.shape
    assert np.all(np.isfinite(ev.residual))
    assert abs(ev.longitudinal.dF_ds_per_cell) < 1e-9
    assert abs(ev.transverse.dF_da_per_cell) < 1e-9
    assert abs(ev.transverse.dF_dlambda_B_per_cell) < 1e-9
    assert abs(ev.transverse.dF_dlambda_J_per_cell) < 1e-9
    np.testing.assert_allclose(sum(ev.weights.as_tuple()), 1.0, atol=1e-14)
