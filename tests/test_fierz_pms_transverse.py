import numpy as np

from rubycgw.fierz_mixed import FierzWeights, WeightedFierzGWResidual
from rubycgw.fierz_pms_transverse import (
    LongitudinalFierzPMSResidual,
    LongitudinalPMSCodec,
    s_a_from_weights,
    weights_from_s_a,
)
from rubycgw.grids import MatsubaraGrid


def test_s_a_weight_roundtrip():
    for s, a in ((0.4, 0.0), (0.55, 0.15), (0.7, -0.2)):
        w = weights_from_s_a(s, a)
        s2, a2 = s_a_from_weights(w)
        np.testing.assert_allclose([s2, a2], [s, a], rtol=0, atol=2e-15)
        np.testing.assert_allclose(sum(w.as_tuple()), 1.0, rtol=0, atol=2e-15)
        assert min(w.as_tuple()) > 0.0


def test_longitudinal_codec_roundtrip_depends_on_a_bounds():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=3, nOmega=2, T=0.2)
    base = WeightedFierzGWResidual(
        np.zeros((1, 1, 2, 2), dtype=complex),
        [(0, 1)],
        FierzWeights(0.5, 0.25, 0.25),
        grid,
        1.0,
    ).codec
    codec = LongitudinalPMSCodec(base, margin=1e-8)
    x = np.linspace(-0.1, 0.2, base.size)
    for a, s in ((0.0, 0.4), (0.2, 0.5), (-0.3, 0.65)):
        y = codec.encode(x, s, a)
        x2, s2, _ = codec.decode(y, a)
        np.testing.assert_allclose(x2, x, rtol=0, atol=0)
        np.testing.assert_allclose(s2, s, rtol=0, atol=2e-15)


def test_transverse_pms_derivatives_vanish_at_zero_interaction():
    grid = MatsubaraGrid(nk1=1, nk2=1, nw=5, nOmega=3, T=0.16)
    h0 = np.array([[[[0.12, -0.25], [-0.25, -0.07]]]], dtype=complex)
    solver = LongitudinalFierzPMSResidual(
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
    x = solver.base_codec.encode(sigma_static, sigma_c, 0.01)
    y = solver.encode(x, s=0.5, a=0.1)
    ev = solver.evaluate(y, a=0.1)
    diag = solver.transverse_diagnostics(y, a=0.1, evaluation=ev)

    assert ev.residual.shape == y.shape
    assert np.all(np.isfinite(ev.residual))
    assert abs(ev.dF_ds_per_cell) < 1e-9
    assert abs(ev.d2F_ds2_per_cell) < 1e-6
    assert abs(diag.dF_da_per_cell) < 1e-9
    assert abs(diag.dF_dlambda_B_per_cell) < 1e-9
    assert abs(diag.dF_dlambda_J_per_cell) < 1e-9
    assert abs(diag.d2F_da2_per_cell) < 1e-6
