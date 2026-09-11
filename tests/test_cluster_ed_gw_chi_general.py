import numpy as np

from benchmark_cluster_ed_gw_chi import (
    _extrapolate_h2,
    _full_ed_supported,
    _matrix_relerr,
)


def test_full_ed_support_tracks_physical_site_count():
    assert _full_ed_supported(1, 1)
    assert _full_ed_supported(2, 1)
    assert _full_ed_supported(1, 2)
    assert not _full_ed_supported(3, 1)
    assert not _full_ed_supported(2, 2)


def test_h2_extrapolation_recovers_intercept():
    h = np.asarray([1.0e-3, 2.0e-3, 3.0e-3])
    intercept = np.asarray([[10.0, 0.2], [0.2, 8.0]])
    slope = np.asarray([[3.0, -2.0], [-2.0, 1.5]])
    mats = intercept[None, :, :] + h[:, None, None] ** 2 * slope[None, :, :]
    got = _extrapolate_h2(h, mats)
    np.testing.assert_allclose(got, intercept, atol=1.0e-12, rtol=1.0e-12)


def test_matrix_relerr_zero_for_equal_inputs():
    a = np.asarray([[1.0, 2.0], [2.0, 4.0]])
    assert _matrix_relerr(a, a) == 0.0
